# =============================================================================
# dataset.py  —  Dataset loader.
#
# Handles three layouts:
#   1. Hokkaido  — img/  +  label/  +  mask/
#   2. CAS       — images/  +  masks/
#   3. Bijie     — images/  +  masks/
#
# Mask pixel convention:
#   0   = background
#   255 = landslide   (converted to 1 internally)
#   OR
#   1   = landslide   (raw label files)  — both are handled
# =============================================================================

import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split

try:
    import rasterio                  # reads GeoTIFF satellite files
    from rasterio.enums import Resampling
except ImportError:                  # PNG/JPEG workflows can still be imported/tested
    rasterio = None
    Resampling = None

from config import MEAN, STD, SEED


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS  = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
MASK_EXTS = {'.png', '.tif', '.tiff', '.jpg'}


def _load_image(path: str, size: int) -> np.ndarray:
    """Load any image file (jpg / png / tif) as float32 RGB [H,W,3] in [0,1]."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.tif', '.tiff'}:
        if rasterio is None:
            raise ImportError(
                "rasterio is required to read TIFF images; install requirements.txt")
        with rasterio.open(path) as src:
            # Read first 3 bands as RGB
            if src.count >= 3:
                data = src.read([1, 2, 3],
                                out_shape=(3, size, size),
                                resampling=Resampling.bilinear)   # [3,H,W]
            else:
                band = src.read(1,
                                out_shape=(size, size),
                                resampling=Resampling.bilinear)   # [H,W]
                data = np.stack([band, band, band])               # [3,H,W]
            data = data.astype(np.float32)
            # Normalise to [0,1]: handle uint8 (0-255) and uint16 (0-65535)
            if data.max() > 1.0:
                data = data / (255.0 if data.max() <= 255 else 65535.0)
            return data.transpose(1, 2, 0)   # [H,W,3]
    else:
        img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0   # [H,W,3]


def _load_mask(path: str, size: int) -> np.ndarray:
    """Load mask as binary int64 [H,W] with values {0, 1}."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.tif', '.tiff'}:
        if rasterio is None:
            raise ImportError(
                "rasterio is required to read TIFF masks; install requirements.txt")
        with rasterio.open(path) as src:
            data = src.read(1,
                            out_shape=(size, size),
                            resampling=Resampling.nearest)   # nearest keeps binary
        data = data.astype(np.float32)
    else:
        img  = Image.open(path).convert('L').resize((size, size), Image.NEAREST)
        data = np.array(img, dtype=np.float32)

    # Convert: values >0.5 → 1, else 0
    # Handles both 0/255 (PNG) and 0/1 (raw label tif)
    return (data > 0.5).astype(np.int64)   # [H,W]


def _augment(img: np.ndarray, mask: np.ndarray):
    """Random horizontal + vertical flip."""
    if np.random.rand() > 0.5:
        img  = img[:, ::-1, :].copy()
        mask = mask[:, ::-1].copy()
    if np.random.rand() > 0.5:
        img  = img[::-1, :, :].copy()
        mask = mask[::-1, :].copy()
    return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# Core Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class LandslideDataset(Dataset):
    """
    Universal landslide dataset loader.

    Supports layouts:
      Layout A (CAS / Bijie):
          img_dir/   image_001.jpg ...
          mask_dir/  image_001.png ...   ← same stem, .png mask

      Layout B (Hokkaido):
          img_dir/   image_001.tif ...
          mask_dir/  image_001.tif ...   ← label or mask folder
          The loader tries mask_dir first, then label_dir as fallback.

    Args:
        img_dir  : folder of satellite images
        mask_dir : folder of binary masks  (0=bg, 255 or 1 = landslide)
        img_size : resize all images to this square size
        augment  : apply random flips during training
    """

    def __init__(self, img_dir: str, mask_dir: str,
                 img_size: int = 512, augment: bool = False):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment  = augment
        self.mean     = np.array(MEAN, dtype=np.float32)
        self.std      = np.array(STD,  dtype=np.float32)

        # Collect all image paths
        all_files = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No images found in {img_dir}\n"
                f"Supported: {IMG_EXTS}")

        # Pair each image with its mask
        self.pairs = []
        missing    = []
        for fname in all_files:
            stem      = os.path.splitext(fname)[0]
            img_path  = os.path.join(img_dir, fname)
            mask_path = self._find_mask(stem)
            if mask_path:
                self.pairs.append((img_path, mask_path))
            else:
                missing.append(fname)

        if len(missing) > 0:
            print(f"  WARNING: {len(missing)} images have no matching mask — skipped.")
        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No image-mask pairs found.\n"
                f"img_dir : {img_dir}\n"
                f"mask_dir: {mask_dir}")

        print(f"Dataset ready: {len(self.pairs)} pairs  |  "
              f"size={img_size}  augment={augment}  [{img_dir}]")

    def _find_mask(self, stem: str):
        """Try all mask extensions in mask_dir. Return path or None."""
        for ext in MASK_EXTS:
            p = os.path.join(self.mask_dir, stem + ext)
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = _load_image(img_path,  self.img_size)   # [H,W,3]  float32 [0,1]
        mask = _load_mask(mask_path,  self.img_size)   # [H,W]    int64  {0,1}

        if self.augment:
            img, mask = _augment(img, mask)

        # ImageNet normalisation
        img = (img - self.mean) / self.std             # [H,W,3]  ~ [-2.5, +2.5]

        img_t  = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)  # [3,H,W]
        mask_t = torch.tensor(mask, dtype=torch.long)                       # [H,W]
        return img_t, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Hokkaido-specific loader
# Hokkaido has three folders:  img/ + label/ + mask/
# mask/ = pre-made binary PNGs  (preferred, cleaner)
# label/ = raw label tifs       (fallback if mask/ empty)
# ─────────────────────────────────────────────────────────────────────────────

class HokkaidoDataset(LandslideDataset):
    """
    Hokkaido dataset with  img/  label/  mask/  folder layout.

    Priority:
      1. Use mask/ folder if non-empty  (already binary 0/255 PNGs)
      2. Fall back to label/ folder     (raw 0/1 GeoTIFF labels)
    """
    def __init__(self, img_dir: str, label_dir: str, mask_dir: str,
                 img_size: int = 512, augment: bool = False):

        # Choose which mask folder to use
        mask_files = glob.glob(os.path.join(mask_dir, '*'))
        if len(mask_files) > 0:
            chosen_mask_dir = mask_dir
            print(f"Hokkaido: using pre-made masks from  {mask_dir}")
        else:
            chosen_mask_dir = label_dir
            print(f"Hokkaido: mask/ empty → using labels from  {label_dir}")

        super().__init__(img_dir, chosen_mask_dir, img_size, augment)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(img_dir, mask_dir, batch_size, img_size,
                    val_split=0.2, augment_train=True,
                    label_dir=None, num_workers=None):
    """
    Build train/val DataLoaders with automatic 80/20 split.

    If label_dir is provided, uses HokkaidoDataset (img + label + mask).
    Otherwise uses standard LandslideDataset (img + mask).
    """
    np.random.seed(SEED)

    if label_dir is not None:
        # Hokkaido mode
        full_ds = HokkaidoDataset(img_dir, label_dir, mask_dir,
                                   img_size=img_size, augment=False)
    else:
        full_ds  = LandslideDataset(img_dir, mask_dir,
                                     img_size=img_size, augment=False)

    N       = len(full_ds)
    n_val   = max(1, int(val_split * N))
    n_train = N - n_val

    train_ds_base, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED))

    # Wrap train subset with augmentation via a fresh dataset instance
    if augment_train:
        if label_dir is not None:
            aug_ds = HokkaidoDataset(img_dir, label_dir, mask_dir,
                                      img_size=img_size, augment=True)
        else:
            aug_ds = LandslideDataset(img_dir, mask_dir,
                                       img_size=img_size, augment=True)
        train_ds = Subset(aug_ds, train_ds_base.indices)
    else:
        train_ds = train_ds_base

    nw = (min(4, os.cpu_count() or 1) if num_workers is None
          else max(0, int(num_workers)))
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=nw,
                               pin_memory=True,
                               drop_last=len(train_ds) >= batch_size)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=nw,
                               pin_memory=True)

    print(f"  train={len(train_ds)}  val={len(val_ds)}  "
          f"batch={batch_size}  workers={nw}")
    return train_loader, val_loader


def make_split_indices(num_samples, train_fraction, seed=SEED,
                       val_fraction=0.2, test_fraction=0.2):
    """Return deterministic, directly indexable train/validation/test indices.

    Validation and test indices depend only on ``num_samples`` and ``seed``.
    A smaller training fraction is therefore a strict subset of the 60% pool,
    which makes the four transfer-learning conditions directly comparable.
    """
    if num_samples < 3:
        raise ValueError("At least three samples are required for a three-way split")
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive")

    max_train_fraction = 1.0 - val_fraction - test_fraction
    if max_train_fraction <= 0:
        raise ValueError("Validation and test fractions leave no training data")
    if not 0 < train_fraction <= max_train_fraction + 1e-12:
        raise ValueError(
            f"train_fraction must be in (0, {max_train_fraction:.3f}]")

    n_val = max(1, int(val_fraction * num_samples))
    n_test = max(1, int(test_fraction * num_samples))
    train_pool_size = num_samples - n_val - n_test
    if train_pool_size < 1:
        raise ValueError("Fractions leave no training samples")

    n_train = min(
        train_pool_size,
        max(1, int(round(train_fraction * num_samples))),
    )
    permutation = np.random.default_rng(seed).permutation(num_samples).tolist()
    train_pool = permutation[:train_pool_size]
    val_indices = permutation[train_pool_size:train_pool_size + n_val]
    test_indices = permutation[train_pool_size + n_val:]

    return train_pool[:n_train], val_indices, test_indices


def get_three_way_split(img_dir, mask_dir, img_size, batch_size,
                         label_dir=None, data_fraction=0.6,
                         num_workers=None, augment_train=False):
    """Build a fixed validation/test split and a nested training subset.

    Target-domain augmentation is disabled by default to match the protocol
    described in the landslide transfer-learning paper. Callers may opt in for
    exploratory experiments.
    """
    if label_dir:
        full_ds = HokkaidoDataset(img_dir, label_dir, mask_dir,
                                   img_size=img_size, augment=False)
    else:
        full_ds = LandslideDataset(img_dir, mask_dir,
                                    img_size=img_size, augment=False)

    tr_idx, va_idx, te_idx = make_split_indices(
        len(full_ds), data_fraction, seed=SEED)

    if augment_train:
        if label_dir:
            train_ds = HokkaidoDataset(
                img_dir, label_dir, mask_dir,
                img_size=img_size, augment=True)
        else:
            train_ds = LandslideDataset(
                img_dir, mask_dir, img_size=img_size, augment=True)
    else:
        train_ds = full_ds

    nw = (min(4, os.cpu_count() or 1) if num_workers is None
          else max(0, int(num_workers)))
    tr = DataLoader(Subset(train_ds, tr_idx), batch_size=batch_size,
                    shuffle=True, num_workers=nw, pin_memory=True,
                    drop_last=len(tr_idx) >= batch_size)
    va = DataLoader(Subset(full_ds, va_idx), batch_size=batch_size,
                    shuffle=False, num_workers=nw, pin_memory=True)
    te = DataLoader(Subset(full_ds, te_idx), batch_size=batch_size,
                    shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  3-way split  train={len(tr_idx)}  val={len(va_idx)}  "
          f"test={len(te_idx)}")
    return tr, va, te, Subset(full_ds, te_idx)


def make_loaders(img_dir, mask_dir, batch_size, img_size, num_workers=None):
    """Backward-compatible wrapper used by pretrain.py."""
    return get_dataloaders(img_dir, mask_dir, batch_size=batch_size,
                           img_size=img_size, num_workers=num_workers)


def make_finetune_loaders(img_dir, mask_dir, data_fraction, batch_size,
                          img_size, label_dir=None, num_workers=None,
                          augment_train=False):
    """Backward-compatible wrapper used by finetune.py."""
    return get_three_way_split(img_dir, mask_dir, img_size=img_size,
                               batch_size=batch_size, label_dir=label_dir,
                               data_fraction=data_fraction,
                               num_workers=num_workers,
                               augment_train=augment_train)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    img_dir  = sys.argv[1] if len(sys.argv) > 1 else "dataset/bijie/images"
    mask_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset/bijie/masks"

    if not os.path.exists(img_dir):
        print(f"Path not found: {img_dir}")
        print("Usage: python dataset.py <img_dir> <mask_dir>")
        raise SystemExit

    ds = LandslideDataset(img_dir, mask_dir, img_size=256)
    img, mask = ds[0]
    print(f"Image : {img.shape}  dtype={img.dtype}  "
          f"range=[{img.min():.2f}, {img.max():.2f}]")
    print(f"Mask  : {mask.shape}  dtype={mask.dtype}  "
          f"unique={mask.unique().tolist()}")
    ls_pct = mask.float().mean().item() * 100
    print(f"Landslide pixels: {ls_pct:.1f}%  (expect 4-15%)")
    assert mask.dtype == torch.long,         "FAIL: mask must be torch.long"
    assert set(mask.unique().tolist()) <= {0,1}, "FAIL: mask values not in {0,1}"
    print("dataset.py self-test PASSED")
