"""
dataset.py — loads landslide image+mask pairs for Graph U-Net training.
Supports Bijie, Hokkaido, Niangniangba, CAS formats.
"""
import os, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from config import MEAN, STD, SEED

class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=64, augment=False):
        self.img_size = img_size
        self.augment  = augment
        self.mean     = np.array(MEAN, dtype=np.float32)
        self.std      = np.array(STD,  dtype=np.float32)

        image_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        mask_exts = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}
        img_root = Path(img_dir)
        mask_root = Path(mask_dir)
        if not img_root.is_dir():
            raise RuntimeError(f"Image directory not found: {img_root}")
        if not mask_root.is_dir():
            raise RuntimeError(f"Mask directory not found: {mask_root}")

        # Scan once and normalize suffix/stem case. Globbing both upper- and
        # lower-case patterns duplicates files on case-insensitive systems.
        imgs = sorted(
            (p for p in img_root.iterdir()
             if p.is_file() and p.suffix.lower() in image_exts),
            key=lambda p: p.name.casefold(),
        )
        if not imgs:
            raise RuntimeError(f"No images found in {img_dir}")

        masks_by_stem = {}
        for path in sorted(mask_root.iterdir(), key=lambda p: p.name.casefold()):
            if path.is_file() and path.suffix.lower() in mask_exts:
                masks_by_stem.setdefault(path.stem.casefold(), path)

        self.pairs = []
        for ip in imgs:
            mp = masks_by_stem.get(ip.stem.casefold())
            if mp is not None:
                self.pairs.append((str(ip), str(mp)))
        if not self.pairs:
            raise RuntimeError(
                f"No matching image/mask stems in {img_root} and {mask_root}"
            )
        missing = len(imgs) - len(self.pairs)
        if missing:
            print(f"[Dataset] WARNING: {missing} images have no matching mask")
        print(f"[Dataset] {len(self.pairs)} pairs from {img_dir}")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img  = np.array(Image.open(ip).convert("RGB")
                        .resize((self.img_size, self.img_size), Image.BILINEAR),
                        dtype=np.float32) / 255.0
        mask = np.array(Image.open(mp).convert("L")
                        .resize((self.img_size, self.img_size), Image.NEAREST),
                        dtype=np.uint8)
        if self.augment:
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy(); mask = mask[:, ::-1].copy()
            if random.random() > 0.5:
                img = img[::-1, :, :].copy(); mask = mask[::-1, :].copy()
            k = random.randint(0, 3)
            if k > 0:
                img  = np.rot90(img,  k, (0,1)).copy()
                mask = np.rot90(mask, k, (0,1)).copy()

        img  = (img - self.mean) / self.std
        # Accept common binary encodings (0/1 and 0/255). Validate nodata
        # handling for a new dataset before training.
        mask = (mask > 0).astype(np.int64)
        return (torch.tensor(img.transpose(2,0,1), dtype=torch.float32),
                torch.tensor(mask, dtype=torch.long))


def make_loaders(img_dir, mask_dir, batch_size=4, img_size=64,
                 val_split=0.2, num_workers=0, seed=SEED):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    full = LandslideDataset(img_dir, mask_dir, img_size=img_size)
    n    = len(full); nv = max(1, int(n*val_split)); nt = n - nv
    order = torch.randperm(
        n, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    tr_idx, va_idx = order[:nt], order[nt:]

    train_ds = LandslideDataset(img_dir, mask_dir, img_size=img_size, augment=True)
    val_ds   = LandslideDataset(img_dir, mask_dir, img_size=img_size, augment=False)

    tl = DataLoader(Subset(train_ds, tr_idx), batch_size=batch_size, shuffle=True,
                     num_workers=num_workers, pin_memory=False)
    vl = DataLoader(Subset(val_ds, va_idx), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=False)
    print(f"[Loaders] train={nt}  val={nv}")
    return tl, vl


def make_finetune_split_indices(num_samples, data_fraction=0.6, seed=SEED):
    """Create nested train subsets with fixed validation and test indices."""
    if num_samples < 5:
        raise ValueError("At least five samples are required")
    if not 0 < data_fraction <= 0.6:
        raise ValueError("data_fraction must be in (0, 0.6]")

    rng  = np.random.RandomState(seed)
    idx = rng.permutation(num_samples)
    ntp = int(0.6 * num_samples)
    nv = int(0.2 * num_samples)
    pool = idx[:ntp]; vi = idx[ntp:ntp+nv]; ti2 = idx[ntp+nv:]
    nuse = max(1, min(int(data_fraction * num_samples), len(pool)))
    return pool[:nuse].tolist(), vi.tolist(), ti2.tolist()


def make_finetune_loaders(img_dir, mask_dir, data_fraction=0.6,
                           batch_size=4, img_size=64,
                           num_workers=0, seed=SEED):
    full = LandslideDataset(img_dir, mask_dir, img_size=img_size)
    tridx, vi, ti2 = make_finetune_split_indices(
        len(full), data_fraction=data_fraction, seed=seed)
    aug  = LandslideDataset(img_dir, mask_dir, img_size=img_size, augment=True)
    tl = DataLoader(Subset(aug,  tridx), batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    vl = DataLoader(Subset(full, vi),    batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tsl= DataLoader(Subset(full, ti2),   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"[Finetune] train={len(tridx)}({data_fraction*100:.0f}%)  "
          f"val={len(vi)}  test={len(ti2)}")
    return tl, vl, tsl
