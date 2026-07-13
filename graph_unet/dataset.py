"""
dataset.py — loads landslide image+mask pairs for Graph U-Net training.
Supports Bijie, Hokkaido, Niangniangba, CAS formats.
"""
import os, glob, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from config import MEAN, STD, SEED

class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=64, augment=False):
        self.img_size = img_size
        self.augment  = augment
        self.mean     = np.array(MEAN, dtype=np.float32)
        self.std      = np.array(STD,  dtype=np.float32)

        exts = ('*.jpg','*.jpeg','*.png','*.tif','*.tiff',
                '*.JPG','*.JPEG','*.PNG','*.TIF','*.TIFF')
        imgs = []
        for e in exts:
            imgs.extend(glob.glob(os.path.join(img_dir, e)))
        imgs = sorted(imgs)
        if not imgs:
            raise RuntimeError(f"No images found in {img_dir}")

        self.pairs = []
        for ip in imgs:
            stem = os.path.splitext(os.path.basename(ip))[0]
            for ext in ['.png','.tif','.tiff','.jpg']:
                mp = os.path.join(mask_dir, stem + ext)
                if os.path.exists(mp):
                    self.pairs.append((ip, mp)); break
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
        mask = (mask > 127).astype(np.int64)
        return (torch.tensor(img.transpose(2,0,1), dtype=torch.float32),
                torch.tensor(mask, dtype=torch.long))


def make_loaders(img_dir, mask_dir, batch_size=4, img_size=64,
                 val_split=0.2, num_workers=0, seed=SEED):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    full = LandslideDataset(img_dir, mask_dir, img_size=img_size)
    n    = len(full); nv = max(1, int(n*val_split)); nt = n - nv
    tr_idx, va_idx = random_split(range(n), [nt, nv],
                                  generator=torch.Generator().manual_seed(seed))

    train_ds = LandslideDataset(img_dir, mask_dir, img_size=img_size, augment=True)
    val_ds   = LandslideDataset(img_dir, mask_dir, img_size=img_size, augment=False)

    tl = DataLoader(Subset(train_ds, tr_idx), batch_size=batch_size, shuffle=True,
                     num_workers=num_workers, pin_memory=False)
    vl = DataLoader(Subset(val_ds, va_idx), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=False)
    print(f"[Loaders] train={nt}  val={nv}")
    return tl, vl


def make_finetune_loaders(img_dir, mask_dir, data_fraction=0.6,
                           batch_size=4, img_size=64,
                           num_workers=0, seed=SEED):
    rng  = np.random.RandomState(seed)
    full = LandslideDataset(img_dir, mask_dir, img_size=img_size)
    N    = len(full); idx = rng.permutation(N)
    ntp  = int(0.6*N); nv = int(0.2*N)
    pool = idx[:ntp]; vi = idx[ntp:ntp+nv]; ti2 = idx[ntp+nv:]
    nuse = max(1, min(int(data_fraction*N), len(pool)))
    tridx = pool[:nuse]
    aug  = LandslideDataset(img_dir, mask_dir, img_size=img_size, augment=True)
    tl = DataLoader(Subset(aug,  tridx), batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    vl = DataLoader(Subset(full, vi),    batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tsl= DataLoader(Subset(full, ti2),   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"[Finetune] train={len(tridx)}({data_fraction*100:.0f}%)  val={len(vi)}  test={len(ti2)}")
    return tl, vl, tsl
