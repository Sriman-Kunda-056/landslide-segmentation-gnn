"""
data_prep/prepare_hokkaido.py
Prepares the Hokkaido Iburi coseismic landslide dataset for training.

Raw Hokkaido dataset structure (as distributed):
  hokkaido_raw/
      img/        ← PlanetScope RGB satellite images  (.tif / .jpg)
      label/      ← annotation files (may be GeoJSON polygons or raster .tif)
      mask/       ← binary raster masks (0=background, 255=landslide)

This script:
  1. Reads every image from img/
  2. Finds the matching mask from mask/ (preferred) or converts label/ raster
  3. Resizes both to target_size × target_size
  4. Saves to output directory:
       dataset/hokkaido/images/   (PNG)
       dataset/hokkaido/masks/    (PNG, values 0 and 255)

Usage:
  python data_prep/prepare_hokkaido.py
         --raw_dir  path/to/hokkaido_raw
         --out_dir  dataset/hokkaido
         --size     512
         --verify

After running, check the output:
  python dataset.py dataset/hokkaido/images dataset/hokkaido/masks
"""

import os
import sys
import glob
import argparse
import shutil
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_file(directory, stem, exts=('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
    """Find a file with the given stem in directory, trying multiple extensions."""
    for ext in exts:
        p = os.path.join(directory, stem + ext)
        if os.path.exists(p):
            return p
        # also try upper-case extension
        p = os.path.join(directory, stem + ext.upper())
        if os.path.exists(p):
            return p
    return None


def load_image_rgb(path, size):
    """Load any image format as RGB PIL, resize to (size, size)."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(path).convert("RGB")
        img = img.resize((size, size), PILImage.BILINEAR)
        return img
    except Exception as e:
        print(f"  [warn] Cannot read image {path}: {e}")
        return None


def load_mask(path, size):
    """
    Load a mask file. Handles:
      - 8-bit grayscale (0=bg, 255=landslide)  ← standard
      - 8-bit grayscale (0=bg, 1=landslide)    ← already binary
      - RGB where landslide is a specific colour
    Returns PIL Image in mode "L" with values {0, 255}.
    """
    try:
        mask = Image.open(path).convert("L")
        arr  = np.array(mask, dtype=np.uint8)

        # Normalise: if values are 0/1 → scale to 0/255
        unique = np.unique(arr)
        if set(unique).issubset({0, 1}):
            arr = arr * 255

        # Binarise: anything > 127 is landslide
        arr = (arr > 127).astype(np.uint8) * 255

        mask = Image.fromarray(arr, mode="L")
        mask = mask.resize((size, size), Image.NEAREST)
        return mask
    except Exception as e:
        print(f"  [warn] Cannot read mask {path}: {e}")
        return None


def tif_to_mask(label_path, size):
    """
    Try to load a label .tif as a binary mask.
    Handles single-band rasters where non-zero = landslide.
    """
    # Try with rasterio first (best for GeoTIFF)
    try:
        import rasterio
        with rasterio.open(label_path) as src:
            arr = src.read(1).astype(np.float32)
        arr = (arr > 0).astype(np.uint8) * 255
        mask = Image.fromarray(arr, mode="L")
        mask = mask.resize((size, size), Image.NEAREST)
        return mask
    except ImportError:
        pass

    # Fallback: PIL
    try:
        mask = Image.open(label_path).convert("L")
        arr  = np.array(mask, dtype=np.uint8)
        arr  = (arr > 0).astype(np.uint8) * 255
        mask = Image.fromarray(arr, mode="L")
        mask = mask.resize((size, size), Image.NEAREST)
        return mask
    except Exception as e:
        print(f"  [warn] Cannot convert label {label_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main preparation function
# ─────────────────────────────────────────────────────────────────────────────

def prepare(raw_dir, out_dir, size=512, verify=False):
    img_in   = os.path.join(raw_dir, "img")
    mask_in  = os.path.join(raw_dir, "mask")
    label_in = os.path.join(raw_dir, "label")

    img_out  = os.path.join(out_dir, "images")
    mask_out = os.path.join(out_dir, "masks")
    os.makedirs(img_out,  exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    # Collect all image files
    exts = ['*.tif','*.tiff','*.jpg','*.jpeg','*.png',
            '*.TIF','*.TIFF','*.JPG','*.JPEG','*.PNG']
    img_files = []
    for ext in exts:
        img_files.extend(glob.glob(os.path.join(img_in, ext)))
    img_files = sorted(img_files)

    if not img_files:
        print(f"ERROR: No image files found in {img_in}")
        print("Check that your raw_dir contains an 'img' subfolder.")
        sys.exit(1)

    print(f"Found {len(img_files)} images in {img_in}")
    print(f"Output → {out_dir}  (size={size}×{size})")
    print("-" * 50)

    ok = 0; skip = 0

    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # ── Load image ────────────────────────────────────────────────
        img_pil = load_image_rgb(img_path, size)
        if img_pil is None:
            skip += 1; continue

        # ── Find / load mask ──────────────────────────────────────────
        mask_pil = None

        # Priority 1: mask/ folder (already processed binary raster)
        if os.path.isdir(mask_in):
            mp = find_file(mask_in, stem)
            if mp:
                mask_pil = load_mask(mp, size)

        # Priority 2: label/ folder (may be raw annotation raster)
        if mask_pil is None and os.path.isdir(label_in):
            lp = find_file(label_in, stem)
            if lp:
                mask_pil = tif_to_mask(lp, size)

        if mask_pil is None:
            print(f"  [skip] No mask found for {stem}")
            skip += 1; continue

        # ── Sanity: mask must have some landslide pixels ───────────────
        arr = np.array(mask_pil)
        if arr.max() == 0:
            # all background — include anyway (paper does not filter these)
            pass

        # ── Save ──────────────────────────────────────────────────────
        img_pil.save(os.path.join(img_out,  stem + ".png"))
        mask_pil.save(os.path.join(mask_out, stem + ".png"))
        ok += 1

        if ok % 100 == 0:
            print(f"  Processed {ok} pairs...")

    print(f"\nDone.  Saved: {ok}   Skipped: {skip}")
    print(f"Images → {img_out}")
    print(f"Masks  → {mask_out}")

    # ── Verify ────────────────────────────────────────────────────────
    if verify and ok > 0:
        print("\nRunning dataset verification...")
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from dataset import LandslideDataset
        try:
            ds = LandslideDataset(img_out, mask_out, img_size=size)
            img, mask = ds[0]
            print(f"  image shape:  {tuple(img.shape)}")
            print(f"  mask unique:  {mask.unique().tolist()}  ← must be [0] or [0,1]")
            assert mask.max().item() <= 1, "Mask values > 1 — check binarisation"
            print("  Verification PASSED ✓")
        except Exception as e:
            print(f"  Verification FAILED: {e}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",  type=str, required=True,
                   help="Path to raw Hokkaido folder (contains img/, mask/, label/)")
    p.add_argument("--out_dir",  type=str, default="dataset/hokkaido",
                   help="Where to save processed images and masks")
    p.add_argument("--size",     type=int, default=512,
                   help="Output patch size (paper: 512)")
    p.add_argument("--verify",   action="store_true",
                   help="Run dataset verification after processing")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.raw_dir, args.out_dir, args.size, args.verify)
