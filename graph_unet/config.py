"""
config.py — all settings for Graph U-Net landslide project
"""
import torch, os

# ── Paths ──────────────────────────────────────────────────────────────────
BIJ_IMG_DIR   = "dataset/bijie/images"
BIJ_MASK_DIR  = "dataset/bijie/masks"
HOK_IMG_DIR   = "dataset/hokkaido/images"
HOK_MASK_DIR  = "dataset/hokkaido/masks"
NGB_IMG_DIR   = "dataset/niangniangba/images"
NGB_MASK_DIR  = "dataset/niangniangba/masks"
CAS_IMG_DIR   = "dataset/cas/images"
CAS_MASK_DIR  = "dataset/cas/masks"
SAVE_DIR      = "save_weights"
RESULTS_DIR   = "results"

# ── Image settings ─────────────────────────────────────────────────────────
# IMPORTANT: this prototype uses dense adjacency and graph-power operations.
# Keep the grid tiny. Useful image resolutions require a sparse redesign.
IMG_SIZE      = 16          # dense prototype; larger grids become very expensive
IN_CHANNELS   = 3
NUM_CLASSES   = 2
MEAN          = [0.485, 0.456, 0.406]
STD           = [0.229, 0.224, 0.225]

# ── Graph U-Net architecture ────────────────────────────────────────────────
FEAT_DIM      = 64           # CNN output channels (node feature dim)
HIDDEN_DIM    = 128          # GCN hidden dimension
N_LAYERS      = 4            # encoder/decoder depth (paper: 4)
POOL_RATIOS   = [0.9, 0.7, 0.6, 0.5]   # fraction of nodes kept at each level
GCN_DROPOUT   = 0.0          # dropout in GCN layers

# ── Training ───────────────────────────────────────────────────────────────
PRETRAIN_EPOCHS  = 50
PRETRAIN_LR      = 0.01
PRETRAIN_BATCH   = 4         # keep small — graph processing is memory intensive
FINETUNE_EPOCHS  = 25
FINETUNE_LR      = 0.001
FINETUNE_BATCH   = 4
MOMENTUM         = 0.9
WEIGHT_DECAY     = 1e-3
LR_POWER         = 3
GRAD_CLIP        = 1.0
DROPOUT          = 0.1

# ── Loss ───────────────────────────────────────────────────────────────────
DICE_SMOOTH   = 1e-5
CE_WEIGHT     = 0.5

# ── Experimental conditions ─────────────────────────────────────────────────
SMALL_DATA_FRAC = 0.20
FULL_DATA_FRAC  = 0.60

# ── Misc ───────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = 42
LOG_INTERVAL  = 5
SAVE_INTERVAL = 10

os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
