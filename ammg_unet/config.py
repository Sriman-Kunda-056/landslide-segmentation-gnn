# =============================================================================
# config.py  —  All hyperparameters, paths, and settings in one place.
# Change values HERE only — never hard-code numbers inside other files.
# =============================================================================

import torch, os

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "AMGUnet"   # "AMGUnet" | "UNet"
IN_CHANNELS  = 3           # RGB
NUM_CLASSES  = 2           # 0=background  1=landslide
IMG_SIZE     = 512         # paper: 512×512  |  use 256 if GPU OOM
FEATURES     = [64,128,256,512]   # channel progression per encoder level
GRB_RATIO    = 4           # Cr = C // GRB_RATIO  (e.g. 512//4 = 128)
DROPOUT_RATE = 0.1

# ── ImageNet normalisation ────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Pretraining  (Phase 1 — large CAS source dataset) ────────────────────────
PRETRAIN_IMG_DIR  = "dataset/CAS/img"
PRETRAIN_MASK_DIR = "dataset/CAS/mask"
CAS_IMG_DIR       = PRETRAIN_IMG_DIR
CAS_MASK_DIR      = PRETRAIN_MASK_DIR
PRETRAIN_EPOCHS   = 150    # paper: 150  |  set 30 for quick smoke-test
PRETRAIN_LR       = 0.01
PRETRAIN_BATCH    = 4      # reduce to 2 if CUDA OOM
PRETRAIN_WORKERS  = 2
SAVE_INTERVAL     = 25

# ── Fine-tuning  (Phase 2 — Hokkaido target dataset) ─────────────────────────
#    Hokkaido folder structure:
#        dataset/hokkaido/images/  ← output of prepare_hokkaido.py
#        dataset/hokkaido/masks/   ← output of prepare_hokkaido.py
#    The loader tries mask/ first, then label/ as fallback.
FINETUNE_IMG_DIR   = "dataset/hokkaido/images"
FINETUNE_LABEL_DIR = "dataset/hokkaido/label"
FINETUNE_MASK_DIR  = "dataset/hokkaido/masks"
HOK_IMG_DIR        = FINETUNE_IMG_DIR
HOK_MASK_DIR       = FINETUNE_MASK_DIR
FINETUNE_EPOCHS    = 50    # paper: 50   |  set 20 for quick smoke-test
FINETUNE_LR        = 0.001 # 10× lower than pretrain
FINETUNE_BATCH     = 4
FINETUNE_WORKERS   = 2
LOG_INTERVAL       = 10
SMALL_DATA_FRAC    = 0.2   # 20% data experiments
FULL_DATA_FRAC     = 0.6   # 60% data experiments

# ── Bijie quick-test dataset ──────────────────────────────────────────────────
BIJIE_IMG_DIR  = "dataset/bijie/images"
BIJIE_MASK_DIR = "dataset/bijie/masks"
BIJIE_EPOCHS   = 50
BIJIE_BATCH    = 4

# ── Optimizer ─────────────────────────────────────────────────────────────────
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.001   # λ for L2 regularisation
LR_POWER     = 3       # polynomial LR decay exponent (cubic)
GRAD_CLIP    = 1.0

# ── Paths ─────────────────────────────────────────────────────────────────────
SAVE_DIR    = "save_weights"
RESULTS_DIR = "results"
os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
