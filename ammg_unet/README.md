# AMMG-UNet — Landslide Detection via Transfer Learning & GCN

Implementation of the paper:
> **"A proposed method for landslide detection based on transfer learning
> and graph neural network"**
> Luo et al., Geoscience Frontiers 16 (2025) 102171

---

## Project structure

```
landslide_project/
├── config.py               ← ALL hyperparameters and paths (edit here)
├── dataset.py              ← data loading for all datasets
├── loss.py                 ← Dice + CE combined loss (Eq 6-8)
├── metrics.py              ← Recall, Spec, Prec, F1  (Eq 9-12)
├── pretrain.py             ← Phase 1: pretrain on CAS / Bijie
├── finetune.py             ← Phase 2: transfer learning (4 conditions)
├── test_components.py      ← verify every module before training
├── requirements.txt
├── models/
│   ├── grb.py              ← GraphReasoningBlock  (GCN core)
│   ├── mgrm.py             ← Multiscale Global Reasoning Module
│   ├── attention_conv.py   ← AttentionConv  (Eq 1-4)
│   ├── multiscale.py       ← MultiscaleConnection  (Eq 5)
│   ├── ammg_unet.py        ← Full AMMG-UNet
│   └── baselines.py        ← Vanilla UNet baseline
├── data_prep/
│   └── prepare_hokkaido.py ← converts raw Hokkaido data → ready format
├── utils/
│   └── trainer.py          ← shared train/validate functions
├── dataset/
│   ├── bijie/images/       ← put Bijie images here
│   ├── bijie/masks/
│   ├── hokkaido/images/    ← output of prepare_hokkaido.py
│   ├── hokkaido/masks/
│   └── CAS/                ← CAS dataset (20 GB)
├── save_weights/           ← model checkpoints saved here
└── results/                ← training curves saved here
```

---

## Step 0 — Environment setup

```bash
conda create -n landslide python=3.10
conda activate landslide

# GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# or CPU only
pip install torch torchvision

pip install -r requirements.txt
```

---

## Step 1 — Prepare Hokkaido dataset

Your raw Hokkaido folder must have this structure:
```
hokkaido_raw/
    img/     ← PlanetScope RGB satellite images (.tif or .jpg)
    mask/    ← binary raster masks (0=bg, 255=landslide)
    label/   ← annotation files (used if mask/ is missing)
```

Run:
```bash
python data_prep/prepare_hokkaido.py \
    --raw_dir /path/to/hokkaido_raw \
    --out_dir dataset/hokkaido \
    --size    512 \
    --verify
```

Expected output:
```
Found N images in .../img
Saved: N   Skipped: 0
Verification PASSED ✓
```

---

## Step 2 — Verify all components

```bash
python test_components.py
```

Expected:
```
GRB  ........ PASSED ✓
MGRM ........ PASSED ✓
AttentionConv PASSED ✓
Multiscale .. PASSED ✓
AMGUnet ..... PASSED ✓
Loss ........ PASSED ✓
Metrics ..... PASSED ✓
ALL TESTS PASSED — ready to train
```

---

## Step 3 — Quick smoke test (Bijie dataset, 5 epochs)

Download Bijie: https://github.com/SDU-L/CLPD/tree/main/dataset

```bash
# Put images in dataset/bijie/images/  and masks in dataset/bijie/masks/
python pretrain.py --dataset bijie --epochs 5 --imgsize 256
```

Loss should decrease each epoch. F1 > 0 by epoch 3.

---

## Step 4 — Pretrain on source domain

**Option A — Bijie (fast, ~1-2 hours on GPU):**
```bash
python pretrain.py --dataset bijie --imgsize 256
```

**Option B — CAS dataset (paper-accurate, ~20 hours on GPU):**

Download CAS: https://zenodo.org/records/10294997

```bash
python pretrain.py --dataset cas --epochs 60
# For exact paper reproduction: --epochs 150
```

Checkpoint saved to: `save_weights/AMGUnet_pretrained_best.pth`

---

## Step 5 — Fine-tune on Hokkaido (Transfer Learning)

```bash
python finetune.py \
    --pretrained save_weights/AMGUnet_pretrained_best.pth \
    --dataset    hokkaido \
    --epochs     30
```

This runs all 4 experimental conditions from the paper:
- Condition 1: 20% data, no transfer learning
- Condition 2: 20% data, WITH transfer learning  ← key result
- Condition 3: 60% data, no transfer learning
- Condition 4: 60% data, WITH transfer learning

Expected results table (paper values for reference):
```
Condition          Recall   Spec    Prec    F1
20% data, no TL    82.7%   95.9%   71.6%  76.8%
20% data, WITH TL  78.8%   95.2%   78.6%  78.7%   ← ≈ condition 3
60% data, no TL    76.7%   94.8%   81.0%  78.8%
60% data, WITH TL  80.1%   95.5%   80.3%  80.2%   ← best
```

Key finding: Condition 2 ≈ Condition 3 → transfer learning halves data requirement.

---

## Configuration reference

All settings in `config.py`:

| Setting | Default | Paper value | Notes |
|---------|---------|-------------|-------|
| PRETRAIN_EPOCHS | 60 | 150 | Increase for full reproduction |
| FINETUNE_EPOCHS | 30 | 50 | Increase for full reproduction |
| PRETRAIN_LR | 0.01 | 0.01 | — |
| FINETUNE_LR | 0.001 | 0.001 | 10× lower than pretrain |
| BATCH_SIZE | 4 | 4 | Reduce to 2 if CUDA OOM |
| IMG_SIZE | 512 | 512 | Use 256 for Bijie |
| GCN_RATIO | 4 | 4 | Cr = in_ch // 4 = 128 at bottleneck |

---

## Common errors

| Error | Fix |
|-------|-----|
| `No images found` | Check img_dir path in config.py |
| `Missing masks` | Run prepare_hokkaido.py first |
| `CUDA out of memory` | Set PRETRAIN_BATCH=2 in config.py |
| `Loss = NaN` | Check mask values are 0/255, not 0/1×255 float |
| `F1 stuck at 0` | Check mask dtype is torch.long not torch.float |
| `Shape mismatch in decoder` | Run test_components.py to isolate |

---

## Paper architecture correspondence

| Paper component | File | Class |
|-----------------|------|-------|
| Attention Conv (Eq 1-4) | models/attention_conv.py | AttentionConv |
| Multiscale Connection (Eq 5) | models/multiscale.py | MultiscaleConnection |
| Graph Reasoning Block | models/grb.py | GraphReasoningBlock |
| MGRM (4-path GCN bridge) | models/mgrm.py | MGRM |
| Full AMMG-UNet | models/ammg_unet.py | AMGUnet |
| Loss (Eq 6-8) | loss.py | CombinedLoss |
| Metrics (Eq 9-12) | metrics.py | compute_metrics |
| Transfer learning | finetune.py | run_condition |
