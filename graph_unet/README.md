# Graph U-Net for Landslide Segmentation

Adapts **Graph U-Nets** (Gao & Ji, ICML 2019) for pixel-wise landslide
detection from satellite imagery, with transfer learning.

**Paper:** https://proceedings.mlr.press/v97/gao19a/gao19a.pdf
**Reference repo:** https://github.com/HongyangGao/Graph-U-Nets

---

## Architecture

```
Image [B,3,H,W]
    ↓ CNN Feature Extractor (dilated convs, stride=1, preserves H×W)
Per-pixel features [B,feat_dim,H,W] → reshape → [N, feat_dim]  (N=H*W nodes)
    ↓ Build 8-connected grid adjacency A  [N,N]
    ↓ GCN embedding layer  [N, hidden_dim]
    ↓
ENCODER (n_layers × [GCN → gPool]):
  gPool selects top-k nodes by trainable projection p (Eq 2)
  Saves idx for gUnpool; augments adjacency with A² (Eq 4)
    ↓
DECODER (n_layers × [gUnpool → skip_add → GCN]):
  gUnpool restores nodes to original positions using saved idx (Eq 3)
  Skip connection (addition) from encoder
    ↓
Final GCN → per-node logits [N, 2] → reshape → [B, 2, H, W]
```

## Key equations (from paper)

| Eq | Operation | Formula |
|----|-----------|---------|
| 1 | GCN layer | X_{l+1} = σ(D̂^{-½} Â D̂^{-½} X_l W_l),  Â = A + **2**I |
| 2 | gPool | y=Xp/‖p‖, idx=topk(y,k), ỹ=sigmoid(y[idx]), X_{l+1}=X[idx]⊙ỹ |
| 3 | gUnpool | X_{l+1} = distribute(0_{N×C}, X_l, idx) |
| 4 | Graph power | A_{l+1} = A²[idx,idx]  (2-hop connectivity augmentation) |

---

## Project structure

```
graph_unet_landslide/
├── models/
│   ├── graph_ops.py    ← GCNLayer, gPool (Eq2), gUnpool (Eq3), EncoderBlock, DecoderBlock
│   └── graph_unet.py   ← Full GraphUNet + load_pretrained_graph_unet
├── config.py           ← all settings
├── dataset.py          ← image+mask loading
├── loss.py             ← Dice + CE
├── metrics.py          ← Recall, Spec, Prec, F1
├── pretrain.py         ← Phase 1: train on source domain
├── finetune.py         ← Phase 2: all 4 TL conditions
├── test_components.py  ← run this first
└── utils/trainer.py    ← shared train/validate
```

---

## Setup

```bash
conda create -n graphunet python=3.10 && conda activate graphunet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Important: IMG_SIZE limitation

The grid adjacency matrix is **N×N where N = H×W**.

| IMG_SIZE | N nodes | Adjacency size | Memory (float32) |
|----------|---------|----------------|------------------|
| 32 | 1,024 | 1M | 4 MB |
| 64 | 4,096 | 16M | 64 MB ← recommended |
| 128 | 16,384 | 268M | 1 GB ← borderline |
| 256 | 65,536 | 4.3B | 17 GB ← needs sparse |

**Use IMG_SIZE=64 for development, 128 if you have >16GB RAM/VRAM.**

Set in `config.py`: `IMG_SIZE = 64`

---

## Run order

```bash
# Step 1 — verify all components
python test_components.py

# Step 2 — smoke test (5 epochs, tiny Bijie dataset)
# Put images in dataset/bijie/images/  masks in dataset/bijie/masks/
python pretrain.py --dataset bijie --epochs 5 --imgsize 64

# Step 3 — pretrain on source domain
python pretrain.py --dataset bijie --imgsize 64
# saves: save_weights/GraphUNet_pretrained_best.pth

# Step 4 — prepare Hokkaido data
# Your hokkaido_raw/ must have: img/ mask/ label/
python data_prep/prepare_hokkaido.py \
    --raw_dir /path/to/hokkaido_raw \
    --out_dir dataset/hokkaido --size 64

# Step 5 — transfer learning (all 4 conditions)
python finetune.py \
    --pretrained save_weights/GraphUNet_pretrained_best.pth \
    --dataset hokkaido --imgsize 64
```

---

## What each condition tests

| Condition | Data | Transfer | Expected |
|-----------|------|----------|----------|
| 1 | 20% | No | Lowest F1 |
| 2 | 20% | Yes | ≈ Condition 3 |
| 3 | 60% | No | Baseline |
| 4 | 60% | Yes | Best F1 |

**Key finding to verify:** Condition 2 F1 ≈ Condition 3 F1
→ Transfer learning halves labeled data requirement

---

## Comparison: Graph U-Net vs CNN U-Net

| Aspect | CNN UNet (AMMG) | Graph U-Net |
|--------|-----------------|-------------|
| Pooling | MaxPool (uniform) | gPool (learned selection by projection) |
| Unpooling | Bilinear interpolate | gUnpool (exact position restore via idx) |
| Skip connections | Feature concat | Feature add (at exact node positions) |
| Global context | GCN at bottleneck only | GCN at EVERY encoder/decoder level |
| Node selection | None | Top-k by trainable p vector |
| Memory | O(H×W) | O((H×W)²) — use small IMG_SIZE |
