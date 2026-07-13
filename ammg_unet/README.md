# Simplified AMMG-Style U-Net for Landslide Segmentation

This directory contains an independent, simplified interpretation of ideas from:

> W. Luo, H. Qiu, Y. Wei, et al. "A proposed method for landslide
> detection based on transfer learning and graph neural network."
> Geoscience Frontiers 16 (2025), 102171.
> https://doi.org/10.1016/j.gsf.2025.102171

The paper links its official MIT-licensed implementation at
https://github.com/anon-nameless/TL-landslide_detection.

## Scope and fidelity

This code combines attention downsampling, multiscale skip processing, a
multiscale graph-reasoning bottleneck, and U-Net-style decoding. It is useful as
a modular research prototype, but it is **not** a faithful reproduction of the
paper or its official code.

Key differences include:

- approximately 15.85M parameters here versus 48.1M reported for AMMG-UNet;
- fewer encoder stages and attention blocks;
- a simplified graph reasoning block and MGRM;
- bilinear decoding instead of the official transposed-convolution decoder;
- two-class softmax Dice/cross-entropy instead of a one-logit sigmoid setup;
- ImageNet normalization rather than dataset-specific statistics.

Do not label metrics from this implementation as reproduced paper results.

## Repository contents

```text
ammg_unet/
  config.py                 paths and experiment settings
  dataset.py                image/mask loading and deterministic splits
  loss.py                   Dice + 0.5 * cross-entropy
  metrics.py                recall, specificity, precision, and F1
  pretrain.py               source-domain training
  finetune.py               four target-domain conditions
  test_components.py        model/loss/metric smoke tests
  test_splits.py            split leakage regression tests
  models/
    attention_conv.py
    multiscale.py
    grb.py
    mgrm.py
    ammg_unet.py
    baselines.py
  data_prep/
    prepare_hokkaido.py
  utils/
    trainer.py
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install torch torchvision
pip install -r requirements.txt
```

## Dataset layout

Downloaded data is intentionally excluded from Git.

```text
dataset/
  CAS/
    img/
    mask/
  hokkaido/
    images/
    masks/
    label/        # optional fallback
  bijie/
    images/
    masks/
```

Before pretraining, verify the CAS manifest and explicitly exclude any target
region (for example Hokkaido) to prevent source/target leakage. The paper used
20,865 CAS images; a smaller local subset is not the same experiment.

`data_prep/prepare_hokkaido.py` is a convenience conversion script. It resizes
inputs to square patches and is not equivalent to the paper's geospatial
1-meter resampling and 512x512 tiling workflow. Inspect several image/mask pairs
after conversion before training.

## Validation

```bash
python test_components.py
python -m unittest test_splits.py
```

These tests check tensor shapes, finite gradients, loss/metrics behavior, and
that the 20% training set is nested within the 60% pool while validation/test
sets remain fixed and disjoint. They do not establish architectural fidelity or
scientific accuracy.

## Pretrain

```bash
python pretrain.py --dataset cas --epochs 150 --imgsize 512
```

For a short plumbing check:

```bash
python pretrain.py --dataset bijie --epochs 5 --imgsize 256
```

The source-domain defaults follow the paper-inspired SGD configuration:
momentum 0.9, weight decay 0.001, learning rate 0.01, and cubic polynomial
decay.

## Fine-tune

```bash
python finetune.py \
  --pretrained save_weights/AMGUnet_pretrained_best.pth \
  --dataset hokkaido \
  --epochs 50
```

The four conditions use a fixed held-out validation/test split:

| Condition | Training fraction | Initialization |
| --- | ---: | --- |
| 1 | 20% | Random |
| 2 | 20% | Pretrained |
| 3 | 60% | Random |
| 4 | 60% | Pretrained |

All target-domain conditions use the same optimizer settings. Random flips are
off by default; pass `--augment` only for a separately reported exploratory
experiment.

## Attribution

The official repository's MIT notice is preserved in
`../licenses/TL-landslide_detection-MIT.txt`. Cite both the paper and any
upstream code used in derived work.
