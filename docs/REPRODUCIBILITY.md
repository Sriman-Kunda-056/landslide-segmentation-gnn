# Reproducibility and Methodology Notes

This document records what the code does, what has been validated, and what must
still be controlled before reporting scientific results.

## 1. Paper comparison

The landslide paper proposes AMMG-UNet: a U-Net-derived architecture with
attention convolution, multiscale connections, and a multiscale global
reasoning module (MGRM) made from graph reasoning blocks. It pretrains on the
CAS dataset, excludes Hokkaido source samples, and fine-tunes on target areas.

The implementation in `ammg_unet/` is a simplified interpretation.

| Aspect | Paper / official release | This repository |
| --- | --- | --- |
| Reported parameters | 48.1M | 15,849,914 |
| Encoder | Full-resolution stage plus deeper double-attention stages | Four single attention-downsampling stages |
| Bottleneck | Larger MGRM with additional convolutions and fuller graph reasoning | Compact four-branch factorized graph block |
| Decoder | Transposed convolution with attention blocks | Bilinear upsampling with double convolution |
| Output/loss | One sigmoid logit with binary losses | Two softmax logits with Dice + cross-entropy |
| Normalization | Dataset-specific | ImageNet mean/std |

Consequently, use "simplified AMMG-style prototype" in reports. Do not call it
the paper architecture or use its metrics as reproduced paper values.

The parameter-scale ratio is `15,849,914 / 48,100,000 = 33.0%`. This is a size
comparison, not a measure of architectural fidelity or predictive accuracy.

## 2. Graph U-Net variants

The package and notebook in `graph_unet/` are different experiments.

| Aspect | Pixel-grid package | Superpixel notebook |
| --- | --- | --- |
| Nodes | Every resized pixel | SLIC superpixels |
| Features | CNN pixel features | Mean RGB per superpixel |
| Default depth | Four pooling levels | Two pooling levels |
| Skip merge | Addition | Concatenation |
| Output | Two logits per pixel | One logit per superpixel |
| Loss | Dice + cross-entropy | Dice + BCE |
| Metric weighting | Pixels | Superpixel nodes |

Their checkpoints and metrics are incompatible. The notebook is inspired by
Graph U-Nets but is not an exact implementation of the reference repository.

## 3. Split protocol

The AMMG-style four-condition workflow now uses one deterministic permutation:

- 20% of the complete dataset for limited-data training;
- 60% of the complete dataset for full training;
- a fixed 20% validation set;
- a fixed 20% test set.

The 20% training indices are a subset of the 60% pool. Validation and test
indices are identical across all four conditions and disjoint from training.
Target-domain augmentation is off by default, and scratch/pretrained conditions
use the same target optimizer settings. Random generators are reset for every
condition so execution order does not change initialization or shuffling.

The Graph U-Net package similarly keeps validation/test indices fixed across its
four conditions.

For a publication, persist the actual filenames and SHA-256 checksums, not just
the seed.

## 4. Dataset controls

Before training:

1. Record the dataset release, download URL, license, and checksum.
2. Verify every image/mask pair by normalized filename stem.
3. Inspect mask values, nodata values, class balance, and geospatial alignment.
4. Group related tiles/scenes before splitting to prevent spatial leakage.
5. Remove every target-region sample from the source-domain manifest.
6. Record spatial resolution, resampling, cropping, and augmentation.

The local CAS copy inspected during packaging contained 4,273 matched
image-mask pairs, compared with 20,865 images stated for the paper source
dataset: `4,273 / 20,865 = 20.5%`. This coverage ratio does not establish that
the files use the identical release or preprocessing. It must not be described
as the same source experiment without a verified manifest.

The included Hokkaido preparation helper performs generic resizing. It does not
reproduce the paper's geospatial 1-meter resampling and 512x512 tiling.

## 5. Metric rules

Report at least recall, specificity, precision, and F1 with a clear averaging
unit:

- pixel-grid models: aggregate confusion counts over pixels;
- superpixel notebook: current metrics aggregate over nodes.

Do not compare node-weighted and pixel-weighted F1 values as if they were the
same quantity. Report the threshold, positive class, ignored/nodata pixels, and
whether metrics are per-image or globally aggregated.

## 6. Validation performed for this release

The following checks passed locally:

```text
AMMG-style component tests:       passed
AMMG split regression tests:      4 passed
Graph U-Net component tests:      passed at 16x16
Graph dataset regression test:    passed
Python AST parsing:               32 files passed
Training CLI import/help checks:  passed
Notebook JSON/output audit:       26 cells, 0 outputs, 0 execution counts
```

These are software checks. Full dataset training was not rerun as part of
repository packaging, so no scientific benchmark is released.

## 7. Known limitations

- Dense pixel-grid Graph U-Net operations are not scalable; use a sparse graph
  representation before increasing image size.
- Dataset split manifests are deterministic but are not yet exported as files.
- No multi-seed confidence intervals are produced.
- The preprocessing helper is not a geospatial reproduction pipeline.
- The AMMG-style architecture is materially smaller than the paper model.
- A repository-wide license still needs to be selected by the repository owner.

## 8. Minimum result checklist

Before publishing a result, include:

- commit hash and clean Git status;
- environment/package lock and hardware;
- dataset/manifests/checksums;
- preprocessing and split-generation command;
- random seeds and number of runs;
- checkpoint-selection rule;
- held-out test metrics with averaging unit;
- training curves and failure cases;
- a statement of architectural differences from cited work.
