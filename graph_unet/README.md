# Graph U-Net Landslide Experiments

Two exploratory Graph U-Net variants for binary landslide segmentation are
included here. They share the Graph U-Net idea, but they use different graph
representations and produce incompatible checkpoints.

| Variant | Graph nodes | Output | Entry point |
| --- | --- | --- | --- |
| Pixel-grid prototype | Every resized pixel in an 8-neighbor grid | Two logits per pixel | `pretrain.py`, `finetune.py` |
| Superpixel notebook | About 500 SLIC superpixels per image | One binary logit per superpixel | `graph_unet_colab.ipynb` |

Both variants are custom adaptations inspired by Gao and Ji's
[Graph U-Nets](https://proceedings.mlr.press/v97/gao19a.html). They are not
exact reproductions of the authors' citation-network code, and they are not the
AMMG-UNet architecture from the landslide transfer-learning paper.

## Important limitations

- The Python package uses dense adjacency matrices and dense graph-power
  operations. Memory grows as `O((H*W)^2)`, while the graph-power matrix
  multiplication can grow cubically in the node count. The safe default is
  therefore only `16x16`, suitable for component testing rather than
  publication-quality mapping. A sparse implementation is required for useful
  high-resolution pixel grids.
- The notebook evaluates superpixel nodes with equal weight. Its F1 score is
  not a pixel-weighted segmentation F1 and must not be compared directly with
  pixel-level paper results.
- The package trains Bijie -> Hokkaido; the notebook demonstrates
  Hokkaido -> Bijie. These are separate experimental protocols.
- Dataset masks are treated as binary when their value is greater than zero.
  Verify nodata and label encoding before training.
- No downloaded dataset, checkpoint, or generated result is versioned here.
  The notebook outputs were intentionally cleared.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install torch torchvision
pip install -r requirements.txt
```

Expected package data layout:

```text
dataset/
  bijie/
    images/
    masks/
  hokkaido/
    images/
    masks/
```

Dataset directories are excluded by `.gitignore`.

## Validate the package

```bash
python test_components.py
python -m unittest test_dataset.py
```

The component test uses a small synthetic graph. It checks adjacency,
gPool/gUnpool, forward and backward passes, loss, and metrics. It does not
validate scientific accuracy or the dataset protocol.

## Run the pixel-grid prototype

```bash
python pretrain.py --dataset bijie --epochs 5 --imgsize 16

python finetune.py \
  --pretrained save_weights/GraphUNet_pretrained_best.pth \
  --dataset hokkaido \
  --imgsize 16
```

The four fine-tuning conditions use a fixed 60/20/20 split. The 20% and 60%
labels refer to the fraction of the full dataset used for training; validation
and test sets remain fixed.

## Run the superpixel notebook

Set the dataset roots before opening the notebook:

```bash
# PowerShell
$env:HOKKAIDO_ROOT = "D:\path\to\hokkaido"
$env:BIJIE_ROOT = "D:\path\to\bijie"

jupyter lab graph_unet_colab.ipynb
```

On Linux/macOS, export the same variables with `export NAME=value`. The
notebook also accepts `dataset/hokkaido` and `dataset/bijie` relative to its
working directory.

## References

- H. Gao and S. Ji, "Graph U-Nets," ICML 2019.
  https://proceedings.mlr.press/v97/gao19a.html
- Reference implementation:
  https://github.com/HongyangGao/Graph-U-Nets
