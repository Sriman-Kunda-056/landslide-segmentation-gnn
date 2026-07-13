"""
models/graph_unet.py
Graph U-Net for Landslide Segmentation

Adapts the Gao & Ji (ICML 2019) Graph U-Net from node classification
on citation graphs → pixel-wise segmentation of satellite images.

Key adaptation decisions:
  1. CNN feature extractor (ResNet-style) converts image patches to
     per-pixel feature vectors → these become graph node features
  2. Spatial 8-connected grid adjacency connects neighbouring pixels
  3. gPool/gUnpool handle encoder-decoder pooling on graph
  4. Final per-node predictions → reshaped to [B, 2, H, W] output

Architecture:
  Input image [B, 3, H, W]
       ↓
  CNN Feature Extractor   →  per-pixel features [B*H*W, feat_dim]
       ↓
  Build graph (8-connected grid adjacency)
       ↓
  GCN embedding layer
       ↓
  Encoder: [GCN → gPool] × n_layers
       ↓
  Decoder: [gUnpool → skip_add → GCN] × n_layers
       ↓
  Final GCN → per-pixel logits [N, 2]
       ↓
  Reshape → [B, 2, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .graph_ops import (GCNLayer, EncoderBlock, DecoderBlock,
                         build_grid_adjacency, normalize_adjacency)


# ─────────────────────────────────────────────────────────────────────────────
# CNN Feature Extractor
# Converts image pixels into per-pixel feature vectors for graph nodes
# ─────────────────────────────────────────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    """
    Lightweight CNN that maps an image [B, 3, H, W]
    to per-pixel features [B, feat_dim, H, W].

    Uses dilated convolutions to increase receptive field without
    reducing spatial resolution (needed because we need per-pixel features).

    All convolutions have stride=1 + padding to preserve H, W.
    """

    def __init__(self, in_ch: int = 3, feat_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: local edge/colour features
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            # Block 2: small context
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            # Block 3: medium context (dilation=2, RF=5)
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            # Block 4: wider context (dilation=4, RF=13)
            nn.Conv2d(64, feat_dim, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                         nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)    # [B, feat_dim, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Full Graph U-Net
# ─────────────────────────────────────────────────────────────────────────────

class GraphUNet(nn.Module):
    """
    Graph U-Net for pixel-wise landslide segmentation.

    Args:
        in_ch:       input image channels (3 for RGB)
        num_classes: output classes (2 for landslide/background)
        feat_dim:    CNN output feature dimension per pixel
        hidden_dim:  GCN hidden dimension
        n_layers:    number of encoder/decoder blocks (paper uses 4)
        pool_ratios: list of pool ratios per encoder stage
                     e.g. [0.9, 0.7, 0.6, 0.5] (paper's inductive setting)
        dropout:     dropout rate in GCN layers
        img_size:    spatial size of input patches (H = W = img_size)
    """

    def __init__(self,
                 in_ch:       int   = 3,
                 num_classes: int   = 2,
                 feat_dim:    int   = 64,
                 hidden_dim:  int   = 128,
                 n_layers:    int   = 4,
                 pool_ratios: List  = None,
                 dropout:     float = 0.0,
                 img_size:    int   = 64):
        super().__init__()

        if pool_ratios is None:
            pool_ratios = [0.9, 0.7, 0.6, 0.5]
        assert len(pool_ratios) == n_layers, \
            f"pool_ratios must have {n_layers} entries, got {len(pool_ratios)}"

        self.n_layers  = n_layers
        self.img_size  = img_size
        self.feat_dim  = feat_dim
        self.hidden_dim = hidden_dim

        # ── CNN: image → per-pixel features ───────────────────────────
        self.cnn = CNNFeatureExtractor(in_ch, feat_dim)

        # ── GCN embedding: high-dim pixel features → hidden_dim ────────
        self.embed = GCNLayer(feat_dim, hidden_dim, dropout=dropout)

        # ── ENCODER: n_layers of [GCN → gPool] ─────────────────────────
        self.encoders = nn.ModuleList()
        for i in range(n_layers):
            self.encoders.append(
                EncoderBlock(hidden_dim, hidden_dim,
                              pool_ratio=pool_ratios[i],
                              dropout=dropout)
            )

        # ── DECODER: n_layers of [gUnpool → skip_add → GCN] ────────────
        self.decoders = nn.ModuleList()
        for i in range(n_layers):
            self.decoders.append(
                DecoderBlock(hidden_dim, hidden_dim, dropout=dropout)
            )

        # ── Final GCN → per-pixel logits ───────────────────────────────
        self.final_gcn = GCNLayer(hidden_dim, num_classes,
                                   dropout=0.0, activation=False)

        # Pre-build and cache the grid adjacency for given img_size
        # This avoids rebuilding it every forward pass
        self._A_cache = {}

    def _get_adjacency(self, H: int, W: int,
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (or build+cache) grid adjacency for H×W grid."""
        key = (H, W)
        if key not in self._A_cache:
            A_raw  = build_grid_adjacency(H, W, device)
            A_norm = normalize_adjacency(A_raw)
            self._A_cache[key] = (A_raw, A_norm)
        else:
            A_raw, A_norm = self._A_cache[key]
            # Move to correct device if needed
            A_raw  = A_raw.to(device)
            A_norm = A_norm.to(device)
        return A_raw, A_norm

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: [B, 3, H, W]
        Returns:
            logits: [B, num_classes, H, W]
        """
        B, C, H, W = imgs.shape
        N = H * W    # number of graph nodes per image

        # ── Step 1: CNN → per-pixel features ──────────────────────────
        feat_maps = self.cnn(imgs)            # [B, feat_dim, H, W]

        # Reshape: [B, feat_dim, H, W] → [B*N, feat_dim] → process each image's graph
        # For simplicity we process each image independently
        # (batching over graphs of same size is possible but complex)
        all_logits = []

        A_raw, A_norm = self._get_adjacency(H, W, imgs.device)

        for b in range(B):
            # Extract this image's node features: [N, feat_dim]
            x = feat_maps[b].view(self.feat_dim, N).t()   # [N, feat_dim]

            # ── Step 2: GCN embedding ────────────────────────────────
            x = self.embed(x, A_norm)    # [N, hidden_dim]

            # ── Step 3: Encoder ──────────────────────────────────────
            skip_list = []   # stores (x_pre_pool, A_norm_pre, A_raw_pre, idx) per level
            A_n, A_r = A_norm, A_raw

            for enc in self.encoders:
                x, A_n_pool, A_r_pool, idx, x_pre, A_r_pre = enc(x, A_n, A_r)
                A_n_pre = normalize_adjacency(A_r_pre)
                skip_list.append((x_pre, A_n_pre, A_r_pre, idx))
                A_n, A_r = A_n_pool, A_r_pool

            # ── Step 4: Decoder ──────────────────────────────────────
            for i, dec in enumerate(reversed(self.decoders)):
                x_skip, A_n_skip, A_r_skip, idx = skip_list[-(i+1)]
                x, A_n, A_r = dec(x, A_n, A_r,
                                   idx, x_skip, A_n_skip, A_r_skip)

            # ── Step 5: Final prediction ─────────────────────────────
            logits_n = self.final_gcn(x, A_n)   # [N, num_classes]

            # Reshape back to [num_classes, H, W]
            logits_hw = logits_n.t().view(-1, H, W)   # [num_classes, H, W]
            all_logits.append(logits_hw)

        return torch.stack(all_logits, dim=0)   # [B, num_classes, H, W]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Transfer learning support
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained_graph_unet(model: GraphUNet,
                                 checkpoint_path: str,
                                 device: torch.device,
                                 freeze_cnn: bool = False) -> GraphUNet:
    """
    Load pretrained weights for transfer learning.

    Args:
        model:           GraphUNet instance (same architecture)
        checkpoint_path: path to .pth saved by training
        device:          target device
        freeze_cnn:      if True, freeze CNN feature extractor weights
                         (use when target domain has similar spectral characteristics)
    Returns:
        model with loaded weights
    """
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"  [warn] Missing keys: {missing}")

    if freeze_cnn:
        for param in model.cnn.parameters():
            param.requires_grad = False
        print("  CNN weights frozen — only GCN layers will be updated")

    print(f"  Loaded pretrained weights from {checkpoint_path}")
    return model
