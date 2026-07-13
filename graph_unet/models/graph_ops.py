"""
models/graph_ops.py
Core Graph U-Net operations — implemented from paper equations exactly.

Paper: "Graph U-Nets" Gao & Ji, ICML 2019
       https://proceedings.mlr.press/v97/gao19a/gao19a.pdf

Three primitives:
  1. GCNLayer   — Eq 1  (with paper's A = A + 2I modification)
  2. gPool      — Eq 2  (top-k node selection via trainable projection)
  3. gUnpool    — Eq 3  (inverse: restore nodes using saved idx)

All written in pure PyTorch — no PyTorch Geometric, no DGL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build adjacency from 2D pixel grid
# ─────────────────────────────────────────────────────────────────────────────

def build_grid_adjacency(H: int, W: int,
                          device: torch.device) -> torch.Tensor:
    """
    Build an 8-connected adjacency matrix for an H×W pixel grid.
    Node i = pixel at (row=i//W, col=i%W).

    Returns:
        A: [N, N] sparse-dense float tensor, values 0 or 1
           where N = H*W
    """
    N = H * W
    # Build COO edge list
    rows, cols = [], []
    for r in range(H):
        for c in range(W):
            i = r * W + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        j = nr * W + nc
                        rows.append(i)
                        cols.append(j)
    rows = torch.tensor(rows, dtype=torch.long, device=device)
    cols = torch.tensor(cols, dtype=torch.long, device=device)

    A = torch.zeros(N, N, device=device)
    A[rows, cols] = 1.0
    return A   # [N, N]


def normalize_adjacency(A: torch.Tensor,
                         self_loop_weight: float = 2.0) -> torch.Tensor:
    """
    Normalise adjacency for GCN layer.
    Paper uses Â = A + 2I (double self-loop weight, Section 3.5).

    Standard GCN normalisation: D̂^{-1/2} · Â · D̂^{-1/2}

    Args:
        A: [N, N] adjacency matrix
        self_loop_weight: 2.0 as per paper (vs 1.0 in standard GCN)
    Returns:
        A_norm: [N, N] normalised adjacency
    """
    N = A.shape[0]
    # Â = A + self_loop_weight * I
    A_hat = A + self_loop_weight * torch.eye(N, device=A.device)

    # D̂ = degree matrix of Â
    D_vec  = A_hat.sum(dim=1)           # [N]
    D_inv_sqrt = D_vec.pow(-0.5)
    D_inv_sqrt[D_inv_sqrt == float('inf')] = 0.0

    # D̂^{-1/2} · Â · D̂^{-1/2}
    A_norm = D_inv_sqrt.unsqueeze(1) * A_hat * D_inv_sqrt.unsqueeze(0)
    return A_norm   # [N, N]


def graph_power_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Compute 2nd graph power: A² = A @ A  (Eq 4 in paper).
    Used after gPool to restore connectivity among sampled nodes.

    Returns binarised A² (cap at 1 to avoid large weights).
    """
    A2 = torch.mm(A, A)
    # Binarise: any path of length ≤ 2 → edge exists
    return (A2 > 0).float()


# ─────────────────────────────────────────────────────────────────────────────
# 1. GCN Layer  — Equation 1 (with paper's Â = A + 2I)
# ─────────────────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    Single GCN layer.

    Paper Eq 1:
        X_{l+1} = σ( D̂^{-1/2} · Â · D̂^{-1/2} · X_l · W_l )
        Â = A + 2I   (paper's modification for stronger self-loops)

    In our implementation the adjacency normalisation is pre-computed
    outside this layer to avoid redundant computation.

    Args:
        in_feats:  input feature dimension  C_in
        out_feats: output feature dimension C_out
        dropout:   dropout rate on features (paper: 0.08 for node clf tasks)
        activation: if True apply ReLU; use False for last layer
    """

    def __init__(self, in_feats: int, out_feats: int,
                 dropout: float = 0.0,
                 activation: bool = True):
        super().__init__()
        self.W          = nn.Linear(in_feats, out_feats, bias=False)
        self.dropout    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = activation
        self.bn         = nn.BatchNorm1d(out_feats)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      [N, C_in]   node feature matrix
            A_norm: [N, N]      pre-normalised adjacency (from normalize_adjacency)
        Returns:
            [N, C_out]
        """
        x = self.dropout(x)
        # Aggregate: Â_norm · X  then transform: · W
        # = A_norm @ x @ W  (standard GCN message passing)
        x = torch.mm(A_norm, x)     # [N, C_in]  — aggregate neighbours
        x = self.W(x)               # [N, C_out] — linear transform
        x = self.bn(x)
        if self.activation:
            x = F.relu(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2. gPool — Graph Pooling Layer  — Equation 2
# ─────────────────────────────────────────────────────────────────────────────

class gPool(nn.Module):
    """
    Graph Pooling layer.

    Paper Eq 2:
        y        = X_l · p / ||p||        # scalar projection per node
        idx      = rank(y, k)             # top-k node indices
        ỹ        = sigmoid( y[idx] )      # gate values in (0,1)
        X̃_l      = X_l[idx, :]            # selected node features
        A_{l+1}  = A²[idx, idx]           # subgraph (with graph power)
        X_{l+1}  = X̃_l ⊙ (ỹ · 1_C^T)   # gate features by score

    The trainable projection vector p has the same dimension as node features.
    The gate operation (⊙ with ỹ) makes p trainable via backprop.

    Args:
        in_feats: feature dimension of input nodes
        ratio:    fraction of nodes to keep (e.g. 0.9 keeps top 90%)
                  OR fixed integer k if ratio > 1
    """

    def __init__(self, in_feats: int, ratio: float = 0.5):
        super().__init__()
        self.ratio = ratio
        # Trainable projection vector p ∈ R^{in_feats}
        self.p = nn.Parameter(torch.randn(in_feats))
        nn.init.xavier_uniform_(self.p.unsqueeze(0))

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Args:
            x: [N, C]   node feature matrix
            A: [N, N]   adjacency matrix (UNnormalised — we normalise inside)
        Returns:
            x_pool:   [k, C]   pooled node features
            A_pool:   [k, k]   pooled adjacency (with graph power augmentation)
            idx:      [k]      indices of selected nodes in original N-node graph
            x_orig:   [N, C]   original x (needed by gUnpool)
            A_orig:   [N, N]   original A (needed by gUnpool)
        """
        N, C = x.shape

        # Determine k
        if self.ratio <= 1.0:
            k = max(1, int(N * self.ratio))
        else:
            k = min(int(self.ratio), N)

        # ── Step 1: scalar projection y = X·p / ||p|| ─────────────────
        p_norm = F.normalize(self.p.unsqueeze(0), dim=1).squeeze(0)  # [C]
        y = torch.mv(x, p_norm)   # [N]  — scalar score per node

        # ── Step 2: select top-k nodes ────────────────────────────────
        # topk returns values and indices; we only need indices
        _, idx = torch.topk(y, k, dim=0)          # [k]
        idx, _ = idx.sort()   # preserve order (paper: "index selection preserves order")

        # ── Step 3: gate values ỹ = sigmoid(y[idx]) ──────────────────
        y_tilde = torch.sigmoid(y[idx])            # [k]

        # ── Step 4: extract selected node features ────────────────────
        x_tilde = x[idx, :]                        # [k, C]

        # ── Step 5: graph power augmentation + extract subgraph ───────
        # A² = A @ A  (connects nodes up to 2 hops, Eq 4)
        A2     = graph_power_adjacency(A)          # [N, N]
        A_pool = A2[idx][:, idx]                   # [k, k]

        # ── Step 6: gate features ─────────────────────────────────────
        # X_{l+1} = X̃_l ⊙ (ỹ · 1_C^T)
        # ỹ[:, None] broadcasts [k,1] * [k,C] → [k,C]
        x_pool = x_tilde * y_tilde.unsqueeze(1)    # [k, C]

        return x_pool, A_pool, idx, x, A


# ─────────────────────────────────────────────────────────────────────────────
# 3. gUnpool — Graph Unpooling Layer  — Equation 3
# ─────────────────────────────────────────────────────────────────────────────

class gUnpool(nn.Module):
    """
    Graph Unpooling layer — inverse of gPool.

    Paper Eq 3:
        X_{l+1} = distribute( 0_{N×C}, X_l, idx )

    Creates a zero matrix of original size N×C, then places
    the k pooled node features back at their original positions (idx).
    Nodes not in idx remain zero (they will receive information via skip connection).

    No trainable parameters — pure structural operation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_pool: torch.Tensor,
                idx:    torch.Tensor,
                x_orig: torch.Tensor,
                A_orig: torch.Tensor) -> tuple:
        """
        Args:
            x_pool: [k, C]   pooled node features (output of gPool)
            idx:    [k]      positions in original N-node graph
            x_orig: [N, C]   original features before pooling (for skip connection)
            A_orig: [N, N]   original adjacency (restored)
        Returns:
            x_unpool: [N, C]  features restored to original N-node graph
            A_orig:   [N, N]  restored adjacency
        """
        N, C = x_orig.shape
        k    = x_pool.shape[0]

        # distribute(0_{N×C}, X_pool, idx)
        x_unpool = torch.zeros(N, C, device=x_pool.device, dtype=x_pool.dtype)
        x_unpool[idx] = x_pool   # place pooled features at original positions

        return x_unpool, A_orig


# ─────────────────────────────────────────────────────────────────────────────
# Encoder/Decoder blocks
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    One encoder stage: GCN → gPool
    Returns pooled graph + all info needed for unpooling.
    """
    def __init__(self, in_feats: int, out_feats: int,
                 pool_ratio: float = 0.5, dropout: float = 0.0):
        super().__init__()
        self.gcn   = GCNLayer(in_feats, out_feats, dropout=dropout)
        self.gpool = gPool(out_feats, ratio=pool_ratio)

    def forward(self, x, A_norm, A_raw):
        """
        Args:
            x:      [N, C_in]
            A_norm: [N, N]  normalised adjacency for GCN
            A_raw:  [N, N]  unnormalised adjacency for gPool
        Returns:
            x_pool, A_pool_norm, A_pool_raw, idx, x_pre_pool, A_pre_pool_raw
        """
        # GCN aggregation
        x = self.gcn(x, A_norm)             # [N, C_out]

        # gPool: select top-k nodes
        x_pool, A_pool_raw, idx, x_pre, A_pre_raw = self.gpool(x, A_raw)
        # [k, C_out],  [k, k],  [k],  [N, C_out],  [N, N]

        # Normalise pooled adjacency for next GCN
        A_pool_norm = normalize_adjacency(A_pool_raw)

        return x_pool, A_pool_norm, A_pool_raw, idx, x_pre, A_pre_raw


class DecoderBlock(nn.Module):
    """
    One decoder stage: gUnpool → skip-add → GCN
    """
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.0):
        super().__init__()
        self.gunpool = gUnpool()
        self.gcn     = GCNLayer(in_feats, out_feats, dropout=dropout)

    def forward(self, x_pool, A_pool_norm, A_pool_raw,
                idx, x_skip, A_skip_norm, A_skip_raw):
        """
        Args:
            x_pool:      [k, C]   current pooled features
            idx:         [k]      positions from gPool
            x_skip:      [N, C]   skip connection (encoder features before pooling)
            A_skip_norm: [N, N]   normalised adjacency of original graph
            A_skip_raw:  [N, N]   raw adjacency of original graph
        Returns:
            x_out: [N, C_out]
            A_norm, A_raw for original N-node graph
        """
        # Restore nodes to original positions (zeros for unselected)
        x_unpool, _ = self.gunpool(x_pool, idx, x_skip, A_skip_raw)
        # x_unpool: [N, C]

        # Skip connection: add encoder features (paper: addition operation)
        x_merged = x_unpool + x_skip       # [N, C]

        # GCN on restored graph
        x_out = self.gcn(x_merged, A_skip_norm)   # [N, C_out]

        return x_out, A_skip_norm, A_skip_raw
