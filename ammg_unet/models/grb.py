"""
models/grb.py  —  Graph Reasoning Block (GRB)

Paper: AMMG-UNet, placed inside MGRM at the bottleneck.
Based on Li et al. 2021 and Chen et al. 2019.

6-step pipeline:
  1. W1, W2 project X → V, Z  (1×1 conv, C → Cr)
  2. Reshape [B,Cr,H,W] → [B,Cr,N]  where N=H*W
  3. A  = V_flat × Z_flat^T   [B,Cr,Cr]   ← factorised adjacency
  4. A' = ReLU(A × W_gcn)     [B,Cr,Cr]   ← GCN message passing
  5. M  = W3( A' × Z_flat )   [B,C,H,W]   ← back to spatial
  6. out = BN(X + M)           [B,C,H,W]   ← residual
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphReasoningBlock(nn.Module):
    def __init__(self, in_ch: int, ratio: int = 4):
        super().__init__()
        self.Cr  = max(in_ch // ratio, 16)
        self.W1  = nn.Conv2d(in_ch, self.Cr, 1, bias=False)
        self.W2  = nn.Conv2d(in_ch, self.Cr, 1, bias=False)
        self.Wgcn = nn.Linear(self.Cr, self.Cr, bias=False)
        self.W3  = nn.Conv2d(self.Cr, in_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(in_ch)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(self, X):
        B, C, H, W = X.shape
        N  = H * W

        # Step 1: project to latent space
        V  = self.W1(X)                        # [B, Cr, H, W]
        Z  = self.W2(X)                        # [B, Cr, H, W]

        # Step 2: flatten spatial → node list
        Vf = V.view(B, self.Cr, N)             # [B, Cr, N]
        Zf = Z.view(B, self.Cr, N)             # [B, Cr, N]

        # Step 3: factorised adjacency [B,Cr,N]×[B,N,Cr] → [B,Cr,Cr]
        A  = torch.bmm(Vf, Zf.transpose(1, 2))
        A  = A / (self.Cr ** 0.5)             # scale: prevents NaN

        # Step 4: GCN message passing
        Ap = F.relu(self.Wgcn(A))             # [B, Cr, Cr]

        # Step 5: project back to spatial
        out = torch.bmm(Ap, Zf)               # [B, Cr, N]
        M   = self.W3(out.view(B, self.Cr, H, W))  # [B, C, H, W]

        # Step 6: residual addition
        return self.bn(X + M)                 # [B, C, H, W]
