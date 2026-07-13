"""
models/mgrm.py  —  Multiscale Global Reasoning Module (MGRM)

4 parallel GRBs at pool scales {1, 2, 3, 5}.
Placed at the encoder-decoder bottleneck in AMMG-UNet.

Branch 1: no pool   — 32×32 = 1024 nodes (fine detail)
Branch 2: pool 2×2  — 16×16 =  256 nodes
Branch 3: pool 3×3  — 10×10 =  100 nodes
Branch 4: pool 5×5  —  6×6  =   36 nodes (global structure)

All outputs upsampled to H×W, concatenated → 4C → 1×1 conv → C.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .grb import GraphReasoningBlock


class MGRM(nn.Module):
    def __init__(self, in_ch: int, grb_ratio: int = 4):
        super().__init__()
        self.grb1  = GraphReasoningBlock(in_ch, ratio=grb_ratio)

        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.grb2  = GraphReasoningBlock(in_ch, ratio=grb_ratio)

        self.pool3 = nn.MaxPool2d(3, stride=3)
        self.grb3  = GraphReasoningBlock(in_ch, ratio=grb_ratio)

        self.pool5 = nn.MaxPool2d(5, stride=5)
        self.grb4  = GraphReasoningBlock(in_ch, ratio=grb_ratio)

        self.proj  = nn.Sequential(
            nn.Conv2d(4 * in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        o1 = self.grb1(x)

        o2 = self.grb2(self.pool2(x))
        o2 = F.interpolate(o2, (H, W), mode='bilinear', align_corners=False)

        o3 = self.grb3(self.pool3(x))
        o3 = F.interpolate(o3, (H, W), mode='bilinear', align_corners=False)

        o4 = self.grb4(self.pool5(x))
        o4 = F.interpolate(o4, (H, W), mode='bilinear', align_corners=False)

        return self.proj(torch.cat([o1, o2, o3, o4], dim=1))
