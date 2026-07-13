"""
models/attention_conv.py  —  Attention Convolution (Eq 1-4)

Replaces DoubleConv + MaxPool in each encoder stage.
Stride=2 is inside Wv → handles downsampling.

Eq 1: phi = sigmoid( psi( ReLU( Wq(x) * Wk(x) ) ) )
Eq 2: out = phi * Wv(x)          [Wv has stride=2]
Eq 3: sigmoid(x) = 1/(1+e^-x)
Eq 4: ReLU(x)    = max(0,x)

Input:  [B, in_ch,  H,   W  ]
Output: [B, out_ch, H/2, W/2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        sub = max(in_ch // 4, 4)

        self.Wq  = nn.Conv2d(in_ch, sub,    1, bias=True)
        self.Wk  = nn.Conv2d(in_ch, sub,    1, bias=True)
        self.psi = nn.Conv2d(sub,   1,       1, bias=True)
        # stride=2 → downsampling happens here
        self.Wv  = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Gate phi at full resolution
        q   = self.Wq(x)
        k   = self.Wk(x)
        phi = torch.sigmoid(self.psi(F.relu(q * k)))  # [B,1,H,W]

        # Value with stride=2 downsampling
        v   = self.Wv(x)                               # [B,out_ch,H/2,W/2]

        # Downsample phi to match v
        phi = F.interpolate(phi, v.shape[2:], mode='bilinear', align_corners=False)

        return F.relu(self.bn(phi * v))                # [B,out_ch,H/2,W/2]
