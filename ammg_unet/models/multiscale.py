"""
models/multiscale.py  —  Multiscale Connection (Eq 5)

Replaces direct skip connections in UNet.

4 parallel branches:
  b1: Conv 1×1 d=1   RF = 1×1  (point features)
  b2: Conv 3×3 d=1   RF = 3×3  (local)
  b3: Conv 3×3 d=3   RF = 7×7  (medium; 1+(3-1)*3=7)
  b4: MaxPool 3×3 + Conv 1×1   (pooled global context)

Concat → 4*in_ch → 1×1 proj → out_ch
H,W preserved throughout.
"""

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch, out_ch, k=1, p=0, d=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=d, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class MultiscaleConnection(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = in_ch

        self.b1 = _conv_bn_relu(in_ch, mid, k=1, p=0, d=1)
        self.b2 = _conv_bn_relu(in_ch, mid, k=3, p=1, d=1)
        self.b3 = _conv_bn_relu(in_ch, mid, k=3, p=3, d=3)  # p=d for H,W preservation
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            _conv_bn_relu(in_ch, mid, k=1)
        )
        self.proj = _conv_bn_relu(4 * mid, out_ch, k=1)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.proj(torch.cat([
            self.b1(x), self.b2(x), self.b3(x), self.b4(x)
        ], dim=1))
