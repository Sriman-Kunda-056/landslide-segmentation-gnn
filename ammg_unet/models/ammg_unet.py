"""
models/ammg_unet.py  —  Full AMMG-UNet

Encoder:  4x AttentionConv (stride=2 each)
Bridge:   MGRM  (4x GRB)
Skips:    MultiscaleConnection
Decoder:  upsample + cat(skip) + DoubleConv  x4
Head:     Conv2d(64, 2, 1)

Tensor shapes for 512×512 input (B=1):
  enc1:  [ 64, 256, 256]   enc2: [128, 128, 128]
  enc3:  [256,  64,  64]   enc4: [512,  32,  32]
  MGRM:  [512,  32,  32]
  dec4:  [256,  32,  32] → up → [256, 64, 64]
  dec3:  [128,  64,  64] → up → [128,128,128]
  dec2:  [ 64, 128, 128] → up → [ 64,256,256]
  dec1:  [ 64, 256, 256] → up → [ 64,512,512]
  head:  [  2, 512, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_conv import AttentionConv
from .multiscale     import MultiscaleConnection
from .mgrm           import MGRM


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)


class AMGUnet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, features=None):
        super().__init__()
        f = features or [64, 128, 256, 512]

        # Encoder
        self.enc1 = AttentionConv(in_ch, f[0])
        self.enc2 = AttentionConv(f[0],  f[1])
        self.enc3 = AttentionConv(f[1],  f[2])
        self.enc4 = AttentionConv(f[2],  f[3])

        # Bridge
        self.mgrm = MGRM(f[3])

        # Skip processors
        self.ms1 = MultiscaleConnection(f[0], f[0])
        self.ms2 = MultiscaleConnection(f[1], f[1])
        self.ms3 = MultiscaleConnection(f[2], f[2])
        self.ms4 = MultiscaleConnection(f[3], f[3])

        # Decoder  (in channels = up_ch + skip_ch)
        self.dec4 = _DoubleConv(f[3]+f[3], f[2])   # 1024 → 256
        self.dec3 = _DoubleConv(f[2]+f[2], f[1])   # 512  → 128
        self.dec2 = _DoubleConv(f[1]+f[1], f[0])   # 256  → 64
        self.dec1 = _DoubleConv(f[0]+f[0], f[0])   # 128  → 64

        self.head    = nn.Conv2d(f[0], num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

        nn.init.kaiming_normal_(self.head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.head.bias, 0)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, ref.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bridge (GCN global reasoning)
        b  = self.mgrm(s4)

        # Decoder with multiscale skips
        d4 = self.dec4(torch.cat([self._up(b,  s4), self.ms4(s4)], 1))
        d3 = self.dec3(torch.cat([self._up(d4, s3), self.ms3(s3)], 1))
        d2 = self.dec2(torch.cat([self._up(d3, s2), self.ms2(s2)], 1))
        d1 = self.dec1(torch.cat([self._up(d2, s1), self.ms1(s1)], 1))

        return self.head(self.dropout(self._up(d1, x)))   # [B, 2, H, W]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
