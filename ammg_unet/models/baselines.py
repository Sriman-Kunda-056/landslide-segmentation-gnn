"""
models/baselines.py  —  Vanilla UNet baseline for comparison.
"""
import torch, torch.nn as nn, torch.nn.functional as F

class _DC(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
    def forward(self, x): return self.b(x)

class VanillaUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, features=None):
        super().__init__()
        f = features or [64,128,256,512]
        self.e1=_DC(in_ch,f[0]); self.e2=_DC(f[0],f[1])
        self.e3=_DC(f[1],f[2]); self.e4=_DC(f[2],f[3])
        self.pool=nn.MaxPool2d(2)
        self.bot=_DC(f[3],f[3]*2)
        self.d4=_DC(f[3]*2+f[3],f[3]); self.d3=_DC(f[3]+f[2],f[2])
        self.d2=_DC(f[2]+f[1],f[1]); self.d1=_DC(f[1]+f[0],f[0])
        self.head=nn.Conv2d(f[0],num_classes,1); self.drop=nn.Dropout2d(0.1)
    def _up(self,x,r): return F.interpolate(x,r.shape[2:],mode='bilinear',align_corners=False)
    def forward(self,x):
        s1=self.e1(x); s2=self.e2(self.pool(s1)); s3=self.e3(self.pool(s2)); s4=self.e4(self.pool(s3))
        b=self.bot(self.pool(s4))
        x4=self.d4(torch.cat([self._up(b,s4),s4],1))
        x3=self.d3(torch.cat([self._up(x4,s3),s3],1))
        x2=self.d2(torch.cat([self._up(x3,s2),s2],1))
        x1=self.d1(torch.cat([self._up(x2,s1),s1],1))
        return self.head(self.drop(x1))
    def count_parameters(self): return sum(p.numel() for p in self.parameters())
