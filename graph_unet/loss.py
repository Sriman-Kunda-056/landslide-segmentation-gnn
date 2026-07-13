"""
loss.py — Combined Dice + CE loss (Paper Eq 6-8)
"""
import torch, torch.nn as nn, torch.nn.functional as F
from config import DICE_SMOOTH, CE_WEIGHT

class DiceLoss(nn.Module):
    def __init__(self, smooth=DICE_SMOOTH):
        super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        p = torch.softmax(logits, dim=1)[:, 1]
        g = (targets == 1).float()
        pf = p.contiguous().view(p.shape[0], -1)
        gf = g.contiguous().view(g.shape[0], -1)
        inter = (pf * gf).sum(1)
        card  = pf.sum(1) + gf.sum(1)
        return (1.0 - ((2*inter + self.smooth)/(card + self.smooth))).mean()

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=CE_WEIGHT):
        super().__init__()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss()
        self.w    = ce_weight
    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.w * self.ce(logits, targets)
