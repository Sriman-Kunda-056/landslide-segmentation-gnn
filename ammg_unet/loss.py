# =============================================================================
# loss.py  —  Combined segmentation loss  (Paper Equations 6-8)
#
# Eq 6:  L_seg = L_dice + 0.5 * L_ce
# Eq 7:  L_dice = 1 - (1/N) * Σ  2*Σp(x)*g(x) / (Σp(x) + Σg(x))
# Eq 8:  L_ce   = -(1/N) * Σ g * log(p)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    Handles class imbalance: normalises by total positives not total pixels.
    A model predicting ALL background gets Dice=1.0 (maximum loss).
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # logits : [B, 2, H, W]  raw scores
        # targets: [B, H, W]     int64  {0, 1}

        probs = torch.softmax(logits, dim=1)[:, 1]   # P(landslide) [B,H,W]
        g     = (targets == 1).float()               # ground truth  [B,H,W]

        pf = probs.contiguous().view(probs.shape[0], -1)  # [B, H*W]
        gf = g.contiguous().view(g.shape[0], -1)          # [B, H*W]

        num  = 2.0 * (pf * gf).sum(1) + self.smooth       # [B]
        den  = pf.sum(1) + gf.sum(1)  + self.smooth       # [B]
        return (1.0 - (num / den)).mean()


class CombinedLoss(nn.Module):
    """
    L_seg = L_dice + 0.5 * L_ce   (Equation 6)
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss()   # takes raw logits + long targets

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, targets) + 0.5 * self.ce(logits, targets)


# ── self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    crit = CombinedLoss()

    # Random → loss ~ 1.0-1.5
    lo = torch.randn(2, 2, 256, 256)
    ta = torch.randint(0, 2, (2, 256, 256))
    l  = crit(lo, ta)
    print(f"Random loss    : {l.item():.4f}  (expect ~1.0-1.5)")

    # Perfect → loss near 0
    lo2 = torch.zeros(2, 2, 256, 256)
    lo2[:, 1][ta == 1] = 10.0
    lo2[:, 0][ta == 0] = 10.0
    l2  = crit(lo2, ta)
    print(f"Perfect loss   : {l2.item():.6f}  (expect ~0.0)")
    assert l.item() > l2.item(), "Loss ordering wrong!"
    print("loss.py PASSED ✓")
