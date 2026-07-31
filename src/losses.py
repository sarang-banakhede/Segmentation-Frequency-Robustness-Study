import torch
import torch.nn as nn
import torch.nn.functional as F


class _SoftDice(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prob, target):
        p = prob.contiguous().view(-1)
        t = target.contiguous().view(-1)
        return 1.0 - (2.0 * (p * t).sum() + self.smooth) / (p.sum() + t.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self._dice = _SoftDice(smooth)

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        p = prob.clamp(1e-7, 1.0 - 1e-7)
        dice_val = self._dice(p, target)
        bce_val = F.binary_cross_entropy_with_logits(logits.float(), target.float())
        return dice_val + bce_val, dice_val, bce_val
