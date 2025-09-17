import torch


# Dice (binary), logits: (B,1,H,W), target in {0,1}
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    """
    Works for binary or multi-label segmentation.
    - logits: (B, C, H, W) raw outputs (no sigmoid in model)
    - target: (B, C, H, W) float in {0,1}
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, smooth=1e-6, reduction="mean"):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.reduction = reduction

        # pos_weight can be None, a scalar, or a (C,) tensor for per-class imbalance
        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            # register as buffer so it moves with .to(device), .cuda()
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE w/ logits (per-channel for multi-label)
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction=self.reduction
        )

        # Soft Dice (per channel, then mean)
        probs = torch.sigmoid(logits)
        # sum over batch+spatial, keep channel dim
        inter = (probs * target).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss
