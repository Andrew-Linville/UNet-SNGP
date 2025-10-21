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





class BCEDiceLossSNGP(nn.Module):
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

    def _only_logits(self, out):
        return out[0] if isinstance(out, (tuple, list)) else out
    
    def _to_multilabel_targets(self, y):
        """
        Coerce y to float [B,2,H,W] for multi-label BCE.
        Accepts: [B,2,H,W] (one-hot or multi-hot), [B,1,2,H,W], [B,H,W] (binary -> broadcast),
                [B,1,H,W] (single channel -> broadcast or raise).
        """
        if y.ndim == 5 and y.shape[2] == 2:         # [B,1,2,H,W]
            y = y[:, 0, ...]                        # -> [B,2,H,W]
        elif y.ndim == 4 and y.shape[1] == 2:       # [B,2,H,W]
            pass
        elif y.ndim == 4 and y.shape[1] == 1:       # [B,1,H,W] -> duplicate to 2 chans (if both labels share same mask)
            y = y.repeat(1, 2, 1, 1)
        elif y.ndim == 3:                            # [B,H,W] -> duplicate to 2 chans
            y = y.unsqueeze(1).repeat(1, 2, 1, 1)
        else:
            raise ValueError(f"Unexpected target shape: {tuple(y.shape)}")
        return y.float()
    
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
