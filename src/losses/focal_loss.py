"""Binary focal loss for extreme class-imbalanced C2 detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Numerically stable binary focal loss on logits.

    Parameters
    ----------
    alpha:
        Positive-class weighting factor in [0, 1].
    gamma:
        Focusing exponent. Higher values down-weight easy examples more.
    reduction:
        One of {"mean", "sum", "none"}.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 3.0, reduction: str = "mean") -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"invalid reduction: {reduction}")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for binary labels.

        logits: shape (B,) or (B, 1)
        targets: shape-compatible tensor of {0,1}
        """
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)

        # p_t is the probability assigned to the ground-truth class.
        p_t = torch.where(targets == 1.0, probs, 1.0 - probs)

        alpha_t = torch.where(
            targets == 1.0,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
