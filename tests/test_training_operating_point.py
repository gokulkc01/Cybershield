import numpy as np
import torch
import torch.nn.functional as F

from src.losses.focal_loss import FocalLoss
from src.training.train_transformer import (
    find_threshold_under_fpr_budget,
    is_better_operating_point,
)


def test_is_better_operating_point_prefers_higher_recall():
    assert is_better_operating_point(
        candidate_recall=0.20,
        candidate_f1=0.10,
        candidate_fpr=0.004,
        best_recall=0.10,
        best_f1=0.20,
        best_fpr=0.003,
    )


def test_is_better_operating_point_tiebreaks_on_f1_then_fpr():
    assert is_better_operating_point(
        candidate_recall=0.10,
        candidate_f1=0.30,
        candidate_fpr=0.004,
        best_recall=0.10,
        best_f1=0.20,
        best_fpr=0.003,
    )
    assert is_better_operating_point(
        candidate_recall=0.10,
        candidate_f1=0.20,
        candidate_fpr=0.002,
        best_recall=0.10,
        best_f1=0.20,
        best_fpr=0.003,
    )


def test_find_threshold_under_fpr_budget_returns_feasible_point():
    y_true = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.01, 0.02, 0.30, 0.80, 0.90], dtype=np.float32)

    threshold, recall, fpr, feasible = find_threshold_under_fpr_budget(
        y_true, y_prob, max_fpr=0.005
    )

    assert feasible is True
    assert 0.0 <= fpr <= 0.005 + 1e-12
    assert recall == 1.0
    assert threshold <= 0.8000001


def test_find_threshold_under_fpr_budget_handles_no_positive_samples():
    y_true = np.array([0, 0, 0], dtype=np.int64)
    y_prob = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    threshold, recall, fpr, feasible = find_threshold_under_fpr_budget(
        y_true, y_prob, max_fpr=0.0
    )

    assert feasible is False
    assert threshold == 1.0
    assert recall == 0.0
    assert fpr == 0.0


def test_find_threshold_under_fpr_budget_handles_no_negative_samples():
    y_true = np.array([1, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    threshold, recall, fpr, feasible = find_threshold_under_fpr_budget(
        y_true, y_prob, max_fpr=0.0
    )

    assert feasible is True
    assert threshold == 0.0
    assert recall == 1.0
    assert fpr == 0.0


def test_focal_loss_produces_finite_gradients():
    logits = torch.tensor([0.3, -0.7, 1.1, -2.2], requires_grad=True)
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32)

    criterion = FocalLoss(alpha=0.5, gamma=1.0)
    loss = criterion(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_focal_loss_gamma_zero_matches_alpha_weighted_bce():
    logits = torch.tensor([0.2, -1.4, 0.7, -0.1])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    criterion = FocalLoss(alpha=0.5, gamma=0.0, reduction="none")
    focal_loss = criterion(logits, targets)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected = 0.5 * bce

    assert torch.allclose(focal_loss, expected, atol=1e-6, rtol=1e-6)
