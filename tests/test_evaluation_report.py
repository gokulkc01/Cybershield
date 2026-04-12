import numpy as np
import torch

from src.evaluation.evaluate_transformer import find_threshold_under_fpr_budget


def test_find_threshold_under_fpr_budget_prefers_highest_recall_within_budget():
    y_true = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.05, 0.10, 0.20, 0.80, 0.90], dtype=np.float32)

    threshold, recall, fpr, feasible = find_threshold_under_fpr_budget(y_true, y_prob, max_fpr=0.01)

    assert feasible is True
    assert 0.0 <= fpr <= 0.01 + 1e-12
    assert recall == 1.0
    assert threshold <= 0.9


def test_find_threshold_under_fpr_budget_returns_infeasible_when_no_positives():
    y_true = np.array([0, 0, 0], dtype=np.int64)
    y_prob = np.array([0.05, 0.10, 0.20], dtype=np.float32)

    threshold, recall, fpr, feasible = find_threshold_under_fpr_budget(y_true, y_prob, max_fpr=0.01)

    assert feasible is False
    assert threshold == 1.0
    assert recall == 0.0
    assert fpr == 0.0
