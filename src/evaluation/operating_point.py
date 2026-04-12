"""Selection logic for production operating points under an FPR budget."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def find_threshold_under_fpr_budget(y_true: np.ndarray, y_prob: np.ndarray, max_fpr: float):
    """Select the highest-threshold point with maximal recall under the FPR cap."""
    if not (0.0 <= max_fpr <= 1.0):
        raise ValueError(f"max_fpr must be in [0, 1], got {max_fpr}")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.size == 0:
        return 1.0, 0.0, 0.0, False

    has_positive = np.any(y_true == 1)
    has_negative = np.any(y_true == 0)

    if not has_positive:
        return 1.0, 0.0, 0.0, False
    if not has_negative:
        return 0.0, 1.0, 0.0, True

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_idx = np.where(fpr <= max_fpr)[0]

    if len(valid_idx) == 0:
        return 1.0, 0.0, 0.0, False

    valid_fpr = fpr[valid_idx]
    valid_tpr = tpr[valid_idx]
    valid_thr = thresholds[valid_idx]

    best_tpr = float(valid_tpr.max())
    best_candidates = np.where(valid_tpr == best_tpr)[0]
    chosen = best_candidates[np.argmax(valid_thr[best_candidates])]

    return float(valid_thr[chosen]), float(valid_tpr[chosen]), float(valid_fpr[chosen]), True


def is_better_operating_point(
    candidate_recall: float,
    candidate_f1: float,
    candidate_fpr: float,
    best_recall: float,
    best_f1: float,
    best_fpr: float,
) -> bool:
    """Rank checkpoints by recall first, then F1, then lower FPR."""
    if candidate_recall > best_recall + 1e-12:
        return True
    if abs(candidate_recall - best_recall) <= 1e-12:
        if candidate_f1 > best_f1 + 1e-12:
            return True
        if abs(candidate_f1 - best_f1) <= 1e-12 and candidate_fpr < best_fpr - 1e-12:
            return True
    return False
