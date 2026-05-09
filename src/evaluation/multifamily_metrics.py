"""Metrics and reporting helpers for the multifamily generalization experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_float(value: float | np.floating | np.ndarray | None) -> float | None:
    if value is None:
        return None
    return float(value)


def recall_at_fpr(y_true: Sequence[int], y_score: Sequence[float], target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_score))
    eligible = tpr[fpr <= target_fpr]
    return float(np.max(eligible)) if len(eligible) else 0.0


def precision_at_recall(y_true: Sequence[int], y_score: Sequence[float], target_recall: float) -> float:
    precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_score))
    eligible = precision[recall >= target_recall]
    return float(np.max(eligible)) if len(eligible) else 0.0


def compute_threshold_confusion(y_true: Sequence[int], y_score: Sequence[float], threshold: float) -> dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
    }


def compute_family_breakdown(
    y_true: Sequence[int],
    y_score: Sequence[float],
    family_labels: Iterable[str],
    threshold: float,
) -> dict[str, dict]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    family_labels_arr = np.asarray(list(family_labels), dtype=str)

    if len(y_true_arr) != len(family_labels_arr):
        raise ValueError("family_labels must match y_true length")

    breakdown: dict[str, dict] = {}
    for family in sorted(set(family_labels_arr.tolist())):
        mask = family_labels_arr == family
        if not np.any(mask):
            continue
        family_true = y_true_arr[mask]
        family_score = y_score_arr[mask]
        confusion = compute_threshold_confusion(family_true, family_score, threshold)
        confusion["pr_auc"] = _safe_float(average_precision_score(family_true, family_score))
        confusion["roc_auc"] = _safe_float(roc_auc_score(family_true, family_score)) if len(np.unique(family_true)) > 1 else None
        breakdown[family] = confusion
    return breakdown


def compute_multifamily_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
    family_labels: Iterable[str] | None = None,
) -> dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    overall = {
        "pr_auc": _safe_float(average_precision_score(y_true_arr, y_score_arr)),
        "roc_auc": _safe_float(roc_auc_score(y_true_arr, y_score_arr)) if len(np.unique(y_true_arr)) > 1 else None,
        "recall_at_fpr_1pct": recall_at_fpr(y_true_arr, y_score_arr, 0.01),
        "recall_at_fpr_2pct": recall_at_fpr(y_true_arr, y_score_arr, 0.02),
        "recall_at_fpr_5pct": recall_at_fpr(y_true_arr, y_score_arr, 0.05),
        "precision_at_recall_70pct": precision_at_recall(y_true_arr, y_score_arr, 0.70),
        "precision_at_recall_80pct": precision_at_recall(y_true_arr, y_score_arr, 0.80),
        "precision_at_recall_90pct": precision_at_recall(y_true_arr, y_score_arr, 0.90),
        "threshold_confusion": compute_threshold_confusion(y_true_arr, y_score_arr, threshold),
    }

    report = {"overall": overall}
    if family_labels is not None:
        report["by_family"] = compute_family_breakdown(y_true_arr, y_score_arr, family_labels, threshold)
    return report


def save_multifamily_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
