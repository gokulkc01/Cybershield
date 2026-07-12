"""Metrics and reporting helpers for the multifamily generalization experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import stats
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


def bootstrap_metric_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
    stratified: bool = True,
) -> dict:
    """Percentile bootstrap CI for any (y_true, y_score) -> float metric.

    Stratified resampling preserves class counts so metrics that need both
    classes (ROC-AUC) stay defined in every resample.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if len(y_true_arr) != len(y_score_arr):
        raise ValueError("y_true and y_score must have equal length")
    rng = np.random.default_rng(random_seed)
    point = float(metric_fn(y_true_arr, y_score_arr))

    if stratified:
        class_indices = [np.where(y_true_arr == cls)[0] for cls in np.unique(y_true_arr)]
    samples: list[float] = []
    for _ in range(n_bootstrap):
        if stratified:
            idx = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in class_indices])
        else:
            idx = rng.integers(0, len(y_true_arr), size=len(y_true_arr))
        try:
            samples.append(float(metric_fn(y_true_arr[idx], y_score_arr[idx])))
        except ValueError:
            continue
    if not samples:
        return {"point": point, "ci_lower": None, "ci_upper": None, "n_bootstrap": 0, "confidence": confidence}
    lower_q = (1.0 - confidence) / 2.0
    return {
        "point": point,
        "ci_lower": float(np.quantile(samples, lower_q)),
        "ci_upper": float(np.quantile(samples, 1.0 - lower_q)),
        "n_bootstrap": len(samples),
        "confidence": float(confidence),
    }


def bootstrap_score_cis(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    fpr_budgets: Sequence[float] = (0.005, 0.015, 0.03),
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> dict[str, dict]:
    """Bootstrap CIs for the headline ranking metrics."""
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "roc_auc": lambda t, s: float(roc_auc_score(t, s)),
        "pr_auc": lambda t, s: float(average_precision_score(t, s)),
    }
    for budget in fpr_budgets:
        metric_fns[f"recall_at_fpr_{budget:.4f}"] = (
            lambda t, s, b=float(budget): recall_at_fpr(t, s, b)
        )
    return {
        name: bootstrap_metric_ci(
            y_true,
            y_score,
            fn,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            random_seed=random_seed,
        )
        for name, fn in metric_fns.items()
    }


def prevalence_adjusted_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    base_rates: Sequence[float] = (0.001, 0.01),
    fpr_budgets: Sequence[float] = (0.005, 0.015, 0.03),
) -> dict[str, dict]:
    """Re-weight test metrics to a deployment base rate of positives.

    Balanced evaluation sets overstate precision because C2 prevalence in
    production is ~0.1-1%, not ~50%. Bayes-adjusted precision at prevalence
    ``pi`` from an ROC point (fpr, tpr) is ``pi*tpr / (pi*tpr + (1-pi)*fpr)``.
    ``pr_auc`` integrates that adjusted precision over the ROC sweep, and
    ``alerts_per_1k_sessions`` is the expected alert volume at the operating
    point (both true and false alerts).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)

    report: dict[str, dict] = {}
    for base_rate in base_rates:
        pi = float(base_rate)
        denominator = pi * tpr + (1.0 - pi) * fpr
        adjusted_precision = np.divide(
            pi * tpr,
            denominator,
            out=np.ones_like(tpr),
            where=denominator > 0,
        )
        # roc_curve returns fpr/tpr sorted by increasing fpr (recall is
        # non-decreasing), so trapezoid-free AP-style integration applies.
        recall_deltas = np.diff(np.concatenate([[0.0], tpr]))
        adjusted_pr_auc = float(np.sum(recall_deltas * adjusted_precision))

        operating_points: dict[str, dict] = {}
        for budget in fpr_budgets:
            eligible = np.where(fpr <= float(budget))[0]
            if len(eligible):
                best = eligible[np.argmax(tpr[eligible])]
                point_tpr = float(tpr[best])
                point_fpr = float(fpr[best])
            else:
                point_tpr = 0.0
                point_fpr = 0.0
            alert_rate = pi * point_tpr + (1.0 - pi) * point_fpr
            point_denominator = alert_rate
            operating_points[f"{float(budget):.4f}"] = {
                "recall": point_tpr,
                "fpr": point_fpr,
                "adjusted_precision": float(pi * point_tpr / point_denominator) if point_denominator > 0 else None,
                "alerts_per_1k_sessions": float(1000.0 * alert_rate),
                "false_alerts_per_1k_sessions": float(1000.0 * (1.0 - pi) * point_fpr),
            }

        report[f"{pi:g}"] = {
            "base_rate": pi,
            "adjusted_pr_auc": adjusted_pr_auc,
            "operating_points": operating_points,
        }
    return report


def seed_mean_ci(values: Sequence[float], *, confidence: float = 0.95) -> dict:
    """Mean with a t-distribution CI across independent runs (e.g. seeds)."""
    arr = np.asarray([float(value) for value in values], dtype=float)
    n = int(len(arr))
    if n == 0:
        return {"mean": None, "std": None, "ci_lower": None, "ci_upper": None, "n": 0, "confidence": confidence}
    mean = float(np.mean(arr))
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci_lower": mean, "ci_upper": mean, "n": 1, "confidence": confidence}
    std = float(np.std(arr, ddof=1))
    half_width = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1) * std / math.sqrt(n))
    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "n": n,
        "confidence": float(confidence),
    }


def honest_test_report(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
    family_labels: Iterable[str] | None = None,
    fpr_budgets: Sequence[float] = (0.005, 0.015, 0.03),
    base_rates: Sequence[float] = (0.001, 0.01),
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> dict:
    """Full honest evaluation: threshold-free metrics, bootstrap CIs, base-rate view."""
    report = compute_multifamily_metrics(
        y_true,
        y_score,
        threshold=threshold,
        family_labels=family_labels,
    )
    report["bootstrap_cis"] = bootstrap_score_cis(
        y_true,
        y_score,
        fpr_budgets=fpr_budgets,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    report["prevalence_adjusted"] = prevalence_adjusted_metrics(
        y_true,
        y_score,
        base_rates=base_rates,
        fpr_budgets=fpr_budgets,
    )
    return report


def save_multifamily_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
