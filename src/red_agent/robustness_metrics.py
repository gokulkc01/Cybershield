"""Robustness metrics for mutation-based adversarial evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_detection_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    recall = _safe_div(tp, tp + fn)
    precision = _safe_div(tp, tp + fp)
    fpr = _safe_div(fp, fp + tn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "f1": f1,
    }


def compute_robustness_summary(
    baseline: Dict[str, float],
    mutated: Dict[str, float],
    baseline_latency_ms: float,
    mutated_latency_ms: float,
) -> Dict[str, float]:
    """Primary research signals: degradation, FP shift, and latency shift."""

    return {
        "recall_degradation": baseline["recall"] - mutated["recall"],
        "fpr_shift": mutated["fpr"] - baseline["fpr"],
        "precision_shift": mutated["precision"] - baseline["precision"],
        "f1_shift": mutated["f1"] - baseline["f1"],
        "detection_latency_shift_ms": mutated_latency_ms - baseline_latency_ms,
        "behavioral_invariance_stability": _safe_div(mutated["recall"], max(1e-9, baseline["recall"])),
    }


def aggregate_by_mutation(records: List[dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate outcome stats grouped by mutation_type."""

    grouped: Dict[str, List[dict]] = {}
    for row in records:
        grouped.setdefault(row["mutation_type"], []).append(row)

    out: Dict[str, Dict[str, float]] = {}
    for mutation_type, rows in grouped.items():
        detected = sum(int(r.get("evaluation_outcome", {}).get("is_detected", False)) for r in rows)
        total = len(rows)
        severities = [float(r.get("mutation_severity", 0.0)) for r in rows]
        out[mutation_type] = {
            "n_samples": float(total),
            "detection_rate": _safe_div(detected, total),
            "failure_rate": 1.0 - _safe_div(detected, total),
            "avg_severity": float(np.mean(severities)) if severities else 0.0,
        }
    return out
