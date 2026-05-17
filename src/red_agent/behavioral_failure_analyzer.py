"""Analyze why mutated samples evade detection."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def analyze_failures(
    original_sequences: np.ndarray,
    mutated_sequences: np.ndarray,
    metadata_rows: List[dict],
    feature_names: List[str],
) -> Dict[str, object]:
    """Produce interpretable failure diagnostics for robustness research."""

    if original_sequences.shape != mutated_sequences.shape:
        raise ValueError("Original and mutated sequence shapes must match")

    failures_idx = [
        i
        for i, row in enumerate(metadata_rows)
        if not bool(row.get("evaluation_outcome", {}).get("is_detected", False))
    ]
    if not failures_idx:
        return {
            "n_failures": 0,
            "top_shifted_features": [],
            "fragile_mutation_types": {},
            "notes": "No detection failures under current mutation set",
        }

    orig_fail = original_sequences[failures_idx]
    mut_fail = mutated_sequences[failures_idx]
    abs_shift = np.abs(mut_fail - orig_fail)
    per_feature_shift = abs_shift.mean(axis=(0, 1))

    top_features = np.argsort(-per_feature_shift)[: min(5, len(feature_names))]
    top_shifted = [
        {
            "feature": feature_names[idx],
            "mean_absolute_shift": float(per_feature_shift[idx]),
        }
        for idx in top_features
    ]

    fragile_mutations: Dict[str, int] = {}
    for i in failures_idx:
        mutation_type = str(metadata_rows[i].get("mutation_type", "unknown"))
        fragile_mutations[mutation_type] = fragile_mutations.get(mutation_type, 0) + 1

    return {
        "n_failures": len(failures_idx),
        "top_shifted_features": top_shifted,
        "fragile_mutation_types": dict(sorted(fragile_mutations.items(), key=lambda kv: kv[1], reverse=True)),
        "notes": "Higher feature shift concentration indicates potential over-reliance.",
    }
