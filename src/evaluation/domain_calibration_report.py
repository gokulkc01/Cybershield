"""Domain-calibrated evaluation for continual-learning checkpoints.

This script calibrates a separate operating threshold per domain using the
corresponding validation split, then reports test-set metrics for each domain.
It is intended to assess whether a continual-learning checkpoint preserves both
mixed-split and UWF behavior when each domain gets its own threshold.

Usage:
    python -m src.evaluation.domain_calibration_report \
        --model_path experiments/continual_mixed_replay_uwf/best_transformer.pth \
        --mixed_val_npz data/processed/extended_mixed_real_corrected/val_sessions.npz \
        --mixed_test_npz data/processed/extended_mixed_real_corrected/test_sessions.npz \
        --uwf_val_npz data/processed/uwf_adaptation_split/val_sessions.npz \
        --uwf_test_npz data/processed/uwf_adaptation_split/test_sessions.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data_loader.npz_utils import detect_npz_schema, load_session_npz
from src.evaluation.operating_point import find_threshold_under_fpr_budget
from src.evaluation.evaluate_zero_shot import (
    _apply_transforms,
    _load_transform_config,
    SessionNPZDataset,
    compute_metrics,
    predict_probabilities,
)
from src.features.feature_config import FEATURE_DIM, MAX_FPR_BUDGET, SESSION_LEN
from src.features.feature_config_extended import FEATURE_SCHEMA_VERSION as EXTENDED_SCHEMA_VERSION
from src.models.transformer import C2Transformer
from src.data_loader.normalization import FeatureNormalizer


def _load_domain_dataset(
    npz_path: str,
    checkpoint: dict,
    expected_feature_names: tuple[str, ...] | None,
    expected_feature_dim: int,
    expected_session_len: int,
):
    schema = detect_npz_schema(npz_path)
    feature_names = tuple(schema["feature_names"]) if schema["feature_names"] else expected_feature_names
    sequences, labels, masks = load_session_npz(
        npz_path,
        expected_feature_names=feature_names,
        expected_session_len=int(schema["session_len"]) if schema["session_len"] else expected_session_len,
    )

    transform_config = _load_transform_config(checkpoint)
    checkpoint_schema = str(checkpoint.get("schema_version", schema["schema_version"]))
    sequences = _apply_transforms(sequences, masks, transform_config, checkpoint_schema, feature_names)

    if checkpoint.get("normalize_features", False):
        normalizer = FeatureNormalizer.from_checkpoint_dict(checkpoint.get("feature_normalizer"))
        sequences = normalizer.transform(sequences, masks)

    dataset = SessionNPZDataset(sequences, masks, labels)
    model_feature_dim = int(checkpoint.get("feature_dim", schema["feature_dim"] or expected_feature_dim))
    model_seq_len = int(checkpoint.get("session_len", schema["session_len"] or expected_session_len))
    return dataset, labels, model_feature_dim, model_seq_len


def _score_domain(
    model: torch.nn.Module,
    device: torch.device,
    val_dataset: SessionNPZDataset,
    test_dataset: SessionNPZDataset,
    max_fpr_budget: float,
    batch_size: int,
):
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    val_probs, val_targets = predict_probabilities(model, val_loader, device)
    threshold, val_recall, val_fpr, _ = find_threshold_under_fpr_budget(val_targets, val_probs, max_fpr=max_fpr_budget)

    test_probs, test_targets = predict_probabilities(model, test_loader, device)
    recall, fpr, auc, f1, (tp, tn, fp, fn) = compute_metrics(test_targets, test_probs, threshold)
    return {
        "threshold": float(threshold),
        "val_recall": float(val_recall),
        "val_fpr": float(val_fpr),
        "test_recall": float(recall),
        "test_fpr": float(fpr),
        "test_auc": float(auc),
        "test_f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_test": int(len(test_targets)),
        "n_c2": int((test_targets == 1).sum()),
        "n_benign": int((test_targets == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-calibrated evaluation for continual-learning checkpoints")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mixed_val_npz", required=True)
    parser.add_argument("--mixed_test_npz", required=True)
    parser.add_argument("--uwf_val_npz", required=True)
    parser.add_argument("--uwf_test_npz", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_fpr_budget", type=float, default=MAX_FPR_BUDGET)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing model_state_dict")

    expected_feature_names = tuple(checkpoint.get("feature_names") or []) or None
    expected_feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
    expected_session_len = int(checkpoint.get("session_len", SESSION_LEN))

    mixed_val_ds, _, feature_dim, seq_len = _load_domain_dataset(
        args.mixed_val_npz,
        checkpoint,
        expected_feature_names,
        expected_feature_dim,
        expected_session_len,
    )
    mixed_test_ds, _, _, _ = _load_domain_dataset(
        args.mixed_test_npz,
        checkpoint,
        expected_feature_names,
        expected_feature_dim,
        expected_session_len,
    )
    uwf_val_ds, _, _, _ = _load_domain_dataset(
        args.uwf_val_npz,
        checkpoint,
        expected_feature_names,
        expected_feature_dim,
        expected_session_len,
    )
    uwf_test_ds, _, _, _ = _load_domain_dataset(
        args.uwf_test_npz,
        checkpoint,
        expected_feature_names,
        expected_feature_dim,
        expected_session_len,
    )

    model = C2Transformer(
        feature_dim=feature_dim,
        seq_len=seq_len,
        use_derivative_features=bool(checkpoint.get("use_derivative_features", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    mixed_scores = _score_domain(model, device, mixed_val_ds, mixed_test_ds, args.max_fpr_budget, args.batch_size)
    uwf_scores = _score_domain(model, device, uwf_val_ds, uwf_test_ds, args.max_fpr_budget, args.batch_size)

    print("\nDOMAIN-CALIBRATED CONTINUAL-LEARNING EVALUATION")
    print(f"Checkpoint: {args.model_path}")
    print(f"Budget: {args.max_fpr_budget:.4f}")
    print("\nMixed domain:")
    print(f"  Threshold: {mixed_scores['threshold']:.4f}")
    print(f"  Test AUC: {mixed_scores['test_auc']:.4f}")
    print(f"  Test F1: {mixed_scores['test_f1']:.4f}")
    print(f"  Test FPR: {mixed_scores['test_fpr']:.4f}")
    print(f"  Test Recall: {mixed_scores['test_recall']:.4f}")
    print(f"  Confusion: TP={mixed_scores['tp']:,} TN={mixed_scores['tn']:,} FP={mixed_scores['fp']:,} FN={mixed_scores['fn']:,}")

    print("\nUWF domain:")
    print(f"  Threshold: {uwf_scores['threshold']:.4f}")
    print(f"  Test AUC: {uwf_scores['test_auc']:.4f}")
    print(f"  Test F1: {uwf_scores['test_f1']:.4f}")
    print(f"  Test FPR: {uwf_scores['test_fpr']:.4f}")
    print(f"  Test Recall: {uwf_scores['test_recall']:.4f}")
    print(f"  Confusion: TP={uwf_scores['tp']:,} TN={uwf_scores['tn']:,} FP={uwf_scores['fp']:,} FN={uwf_scores['fn']:,}")

    report = {
        "checkpoint": args.model_path,
        "budget": args.max_fpr_budget,
        "mixed": mixed_scores,
        "uwf": uwf_scores,
    }
    out_path = Path(args.model_path).parent / "domain_calibration_report.json"
    out_path.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    print(f"\n[INFO] Saved report to {out_path}")


if __name__ == "__main__":
    main()
