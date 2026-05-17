"""Evaluate a trained CyberShield Transformer checkpoint on any session NPZ.

This module is designed for two use cases:
1. Production-style reporting on a held-out test split.
2. Zero-shot evaluation on a frozen external dataset such as Stratosphere MCFP.

It reports Recall@FPR budget, threshold, AUC, F1, confusion matrix, and an
estimated false-alert volume for a specified daily flow count.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.data_loader.extended_feature_transforms import (
    ExtendedFeatureTransformConfig,
    apply_extended_feature_transforms,
)
from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.normalization import FeatureNormalizer
from src.data_loader.npz_utils import detect_npz_schema, load_session_npz
from src.features.feature_config import MAX_FPR_BUDGET, SESSION_LEN, FEATURE_DIM
from src.features.feature_config_extended import FEATURE_SCHEMA_VERSION as EXTENDED_SCHEMA_VERSION
from src.models.transformer import C2Transformer


@dataclass(frozen=True)
class EvaluationResult:
    threshold: float
    recall_at_budget: float
    fpr_at_budget: float
    f1: float
    auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    budget_met: bool
    expected_false_alerts_per_day: float


class SessionNPZDataset(Dataset):
    def __init__(self, sequences: np.ndarray, masks: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.padding_masks = torch.tensor(~masks, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.padding_masks[idx], self.labels[idx]


def _load_transform_config(checkpoint: dict) -> FeatureTransformConfig | ExtendedFeatureTransformConfig:
    payload = checkpoint.get("feature_transform_config")
    if isinstance(payload, dict) and payload.get("schema") == EXTENDED_SCHEMA_VERSION:
        return ExtendedFeatureTransformConfig.from_checkpoint_dict(payload)
    return FeatureTransformConfig.from_checkpoint_dict(payload)


def _apply_transforms(
    sequences: np.ndarray,
    masks: np.ndarray,
    config: FeatureTransformConfig | ExtendedFeatureTransformConfig,
    schema_version: str,
    feature_names: tuple[str, ...] | None,
) -> np.ndarray:
    if schema_version == EXTENDED_SCHEMA_VERSION:
        return apply_extended_feature_transforms(sequences, masks, config)  # type: ignore[arg-type]
    return apply_feature_transforms(sequences, masks, config, feature_names=feature_names)

@torch.no_grad()
def predict_probabilities(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs, targets = [], []
    for batch_x, batch_mask, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_mask = batch_mask.to(device)
        logits = model(batch_x, batch_mask)
        batch_probs = torch.sigmoid(logits)
        probs.extend(batch_probs.cpu().numpy())
        targets.extend(batch_y.numpy())
    return np.asarray(probs), np.asarray(targets)


def evaluate_checkpoint(
    checkpoint_path: str,
    npz_path: str,
    batch_size: int = 256,
    max_fpr_budget: float = MAX_FPR_BUDGET,
    daily_flows: int = 1_000_000,
    device: str | None = None,
) -> EvaluationResult:
    schema = detect_npz_schema(npz_path)
    feature_names = tuple(schema["feature_names"]) if schema["feature_names"] else None
    sequences, labels, masks = load_session_npz(
        npz_path,
        expected_feature_names=feature_names,
        expected_session_len=int(schema["session_len"]) if schema["session_len"] else None,
    )

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    if "optimal_threshold" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing a validation-derived operating threshold. "
            "Refuse to tune on evaluation data to avoid leakage."
        )
    transform_config = _load_transform_config(checkpoint)
    checkpoint_schema = str(checkpoint.get("schema_version", schema["schema_version"]))
    sequences = _apply_transforms(sequences, masks, transform_config, checkpoint_schema, feature_names)
    if checkpoint.get("normalize_features", False):
        feature_normalizer = checkpoint.get("feature_normalizer")
        if feature_normalizer is None:
            raise ValueError("Checkpoint expects normalized features but has no saved normalizer.")
        normalizer = FeatureNormalizer.from_checkpoint_dict(feature_normalizer)
        sequences = normalizer.transform(sequences, masks)

    model_feature_dim = int(checkpoint.get("feature_dim", schema["feature_dim"] or FEATURE_DIM))
    model_seq_len = int(checkpoint.get("session_len", schema["session_len"] or SESSION_LEN))

    dataset = SessionNPZDataset(sequences, masks, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = C2Transformer(
        feature_dim=model_feature_dim,
        seq_len=model_seq_len,
        use_derivative_features=bool(checkpoint.get("use_derivative_features", False)),
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])

    probs, targets = predict_probabilities(model, loader, device_obj)
    threshold = float(checkpoint.get("optimal_threshold", 0.5))
    saved_budget = float(checkpoint.get("max_fpr_budget", max_fpr_budget))

    preds = (probs >= threshold).astype(int)
    auc = float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.0
    f1 = float(f1_score(targets, preds, zero_division=0))

    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    budget_met = fpr <= (saved_budget + 1e-12)

    benign_rate = float((targets == 0).mean()) if len(targets) > 0 else 0.0
    expected_false_alerts_per_day = daily_flows * benign_rate * fpr

    return EvaluationResult(
        threshold=threshold,
        recall_at_budget=recall,
        fpr_at_budget=fpr,
        f1=f1,
        auc=auc,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        budget_met=budget_met,
        expected_false_alerts_per_day=float(expected_false_alerts_per_day),
    )


def format_result(result: EvaluationResult) -> str:
    return (
        f"Threshold: {result.threshold:.4f}\n"
        f"Recall@budget: {result.recall_at_budget:.4f}\n"
        f"FPR: {result.fpr_at_budget:.4f}\n"
        f"F1: {result.f1:.4f}\n"
        f"AUC: {result.auc:.4f}\n"
        f"Confusion Matrix: TN={result.tn}, FP={result.fp}, FN={result.fn}, TP={result.tp}\n"
        f"Budget met: {result.budget_met}\n"
        f"Expected false alerts/day: {result.expected_false_alerts_per_day:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CyberShield Transformer checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best_transformer.pth")
    parser.add_argument("--npz_path", required=True, help="Path to sessions NPZ")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_fpr_budget", type=float, default=MAX_FPR_BUDGET)
    parser.add_argument("--daily_flows", type=int, default=1_000_000)
    args = parser.parse_args()

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        npz_path=args.npz_path,
        batch_size=args.batch_size,
        max_fpr_budget=args.max_fpr_budget,
        daily_flows=args.daily_flows,
    )
    print(format_result(result))


if __name__ == "__main__":
    main()
