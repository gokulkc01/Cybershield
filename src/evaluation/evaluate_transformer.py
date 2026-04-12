"""Evaluate a trained CyberShield Transformer checkpoint on any session NPZ.

This module is designed for two use cases:
1. Production-style reporting on a held-out test split.
2. Zero-shot evaluation on a frozen external dataset such as Stratosphere MCFP.

It reports Recall@FPR budget, threshold, AUC, F1, confusion matrix, and an
estimated false-alert volume for a specified daily flow count.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from src.features.feature_config import MAX_FPR_BUDGET, SESSION_LEN, FEATURE_DIM
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


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")

    data = np.load(path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise KeyError(f"NPZ must contain X and y arrays. Found keys: {list(data.keys())}")

    sequences = data["X"].astype(np.float32)
    labels = data["y"].astype(np.int64)

    if sequences.ndim != 3:
        raise ValueError(f"Expected X to be 3D (N, {SESSION_LEN}, {FEATURE_DIM}), got {sequences.shape}")
    if sequences.shape[1] != SESSION_LEN:
        raise ValueError(f"Expected SESSION_LEN={SESSION_LEN}, got {sequences.shape[1]}")
    if sequences.shape[2] != FEATURE_DIM:
        raise ValueError(f"Expected FEATURE_DIM={FEATURE_DIM}, got {sequences.shape[2]}")

    if "masks" in data:
        masks = data["masks"].astype(bool)
    else:
        masks = (sequences.sum(axis=2) != 0)

    if masks.shape != (sequences.shape[0], sequences.shape[1]):
        raise ValueError(f"Mask shape mismatch: expected {(sequences.shape[0], sequences.shape[1])}, got {masks.shape}")

    return sequences, labels, masks


def find_threshold_under_fpr_budget(y_true: np.ndarray, y_prob: np.ndarray, max_fpr: float = MAX_FPR_BUDGET):
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
    sequences, labels, masks = load_npz(npz_path)
    dataset = SessionNPZDataset(sequences, masks, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = C2Transformer().to(device_obj)

    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    probs, targets = predict_probabilities(model, loader, device_obj)
    threshold = float(checkpoint.get("optimal_threshold", 0.5))
    saved_budget = float(checkpoint.get("max_fpr_budget", max_fpr_budget))

    if "val_recall_budget" in checkpoint and "val_fpr_budget" in checkpoint:
        threshold, val_recall, val_fpr, feasible = (
            threshold,
            float(checkpoint["val_recall_budget"]),
            float(checkpoint["val_fpr_budget"]),
            True,
        )
    else:
        threshold, val_recall, val_fpr, feasible = find_threshold_under_fpr_budget(targets, probs, saved_budget)

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
