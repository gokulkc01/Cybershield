"""Evaluate a checkpoint on the deterministic held-out CTU split used in training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.normalization import FeatureNormalizer, fit_feature_normalizer
from src.data_loader.split_utils import create_session_splits
from src.features.feature_config import FEATURE_DIM, MAX_FPR_BUDGET, SESSION_LEN
from src.models.transformer import C2Transformer


@dataclass(frozen=True)
class HoldoutResult:
    threshold: float
    recall: float
    fpr: float
    f1: float
    auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    budget_met: bool
    expected_false_alerts_per_day: float
    n_test: int
    n_c2: int
    n_benign: int


class SessionDataset(Dataset):
    def __init__(self, sequences: np.ndarray, masks: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.padding_masks = torch.tensor(~masks, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.padding_masks[idx], self.labels[idx]


@torch.no_grad()
def predict_probabilities(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs, targets = [], []
    for batch_x, batch_mask, batch_y in loader:
        logits = model(batch_x.to(device), batch_mask.to(device))
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        targets.extend(batch_y.numpy())
    return np.asarray(probs), np.asarray(targets)


def evaluate_holdout(
    checkpoint_path: str,
    npz_path: str,
    min_flows: int = 5,
    batch_size: int = 256,
    max_fpr_budget: float = MAX_FPR_BUDGET,
    daily_flows: int = 1_000_000,
    device: str | None = None,
) -> HoldoutResult:
    splits = create_session_splits(npz_path, min_flows=min_flows)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    threshold = float(checkpoint["optimal_threshold"])
    transform_config = FeatureTransformConfig.from_checkpoint_dict(checkpoint.get("feature_transform_config"))
    x_train = apply_feature_transforms(splits.x_train, splits.m_train, transform_config)
    x_test = apply_feature_transforms(splits.x_test, splits.m_test, transform_config)
    if checkpoint.get("normalize_features", False):
        feature_normalizer = checkpoint.get("feature_normalizer")
        if feature_normalizer is not None:
            normalizer = FeatureNormalizer.from_checkpoint_dict(feature_normalizer)
        else:
            normalizer = fit_feature_normalizer(x_train, splits.m_train)
        x_test = normalizer.transform(x_test, splits.m_test)

    dataset = SessionDataset(x_test, splits.m_test, splits.y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = C2Transformer(
        feature_dim=FEATURE_DIM,
        seq_len=SESSION_LEN,
        use_derivative_features=bool(checkpoint.get("use_derivative_features", False)),
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])

    probs, targets = predict_probabilities(model, loader, device_obj)
    preds = (probs >= threshold).astype(int)

    auc = float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.0
    f1 = float(f1_score(targets, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    budget_met = fpr <= (max_fpr_budget + 1e-12)
    benign_rate = float((targets == 0).mean()) if len(targets) > 0 else 0.0
    expected_false_alerts_per_day = daily_flows * benign_rate * fpr

    return HoldoutResult(
        threshold=threshold,
        recall=recall,
        fpr=fpr,
        f1=f1,
        auc=auc,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        budget_met=budget_met,
        expected_false_alerts_per_day=float(expected_false_alerts_per_day),
        n_test=len(targets),
        n_c2=int((targets == 1).sum()),
        n_benign=int((targets == 0).sum()),
    )


def format_result(result: HoldoutResult) -> str:
    return (
        f"Threshold: {result.threshold:.4f}\n"
        f"Test size: {result.n_test} (C2={result.n_c2}, Benign={result.n_benign})\n"
        f"Recall@budget-threshold: {result.recall:.4f}\n"
        f"FPR: {result.fpr:.4f}\n"
        f"F1: {result.f1:.4f}\n"
        f"AUC: {result.auc:.4f}\n"
        f"Confusion Matrix: TN={result.tn}, FP={result.fp}, FN={result.fn}, TP={result.tp}\n"
        f"Budget met: {result.budget_met}\n"
        f"Expected false alerts/day: {result.expected_false_alerts_per_day:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the deterministic hold-out split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--min_flows", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_fpr_budget", type=float, default=MAX_FPR_BUDGET)
    parser.add_argument("--daily_flows", type=int, default=1_000_000)
    args = parser.parse_args()

    result = evaluate_holdout(
        checkpoint_path=args.checkpoint,
        npz_path=args.npz_path,
        min_flows=args.min_flows,
        batch_size=args.batch_size,
        max_fpr_budget=args.max_fpr_budget,
        daily_flows=args.daily_flows,
    )
    print(format_result(result))


if __name__ == "__main__":
    main()
