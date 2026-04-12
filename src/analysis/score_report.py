"""Score-distribution diagnostics for a trained checkpoint on a dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.normalization import FeatureNormalizer, fit_feature_normalizer
from src.data_loader.split_utils import create_session_splits, load_filtered_sessions
from src.features.feature_config import FEATURE_DIM, SESSION_LEN
from src.models.transformer import C2Transformer


@dataclass(frozen=True)
class ScoreSummary:
    split_name: str
    threshold: float
    n_total: int
    n_positive: int
    n_negative: int
    score_mean: float
    score_std: float
    positive_mean: float
    negative_mean: float
    positive_p50: float
    positive_p95: float
    negative_p50: float
    negative_p95: float
    predicted_positive_rate: float
    recall_at_threshold: float
    fpr_at_threshold: float


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


def _summarize(split_name: str, probs: np.ndarray, targets: np.ndarray, threshold: float) -> ScoreSummary:
    preds = (probs >= threshold).astype(int)
    positives = probs[targets == 1]
    negatives = probs[targets == 0]
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())

    return ScoreSummary(
        split_name=split_name,
        threshold=float(threshold),
        n_total=len(targets),
        n_positive=int((targets == 1).sum()),
        n_negative=int((targets == 0).sum()),
        score_mean=float(np.mean(probs)),
        score_std=float(np.std(probs)),
        positive_mean=float(np.mean(positives)) if len(positives) else 0.0,
        negative_mean=float(np.mean(negatives)) if len(negatives) else 0.0,
        positive_p50=float(np.quantile(positives, 0.50)) if len(positives) else 0.0,
        positive_p95=float(np.quantile(positives, 0.95)) if len(positives) else 0.0,
        negative_p50=float(np.quantile(negatives, 0.50)) if len(negatives) else 0.0,
        negative_p95=float(np.quantile(negatives, 0.95)) if len(negatives) else 0.0,
        predicted_positive_rate=float(np.mean(preds)),
        recall_at_threshold=float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        fpr_at_threshold=float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    )


def score_dataset(
    checkpoint_path: str,
    npz_path: str,
    batch_size: int = 256,
    min_flows: int = 5,
    split: str = "full",
    device: str | None = None,
) -> ScoreSummary:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    threshold = float(checkpoint["optimal_threshold"])
    transform_config = FeatureTransformConfig.from_checkpoint_dict(checkpoint.get("feature_transform_config"))

    if split == "holdout":
        splits = create_session_splits(npz_path, min_flows=min_flows)
        x_train = apply_feature_transforms(splits.x_train, splits.m_train, transform_config)
        sequences = apply_feature_transforms(splits.x_test, splits.m_test, transform_config)
        labels, masks = splits.y_test, splits.m_test
        if checkpoint.get("normalize_features", False):
            if checkpoint.get("feature_normalizer") is not None:
                normalizer = FeatureNormalizer.from_checkpoint_dict(checkpoint["feature_normalizer"])
            else:
                normalizer = fit_feature_normalizer(x_train, splits.m_train)
            sequences = normalizer.transform(sequences, masks)
    else:
        sequences, labels, masks = load_filtered_sessions(npz_path, min_flows=min_flows)
        sequences = apply_feature_transforms(sequences, masks, transform_config)
        if checkpoint.get("normalize_features", False):
            feature_normalizer = checkpoint.get("feature_normalizer")
            if feature_normalizer is None:
                raise ValueError("Checkpoint expects normalized features but has no saved normalizer.")
            normalizer = FeatureNormalizer.from_checkpoint_dict(feature_normalizer)
            sequences = normalizer.transform(sequences, masks)

    dataset = SessionDataset(sequences, masks, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = C2Transformer(
        feature_dim=FEATURE_DIM,
        seq_len=SESSION_LEN,
        use_derivative_features=bool(checkpoint.get("use_derivative_features", False)),
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])

    probs, targets = predict_probabilities(model, loader, device_obj)
    return _summarize(split_name=split, probs=probs, targets=targets, threshold=threshold)


def format_summary(summary: ScoreSummary) -> str:
    return (
        f"Split: {summary.split_name}\n"
        f"Threshold: {summary.threshold:.4f}\n"
        f"N={summary.n_total} | Pos={summary.n_positive} | Neg={summary.n_negative}\n"
        f"Score mean/std: {summary.score_mean:.4f} / {summary.score_std:.4f}\n"
        f"Positive mean/p50/p95: {summary.positive_mean:.4f} / {summary.positive_p50:.4f} / {summary.positive_p95:.4f}\n"
        f"Negative mean/p50/p95: {summary.negative_mean:.4f} / {summary.negative_p50:.4f} / {summary.negative_p95:.4f}\n"
        f"Predicted positive rate: {summary.predicted_positive_rate:.4f}\n"
        f"Recall@threshold: {summary.recall_at_threshold:.4f}\n"
        f"FPR@threshold: {summary.fpr_at_threshold:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score-distribution diagnostics for a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_flows", type=int, default=5)
    parser.add_argument("--split", choices=["full", "holdout"], default="full")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    summary = score_dataset(
        checkpoint_path=args.checkpoint,
        npz_path=args.npz_path,
        batch_size=args.batch_size,
        min_flows=args.min_flows,
        split=args.split,
    )
    print(format_summary(summary))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(asdict(summary), handle, indent=2)


if __name__ == "__main__":
    main()
