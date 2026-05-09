"""Zero-shot evaluation: assess model generalization across held-out datasets.

This evaluator reports metrics broken down by dataset source (CTU-13, MCFP, UWF),
enabling rigorous assessment of whether learned behavioral patterns transfer
across C2 families and network environments.

Usage:
    python -m src.evaluation.evaluate_zero_shot \\
        --model_path experiments/transformer_uwf_only/best_transformer.pth \\
        --test_npz data/processed/v2_mixed_session_only/test_sessions.npz \\
        --split_metadata data/processed/v2_mixed_session_only/split_metadata.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.normalization import FeatureNormalizer
from src.data_loader.npz_utils import load_session_npz
from src.features.dataset_builder_v2 import load_multi_source, parse_source_spec
from src.features.feature_config import MAX_FPR_BUDGET, SESSION_LEN, FEATURE_DIM
from src.models.transformer import C2Transformer


@dataclass(frozen=True)
class PerSourceMetrics:
    """Metrics for a single dataset source."""
    source_name: str
    n_samples: int
    n_c2: int
    n_benign: int
    threshold: float
    recall: float
    fpr: float
    auc: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int


class SessionNPZDataset(Dataset):
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
    """Generate predictions for a dataset."""
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


def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> tuple:
    """Compute Recall, FPR, AUC, F1 and confusion matrix."""
    preds = (probs >= threshold).astype(int)
    
    if len(np.unique(targets)) > 1:
        auc = float(roc_auc_score(targets, probs))
    else:
        auc = 0.0
    
    f1 = float(f1_score(targets, preds, zero_division=0))
    cm = confusion_matrix(targets, preds)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: single class in targets
        tn = fp = fn = tp = 0
        if len(np.unique(targets)) == 1:
            if targets[0] == 0:
                tn = (preds == 0).sum()
                fp = (preds == 1).sum()
            else:
                fn = (preds == 0).sum()
                tp = (preds == 1).sum()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return recall, fpr, auc, f1, (tp, tn, fp, fn)


def evaluate_zero_shot(
    checkpoint_path: str,
    test_npz: str,
    split_metadata: str | None = None,
    batch_size: int = 256,
    device: str | None = None,
) -> dict:
    """Evaluate model on test set, optionally broken down by source."""
    print(f"\n[INFO] Loading checkpoint from {checkpoint_path}...")
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    
    if "optimal_threshold" not in checkpoint:
        raise ValueError("Checkpoint missing validation-derived threshold; refusing to tune on eval data.")
    
    threshold = float(checkpoint.get("optimal_threshold", 0.5))
    
    # Load features and apply transforms
    print(f"[INFO] Loading test set from {test_npz}...")
    sequences, labels, masks = load_session_npz(test_npz)
    
    transform_config = FeatureTransformConfig.from_checkpoint_dict(checkpoint.get("feature_transform_config"))
    sequences = apply_feature_transforms(sequences, masks, transform_config)
    
    if checkpoint.get("normalize_features", False):
        normalizer = FeatureNormalizer.from_checkpoint_dict(checkpoint.get("feature_normalizer"))
        sequences = normalizer.transform(sequences, masks)
    
    # Load model
    model = C2Transformer(
        feature_dim=FEATURE_DIM,
        seq_len=SESSION_LEN,
        use_derivative_features=bool(checkpoint.get("use_derivative_features", False)),
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Get predictions
    dataset = SessionNPZDataset(sequences, masks, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, targets = predict_probabilities(model, loader, device_obj)
    
    # Overall metrics
    overall_recall, overall_fpr, overall_auc, overall_f1, (tp, tn, fp, fn) = compute_metrics(
        targets, probs, threshold
    )
    
    results = {
        "checkpoint": checkpoint_path,
        "test_npz": test_npz,
        "threshold": threshold,
        "n_test_samples": len(targets),
        "n_c2": int((targets == 1).sum()),
        "n_benign": int((targets == 0).sum()),
        "overall": {
            "recall": overall_recall,
            "fpr": overall_fpr,
            "auc": overall_auc,
            "f1": overall_f1,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
    }
    
    # Per-source breakdown (if metadata provided)
    # TODO: Implement per-source metrics by tracking source labels in test_sessions.npz
    # For now, just report which sources contributed to the test set
    if split_metadata:
        try:
            print(f"[INFO] Loading split metadata...")
            with open(split_metadata) as f:
                meta = json.load(f)
            
            source_specs = meta.get("source_specs", [])
            if source_specs:
                print("[INFO] Test set sources:")
                for spec in source_specs:
                    npz_path, source, label_type = parse_source_spec(spec)
                    print(f"  - {source} ({Path(npz_path).name})")
        except Exception as e:
            print(f"[WARNING] Could not parse split metadata: {e}")

    
    return results


def print_results(results: dict) -> None:
    """Print evaluation results in a human-readable format."""
    print("\n" + "=" * 70)
    print("ZERO-SHOT EVALUATION REPORT")
    print("=" * 70)
    print(f"Checkpoint: {results['checkpoint']}")
    print(f"Test set: {results['test_npz']}")
    print(f"Threshold: {results['threshold']:.4f}")
    print(f"\nTest set composition:")
    print(f"  Total samples: {results['n_test_samples']:,}")
    print(f"  C2 samples: {results['n_c2']:,}")
    print(f"  Benign samples: {results['n_benign']:,}")
    
    ov = results["overall"]
    print(f"\nOVERALL METRICS:")
    print(f"  Recall:   {ov['recall']:.4f}")
    print(f"  FPR:      {ov['fpr']:.4f}")
    print(f"  AUC:      {ov['auc']:.4f}")
    print(f"  F1:       {ov['f1']:.4f}")
    print(f"  Confusion: TP={ov['tp']:,} TN={ov['tn']:,} FP={ov['fp']:,} FN={ov['fn']:,}")
    
    if "per_source" in results:
        print(f"\nPER-SOURCE BREAKDOWN:")
        for source_name, metrics in results["per_source"].items():
            print(f"\n  {source_name.upper()}:")
            print(f"    Samples: {metrics['n_samples']:,} (C2: {metrics['n_c2']:,}, Benign: {metrics['n_benign']:,})")
            print(f"    Recall:  {metrics['recall']:.4f}")
            print(f"    FPR:     {metrics['fpr']:.4f}")
            print(f"    AUC:     {metrics['auc']:.4f}")
            print(f"    F1:      {metrics['f1']:.4f}")
            print(f"    Confusion: TP={metrics['tp']:,} TN={metrics['tn']:,} FP={metrics['fp']:,} FN={metrics['fn']:,}")
    
    print("\n" + "=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot generalization evaluation")
    parser.add_argument("--model_path", required=True, help="Path to best_transformer.pth")
    parser.add_argument("--test_npz", required=True, help="Path to test_sessions.npz")
    parser.add_argument(
        "--split_metadata", default=None,
        help="Path to split_metadata.json for per-source breakdown"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default=None, help="torch device (cuda/cpu)")
    args = parser.parse_args()
    
    results = evaluate_zero_shot(
        checkpoint_path=args.model_path,
        test_npz=args.test_npz,
        split_metadata=args.split_metadata,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    print_results(results)
    
    # Save results to JSON
    results_path = Path(args.model_path).parent / "zero_shot_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[INFO] Results saved to {results_path}")


if __name__ == "__main__":
    main()
