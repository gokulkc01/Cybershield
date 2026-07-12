"""Train the host-aware domain-adaptive Transformer research MVP."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score

from src.data_loader.extended_feature_transforms import (
    ExtendedFeatureTransformConfig,
    apply_extended_feature_transforms,
)
from src.data_loader.host_window_dataset import (
    HOST_AWARE_SCHEMA_VERSION,
    HostWindowArrays,
    HostWindowTorchDataset,
    load_host_windows_npz,
)
from src.data_loader.normalization import FeatureNormalizer, fit_feature_normalizer
from src.evaluation.operating_point import find_threshold_under_fpr_budget, is_better_operating_point
from src.features.feature_config_extended import FEATURE_SCHEMA_VERSION as EXTENDED_SCHEMA_VERSION
from src.models.host_aware_domain_adaptive_transformer import HostAwareDomainAdaptiveTransformer


DEFAULT_FPR_BUDGETS = (0.005, 0.015, 0.03)
DEFAULT_FPR_BUDGET = 0.015
DEFAULT_RESULTS_FILENAME = "training_results.json"


@dataclass(frozen=True)
class HostFeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def to_checkpoint_dict(self) -> dict:
        return {
            "mean": self.mean.astype(np.float32),
            "std": self.std.astype(np.float32),
            "eps": self.eps,
        }


def _fit_host_feature_normalizer(values: np.ndarray, eps: float = 1e-6) -> HostFeatureNormalizer:
    mean = values.astype(np.float32).mean(axis=0)
    std = values.astype(np.float32).std(axis=0)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return HostFeatureNormalizer(mean=mean, std=std, eps=eps)


def _apply_window_feature_transforms(
    arrays: HostWindowArrays,
    transform_config: ExtendedFeatureTransformConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if arrays.session_schema_version != EXTENDED_SCHEMA_VERSION:
        return arrays.current_sessions, arrays.history_sessions

    current = apply_extended_feature_transforms(
        arrays.current_sessions,
        arrays.current_masks,
        transform_config,
    )
    n, history_size, session_len, feature_dim = arrays.history_sessions.shape
    history_flat = arrays.history_sessions.reshape(n * history_size, session_len, feature_dim)
    history_masks_flat = arrays.history_flow_masks.reshape(n * history_size, session_len)
    history_flat = apply_extended_feature_transforms(
        history_flat,
        history_masks_flat,
        transform_config,
    )
    return current, history_flat.reshape(n, history_size, session_len, feature_dim)


def _fit_window_normalizer(current: np.ndarray, current_masks: np.ndarray, history: np.ndarray, history_masks: np.ndarray) -> FeatureNormalizer:
    n, history_size, session_len, feature_dim = history.shape
    history_flat = history.reshape(n * history_size, session_len, feature_dim)
    history_masks_flat = history_masks.reshape(n * history_size, session_len)
    sequences = np.concatenate([current, history_flat], axis=0)
    masks = np.concatenate([current_masks, history_masks_flat], axis=0)
    return fit_feature_normalizer(sequences, masks)


def _transform_history(normalizer: FeatureNormalizer, history: np.ndarray, history_masks: np.ndarray) -> np.ndarray:
    n, history_size, session_len, feature_dim = history.shape
    history_flat = history.reshape(n * history_size, session_len, feature_dim)
    history_masks_flat = history_masks.reshape(n * history_size, session_len)
    transformed = normalizer.transform(history_flat, history_masks_flat)
    return transformed.reshape(n, history_size, session_len, feature_dim)


def _preprocess_splits(
    train: HostWindowArrays,
    val: HostWindowArrays,
    test: HostWindowArrays,
    *,
    normalize_features: bool,
    normalize_host_features: bool,
    transform_config: ExtendedFeatureTransformConfig,
) -> tuple[HostWindowArrays, HostWindowArrays, HostWindowArrays, FeatureNormalizer | None, HostFeatureNormalizer | None]:
    train_current, train_history = _apply_window_feature_transforms(train, transform_config)
    val_current, val_history = _apply_window_feature_transforms(val, transform_config)
    test_current, test_history = _apply_window_feature_transforms(test, transform_config)

    feature_normalizer = None
    if normalize_features:
        feature_normalizer = _fit_window_normalizer(
            train_current,
            train.current_masks,
            train_history,
            train.history_flow_masks,
        )
        train_current = feature_normalizer.transform(train_current, train.current_masks)
        val_current = feature_normalizer.transform(val_current, val.current_masks)
        test_current = feature_normalizer.transform(test_current, test.current_masks)
        train_history = _transform_history(feature_normalizer, train_history, train.history_flow_masks)
        val_history = _transform_history(feature_normalizer, val_history, val.history_flow_masks)
        test_history = _transform_history(feature_normalizer, test_history, test.history_flow_masks)

    host_feature_normalizer = None
    if normalize_host_features:
        host_feature_normalizer = _fit_host_feature_normalizer(train.host_features)
        train.host_features = host_feature_normalizer.transform(train.host_features)
        val.host_features = host_feature_normalizer.transform(val.host_features)
        test.host_features = host_feature_normalizer.transform(test.host_features)

    return (
        train.copy_with(current_sessions=train_current, history_sessions=train_history),
        val.copy_with(current_sessions=val_current, history_sessions=val_history),
        test.copy_with(current_sessions=test_current, history_sessions=test_history),
        feature_normalizer,
        host_feature_normalizer,
    )


def _score_model(
    model: HostAwareDomainAdaptiveTransformer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probs: list[float] = []
    targets: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (
                current,
                current_padding,
                history,
                history_flow_padding,
                history_session_mask,
                host_features,
                labels,
                domain_ids,
            ) = batch
            outputs = model(
                current.to(device),
                current_padding.to(device),
                history.to(device),
                history_flow_padding.to(device),
                history_session_mask.to(device),
                host_features.to(device),
                domain_ids.to(device),
            )
            probs.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return np.asarray(probs, dtype=np.float64), np.asarray(targets, dtype=np.int64)


def _metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _budget_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    budgets: Iterable[float],
    *,
    fixed_thresholds: dict[str, float] | None = None,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for budget in budgets:
        key = f"{budget:.4f}"
        if fixed_thresholds is None:
            threshold, _, _, feasible = find_threshold_under_fpr_budget(y_true, probs, max_fpr=budget)
            metrics = _metrics_at_threshold(y_true, probs, threshold)
            metrics["feasible"] = bool(feasible)
        else:
            metrics = _metrics_at_threshold(y_true, probs, float(fixed_thresholds[key]))
            metrics["feasible"] = metrics["fpr"] <= budget + 1e-12
        out[key] = metrics
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = 0.0
    for metrics in out.values():
        metrics["auc"] = auc
    return out


def train_host_aware_domain_adaptive_model(
    train_npz: str,
    val_npz: str,
    test_npz: str,
    model_save_dir: str,
    *,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-4,
    patience: int = 6,
    default_fpr_budget: float = DEFAULT_FPR_BUDGET,
    fpr_budgets: tuple[float, ...] = DEFAULT_FPR_BUDGETS,
    normalize_features: bool = True,
    normalize_host_features: bool = True,
    auxiliary_loss_weight: float = 0.2,
    seed: int = 42,
) -> dict:
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_host_aware_transformer.pth")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_arrays = load_host_windows_npz(train_npz)
    val_arrays = load_host_windows_npz(val_npz)
    test_arrays = load_host_windows_npz(test_npz)

    if train_arrays.schema_version != HOST_AWARE_SCHEMA_VERSION:
        raise ValueError(f"Expected {HOST_AWARE_SCHEMA_VERSION}, got {train_arrays.schema_version}")
    if not train_arrays.real_host_identity:
        print("[WARN] Training split does not claim real host identity; use only for smoke tests.")

    transform_config = ExtendedFeatureTransformConfig()
    train_arrays, val_arrays, test_arrays, feature_normalizer, host_feature_normalizer = _preprocess_splits(
        train_arrays,
        val_arrays,
        test_arrays,
        normalize_features=normalize_features,
        normalize_host_features=normalize_host_features,
        transform_config=transform_config,
    )

    domains = sorted(
        set(str(source) for source in train_arrays.sources.tolist())
        | set(str(source) for source in val_arrays.sources.tolist())
        | set(str(source) for source in test_arrays.sources.tolist())
    )
    domain_to_index = {domain: idx for idx, domain in enumerate(domains)}

    train_ds = HostWindowTorchDataset(train_arrays, domain_to_index=domain_to_index)
    val_ds = HostWindowTorchDataset(val_arrays, domain_to_index=domain_to_index)
    test_ds = HostWindowTorchDataset(test_arrays, domain_to_index=domain_to_index)

    sample_keys = train_ds.domain_ids * 2 + train_ds.labels
    key_counts = torch.bincount(sample_keys, minlength=len(domains) * 2).float()
    sample_weights = (1.0 / (key_counts[sample_keys] + 1e-6)).tolist()
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HostAwareDomainAdaptiveTransformer(
        domains=domains,
        feature_dim=train_arrays.feature_dim,
        host_feature_dim=train_arrays.host_feature_dim,
        seq_len=train_arrays.session_len,
        history_size=train_arrays.history_size,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_recall = 0.0
    best_f1 = 0.0
    best_fpr = float("inf")
    epochs_without_improvement = 0
    best_val_metrics: dict[str, dict] = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            (
                current,
                current_padding,
                history,
                history_flow_padding,
                history_session_mask,
                host_features,
                labels,
                domain_ids,
            ) = batch
            labels_float = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(
                current.to(device),
                current_padding.to(device),
                history.to(device),
                history_flow_padding.to(device),
                history_session_mask.to(device),
                host_features.to(device),
                domain_ids.to(device),
                return_components=True,
            )
            loss = criterion(outputs["c2_logit"], labels_float)
            if auxiliary_loss_weight > 0:
                loss = loss + auxiliary_loss_weight * (
                    criterion(outputs["session_logit"], labels_float)
                    + criterion(outputs["host_logit"], labels_float)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        val_probs, val_targets = _score_model(model, val_loader, device)
        val_metrics_by_budget = _budget_metrics(val_targets, val_probs, fpr_budgets)
        default_key = f"{default_fpr_budget:.4f}"
        default_metrics = val_metrics_by_budget[default_key]

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] | "
            f"loss={total_loss / max(len(train_loader), 1):.4f} | "
            f"val_auc={default_metrics['auc']:.4f} | "
            f"val_recall@{default_key}={default_metrics['recall']:.4f} | "
            f"val_fpr={default_metrics['fpr']:.4f} | "
            f"val_f1={default_metrics['f1']:.4f}"
        )

        if is_better_operating_point(
            candidate_recall=default_metrics["recall"],
            candidate_f1=default_metrics["f1"],
            candidate_fpr=default_metrics["fpr"],
            best_recall=best_recall,
            best_f1=best_f1,
            best_fpr=best_fpr,
        ):
            best_recall = default_metrics["recall"]
            best_f1 = default_metrics["f1"]
            best_fpr = default_metrics["fpr"]
            best_val_metrics = val_metrics_by_budget
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": "host_aware_domain_adaptive_transformer",
                    "domains": domains,
                    "domain_to_index": domain_to_index,
                    "schema_version": HOST_AWARE_SCHEMA_VERSION,
                    "session_schema_version": train_arrays.session_schema_version,
                    "feature_names": list(train_arrays.feature_names),
                    "host_feature_names": list(train_arrays.host_feature_names),
                    "feature_dim": train_arrays.feature_dim,
                    "host_feature_dim": train_arrays.host_feature_dim,
                    "session_len": train_arrays.session_len,
                    "history_size": train_arrays.history_size,
                    "normalize_features": normalize_features,
                    "normalize_host_features": normalize_host_features,
                    "feature_normalizer": feature_normalizer.to_checkpoint_dict() if feature_normalizer else None,
                    "host_feature_normalizer": host_feature_normalizer.to_checkpoint_dict() if host_feature_normalizer else None,
                    "feature_transform_config": transform_config.to_checkpoint_dict(),
                    "fpr_budgets": list(fpr_budgets),
                    "default_fpr_budget": default_fpr_budget,
                    "validation_metrics_by_budget": best_val_metrics,
                    "real_host_identity": bool(train_arrays.real_host_identity),
                },
                best_model_path,
            )
            print(f"  --> Model improved. Saved to {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[INFO] Early stopping after {epoch + 1} epochs.")
                break

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    fixed_thresholds = {
        key: metrics["threshold"]
        for key, metrics in checkpoint["validation_metrics_by_budget"].items()
    }
    test_probs, test_targets = _score_model(model, test_loader, device)
    test_metrics_by_budget = _budget_metrics(
        test_targets,
        test_probs,
        fpr_budgets,
        fixed_thresholds=fixed_thresholds,
    )
    print("Final test metrics with validation-frozen thresholds:")
    for budget_key, metrics in test_metrics_by_budget.items():
        print(
            f"  FPR<={budget_key}: auc={metrics['auc']:.4f} "
            f"recall={metrics['recall']:.4f} fpr={metrics['fpr']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )

    results = {
        "best_model_path": best_model_path,
        "validation_metrics_by_budget": checkpoint["validation_metrics_by_budget"],
        "test_metrics_by_budget": test_metrics_by_budget,
        "test_scores": np.asarray(test_probs, dtype=np.float64).tolist(),
        "test_labels": np.asarray(test_targets, dtype=np.int64).tolist(),
        "train_npz": train_npz,
        "val_npz": val_npz,
        "test_npz": test_npz,
        "domains": domains,
        "schema_version": HOST_AWARE_SCHEMA_VERSION,
        "session_schema_version": train_arrays.session_schema_version,
        "feature_dim": train_arrays.feature_dim,
        "host_feature_dim": train_arrays.host_feature_dim,
        "session_len": train_arrays.session_len,
        "history_size": train_arrays.history_size,
        "real_host_identity": bool(train_arrays.real_host_identity),
        "default_fpr_budget": default_fpr_budget,
        "fpr_budgets": list(fpr_budgets),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
    }
    results_path = os.path.join(model_save_dir, DEFAULT_RESULTS_FILENAME)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, indent=2)
    print(f"Saved training results to {results_path}")
    results["results_path"] = results_path
    return results


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _parse_budgets(raw_value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw_value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train host-aware domain-adaptive C2 detector")
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--model_save_dir", default="experiments/host_aware_domain_adaptive")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--default_fpr_budget", type=float, default=DEFAULT_FPR_BUDGET)
    parser.add_argument("--fpr_budgets", default="0.005,0.015,0.03")
    parser.add_argument("--auxiliary_loss_weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_normalize_features", action="store_true")
    parser.add_argument("--no_normalize_host_features", action="store_true")
    args = parser.parse_args()

    train_host_aware_domain_adaptive_model(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        test_npz=args.test_npz,
        model_save_dir=args.model_save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        default_fpr_budget=args.default_fpr_budget,
        fpr_budgets=_parse_budgets(args.fpr_budgets),
        normalize_features=not args.no_normalize_features,
        normalize_host_features=not args.no_normalize_host_features,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
