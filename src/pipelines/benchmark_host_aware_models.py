"""Benchmark session baselines against the host-aware detector.

The harness takes an existing ``host_aware_v1`` split directory, converts the
current-session tensors into baseline-compatible session NPZs, trains:

1. ``C2Transformer`` on current sessions only.
2. ``DomainAdaptiveC2Transformer`` on current sessions only.
3. ``HostAwareDomainAdaptiveTransformer`` on current + host-history windows.

It then writes a single JSON and Markdown report with validation-frozen test
metrics at the requested FPR budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score

from src.data_loader.extended_feature_transforms import (
    ExtendedFeatureTransformConfig,
    apply_extended_feature_transforms,
)
from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.host_window_dataset import HOST_AWARE_SCHEMA_VERSION, load_host_windows_npz
from src.data_loader.normalization import FeatureNormalizer, fit_feature_normalizer
from src.data_loader.npz_utils import detect_npz_schema, load_session_npz
from src.data_loader.torch_dataset import C2SessionDataset
from src.evaluation.operating_point import find_threshold_under_fpr_budget, is_better_operating_point
from src.features.feature_config_extended import FEATURE_SCHEMA_VERSION as EXTENDED_SCHEMA_VERSION
from src.models.domain_adaptive_transformer import DomainAdaptiveC2Transformer
from src.models.transformer import C2Transformer
from src.training.train_host_aware_domain_adaptive import train_host_aware_domain_adaptive_model


DEFAULT_FPR_BUDGETS = (0.005, 0.015, 0.03)
DEFAULT_FPR_BUDGET = 0.015


@dataclass
class SessionSplitData:
    train_x: np.ndarray
    train_y: np.ndarray
    train_masks: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_masks: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_masks: np.ndarray
    feature_names: tuple[str, ...]
    schema_version: str
    session_len: int
    feature_dim: int
    normalizer: FeatureNormalizer | None
    transform_config: FeatureTransformConfig | ExtendedFeatureTransformConfig


class DomainSessionDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: np.ndarray, masks: np.ndarray, labels: np.ndarray, domain_id: int = 0):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.padding_masks = torch.tensor(~masks.astype(bool), dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.domain_ids = torch.full((len(labels),), int(domain_id), dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.padding_masks[idx], self.labels[idx], self.domain_ids[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark host-aware C2 model against session baselines")
    parser.add_argument("--host_split_dir", default="data/processed/host_aware_mvp")
    parser.add_argument("--out_dir", default="experiments/host_aware_model_benchmark")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--host_aware_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--default_fpr_budget", type=float, default=DEFAULT_FPR_BUDGET)
    parser.add_argument("--fpr_budgets", default="0.005,0.015,0.03")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_normalize_features", action="store_true")
    parser.add_argument("--no_normalize_host_features", action="store_true")
    parser.add_argument(
        "--reuse_existing_host_aware",
        action="store_true",
        help="Reuse an existing host-aware training_results.json in the benchmark output if present.",
    )
    args = parser.parse_args()

    budgets = parse_budgets(args.fpr_budgets)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    host_split_dir = Path(args.host_split_dir)
    session_npzs = convert_host_split_to_session_npzs(host_split_dir, out_dir / "baseline_session_npz")
    split_metadata = load_optional_json(host_split_dir / "host_window_split_metadata.json")
    dataset_summary = summarize_host_split(host_split_dir, split_metadata)

    print("[INFO] Loading converted session split for baselines...")
    session_data = load_preprocessed_session_splits(
        session_npzs,
        normalize_features=not args.no_normalize_features,
    )

    c2_result = train_session_baseline(
        model_name="c2_transformer",
        model_factory=lambda: C2Transformer(
            feature_dim=session_data.feature_dim,
            seq_len=session_data.session_len,
        ),
        forward_fn=lambda model, batch_x, batch_mask, batch_domain: model(batch_x, batch_mask),
        session_data=session_data,
        model_dir=out_dir / "c2_transformer",
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        default_fpr_budget=args.default_fpr_budget,
        fpr_budgets=budgets,
        seed=args.seed,
    )

    domain_name = infer_single_domain(host_split_dir)
    domain_result = train_session_baseline(
        model_name="domain_adaptive_transformer",
        model_factory=lambda: DomainAdaptiveC2Transformer(
            domains=[domain_name],
            feature_dim=session_data.feature_dim,
            seq_len=session_data.session_len,
        ),
        forward_fn=lambda model, batch_x, batch_mask, batch_domain: model(batch_x, batch_mask, batch_domain),
        session_data=session_data,
        model_dir=out_dir / "domain_adaptive_transformer",
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        default_fpr_budget=args.default_fpr_budget,
        fpr_budgets=budgets,
        seed=args.seed,
        domain_name=domain_name,
    )

    host_aware_dir = out_dir / "host_aware_domain_adaptive_transformer"
    host_aware_results_path = host_aware_dir / "training_results.json"
    if args.reuse_existing_host_aware and host_aware_results_path.exists():
        print(f"[INFO] Reusing host-aware results from {host_aware_results_path}")
        host_aware_result = json.loads(host_aware_results_path.read_text(encoding="utf-8"))
    else:
        print("[INFO] Training host-aware domain-adaptive transformer...")
        host_aware_result = train_host_aware_domain_adaptive_model(
            train_npz=str(host_split_dir / "train_host_windows.npz"),
            val_npz=str(host_split_dir / "val_host_windows.npz"),
            test_npz=str(host_split_dir / "test_host_windows.npz"),
            model_save_dir=str(host_aware_dir),
            batch_size=args.batch_size,
            epochs=args.host_aware_epochs or args.epochs,
            lr=args.lr,
            patience=args.patience,
            default_fpr_budget=args.default_fpr_budget,
            fpr_budgets=budgets,
            normalize_features=not args.no_normalize_features,
            normalize_host_features=not args.no_normalize_host_features,
            seed=args.seed,
        )

    report = {
        "benchmark_version": "host_aware_model_benchmark_v1",
        "host_split_dir": str(host_split_dir),
        "out_dir": str(out_dir),
        "dataset_summary": dataset_summary,
        "session_npzs": session_npzs,
        "settings": {
            "epochs": int(args.epochs),
            "host_aware_epochs": int(args.host_aware_epochs or args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "patience": int(args.patience),
            "seed": int(args.seed),
            "default_fpr_budget": float(args.default_fpr_budget),
            "fpr_budgets": list(budgets),
            "normalize_features": not args.no_normalize_features,
            "normalize_host_features": not args.no_normalize_host_features,
        },
        "models": {
            "c2_transformer": c2_result,
            "domain_adaptive_transformer": domain_result,
            "host_aware_domain_adaptive_transformer": normalize_host_aware_result(host_aware_result),
        },
    }

    report_path = out_dir / "comparison_report.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    markdown_path = out_dir / "comparison_report.md"
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"[SUCCESS] Wrote JSON report: {report_path}")
    print(f"[SUCCESS] Wrote Markdown report: {markdown_path}")


def convert_host_split_to_session_npzs(host_split_dir: Path, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        arrays = load_host_windows_npz(host_split_dir / f"{split_name}_host_windows.npz")
        path = out_dir / f"{split_name}_sessions.npz"
        np.savez_compressed(
            path,
            X=arrays.current_sessions.astype(np.float32),
            y=arrays.labels.astype(np.int64),
            masks=arrays.current_masks.astype(bool),
            feature_names=np.asarray(arrays.feature_names),
            schema_version=np.asarray(arrays.session_schema_version),
            source_host_schema_version=np.asarray(arrays.schema_version),
            host_ids=arrays.host_ids,
            dest_ids=arrays.dest_ids,
            timestamps=arrays.timestamps.astype(np.float64),
            sources=arrays.sources,
            families=arrays.families,
            capture_ids=arrays.capture_ids,
            real_host_identity=np.asarray(arrays.real_host_identity, dtype=bool),
        )
        paths[split_name] = str(path)
    return paths


def load_preprocessed_session_splits(
    session_npzs: dict[str, str],
    *,
    normalize_features: bool,
) -> SessionSplitData:
    train_schema = detect_npz_schema(session_npzs["train"])
    feature_names = tuple(str(name) for name in train_schema["feature_names"])
    schema_version = str(train_schema["schema_version"])
    session_len = int(train_schema["session_len"])
    feature_dim = int(train_schema["feature_dim"])

    transform_config: FeatureTransformConfig | ExtendedFeatureTransformConfig
    if schema_version == EXTENDED_SCHEMA_VERSION:
        transform_config = ExtendedFeatureTransformConfig()
    else:
        transform_config = FeatureTransformConfig()

    train_x, train_y, train_masks = load_session_npz(
        session_npzs["train"],
        expected_feature_names=feature_names,
        expected_session_len=session_len,
    )
    val_x, val_y, val_masks = load_session_npz(
        session_npzs["val"],
        expected_feature_names=feature_names,
        expected_session_len=session_len,
    )
    test_x, test_y, test_masks = load_session_npz(
        session_npzs["test"],
        expected_feature_names=feature_names,
        expected_session_len=session_len,
    )

    train_x = apply_transforms(train_x, train_masks, schema_version, transform_config, feature_names)
    val_x = apply_transforms(val_x, val_masks, schema_version, transform_config, feature_names)
    test_x = apply_transforms(test_x, test_masks, schema_version, transform_config, feature_names)

    normalizer = None
    if normalize_features:
        normalizer = fit_feature_normalizer(train_x, train_masks)
        train_x = normalizer.transform(train_x, train_masks)
        val_x = normalizer.transform(val_x, val_masks)
        test_x = normalizer.transform(test_x, test_masks)

    return SessionSplitData(
        train_x=train_x,
        train_y=train_y,
        train_masks=train_masks,
        val_x=val_x,
        val_y=val_y,
        val_masks=val_masks,
        test_x=test_x,
        test_y=test_y,
        test_masks=test_masks,
        feature_names=feature_names,
        schema_version=schema_version,
        session_len=session_len,
        feature_dim=feature_dim,
        normalizer=normalizer,
        transform_config=transform_config,
    )


def apply_transforms(
    sequences: np.ndarray,
    masks: np.ndarray,
    schema_version: str,
    transform_config: FeatureTransformConfig | ExtendedFeatureTransformConfig,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    if schema_version == EXTENDED_SCHEMA_VERSION:
        return apply_extended_feature_transforms(sequences, masks, transform_config)  # type: ignore[arg-type]
    return apply_feature_transforms(sequences, masks, transform_config, feature_names=feature_names)


def train_session_baseline(
    *,
    model_name: str,
    model_factory: Callable[[], torch.nn.Module],
    forward_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    session_data: SessionSplitData,
    model_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    patience: int,
    default_fpr_budget: float,
    fpr_budgets: tuple[float, ...],
    seed: int,
    domain_name: str = "session",
) -> dict[str, object]:
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_model.pth"
    results_path = model_dir / "training_results.json"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = DomainSessionDataset(session_data.train_x, session_data.train_masks, session_data.train_y)
    val_ds = DomainSessionDataset(session_data.val_x, session_data.val_masks, session_data.val_y)
    test_ds = DomainSessionDataset(session_data.test_x, session_data.test_masks, session_data.test_y)

    label_counts = torch.bincount(train_ds.labels, minlength=2).float()
    sample_weights = (1.0 / (label_counts[train_ds.labels] + 1e-6)).tolist()
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_recall = 0.0
    best_f1 = 0.0
    best_fpr = float("inf")
    best_val_metrics: dict[str, dict] = {}
    epochs_without_improvement = 0
    default_key = f"{default_fpr_budget:.4f}"

    print(f"[INFO] Training {model_name} on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_mask, batch_y, batch_domain in train_loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.float().to(device)
            batch_domain = batch_domain.to(device)

            optimizer.zero_grad()
            logits = forward_fn(model, batch_x, batch_mask, batch_domain)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        val_probs, val_targets = score_session_model(model, val_loader, device, forward_fn)
        val_metrics_by_budget = budget_metrics(val_targets, val_probs, fpr_budgets)
        default_metrics = val_metrics_by_budget[default_key]
        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] {model_name} | "
            f"loss={total_loss / max(len(train_loader), 1):.4f} | "
            f"val_auc={default_metrics['auc']:.4f} | "
            f"val_recall@{default_key}={default_metrics['recall']:.4f} | "
            f"val_fpr={default_metrics['fpr']:.4f} | "
            f"val_f1={default_metrics['f1']:.4f}"
        )

        improved = epoch == 0 or is_better_operating_point(
            candidate_recall=default_metrics["recall"],
            candidate_f1=default_metrics["f1"],
            candidate_fpr=default_metrics["fpr"],
            best_recall=best_recall,
            best_f1=best_f1,
            best_fpr=best_fpr,
        )
        if improved:
            best_recall = default_metrics["recall"]
            best_f1 = default_metrics["f1"]
            best_fpr = default_metrics["fpr"]
            best_val_metrics = val_metrics_by_budget
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": model_name,
                    "domain_name": domain_name,
                    "schema_version": session_data.schema_version,
                    "feature_names": list(session_data.feature_names),
                    "feature_dim": session_data.feature_dim,
                    "session_len": session_data.session_len,
                    "normalize_features": session_data.normalizer is not None,
                    "feature_normalizer": session_data.normalizer.to_checkpoint_dict()
                    if session_data.normalizer
                    else None,
                    "feature_transform_config": session_data.transform_config.to_checkpoint_dict(),
                    "fpr_budgets": list(fpr_budgets),
                    "default_fpr_budget": default_fpr_budget,
                    "validation_metrics_by_budget": best_val_metrics,
                },
                best_model_path,
            )
            print(f"  --> {model_name} improved. Saved to {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[INFO] Early stopping {model_name} after {epoch + 1} epochs.")
                break

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    fixed_thresholds = {
        key: metrics["threshold"]
        for key, metrics in checkpoint["validation_metrics_by_budget"].items()
    }
    test_probs, test_targets = score_session_model(model, test_loader, device, forward_fn)
    test_metrics_by_budget = budget_metrics(
        test_targets,
        test_probs,
        fpr_budgets,
        fixed_thresholds=fixed_thresholds,
    )

    results = {
        "model_name": model_name,
        "best_model_path": str(best_model_path),
        "results_path": str(results_path),
        "validation_metrics_by_budget": checkpoint["validation_metrics_by_budget"],
        "test_metrics_by_budget": test_metrics_by_budget,
        "schema_version": session_data.schema_version,
        "feature_dim": session_data.feature_dim,
        "session_len": session_data.session_len,
        "domain_name": domain_name,
        "epochs_requested": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
    }
    results_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    return results


def score_session_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    forward_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    probs: list[float] = []
    targets: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_mask, batch_y, batch_domain in loader:
            logits = forward_fn(
                model,
                batch_x.to(device),
                batch_mask.to(device),
                batch_domain.to(device),
            )
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            targets.extend(batch_y.numpy().tolist())
    return np.asarray(probs, dtype=np.float64), np.asarray(targets, dtype=np.int64)


def metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, object]:
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


def budget_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    budgets: Iterable[float],
    *,
    fixed_thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for budget in budgets:
        key = f"{budget:.4f}"
        if fixed_thresholds is None:
            threshold, _, _, feasible = find_threshold_under_fpr_budget(y_true, probs, max_fpr=budget)
            metrics = metrics_at_threshold(y_true, probs, threshold)
            metrics["feasible"] = bool(feasible)
        else:
            metrics = metrics_at_threshold(y_true, probs, float(fixed_thresholds[key]))
            metrics["feasible"] = bool(metrics["fpr"] <= budget + 1e-12)
        out[key] = metrics
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = 0.0
    for metrics in out.values():
        metrics["auc"] = auc
    return out


def normalize_host_aware_result(result: dict[str, object]) -> dict[str, object]:
    normalized = dict(result)
    normalized["model_name"] = "host_aware_domain_adaptive_transformer"
    if "results_path" not in normalized and "best_model_path" in normalized:
        normalized["results_path"] = str(Path(str(normalized["best_model_path"])).parent / "training_results.json")
    return normalized


def summarize_host_split(host_split_dir: Path, metadata: dict[str, object]) -> dict[str, object]:
    summary = {
        "schema_version": HOST_AWARE_SCHEMA_VERSION,
        "path": str(host_split_dir),
        "metadata": metadata,
        "splits": {},
    }
    for split_name in ("train", "val", "test"):
        arrays = load_host_windows_npz(host_split_dir / f"{split_name}_host_windows.npz")
        summary["splits"][split_name] = {
            "samples": int(len(arrays.labels)),
            "positive": int((arrays.labels == 1).sum()),
            "benign": int((arrays.labels == 0).sum()),
            "hosts": int(len(set(str(host) for host in arrays.host_ids.tolist()))),
            "real_host_identity": bool(arrays.real_host_identity),
            "sources": sorted(set(str(source) for source in arrays.sources.tolist())),
            "families": sorted(set(str(family) for family in arrays.families.tolist() if str(family))),
        }
    return summary


def infer_single_domain(host_split_dir: Path) -> str:
    arrays = load_host_windows_npz(host_split_dir / "train_host_windows.npz")
    sources = sorted(set(str(source) for source in arrays.sources.tolist()))
    return sources[0] if len(sources) == 1 else "benchmark_mixed"


def render_markdown_report(report: dict[str, object]) -> str:
    default_key = f"{float(report['settings']['default_fpr_budget']):.4f}"
    models = report["models"]
    lines = [
        "# Host-Aware Model Benchmark",
        "",
        f"Host split: `{report['host_split_dir']}`",
        f"Default FPR budget: `{default_key}`",
        "",
        "## Dataset",
        "",
        "| Split | Samples | Positive | Benign | Hosts |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    splits = report["dataset_summary"]["splits"]
    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        lines.append(
            f"| {split_name} | {split['samples']} | {split['positive']} | {split['benign']} | {split['hosts']} |"
        )

    warnings = (
        report["dataset_summary"]
        .get("metadata", {})
        .get("evaluation_readiness", {})
        .get("warnings", [])
    )
    if warnings:
        lines.extend(["", "Dataset warnings:"])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            f"## Default Budget ({default_key})",
            "",
            "| Model | Val Recall | Val FPR | Val F1 | Val AUC | Test Recall | Test FPR | Test F1 | Test AUC | Feasible |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for model_key, result in models.items():
        val = result["validation_metrics_by_budget"][default_key]
        test = result["test_metrics_by_budget"][default_key]
        lines.append(
            f"| {model_key} | "
            f"{val['recall']:.4f} | {val['fpr']:.4f} | {val['f1']:.4f} | {val['auc']:.4f} | "
            f"{test['recall']:.4f} | {test['fpr']:.4f} | {test['f1']:.4f} | {test['auc']:.4f} | "
            f"{test['feasible']} |"
        )

    lines.extend(["", "## All Budgets", ""])
    for budget in report["settings"]["fpr_budgets"]:
        key = f"{float(budget):.4f}"
        lines.extend(
            [
                f"### FPR <= {key}",
                "",
                "| Model | Val Recall/FPR/F1 | Test Recall/FPR/F1 | Test Feasible |",
                "| --- | --- | --- | --- |",
            ]
        )
        for model_key, result in models.items():
            val = result["validation_metrics_by_budget"][key]
            test = result["test_metrics_by_budget"][key]
            lines.append(
                f"| {model_key} | "
                f"{val['recall']:.4f} / {val['fpr']:.4f} / {val['f1']:.4f} | "
                f"{test['recall']:.4f} / {test['fpr']:.4f} / {test['f1']:.4f} | "
                f"{test['feasible']} |"
            )
        lines.append("")

    lines.extend(["## Artifacts", ""])
    for model_key, result in models.items():
        lines.append(f"- {model_key}: `{result['best_model_path']}`")
    return "\n".join(lines) + "\n"


def parse_budgets(raw_value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw_value.split(",") if part.strip())


def load_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
