"""Run a small feature-bias ablation study and compare variants on internal and external data."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.evaluation.evaluate_holdout import evaluate_holdout
from src.evaluation.evaluate_transformer import evaluate_checkpoint
from src.training.train_transformer import train_model


@dataclass(frozen=True)
class VariantConfig:
    name: str
    ablate_features: tuple[str, ...]


@dataclass(frozen=True)
class VariantSummary:
    name: str
    ablated_features: list[str]
    checkpoint_path: str
    internal_holdout: dict[str, Any]
    external: dict[str, Any]
    delta_internal: dict[str, float]
    delta_external: dict[str, float]


def _parse_feature_list(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return ()
    return tuple(feature.strip() for feature in raw_value.split(",") if feature.strip())


def _make_variants() -> list[VariantConfig]:
    return [
        VariantConfig(name="baseline", ablate_features=()),
        VariantConfig(name="drop_ports", ablate_features=("src_port", "dst_port")),
        VariantConfig(name="drop_is_outbound", ablate_features=("is_outbound",)),
        VariantConfig(name="drop_ports_and_is_outbound", ablate_features=("src_port", "dst_port", "is_outbound")),
    ]


def _metrics_dict(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _delta_metrics(baseline: dict[str, Any], candidate: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(candidate.get(key, 0.0) - baseline.get(key, 0.0)) for key in keys}


def run_ablation_study(
    base_npz_path: str,
    external_npz_path: str,
    output_dir: str,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-4,
    patience: int = 8,
    min_flows: int = 5,
    focal_alpha: float = 0.5,
    focal_gamma: float = 2.0,
    max_fpr_budget: float = 0.005,
    guard_band: float = 0.8,
    normalize_features: bool = True,
    use_derivative_features: bool = False,
    log_scale_features: tuple[str, ...] | None = None,
    device: str | None = None,
) -> list[VariantSummary]:
    os.makedirs(output_dir, exist_ok=True)
    variants = _make_variants()
    summaries: list[VariantSummary] = []

    for variant in variants:
        variant_dir = os.path.join(output_dir, variant.name)
        print(f"\n[ABLAT] Training variant: {variant.name} | ablate={list(variant.ablate_features)}")
        checkpoint_path = train_model(
            npz_path=base_npz_path,
            model_save_dir=variant_dir,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience,
            min_flows=min_flows,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            max_fpr_budget=max_fpr_budget,
            fpr_guard_band=guard_band,
            use_derivative_features=use_derivative_features,
            normalize_features=normalize_features,
            log_scale_features=log_scale_features,
            ablate_features=variant.ablate_features,
            run_final_test_eval=False,
        )

        internal = evaluate_holdout(
            checkpoint_path=checkpoint_path,
            npz_path=base_npz_path,
            batch_size=batch_size,
            max_fpr_budget=max_fpr_budget,
            device=device,
        )
        external = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            npz_path=external_npz_path,
            batch_size=batch_size,
            max_fpr_budget=max_fpr_budget,
            device=device,
        )

        summaries.append(
            VariantSummary(
                name=variant.name,
                ablated_features=list(variant.ablate_features),
                checkpoint_path=checkpoint_path,
                internal_holdout=_metrics_dict(internal),
                external=_metrics_dict(external),
                delta_internal={},
                delta_external={},
            )
        )

    if summaries:
        baseline_internal = summaries[0].internal_holdout
        baseline_external = summaries[0].external
        updated_summaries: list[VariantSummary] = []
        for summary in summaries:
            updated_summaries.append(
                VariantSummary(
                    name=summary.name,
                    ablated_features=summary.ablated_features,
                    checkpoint_path=summary.checkpoint_path,
                    internal_holdout=summary.internal_holdout,
                    external=summary.external,
                    delta_internal=_delta_metrics(baseline_internal, summary.internal_holdout, ("recall", "fpr", "f1", "auc", "expected_false_alerts_per_day")),
                    delta_external=_delta_metrics(baseline_external, summary.external, ("recall_at_budget", "fpr_at_budget", "f1", "auc", "expected_false_alerts_per_day")),
                )
            )
        summaries = updated_summaries

    return summaries


def format_summary(summaries: list[VariantSummary]) -> str:
    lines = ["Feature ablation summary:"]
    for summary in summaries:
        ext = summary.external
        lines.append(
            f"- {summary.name}: ext recall={ext['recall_at_budget']:.4f}, ext fpr={ext['fpr_at_budget']:.4f}, ext auc={ext['auc']:.4f}, budget_met={ext['budget_met']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a feature ablation study for CyberShield")
    parser.add_argument("--base_npz_path", default="data/processed/mcfp_multi_plus_benign_sessions.npz")
    parser.add_argument("--external_npz_path", default="data/processed/ctu13_external_unseen_sample_30k.npz")
    parser.add_argument("--output_dir", default="experiments/ablation_results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_flows", type=int, default=5)
    parser.add_argument("--focal_alpha", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--max_fpr_budget", type=float, default=0.005)
    parser.add_argument("--guard_band", type=float, default=0.8)
    parser.add_argument("--normalize_features", action="store_true", default=True)
    parser.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    parser.add_argument("--use_derivative_features", action="store_true")
    parser.add_argument("--log_scale_features", default=None, help="Comma-separated feature names to log-scale")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    summaries = run_ablation_study(
        base_npz_path=args.base_npz_path,
        external_npz_path=args.external_npz_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        min_flows=args.min_flows,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        max_fpr_budget=args.max_fpr_budget,
        guard_band=args.guard_band,
        normalize_features=args.normalize_features,
        use_derivative_features=args.use_derivative_features,
        log_scale_features=_parse_feature_list(args.log_scale_features),
        device=args.device,
    )

    print(format_summary(summaries))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(_json_safe([asdict(summary) for summary in summaries]), handle, indent=2)


if __name__ == "__main__":
    main()
