"""Phase 1 host-context ablations for the host-aware domain-adaptive model.

Addresses the Phase 0 watch item: the host-aware model saturates the
same-family validation axis (val AUC 0.995-1.0). Before crediting its
test-axis advantage to the architecture, ablate the host-context inputs:

- ``no_host_features``: zero the 15 host summary features (history volume,
  gaps, age, ...), keep the raw history sessions.
- ``no_history``: mask out all history sessions, keep the host summary
  features.
- ``no_host_context``: both ablated - the host branch carries no
  sample-dependent information (sanity: should approach the session-only
  transformer).
- ``history_4`` / ``history_1``: truncate history to the K most recent
  sessions (dose-response on history length).
- ``full``: the unablated Phase 0 model, reused from an existing benchmark
  directory when available.

Each variant trains once per seed with the same settings as the Phase 0
benchmark (30 epochs, patience 5, thresholds frozen on val) and reports the
honest test metrics. Reports are re-written after every completed run so a
long background sweep yields readable partial results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.benchmark_host_aware_models import (
    aggregate_model_metrics,
    attach_honest_test_report,
    format_mean_ci,
    json_safe,
    normalize_host_aware_result,
    parse_budgets,
    parse_seeds,
)
from src.training.train_host_aware_domain_adaptive import train_host_aware_domain_adaptive_model

VARIANTS: dict[str, dict] = {
    "full": {},
    "no_host_features": {"ablate_host_features": True},
    "no_history": {"history_limit": 0},
    "no_host_context": {"ablate_host_features": True, "history_limit": 0},
    "history_4": {"history_limit": 4},
    "history_1": {"history_limit": 1},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate host-context inputs of the host-aware model")
    parser.add_argument("--host_split_dir", default="data/processed/host_aware_ctu13_same_provenance")
    parser.add_argument("--out_dir", default="experiments/phase1_host_aware_ablations")
    parser.add_argument(
        "--full_results_dir",
        default="experiments/host_aware_benchmark_ctu13_same_provenance",
        help="Phase 0 benchmark dir; per-seed host-aware results there are reused as the 'full' variant.",
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--default_fpr_budget", type=float, default=0.015)
    parser.add_argument("--fpr_budgets", default="0.005,0.015,0.03")
    parser.add_argument("--base_rates", default="0.001,0.01")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    budgets = parse_budgets(args.fpr_budgets)
    base_rates = parse_budgets(args.base_rates)
    seeds = parse_seeds(args.seeds)
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variant_names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {list(VARIANTS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    host_split_dir = Path(args.host_split_dir)

    per_seed_results: dict[str, dict[str, dict]] = {name: {} for name in variant_names}
    for variant in variant_names:
        for seed in seeds:
            result = load_or_train_variant(
                variant,
                seed,
                args=args,
                host_split_dir=host_split_dir,
                out_dir=out_dir,
                budgets=budgets,
                base_rates=base_rates,
            )
            per_seed_results[variant][str(seed)] = result
            write_reports(per_seed_results, out_dir, args, budgets, base_rates)
    print(f"[SUCCESS] Ablation sweep complete: {out_dir / 'ablation_report.md'}")


def load_or_train_variant(
    variant: str,
    seed: int,
    *,
    args: argparse.Namespace,
    host_split_dir: Path,
    out_dir: Path,
    budgets: tuple[float, ...],
    base_rates: tuple[float, ...],
) -> dict:
    variant_dir = out_dir / variant / f"seed_{seed}"
    results_path = variant_dir / "training_results.json"

    if results_path.exists():
        print(f"[INFO] Reusing existing {variant} seed {seed}: {results_path}")
        result = json.loads(results_path.read_text(encoding="utf-8"))
    elif variant == "full":
        phase0_path = (
            Path(args.full_results_dir)
            / f"seed_{seed}"
            / "host_aware_domain_adaptive_transformer"
            / "training_results.json"
        )
        if phase0_path.exists():
            print(f"[INFO] Reusing Phase 0 full-model result: {phase0_path}")
            result = json.loads(phase0_path.read_text(encoding="utf-8"))
        else:
            result = train_variant(variant, seed, args, host_split_dir, variant_dir)
    else:
        result = train_variant(variant, seed, args, host_split_dir, variant_dir)

    result = normalize_host_aware_result(result)
    result["model_name"] = f"host_aware_{variant}"
    attach_honest_test_report(
        result,
        default_fpr_budget=args.default_fpr_budget,
        fpr_budgets=budgets,
        base_rates=base_rates,
        n_bootstrap=args.n_bootstrap,
        random_seed=seed,
    )
    return result


def train_variant(
    variant: str,
    seed: int,
    args: argparse.Namespace,
    host_split_dir: Path,
    variant_dir: Path,
) -> dict:
    knobs = VARIANTS[variant]
    print(f"\n[INFO] ===== Training variant '{variant}' seed {seed} =====")
    return train_host_aware_domain_adaptive_model(
        train_npz=str(host_split_dir / "train_host_windows.npz"),
        val_npz=str(host_split_dir / "val_host_windows.npz"),
        test_npz=str(host_split_dir / "test_host_windows.npz"),
        model_save_dir=str(variant_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        default_fpr_budget=args.default_fpr_budget,
        fpr_budgets=parse_budgets(args.fpr_budgets),
        seed=seed,
        ablate_host_features=bool(knobs.get("ablate_host_features", False)),
        history_limit=knobs.get("history_limit"),
    )


def write_reports(
    per_seed_results: dict[str, dict[str, dict]],
    out_dir: Path,
    args: argparse.Namespace,
    budgets: tuple[float, ...],
    base_rates: tuple[float, ...],
) -> None:
    aggregated = {
        variant: aggregate_model_metrics(list(seed_map.values()), budgets=budgets, base_rates=base_rates)
        for variant, seed_map in per_seed_results.items()
        if seed_map
    }
    report = {
        "benchmark_version": "phase1_host_aware_ablations_v1",
        "host_split_dir": str(args.host_split_dir),
        "settings": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "patience": int(args.patience),
            "seeds": parse_seeds(args.seeds),
            "default_fpr_budget": float(args.default_fpr_budget),
            "fpr_budgets": list(budgets),
            "base_rates": list(base_rates),
            "n_bootstrap": int(args.n_bootstrap),
            "variant_definitions": {name: VARIANTS[name] for name in per_seed_results},
        },
        "variants": aggregated,
        "per_seed_variants": per_seed_results,
    }
    (out_dir / "ablation_report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    (out_dir / "ablation_report.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    default_key = f"{float(report['settings']['default_fpr_budget']):.4f}"
    lines = [
        "# Phase 1 - Host-Aware Ablations (Multi-Seed)",
        "",
        f"Host split: `{report['host_split_dir']}`",
        f"Seeds: `{', '.join(str(s) for s in report['settings']['seeds'])}` "
        "(mean [95% CI] across completed seeds; report is rewritten incrementally)",
        "",
        "| Variant | Seeds done | Val AUC | Test AUC | Test PR-AUC | Test Recall@1.5%FPR | Adj PR-AUC pi=0.001 | pi=0.01 |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for variant, agg in report["variants"].items():
        val = agg["validation_metrics_by_budget"].get(default_key, {})
        test = agg["test_metrics_by_budget"].get(default_key, {})
        honest = agg.get("honest_test_metrics", {})
        adjusted = honest.get("prevalence_adjusted", {})
        lines.append(
            f"| {variant} | {agg['n_seeds']} | "
            f"{format_mean_ci(val.get('auc'))} | "
            f"{format_mean_ci(test.get('auc'))} | "
            f"{format_mean_ci(honest.get('pr_auc'))} | "
            f"{format_mean_ci(test.get('recall'))} | "
            f"{format_mean_ci(adjusted.get('0.001', {}).get('adjusted_pr_auc'))} | "
            f"{format_mean_ci(adjusted.get('0.01', {}).get('adjusted_pr_auc'))} |"
        )
    lines.extend(["", "## Per-seed test AUC", ""])
    for variant, agg in report["variants"].items():
        per_seed = agg.get("per_seed_test_auc", {})
        seed_text = ", ".join(f"{seed}: {auc:.4f}" for seed, auc in per_seed.items())
        lines.append(f"- {variant}: {seed_text}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
