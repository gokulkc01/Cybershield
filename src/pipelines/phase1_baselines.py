"""Phase 1 reference baselines on the honest same-provenance split.

Runs the cheap, non-transformer baselines with the exact Phase 0 harness
conventions (multi-seed, thresholds frozen on val per FPR budget, honest
test report with bootstrap CIs and base-rate-adjusted metrics):

- ``resp_bytes_single_feature``: masked mean of raw ``resp_bytes`` per
  session, direction (sign) fitted on train. Mandatory floor - Phase 0
  showed a single byte-count feature reaches ~0.97 test AUC on this split
  (low response volume indicates C2 here).
- ``logistic_probe``: logistic regression on mean-pooled (transformed +
  normalized) session features. Mandatory floor - the linear signal ceiling.
- ``random_forest``: RandomForest on masked mean/std/min/max pooled session
  features (the classic tabular reference).
- ``beaconing_detector``: the non-learned Fourier/autocorrelation timing
  detector (``src.models.beaconing_detector``), scored per (host, dest)
  pair and broadcast to sessions, in the classic direction (periodic = C2).
- ``beaconing_detector_oriented``: same scores with the sign fitted on
  train - the honest ceiling of the timing-only signal on this data.

The deterministic detectors produce identical scores for every seed; their
seed CIs collapse to a point and uncertainty comes from the per-seed
bootstrap CIs instead. Aggregated reference rows from the Phase 0
transformer benchmark are appended to the markdown table when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_loader.host_window_dataset import load_host_windows_npz
from src.evaluation.multifamily_metrics import honest_test_report
from src.models.beaconing_detector import BeaconingConfig, BeaconingDetector
from src.pipelines.benchmark_host_aware_models import (
    aggregate_model_metrics,
    budget_metrics,
    convert_host_split_to_session_npzs,
    format_mean_ci,
    json_safe,
    load_preprocessed_session_splits,
    parse_budgets,
    parse_seeds,
)

METHOD_ORDER = (
    "resp_bytes_single_feature",
    "logistic_probe",
    "random_forest",
    "beaconing_detector",
    "beaconing_detector_oriented",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 baselines on the honest split")
    parser.add_argument("--host_split_dir", default="data/processed/host_aware_ctu13_same_provenance")
    parser.add_argument("--out_dir", default="experiments/phase1_baselines_ctu13_same_provenance")
    parser.add_argument(
        "--session_npz_dir",
        default="experiments/host_aware_benchmark_ctu13_same_provenance/baseline_session_npz",
        help="Existing converted session NPZs; converted fresh into out_dir when absent.",
    )
    parser.add_argument(
        "--phase0_report",
        default="experiments/host_aware_benchmark_ctu13_same_provenance/comparison_report.json",
        help="Phase 0 multi-seed benchmark report; its models are appended as reference rows.",
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--default_fpr_budget", type=float, default=0.015)
    parser.add_argument("--fpr_budgets", default="0.005,0.015,0.03")
    parser.add_argument("--base_rates", default="0.001,0.01")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--rf_trees", type=int, default=200)
    parser.add_argument("--rf_max_depth", type=int, default=20)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=5)
    parser.add_argument("--beacon_min_events", type=int, default=6)
    args = parser.parse_args()

    budgets = parse_budgets(args.fpr_budgets)
    base_rates = parse_budgets(args.base_rates)
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    host_split_dir = Path(args.host_split_dir)

    session_npz_dir = Path(args.session_npz_dir)
    if session_npz_dir.exists():
        session_npzs = {name: str(session_npz_dir / f"{name}_sessions.npz") for name in ("train", "val", "test")}
    else:
        print(f"[INFO] {session_npz_dir} missing; converting host split to session NPZs...")
        session_npzs = convert_host_split_to_session_npzs(host_split_dir, out_dir / "baseline_session_npz")

    print("[INFO] Loading transformed + normalized session splits...")
    session_data = load_preprocessed_session_splits(session_npzs, normalize_features=True)

    print("[INFO] Loading raw host-window arrays (for raw-feature detectors)...")
    raw = {name: load_host_windows_npz(host_split_dir / f"{name}_host_windows.npz") for name in ("train", "val", "test")}

    print("[INFO] Pooling session features...")
    pooled = {
        "train": masked_pool(session_data.train_x, session_data.train_masks),
        "val": masked_pool(session_data.val_x, session_data.val_masks),
        "test": masked_pool(session_data.test_x, session_data.test_masks),
    }
    mean_pooled = {name: arr[:, : session_data.feature_dim] for name, arr in pooled.items()}

    resp_bytes_scores = {
        name: raw_feature_mean_scores(raw[name], feature="resp_bytes") for name in ("train", "val", "test")
    }
    resp_bytes_sign = orientation_sign(resp_bytes_scores["train"], raw["train"].labels)
    resp_bytes_scores = {name: resp_bytes_sign * scores for name, scores in resp_bytes_scores.items()}
    resp_bytes_extras = {"orientation_sign": resp_bytes_sign, "orientation_fit_on": "train"}

    beacon_config = BeaconingConfig(min_events=args.beacon_min_events)
    beacon_scores = {name: beaconing_scores(raw[name], beacon_config) for name in ("train", "val", "test")}
    beacon_sign = orientation_sign(beacon_scores["train"], raw["train"].labels)
    beacon_oriented_scores = {name: beacon_sign * scores for name, scores in beacon_scores.items()}
    beacon_extras = {
        "config": {"min_events": beacon_config.min_events, "max_lag": beacon_config.max_lag},
        "coverage": {
            name: beaconing_coverage(raw[name], beacon_scores[name]) for name in ("val", "test")
        },
    }
    beacon_oriented_extras = dict(beacon_extras, orientation_sign=beacon_sign, orientation_fit_on="train")

    val_y = session_data.val_y
    test_y = session_data.test_y

    per_seed_results: dict[str, dict[str, dict]] = {name: {} for name in METHOD_ORDER}
    for seed in seeds:
        print(f"\n[INFO] ===== Seed {seed} =====")
        rng_probe = LogisticRegression(max_iter=2000, random_state=seed)
        rng_probe.fit(mean_pooled["train"], session_data.train_y)
        probe_scores = {
            name: rng_probe.predict_proba(mean_pooled[name])[:, 1] for name in ("val", "test")
        }

        rf = RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        rf.fit(pooled["train"], session_data.train_y)
        rf_scores = {name: rf.predict_proba(pooled[name])[:, 1] for name in ("val", "test")}
        rf_extras = {"top_features": rf_top_features(rf, session_data.feature_names)}

        method_scores = {
            "resp_bytes_single_feature": resp_bytes_scores,
            "logistic_probe": probe_scores,
            "random_forest": rf_scores,
            "beaconing_detector": beacon_scores,
            "beaconing_detector_oriented": beacon_oriented_scores,
        }
        method_extras = {
            "resp_bytes_single_feature": resp_bytes_extras,
            "random_forest": rf_extras,
            "beaconing_detector": beacon_extras,
            "beaconing_detector_oriented": beacon_oriented_extras,
        }

        for method in METHOD_ORDER:
            scores = method_scores[method]
            result = evaluate_method(
                method,
                val_y=val_y,
                val_scores=np.asarray(scores["val"], dtype=np.float64),
                test_y=test_y,
                test_scores=np.asarray(scores["test"], dtype=np.float64),
                budgets=budgets,
                default_fpr_budget=args.default_fpr_budget,
                base_rates=base_rates,
                n_bootstrap=args.n_bootstrap,
                seed=seed,
                extras=method_extras.get(method),
            )
            per_seed_results[method][str(seed)] = result
            np.savez_compressed(
                out_dir / "scores" / f"{method}_seed{seed}.npz",
                val_scores=np.asarray(scores["val"], dtype=np.float64),
                val_y=val_y.astype(np.int64),
                test_scores=np.asarray(scores["test"], dtype=np.float64),
                test_y=test_y.astype(np.int64),
            )
            auc = result["test_metrics_by_budget"][f"{args.default_fpr_budget:.4f}"]["auc"]
            print(f"[INFO] {method}: test AUC = {auc:.4f}")

    aggregated = {
        method: aggregate_model_metrics(list(per_seed_results[method].values()), budgets=budgets, base_rates=base_rates)
        for method in METHOD_ORDER
    }
    reference_models = load_phase0_reference(Path(args.phase0_report))

    report = {
        "benchmark_version": "phase1_baselines_v1",
        "host_split_dir": str(host_split_dir),
        "out_dir": str(out_dir),
        "settings": {
            "seeds": list(seeds),
            "default_fpr_budget": float(args.default_fpr_budget),
            "fpr_budgets": list(budgets),
            "base_rates": list(base_rates),
            "n_bootstrap": int(args.n_bootstrap),
            "rf": {
                "n_estimators": args.rf_trees,
                "max_depth": args.rf_max_depth,
                "min_samples_leaf": args.rf_min_samples_leaf,
            },
            "beaconing": beacon_extras["config"],
        },
        "models": aggregated,
        "phase0_reference_models": reference_models,
        "per_seed_models": per_seed_results,
    }
    (out_dir / "comparison_report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    (out_dir / "comparison_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\n[SUCCESS] Wrote {out_dir / 'comparison_report.json'}")
    print(f"[SUCCESS] Wrote {out_dir / 'comparison_report.md'}")


def masked_pool(sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Mask-aware mean/std/min/max pooling to (N, D*4); mean block first."""
    n, _, d = sequences.shape
    out = np.zeros((n, d * 4), dtype=np.float32)
    for i in range(n):
        real = sequences[i][masks[i].astype(bool)]
        if len(real) == 0:
            real = np.zeros((1, d), dtype=np.float32)
        out[i] = np.concatenate([real.mean(0), real.std(0), real.min(0), real.max(0)])
    return out


def orientation_sign(train_scores: np.ndarray, train_labels: np.ndarray) -> float:
    """+1 if higher scores indicate C2 on the training split, else -1.

    Single-feature and unsupervised detectors have no inherent direction;
    fitting the sign on train keeps val/test untouched.
    """
    from sklearn.metrics import roc_auc_score

    return 1.0 if roc_auc_score(train_labels, train_scores) >= 0.5 else -1.0


def raw_feature_mean_scores(arrays, *, feature: str) -> np.ndarray:
    feature_names = [str(name) for name in arrays.feature_names]
    idx = feature_names.index(feature)
    scores = np.zeros(len(arrays.labels), dtype=np.float64)
    for i in range(len(arrays.labels)):
        valid = arrays.current_masks[i].astype(bool)
        if valid.any():
            scores[i] = float(arrays.current_sessions[i][valid, idx].mean())
    return scores


def beaconing_scores(arrays, config: BeaconingConfig) -> np.ndarray:
    detector = BeaconingDetector(config)
    return detector.score_sessions(
        sessions=arrays.current_sessions,
        masks=arrays.current_masks,
        host_ids=arrays.host_ids,
        dest_ids=arrays.dest_ids,
        timestamps=arrays.timestamps,
        feature_names=arrays.feature_names,
    )


def beaconing_coverage(arrays, scores: np.ndarray) -> dict:
    labels = arrays.labels
    covered = scores > 0.0
    return {
        "sessions": int(len(labels)),
        "covered_fraction": float(covered.mean()),
        "covered_fraction_c2": float(covered[labels == 1].mean()) if (labels == 1).any() else None,
        "covered_fraction_benign": float(covered[labels == 0].mean()) if (labels == 0).any() else None,
    }


def rf_top_features(rf: RandomForestClassifier, feature_names: tuple[str, ...], top_k: int = 10) -> list[dict]:
    stats = ("mean", "std", "min", "max")
    flat_names = [f"{name}_{stat}" for stat in stats for name in feature_names]
    order = np.argsort(rf.feature_importances_)[::-1][:top_k]
    return [
        {"feature": flat_names[i], "importance": float(rf.feature_importances_[i])}
        for i in order
    ]


def evaluate_method(
    method: str,
    *,
    val_y: np.ndarray,
    val_scores: np.ndarray,
    test_y: np.ndarray,
    test_scores: np.ndarray,
    budgets: tuple[float, ...],
    default_fpr_budget: float,
    base_rates: tuple[float, ...],
    n_bootstrap: int,
    seed: int,
    extras: dict | None = None,
) -> dict:
    val_metrics_by_budget = budget_metrics(val_y, val_scores, budgets)
    fixed_thresholds = {key: metrics["threshold"] for key, metrics in val_metrics_by_budget.items()}
    test_metrics_by_budget = budget_metrics(test_y, test_scores, budgets, fixed_thresholds=fixed_thresholds)
    default_key = f"{default_fpr_budget:.4f}"
    result = {
        "model_name": method,
        "seed": int(seed),
        "validation_metrics_by_budget": val_metrics_by_budget,
        "test_metrics_by_budget": test_metrics_by_budget,
        "honest_test_report": honest_test_report(
            test_y,
            test_scores,
            threshold=float(fixed_thresholds[default_key]),
            fpr_budgets=budgets,
            base_rates=base_rates,
            n_bootstrap=n_bootstrap,
            random_seed=seed,
        ),
    }
    if extras:
        result["extras"] = extras
    return result


def load_phase0_reference(report_path: Path) -> dict:
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("models", {})


def render_markdown(report: dict) -> str:
    default_key = f"{float(report['settings']['default_fpr_budget']):.4f}"
    lines = [
        "# Phase 1 Baselines (Multi-Seed, Honest Split)",
        "",
        f"Host split: `{report['host_split_dir']}`",
        f"Seeds: `{', '.join(str(seed) for seed in report['settings']['seeds'])}` "
        "(mean [95% CI]; deterministic methods have zero seed variance by construction)",
        "",
        f"## Test metrics at default budget ({default_key}) + honest view",
        "",
        "| Model | Test AUC | Test PR-AUC | Recall@1.5%FPR (frozen thr) | Adj PR-AUC pi=0.001 | pi=0.01 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def model_row(name: str, agg: dict) -> str:
        test = agg["test_metrics_by_budget"].get(default_key, {})
        honest = agg.get("honest_test_metrics", {})
        adjusted = honest.get("prevalence_adjusted", {})
        return (
            f"| {name} | {format_mean_ci(test.get('auc'))} | {format_mean_ci(honest.get('pr_auc'))} | "
            f"{format_mean_ci(test.get('recall'))} | "
            f"{format_mean_ci(adjusted.get('0.001', {}).get('adjusted_pr_auc'))} | "
            f"{format_mean_ci(adjusted.get('0.01', {}).get('adjusted_pr_auc'))} |"
        )

    for name, agg in report["models"].items():
        lines.append(model_row(name, agg))
    for name, agg in report.get("phase0_reference_models", {}).items():
        lines.append(model_row(f"{name} (Phase 0 ref)", agg))

    lines.extend(["", "## Validation AUC (same-family axis)", ""])
    for name, agg in report["models"].items():
        val = agg["validation_metrics_by_budget"].get(default_key, {})
        lines.append(f"- {name}: {format_mean_ci(val.get('auc'))}")

    beacon = report["per_seed_models"].get("beaconing_detector", {})
    first = next(iter(beacon.values()), {})
    coverage = first.get("extras", {}).get("coverage")
    if coverage:
        lines.extend(["", "## Beaconing detector coverage (pairs with enough events)", ""])
        for split_name, stats in coverage.items():
            lines.append(
                f"- {split_name}: {stats['covered_fraction']:.1%} of sessions covered "
                f"(C2 {stats['covered_fraction_c2']:.1%}, benign {stats['covered_fraction_benign']:.1%})"
            )

    lines.extend(["", "## Per-seed test AUC", ""])
    for name, agg in report["models"].items():
        per_seed = agg.get("per_seed_test_auc", {})
        lines.append(f"- {name}: " + ", ".join(f"{seed}: {auc:.4f}" for seed, auc in per_seed.items()))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
