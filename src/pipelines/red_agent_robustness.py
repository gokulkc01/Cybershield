"""Single-command red-agent robustness evaluation pipeline.

Usage:
    python -m src.pipelines.red_agent_robustness \
        --config path/to/red_agent_config.json \
        --output_dir experiments/red_agent_run_001

Config JSON schema:
{
    "checkpoint_path": "experiments/.../best_transformer.pth",
    "baseline_npz": "data/processed/.../sessions.npz",
    "mutation_plan": [
        {
            "mutation_type": "timing_jitter",
            "severity": 0.4,
            "params": {}
        },
        ...
    ],
    "base_seed": 42,
    "batch_size": 256,
    "device": "cpu"
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.red_agent import build_adversarial_dataset, run_mutation_evaluation


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate config schema and required fields."""
    required = ["checkpoint_path", "baseline_npz", "mutation_plan"]
    for key in required:
        if key not in config:
            raise ValueError(f"Config missing required field: {key}")

    if not isinstance(config["mutation_plan"], list) or len(config["mutation_plan"]) == 0:
        raise ValueError("mutation_plan must be non-empty list")

    for i, plan in enumerate(config["mutation_plan"]):
        if "mutation_type" not in plan:
            raise ValueError(f"mutation_plan[{i}] missing 'mutation_type'")
        if "severity" not in plan:
            raise ValueError(f"mutation_plan[{i}] missing 'severity'")


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate config JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    _validate_config(config)
    return config


def run_robustness_pipeline(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Execute full mutation-based robustness evaluation pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = config["checkpoint_path"]
    baseline_npz = config["baseline_npz"]
    mutation_plan = config["mutation_plan"]
    base_seed = config.get("base_seed", 42)
    batch_size = config.get("batch_size", 256)
    device = config.get("device", "cpu")

    # Verify checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not Path(baseline_npz).exists():
        raise FileNotFoundError(f"Baseline NPZ not found: {baseline_npz}")

    print("=" * 70)
    print("RED-AGENT V1: MUTATION-BASED ADVERSARIAL ROBUSTNESS EVALUATION")
    print("=" * 70)
    print(f"\n[CONFIG]")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Baseline NPZ: {baseline_npz}")
    print(f"  Output directory: {output_dir}")
    print(f"  Mutation plan: {len(mutation_plan)} operations")
    print(f"  Base seed: {base_seed}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")

    for i, plan in enumerate(mutation_plan):
        print(f"    [{i}] {plan['mutation_type']} (severity={plan['severity']})")

    # Step 1: Build adversarial dataset
    print(f"\n[STEP 1/2] Building adversarial dataset...")
    mutated_npz = str(output_path / "mutated_sessions.npz")
    metadata_jsonl = str(output_path / "mutation_metadata.jsonl")

    try:
        build_result = build_adversarial_dataset(
            input_npz=baseline_npz,
            output_npz=mutated_npz,
            output_metadata_jsonl=metadata_jsonl,
            mutation_plan=mutation_plan,
            base_seed=base_seed,
        )
        print(f"  ✓ Generated {build_result['n_samples']} mutated samples")
        print(f"  ✓ Metadata saved to {metadata_jsonl}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}", file=sys.stderr)
        raise

    # Step 2: Run evaluation
    print(f"\n[STEP 2/2] Running robustness evaluation...")
    output_json = str(output_path / "robustness_report.json")

    try:
        eval_output = run_mutation_evaluation(
            checkpoint_path=checkpoint_path,
            baseline_npz=baseline_npz,
            mutated_npz=mutated_npz,
            metadata_jsonl=metadata_jsonl,
            output_json=output_json,
            batch_size=batch_size,
            device=device,
        )
        print(f"  ✓ Evaluation complete")
        print(f"  ✓ Report saved to {output_json}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}", file=sys.stderr)
        raise

    # Summary
    print(f"\n[RESULTS]")
    print(f"  Baseline Recall: {eval_output.baseline_metrics['recall']:.4f}")
    print(f"  Mutated Recall: {eval_output.mutated_metrics['recall']:.4f}")
    print(f"  Recall Degradation: {eval_output.robustness_summary['recall_degradation']:.4f}")
    print(f"  FPR Shift: {eval_output.robustness_summary['fpr_shift']:.6f}")
    print(f"  Behavioral Invariance: {eval_output.robustness_summary['behavioral_invariance_stability']:.4f}")

    print(f"\n[FRAGILE MUTATIONS]")
    if eval_output.failure_analysis.get("fragile_mutation_types"):
        for mutation_type, count in sorted(
            eval_output.failure_analysis["fragile_mutation_types"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {mutation_type}: {count} undetected samples")
    else:
        print(f"  (No undetected samples)")

    print(f"\n[TOP FEATURE SHIFTS]")
    for feature_info in eval_output.failure_analysis.get("top_shifted_features", [])[:3]:
        print(f"  {feature_info['feature']}: mean shift={feature_info['mean_absolute_shift']:.6f}")

    # Write summary to stdout
    summary_path = output_path / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("RED-AGENT V1 ROBUSTNESS SUMMARY\n")
        handle.write("=" * 50 + "\n\n")
        handle.write(f"Baseline Recall: {eval_output.baseline_metrics['recall']:.4f}\n")
        handle.write(f"Mutated Recall: {eval_output.mutated_metrics['recall']:.4f}\n")
        handle.write(f"Recall Degradation: {eval_output.robustness_summary['recall_degradation']:.4f}\n")
        handle.write(f"FPR Shift: {eval_output.robustness_summary['fpr_shift']:.6f}\n")
        handle.write(f"Behavioral Invariance: {eval_output.robustness_summary['behavioral_invariance_stability']:.4f}\n\n")
        handle.write(f"Fragile Mutations:\n")
        for mutation_type, count in sorted(
            eval_output.failure_analysis.get("fragile_mutation_types", {}).items(), key=lambda x: x[1], reverse=True
        ):
            handle.write(f"  {mutation_type}: {count}\n")

    print(f"\n✓ Summary written to {summary_path}")
    print(f"\n[ARTIFACTS]")
    print(f"  Mutated NPZ: {mutated_npz}")
    print(f"  Metadata: {metadata_jsonl}")
    print(f"  Report JSON: {output_json}")
    print(f"  Summary: {summary_path}")

    return {
        "baseline_metrics": eval_output.baseline_metrics,
        "mutated_metrics": eval_output.mutated_metrics,
        "robustness_summary": eval_output.robustness_summary,
        "mutation_breakdown": eval_output.mutation_breakdown,
        "failure_analysis": eval_output.failure_analysis,
        "output_dir": str(output_path),
        "artifacts": {
            "mutated_npz": mutated_npz,
            "metadata_jsonl": metadata_jsonl,
            "report_json": output_json,
            "summary_txt": str(summary_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Red-Agent v1: Single-command mutation-based robustness evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipelines.red_agent_robustness \\
    --config red_agent_config.json \\
    --output_dir experiments/red_agent_run_001

  python -m src.pipelines.red_agent_robustness \\
    --config red_agent_config.json \\
    --output_dir experiments/red_agent_run_001 \\
    --verbose
""",
    )

    parser.add_argument("--config", required=True, help="Path to red_agent_config.json")
    parser.add_argument("--output_dir", required=True, help="Output directory for all artifacts")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    try:
        config = _load_config(args.config)
        result = run_robustness_pipeline(config, args.output_dir)
        print(f"\n✓ Pipeline execution successful")
        return 0
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
