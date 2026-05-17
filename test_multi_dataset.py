#!/usr/bin/env python3
"""
Test the Transformer model on multiple datasets and compare metrics.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.evaluation.evaluate_transformer import evaluate_checkpoint
from src.features.feature_config import FEATURE_NAMES

# Model checkpoint
MODEL_CHECKPOINT = "experiments/transformer/best_transformer.pth"

# Datasets to test
DATASETS = {
    "CTU-13 C2 Sessions": "data/processed/ctu13_c2_sessions.npz",
    "CTU-13 External Unseen": "data/processed/ctu13_external_unseen_sample.npz",
    "CTU-13 External 30K": "data/processed/ctu13_external_unseen_sample_30k.npz",
    "CTU-13 Scenario 1 (Neris)": "data/processed/ctu13_scenario1_neris.npz",
    "CTU-13 Scenario 2 (Kraken)": "data/processed/ctu13_scenario2_kraken.npz",
    "CTU-13 Scenario 9 (Conficker)": "data/processed/ctu13_scenario9_conficker.npz",
    "MCFP Multi": "data/processed/mcfp_multi_sessions.npz",
    "MCFP Multi + Benign": "data/processed/mcfp_multi_plus_benign_sessions.npz",
    "Stratosphere MCFP": "data/processed/stratosphere_mcfp_sessions.npz",
}


def _to_json_safe(value):
    """Recursively convert numpy/scalar values to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepare_compatible_npz(npz_path: str) -> tuple[str, str | None]:
    """Return an evaluation-ready NPZ path, adding missing port features when needed.

    Some legacy CTU scenario files use a 10-feature schema that omits src_port and
    dst_port. The evaluation pipeline expects 12 features, so we inject zero-filled
    placeholders for missing port columns and write a temporary compatible NPZ.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        keys = list(data.keys())
        seq_key = "X" if "X" in data else ("sessions" if "sessions" in data else None)
        if seq_key is None:
            return npz_path, None

        sequences = data[seq_key]
        saved_feature_names = None
        if "feature_names" in data:
            saved_feature_names = [str(name) for name in data["feature_names"].tolist()]

        expected = list(FEATURE_NAMES)
        expected_no_ports = [name for name in expected if name not in {"src_port", "dst_port"}]

        needs_compat = False
        if saved_feature_names is not None:
            needs_compat = saved_feature_names == expected_no_ports
        elif sequences.ndim == 3 and sequences.shape[2] == len(expected_no_ports):
            needs_compat = True

        if not needs_compat:
            return npz_path, None

        src_idx = expected.index("src_port")
        dst_idx = expected.index("dst_port")
        if dst_idx <= src_idx:
            raise ValueError("Invalid expected feature order for src_port/dst_port")

        left = sequences[:, :, :src_idx]
        right = sequences[:, :, src_idx:]
        zeros_src = np.zeros((sequences.shape[0], sequences.shape[1], 1), dtype=sequences.dtype)
        zeros_dst = np.zeros((sequences.shape[0], sequences.shape[1], 1), dtype=sequences.dtype)
        patched_sequences = np.concatenate([left, zeros_src, zeros_dst, right], axis=2)

        patched_arrays = {k: data[k] for k in keys}
        patched_arrays[seq_key] = patched_sequences.astype(np.float32)
        patched_arrays["feature_names"] = np.array(expected, dtype=object)

    temp_file = tempfile.NamedTemporaryFile(prefix="compat_", suffix=".npz", delete=False)
    temp_file.close()
    np.savez(temp_file.name, **patched_arrays)
    note = "Applied compatibility patch: inserted zero src_port/dst_port columns"
    return temp_file.name, note

def main():
    results = {}
    temp_paths = []
    
    print("=" * 100)
    print("CyberShield Transformer Model - Multi-Dataset Evaluation")
    print("=" * 100)
    print(f"\nModel: {MODEL_CHECKPOINT}")
    print(f"\nTesting on {len(DATASETS)} datasets...\n")
    
    for dataset_name, dataset_path in DATASETS.items():
        if not Path(dataset_path).exists():
            print(f"⚠️  SKIP: {dataset_name}")
            print(f"   Path not found: {dataset_path}\n")
            continue
        
        try:
            print(f"🔍 Testing: {dataset_name}")
            print(f"   Path: {dataset_path}")

            eval_path, compat_note = _prepare_compatible_npz(dataset_path)
            if compat_note is not None:
                temp_paths.append(eval_path)
                print(f"   ⚙️  {compat_note}")
            
            result = evaluate_checkpoint(
                checkpoint_path=MODEL_CHECKPOINT,
                npz_path=eval_path,
                batch_size=256,
                daily_flows=1_000_000,
            )
            
            results[dataset_name] = {
                "threshold": result.threshold,
                "recall": result.recall_at_budget,
                "fpr": result.fpr_at_budget,
                "f1": result.f1,
                "auc": result.auc,
                "tp": result.tp,
                "fp": result.fp,
                "tn": result.tn,
                "fn": result.fn,
                "budget_met": result.budget_met,
                "expected_alerts_per_day": result.expected_false_alerts_per_day,
            }
            
            print(f"   ✅ Recall: {result.recall_at_budget:.4f} (Target: ≥ 0.75)")
            print(f"   ✅ FPR: {result.fpr_at_budget:.4f}")
            print(f"   ✅ AUC: {result.auc:.4f}")
            print(f"   ✅ F1: {result.f1:.4f}")
            print(f"   ✅ TP={result.tp}, FP={result.fp}, TN={result.tn}, FN={result.fn}")
            print()
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}\n")
            results[dataset_name] = {"error": str(e)}
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Dataset':<40} {'Recall':<10} {'FPR':<10} {'AUC':<10} {'F1':<10} {'Status':<15}")
    print("-" * 100)
    
    for dataset_name, result in results.items():
        if "error" in result:
            print(f"{dataset_name:<40} {'ERROR':<10}")
        else:
            recall = result["recall"]
            fpr = result["fpr"]
            auc = result["auc"]
            f1 = result["f1"]
            status = "✅ GOOD" if recall >= 0.75 else ("🟡 PARTIAL" if recall >= 0.50 else "❌ POOR")
            
            print(f"{dataset_name:<40} {recall:<10.4f} {fpr:<10.4f} {auc:<10.4f} {f1:<10.4f} {status:<15}")
    
    # Save results to JSON
    output_file = "test_results_multi_dataset.json"
    with open(output_file, "w") as f:
        json.dump(_to_json_safe(results), f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("- Recall ≥ 0.75: Model generalizes well to this dataset")
    print("- Recall 0.50-0.75: Partial generalization, some families detected")
    print("- Recall < 0.50: Poor generalization, most C2 missed")
    print("- FPR: False positive rate (lower is better, target ≤ 0.015)")
    print("- AUC: Area under ROC curve (0.5=random, 1.0=perfect)")
    print("=" * 100)

    for temp_path in temp_paths:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            pass

if __name__ == "__main__":
    main()
