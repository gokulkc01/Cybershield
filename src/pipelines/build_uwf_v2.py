"""End-to-end UWF-ZeekData24 pipeline for CyberShield v2.

This runner wires the next steps together:
1. Process raw UWF telemetry into session NPZs and metadata sidecars.
2. Build a v2 family-separated dataset from the produced UWF sessions.

It is intentionally small and explicit so the project has a single
repeatable command path from raw data to training-ready splits.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from src.data_loader.uwf_processor import build_uwf_sessions
from src.features.dataset_builder_v2 import build_v2_dataset


def _build_source_specs(uwf_processed_dir: str) -> List[str]:
    """Return the UWF source specs produced by the processor."""
    c2_path = Path(uwf_processed_dir) / "uwf_c2_sessions.npz"
    benign_path = Path(uwf_processed_dir) / "uwf_benign_sessions.npz"

    specs: List[str] = []
    if c2_path.exists():
        specs.append(f"{c2_path.as_posix()}:uwf_zeekdata24:c2")
    if benign_path.exists():
        specs.append(f"{benign_path.as_posix()}:uwf_zeekdata24:benign")
    return specs


def run_uwf_pipeline(
    raw_dir: str,
    processed_dir: str,
    v2_out_dir: str,
    *,
    min_flows: int = 1,
    inactivity_timeout: float = 300.0,
    max_benign_files: int | None = None,
    max_c2_files: int | None = None,
    split_strategy: str = "family_separated",
) -> dict[str, str]:
    """Process UWF raw data and build the v2 dataset."""
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(v2_out_dir, exist_ok=True)

    print("[STEP 1/2] Processing raw UWF telemetry...")
    uwf_outputs = build_uwf_sessions(
        data_dir=raw_dir,
        output_dir=processed_dir,
        min_flows=min_flows,
        inactivity_timeout=inactivity_timeout,
        max_benign_files=max_benign_files,
        max_c2_files=max_c2_files,
    )

    source_specs = _build_source_specs(processed_dir)
    if not source_specs:
        raise ValueError(
            f"No UWF session NPZs found in {processed_dir}. "
            "Processor did not emit the expected outputs."
        )

    print("[STEP 2/2] Building the v2 split dataset...")
    v2_outputs = build_v2_dataset(
        source_specs=source_specs,
        out_dir=v2_out_dir,
        split_strategy=split_strategy,
    )

    outputs: dict[str, str] = {}
    outputs.update({f"uwf_{key}": value for key, value in uwf_outputs.items()})
    outputs.update({f"v2_{key}": value for key, value in v2_outputs.items()})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UWF -> v2 dataset pipeline")
    parser.add_argument("--raw_dir", required=True, help="Root directory containing raw UWF telemetry")
    parser.add_argument(
        "--processed_dir",
        default="data/processed/uwf",
        help="Directory for UWF session NPZs and metadata",
    )
    parser.add_argument(
        "--v2_out_dir",
        default="data/processed/v2_uwf",
        help="Directory for the v2 split dataset",
    )
    parser.add_argument("--min_flows", type=int, default=1)
    parser.add_argument("--inactivity_timeout", type=float, default=300.0)
    parser.add_argument("--max_benign_files", type=int, default=None)
    parser.add_argument("--max_c2_files", type=int, default=None)
    parser.add_argument(
        "--split_strategy",
        default="family_separated",
        choices=["random_stratified", "family_separated", "source_separated", "time_separated", "zero_shot"],
    )
    args = parser.parse_args()

    outputs = run_uwf_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        v2_out_dir=args.v2_out_dir,
        min_flows=args.min_flows,
        inactivity_timeout=args.inactivity_timeout,
        max_benign_files=args.max_benign_files,
        max_c2_files=args.max_c2_files,
        split_strategy=args.split_strategy,
    )

    print("\n[COMPLETE] Pipeline outputs:")
    for key, value in sorted(outputs.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
