"""Build a mixed-source CyberShield v2 dataset with realistic C2 variety.

This runner combines UWF, CTU-13, and MCFP processed session NPZs into a
single v2 dataset and defaults to a zero-shot source split so evaluation can
exercise held-out CTU-13 / MCFP traffic instead of UWF-only behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.features.dataset_builder_v2 import build_v2_dataset


def _maybe_add_spec(specs: List[str], path: Path, source_name: str, label_type: str) -> None:
    if path.exists():
        specs.append(f"{path.as_posix()}:{source_name}:{label_type}")


def _build_source_specs(processed_root: str) -> List[str]:
    root = Path(processed_root)
    uwf_root = root / "uwf"

    specs: List[str] = []
    _maybe_add_spec(specs, uwf_root / "uwf_c2_sessions.npz", "uwf_zeekdata24", "c2")
    _maybe_add_spec(specs, uwf_root / "uwf_benign_sessions.npz", "uwf_zeekdata24", "benign")

    _maybe_add_spec(specs, root / "ctu13_c2_sessions.npz", "ctu13", "c2")
    _maybe_add_spec(specs, root / "ctu13_external_unseen_sample.npz", "ctu13", "c2")
    _maybe_add_spec(specs, root / "ctu13_external_unseen_sample_30k.npz", "ctu13", "c2")

    _maybe_add_spec(specs, root / "mcfp_multi_plus_benign_sessions.npz", "mcfp_stratosphere", "mixed")
    _maybe_add_spec(specs, root / "stratosphere_mcfp_sessions.npz", "mcfp_stratosphere", "mixed")
    _maybe_add_spec(specs, root / "mcfp_multi_sessions.npz", "mcfp_stratosphere", "c2")
    return specs


def run_mixed_pipeline(
    processed_root: str,
    out_dir: str,
    *,
    split_strategy: str = "zero_shot",
    held_out_sources: str = "ctu13,mcfp_stratosphere",
    build_timelines: bool = False,
) -> dict[str, str]:
    source_specs = _build_source_specs(processed_root)
    if not source_specs:
        raise ValueError(
            f"No processed session NPZs found under {processed_root}. "
            "Expected UWF, CTU-13, or MCFP session exports."
        )

    return build_v2_dataset(
        source_specs=source_specs,
        out_dir=out_dir,
        split_strategy=split_strategy,
        held_out_sources=held_out_sources,
        build_timelines=build_timelines,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed-source v2 dataset")
    parser.add_argument(
        "--processed_root",
        default="data/processed",
        help="Root directory containing processed UWF / CTU-13 / MCFP NPZs",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/v2_mixed",
        help="Directory for the mixed-source v2 split dataset",
    )
    parser.add_argument(
        "--split_strategy",
        default="zero_shot",
        choices=["random_stratified", "family_separated", "source_separated", "time_separated", "zero_shot"],
        help="Dataset split strategy",
    )
    parser.add_argument(
        "--held_out_sources",
        default="ctu13,mcfp_stratosphere",
        help="Comma-separated dataset sources to hold out for evaluation",
    )
    parser.add_argument(
        "--with_timelines",
        action="store_true",
        help="Also build host timelines (default: session-only for memory safety)",
    )
    args = parser.parse_args()

    outputs = run_mixed_pipeline(
        processed_root=args.processed_root,
        out_dir=args.out_dir,
        split_strategy=args.split_strategy,
        held_out_sources=args.held_out_sources,
        build_timelines=args.with_timelines,
    )

    print("\n[COMPLETE] Mixed-source pipeline outputs:")
    for key, value in sorted(outputs.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()