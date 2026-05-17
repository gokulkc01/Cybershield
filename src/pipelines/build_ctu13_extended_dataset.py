"""Build an extended_v1 CTU-13 dataset directly from raw binetflow.

This wrapper reuses the existing CTU-13 standardization and labeling logic,
then hands the labeled DataFrame to the extended session builder so CTU-13
can be converted into the 45-feature schema without Zeek logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_loader.ctu13_processor import apply_ctu13_labels, load_and_standardize_ctu13
from src.pipelines.build_extended_dataset import build_extended_dataset_from_dataframe


def build_ctu13_extended_dataset(
    input_path: str,
    out_dir: str,
    *,
    source_dataset: str = "ctu13",
    min_flows: int = 1,
    inactivity_timeout: float = 300.0,
    max_rows: int | None = None,
) -> dict:
    """Convert a raw CTU-13 binetflow file into extended_v1 session tensors."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"CTU-13 input file not found: {input_file}")

    print(f"[1/3] Loading CTU-13 binetflow: {input_file}")
    df = load_and_standardize_ctu13(str(input_file))

    print("[2/3] Applying CTU-13 labels")
    df = apply_ctu13_labels(df)

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        print(f"[2.5/3] Truncating to first {max_rows:,} rows for smoke validation")
        df = df.head(max_rows).copy()

    print("[3/3] Building extended_v1 sessions")
    manifest = build_extended_dataset_from_dataframe(
        df,
        out_dir,
        source_dataset=source_dataset,
        min_flows=min_flows,
        inactivity_timeout=inactivity_timeout,
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CTU-13 extended_v1 dataset from raw binetflow")
    parser.add_argument(
        "--input",
        default="data/raw/ctu13/capture20110810.binetflow",
        help="Path to a raw CTU-13 .binetflow file",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/ctu13_extended",
        help="Output directory for sessions_extended.npz and dataset_manifest.json",
    )
    parser.add_argument(
        "--source-dataset",
        default="ctu13",
        help="Source dataset label to store in the manifest",
    )
    parser.add_argument(
        "--min-flows",
        type=int,
        default=1,
        help="Minimum flows per session",
    )
    parser.add_argument(
        "--inactivity-timeout",
        type=float,
        default=300.0,
        help="Seconds of inactivity before starting a new session",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke validation",
    )
    args = parser.parse_args()

    manifest = build_ctu13_extended_dataset(
        args.input,
        args.out_dir,
        source_dataset=args.source_dataset,
        min_flows=args.min_flows,
        inactivity_timeout=args.inactivity_timeout,
        max_rows=args.max_rows,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
