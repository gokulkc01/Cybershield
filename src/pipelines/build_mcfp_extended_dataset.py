"""Build an extended_v1 MCFP dataset from raw Stratosphere .binetflow files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader.stratosphere_mcfp_processor import _load_flow_dataframe
from src.pipelines.build_extended_dataset import build_extended_dataset_from_dataframe


def _discover_inputs(raw_dir: str) -> tuple[list[str], list[str]]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"MCFP raw directory not found: {raw_dir}")

    c2_files: list[str] = []
    benign_files: list[str] = []
    for path in sorted(root.glob("*.binetflow")):
        name = path.name.lower()
        if "normal" in name or "benign" in name:
            benign_files.append(str(path))
        else:
            c2_files.append(str(path))
    return c2_files, benign_files


def _load_labeled_frames(paths: list[str], label: int, source_dataset: str, family: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = _load_flow_dataframe(path)
        df = df.copy()
        if "timestamp" not in df.columns and "ts" in df.columns:
            df["timestamp"] = pd.to_numeric(df["ts"], errors="coerce")
        df["label"] = int(label)
        df["family"] = family if label == 1 else ""
        df["source_dataset"] = source_dataset
        df["capture_id"] = Path(path).stem
        df["source_path"] = path
        df["host_id"] = df["src_ip"].astype(str)
        if "timestamp" in df.columns:
            df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)
        else:
            df["ts"] = pd.to_numeric(df.get("ts", 0.0), errors="coerce").fillna(0.0)
        frames.append(df.replace([np.inf, -np.inf], 0.0).fillna(0.0))
    return frames


def build_mcfp_extended_dataset(
    raw_dir: str,
    out_dir: str,
    *,
    include_benign: bool = True,
    max_c2_files: int | None = None,
    max_benign_files: int | None = None,
) -> dict:
    c2_files, benign_files = _discover_inputs(raw_dir)
    if max_c2_files is not None:
        c2_files = c2_files[:max_c2_files]
    if max_benign_files is not None:
        benign_files = benign_files[:max_benign_files]

    if not c2_files and not (include_benign and benign_files):
        raise ValueError("No MCFP input files selected. Check raw_dir and limits.")

    frames: list[pd.DataFrame] = []
    frames.extend(_load_labeled_frames(c2_files, label=1, source_dataset="mcfp_stratosphere", family="mcfp"))
    if include_benign:
        frames.extend(_load_labeled_frames(benign_files, label=0, source_dataset="mcfp_stratosphere", family=""))

    combined = pd.concat(frames, ignore_index=True)
    manifest = build_extended_dataset_from_dataframe(
        combined,
        out_dir,
        source_dataset="mcfp_stratosphere",
        min_flows=1,
        inactivity_timeout=300.0,
    )
    manifest["num_c2_files"] = len(c2_files)
    manifest["num_benign_files"] = len(benign_files) if include_benign else 0
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MCFP extended_v1 dataset from raw .binetflow files")
    parser.add_argument("--raw-dir", default="data/raw/mcfp", help="Directory containing MCFP .binetflow files")
    parser.add_argument("--out-dir", default="data/processed/mcfp_extended", help="Output directory for extended dataset")
    parser.add_argument("--no-benign", action="store_true", help="Exclude benign/normal .binetflow files")
    parser.add_argument("--max-c2-files", type=int, default=None, help="Optional limit on number of C2 files")
    parser.add_argument("--max-benign-files", type=int, default=None, help="Optional limit on number of benign files")
    args = parser.parse_args()

    manifest = build_mcfp_extended_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        include_benign=not args.no_benign,
        max_c2_files=args.max_c2_files,
        max_benign_files=args.max_benign_files,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
