"""Build an extended_v1 UWF dataset from raw UWF-ZeekData24 CSV files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.data_loader.normalized_telemetry import load_and_normalize_telemetry
from src.pipelines.build_extended_dataset import build_extended_dataset_from_dataframe


def _discover_uwf_files(raw_dir: str) -> dict[str, list[str]]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"UWF raw directory not found: {raw_dir}")

    groups = {
        "benign": sorted(str(path) for path in (root / "benign").glob("*.csv")),
        "defense_evasion": sorted(str(path) for path in (root / "defense_evasion").glob("*.csv")),
        "exfiltration": sorted(str(path) for path in (root / "exfiltration").glob("*.csv")),
        "persistence": sorted(str(path) for path in (root / "persistence").glob("*.csv")),
    }
    return groups


def _load_uwf_frame(args: Tuple[str, int, str]) -> pd.DataFrame:
    path, label, family = args
    return load_and_normalize_telemetry(
        path,
        source_dataset="uwf_zeekdata24",
        label=label,
        family=family,
        capture_id=Path(path).stem,
    )


def build_uwf_extended_dataset(
    raw_dir: str,
    out_dir: str,
    *,
    max_benign_files: int | None = None,
    max_c2_files: int | None = None,
) -> dict:
    groups = _discover_uwf_files(raw_dir)

    benign_files = groups["benign"]
    c2_files = groups["defense_evasion"] + groups["exfiltration"] + groups["persistence"]
    if max_benign_files is not None:
        benign_files = benign_files[:max_benign_files]
    if max_c2_files is not None:
        c2_files = c2_files[:max_c2_files]

    if not benign_files and not c2_files:
        raise ValueError("No UWF CSV files found to process.")

    tasks: list[Tuple[str, int, str]] = [(path, 0, "") for path in benign_files]
    tasks.extend((path, 1, Path(path).parent.name.lower()) for path in c2_files)

    frames: list[pd.DataFrame] = []
    max_workers = min(8, len(tasks)) if tasks else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_uwf_frame, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            frames.append(future.result())

    combined = pd.concat(frames, ignore_index=True)
    manifest = build_extended_dataset_from_dataframe(
        combined,
        out_dir,
        source_dataset="uwf_zeekdata24",
        min_flows=1,
        inactivity_timeout=300.0,
    )
    manifest["num_benign_files"] = len(benign_files)
    manifest["num_c2_files"] = len(c2_files)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UWF extended_v1 dataset from raw UWF CSV folders")
    parser.add_argument("--raw-dir", default="data/raw/uwf_zeekdata24", help="Root UWF raw directory")
    parser.add_argument("--out-dir", default="data/processed/uwf_extended", help="Output directory for extended dataset")
    parser.add_argument("--max-benign-files", type=int, default=None, help="Optional limit on benign files")
    parser.add_argument("--max-c2-files", type=int, default=None, help="Optional limit on C2 files")
    args = parser.parse_args()

    manifest = build_uwf_extended_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        max_benign_files=args.max_benign_files,
        max_c2_files=args.max_c2_files,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
