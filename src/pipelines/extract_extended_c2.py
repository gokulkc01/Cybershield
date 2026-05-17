"""Extract a labeled subset from an extended_v1 sessions NPZ.

Usage:
    python -m src.pipelines.extract_extended_c2 --npz data/processed/ctu13_extended/sessions_extended.npz --out_dir data/processed/ctu13_extended_c2 --label c2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def extract_subset(npz_path: str, out_dir: str, label: str = "c2") -> dict:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(str(p), allow_pickle=True)
    X = data["X"]
    y = data["y"]
    masks = data["masks"]
    feature_names = data.get("feature_names")
    schema_version = data.get("schema_version")

    label = label.strip().lower()
    if label not in {"c2", "benign"}:
        raise ValueError("label must be either 'c2' or 'benign'")

    target_value = 1 if label == "c2" else 0
    subset_idx = (y == target_value)
    if not subset_idx.any():
        raise ValueError(f"No {label.upper()} samples found in input NPZ")

    out_path = Path(out_dir) / "sessions_extended.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X[subset_idx].astype(np.float32),
        y=y[subset_idx].astype(np.int64),
        masks=masks[subset_idx].astype(bool),
        feature_names=feature_names,
        schema_version=schema_version,
    )

    return {
        "npz_path": str(out_path).replace("\\", "/"),
        "num_sessions": int(subset_idx.sum()),
        "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a labeled subset from extended NPZ")
    parser.add_argument("--npz", required=True, help="Input extended NPZ path")
    parser.add_argument("--out_dir", required=True, help="Output directory for C2-only NPZ")
    parser.add_argument("--label", default="c2", choices=["c2", "benign"], help="Label subset to extract")
    args = parser.parse_args()

    summary = extract_subset(args.npz, args.out_dir, label=args.label)
    print(summary)


if __name__ == "__main__":
    main()
