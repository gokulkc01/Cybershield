"""Convert session NPZ files from the baseline 12-feature schema to the 10-feature experiment schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.features.feature_config_experiment import FEATURE_NAMES as EXP_FEATURE_NAMES
from src.features.feature_config import FEATURE_NAMES as BASE_FEATURE_NAMES


def _read_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(npz_path, allow_pickle=True)
    if "X" in data:
        sequences = data["X"].astype(np.float32)
    else:
        sequences = data["sessions"].astype(np.float32)

    if "y" in data:
        labels = data["y"].astype(np.int64)
    else:
        labels = data["labels"].astype(np.int64)

    masks = data["masks"].astype(bool) if "masks" in data else np.any(sequences != 0.0, axis=2)
    feature_names = [str(name) for name in data["feature_names"].tolist()] if "feature_names" in data else []
    return sequences, labels, masks, feature_names


def convert_npz_to_experiment_schema(input_npz: str, output_npz: str) -> str:
    input_path = Path(input_npz)
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sequences, labels, masks, source_feature_names = _read_npz(input_path)
    if not source_feature_names:
        source_feature_names = list(BASE_FEATURE_NAMES)

    source_index = {name: idx for idx, name in enumerate(source_feature_names)}
    missing = [name for name in EXP_FEATURE_NAMES if name not in source_index]
    if missing:
        raise ValueError(f"Input NPZ does not contain required experiment features {missing}: {input_path}")

    keep_indices = [source_index[name] for name in EXP_FEATURE_NAMES]
    converted = sequences[:, :, keep_indices].astype(np.float32)

    np.savez_compressed(
        output_path,
        X=converted,
        y=labels,
        masks=masks,
        feature_names=np.array(EXP_FEATURE_NAMES),
    )
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert baseline NPZ files to 10-feature experiment schema")
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_npz", required=True)
    args = parser.parse_args()

    output_path = convert_npz_to_experiment_schema(args.input_npz, args.output_npz)
    print(f"[COMPLETE] Wrote experiment-schema NPZ: {output_path}")


if __name__ == "__main__":
    main()
