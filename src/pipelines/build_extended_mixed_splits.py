"""Build train/val/test splits from multiple extended_v1 NPZ sources.

Each source is specified as:
    <npz_path>:<source_name>:<label_type>[:<family>]

Examples:
    data/processed/ctu13_extended/sessions_extended.npz:ctu13:c2:ctu13
    data/processed/uwf_extended/sessions_extended.npz:uwf_zeekdata24:benign

This script enforces the extended_v1 feature schema (45 features).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.data_loader.npz_utils import load_session_npz
from src.data_loader.split_utils import SplitConfig, SplitStrategy, split_dataset
from src.features.feature_config_extended import (
    FEATURE_NAMES_EXTENDED,
    FEATURE_SCHEMA_VERSION,
    SESSION_LEN,
)
from src.features.feature_config_v2 import DatasetSource, LABEL_BENIGN, LABEL_C2, SessionMetadata


@dataclass(frozen=True)
class SourceSpec:
    path: str
    source: DatasetSource
    label_type: str
    family: str


def _parse_source_spec(spec: str) -> SourceSpec:
    parts = spec.rsplit(":", 3)
    if len(parts) == 3:
        npz_path, source_name, label_type = parts
        family = ""
    elif len(parts) == 4:
        npz_path, source_name, label_type, family = parts
    else:
        raise ValueError(
            f"Invalid source spec '{spec}'. Expected <npz_path>:<source_name>:<label_type>[:<family>]"
        )

    source_map = {item.value: item for item in DatasetSource}
    source = source_map.get(source_name.lower(), DatasetSource.UNKNOWN)
    label_type = label_type.strip().lower()
    if label_type not in {"c2", "benign"}:
        raise ValueError(f"Invalid label_type '{label_type}' in '{spec}'. Use c2 or benign.")

    return SourceSpec(
        path=npz_path,
        source=source,
        label_type=label_type,
        family=family.strip().lower(),
    )


def _save_npz(path: Path, sequences: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=sequences.astype(np.float32),
        y=labels.astype(np.int64),
        masks=masks.astype(bool),
        feature_names=np.array(FEATURE_NAMES_EXTENDED),
        schema_version=np.array(FEATURE_SCHEMA_VERSION),
    )


def build_extended_mixed_splits(
    source_specs: List[str],
    out_dir: str,
    split_strategy: str = "random_stratified",
    random_seed: int = 42,
) -> dict:
    specs = [_parse_source_spec(spec) for spec in source_specs]

    all_sequences: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    metadata: list[SessionMetadata] = []

    for spec in specs:
        npz_path = Path(spec.path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Source NPZ not found: {npz_path}")

        sequences, labels, masks = load_session_npz(
            str(npz_path),
            expected_feature_names=FEATURE_NAMES_EXTENDED,
            expected_session_len=SESSION_LEN,
        )

        if spec.label_type == "c2":
            labels = np.ones(len(sequences), dtype=np.int64)
        else:
            labels = np.zeros(len(sequences), dtype=np.int64)

        all_sequences.append(sequences)
        all_labels.append(labels)
        all_masks.append(masks)

        family = spec.family or ("unknown_c2" if spec.label_type == "c2" else "")
        capture_id = npz_path.parent.name or npz_path.stem
        for _ in range(len(sequences)):
            metadata.append(
                SessionMetadata(
                    source=spec.source,
                    c2_family=family if spec.label_type == "c2" else "",
                    capture_id=capture_id,
                )
            )

    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    strategy_map = {
        "random_stratified": SplitStrategy.RANDOM_STRATIFIED,
        "family_separated": SplitStrategy.FAMILY_SEPARATED,
        "source_separated": SplitStrategy.SOURCE_SEPARATED,
        "time_separated": SplitStrategy.TIME_SEPARATED,
        "zero_shot": SplitStrategy.ZERO_SHOT,
    }
    if split_strategy not in strategy_map:
        raise ValueError(f"Unsupported split strategy '{split_strategy}'")

    split_result = split_dataset(
        labels=labels,
        metadata=metadata,
        config=SplitConfig(strategy=strategy_map[split_strategy], random_seed=random_seed),
    )

    out_root = Path(out_dir)
    train_path = out_root / "train_sessions.npz"
    val_path = out_root / "val_sessions.npz"
    test_path = out_root / "test_sessions.npz"

    _save_npz(train_path, sequences[split_result.train_indices], labels[split_result.train_indices], masks[split_result.train_indices])
    _save_npz(val_path, sequences[split_result.val_indices], labels[split_result.val_indices], masks[split_result.val_indices])
    _save_npz(test_path, sequences[split_result.test_indices], labels[split_result.test_indices], masks[split_result.test_indices])

    summary = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": len(FEATURE_NAMES_EXTENDED),
        "session_len": SESSION_LEN,
        "split_strategy": split_strategy,
        "random_seed": random_seed,
        "total_samples": int(len(labels)),
        "total_c2": int((labels == LABEL_C2).sum()),
        "total_benign": int((labels == LABEL_BENIGN).sum()),
        "train_samples": int(len(split_result.train_indices)),
        "val_samples": int(len(split_result.val_indices)),
        "test_samples": int(len(split_result.test_indices)),
        "train_path": str(train_path).replace("\\", "/"),
        "val_path": str(val_path).replace("\\", "/"),
        "test_path": str(test_path).replace("\\", "/"),
    }

    summary_path = out_root / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test splits from extended_v1 NPZ sources")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source specs: <npz_path>:<source_name>:<label_type>[:<family>]",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/extended_mixed",
        help="Output directory for split NPZ files",
    )
    parser.add_argument(
        "--split_strategy",
        default="random_stratified",
        choices=["random_stratified", "family_separated", "source_separated", "time_separated", "zero_shot"],
        help="Split strategy",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    summary = build_extended_mixed_splits(
        source_specs=args.sources,
        out_dir=args.out_dir,
        split_strategy=args.split_strategy,
        random_seed=args.random_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
