"""Prepare train/val/test NPZ splits for the multifamily generalization experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.data_loader.family_splitter import FamilyAwareSplitter
from src.data_loader.npz_utils import load_session_npz
from src.data_loader.split_utils import SplitConfig, SplitStrategy, split_dataset
from src.features.feature_config_experiment import FEATURE_NAMES
from src.features.feature_config_v2 import DatasetSource, SessionMetadata


@dataclass(frozen=True)
class SourceSpec:
    path: str
    family: str
    label_type: str
    scenario: str


def _load_source_specs(path: str) -> List[SourceSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = []
    for item in payload.get("sources", []):
        sources.append(
            SourceSpec(
                path=item["path"],
                family=str(item.get("family", "")).strip().lower(),
                label_type=str(item.get("label_type", "c2")).strip().lower(),
                scenario=str(item.get("scenario", "")).strip(),
            )
        )
    if not sources:
        raise ValueError(f"No sources found in manifest: {path}")
    return sources


def _save_npz(path: Path, sessions: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=sessions.astype(np.float32),
        y=labels.astype(np.int64),
        masks=masks.astype(bool),
        feature_names=np.array(FEATURE_NAMES),
    )


def prepare_multifamily_splits(
    sources_manifest: str,
    out_dir: str,
    train_families: tuple[str, ...],
    test_families: tuple[str, ...],
    random_seed: int = 42,
    fallback_strategy: str = "none",
) -> dict:
    specs = _load_source_specs(sources_manifest)

    sessions_chunks = []
    labels_chunks = []
    masks_chunks = []
    metadata: list[SessionMetadata] = []

    for spec in specs:
        npz_path = Path(spec.path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Source file not found: {npz_path}")

        sessions, labels, masks = load_session_npz(
            str(npz_path),
            expected_feature_names=FEATURE_NAMES,
        )

        if spec.label_type == "c2":
            labels = np.ones(len(sessions), dtype=np.int64)
        elif spec.label_type == "benign":
            labels = np.zeros(len(sessions), dtype=np.int64)
        else:
            raise ValueError(f"Unsupported label_type '{spec.label_type}' in {sources_manifest}")

        sessions_chunks.append(sessions)
        labels_chunks.append(labels)
        masks_chunks.append(masks)

        for _ in range(len(sessions)):
            metadata.append(
                SessionMetadata(
                    source=DatasetSource.CTU13,
                    c2_family=spec.family if spec.label_type == "c2" else "",
                    capture_id=spec.scenario,
                )
            )

    sessions = np.concatenate(sessions_chunks, axis=0)
    labels = np.concatenate(labels_chunks, axis=0)
    masks = np.concatenate(masks_chunks, axis=0)

    observed_c2_families = sorted({m.c2_family for m in metadata if m.c2_family})
    if len(observed_c2_families) < 2 and fallback_strategy == "none":
        raise ValueError(
            "Family-separated split is not possible: fewer than 2 C2 families found in manifest. "
            f"Observed C2 families: {observed_c2_families}"
        )

    requested_train = tuple(f.strip().lower() for f in train_families if f.strip())
    requested_test = tuple(f.strip().lower() for f in test_families if f.strip())
    if requested_train and fallback_strategy == "none":
        missing = sorted(set(requested_train) - set(observed_c2_families))
        if missing:
            raise ValueError(f"Requested train families not found in data: {missing}")
    if requested_test and fallback_strategy == "none":
        missing = sorted(set(requested_test) - set(observed_c2_families))
        if missing:
            raise ValueError(f"Requested test families not found in data: {missing}")

    split_mode = "family_separated"
    if len(observed_c2_families) >= 2 and fallback_strategy == "none":
        splitter = FamilyAwareSplitter(
            train_families=requested_train or None,
            test_families=requested_test or None,
            random_seed=random_seed,
        )
        split_result, split_manifest = splitter.split(labels=labels, metadata=metadata)
    else:
        if fallback_strategy not in {"random_stratified", "source_separated"}:
            raise ValueError(
                f"Invalid fallback_strategy '{fallback_strategy}'. "
                "Use one of: none, random_stratified, source_separated"
            )
        strategy = (
            SplitStrategy.RANDOM_STRATIFIED
            if fallback_strategy == "random_stratified"
            else SplitStrategy.SOURCE_SEPARATED
        )
        split_mode = fallback_strategy
        split_result = split_dataset(
            labels=labels,
            metadata=metadata,
            config=SplitConfig(strategy=strategy, random_seed=random_seed),
        )
        from src.data_loader.family_splitter import FamilySplitManifest

        split_manifest = FamilySplitManifest(
            split_name=f"{fallback_strategy}_fallback_v1",
            random_seed=random_seed,
            train_families=sorted(split_result.train_families),
            test_families=sorted(split_result.test_families),
            train_indices=split_result.train_indices.tolist(),
            val_indices=split_result.val_indices.tolist(),
            test_indices=split_result.test_indices.tolist(),
            train_size=len(split_result.train_indices),
            val_size=len(split_result.val_indices),
            test_size=len(split_result.test_indices),
            train_c2=int(np.sum(labels[split_result.train_indices] == 1)),
            val_c2=int(np.sum(labels[split_result.val_indices] == 1)),
            test_c2=int(np.sum(labels[split_result.test_indices] == 1)),
        )

    out_root = Path(out_dir)
    train_path = out_root / "train_sessions.npz"
    val_path = out_root / "val_sessions.npz"
    test_path = out_root / "test_sessions.npz"
    _save_npz(train_path, sessions[split_result.train_indices], labels[split_result.train_indices], masks[split_result.train_indices])
    _save_npz(val_path, sessions[split_result.val_indices], labels[split_result.val_indices], masks[split_result.val_indices])
    _save_npz(test_path, sessions[split_result.test_indices], labels[split_result.test_indices], masks[split_result.test_indices])

    manifest_path = out_root / "family_split_manifest.json"
    split_manifest.save(manifest_path)

    summary = {
        "sources_manifest": str(sources_manifest),
        "split_mode": split_mode,
        "fallback_strategy": fallback_strategy,
        "observed_c2_families": observed_c2_families,
        "train_families": split_manifest.train_families,
        "test_families": split_manifest.test_families,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "split_manifest_path": str(manifest_path),
        "train_size": split_manifest.train_size,
        "val_size": split_manifest.val_size,
        "test_size": split_manifest.test_size,
    }

    summary_path = out_root / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["split_summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multifamily experiment train/val/test splits")
    parser.add_argument(
        "--sources_manifest",
        default="experiments/multifamily_generalization/ctu13_sources.json",
        help="Path to source manifest JSON",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/experiment_10f_splits",
        help="Output directory for split NPZ files",
    )
    parser.add_argument(
        "--train_families",
        default="neris,kraken",
        help="Comma-separated C2 families for training",
    )
    parser.add_argument(
        "--test_families",
        default="conficker",
        help="Comma-separated C2 families for held-out testing",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--fallback_strategy",
        default="none",
        choices=["none", "random_stratified", "source_separated"],
        help="Fallback split mode when strict family-separated split is not possible",
    )
    args = parser.parse_args()

    train_families = tuple(f.strip() for f in args.train_families.split(",") if f.strip())
    test_families = tuple(f.strip() for f in args.test_families.split(",") if f.strip())

    summary = prepare_multifamily_splits(
        sources_manifest=args.sources_manifest,
        out_dir=args.out_dir,
        train_families=train_families,
        test_families=test_families,
        random_seed=args.random_seed,
        fallback_strategy=args.fallback_strategy,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
