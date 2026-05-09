"""Source-aware evaluation split utilities for CyberShield v2.

Replaces the v1 split_utils.py with strategies that prevent data leakage
across C2 families, dataset sources, and time periods.

Split Strategies:
    1. Family-separated:  No C2 family in both train and test.
    2. Source-separated:   No dataset source in both train and test.
    3. Time-separated:     Temporal ordering within each dataset.
    4. Zero-shot:          Completely held-out C2 families + diverse benign.

All strategies maintain the constraint that benign traffic can (and should)
appear in both train and test — what matters is that C2 families are separated.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.features.feature_config_v2 import (
    DatasetSource,
    LABEL_BENIGN,
    LABEL_C2,
    SessionMetadata,
    VAL_SPLIT,
    TEST_SPLIT,
)


class SplitStrategy(str, Enum):
    """Available split strategies."""
    RANDOM_STRATIFIED = "random_stratified"     # v1 baseline (stratified random)
    FAMILY_SEPARATED = "family_separated"       # No C2 family in train+test
    SOURCE_SEPARATED = "source_separated"       # No dataset source in train+test
    TIME_SEPARATED = "time_separated"           # Temporal ordering
    ZERO_SHOT = "zero_shot"                     # Held-out families + sources


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    strategy: SplitStrategy = SplitStrategy.FAMILY_SEPARATED
    val_fraction: float = VAL_SPLIT
    test_fraction: float = TEST_SPLIT
    random_seed: int = 42

    # Family-separated options
    held_out_families: Optional[Set[str]] = None     # Explicit families for test
    min_families_per_split: int = 1                   # Min C2 families per split

    # Zero-shot options
    zero_shot_families: Optional[Set[str]] = None     # Families never seen in train
    zero_shot_sources: Optional[Set[str]] = None      # Sources never seen in train

    # Time-separated options
    time_split_quantile: float = 0.70  # Train uses first 70% of time


@dataclass
class SplitResult:
    """Result of a dataset split, with full provenance tracking."""
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    strategy: SplitStrategy
    config: SplitConfig

    # Diagnostic info
    train_families: Set[str] = field(default_factory=set)
    val_families: Set[str] = field(default_factory=set)
    test_families: Set[str] = field(default_factory=set)
    train_sources: Set[str] = field(default_factory=set)
    test_sources: Set[str] = field(default_factory=set)
    leakage_warnings: List[str] = field(default_factory=list)

    def validate_no_leakage(self) -> bool:
        """Check for index leakage between splits."""
        train_set = set(self.train_indices.tolist())
        val_set = set(self.val_indices.tolist())
        test_set = set(self.test_indices.tolist())

        ok = True
        if train_set & val_set:
            self.leakage_warnings.append("CRITICAL: Train/val index overlap!")
            ok = False
        if train_set & test_set:
            self.leakage_warnings.append("CRITICAL: Train/test index overlap!")
            ok = False
        if val_set & test_set:
            self.leakage_warnings.append("CRITICAL: Val/test index overlap!")
            ok = False

        return ok

    def summary(self) -> Dict[str, Any]:
        """Summary statistics for the split."""
        total = len(self.train_indices) + len(self.val_indices) + len(self.test_indices)
        return {
            "strategy": self.strategy.value,
            "total_samples": total,
            "train_size": len(self.train_indices),
            "val_size": len(self.val_indices),
            "test_size": len(self.test_indices),
            "train_families": sorted(self.train_families),
            "val_families": sorted(self.val_families),
            "test_families": sorted(self.test_families),
            "train_sources": sorted(self.train_sources),
            "test_sources": sorted(self.test_sources),
            "family_overlap_train_test": sorted(self.train_families & self.test_families),
            "leakage_warnings": self.leakage_warnings,
            "leakage_free": len(self.leakage_warnings) == 0,
        }


# ──────────────────────────────────────────────────────────────────────
# Split functions
# ──────────────────────────────────────────────────────────────────────

def split_dataset(
    labels: np.ndarray,
    metadata: Optional[Sequence[SessionMetadata]] = None,
    config: Optional[SplitConfig] = None,
) -> SplitResult:
    """Split a dataset according to the configured strategy.

    Parameters
    ----------
    labels : Array of labels (0=benign, 1=C2) for each sample.
    metadata : Optional per-sample metadata for source-aware splitting.
               If None, falls back to random stratified splitting.
    config : Split configuration. Defaults to family-separated.

    Returns
    -------
    SplitResult with train/val/test indices and diagnostic info.
    """
    if config is None:
        config = SplitConfig()

    n = len(labels)
    if n == 0:
        raise ValueError("Cannot split empty dataset")

    # Route to the appropriate strategy
    if metadata is None or config.strategy == SplitStrategy.RANDOM_STRATIFIED:
        if config.strategy != SplitStrategy.RANDOM_STRATIFIED and metadata is None:
            warnings.warn(
                f"Strategy {config.strategy.value} requires metadata; "
                "falling back to random_stratified.",
                UserWarning,
                stacklevel=2,
            )
        return _random_stratified_split(labels, config)

    if config.strategy == SplitStrategy.FAMILY_SEPARATED:
        return _family_separated_split(labels, metadata, config)
    elif config.strategy == SplitStrategy.SOURCE_SEPARATED:
        return _source_separated_split(labels, metadata, config)
    elif config.strategy == SplitStrategy.TIME_SEPARATED:
        return _time_separated_split(labels, metadata, config)
    elif config.strategy == SplitStrategy.ZERO_SHOT:
        return _zero_shot_split(labels, metadata, config)
    else:
        raise ValueError(f"Unknown split strategy: {config.strategy}")


# ──────────────────────────────────────────────────────────────────────
# Strategy implementations
# ──────────────────────────────────────────────────────────────────────

def _random_stratified_split(
    labels: np.ndarray,
    config: SplitConfig,
) -> SplitResult:
    """v1-style random stratified split (baseline)."""
    n = len(labels)
    indices = np.arange(n)

    # Train+val vs test
    test_size = config.test_fraction
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=config.random_seed,
    )

    # Train vs val
    val_relative = config.val_fraction / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_relative,
        stratify=labels[trainval_idx],
        random_state=config.random_seed,
    )

    result = SplitResult(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        strategy=SplitStrategy.RANDOM_STRATIFIED,
        config=config,
    )
    result.validate_no_leakage()
    return result


def _family_separated_split(
    labels: np.ndarray,
    metadata: Sequence[SessionMetadata],
    config: SplitConfig,
) -> SplitResult:
    """Split by C2 family: no family appears in both train and test.

    Benign samples are split randomly across train/val/test since they
    don't have family labels. C2 samples are grouped by family and entire
    families are assigned to either train or test.
    """
    n = len(labels)
    rng = np.random.default_rng(config.random_seed)

    # Separate benign and C2 indices
    benign_idx = np.where(labels == LABEL_BENIGN)[0]
    c2_idx = np.where(labels == LABEL_C2)[0]

    # Group C2 by family
    family_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i in c2_idx:
        family = metadata[i].split_key_family()
        family_to_indices[family].append(i)

    families = sorted(family_to_indices.keys())

    # Determine held-out families
    if config.held_out_families:
        test_families = config.held_out_families & set(families)
        train_families = set(families) - test_families
    else:
        # Automatically select ~test_fraction of families for test
        n_test_families = max(
            config.min_families_per_split,
            int(len(families) * config.test_fraction),
        )
        n_test_families = min(n_test_families, len(families) - config.min_families_per_split)
        shuffled = list(families)
        rng.shuffle(shuffled)
        test_families = set(shuffled[:n_test_families])
        train_families = set(shuffled[n_test_families:])

    # Assign C2 indices
    c2_train_idx = []
    c2_test_idx = []
    for family in families:
        if family in test_families:
            c2_test_idx.extend(family_to_indices[family])
        else:
            c2_train_idx.extend(family_to_indices[family])

    c2_train_idx = np.array(c2_train_idx, dtype=np.int64)
    c2_test_idx = np.array(c2_test_idx, dtype=np.int64)

    # Split benign randomly
    benign_test_size = max(1, int(len(benign_idx) * config.test_fraction))
    benign_shuffled = rng.permutation(benign_idx)
    benign_test_idx = benign_shuffled[:benign_test_size]
    benign_trainval_idx = benign_shuffled[benign_test_size:]

    # Combine train+val C2 with train+val benign
    trainval_idx = np.concatenate([c2_train_idx, benign_trainval_idx])

    # Split train vs val (within trainval)
    trainval_labels = labels[trainval_idx]
    if len(np.unique(trainval_labels)) > 1:
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=config.val_fraction,
            stratify=trainval_labels,
            random_state=config.random_seed,
        )
    else:
        # Edge case: only one class in trainval
        split_point = int(len(trainval_idx) * (1 - config.val_fraction))
        shuffled_tv = rng.permutation(trainval_idx)
        train_idx = shuffled_tv[:split_point]
        val_idx = shuffled_tv[split_point:]

    # Test = held-out C2 families + benign test portion
    test_idx = np.concatenate([c2_test_idx, benign_test_idx])

    result = SplitResult(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        strategy=SplitStrategy.FAMILY_SEPARATED,
        config=config,
        train_families=train_families,
        test_families=test_families,
        val_families=train_families,  # Val draws from train families
    )

    # Populate source info
    result.train_sources = {metadata[i].source.value for i in train_idx}
    result.test_sources = {metadata[i].source.value for i in test_idx}

    result.validate_no_leakage()
    return result


def _source_separated_split(
    labels: np.ndarray,
    metadata: Sequence[SessionMetadata],
    config: SplitConfig,
) -> SplitResult:
    """Split by dataset source: no source appears in both train and test."""
    rng = np.random.default_rng(config.random_seed)

    # Group indices by source
    source_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        source_to_indices[meta.source.value].append(i)

    sources = sorted(source_to_indices.keys())
    if len(sources) < 2:
        warnings.warn(
            "Only one source found; falling back to random stratified split.",
            UserWarning,
            stacklevel=2,
        )
        return _random_stratified_split(labels, config)

    # Select test sources
    n_test_sources = max(1, int(len(sources) * config.test_fraction))
    n_test_sources = min(n_test_sources, len(sources) - 1)
    shuffled_sources = list(sources)
    rng.shuffle(shuffled_sources)
    test_sources = set(shuffled_sources[:n_test_sources])
    train_sources = set(shuffled_sources[n_test_sources:])

    # Assign indices
    trainval_indices = []
    test_indices = []
    for source in sources:
        if source in test_sources:
            test_indices.extend(source_to_indices[source])
        else:
            trainval_indices.extend(source_to_indices[source])

    trainval_idx = np.array(trainval_indices, dtype=np.int64)
    test_idx = np.array(test_indices, dtype=np.int64)

    # Train vs val split within trainval
    trainval_labels = labels[trainval_idx]
    if len(trainval_idx) > 0 and len(np.unique(trainval_labels)) > 1:
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=config.val_fraction,
            stratify=trainval_labels,
            random_state=config.random_seed,
        )
    else:
        split_point = int(len(trainval_idx) * (1 - config.val_fraction))
        shuffled_tv = rng.permutation(trainval_idx)
        train_idx = shuffled_tv[:split_point]
        val_idx = shuffled_tv[split_point:]

    result = SplitResult(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        strategy=SplitStrategy.SOURCE_SEPARATED,
        config=config,
        train_sources=train_sources,
        test_sources=test_sources,
    )

    # Populate family info
    result.train_families = {
        metadata[i].split_key_family() for i in train_idx if labels[i] == LABEL_C2
    }
    result.test_families = {
        metadata[i].split_key_family() for i in test_idx if labels[i] == LABEL_C2
    }

    result.validate_no_leakage()
    return result


def _time_separated_split(
    labels: np.ndarray,
    metadata: Sequence[SessionMetadata],
    config: SplitConfig,
) -> SplitResult:
    """Split by temporal ordering: train on earlier captures, test on later."""
    n = len(labels)

    # Sort by capture start timestamp
    timestamps = np.array([m.capture_start_ts for m in metadata])
    sorted_indices = np.argsort(timestamps)

    # Split at quantile
    split_point = int(n * config.time_split_quantile)
    val_start = int(split_point * (1 - config.val_fraction))

    train_idx = sorted_indices[:val_start]
    val_idx = sorted_indices[val_start:split_point]
    test_idx = sorted_indices[split_point:]

    result = SplitResult(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        strategy=SplitStrategy.TIME_SEPARATED,
        config=config,
    )

    # Populate family/source info
    result.train_families = {
        metadata[i].split_key_family() for i in train_idx if labels[i] == LABEL_C2
    }
    result.test_families = {
        metadata[i].split_key_family() for i in test_idx if labels[i] == LABEL_C2
    }
    result.train_sources = {metadata[i].source.value for i in train_idx}
    result.test_sources = {metadata[i].source.value for i in test_idx}

    result.validate_no_leakage()
    return result


def _zero_shot_split(
    labels: np.ndarray,
    metadata: Sequence[SessionMetadata],
    config: SplitConfig,
) -> SplitResult:
    """Zero-shot evaluation: test set contains entirely unseen families AND sources.

    This is the hardest evaluation — the system has never seen the C2 family
    or the dataset source during training. It tests pure behavioral generalization.
    """
    rng = np.random.default_rng(config.random_seed)

    # Gather all families and sources.
    family_to_indices: Dict[str, List[int]] = defaultdict(list)
    source_to_indices: Dict[str, List[int]] = defaultdict(list)

    for i, meta in enumerate(metadata):
        if labels[i] == LABEL_C2:
            family_to_indices[meta.split_key_family()].append(i)
        source_to_indices[meta.source.value].append(i)

    families = sorted(family_to_indices.keys())
    sources = sorted(source_to_indices.keys())

    # Determine zero-shot families.
    if config.zero_shot_families:
        zs_families = config.zero_shot_families & set(families)
    else:
        # Hold out ~30% of families for zero-shot.
        n_zs = max(1, int(len(families) * 0.3))
        shuffled = list(families)
        rng.shuffle(shuffled)
        zs_families = set(shuffled[:n_zs])

    # Determine zero-shot sources.
    if config.zero_shot_sources:
        zs_sources = config.zero_shot_sources & set(sources)
    else:
        zs_sources = set()

    # Build test set: whole held-out sources plus zero-shot families.
    test_idx_set: Set[int] = set()
    for source in zs_sources:
        test_idx_set.update(source_to_indices[source])
    for family in zs_families:
        test_idx_set.update(family_to_indices[family])

    benign_idx = np.where(labels == LABEL_BENIGN)[0]
    remaining_benign_idx = np.array([i for i in benign_idx if i not in test_idx_set], dtype=np.int64)

    if len(remaining_benign_idx) > 0:
        benign_test_size = max(1, int(len(remaining_benign_idx) * config.test_fraction))
        benign_shuffled = rng.permutation(remaining_benign_idx)
        benign_test_idx = benign_shuffled[:benign_test_size]
        benign_trainval_idx = benign_shuffled[benign_test_size:]
    else:
        benign_test_idx = np.empty((0,), dtype=np.int64)
        benign_trainval_idx = np.empty((0,), dtype=np.int64)

    test_idx = np.array(sorted(test_idx_set | set(benign_test_idx.tolist())), dtype=np.int64)

    # Train+val: remaining C2 families + remaining benign, excluding held-out sources.
    train_c2_idx = []
    train_families = set()
    for family in families:
        if family not in zs_families:
            train_c2_idx.extend(i for i in family_to_indices[family] if i not in test_idx_set)
            train_families.add(family)

    trainval_idx = np.array(train_c2_idx + benign_trainval_idx.tolist(), dtype=np.int64)

    if len(trainval_idx) == 0:
        raise ValueError(
            "Zero-shot split left no training samples. Check held-out families/sources and source balance."
        )

    # Split train vs val
    trainval_labels = labels[trainval_idx]
    if len(trainval_idx) > 0 and len(np.unique(trainval_labels)) > 1:
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=config.val_fraction,
            stratify=trainval_labels,
            random_state=config.random_seed,
        )
    else:
        split_point = int(len(trainval_idx) * (1 - config.val_fraction))
        shuffled_tv = rng.permutation(trainval_idx)
        train_idx = shuffled_tv[:split_point]
        val_idx = shuffled_tv[split_point:]

    result = SplitResult(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        strategy=SplitStrategy.ZERO_SHOT,
        config=config,
        train_families=train_families,
        test_families=zs_families,
        val_families=train_families,
    )

    result.train_sources = {metadata[i].source.value for i in train_idx}
    result.test_sources = {metadata[i].source.value for i in test_idx}

    result.validate_no_leakage()
    return result


# ──────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────

def print_split_report(result: SplitResult, labels: np.ndarray) -> None:
    """Print a detailed report of a dataset split."""
    summary = result.summary()

    print("\n" + "=" * 60)
    print(f"SPLIT REPORT — {summary['strategy'].upper()}")
    print("=" * 60)

    for split_name, indices in [
        ("Train", result.train_indices),
        ("Val", result.val_indices),
        ("Test", result.test_indices),
    ]:
        split_labels = labels[indices]
        n_c2 = int((split_labels == LABEL_C2).sum())
        n_benign = int((split_labels == LABEL_BENIGN).sum())
        total = len(indices)
        c2_pct = 100 * n_c2 / total if total > 0 else 0
        print(
            f"  {split_name:>5}: {total:>8,} samples "
            f"(C2: {n_c2:>6,} [{c2_pct:.1f}%], Benign: {n_benign:>6,})"
        )

    if summary["train_families"]:
        print(f"\n  Train families: {', '.join(summary['train_families'])}")
    if summary["test_families"]:
        print(f"  Test families:  {', '.join(summary['test_families'])}")
    if summary["family_overlap_train_test"]:
        print(f"  ⚠️  FAMILY OVERLAP: {', '.join(summary['family_overlap_train_test'])}")
    if summary["train_sources"]:
        print(f"\n  Train sources: {', '.join(summary['train_sources'])}")
    if summary["test_sources"]:
        print(f"  Test sources:  {', '.join(summary['test_sources'])}")

    if summary["leakage_warnings"]:
        for warning in summary["leakage_warnings"]:
            print(f"  🚨 {warning}")
    else:
        print("\n  ✅ No leakage detected")

    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────
# v1 Backward Compatibility — DO NOT REMOVE
# These functions are imported by torch_dataset.py, evaluate_holdout.py,
# score_report.py, and drift_report.py.
# ──────────────────────────────────────────────────────────────────────

from dataclasses import dataclass as _dataclass
from sklearn.model_selection import train_test_split as _train_test_split


@_dataclass
class SessionSplits:
    """v1-compatible split result container."""
    x_train: np.ndarray
    y_train: np.ndarray
    m_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    m_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    m_test: np.ndarray


def load_filtered_sessions(
    npz_path: str,
    min_flows: int = 5,
    expected_feature_names: Sequence[str] | None = None,
    expected_session_len: int | None = None,
) -> tuple:
    """Load sessions from NPZ and filter by minimum real flows.

    Returns (sequences, labels, masks) tuple.
    This is the v1 function preserved for backward compatibility.
    """
    from src.data_loader.npz_utils import load_session_npz

    sequences, labels, masks = load_session_npz(
        npz_path,
        expected_feature_names=expected_feature_names,
        expected_session_len=expected_session_len,
    )

    # Filter by min_flows
    real_flow_counts = masks.sum(axis=1)
    valid = real_flow_counts >= min_flows
    return sequences[valid], labels[valid], masks[valid]


def create_session_splits(
    npz_path: str,
    min_flows: int = 5,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
    expected_feature_names: Sequence[str] | None = None,
    expected_session_len: int | None = None,
) -> SessionSplits:
    """Load, filter, and split sessions from an NPZ file.

    This is the v1 function preserved for backward compatibility.
    Uses random stratified splitting (v1 default behavior).
    """
    sequences, labels, masks = load_filtered_sessions(
        npz_path,
        min_flows,
        expected_feature_names=expected_feature_names,
        expected_session_len=expected_session_len,
    )

    indices = np.arange(len(labels))

    # First split: train+val vs test
    trainval_idx, test_idx = _train_test_split(
        indices,
        test_size=test_fraction,
        stratify=labels,
        random_state=random_seed,
    )

    # Second split: train vs val
    val_relative = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = _train_test_split(
        trainval_idx,
        test_size=val_relative,
        stratify=labels[trainval_idx],
        random_state=random_seed,
    )

    return SessionSplits(
        x_train=sequences[train_idx],
        y_train=labels[train_idx],
        m_train=masks[train_idx],
        x_val=sequences[val_idx],
        y_val=labels[val_idx],
        m_val=masks[val_idx],
        x_test=sequences[test_idx],
        y_test=labels[test_idx],
        m_test=masks[test_idx],
    )
