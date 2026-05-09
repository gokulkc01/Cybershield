"""Family-aware split helpers for controlled generalization experiments.

This module wraps the existing source-aware split utilities with a simpler
interface for the controlled multifamily experiment:

- train on selected C2 families
- hold out one or more families for zero-shot evaluation
- emit a reproducible JSON manifest
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from src.data_loader.split_utils import SplitConfig, SplitResult, SplitStrategy, split_dataset
from src.features.feature_config_v2 import SessionMetadata


def _normalize_family_names(families: Optional[Iterable[str]]) -> List[str]:
    if families is None:
        return []
    normalized = sorted({str(family).strip().lower() for family in families if str(family).strip()})
    return normalized


@dataclass(frozen=True)
class FamilySplitManifest:
    """Reproducible record of a family-aware split."""

    split_name: str
    random_seed: int
    train_families: List[str] = field(default_factory=list)
    test_families: List[str] = field(default_factory=list)
    train_indices: List[int] = field(default_factory=list)
    val_indices: List[int] = field(default_factory=list)
    test_indices: List[int] = field(default_factory=list)
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    train_c2: int = 0
    val_c2: int = 0
    test_c2: int = 0

    def to_dict(self) -> dict:
        return {
            "split_name": self.split_name,
            "random_seed": self.random_seed,
            "train_families": self.train_families,
            "test_families": self.test_families,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "train_c2": self.train_c2,
            "val_c2": self.val_c2,
            "test_c2": self.test_c2,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


class FamilyAwareSplitter:
    """Create leakage-free train/val/test splits by C2 family."""

    def __init__(
        self,
        train_families: Optional[Iterable[str]] = None,
        test_families: Optional[Iterable[str]] = None,
        random_seed: int = 42,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
    ) -> None:
        self.train_families = _normalize_family_names(train_families)
        self.test_families = _normalize_family_names(test_families)
        self.random_seed = int(random_seed)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)

        overlap = set(self.train_families) & set(self.test_families)
        if overlap:
            raise ValueError(f"Train/test family overlap is not allowed: {sorted(overlap)}")

    def split(
        self,
        labels: np.ndarray,
        metadata: Sequence[SessionMetadata],
    ) -> tuple[SplitResult, FamilySplitManifest]:
        if metadata is None:
            raise ValueError("Family-aware splitting requires session metadata")

        held_out_families = set(self.test_families) if self.test_families else None
        config = SplitConfig(
            strategy=SplitStrategy.FAMILY_SEPARATED,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
            random_seed=self.random_seed,
            held_out_families=held_out_families,
        )
        result = split_dataset(labels=np.asarray(labels), metadata=metadata, config=config)

        manifest = FamilySplitManifest(
            split_name="family_aware_v1",
            random_seed=self.random_seed,
            train_families=sorted(self.train_families or result.train_families),
            test_families=sorted(self.test_families or result.test_families),
            train_indices=result.train_indices.tolist(),
            val_indices=result.val_indices.tolist(),
            test_indices=result.test_indices.tolist(),
            train_size=len(result.train_indices),
            val_size=len(result.val_indices),
            test_size=len(result.test_indices),
            train_c2=int(np.sum(labels[result.train_indices] == 1)),
            val_c2=int(np.sum(labels[result.val_indices] == 1)),
            test_c2=int(np.sum(labels[result.test_indices] == 1)),
        )
        return result, manifest
