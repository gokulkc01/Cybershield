"""CTU-13 multifamily dataset loader scaffold.

This module is intentionally manifest-driven so the experiment can keep the
scenario-to-family mapping explicit. It combines one or more preprocessed
session NPZ files into a single dataset and writes a reproducible manifest
that records the provenance for each source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from src.data_loader.npz_utils import load_session_npz
from src.features.feature_config_experiment import FEATURE_NAMES


@dataclass(frozen=True)
class Ctu13SourceSpec:
    """Single CTU-13 source entry for the multifamily manifest."""

    path: str
    family: str
    label_type: str = "c2"
    scenario: str = ""

    def normalized(self) -> "Ctu13SourceSpec":
        return Ctu13SourceSpec(
            path=str(Path(self.path)),
            family=self.family.strip().lower(),
            label_type=self.label_type.strip().lower(),
            scenario=self.scenario.strip(),
        )


@dataclass(frozen=True)
class Ctu13MultifamilyManifest:
    """Reproducibility manifest for the CTU-13 multifamily experiment."""

    experiment_name: str
    feature_names: List[str]
    sources: List[Ctu13SourceSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "feature_names": self.feature_names,
            "sources": [asdict(source) for source in self.sources],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


class Ctu13MultifamilyLoader:
    """Combine scenario-specific CTU-13 session NPZs into one dataset."""

    def __init__(self, sources: Sequence[Ctu13SourceSpec], experiment_name: str = "ctu13_multifamily_generalization") -> None:
        normalized = [source.normalized() for source in sources]
        if not normalized:
            raise ValueError("At least one CTU-13 source is required")
        self.sources = normalized
        self.experiment_name = experiment_name

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Ctu13MultifamilyManifest]:
        sessions_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []

        for source in self.sources:
            npz_path = Path(source.path)
            if not npz_path.exists():
                raise FileNotFoundError(f"CTU-13 source not found: {npz_path}")

            sessions, labels, masks = load_session_npz(
                str(npz_path),
                expected_feature_names=FEATURE_NAMES,
            )
            if source.label_type == "c2":
                labels = np.ones(len(sessions), dtype=np.int64)
            elif source.label_type == "benign":
                labels = np.zeros(len(sessions), dtype=np.int64)
            else:
                raise ValueError(f"Unsupported label_type: {source.label_type}")

            if sessions.shape[2] != len(FEATURE_NAMES):
                raise ValueError(
                    f"Feature dimension mismatch for {npz_path}: "
                    f"expected {len(FEATURE_NAMES)}, got {sessions.shape[2]}"
                )

            sessions_list.append(sessions)
            labels_list.append(labels)
            masks_list.append(masks)

        sessions = np.concatenate(sessions_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        masks = np.concatenate(masks_list, axis=0)

        manifest = Ctu13MultifamilyManifest(
            experiment_name=self.experiment_name,
            feature_names=list(FEATURE_NAMES),
            sources=list(self.sources),
        )
        return sessions, labels, masks, manifest

    def save_manifest(self, path: str | Path) -> None:
        _, _, _, manifest = self.load()
        manifest.save(path)
