"""Contracts and data models for mutation execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class MutationSpec:
    """Single mutation instruction with deterministic seed and severity."""

    mutation_type: str
    mutation_params: Dict[str, Any] = field(default_factory=dict)
    mutation_severity: float = 0.5
    seed: int = 0


@dataclass(frozen=True)
class MutationRecord:
    """One mutation operation entry for sample-level lineage."""

    mutation_type: str
    category: str
    mutation_params: Dict[str, Any]
    mutation_severity: float
    seed: int


@dataclass(frozen=True)
class MutationResult:
    """Mutated sample and full lineage metadata."""

    mutated_sequence: np.ndarray
    mutation_history: List[MutationRecord]
