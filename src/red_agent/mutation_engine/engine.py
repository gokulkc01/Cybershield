"""Deterministic mutation engine for behavioral robustness testing."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from src.red_agent.mutation_engine.contracts import MutationRecord, MutationResult, MutationSpec
from src.red_agent.mutation_registry import get_mutation


class MutationEngine:
    """Apply configured mutations sequentially while preserving lineage."""

    def apply_mutations(self, sequence: np.ndarray, specs: Iterable[MutationSpec]) -> MutationResult:
        if sequence.ndim != 2:
            raise ValueError(f"Expected sequence shape (T, F), got {sequence.shape}")

        current = np.asarray(sequence, dtype=np.float32).copy()
        history: List[MutationRecord] = []

        for spec in specs:
            registered = get_mutation(spec.mutation_type)
            rng = np.random.default_rng(int(spec.seed))
            current = registered.fn(
                current,
                rng,
                float(spec.mutation_severity),
                dict(spec.mutation_params),
            )
            if current.shape != sequence.shape:
                raise ValueError(
                    f"Mutation '{spec.mutation_type}' changed shape from {sequence.shape} to {current.shape}"
                )
            history.append(
                MutationRecord(
                    mutation_type=spec.mutation_type,
                    category=registered.category,
                    mutation_params=dict(spec.mutation_params),
                    mutation_severity=float(spec.mutation_severity),
                    seed=int(spec.seed),
                )
            )

        return MutationResult(mutated_sequence=current.astype(np.float32), mutation_history=history)
