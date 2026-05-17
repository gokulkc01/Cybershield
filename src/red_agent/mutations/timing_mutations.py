"""Timing-based behavioral mutations."""

import numpy as np
from typing import Tuple, Dict, List
from .base_mutation_engine import BaseMutationEngine
from ..validation.behavioral_validity_constraints import constraints


class TimingMutationEngine(BaseMutationEngine):
    """Timing mutations: jitter, sleep intervals, beacon patterns."""

    def mutate(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply timing mutation.

        Args:
            session: Session tensor (20, D)
            mask: Valid mask (20,)
            severity: Mutation severity [0, 1]

        Returns:
            (mutated_session, metadata)
        """
        mutated = session.copy()
        metadata = {"mutation_type": "timing", "severity": severity}

        # Apply jitter to inter-arrival times
        if "iat" in self.feature_idx:
            mutated, jitter_info = self._apply_jitter(mutated, mask, severity)
            metadata.update(jitter_info)

        # Apply sleep intervals
        if "duration" in self.feature_idx:
            mutated, sleep_info = self._apply_sleep_perturbation(mutated, mask, severity)
            metadata.update(sleep_info)

        return mutated, metadata

    def _apply_jitter(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply jitter to inter-arrival times."""
        mutated = session.copy()
        iat_idx = self.feature_idx.get("iat")

        if iat_idx is None:
            return mutated, {}

        iat_values = mutated[mask, iat_idx]

        if len(iat_values) == 0:
            return mutated, {}

        # Jitter amount based on severity
        base_jitter = np.mean(iat_values) * severity * 0.2  # Up to 20% at max severity
        jitter = np.random.normal(0, base_jitter, len(iat_values))

        # Apply jitter with bounds checking
        jittered_iat = iat_values + jitter
        jittered_iat = np.clip(
            jittered_iat,
            constraints.MIN_IAT,
            constraints.MAX_IAT,
        )

        mutated[mask, iat_idx] = jittered_iat

        metadata = {
            "jitter_applied": True,
            "jitter_mean": float(np.mean(jitter)),
            "jitter_std": float(np.std(jitter)),
            "iat_original_mean": float(np.mean(iat_values)),
            "iat_mutated_mean": float(np.mean(jittered_iat)),
        }

        return mutated, metadata

    def _apply_sleep_perturbation(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Artificially extend session duration (simulating sleep)."""
        mutated = session.copy()
        duration_idx = self.feature_idx.get("duration")

        if duration_idx is None:
            return mutated, {}

        duration_values = mutated[mask, duration_idx]

        if len(duration_values) == 0:
            return mutated, {}

        # Add sleep time based on severity
        # At severity 1.0, add up to 10x the original duration
        sleep_multiplier = 1.0 + (severity * 9.0)
        sleep_amount = duration_values * (sleep_multiplier - 1.0)
        sleep_amount = np.clip(
            sleep_amount,
            0,
            constraints.MAX_SLEEP_DURATION,
        )

        new_duration = duration_values + sleep_amount
        new_duration = np.clip(
            new_duration,
            constraints.MIN_SESSION_DURATION,
            constraints.MAX_SESSION_DURATION,
        )

        mutated[mask, duration_idx] = new_duration

        metadata = {
            "sleep_applied": True,
            "sleep_multiplier": float(sleep_multiplier),
            "duration_original_mean": float(np.mean(duration_values)),
            "duration_mutated_mean": float(np.mean(new_duration)),
        }

        return mutated, metadata
