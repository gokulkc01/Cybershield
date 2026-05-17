"""Flow-level behavioral mutations."""

import numpy as np
from typing import Tuple, Dict, List
from .base_mutation_engine import BaseMutationEngine
from ..validation.behavioral_validity_constraints import constraints


class FlowMutationEngine(BaseMutationEngine):
    """Flow mutations: packet distribution, byte volume, duration perturbation."""

    def mutate(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply flow-level mutation.

        Args:
            session: Session tensor (20, D)
            mask: Valid mask (20,)
            severity: Mutation severity [0, 1]

        Returns:
            (mutated_session, metadata)
        """
        mutated = session.copy()
        metadata = {"mutation_type": "flow", "severity": severity}

        # Mutate packet counts
        if "orig_pkts" in self.feature_idx and "resp_pkts" in self.feature_idx:
            mutated, pkt_info = self._mutate_packets(mutated, mask, severity)
            metadata.update(pkt_info)

        # Mutate byte volumes
        if "orig_bytes" in self.feature_idx and "resp_bytes" in self.feature_idx:
            mutated, byte_info = self._mutate_bytes(mutated, mask, severity)
            metadata.update(byte_info)

        # Mutate duration
        if "duration" in self.feature_idx:
            mutated, dur_info = self._mutate_duration(mutated, mask, severity)
            metadata.update(dur_info)

        return mutated, metadata

    def _mutate_packets(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Add packet noise while maintaining realism."""
        mutated = session.copy()
        orig_pkts_idx = self.feature_idx["orig_pkts"]
        resp_pkts_idx = self.feature_idx["resp_pkts"]

        orig_pkts = mutated[mask, orig_pkts_idx]
        resp_pkts = mutated[mask, resp_pkts_idx]

        # Add Poisson noise (packets follow approximately Poisson)
        noise_orig = np.random.poisson(orig_pkts * severity * 0.2)
        noise_resp = np.random.poisson(resp_pkts * severity * 0.2)

        mutated_orig = orig_pkts + noise_orig
        mutated_resp = resp_pkts + noise_resp

        # Enforce bounds
        mutated_orig = np.clip(mutated_orig, constraints.MIN_PACKETS, constraints.MAX_PACKETS)
        mutated_resp = np.clip(mutated_resp, constraints.MIN_PACKETS, constraints.MAX_PACKETS)

        mutated[mask, orig_pkts_idx] = mutated_orig
        mutated[mask, resp_pkts_idx] = mutated_resp

        metadata = {
            "packets_mutated": True,
            "orig_pkts_original_mean": float(np.mean(orig_pkts)),
            "orig_pkts_mutated_mean": float(np.mean(mutated_orig)),
            "resp_pkts_original_mean": float(np.mean(resp_pkts)),
            "resp_pkts_mutated_mean": float(np.mean(mutated_resp)),
        }

        return mutated, metadata

    def _mutate_bytes(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Add byte volume perturbation."""
        mutated = session.copy()
        orig_bytes_idx = self.feature_idx["orig_bytes"]
        resp_bytes_idx = self.feature_idx["resp_bytes"]

        orig_bytes = mutated[mask, orig_bytes_idx]
        resp_bytes = mutated[mask, resp_bytes_idx]

        # Apply multiplicative Gaussian noise
        noise_factor = 1.0 + np.random.normal(0, severity * 0.15, len(orig_bytes))
        noise_factor = np.clip(noise_factor, 0.5, 2.0)  # Cap at 50%-200%

        mutated_orig = orig_bytes * noise_factor
        mutated_resp = resp_bytes * noise_factor

        # Enforce bounds
        mutated_orig = np.clip(mutated_orig, constraints.MIN_BYTES, constraints.MAX_BYTES)
        mutated_resp = np.clip(mutated_resp, constraints.MIN_BYTES, constraints.MAX_BYTES)

        mutated[mask, orig_bytes_idx] = mutated_orig
        mutated[mask, resp_bytes_idx] = mutated_resp

        metadata = {
            "bytes_mutated": True,
            "orig_bytes_original_mean": float(np.mean(orig_bytes)),
            "orig_bytes_mutated_mean": float(np.mean(mutated_orig)),
            "resp_bytes_original_mean": float(np.mean(resp_bytes)),
            "resp_bytes_mutated_mean": float(np.mean(mutated_resp)),
        }

        return mutated, metadata

    def _mutate_duration(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perturb session duration."""
        mutated = session.copy()
        duration_idx = self.feature_idx["duration"]

        duration_values = mutated[mask, duration_idx]

        # Multiplicative perturbation
        noise_factor = 1.0 + np.random.normal(0, severity * 0.2)
        noise_factor = np.clip(noise_factor, 0.5, 3.0)

        mutated_duration = duration_values * noise_factor
        mutated_duration = np.clip(
            mutated_duration,
            constraints.MIN_SESSION_DURATION,
            constraints.MAX_SESSION_DURATION,
        )

        mutated[mask, duration_idx] = mutated_duration

        metadata = {
            "duration_mutated": True,
            "duration_original_mean": float(np.mean(duration_values)),
            "duration_mutated_mean": float(np.mean(mutated_duration)),
        }

        return mutated, metadata
