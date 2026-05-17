"""TLS/HTTPS behavioral mutations."""

import numpy as np
from typing import Tuple, Dict, List
from .base_mutation_engine import BaseMutationEngine


class TLSMutationEngine(BaseMutationEngine):
    """TLS/HTTPS mutations: padding, session reuse, handshake variation."""

    def mutate(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply TLS-level mutation.

        Args:
            session: Session tensor (20, D)
            mask: Valid mask (20,)
            severity: Mutation severity [0, 1]

        Returns:
            (mutated_session, metadata)
        """
        mutated = session.copy()
        metadata = {"mutation_type": "tls", "severity": severity}

        # Apply TLS padding (add bytes without changing packet count much)
        if "orig_bytes" in self.feature_idx and "resp_bytes" in self.feature_idx:
            mutated, padding_info = self._apply_tls_padding(mutated, mask, severity)
            metadata.update(padding_info)

        # Adjust for TLS session reuse (some packets become handshakes)
        if "orig_pkts" in self.feature_idx:
            mutated, reuse_info = self._apply_session_reuse(mutated, mask, severity)
            metadata.update(reuse_info)

        return mutated, metadata

    def _apply_tls_padding(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Add TLS record padding.
        
        TLS records are typically padded to hide payload size.
        Add bytes without proportionally increasing packets.
        """
        mutated = session.copy()
        orig_bytes_idx = self.feature_idx["orig_bytes"]
        resp_bytes_idx = self.feature_idx["resp_bytes"]

        orig_bytes = mutated[mask, orig_bytes_idx]
        resp_bytes = mutated[mask, resp_bytes_idx]

        # Padding amount: up to 30% extra at max severity
        padding_fraction = severity * 0.3
        padding_orig = orig_bytes * np.random.uniform(0, padding_fraction, len(orig_bytes))
        padding_resp = resp_bytes * np.random.uniform(0, padding_fraction, len(resp_bytes))

        mutated[mask, orig_bytes_idx] = orig_bytes + padding_orig
        mutated[mask, resp_bytes_idx] = resp_bytes + padding_resp

        metadata = {
            "tls_padding_applied": True,
            "padding_fraction": float(padding_fraction),
            "orig_bytes_padded": float(np.mean(padding_orig)),
            "resp_bytes_padded": float(np.mean(padding_resp)),
        }

        return mutated, metadata

    def _apply_session_reuse(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Simulate TLS session reuse patterns.
        
        Session reuse reduces handshake overhead, changing packet/byte ratio.
        """
        mutated = session.copy()
        orig_pkts_idx = self.feature_idx.get("orig_pkts")

        if orig_pkts_idx is None:
            return mutated, {}

        orig_pkts = mutated[mask, orig_pkts_idx]

        # Remove some packets (handshake optimization)
        # At max severity, reduce by up to 20%
        reduction_fraction = severity * 0.2
        packets_removed = np.random.binomial(
            orig_pkts.astype(int),
            reduction_fraction,
        )

        mutated[mask, orig_pkts_idx] = orig_pkts - packets_removed

        metadata = {
            "session_reuse_applied": True,
            "reduction_fraction": float(reduction_fraction),
            "packets_removed_mean": float(np.mean(packets_removed)),
        }

        return mutated, metadata
