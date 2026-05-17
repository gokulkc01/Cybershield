"""Composite mutations: coordinated multi-stage transformations."""

import numpy as np
from typing import Tuple, Dict, List
from .base_mutation_engine import BaseMutationEngine
from .timing_mutations import TimingMutationEngine
from .flow_mutations import FlowMutationEngine
from .tls_mutations import TLSMutationEngine


class CompositeMutationEngine(BaseMutationEngine):
    """
    Composite mutations: coordinated multi-mutation chains.
    
    These represent sophisticated adversarial strategies like:
    - "high jitter + long sleep + burst reconnect"
    - "TLS reshaping + session splitting"
    - "sparse persistence + delayed reconnect"
    """

    def __init__(self, feature_names: List[str]):
        """Initialize with sub-engines."""
        super().__init__(feature_names)
        self.timing_engine = TimingMutationEngine(feature_names)
        self.flow_engine = FlowMutationEngine(feature_names)
        self.tls_engine = TLSMutationEngine(feature_names)

    def mutate(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float = 0.5,
        mutation_chain: List[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply composite mutation.

        Args:
            session: Session tensor (20, D)
            mask: Valid mask (20,)
            severity: Mutation severity [0, 1]
            mutation_chain: List of mutations to apply in sequence
                Default: ["timing", "flow", "tls"]

        Returns:
            (mutated_session, metadata)
        """
        if mutation_chain is None:
            mutation_chain = ["timing", "flow", "tls"]

        mutated = session.copy()
        metadata = {
            "mutation_type": "composite",
            "severity": severity,
            "chain": mutation_chain,
            "stages": {},
        }

        # Apply each mutation in sequence
        for stage_name in mutation_chain:
            if stage_name == "timing":
                mutated, stage_info = self.timing_engine.mutate(mutated, mask, severity)
            elif stage_name == "flow":
                mutated, stage_info = self.flow_engine.mutate(mutated, mask, severity)
            elif stage_name == "tls":
                mutated, stage_info = self.tls_engine.mutate(mutated, mask, severity)
            else:
                continue

            metadata["stages"][stage_name] = stage_info

        return mutated, metadata

    @staticmethod
    def get_predefined_strategies() -> Dict[str, List[str]]:
        """Return predefined composite mutation strategies."""
        return {
            "stealth_persistence": ["timing", "flow"],  # Jitter + sparse comms
            "tls_evasion": ["tls", "flow"],  # Padding + byte reshaping
            "sophisticated_c2": ["timing", "tls", "flow"],  # All three combined
            "sparse_beacon": ["timing"],  # Just jitter (lightweight)
            "encrypted_hiding": ["tls"],  # Just TLS (hiding payload size)
        }
