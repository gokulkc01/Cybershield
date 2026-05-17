"""
Base Mutation Engine: Abstract interface for behavioral mutations.

All mutations inherit from this interface.
Ensures consistency and type safety.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, List, Optional


class BaseMutationEngine(ABC):
    """Abstract base for mutation engines."""

    def __init__(self, feature_names: List[str]):
        """Initialize with feature names."""
        self.feature_names = feature_names
        self.feature_idx = {name: i for i, name in enumerate(feature_names)}

    @abstractmethod
    def mutate(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        severity: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply mutation to session.

        Args:
            session: Session tensor (20, D)
            mask: Valid (non-padding) mask (20,)
            severity: Mutation severity [0, 1]

        Returns:
            (mutated_session, mutation_metadata)
        """
        pass

    def get_mutation_type(self) -> str:
        """Return mutation type identifier."""
        return self.__class__.__name__
