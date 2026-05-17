"""Multi-objective reward function for Red-Agent PPO."""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiObjectiveRewardEngine:
    """
    Multi-objective reward function for Red-Agent PPO.

    Balances:
    1. Evasion (reduce detector confidence)
    2. Realism (preserve behavioral authenticity)
    3. Functionality (maintain operational C2 properties)
    4. Stability (prevent adversarial artifacts)
    5. Diversity (avoid repetitive exploits)
    """

    def __init__(
        self,
        evasion_weight: float = 0.4,
        realism_weight: float = 0.3,
        functionality_weight: float = 0.15,
        stability_weight: float = 0.1,
        diversity_weight: float = 0.05,
    ):
        """Initialize reward engine with weights."""
        self.evasion_weight = evasion_weight
        self.realism_weight = realism_weight
        self.functionality_weight = functionality_weight
        self.stability_weight = stability_weight
        self.diversity_weight = diversity_weight

        # Normalize weights
        total = sum(
            [
                evasion_weight,
                realism_weight,
                functionality_weight,
                stability_weight,
                diversity_weight,
            ]
        )
        self.evasion_weight /= total
        self.realism_weight /= total
        self.functionality_weight /= total
        self.stability_weight /= total
        self.diversity_weight /= total

        logger.info(
            f"Reward weights: evasion={self.evasion_weight:.2f}, "
            f"realism={self.realism_weight:.2f}, "
            f"functionality={self.functionality_weight:.2f}, "
            f"stability={self.stability_weight:.2f}, "
            f"diversity={self.diversity_weight:.2f}"
        )

    def compute_reward(
        self,
        detector_confidence_original: float,
        detector_confidence_mutated: float,
        realism_score: float,
        functionality_score: float,
        stability_score: float,
        diversity_bonus: float = 0.0,
        constraint_violations: Optional[list] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute multi-objective reward."""
        components = {}

        # 1. EVASION REWARD
        confidence_delta = detector_confidence_original - detector_confidence_mutated
        if confidence_delta > 0:
            evasion_reward = min(confidence_delta, 1.0)
            if (
                detector_confidence_original >= 0.5
                and detector_confidence_mutated < 0.5
            ):
                evasion_reward += 0.5
            evasion_reward = np.clip(evasion_reward, -1.0, 2.0)
        else:
            evasion_reward = -0.5
        components["evasion"] = evasion_reward

        # 2. REALISM REWARD (CRITICAL)
        if realism_score >= 0.7:
            realism_reward = realism_score - 0.7
        elif realism_score >= 0.4:
            realism_reward = -0.2
        else:
            realism_reward = -1.0 - (0.4 - realism_score) * 2.0
        realism_reward = np.clip(realism_reward, -2.0, 0.3)
        components["realism"] = realism_reward

        # 3. FUNCTIONALITY REWARD
        if functionality_score >= 0.8:
            functionality_reward = 0.2
        elif functionality_score >= 0.5:
            functionality_reward = 0.0
        else:
            functionality_reward = -0.3 - (0.5 - functionality_score)
        functionality_reward = np.clip(functionality_reward, -1.0, 0.2)
        components["functionality"] = functionality_reward

        # 4. STABILITY REWARD
        if stability_score >= 0.9:
            stability_reward = 0.1
        elif stability_score >= 0.6:
            stability_reward = 0.0
        else:
            stability_reward = -0.5 - (0.6 - stability_score)
        stability_reward = np.clip(stability_reward, -1.0, 0.1)
        components["stability"] = stability_reward

        # 5. DIVERSITY REWARD
        diversity_reward = diversity_bonus * 0.2
        components["diversity"] = diversity_reward

        # 6. CONSTRAINT PENALTY
        constraint_penalty = 0.0
        if constraint_violations and len(constraint_violations) > 0:
            constraint_penalty = -0.5 * len(constraint_violations)
            constraint_penalty = np.clip(constraint_penalty, -2.0, 0.0)
        components["constraint_penalty"] = constraint_penalty

        # FINAL WEIGHTED SUM
        total_reward = (
            self.evasion_weight * evasion_reward
            + self.realism_weight * realism_reward
            + self.functionality_weight * functionality_reward
            + self.stability_weight * stability_reward
            + self.diversity_weight * diversity_reward
            + constraint_penalty
        )
        components["total"] = total_reward

        return float(total_reward), components

    def validate_reward_integrity(
        self,
        components: Dict[str, float],
    ) -> Tuple[bool, Optional[str]]:
        """Validate reward components."""
        for key, value in components.items():
            if np.isnan(value) or np.isinf(value):
                return False, f"Invalid reward component {key}={value}"
            if abs(value) > 10.0:
                return False, f"Extreme reward component {key}={value}"
        return True, None
