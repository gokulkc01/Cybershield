"""
Red-Agent Orchestrator: Unified adversarial behavioral robustness system.

Coordinates mutation generation, detection, reward computation, and policy learning.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass

from src.red_agent.mutations import (
    TimingMutationEngine,
    FlowMutationEngine,
    TLSMutationEngine,
    CompositeMutationEngine,
)
from src.red_agent.validation.behavioral_validity_constraints import (
    BehavioralValidityConstraints,
)
from src.red_agent.validation.realism_discriminator import RealismDiscriminator
from src.red_agent.training.reward_engine import MultiObjectiveRewardEngine
from src.features.feature_config import FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class MutationResult:
    """Result of a mutation operation."""

    original_session: np.ndarray
    mutated_session: np.ndarray
    mutation_type: str
    mutation_metadata: Dict
    constraint_violations: List[str]
    realism_score: float
    is_realistic: bool


@dataclass
class EvaluationResult:
    """Result of a mutation evaluation."""

    mutation_result: MutationResult
    detector_confidence_original: float
    detector_confidence_mutated: float
    reward: float
    reward_components: Dict[str, float]


class RedAgentOrchestrator:
    """
    Main Red-Agent system.

    Coordinates:
    - Mutation generation
    - Validity checking
    - Realism discrimination
    - Detection evaluation
    - Reward computation
    - Policy learning
    """

    def __init__(
        self,
        feature_names: List[str] = FEATURE_NAMES,
    ):
        """
        Initialize Red-Agent.

        Args:
            feature_names: List of behavioral feature names
        """
        self.feature_names = feature_names
        self.feature_idx = {name: i for i, name in enumerate(feature_names)}

        # Initialize components
        self.constraints = BehavioralValidityConstraints()
        self.discriminator = RealismDiscriminator()
        self.reward_engine = MultiObjectiveRewardEngine()

        # Initialize mutation engines
        self.timing_engine = TimingMutationEngine(feature_names)
        self.flow_engine = FlowMutationEngine(feature_names)
        self.tls_engine = TLSMutationEngine(feature_names)
        self.composite_engine = CompositeMutationEngine(feature_names)

        logger.info("Red-Agent initialized")

    def mutate_session(
        self,
        session: np.ndarray,
        mask: np.ndarray,
        mutation_type: str = "timing",
        severity: float = 0.5,
        composite_chain: Optional[List[str]] = None,
    ) -> MutationResult:
        """
        Generate mutation of a session.

        Args:
            session: Session tensor (20, D)
            mask: Valid mask (20,)
            mutation_type: Type of mutation
            severity: Mutation severity [0, 1]
            composite_chain: Chain for composite mutations

        Returns:
            MutationResult with all metadata
        """
        # Select mutation engine
        if mutation_type == "timing":
            engine = self.timing_engine
        elif mutation_type == "flow":
            engine = self.flow_engine
        elif mutation_type == "tls":
            engine = self.tls_engine
        elif mutation_type == "composite":
            engine = self.composite_engine
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

        # Generate mutation
        if mutation_type == "composite":
            mutated, metadata = engine.mutate(
                session, mask, severity, mutation_chain=composite_chain
            )
        else:
            mutated, metadata = engine.mutate(session, mask, severity)

        # Validate constraints
        is_valid, violations = self.constraints.check_all_constraints(
            session, mutated, self.feature_names
        )

        # Score realism
        realism_score, _ = self.discriminator.score_realism(
            mutated, session, self.feature_names
        )
        is_realistic = self.discriminator.is_realistic(
            mutated, session, self.feature_names, threshold=0.6
        )

        logger.debug(
            f"Mutation {mutation_type}: valid={is_valid}, realistic={is_realistic}, "
            f"realism_score={realism_score:.3f}, violations={len(violations)}"
        )

        return MutationResult(
            original_session=session,
            mutated_session=mutated,
            mutation_type=mutation_type,
            mutation_metadata=metadata,
            constraint_violations=violations,
            realism_score=realism_score,
            is_realistic=is_realistic,
        )

    def evaluate_mutation(
        self,
        mutation_result: MutationResult,
        detector_inference_fn,
        functionality_score: float = 0.8,
        stability_score: float = 0.85,
        diversity_bonus: float = 0.0,
        detection_threshold: float = 0.5,
    ) -> EvaluationResult:
        """
        Evaluate mutation and compute reward.

        Args:
            mutation_result: Result from mutate_session()
            detector_inference_fn: Function that returns detector confidence
            functionality_score: Operational functionality [0, 1]
            stability_score: Stability/coherence [0, 1]
            diversity_bonus: Diversity exploration bonus

        Returns:
            EvaluationResult with reward
        """
        # Run detector on original
        try:
            conf_original = detector_inference_fn(
                mutation_result.original_session,
            )
        except Exception as e:
            logger.warning(f"Detector inference failed on original: {e}")
            conf_original = 0.5

        # Run detector on mutated
        try:
            conf_mutated = detector_inference_fn(
                mutation_result.mutated_session,
            )
        except Exception as e:
            logger.warning(f"Detector inference failed on mutated: {e}")
            conf_mutated = 0.5

        # Compute reward
        reward, reward_components = self.reward_engine.compute_reward(
            detector_confidence_original=conf_original,
            detector_confidence_mutated=conf_mutated,
            realism_score=mutation_result.realism_score,
            functionality_score=functionality_score,
            stability_score=stability_score,
            diversity_bonus=diversity_bonus,
            constraint_violations=mutation_result.constraint_violations,
            detection_threshold=detection_threshold,
        )

        # Validate reward
        is_valid, error_msg = self.reward_engine.validate_reward_integrity(
            reward_components
        )
        if not is_valid:
            logger.error(f"Invalid reward: {error_msg}")
            reward = -1.0

        logger.debug(
            f"Evaluation: conf_original={conf_original:.3f}, "
            f"conf_mutated={conf_mutated:.3f}, reward={reward:.3f}"
        )

        return EvaluationResult(
            mutation_result=mutation_result,
            detector_confidence_original=conf_original,
            detector_confidence_mutated=conf_mutated,
            reward=reward,
            reward_components=reward_components,
        )

    def batch_evaluate_mutations(
        self,
        sessions: np.ndarray,
        masks: np.ndarray,
        detector_inference_fn,
        mutation_types: Optional[List[str]] = None,
        severities: Optional[List[float]] = None,
        detection_threshold: float = 0.5,
    ) -> List[EvaluationResult]:
        """
        Evaluate batch of mutations.

        Args:
            sessions: Session tensors (N, 20, D)
            masks: Valid masks (N, 20)
            detector_inference_fn: Detector function
            mutation_types: List of mutation types (default: mix)
            severities: List of severities (default: random)

        Returns:
            List of EvaluationResult
        """
        if mutation_types is None:
            mutation_types = ["timing", "flow", "tls", "composite"] * (
                len(sessions) // 4 + 1
            )
            mutation_types = mutation_types[: len(sessions)]

        if severities is None:
            severities = np.random.uniform(0.2, 0.8, len(sessions))

        results = []
        for i, (session, mask, mut_type, severity) in enumerate(
            zip(sessions, masks, mutation_types, severities)
        ):
            try:
                mut_result = self.mutate_session(
                    session, mask, mutation_type=mut_type, severity=severity
                )
                eval_result = self.evaluate_mutation(
                    mut_result,
                    detector_inference_fn,
                    detection_threshold=detection_threshold,
                )
                results.append(eval_result)
            except Exception as e:
                logger.warning(f"Batch evaluation failed for sample {i}: {e}")
                results.append(None)

        return results
