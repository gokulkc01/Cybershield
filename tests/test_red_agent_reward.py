from __future__ import annotations

from src.red_agent.training.reward_engine import MultiObjectiveRewardEngine


def test_reward_uses_validation_threshold_for_crossing_bonus():
    engine = MultiObjectiveRewardEngine()

    crossed, crossed_components = engine.compute_reward(
        detector_confidence_original=0.90,
        detector_confidence_mutated=0.70,
        realism_score=0.9,
        functionality_score=0.9,
        stability_score=0.9,
        detection_threshold=0.80,
    )
    not_crossed, not_crossed_components = engine.compute_reward(
        detector_confidence_original=0.90,
        detector_confidence_mutated=0.82,
        realism_score=0.9,
        functionality_score=0.9,
        stability_score=0.9,
        detection_threshold=0.80,
    )

    assert crossed > not_crossed
    assert crossed_components["evasion"] > not_crossed_components["evasion"]
