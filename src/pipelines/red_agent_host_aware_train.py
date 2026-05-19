"""Train Red-Agent PPO against a frozen host-aware C2 detector.

This pipeline mutates only feature-space telemetry windows from an authorized
host-aware split.  It does not generate payloads, infrastructure, or live C2
traffic; the goal is defensive robustness analysis of the detector.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical, Normal

from src.data_loader.host_window_dataset import HostWindowArrays, load_host_windows_npz
from src.red_agent.detector_adapters import FrozenDetectorAdapter
from src.red_agent.orchestrator import EvaluationResult, MutationResult, RedAgentOrchestrator
from src.red_agent.policy import MutationPolicyNetwork
from src.red_agent.training import PPOTrainer


MUTATION_TYPES = ["timing", "flow", "tls", "composite", "none"]
NONE_ACTION_INDEX = 4


@dataclass(frozen=True)
class HostAwareRedAgentSummary:
    epochs: int
    batches: int
    samples: int
    mean_reward: float
    mean_realism: float
    mean_probability_delta: float
    constraint_violation_rate: float
    threshold_evasion_rate: float


def run_host_aware_red_agent_training(
    *,
    checkpoint_path: str,
    host_npz: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str | None = None,
    positive_only: bool = True,
    fpr_budget: float | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Train a mutation policy against a frozen host-aware detector."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    arrays = load_host_windows_npz(host_npz)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    detector = FrozenDetectorAdapter(checkpoint_path, device=device_obj, fpr_budget=fpr_budget)
    if detector.model_type != "host_aware_domain_adaptive_transformer":
        raise ValueError(
            "red_agent_host_aware_train requires a host-aware detector checkpoint; "
            f"got {detector.model_type}"
        )

    train_indices = _select_training_indices(arrays, positive_only=positive_only, max_samples=max_samples)
    if len(train_indices) == 0:
        raise ValueError("No host-aware samples available after filtering")

    policy = MutationPolicyNetwork(input_dim=arrays.feature_dim, device=str(device_obj))
    trainer = PPOTrainer(policy, learning_rate=learning_rate)
    orchestrator = RedAgentOrchestrator(feature_names=list(arrays.feature_names))

    log_rows: list[dict[str, Any]] = []
    total_reward_sum = 0.0
    total_realism_sum = 0.0
    total_probability_delta = 0.0
    total_constraint_violations = 0
    total_threshold_evasions = 0
    total_samples = 0
    total_batches = 0

    print("=" * 72)
    print("RED-AGENT HOST-AWARE PPO TRAINING")
    print("=" * 72)
    print(f"  Detector: {checkpoint_path}")
    print(f"  Host windows: {host_npz}")
    print(f"  Samples: {len(train_indices)}")
    print(f"  Threshold: {detector.threshold:.6f}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device_obj}")

    for epoch in range(epochs):
        shuffled = np.random.permutation(train_indices)
        epoch_reward_sum = 0.0
        epoch_realism_sum = 0.0
        epoch_probability_delta = 0.0
        epoch_constraint_violations = 0
        epoch_threshold_evasions = 0
        epoch_samples = 0

        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start : start + batch_size]
            batch_sessions = arrays.current_sessions[batch_indices]
            batch_masks = arrays.current_masks[batch_indices]

            session_tensor = torch.tensor(batch_sessions, dtype=torch.float32, device=device_obj)
            mutation_logits, severity_mean, _ = policy(session_tensor)
            mutation_probs = torch.softmax(mutation_logits, dim=-1)
            mutation_dist = Categorical(probs=mutation_probs)
            severity_std = torch.clamp(policy.severity_std, min=0.01)
            severity_dist = Normal(severity_mean, severity_std)

            mutation_indices = mutation_dist.sample()
            severities = torch.clamp(severity_dist.sample(), 0.0, 1.0)
            severities = torch.where(mutation_indices == NONE_ACTION_INDEX, torch.zeros_like(severities), severities)
            log_probs = mutation_dist.log_prob(mutation_indices) + severity_dist.log_prob(severities)

            action_pairs = list(zip(mutation_indices.tolist(), severities.tolist()))
            eval_results: list[EvaluationResult] = []
            for local_pos, sample_index in enumerate(batch_indices.tolist()):
                mutation_index = int(mutation_indices[local_pos].item())
                severity = float(severities[local_pos].item())
                result = _evaluate_one_window(
                    arrays=arrays,
                    sample_index=sample_index,
                    mutation_index=mutation_index,
                    severity=severity,
                    orchestrator=orchestrator,
                    detector=detector,
                )
                eval_results.append(result)

            rewards = np.asarray([result.reward for result in eval_results], dtype=np.float32)
            trainer.update(
                session_tensor,
                action_pairs,
                rewards,
                log_probs.detach().cpu(),
                num_epochs=2,
                batch_size=min(batch_size, len(eval_results)),
            )

            epoch_reward_sum += float(np.sum(rewards))
            epoch_realism_sum += float(sum(result.mutation_result.realism_score for result in eval_results))
            epoch_probability_delta += float(
                sum(result.detector_confidence_original - result.detector_confidence_mutated for result in eval_results)
            )
            epoch_constraint_violations += sum(
                len(result.mutation_result.constraint_violations) for result in eval_results
            )
            epoch_threshold_evasions += sum(
                1
                for result in eval_results
                if result.detector_confidence_original >= detector.threshold
                and result.detector_confidence_mutated < detector.threshold
            )
            epoch_samples += len(eval_results)
            total_batches += 1

            for result, sample_index, action_pair in zip(eval_results, batch_indices.tolist(), action_pairs):
                log_rows.append(_serialize_result(result, epoch=epoch + 1, sample_index=sample_index, action_pair=action_pair))

        total_reward_sum += epoch_reward_sum
        total_realism_sum += epoch_realism_sum
        total_probability_delta += epoch_probability_delta
        total_constraint_violations += epoch_constraint_violations
        total_threshold_evasions += epoch_threshold_evasions
        total_samples += epoch_samples

        epoch_summary = {
            "epoch": int(epoch + 1),
            "samples": int(epoch_samples),
            "mean_reward": epoch_reward_sum / max(epoch_samples, 1),
            "mean_realism": epoch_realism_sum / max(epoch_samples, 1),
            "mean_probability_delta": epoch_probability_delta / max(epoch_samples, 1),
            "constraint_violation_rate": epoch_constraint_violations / max(epoch_samples, 1),
            "threshold_evasion_rate": epoch_threshold_evasions / max(epoch_samples, 1),
        }
        print(
            f"[epoch {epoch + 1}/{epochs}] reward={epoch_summary['mean_reward']:.4f} "
            f"p_delta={epoch_summary['mean_probability_delta']:.4f} "
            f"threshold_evasion={epoch_summary['threshold_evasion_rate']:.4f}"
        )
        (output_path / f"epoch_{epoch + 1:03d}.json").write_text(
            json.dumps(epoch_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    policy_path = output_path / "host_aware_phase1_policy.pth"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "input_dim": policy.input_dim,
            "hidden_dim": policy.hidden_dim,
            "num_mutations": policy.num_mutations,
            "mutation_types": MUTATION_TYPES,
            "detector_checkpoint": checkpoint_path,
            "detector_model_type": detector.model_type,
            "detector_threshold": detector.threshold,
            "host_npz": host_npz,
            "feature_names": list(arrays.feature_names),
            "host_feature_names": list(arrays.host_feature_names),
            "history_size": arrays.history_size,
        },
        policy_path,
    )

    history_path = output_path / "host_aware_phase1_training_log.jsonl"
    history_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in log_rows) + ("\n" if log_rows else ""),
        encoding="utf-8",
    )

    summary = HostAwareRedAgentSummary(
        epochs=epochs,
        batches=total_batches,
        samples=total_samples,
        mean_reward=total_reward_sum / max(total_samples, 1),
        mean_realism=total_realism_sum / max(total_samples, 1),
        mean_probability_delta=total_probability_delta / max(total_samples, 1),
        constraint_violation_rate=total_constraint_violations / max(total_samples, 1),
        threshold_evasion_rate=total_threshold_evasions / max(total_samples, 1),
    )
    summary_payload = {
        **summary.__dict__,
        "policy_checkpoint": str(policy_path),
        "training_log": str(history_path),
        "detector_checkpoint": checkpoint_path,
        "detector_threshold": detector.threshold,
        "host_npz": host_npz,
        "positive_only": bool(positive_only),
        "fpr_budget": float(fpr_budget) if fpr_budget is not None else None,
        "max_samples": int(max_samples) if max_samples is not None else None,
        "device": str(device_obj),
    }
    summary_path = output_path / "host_aware_phase1_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nPolicy checkpoint: {policy_path}")
    print(f"Training log: {history_path}")
    print(f"Summary: {summary_path}")
    return summary_payload


def _select_training_indices(
    arrays: HostWindowArrays,
    *,
    positive_only: bool,
    max_samples: int | None,
) -> np.ndarray:
    if positive_only and np.any(arrays.labels == 1):
        indices = np.where(arrays.labels == 1)[0]
    else:
        indices = np.arange(len(arrays.labels))
    if max_samples is not None and max_samples > 0:
        indices = indices[:max_samples]
    return indices.astype(np.int64)


def _evaluate_one_window(
    *,
    arrays: HostWindowArrays,
    sample_index: int,
    mutation_index: int,
    severity: float,
    orchestrator: RedAgentOrchestrator,
    detector: FrozenDetectorAdapter,
) -> EvaluationResult:
    current = arrays.current_sessions[sample_index]
    current_mask = arrays.current_masks[sample_index]
    mutation_type = MUTATION_TYPES[mutation_index]

    if mutation_type == "none":
        mutation_result = MutationResult(
            original_session=current,
            mutated_session=current.copy(),
            mutation_type="none",
            mutation_metadata={"mutation_type": "none", "severity": 0.0},
            constraint_violations=[],
            realism_score=1.0,
            is_realistic=True,
        )
    else:
        mutation_result = orchestrator.mutate_session(
            current,
            current_mask,
            mutation_type=mutation_type,
            severity=severity,
        )

    original_score = detector.score_host_window(
        current_session=mutation_result.original_session,
        current_mask=current_mask,
        history_sessions=arrays.history_sessions[sample_index],
        history_flow_masks=arrays.history_flow_masks[sample_index],
        history_session_masks=arrays.history_session_masks[sample_index],
        host_features=arrays.host_features[sample_index],
        source=str(arrays.sources[sample_index]),
    )
    mutated_score = detector.score_host_window(
        current_session=mutation_result.mutated_session,
        current_mask=current_mask,
        history_sessions=arrays.history_sessions[sample_index],
        history_flow_masks=arrays.history_flow_masks[sample_index],
        history_session_masks=arrays.history_session_masks[sample_index],
        host_features=arrays.host_features[sample_index],
        source=str(arrays.sources[sample_index]),
    )
    reward, reward_components = orchestrator.reward_engine.compute_reward(
        detector_confidence_original=original_score.probability,
        detector_confidence_mutated=mutated_score.probability,
        realism_score=mutation_result.realism_score,
        functionality_score=0.8,
        stability_score=0.85,
        diversity_bonus=0.0,
        constraint_violations=mutation_result.constraint_violations,
        detection_threshold=detector.threshold,
    )
    return EvaluationResult(
        mutation_result=mutation_result,
        detector_confidence_original=original_score.probability,
        detector_confidence_mutated=mutated_score.probability,
        reward=reward,
        reward_components=reward_components,
    )


def _serialize_result(
    result: EvaluationResult,
    *,
    epoch: int,
    sample_index: int,
    action_pair: tuple[int, float],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "sample_index": int(sample_index),
        "action_index": int(action_pair[0]),
        "action_severity": float(action_pair[1]),
        "mutation_type": result.mutation_result.mutation_type,
        "realism_score": float(result.mutation_result.realism_score),
        "is_realistic": bool(result.mutation_result.is_realistic),
        "constraint_violations": list(result.mutation_result.constraint_violations),
        "c2_probability_original": float(result.detector_confidence_original),
        "c2_probability_mutated": float(result.detector_confidence_mutated),
        "probability_delta": float(result.detector_confidence_original - result.detector_confidence_mutated),
        "reward": float(result.reward),
        "reward_components": {key: float(value) for key, value in result.reward_components.items()},
    }


def _load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Red-Agent PPO against a frozen host-aware detector")
    parser.add_argument("--config", help="Optional JSON config file")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--host_npz", default="data/processed/host_aware_clean_balanced_benchmark/train_host_windows.npz")
    parser.add_argument("--output_dir", default="experiments/red_agent_host_aware_phase1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--fpr_budget", type=float)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--include_benign", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    checkpoint_path = config.get("checkpoint_path", args.checkpoint_path)
    if not checkpoint_path:
        raise ValueError("--checkpoint_path is required unless provided in --config")

    run_host_aware_red_agent_training(
        checkpoint_path=checkpoint_path,
        host_npz=config.get("host_npz", args.host_npz),
        output_dir=config.get("output_dir", args.output_dir),
        epochs=int(config.get("epochs", args.epochs)),
        batch_size=int(config.get("batch_size", args.batch_size)),
        learning_rate=float(config.get("learning_rate", args.learning_rate)),
        seed=int(config.get("seed", args.seed)),
        device=config.get("device", args.device),
        positive_only=not bool(config.get("include_benign", args.include_benign)),
        fpr_budget=config.get("fpr_budget", args.fpr_budget),
        max_samples=config.get("max_samples", args.max_samples),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
