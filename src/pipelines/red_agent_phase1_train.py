"""Phase 1 Red-Agent training pipeline.

This trains the Red-Agent policy against a frozen CyberShield detector using
realistic behavioral mutations and PPO updates.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.distributions import Categorical, Normal

from src.data_loader.extended_feature_transforms import (
    ExtendedFeatureTransformConfig,
    apply_extended_feature_transforms,
)
from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.normalization import FeatureNormalizer
from src.data_loader.npz_utils import detect_npz_schema, load_session_npz
from src.features.feature_config import FEATURE_NAMES
from src.features.feature_config_extended import FEATURE_SCHEMA_VERSION as EXTENDED_SCHEMA_VERSION
from src.models.transformer import C2Transformer
from src.red_agent.orchestrator import RedAgentOrchestrator
from src.red_agent.policy import MutationPolicyNetwork
from src.red_agent.training import PPOTrainer


MUTATION_TYPES = ["timing", "flow", "tls", "composite", "timing"]
NONE_ACTION_INDEX = 4


@dataclass(frozen=True)
class Phase1RunSummary:
    epochs: int
    batches: int
    mean_reward: float
    mean_realism: float
    mean_confidence_delta: float
    constraint_violation_rate: float
    evasion_rate: float


def _load_transform_config(checkpoint: dict) -> FeatureTransformConfig | ExtendedFeatureTransformConfig:
    payload = checkpoint.get("feature_transform_config")
    if isinstance(payload, dict) and payload.get("schema") == EXTENDED_SCHEMA_VERSION:
        return ExtendedFeatureTransformConfig.from_checkpoint_dict(payload)
    return FeatureTransformConfig.from_checkpoint_dict(payload)


def _apply_transforms(
    sequence: np.ndarray,
    mask: np.ndarray,
    config: FeatureTransformConfig | ExtendedFeatureTransformConfig,
    schema_version: str,
) -> np.ndarray:
    if schema_version == EXTENDED_SCHEMA_VERSION:
        return apply_extended_feature_transforms(sequence, mask, config)  # type: ignore[arg-type]
    return apply_feature_transforms(sequence, mask, config, feature_names=tuple(FEATURE_NAMES))


def _load_frozen_detector(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    schema_version = str(checkpoint.get("schema_version", ""))
    feature_dim = int(checkpoint.get("feature_dim", len(FEATURE_NAMES)))
    session_len = int(checkpoint.get("session_len", 20))

    model = C2Transformer(
        feature_dim=feature_dim,
        seq_len=session_len,
        use_derivative_features=bool(checkpoint.get("use_derivative_features", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform_cfg = _load_transform_config(checkpoint)
    normalizer = None
    if checkpoint.get("normalize_features", False) and checkpoint.get("feature_normalizer"):
        normalizer = FeatureNormalizer.from_checkpoint_dict(checkpoint["feature_normalizer"])

    def infer_confidence(session: np.ndarray) -> float:
        sequence = np.asarray(session, dtype=np.float32)[np.newaxis, ...]
        real_mask = np.any(sequence != 0.0, axis=2)
        processed = _apply_transforms(sequence, real_mask, transform_cfg, schema_version)
        if normalizer is not None:
            processed = normalizer.transform(processed, real_mask)

        features = torch.from_numpy(processed).to(device)
        padding_mask = torch.from_numpy(~real_mask).to(device)
        with torch.no_grad():
            logits = model(features, padding_mask)
            probability = torch.sigmoid(logits).item()
        return float(max(probability, 1.0 - probability))

    return infer_confidence, checkpoint


def _serialize_mutation_results(results) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for result in results:
        payload.append(
            {
                "mutation_type": result.mutation_result.mutation_type,
                "severity": float(result.mutation_result.mutation_metadata.get("severity", 0.0)),
                "realism_score": float(result.mutation_result.realism_score),
                "is_realistic": bool(result.mutation_result.is_realistic),
                "constraint_violations": list(result.mutation_result.constraint_violations),
                "detector_confidence_original": float(result.detector_confidence_original),
                "detector_confidence_mutated": float(result.detector_confidence_mutated),
                "reward": float(result.reward),
                "reward_components": {k: float(v) for k, v in result.reward_components.items()},
            }
        )
    return payload


def run_phase1_training(
    checkpoint_path: str,
    baseline_npz: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str | None = None,
    positive_only: bool = True,
) -> Dict[str, Any]:
    """Train Red-Agent against a frozen detector using PPO."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")
    if not Path(baseline_npz).exists():
        raise FileNotFoundError(f"Training NPZ not found: {baseline_npz}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    schema = detect_npz_schema(baseline_npz)
    expected_feature_names = tuple(schema["feature_names"]) if schema["feature_names"] else tuple(FEATURE_NAMES)
    expected_session_len = int(schema["session_len"]) if schema["session_len"] else 20
    sessions, labels, masks = load_session_npz(
        baseline_npz,
        expected_feature_names=expected_feature_names,
        expected_session_len=expected_session_len,
    )
    if positive_only and np.any(labels == 1):
        train_indices = np.where(labels == 1)[0]
    else:
        train_indices = np.arange(len(labels))

    if len(train_indices) == 0:
        raise ValueError("No training samples available after filtering")

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    detector_infer, detector_checkpoint = _load_frozen_detector(checkpoint_path, device_obj)

    policy = MutationPolicyNetwork(input_dim=sessions.shape[-1], device=str(device_obj))
    trainer = PPOTrainer(policy, learning_rate=learning_rate)
    orchestrator = RedAgentOrchestrator(feature_names=list(FEATURE_NAMES))

    log_rows: List[Dict[str, Any]] = []
    total_reward_sum = 0.0
    total_realism_sum = 0.0
    total_confidence_delta = 0.0
    total_constraint_violations = 0
    total_samples = 0
    total_evasions = 0
    total_batches = 0

    print("=" * 72)
    print("RED-AGENT PHASE 1: FROZEN DETECTOR TRAINING")
    print("=" * 72)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Training NPZ: {baseline_npz}")
    print(f"  Samples: {len(train_indices)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device_obj}")

    for epoch in range(epochs):
        shuffled = np.random.permutation(train_indices)
        epoch_reward_sum = 0.0
        epoch_realism_sum = 0.0
        epoch_confidence_delta = 0.0
        epoch_constraint_violations = 0
        epoch_samples = 0
        epoch_evasions = 0

        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start : start + batch_size]
            batch_sessions = sessions[batch_indices]
            batch_masks = masks[batch_indices]

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

            mutation_types = [MUTATION_TYPES[int(idx)] for idx in mutation_indices.tolist()]
            severity_values = [0.0 if int(idx) == NONE_ACTION_INDEX else float(value) for idx, value in zip(mutation_indices.tolist(), severities.tolist())]
            action_pairs = list(zip(mutation_indices.tolist(), severity_values))

            eval_results = orchestrator.batch_evaluate_mutations(
                sessions=batch_sessions,
                masks=batch_masks,
                detector_inference_fn=detector_infer,
                mutation_types=mutation_types,
                severities=severity_values,
            )

            if not eval_results:
                continue

            rewards = np.array([result.reward for result in eval_results], dtype=np.float32)
            old_log_probs = log_probs.detach().cpu()

            trainer.update(
                session_tensor,
                action_pairs,
                rewards,
                old_log_probs,
                num_epochs=2,
                batch_size=min(batch_size, len(eval_results)),
            )

            epoch_reward_sum += float(np.sum(rewards))
            epoch_realism_sum += float(sum(result.mutation_result.realism_score for result in eval_results))
            epoch_confidence_delta += float(
                sum(result.detector_confidence_original - result.detector_confidence_mutated for result in eval_results)
            )
            epoch_constraint_violations += sum(len(result.mutation_result.constraint_violations) for result in eval_results)
            epoch_evasions += sum(1 for result in eval_results if result.detector_confidence_mutated < result.detector_confidence_original)
            epoch_samples += len(eval_results)
            total_batches += 1

            serialized = _serialize_mutation_results(eval_results)
            for record, batch_index in zip(serialized, batch_indices.tolist()):
                record["sample_index"] = int(batch_index)
                record["epoch"] = int(epoch)
                log_rows.append(record)

        total_reward_sum += epoch_reward_sum
        total_realism_sum += epoch_realism_sum
        total_confidence_delta += epoch_confidence_delta
        total_constraint_violations += epoch_constraint_violations
        total_samples += epoch_samples
        total_evasions += epoch_evasions

        epoch_summary = {
            "epoch": epoch + 1,
            "samples": epoch_samples,
            "mean_reward": epoch_reward_sum / max(epoch_samples, 1),
            "mean_realism": epoch_realism_sum / max(epoch_samples, 1),
            "mean_confidence_delta": epoch_confidence_delta / max(epoch_samples, 1),
            "constraint_violation_rate": epoch_constraint_violations / max(epoch_samples, 1),
            "evasion_rate": epoch_evasions / max(epoch_samples, 1),
        }
        print(
            f"[epoch {epoch + 1}/{epochs}] reward={epoch_summary['mean_reward']:.4f} "
            f"realism={epoch_summary['mean_realism']:.4f} evasion={epoch_summary['evasion_rate']:.4f}"
        )
        (output_path / f"epoch_{epoch + 1:03d}.json").write_text(json.dumps(epoch_summary, indent=2), encoding="utf-8")

    policy_path = output_path / "phase1_policy.pth"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "input_dim": policy.input_dim,
            "hidden_dim": policy.hidden_dim,
            "num_mutations": policy.num_mutations,
            "checkpoint_path": checkpoint_path,
            "baseline_npz": baseline_npz,
            "feature_names": list(FEATURE_NAMES),
        },
        policy_path,
    )

    history_path = output_path / "phase1_training_log.jsonl"
    history_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in log_rows) + ("\n" if log_rows else ""), encoding="utf-8")

    summary = Phase1RunSummary(
        epochs=epochs,
        batches=total_batches,
        mean_reward=total_reward_sum / max(total_samples, 1),
        mean_realism=total_realism_sum / max(total_samples, 1),
        mean_confidence_delta=total_confidence_delta / max(total_samples, 1),
        constraint_violation_rate=total_constraint_violations / max(total_samples, 1),
        evasion_rate=total_evasions / max(total_samples, 1),
    )

    summary_path = output_path / "phase1_summary.json"
    summary_payload = {
        **summary.__dict__,
        "policy_checkpoint": str(policy_path),
        "training_log": str(history_path),
        "detector_checkpoint": checkpoint_path,
        "baseline_npz": baseline_npz,
        "device": str(device_obj),
        "frozen_detector_metadata": {
            "optimal_threshold": float(detector_checkpoint.get("optimal_threshold", 0.5)),
            "use_derivative_features": bool(detector_checkpoint.get("use_derivative_features", False)),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n✓ Policy checkpoint: {policy_path}")
    print(f"✓ Training log: {history_path}")
    print(f"✓ Summary: {summary_path}")

    return summary_payload


def _load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Red-Agent Phase 1 training pipeline")
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    args = parser.parse_args()

    config = _load_config(args.config)
    run_phase1_training(
        checkpoint_path=config["checkpoint_path"],
        baseline_npz=config["baseline_npz"],
        output_dir=config.get("output_dir", "experiments/red_agent_phase1"),
        epochs=int(config.get("epochs", 5)),
        batch_size=int(config.get("batch_size", 32)),
        learning_rate=float(config.get("learning_rate", 3e-4)),
        seed=int(config.get("seed", 42)),
        device=config.get("device"),
        positive_only=bool(config.get("positive_only", True)),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())