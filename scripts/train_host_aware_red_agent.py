"""
CyberShield Red-Agent RL Training Loop for Host-Aware Models
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.data_loader.host_window_dataset import load_host_windows_npz
from src.features.feature_config_extended import FEATURE_NAMES_EXTENDED
from src.red_agent.detector_adapters import FrozenDetectorAdapter
from src.red_agent.orchestrator import RedAgentOrchestrator
from src.red_agent.policy import MutationPolicyNetwork
from src.red_agent.training.ppo_trainer import PPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to host-aware checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to host_windows.npz data")
    args = parser.parse_args()

    print("==================================================")
    print("Host-Aware Red-Agent PPO Training Loop")
    print("==================================================")

    print("\nLoading Host-Aware Detector...")
    adapter = FrozenDetectorAdapter(args.checkpoint)
    threshold = adapter.threshold
    print(f"Loaded {adapter.model_type} with threshold {threshold:.4f}")

    print("\nLoading Data...")
    arrays = load_host_windows_npz(args.data)
    
    # Filter for C2 only for training the adversary
    c2_idx = np.where(arrays.labels == 1)[0]
    if len(c2_idx) == 0:
        print("No C2 samples found in data!")
        return
        
    print(f"Loaded {len(c2_idx)} C2 host windows for adversarial training.")

    policy = MutationPolicyNetwork(input_dim=len(arrays.feature_names), hidden_dim=256, num_mutations=5)
    orchestrator = RedAgentOrchestrator(feature_names=list(arrays.feature_names))
    trainer = PPOTrainer(
        policy_network=policy,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
    )

    episode_stats = []
    sessions_used = []
    
    print("\nStarting PPO loop...")
    for ep in range(args.episodes):
        idx = c2_idx[ep % len(c2_idx)]
        
        current_session = arrays.current_sessions[idx]
        current_mask = arrays.current_masks[idx]
        history = arrays.history_sessions[idx]
        history_mask = arrays.history_flow_masks[idx]
        history_session_mask = arrays.history_session_masks[idx]
        host_feats = arrays.host_features[idx]
        
        sessions_used.append(current_session)
        
        # Detector closure for this specific episode/host context
        # This is the crucial bridge between session-level mutation and host-level context
        def detector_fn(session_to_eval: np.ndarray) -> float:
            score = adapter.score_host_window(
                current_session=session_to_eval,
                current_mask=None,
                history_sessions=history,
                history_flow_masks=history_mask,
                history_session_masks=history_session_mask,
                host_features=host_feats,
            )
            return score.probability

        # Policy forward
        session_tensor = torch.tensor(current_session, dtype=torch.float32)
        mut_type_idx, severity, log_prob = policy.get_action(session_tensor, deterministic=False)
        mut_name = ["timing", "flow", "tls", "composite", "none"][mut_type_idx]
        if mut_name == "none":
            mut_name = "timing"

        # Mutate the session
        mut_result = orchestrator.mutate_session(
            current_session, current_mask, mutation_type=mut_name, severity=severity,
        )

        # Evaluate against the frozen host-aware model
        eval_result = orchestrator.evaluate_mutation(
            mut_result,
            detector_fn,
            detection_threshold=threshold,
        )
        
        evaded = eval_result.detector_confidence_original >= threshold and eval_result.detector_confidence_mutated < threshold

        episode_stats.append({
            "reward": eval_result.reward,
            "mut_type_idx": mut_type_idx,
            "severity": severity,
            "log_prob": log_prob,
            "evaded": evaded,
            "conf_orig": eval_result.detector_confidence_original,
            "conf_mut": eval_result.detector_confidence_mutated,
        })
        
        # PPO Batch Update
        if (ep + 1) % args.batch_size == 0 or (ep + 1) == args.episodes:
            sessions_tensor = torch.tensor(np.stack(sessions_used), dtype=torch.float32)
            rewards = np.array([s["reward"] for s in episode_stats[-len(sessions_used):]], dtype=np.float32)
            mutations = [(s["mut_type_idx"], s["severity"]) for s in episode_stats[-len(sessions_used):]]
            old_log_probs = torch.tensor([s["log_prob"] for s in episode_stats[-len(sessions_used):]], dtype=torch.float32)
            
            metrics = trainer.update(
                sessions=sessions_tensor,
                mutations=mutations,
                rewards=rewards,
                old_log_probs=old_log_probs,
                batch_size=len(sessions_used),
            )
            
            recent_evasion = np.mean([s["evaded"] for s in episode_stats[-len(sessions_used):]]) * 100
            print(f"Batch {ep+1}/{args.episodes} | Avg Reward: {rewards.mean():+.4f} | Evasion Rate: {recent_evasion:5.1f}% | Policy Loss: {metrics['policy_loss']:+.4f}")
            
            sessions_used = []

    final_evasion = np.mean([s["evaded"] for s in episode_stats]) * 100
    print("\n==================================================")
    print("Training Complete!")
    print(f"Total Episodes: {args.episodes}")
    print(f"Overall Evasion Rate against Host-Aware Model: {final_evasion:.1f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
