"""PPO (Proximal Policy Optimization) trainer for Red-Agent."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO trainer for mutation policy."""

    def __init__(
        self,
        policy_network: nn.Module,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        """Initialize PPO trainer."""
        self.policy = policy_network
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        logger.info(f"PPO trainer initialized: lr={learning_rate}, gamma={gamma}")

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward sequence [T]
            values: Value estimates [T]
            dones: Done flags [T]

        Returns:
            (advantages, returns)
        """
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - dones[t])
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(
        self,
        sessions: torch.Tensor,
        mutations: List[Tuple[int, float]],
        rewards: np.ndarray,
        old_log_probs: torch.Tensor,
        clip_ratio: float = 0.2,
        num_epochs: int = 3,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        PPO update step.

        Args:
            sessions: Session tensors [N, 20, 12]
            mutations: List of (mutation_type_index, severity) tuples
            rewards: Reward sequence [N]
            old_log_probs: Old log probabilities [N]
            clip_ratio: PPO clip ratio (default 0.2)
            num_epochs: Number of training epochs
            batch_size: Mini-batch size

        Returns:
            Metrics dict
        """
        N = len(rewards)
        if N == 0:
            raise ValueError("PPO update received no rewards")
        if len(mutations) != N:
            raise ValueError("mutations and rewards must have the same length")
        
        # Compute value estimates
        with torch.no_grad():
            _, _, values = self.policy(sessions)
            values = values.squeeze(-1).numpy()

        # Compute advantages
        dones = np.zeros(N)
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=sessions.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=sessions.device)

        # Training loop
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        action_indices = torch.tensor([int(action[0]) for action in mutations], dtype=torch.long, device=sessions.device)
        action_severities = torch.tensor([float(action[1]) for action in mutations], dtype=torch.float32, device=sessions.device)
        old_log_probs = old_log_probs.to(sessions.device).float()

        for epoch in range(num_epochs):
            indices = np.random.permutation(N)

            for start_idx in range(0, N, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                batch_index_tensor = torch.as_tensor(batch_indices, dtype=torch.long, device=sessions.device)
                batch_sessions = sessions[batch_index_tensor]
                batch_advantages = advantages[batch_index_tensor]
                batch_returns = returns[batch_index_tensor]
                batch_old_log_probs = old_log_probs[batch_index_tensor]
                batch_action_indices = action_indices[batch_index_tensor]
                batch_action_severities = action_severities[batch_index_tensor]

                # Forward pass
                mutation_logits, severity_mean, value_preds = self.policy(
                    batch_sessions
                )
                value_preds = value_preds.squeeze(-1)

                mutation_probs = torch.softmax(mutation_logits, dim=-1)
                mutation_dist = Categorical(probs=mutation_probs)
                severity_std = torch.clamp(self.policy.severity_std, min=0.01)
                severity_dist = Normal(severity_mean, severity_std)

                current_log_probs = mutation_dist.log_prob(batch_action_indices)
                current_log_probs = current_log_probs + severity_dist.log_prob(batch_action_severities)
                current_log_probs = torch.clamp(current_log_probs, min=-50.0, max=50.0)

                ratios = torch.exp(current_log_probs - batch_old_log_probs)
                clipped_ratios = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio)
                surrogate_1 = ratios * batch_advantages
                surrogate_2 = clipped_ratios * batch_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Value loss
                value_loss = 0.5 * ((value_preds - batch_returns) ** 2).mean()

                # Entropy bonus
                entropy = mutation_dist.entropy().mean() + severity_dist.entropy().mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1

            num_updates = max(update_count, 1)
        metrics = {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

        logger.info(f"PPO update: {metrics}")
        return metrics
