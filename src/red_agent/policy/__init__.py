"""PPO Policy network for mutation control."""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class MutationPolicyNetwork(nn.Module):
    """
    Policy network that selects mutations and severity levels.
    
    Input: session tensor (20, D)
    Output: mutation type logits + severity action
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 256,
        num_mutations: int = 5,  # timing, flow, tls, composite, none
        device: str = "cpu",
    ):
        """
        Initialize policy network.

        Args:
            input_dim: Feature dimension (typically 12)
            hidden_dim: Hidden layer dimension
            num_mutations: Number of mutation types
            device: torch device
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mutations = num_mutations
        self.device = device

        # Session encoder: (20, D) → embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 20, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Mutation type policy head
        self.mutation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_mutations),  # Logits
        )

        # Severity (continuous action) policy head
        self.severity_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output [0, 1]
        )

        self.severity_std = nn.Parameter(torch.tensor(0.2))

        # Value network (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.to(device)

    def forward(
        self,
        session: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            session: Session tensor (batch_size, 20, D) or (20, D)

        Returns:
            (mutation_logits, severity_mean, value_estimate)
        """
        # Flatten session
        if session.dim() == 2:
            session = session.unsqueeze(0)  # Add batch dimension

        batch_size = session.shape[0]
        session_flat = session.reshape(batch_size, -1)

        # Encode
        embedding = self.encoder(session_flat)

        # Policy heads
        mutation_logits = self.mutation_head(embedding)
        severity_mean = self.severity_mean_head(embedding).squeeze(-1)

        # Value estimate
        value = self.value_head(embedding).squeeze(-1)

        return mutation_logits, severity_mean, value

    def get_action(
        self,
        session: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Sample action from policy.

        Args:
            session: Session tensor
            deterministic: If True, use argmax/mean instead of sampling

        Returns:
            (mutation_type, severity, log_probability)
        """
        with torch.no_grad():
            mutation_logits, severity_mean, _ = self.forward(session)

            # Sample mutation type
            probs = torch.softmax(mutation_logits, dim=-1)
            if deterministic:
                mutation_type = torch.argmax(probs, dim=-1).item()
                log_prob = torch.log(probs[0, mutation_type])
            else:
                dist = torch.distributions.Categorical(probs)
                mutation_type = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(mutation_type))

            # Sample severity
            if deterministic:
                severity = severity_mean.item()
            else:
                std = torch.clamp(self.severity_std, min=0.01)
                severity_dist = torch.distributions.Normal(
                    severity_mean, std
                )
                severity = torch.clamp(severity_dist.sample(), 0.0, 1.0).item()
                log_prob = log_prob + severity_dist.log_prob(torch.tensor(severity))

            return mutation_type, severity, log_prob.item()
