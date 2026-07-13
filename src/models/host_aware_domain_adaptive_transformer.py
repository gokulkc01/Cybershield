"""Host-aware domain-adaptive Transformer for modern C2 research.

The model reuses the existing session Transformer backbone, then adds a
causal host-history encoder over previous sessions from the same host.
It keeps domain-specific heads/calibration so each environment can preserve
its own operating point while sharing the session and host encoders.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.domain_adaptive_transformer import TransformerBackbone


@dataclass(frozen=True)
class HostAwareModelConfig:
    domains: tuple[str, ...]
    feature_dim: int
    host_feature_dim: int
    seq_len: int = 20
    history_size: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    host_num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2
    use_derivative_features: bool = False


class HostAwareDomainHead(nn.Module):
    """Domain-specific output heads for session, host, and fused C2 risk."""

    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.session_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.host_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.calibration_scale = nn.Parameter(torch.tensor(1.0))
        self.calibration_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, session_embedding: torch.Tensor, host_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        session_logit = self.session_classifier(session_embedding).squeeze(-1)
        host_logit = self.host_classifier(host_embedding).squeeze(-1)
        fused = torch.cat([session_embedding, host_embedding], dim=-1)
        c2_logit = self.fusion_classifier(fused).squeeze(-1)
        c2_logit = c2_logit * self.calibration_scale + self.calibration_bias
        return {
            "session_logit": session_logit,
            "host_logit": host_logit,
            "c2_logit": c2_logit,
        }


class HostAwareDomainAdaptiveTransformer(nn.Module):
    """Shared session/host encoders with domain-specific calibrated heads."""

    def __init__(
        self,
        domains: list[str],
        feature_dim: int,
        host_feature_dim: int,
        seq_len: int = 20,
        history_size: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        host_num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        use_derivative_features: bool = False,
    ):
        super().__init__()
        if not domains:
            raise ValueError("At least one domain is required")
        if history_size < 1:
            raise ValueError("history_size must be >= 1")

        self.config = HostAwareModelConfig(
            domains=tuple(domains),
            feature_dim=feature_dim,
            host_feature_dim=host_feature_dim,
            seq_len=seq_len,
            history_size=history_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            host_num_layers=host_num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_derivative_features=use_derivative_features,
        )
        self.domains = list(domains)
        self.domain_to_index = {domain: idx for idx, domain in enumerate(self.domains)}
        self.feature_dim = feature_dim
        self.host_feature_dim = host_feature_dim
        self.seq_len = seq_len
        self.history_size = history_size
        self.d_model = d_model

        self.session_backbone = TransformerBackbone(
            feature_dim=feature_dim,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_derivative_features=use_derivative_features,
        )

        self.host_feature_projection = nn.Sequential(
            nn.Linear(host_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.host_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.host_pos_encoder = nn.Parameter(torch.zeros(1, history_size + 1, d_model))
        host_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.host_encoder = nn.TransformerEncoder(host_layer, num_layers=host_num_layers)
        self.host_norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict(
            {domain: HostAwareDomainHead(d_model=d_model, dropout=dropout) for domain in self.domains}
        )
        self._init_host_weights()

    def _init_host_weights(self) -> None:
        nn.init.normal_(self.host_cls_token, std=0.02)
        nn.init.normal_(self.host_pos_encoder, std=0.02)

    def encode_current_session(
        self,
        current_sessions: torch.Tensor,
        current_padding_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.session_backbone.forward_features(current_sessions, current_padding_masks)

    def encode_host_history(
        self,
        history_sessions: torch.Tensor,
        history_flow_padding_masks: torch.Tensor,
        history_session_masks: torch.Tensor,
        host_features: torch.Tensor,
    ) -> torch.Tensor:
        if history_sessions.ndim != 4:
            raise ValueError(
                "Expected history_sessions with shape (B, H, T, D), "
                f"got {tuple(history_sessions.shape)}"
            )
        batch_size, history_size, seq_len, feature_dim = history_sessions.shape
        if history_size != self.history_size:
            raise ValueError(f"Expected history_size={self.history_size}, got {history_size}")
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        if feature_dim != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feature_dim}")
        if history_flow_padding_masks.shape != (batch_size, history_size, seq_len):
            raise ValueError("history_flow_padding_masks shape does not match history_sessions")
        if history_session_masks.shape != (batch_size, history_size):
            raise ValueError("history_session_masks must have shape (B, H)")
        if host_features.shape != (batch_size, self.host_feature_dim):
            raise ValueError(
                f"Expected host_features shape (B, {self.host_feature_dim}), got {tuple(host_features.shape)}"
            )

        flat_history = history_sessions.reshape(batch_size * history_size, seq_len, feature_dim)
        flat_masks = history_flow_padding_masks.reshape(batch_size * history_size, seq_len)
        flat_embeddings = self.session_backbone.forward_features(flat_history, flat_masks)
        history_embeddings = flat_embeddings.reshape(batch_size, history_size, self.d_model)

        host_cls = self.host_cls_token.expand(batch_size, -1, -1)
        host_cls = host_cls + self.host_feature_projection(host_features).unsqueeze(1)
        tokens = torch.cat([host_cls, history_embeddings], dim=1)
        tokens = tokens + self.host_pos_encoder

        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=tokens.device)
        history_padding = ~history_session_masks.to(dtype=torch.bool, device=tokens.device)
        padding_mask = torch.cat([cls_padding, history_padding], dim=1)

        encoded = self.host_encoder(tokens, src_key_padding_mask=padding_mask)
        encoded = self.host_norm(encoded)
        return encoded[:, 0, :]

    def forward(
        self,
        current_sessions: torch.Tensor,
        current_padding_masks: torch.Tensor,
        history_sessions: torch.Tensor,
        history_flow_padding_masks: torch.Tensor,
        history_session_masks: torch.Tensor,
        host_features: torch.Tensor,
        domain_ids: torch.Tensor,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if domain_ids.ndim != 1:
            raise ValueError(f"Expected domain_ids with shape (B,), got {tuple(domain_ids.shape)}")

        session_embedding = self.encode_current_session(current_sessions, current_padding_masks)
        host_embedding = self.encode_host_history(
            history_sessions,
            history_flow_padding_masks,
            history_session_masks,
            host_features,
        )

        batch_size = session_embedding.size(0)
        c2_logits = torch.empty(batch_size, device=session_embedding.device, dtype=session_embedding.dtype)
        session_logits = torch.empty_like(c2_logits)
        host_logits = torch.empty_like(c2_logits)

        for domain_name, domain_index in self.domain_to_index.items():
            domain_mask = domain_ids == domain_index
            if domain_mask.any():
                outputs = self.heads[domain_name](
                    session_embedding[domain_mask],
                    host_embedding[domain_mask],
                )
                c2_logits[domain_mask] = outputs["c2_logit"]
                session_logits[domain_mask] = outputs["session_logit"]
                host_logits[domain_mask] = outputs["host_logit"]

        if return_components:
            return {
                "c2_logit": c2_logits,
                "session_logit": session_logits,
                "host_logit": host_logits,
            }
        return c2_logits

