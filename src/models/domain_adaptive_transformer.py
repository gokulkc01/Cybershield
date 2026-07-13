"""Domain-adaptive transformer with shared backbone and domain-specific heads.

This model keeps a single shared transformer encoder and adds a separate
classification head plus a learnable calibration layer for each domain.
That makes it suitable for continual or multi-task training where the goal
is to preserve behavior on multiple domains at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.features.feature_config import DERIVATIVE_FEATURE_NAMES, IAT_IDX, ORIG_BYTES_IDX


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        feature_dim: int = 12,
        seq_len: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        use_derivative_features: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.use_derivative_features = use_derivative_features
        effective_feature_dim = feature_dim + (len(DERIVATIVE_FEATURE_NAMES) if use_derivative_features else 0)

        self.input_projection = nn.Linear(effective_feature_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1 and "pos_encoder" not in name and "cls_token" not in name:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.pos_encoder, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def _append_derivative_features(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        real_mask = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        x = x * real_mask

        iat = x[:, :, IAT_IDX:IAT_IDX + 1]
        orig_bytes = x[:, :, ORIG_BYTES_IDX:ORIG_BYTES_IDX + 1]

        iat_delta = torch.zeros_like(iat)
        byte_delta = torch.zeros_like(orig_bytes)

        valid_transitions = real_mask[:, 1:, :] * real_mask[:, :-1, :]
        iat_delta[:, 1:, :] = (iat[:, 1:, :] - iat[:, :-1, :]) * valid_transitions
        byte_delta[:, 1:, :] = (orig_bytes[:, 1:, :] - orig_bytes[:, :-1, :]) * valid_transitions

        return torch.cat((x, iat_delta, byte_delta), dim=-1)

    def forward_features(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, {self.seq_len}, {self.feature_dim}), got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        if x.shape[2] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {x.shape[2]}")

        if self.use_derivative_features:
            x = self._append_derivative_features(x, padding_mask)

        batch_size = x.size(0)
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        padding_mask = torch.cat((cls_mask, padding_mask), dim=1)

        x = x + self.pos_encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        return x[:, 0, :]


class DomainHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.calibration_scale = nn.Parameter(torch.tensor(1.0))
        self.calibration_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(features).squeeze(-1)
        return logits * self.calibration_scale + self.calibration_bias


class DomainAdaptiveC2Transformer(nn.Module):
    def __init__(
        self,
        domains: list[str],
        feature_dim: int = 12,
        seq_len: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        use_derivative_features: bool = False,
    ):
        super().__init__()
        if not domains:
            raise ValueError("At least one domain is required")

        self.domains = list(domains)
        self.domain_to_index = {domain: index for index, domain in enumerate(self.domains)}
        self.backbone = TransformerBackbone(
            feature_dim=feature_dim,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_derivative_features=use_derivative_features,
        )
        self.heads = nn.ModuleDict({domain: DomainHead(d_model=d_model, dropout=dropout) for domain in self.domains})

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if domain_ids.ndim != 1:
            raise ValueError(f"Expected domain_ids to have shape (B,), got {tuple(domain_ids.shape)}")

        features = self.backbone.forward_features(x, padding_mask)
        logits = torch.empty(features.size(0), device=features.device, dtype=features.dtype)

        for domain_name, domain_index in self.domain_to_index.items():
            domain_mask = domain_ids == domain_index
            if domain_mask.any():
                logits[domain_mask] = self.heads[domain_name](features[domain_mask])

        return logits
