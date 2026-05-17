"""Feature transforms for extended_v1 session tensors.

Extends the v1 ``FeatureTransformConfig`` to work with 45-dimensional
tensors while preserving full backward compatibility with the 10/12
feature schemas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.feature_config_extended import (
    FEATURE_DIM_EXTENDED,
    FEATURE_INDEX_EXTENDED,
    FEATURE_NAMES_EXTENDED,
    FEATURE_GROUP_SLICES_EXTENDED,
)


# Features that benefit from log-scaling in the extended schema
LOG_SCALE_EXTENDED_DEFAULTS = [
    "orig_bytes", "resp_bytes", "bytes_per_pkt", "duration", "iat",
    "tls_handshake_size", "tls_cert_validity_days",
    "dns_query_count", "dns_unique_domains",
    "iat_mean", "iat_std", "iat_p25", "iat_p50", "iat_p75", "iat_p90",
    "burst_avg_spacing",
]


@dataclass(frozen=True)
class ExtendedFeatureTransformConfig:
    """Transform configuration for extended_v1 schema."""

    log_scale_features: tuple[str, ...] = tuple(LOG_SCALE_EXTENDED_DEFAULTS)
    ablate_features: tuple[str, ...] = ()
    ablate_groups: tuple[str, ...] = ()  # Ablate entire groups: "tls", "dns", "temporal"

    def validate(self) -> None:
        """Check that all named features exist in the extended schema."""
        all_names = set(FEATURE_NAMES_EXTENDED)
        invalid = [
            name for name in self.log_scale_features + self.ablate_features
            if name not in all_names
        ]
        if invalid:
            raise ValueError(f"Unknown feature names in extended transform config: {invalid}")

        valid_groups = set(FEATURE_GROUP_SLICES_EXTENDED.keys())
        invalid_groups = [g for g in self.ablate_groups if g not in valid_groups]
        if invalid_groups:
            raise ValueError(
                f"Unknown feature groups: {invalid_groups}. Valid: {sorted(valid_groups)}"
            )

    def to_checkpoint_dict(self) -> dict:
        return {
            "log_scale_features": list(self.log_scale_features),
            "ablate_features": list(self.ablate_features),
            "ablate_groups": list(self.ablate_groups),
            "schema": "extended_v1",
        }

    @classmethod
    def from_checkpoint_dict(cls, payload: dict | None) -> "ExtendedFeatureTransformConfig":
        if payload is None:
            return cls()
        return cls(
            log_scale_features=tuple(payload.get("log_scale_features", LOG_SCALE_EXTENDED_DEFAULTS)),
            ablate_features=tuple(payload.get("ablate_features", ())),
            ablate_groups=tuple(payload.get("ablate_groups", ())),
        )


def apply_extended_feature_transforms(
    sequences: np.ndarray,
    masks: np.ndarray,
    config: ExtendedFeatureTransformConfig | None = None,
) -> np.ndarray:
    """Apply log-scaling and ablation to extended_v1 tensors.

    Parameters
    ----------
    sequences : ``(N, SESSION_LEN, FEATURE_DIM_EXTENDED)`` float32
    masks : ``(N, SESSION_LEN)`` bool (True = real flow)
    config : Transform configuration.

    Returns
    -------
    Transformed copy of the input tensors.
    """
    if config is None:
        config = ExtendedFeatureTransformConfig()
    config.validate()

    transformed = sequences.astype(np.float32, copy=True)
    valid = masks.astype(bool)

    # Log-scale specified features
    for feature in config.log_scale_features:
        if feature in FEATURE_INDEX_EXTENDED:
            idx = FEATURE_INDEX_EXTENDED[feature]
            values = transformed[:, :, idx]
            values = np.where(valid, values, 0.0)
            values = np.sign(values) * np.log1p(np.abs(values))
            transformed[:, :, idx] = values.astype(np.float32)

    # Ablate individual features
    for feature in config.ablate_features:
        if feature in FEATURE_INDEX_EXTENDED:
            idx = FEATURE_INDEX_EXTENDED[feature]
            transformed[:, :, idx] = 0.0

    # Ablate entire feature groups
    for group in config.ablate_groups:
        if group in FEATURE_GROUP_SLICES_EXTENDED:
            slc = FEATURE_GROUP_SLICES_EXTENDED[group]
            transformed[:, :, slc] = 0.0

    # Zero out padding positions
    transformed[~valid] = 0.0
    return transformed
