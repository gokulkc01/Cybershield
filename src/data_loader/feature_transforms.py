"""Configurable feature preprocessing for CyberShield session tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.feature_config import FEATURE_INDEX, FEATURE_NAMES


LOG_SCALE_DEFAULT_FEATURES = ["orig_bytes", "resp_bytes", "bytes_per_pkt", "duration", "iat"]


@dataclass(frozen=True)
class FeatureTransformConfig:
    log_scale_features: tuple[str, ...] = tuple(LOG_SCALE_DEFAULT_FEATURES)
    ablate_features: tuple[str, ...] = ()

    def validate(self) -> None:
        invalid = [name for name in self.log_scale_features + self.ablate_features if name not in FEATURE_INDEX]
        if invalid:
            raise ValueError(f"Unknown feature names in transform config: {invalid}")

    def to_checkpoint_dict(self) -> dict:
        return {
            "log_scale_features": list(self.log_scale_features),
            "ablate_features": list(self.ablate_features),
        }

    @classmethod
    def from_checkpoint_dict(cls, payload: dict | None) -> "FeatureTransformConfig":
        if payload is None:
            return cls()
        return cls(
            log_scale_features=tuple(payload.get("log_scale_features", LOG_SCALE_DEFAULT_FEATURES)),
            ablate_features=tuple(payload.get("ablate_features", ())),
        )


def apply_feature_transforms(
    sequences: np.ndarray,
    masks: np.ndarray,
    config: FeatureTransformConfig | None,
) -> np.ndarray:
    if config is None:
        config = FeatureTransformConfig()
    config.validate()

    transformed = sequences.astype(np.float32, copy=True)
    valid = masks.astype(bool)

    for feature in config.log_scale_features:
        idx = FEATURE_INDEX[feature]
        values = transformed[:, :, idx]
        values = np.where(valid, values, 0.0)
        values = np.sign(values) * np.log1p(np.abs(values))
        transformed[:, :, idx] = values.astype(np.float32)

    for feature in config.ablate_features:
        idx = FEATURE_INDEX[feature]
        transformed[:, :, idx] = 0.0

    transformed[~valid] = 0.0
    return transformed
