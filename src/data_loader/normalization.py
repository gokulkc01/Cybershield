"""Train-only feature normalization for session tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def transform(self, sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
        transformed = sequences.astype(np.float32, copy=True)
        valid = masks.astype(bool)
        transformed = ((transformed - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)).astype(np.float32)
        transformed[~valid] = 0.0
        return transformed

    def to_checkpoint_dict(self) -> dict:
        return {"mean": self.mean.astype(np.float32), "std": self.std.astype(np.float32), "eps": self.eps}

    @classmethod
    def from_checkpoint_dict(cls, payload: dict) -> "FeatureNormalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            eps=float(payload.get("eps", 1e-6)),
        )


def fit_feature_normalizer(sequences: np.ndarray, masks: np.ndarray, eps: float = 1e-6) -> FeatureNormalizer:
    valid_rows = masks.reshape(-1)
    flattened = sequences.reshape(-1, sequences.shape[-1])[valid_rows]

    if flattened.size == 0:
        raise ValueError("Cannot fit normalizer: no real flows available.")

    mean = flattened.mean(axis=0).astype(np.float32)
    std = flattened.std(axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return FeatureNormalizer(mean=mean, std=std, eps=eps)
