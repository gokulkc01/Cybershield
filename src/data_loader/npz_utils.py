"""Shared helpers for loading CyberShield session NPZ files safely."""

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np

from src.features.feature_config import FEATURE_DIM, FEATURE_NAMES, SESSION_LEN


def normalize_real_flow_masks(sequences: np.ndarray, masks: np.ndarray | None) -> np.ndarray:
    """Return masks with True meaning a real flow and False meaning padding.

    Historical NPZ files in this repo have used both conventions:
    - True = real flow
    - True = padding

    We normalize against the tensor's zero-padded rows so downstream code has
    one stable contract.
    """
    inferred_real_mask = np.any(sequences != 0.0, axis=2)

    if masks is None:
        return inferred_real_mask

    masks = np.asarray(masks, dtype=bool)
    if masks.shape != inferred_real_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: expected {inferred_real_mask.shape}, got {masks.shape}"
        )

    if np.array_equal(masks, inferred_real_mask):
        return masks
    if np.array_equal(~masks, inferred_real_mask):
        return ~masks

    same_orientation_score = np.mean(masks == inferred_real_mask)
    inverse_orientation_score = np.mean((~masks) == inferred_real_mask)
    return masks if same_orientation_score >= inverse_orientation_score else ~masks


def load_session_npz(
    path: str,
    expected_feature_names: Sequence[str] | None = None,
    expected_session_len: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load sequences, labels, and normalized real-flow masks from an NPZ."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")

    data = np.load(path, allow_pickle=True)

    feature_names = list(expected_feature_names) if expected_feature_names is not None else FEATURE_NAMES
    session_len = int(expected_session_len) if expected_session_len is not None else SESSION_LEN
    feature_dim = len(feature_names)

    if "feature_names" in data:
        saved_feature_names = [str(name) for name in data["feature_names"].tolist()]
        if saved_feature_names != feature_names:
            raise ValueError(
                "Feature schema mismatch between NPZ and current pipeline. "
                f"Expected {feature_names}, got {saved_feature_names}"
            )

    if "X" in data:
        sequences = data["X"].astype(np.float32)
    elif "sessions" in data:
        sequences = data["sessions"].astype(np.float32)
    else:
        raise KeyError(f"NPZ must contain 'X' or 'sessions'. Found keys: {list(data.keys())}")

    if "y" in data:
        labels = data["y"].astype(np.int64)
    elif "labels" in data:
        labels = data["labels"].astype(np.int64)
    else:
        raise KeyError(f"NPZ must contain 'y' or 'labels'. Found keys: {list(data.keys())}")

    if sequences.ndim != 3:
        raise ValueError(f"Expected X to be 3D (N, {session_len}, {feature_dim}), got {sequences.shape}")
    if sequences.shape[1] != session_len:
        raise ValueError(f"Expected SESSION_LEN={session_len}, got {sequences.shape[1]}")
    if sequences.shape[2] != feature_dim:
        raise ValueError(f"Expected FEATURE_DIM={feature_dim}, got {sequences.shape[2]}")

    masks = data["masks"].astype(bool) if "masks" in data else None
    real_flow_masks = normalize_real_flow_masks(sequences, masks)
    return sequences, labels, real_flow_masks
