"""Extended NPZ utilities for loading/saving extended_v1 session data.

This module handles the ``extended_v1`` schema (45 features) alongside the
existing v1 (12-feature) and experiment (10-feature) schemas.  It is a
thin layer on top of ``npz_utils.py`` that knows about the extended feature
names, dimensions, and schema version metadata.
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np

from src.features.feature_config_extended import (
    FEATURE_DIM_EXTENDED,
    FEATURE_NAMES_EXTENDED,
    FEATURE_SCHEMA_VERSION,
)
from src.features.feature_config_experiment import SESSION_LEN
from src.data_loader.npz_utils import normalize_real_flow_masks


def load_extended_session_npz(
    path: str,
    *,
    expected_feature_names: Sequence[str] | None = None,
    expected_session_len: int | None = None,
    strict_schema: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an extended_v1 NPZ file and return (sequences, labels, masks).

    Parameters
    ----------
    path : Path to the NPZ file.
    expected_feature_names : Override expected feature names.  Defaults to
        ``FEATURE_NAMES_EXTENDED``.
    expected_session_len : Override session length.  Defaults to ``SESSION_LEN``.
    strict_schema : If True (default), validate the saved feature names and
        schema version match the current ``extended_v1`` spec.

    Returns
    -------
    sequences : ``(N, SESSION_LEN, FEATURE_DIM_EXTENDED)`` float32
    labels : ``(N,)`` int64
    masks : ``(N, SESSION_LEN)`` bool (True = real flow)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Extended NPZ not found: {path}")

    data = np.load(path, allow_pickle=True)

    feature_names = list(expected_feature_names) if expected_feature_names else list(FEATURE_NAMES_EXTENDED)
    session_len = int(expected_session_len) if expected_session_len else SESSION_LEN
    feature_dim = len(feature_names)

    # Schema version check
    if strict_schema and "schema_version" in data:
        saved_version = str(data["schema_version"])
        if saved_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: NPZ has '{saved_version}', "
                f"expected '{FEATURE_SCHEMA_VERSION}'"
            )

    # Feature name check
    if "feature_names" in data:
        saved_names = [str(name) for name in data["feature_names"].tolist()]
        if strict_schema and saved_names != feature_names:
            raise ValueError(
                f"Feature schema mismatch in {path}. "
                f"Expected {len(feature_names)} features, NPZ has {len(saved_names)}. "
                f"Missing: {set(feature_names) - set(saved_names)}, "
                f"Extra: {set(saved_names) - set(feature_names)}"
            )

    # Load arrays
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

    # Shape validation
    if sequences.ndim != 3:
        raise ValueError(
            f"Expected X to be 3D (N, {session_len}, {feature_dim}), got shape {sequences.shape}"
        )
    if sequences.shape[1] != session_len:
        raise ValueError(
            f"Session length mismatch: expected {session_len}, got {sequences.shape[1]}"
        )
    if sequences.shape[2] != feature_dim:
        raise ValueError(
            f"Feature dim mismatch: expected {feature_dim}, got {sequences.shape[2]}"
        )

    # Normalize masks
    masks = data["masks"].astype(bool) if "masks" in data else None
    real_flow_masks = normalize_real_flow_masks(sequences, masks)

    return sequences, labels, real_flow_masks


def save_extended_session_npz(
    path: str,
    sequences: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    schema_version: str | None = None,
) -> None:
    """Save extended_v1 session tensors to compressed NPZ with metadata.

    Parameters
    ----------
    path : Output file path.
    sequences : ``(N, SESSION_LEN, FEATURE_DIM_EXTENDED)`` tensor.
    labels : ``(N,)`` label array.
    masks : ``(N, SESSION_LEN)`` boolean mask (True = real flow).
    feature_names : Override feature name list.
    schema_version : Override schema version string.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        X=sequences.astype(np.float32),
        y=labels.astype(np.int64),
        masks=masks.astype(bool),
        feature_names=np.array(feature_names or FEATURE_NAMES_EXTENDED),
        schema_version=np.array(schema_version or FEATURE_SCHEMA_VERSION),
    )
    print(
        f"Saved {len(sequences):,} extended sessions → {path} "
        f"[schema={schema_version or FEATURE_SCHEMA_VERSION}, "
        f"dim={sequences.shape[2]}]"
    )


def detect_npz_schema(path: str) -> dict:
    """Detect the schema of an NPZ file without full loading.

    Returns a dict with keys: schema_version, feature_dim, feature_names,
    session_len, num_samples.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")

    data = np.load(path, allow_pickle=True)

    # Determine sequences key
    if "X" in data:
        shape = data["X"].shape
    elif "sessions" in data:
        shape = data["sessions"].shape
    else:
        shape = (0, 0, 0)

    feature_names = []
    if "feature_names" in data:
        feature_names = [str(name) for name in data["feature_names"].tolist()]

    schema_version = str(data["schema_version"]) if "schema_version" in data else "unknown"

    return {
        "schema_version": schema_version,
        "feature_dim": shape[2] if len(shape) == 3 else 0,
        "session_len": shape[1] if len(shape) == 3 else 0,
        "num_samples": shape[0] if len(shape) >= 1 else 0,
        "feature_names": feature_names,
        "keys": list(data.keys()),
    }
