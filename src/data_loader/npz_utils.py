"""Shared helpers for loading CyberShield session NPZ files safely."""

from __future__ import annotations

import os
from typing import Any, Sequence, Tuple

import numpy as np

from src.features.feature_config import FEATURE_DIM, FEATURE_NAMES, SESSION_LEN
from src.features.feature_config_extended import FEATURE_NAMES_EXTENDED, FEATURE_SCHEMA_VERSION
from src.features.feature_config_experiment import FEATURE_NAMES as EXPERIMENT_FEATURE_NAMES


def _infer_feature_names_from_width(feature_width: int) -> list[str]:
    if feature_width == len(EXPERIMENT_FEATURE_NAMES):
        return list(EXPERIMENT_FEATURE_NAMES)
    if feature_width == len(FEATURE_NAMES):
        return list(FEATURE_NAMES)
    if feature_width == len(FEATURE_NAMES_EXTENDED):
        return list(FEATURE_NAMES_EXTENDED)
    return list(FEATURE_NAMES)


def detect_npz_schema(path: str) -> dict[str, Any]:
    """Inspect an NPZ file and infer its session schema."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")

    data = np.load(path, allow_pickle=True)
    try:
        if "X" in data:
            shape = data["X"].shape
        elif "sessions" in data:
            shape = data["sessions"].shape
        else:
            shape = (0, 0, 0)

        feature_names: list[str] = []
        if "feature_names" in data:
            feature_names = [str(name) for name in data["feature_names"].tolist()]
        elif len(shape) == 3:
            feature_names = _infer_feature_names_from_width(shape[2])

        schema_version = str(data["schema_version"]) if "schema_version" in data else "unknown"
        if schema_version == "unknown" and len(shape) == 3:
            if shape[2] == len(FEATURE_NAMES_EXTENDED):
                schema_version = FEATURE_SCHEMA_VERSION
            elif shape[2] == len(EXPERIMENT_FEATURE_NAMES):
                schema_version = "experiment_v1"
            elif shape[2] == len(FEATURE_NAMES):
                schema_version = "base_v1"

        result = {
            "schema_version": schema_version,
            "feature_dim": shape[2] if len(shape) == 3 else 0,
            "session_len": shape[1] if len(shape) == 3 else 0,
            "num_samples": shape[0] if len(shape) >= 1 else 0,
            "feature_names": feature_names,
            "keys": list(data.keys()),
        }
    finally:
        data.close()
    return result


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
    try:
        if expected_feature_names is not None:
            feature_names = list(expected_feature_names)
        elif "feature_names" in data:
            feature_names = [str(name) for name in data["feature_names"].tolist()]
        else:
            if "X" in data:
                shape = data["X"].shape
            elif "sessions" in data:
                shape = data["sessions"].shape
            else:
                shape = (0, 0, 0)
            feature_names = _infer_feature_names_from_width(shape[2] if len(shape) == 3 else len(FEATURE_NAMES))

        session_len = int(expected_session_len) if expected_session_len is not None else SESSION_LEN
        feature_dim = len(feature_names)

        if expected_feature_names is not None and "feature_names" in data:
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

        masks = data["masks"].astype(bool) if "masks" in data else None
    finally:
        data.close()

    if sequences.ndim != 3:
        raise ValueError(f"Expected X to be 3D (N, {session_len}, {feature_dim}), got {sequences.shape}")
    if sequences.shape[1] != session_len:
        raise ValueError(f"Expected SESSION_LEN={session_len}, got {sequences.shape[1]}")
    if sequences.shape[2] != feature_dim:
        raise ValueError(f"Expected FEATURE_DIM={feature_dim}, got {sequences.shape[2]}")

    real_flow_masks = normalize_real_flow_masks(sequences, masks)
    return sequences, labels, real_flow_masks


def _remap_features(
    sequences: np.ndarray,
    source_names: list[str],
    target_names: list[str],
) -> np.ndarray:
    """Remap a (N, T, D_src) array to (N, T, D_tgt) by name matching.
    
    Features present in both schemas are copied; missing features are zero-padded.
    """
    import logging
    logger = logging.getLogger(__name__)

    N, T, _ = sequences.shape
    D_tgt = len(target_names)
    remapped = np.zeros((N, T, D_tgt), dtype=sequences.dtype)

    target_index = {name: idx for idx, name in enumerate(target_names)}
    mapped_count = 0

    for src_idx, src_name in enumerate(source_names):
        if src_name in target_index:
            remapped[:, :, target_index[src_name]] = sequences[:, :, src_idx]
            mapped_count += 1

    logger.info(
        f"Feature remapping: {mapped_count}/{len(source_names)} source features mapped "
        f"to {D_tgt}-dim target ({D_tgt - mapped_count} zero-padded)"
    )
    return remapped


def load_session_npz_from_bytes(
    file_bytes: bytes,
    expected_feature_names: Sequence[str] | None = None,
    expected_session_len: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load sessions from raw NPZ bytes (in-memory, no file handles).
    
    This avoids Windows file-locking issues by never touching the filesystem.
    If the NPZ uses a different feature schema, overlapping features are
    mapped automatically and missing features are zero-padded.
    """
    import io
    import logging
    logger = logging.getLogger(__name__)

    buf = io.BytesIO(file_bytes)
    data = np.load(buf, allow_pickle=True)

    try:
        # --- Determine the file's own feature names ---
        if "feature_names" in data:
            file_feature_names = [str(name) for name in data["feature_names"].tolist()]
        else:
            if "X" in data:
                shape = data["X"].shape
            elif "sessions" in data:
                shape = data["sessions"].shape
            else:
                shape = (0, 0, 0)
            file_feature_names = _infer_feature_names_from_width(
                shape[2] if len(shape) == 3 else len(FEATURE_NAMES)
            )

        # --- Load raw arrays ---
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

        masks = data["masks"].astype(bool) if "masks" in data else None
    finally:
        data.close()

    session_len = int(expected_session_len) if expected_session_len is not None else SESSION_LEN

    # --- Validate session length ---
    if sequences.ndim != 3:
        raise ValueError(f"Expected X to be 3D, got {sequences.shape}")
    if sequences.shape[1] != session_len:
        raise ValueError(f"Expected SESSION_LEN={session_len}, got {sequences.shape[1]}")

    # --- Feature remapping if schemas differ ---
    if expected_feature_names is not None:
        target_names = list(expected_feature_names)
        if file_feature_names != target_names:
            # Check if there's any overlap at all
            overlap = set(file_feature_names) & set(target_names)
            if not overlap:
                raise ValueError(
                    f"No overlapping features between uploaded NPZ ({file_feature_names}) "
                    f"and model ({target_names}). Cannot auto-map."
                )
            logger.info(
                f"Schema auto-adaptation: file has {len(file_feature_names)} features, "
                f"model expects {len(target_names)} — remapping {len(overlap)} shared features"
            )
            sequences = _remap_features(sequences, file_feature_names, target_names)
    else:
        target_names = file_feature_names

    feature_dim = len(target_names)
    if sequences.shape[2] != feature_dim:
        raise ValueError(f"Expected FEATURE_DIM={feature_dim}, got {sequences.shape[2]}")

    real_flow_masks = normalize_real_flow_masks(sequences, masks)
    return sequences, labels, real_flow_masks
