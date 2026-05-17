"""Flow-statistic mutations to stress shallow feature dependencies."""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.red_agent.mutation_registry import register_mutation


def _resolve_index(feature_index: Dict[str, int], key: str) -> int:
    if key not in feature_index:
        raise KeyError(f"Feature '{key}' not present in feature_index")
    return feature_index[key]


@register_mutation(name="packet_count_variation", category="flow")
def packet_count_variation(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Vary packet counts while preserving directional asymmetry."""
    feature_index = params["feature_index"]
    orig_pkts_idx = _resolve_index(feature_index, "orig_pkts")
    resp_pkts_idx = _resolve_index(feature_index, "resp_pkts")
    packet_ratio_idx = _resolve_index(feature_index, "packet_ratio")
    out = sequence.copy()

    scale_noise = rng.normal(loc=1.0, scale=0.1 + severity * 0.4, size=out.shape[0]).astype(np.float32)
    scale_noise = np.clip(scale_noise, 0.05, 5.0)
    out[:, orig_pkts_idx] *= scale_noise
    out[:, resp_pkts_idx] *= scale_noise

    ratio_noise = rng.normal(loc=0.0, scale=0.03 + 0.2 * severity, size=out.shape[0]).astype(np.float32)
    out[:, packet_ratio_idx] = np.clip(out[:, packet_ratio_idx] + ratio_noise, 0.0, None)
    return np.clip(out, 0.0, None)


@register_mutation(name="byte_distribution_shift", category="flow")
def byte_distribution_shift(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Shift byte distributions and bytes-per-packet characteristics."""
    feature_index = params["feature_index"]
    orig_bytes_idx = _resolve_index(feature_index, "orig_bytes")
    resp_bytes_idx = _resolve_index(feature_index, "resp_bytes")
    byte_ratio_idx = _resolve_index(feature_index, "byte_ratio")
    bpp_idx = _resolve_index(feature_index, "bytes_per_pkt")
    out = sequence.copy()

    swing = float(params.get("swing", 0.2 + 1.0 * severity))
    direction = rng.choice([-1.0, 1.0], size=out.shape[0]).astype(np.float32)
    factors = np.clip(1.0 + direction * swing, 0.05, 6.0)

    out[:, orig_bytes_idx] *= factors
    out[:, resp_bytes_idx] *= np.clip(2.0 - factors, 0.05, 6.0)
    out[:, bpp_idx] *= np.clip(1.0 + rng.normal(0.0, 0.1 + 0.3 * severity, size=out.shape[0]), 0.05, 4.0)

    ratio_noise = rng.normal(0.0, 0.05 + 0.25 * severity, size=out.shape[0]).astype(np.float32)
    out[:, byte_ratio_idx] = np.clip(out[:, byte_ratio_idx] + ratio_noise, 0.0, None)
    return np.clip(out, 0.0, None)
