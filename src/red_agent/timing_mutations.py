"""Timing-oriented behavioral mutations for session sequences."""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.red_agent.mutation_registry import register_mutation


def _resolve_index(feature_index: Dict[str, int], key: str) -> int:
    if key not in feature_index:
        raise KeyError(f"Feature '{key}' not present in feature_index")
    return feature_index[key]


@register_mutation(name="timing_jitter", category="timing")
def timing_jitter(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Inject multiplicative jitter to inter-arrival time (iat)."""
    feature_index = params["feature_index"]
    iat_idx = _resolve_index(feature_index, "iat")
    out = sequence.copy()

    max_scale = float(params.get("max_scale", 1.0 + 2.0 * severity))
    min_scale = max(0.05, float(params.get("min_scale", 1.0 - 0.7 * severity)))
    scales = rng.uniform(min_scale, max_scale, size=out.shape[0]).astype(np.float32)
    out[:, iat_idx] = np.clip(out[:, iat_idx] * scales, 0.0, None)
    return out


@register_mutation(name="burst_callback", category="timing")
def burst_callback(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Create bursty callback rhythm by compressing selected intervals and expanding others."""
    feature_index = params["feature_index"]
    iat_idx = _resolve_index(feature_index, "iat")
    out = sequence.copy()

    ratio = float(params.get("burst_ratio", min(0.8, 0.2 + severity * 0.5)))
    n = out.shape[0]
    k = max(1, int(n * ratio))
    burst_idx = rng.choice(n, size=k, replace=False)

    compress = float(params.get("compress_factor", max(0.05, 0.6 - 0.5 * severity)))
    expand = float(params.get("expand_factor", 1.0 + 2.5 * severity))

    out[burst_idx, iat_idx] = np.clip(out[burst_idx, iat_idx] * compress, 0.0, None)
    mask = np.ones(n, dtype=bool)
    mask[burst_idx] = False
    out[mask, iat_idx] = np.clip(out[mask, iat_idx] * expand, 0.0, None)
    return out


@register_mutation(name="delayed_reconnect", category="timing")
def delayed_reconnect(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Inject long reconnect gaps on a subset of flows."""
    feature_index = params["feature_index"]
    iat_idx = _resolve_index(feature_index, "iat")
    duration_idx = _resolve_index(feature_index, "duration")
    out = sequence.copy()

    p = float(params.get("affected_fraction", 0.1 + 0.4 * severity))
    affected = rng.random(out.shape[0]) < p
    multiplier = float(params.get("delay_multiplier", 2.0 + 6.0 * severity))

    out[affected, iat_idx] = np.clip(out[affected, iat_idx] * multiplier, 0.0, None)
    out[affected, duration_idx] = np.clip(out[affected, duration_idx] * (1.0 + severity), 0.0, None)
    return out
