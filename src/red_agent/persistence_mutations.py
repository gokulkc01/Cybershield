"""Persistence-style mutations (low-and-slow and intermittent behavior)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.red_agent.mutation_registry import register_mutation


def _resolve_index(feature_index: Dict[str, int], key: str) -> int:
    if key not in feature_index:
        raise KeyError(f"Feature '{key}' not present in feature_index")
    return feature_index[key]


@register_mutation(name="low_frequency_callback", category="persistence")
def low_frequency_callback(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Increase iat to model reduced callback frequency."""
    feature_index = params["feature_index"]
    iat_idx = _resolve_index(feature_index, "iat")
    out = sequence.copy()

    multiplier = float(params.get("frequency_reduction", 1.0 + 8.0 * severity))
    out[:, iat_idx] = np.clip(out[:, iat_idx] * multiplier, 0.0, None)
    return out


@register_mutation(name="intermittent_communication", category="persistence")
def intermittent_communication(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Drop activity from random windows and compensate with sparse spikes."""
    feature_index = params["feature_index"]
    orig_pkts_idx = _resolve_index(feature_index, "orig_pkts")
    resp_pkts_idx = _resolve_index(feature_index, "resp_pkts")
    orig_bytes_idx = _resolve_index(feature_index, "orig_bytes")
    resp_bytes_idx = _resolve_index(feature_index, "resp_bytes")
    out = sequence.copy()

    n = out.shape[0]
    quiet_fraction = float(params.get("quiet_fraction", 0.2 + 0.5 * severity))
    k = max(1, int(n * quiet_fraction))
    quiet_idx = rng.choice(n, size=k, replace=False)

    decay = float(params.get("quiet_decay", max(0.02, 0.5 - 0.45 * severity)))
    out[quiet_idx, orig_pkts_idx] *= decay
    out[quiet_idx, resp_pkts_idx] *= decay
    out[quiet_idx, orig_bytes_idx] *= decay
    out[quiet_idx, resp_bytes_idx] *= decay

    spike_pool = np.setdiff1d(np.arange(n), quiet_idx)
    if spike_pool.size > 0:
        spike_count = max(1, int(spike_pool.size * min(0.3, severity * 0.4)))
        spike_idx = rng.choice(spike_pool, size=spike_count, replace=False)
        boost = float(params.get("spike_boost", 1.2 + 2.0 * severity))
        out[spike_idx, orig_pkts_idx] *= boost
        out[spike_idx, resp_pkts_idx] *= boost
        out[spike_idx, orig_bytes_idx] *= boost
        out[spike_idx, resp_bytes_idx] *= boost

    return np.clip(out, 0.0, None)


@register_mutation(name="long_sleep", category="persistence")
def long_sleep(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Simulate long beacon sleep intervals with sparse reconnects."""
    feature_index = params["feature_index"]
    iat_idx = _resolve_index(feature_index, "iat")
    duration_idx = _resolve_index(feature_index, "duration")
    out = sequence.copy()

    n = out.shape[0]
    sleep_segments = int(params.get("segments", max(1, int(1 + severity * 3))))
    for _ in range(sleep_segments):
        start = int(rng.integers(0, max(1, n - 1)))
        length = int(rng.integers(1, max(2, n // 3)))
        end = min(n, start + length)
        sleep_scale = float(params.get("sleep_scale", 2.0 + 10.0 * severity))
        out[start:end, iat_idx] *= sleep_scale
        out[start:end, duration_idx] *= (1.0 + 0.7 * severity)

    return np.clip(out, 0.0, None)
