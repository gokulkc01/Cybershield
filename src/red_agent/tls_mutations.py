"""TLS/HTTPS-like flow-shape mutations (without payload semantics)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.red_agent.mutation_registry import register_mutation


def _resolve_index(feature_index: Dict[str, int], key: str) -> int:
    if key not in feature_index:
        raise KeyError(f"Feature '{key}' not present in feature_index")
    return feature_index[key]


@register_mutation(name="tls_padding", category="tls")
def tls_padding(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Increase bytes-per-packet and byte counters to mimic padded TLS records."""
    feature_index = params["feature_index"]
    bpp_idx = _resolve_index(feature_index, "bytes_per_pkt")
    orig_bytes_idx = _resolve_index(feature_index, "orig_bytes")
    resp_bytes_idx = _resolve_index(feature_index, "resp_bytes")
    out = sequence.copy()

    padding_factor = float(params.get("padding_factor", 1.05 + 0.7 * severity))
    out[:, bpp_idx] *= padding_factor
    out[:, orig_bytes_idx] *= (1.0 + 0.5 * severity)
    out[:, resp_bytes_idx] *= (1.0 + 0.5 * severity)
    return np.clip(out, 0.0, None)


@register_mutation(name="session_reuse_shape", category="tls")
def session_reuse_shape(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Flatten handshake-like variance to imitate TLS session reuse."""
    feature_index = params["feature_index"]
    duration_idx = _resolve_index(feature_index, "duration")
    iat_idx = _resolve_index(feature_index, "iat")
    out = sequence.copy()

    duration_mean = float(np.mean(out[:, duration_idx]))
    iat_mean = float(np.mean(out[:, iat_idx]))
    smooth = float(params.get("smoothing", min(0.9, 0.2 + severity)))

    out[:, duration_idx] = (1.0 - smooth) * out[:, duration_idx] + smooth * duration_mean
    out[:, iat_idx] = (1.0 - smooth) * out[:, iat_idx] + smooth * iat_mean
    return np.clip(out, 0.0, None)


@register_mutation(name="handshake_variation", category="tls")
def handshake_variation(sequence: np.ndarray, rng: np.random.Generator, severity: float, params: dict) -> np.ndarray:
    """Perturb packet/byte ratios to emulate varying TLS handshake footprints."""
    feature_index = params["feature_index"]
    packet_ratio_idx = _resolve_index(feature_index, "packet_ratio")
    byte_ratio_idx = _resolve_index(feature_index, "byte_ratio")
    out = sequence.copy()

    noise_scale = float(params.get("ratio_noise", 0.05 + 0.35 * severity))
    pkt_noise = rng.normal(0.0, noise_scale, size=out.shape[0]).astype(np.float32)
    byt_noise = rng.normal(0.0, noise_scale, size=out.shape[0]).astype(np.float32)

    out[:, packet_ratio_idx] = np.clip(out[:, packet_ratio_idx] + pkt_noise, 0.0, None)
    out[:, byte_ratio_idx] = np.clip(out[:, byte_ratio_idx] + byt_noise, 0.0, None)
    return out
