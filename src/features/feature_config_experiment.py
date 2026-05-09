"""Experiment-only 10-feature schema for the multifamily generalization run."""

from __future__ import annotations

FEATURE_NAMES = [
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "bytes_per_pkt",
    "packet_ratio",
    "byte_ratio",
    "is_outbound",
    "duration",
    "iat",
]

FEATURE_DIM = len(FEATURE_NAMES)
assert FEATURE_DIM == 10, f"Expected 10 experiment features, got {FEATURE_DIM}"

FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
ORIG_BYTES_IDX = FEATURE_INDEX["orig_bytes"]
IAT_IDX = FEATURE_INDEX["iat"]

SESSION_LEN = 20
