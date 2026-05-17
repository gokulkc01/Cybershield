"""Extended session builder for the ``extended_v1`` 45-feature schema.

Converts flow-level DataFrames (the same format ingested by the v1
``session_builder.py``) into ``(SESSION_LEN, FEATURE_DIM_EXTENDED)``
tensors, enriched with TLS, DNS and temporal features.

Pipeline position:
    normalized_telemetry.py  →  THIS FILE  →  NPZ / training pipeline

Key design decisions:
  - **Per-flow vectors** still use the 10 base features from the experiment
    schema (no ports).  TLS, DNS and temporal features are derived from the
    *session aggregate* and broadcast identically to every flow row so the
    tensor shape stays ``(SESSION_LEN, 45)`` and the Transformer can still
    attend over the flow sequence.
  - When TLS / DNS context DataFrames are not provided the corresponding
    feature groups are zero-filled — this is safe because the extractors
    and the builder already guarantee zero-fill semantics.
  - The module is a drop-in upgrade path: callers that only have ``conn.log``
    data get the base+temporal features automatically; TLS/DNS enrich
    further when ``ssl.log`` / ``dns.log`` are available.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.extended_feature_builder import ExtendedFeatureBuilder
from src.features.feature_config_extended import (
    FEATURE_DIM_EXTENDED,
    FEATURE_NAMES_EXTENDED,
    FEATURE_SCHEMA_VERSION,
)
from src.features.feature_config_experiment import SESSION_LEN
from src.features.feature_config import (
    INACTIVITY_TIMEOUT,
    LABEL_BENIGN,
    LABEL_C2,
)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def extended_flow_row_to_vector(
    row: pd.Series,
    *,
    session_tls_features: Dict[str, float] | None = None,
    session_dns_features: Dict[str, float] | None = None,
    session_temporal_features: Dict[str, float] | None = None,
    builder: ExtendedFeatureBuilder | None = None,
) -> np.ndarray:
    """Convert one flow row into a ``(FEATURE_DIM_EXTENDED,)`` vector.

    The base features are extracted from the row itself.  TLS, DNS and
    temporal features are *session-level* and are broadcast identically
    across all flow rows in the same session.  If pre-computed session-
    level dicts are passed they are used directly; otherwise the builder
    extracts what it can from the single row.
    """
    if builder is None:
        builder = ExtendedFeatureBuilder()

    # Build per-flow base features from the row mapping
    feat = builder.build_feature_dict(row.to_dict())

    # Override with session-level aggregates when available
    if session_tls_features:
        feat.update(session_tls_features)
    if session_dns_features:
        feat.update(session_dns_features)
    if session_temporal_features:
        feat.update(session_temporal_features)

    vector = np.array(
        [feat.get(name, 0.0) for name in FEATURE_NAMES_EXTENDED],
        dtype=np.float32,
    )
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def extended_session_to_tensor(
    session_df: pd.DataFrame,
    *,
    tls_context: pd.DataFrame | None = None,
    dns_context: pd.DataFrame | None = None,
    builder: ExtendedFeatureBuilder | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a session DataFrame into a padded ``(SESSION_LEN, 45)`` tensor.

    Parameters
    ----------
    session_df : Flow-level DataFrame for one session (sorted by ``ts``).
    tls_context : Optional ssl.log DataFrame filtered to overlapping flows.
    dns_context : Optional dns.log DataFrame filtered to overlapping flows.
    builder : Reusable ``ExtendedFeatureBuilder`` instance.

    Returns
    -------
    tensor : ``(SESSION_LEN, FEATURE_DIM_EXTENDED)`` float32
    mask : ``(SESSION_LEN,)`` bool — True for real flows.
    """
    if builder is None:
        builder = ExtendedFeatureBuilder()

    # Compute session-level TLS, DNS, and temporal features once
    session_tls = builder._extract_tls_features(tls_context)
    session_dns = builder._extract_dns_features(dns_context)
    session_temporal = builder._extract_temporal_features(session_df)

    # Build per-flow vectors
    flow_vectors = []
    for _, row in session_df.iterrows():
        vec = extended_flow_row_to_vector(
            row,
            session_tls_features=session_tls,
            session_dns_features=session_dns,
            session_temporal_features=session_temporal,
            builder=builder,
        )
        flow_vectors.append(vec)

    flow_vectors = np.array(flow_vectors, dtype=np.float32)

    # Truncate to SESSION_LEN (keep most recent)
    if len(flow_vectors) > SESSION_LEN:
        flow_vectors = flow_vectors[-SESSION_LEN:]

    n_real = len(flow_vectors)
    padded = np.zeros((SESSION_LEN, FEATURE_DIM_EXTENDED), dtype=np.float32)
    padded[:n_real] = flow_vectors

    mask = np.zeros(SESSION_LEN, dtype=bool)
    mask[:n_real] = True

    return padded, mask


def build_extended_sessions(
    df: pd.DataFrame,
    *,
    label_col: str | None = "label",
    inactivity_timeout: float = INACTIVITY_TIMEOUT,
    min_flows: int = 1,
    tls_df: pd.DataFrame | None = None,
    dns_df: pd.DataFrame | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a flat flow DataFrame into extended_v1 session tensors.

    This is the extended analogue of ``session_builder.build_sessions()``
    but outputs ``(N, SESSION_LEN, FEATURE_DIM_EXTENDED)`` tensors.

    Parameters
    ----------
    df : Normalized flow DataFrame with at least ``ts, src_ip, dst_ip, proto``
         and the base feature columns.
    label_col : Column name for flow-level labels (None for unlabeled data).
    inactivity_timeout : Seconds of inactivity before a new session starts.
    min_flows : Minimum flows to keep a session.
    tls_df : Optional ssl.log DataFrame. Will be joined to sessions by
             ``(src_ip, dst_ip)`` and overlapping timestamps.
    dns_df : Optional dns.log DataFrame. Will be joined to sessions by
             ``src_ip`` and overlapping timestamps.

    Returns
    -------
    sessions : ``(N, SESSION_LEN, FEATURE_DIM_EXTENDED)`` float32
    labels : ``(N,)`` int64
    masks : ``(N, SESSION_LEN)`` bool
    """
    required = {"ts", "src_ip", "dst_ip", "proto"}
    if label_col:
        required.add(label_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    df = df.copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0.0)
    df = df.sort_values(["src_ip", "dst_ip", "proto", "ts"]).reset_index(drop=True)

    builder = ExtendedFeatureBuilder()
    sessions_list: List[np.ndarray] = []
    labels_list: List[int] = []
    masks_list: List[np.ndarray] = []

    groups = df.groupby(["src_ip", "dst_ip", "proto"], sort=False)

    for (src_ip, dst_ip, proto), group in tqdm(
        groups, desc="Building extended sessions", leave=False, unit="group"
    ):
        group = group.sort_values("ts").reset_index(drop=True)
        for sess_df in _segment_by_inactivity(group, inactivity_timeout):
            if len(sess_df) < min_flows:
                continue

            # Extract TLS/DNS context for this session
            sess_tls = _filter_context_for_session(
                tls_df, src_ip, dst_ip, sess_df, context_type="tls"
            )
            sess_dns = _filter_context_for_session(
                dns_df, src_ip, dst_ip, sess_df, context_type="dns"
            )

            padded, mask = extended_session_to_tensor(
                sess_df,
                tls_context=sess_tls,
                dns_context=sess_dns,
                builder=builder,
            )

            label = (
                int(sess_df[label_col].mode().iloc[0])
                if label_col and label_col in sess_df.columns
                else LABEL_BENIGN
            )

            sessions_list.append(padded)
            labels_list.append(label)
            masks_list.append(mask)

    if not sessions_list:
        raise ValueError("No sessions built. Check input DataFrame.")

    sessions = np.stack(sessions_list)
    labels = np.array(labels_list, dtype=np.int64)
    masks = np.stack(masks_list)

    n_c2 = int((labels == LABEL_C2).sum())
    print(
        f"Extended sessions: {len(sessions):,} | "
        f"C2: {n_c2:,} ({100 * n_c2 / len(sessions):.1f}%) | "
        f"Benign: {len(sessions) - n_c2:,} | "
        f"Schema: {FEATURE_SCHEMA_VERSION} ({FEATURE_DIM_EXTENDED}d)"
    )
    return sessions, labels, masks


def save_extended_sessions(
    sessions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    path: str,
) -> None:
    """Save extended_v1 session tensors to compressed NPZ."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        X=sessions.astype(np.float32),
        y=labels.astype(np.int64),
        masks=masks.astype(bool),
        feature_names=np.array(FEATURE_NAMES_EXTENDED),
        schema_version=np.array(FEATURE_SCHEMA_VERSION),
    )
    print(f"Saved {len(sessions):,} extended sessions → {path}")


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _segment_by_inactivity(
    group: pd.DataFrame, timeout: float
) -> List[pd.DataFrame]:
    """Split a sorted flow group into sessions by inactivity gap."""
    if len(group) == 0:
        return []
    sessions: List[pd.DataFrame] = []
    current = [group.iloc[0]]
    for i in range(1, len(group)):
        gap = float(group.iloc[i]["ts"]) - float(group.iloc[i - 1]["ts"])
        if gap > timeout:
            sessions.append(pd.DataFrame(current).reset_index(drop=True))
            current = [group.iloc[i]]
        else:
            current.append(group.iloc[i])
    if current:
        sessions.append(pd.DataFrame(current).reset_index(drop=True))
    return sessions


def _filter_context_for_session(
    context_df: pd.DataFrame | None,
    src_ip: str,
    dst_ip: str,
    session_df: pd.DataFrame,
    context_type: str = "tls",
) -> pd.DataFrame | None:
    """Filter a TLS or DNS log to rows relevant to a session's time window."""
    if context_df is None or context_df.empty:
        return None

    ts_vals = pd.to_numeric(session_df["ts"], errors="coerce").dropna()
    if ts_vals.empty:
        return None

    t_start = float(ts_vals.iloc[0])
    t_end = float(ts_vals.iloc[-1])
    # Add small window margin for context lookup
    margin = max(30.0, (t_end - t_start) * 0.1)

    # Determine which columns to join on
    if context_type == "tls":
        # ssl.log typically has id.orig_h, id.resp_h
        src_col = _first_present(context_df, ["id.orig_h", "src_ip", "orig_h"])
        dst_col = _first_present(context_df, ["id.resp_h", "dst_ip", "resp_h"])
    else:
        # dns.log typically has id.orig_h
        src_col = _first_present(context_df, ["id.orig_h", "src_ip", "orig_h"])
        dst_col = None

    ts_col = _first_present(context_df, ["ts", "timestamp"])
    if ts_col is None:
        return None

    ctx = context_df.copy()
    ctx["_ts_num"] = pd.to_numeric(ctx[ts_col], errors="coerce")

    # Filter by time window
    mask = (ctx["_ts_num"] >= t_start - margin) & (ctx["_ts_num"] <= t_end + margin)

    # Filter by source IP
    if src_col:
        mask = mask & (ctx[src_col].astype(str) == str(src_ip))

    # Filter by destination IP (TLS only)
    if dst_col and context_type == "tls":
        mask = mask & (ctx[dst_col].astype(str) == str(dst_ip))

    filtered = ctx.loc[mask].drop(columns=["_ts_num"], errors="ignore")
    return filtered if len(filtered) > 0 else None


def _first_present(df: pd.DataFrame, names: List[str]) -> str | None:
    """Return the first column name present in the DataFrame."""
    return next((name for name in names if name in df.columns), None)
