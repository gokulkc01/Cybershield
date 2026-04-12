"""
session_builder.py
==================
Groups processed flow records into sessions, then converts each session
into a fixed-length tensor ready for the Transformer encoder.

Session definition:
  - Group by (src_ip, dst_ip, protocol)
  - Segment by INACTIVITY_TIMEOUT: if gap between consecutive flows
    exceeds INACTIVITY_TIMEOUT seconds, start a new session
  - Compute session-level IAT stats (mean, std, cv) over all flows
  - Truncate to SESSION_LEN (keep most recent flows)
  - Zero-pad shorter sessions + boolean padding mask

Pipeline position:
    zeek_parser.py → THIS FILE → dataset_builder.py → models
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from tqdm import tqdm

from src.features.feature_config import (
    FEATURE_NAMES, FEATURE_DIM, SESSION_LEN, INACTIVITY_TIMEOUT,
    LABEL_C2, LABEL_BENIGN,
)


def _to_float(value, default: float = 0.0) -> float:
    """Best-effort numeric conversion for mixed Zeek/CTU field encodings."""
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            if text.lower().startswith("0x"):
                return float(int(text, 16))
            return float(text)
        except ValueError:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def segment_by_inactivity(
    group: pd.DataFrame,
    timeout: float = INACTIVITY_TIMEOUT,
) -> List[pd.DataFrame]:
    """Split a sorted group into sessions using inactivity timeout."""
    if len(group) == 0:
        return []
    sessions = []
    current = [group.iloc[0]]
    for i in range(1, len(group)):
        gap = float(group.iloc[i]["ts"]) - float(group.iloc[i-1]["ts"])
        if gap > timeout:
            sessions.append(pd.DataFrame(current).reset_index(drop=True))
            current = [group.iloc[i]]
        else:
            current.append(group.iloc[i])
    if current:
        sessions.append(pd.DataFrame(current).reset_index(drop=True))
    return sessions


def flow_row_to_vector(row: pd.Series) -> np.ndarray:
    """
    Convert one flow row to a FEATURE_DIM vector.
    Order must exactly match FEATURE_NAMES in feature_config.py.
    """
    feature_map = {
        "orig_bytes": _to_float(row.get("orig_bytes", 0), 0.0),
        "resp_bytes": _to_float(row.get("resp_bytes", 0), 0.0),
        "orig_pkts": _to_float(row.get("orig_pkts", 1), 1.0),
        "resp_pkts": _to_float(row.get("resp_pkts", 0), 0.0),
        "bytes_per_pkt": _to_float(row.get("bytes_per_pkt", 0), 0.0),
        "packet_ratio": _to_float(row.get("packet_ratio", 0), 0.0),
        "byte_ratio": _to_float(row.get("byte_ratio", 0), 0.0),
        "is_outbound": _to_float(row.get("is_outbound", 0), 0.0),
        "duration": _to_float(row.get("duration", 0), 0.0),
        "src_port": _to_float(row.get("src_port", 0), 0.0),
        "dst_port": _to_float(row.get("dst_port", 0), 0.0),
        "iat": _to_float(row.get("iat", 0), 0.0),
    }
    vec = np.array([feature_map[name] for name in FEATURE_NAMES], dtype=np.float32)
    assert len(vec) == FEATURE_DIM
    return vec


def session_to_tensor(session_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a session DataFrame to padded tensor + mask."""
    flow_vectors = np.array([
        flow_row_to_vector(row)
        for _, row in session_df.iterrows()
    ], dtype=np.float32)

    if len(flow_vectors) > SESSION_LEN:
        flow_vectors = flow_vectors[-SESSION_LEN:]   # keep most recent

    n_real = len(flow_vectors)
    padded = np.zeros((SESSION_LEN, FEATURE_DIM), dtype=np.float32)
    padded[:n_real] = flow_vectors

    mask = np.zeros(SESSION_LEN, dtype=bool)
    mask[:n_real] = True

    return padded, mask


def build_sessions(
    df: pd.DataFrame,
    label_col: Optional[str] = "label",
    inactivity_timeout: float = INACTIVITY_TIMEOUT,
    min_flows: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert flat flow DataFrame into session-level tensors.

    Parameters
    ----------
    df                 : output of zeek_parser.process_conn_log(return_metadata=True)
                         with an added 'label' column
    label_col          : flow-level label column name (None for inference)
    inactivity_timeout : seconds of inactivity before a new session starts
    min_flows          : minimum flows to keep a session

    Returns
    -------
    sessions : (N, SESSION_LEN, FEATURE_DIM)
    labels   : (N,)
    masks    : (N, SESSION_LEN) bool — True = padding
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

    sessions_list, labels_list, masks_list = [], [], []

    groups = df.groupby(["src_ip", "dst_ip", "proto"], sort=False)
    for _, group in tqdm(groups, desc="Building sessions", leave=False, unit="group"):
        group = group.sort_values("ts").reset_index(drop=True)
        for sess_df in segment_by_inactivity(group, timeout=inactivity_timeout):
            if len(sess_df) < min_flows:
                continue
            padded, mask = session_to_tensor(sess_df)
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
    labels   = np.array(labels_list, dtype=np.int64)
    masks    = np.stack(masks_list)

    n_c2 = int((labels == LABEL_C2).sum())
    print(
        f"Sessions: {len(sessions):,} | "
        f"C2: {n_c2:,} ({100*n_c2/len(sessions):.1f}%) | "
        f"Benign: {len(sessions)-n_c2:,}"
    )
    return sessions, labels, masks


def save_sessions(sessions, labels, masks, path: str) -> None:
    np.savez_compressed(path, X=sessions, y=labels, masks=masks, feature_names=np.array(FEATURE_NAMES))
    print(f"Saved {len(sessions):,} sessions → {path}")


def load_sessions(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    if "X" in data and "y" in data:
        return data["X"], data["y"], data["masks"]
    return data["sessions"], data["labels"], data["masks"]


def sessions_to_flat(sessions: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Flatten sessions to 2D for tree baselines. Output: (N, FEATURE_DIM*4)"""
    N = sessions.shape[0]
    flat = np.zeros((N, FEATURE_DIM * 4), dtype=np.float32)
    for i in range(N):
        real = sessions[i][masks[i]]
        if len(real) == 0:
            real = sessions[i][:1]
        flat[i] = np.concatenate([
            real.mean(0), real.std(0), real.min(0), real.max(0)
        ])
    return flat
