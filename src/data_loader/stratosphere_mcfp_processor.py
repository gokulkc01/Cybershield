"""Build zero-shot evaluation NPZ from Stratosphere MCFP flow files.

Outputs an NPZ in the same schema used by training/evaluation code:
- X: (N, SESSION_LEN, FEATURE_DIM)
- y: (N,)
- masks: (N, SESSION_LEN) where True means real flow in this file's convention

Usage examples
--------------
python -m src.data_loader.stratosphere_mcfp_processor \
  --c2_inputs data/raw/mcfp/emotet_conn.log \
  --benign_inputs data/raw/mcfp/benign_conn.log \
  --output_path data/processed/stratosphere_mcfp_sessions.npz

python -m src.data_loader.stratosphere_mcfp_processor \
  --c2_inputs data/raw/mcfp/emotet.csv data/raw/mcfp/trickbot.csv \
  --output_path data/processed/mcfp_c2_only_sessions.npz
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.features.feature_config import FEATURE_NAMES, LABEL_BENIGN, LABEL_C2, is_private_ip
from src.features.session_builder import build_sessions
from src.features.zeek_parser import process_conn_log


def _read_generic_csv(path: str) -> pd.DataFrame:
    """Read a delimited flow file with best-effort separator detection."""
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def _read_binetflow(path: str) -> pd.DataFrame:
    """Read Argus/Stratosphere .binetflow exports."""
    return pd.read_csv(path, low_memory=False)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common flow column aliases to CyberShield canonical names."""
    rename_map = {
        "id.orig_h": "src_ip",
        "id.resp_h": "dst_ip",
        "id.orig_p": "src_port",
        "id.resp_p": "dst_port",
        "SrcAddr": "src_ip",
        "DstAddr": "dst_ip",
        "Sport": "src_port",
        "Dport": "dst_port",
        "Proto": "proto",
        "Dur": "duration",
        "SrcBytes": "orig_bytes",
        "DstBytes": "resp_bytes",
        "StartTime": "ts",
    }
    out = df.rename(columns=rename_map).copy()

    if "ts" in out.columns:
        # Preserve numeric timestamps if already numeric; else parse datetime.
        if not np.issubdtype(out["ts"].dtype, np.number):
            out["ts"] = pd.to_datetime(out["ts"], errors="coerce").astype("int64") // 10**9
    elif "start_time" in out.columns:
        out["ts"] = pd.to_datetime(out["start_time"], errors="coerce").astype("int64") // 10**9

    # Fill numeric backbone.
    for col in ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "src_port", "dst_port"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Infer missing byte/packet split when only totals exist.
    if "resp_bytes" not in out.columns and "TotBytes" in out.columns:
        out["TotBytes"] = pd.to_numeric(out["TotBytes"], errors="coerce").fillna(0.0)
        out["orig_bytes"] = pd.to_numeric(out.get("orig_bytes", 0.0), errors="coerce").fillna(0.0)
        out["resp_bytes"] = (out["TotBytes"] - out["orig_bytes"]).clip(lower=0.0)

    if "orig_pkts" not in out.columns and "TotPkts" in out.columns:
        out["TotPkts"] = pd.to_numeric(out["TotPkts"], errors="coerce").fillna(0.0)
        out["orig_pkts"] = np.ceil(out["TotPkts"] / 2.0)
        out["resp_pkts"] = np.floor(out["TotPkts"] / 2.0)

    out["duration"] = pd.to_numeric(out.get("duration", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["orig_bytes"] = pd.to_numeric(out.get("orig_bytes", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["resp_bytes"] = pd.to_numeric(out.get("resp_bytes", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["orig_pkts"] = pd.to_numeric(out.get("orig_pkts", 1.0), errors="coerce").fillna(1.0).clip(lower=1.0)
    out["resp_pkts"] = pd.to_numeric(out.get("resp_pkts", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    required = {"ts", "src_ip", "dst_ip", "proto"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")

    # Derived per-flow features expected by session_builder feature mapping.
    total_pkts = out["orig_pkts"] + out["resp_pkts"]
    total_bytes = out["orig_bytes"] + out["resp_bytes"]
    out["bytes_per_pkt"] = total_bytes / (total_pkts + 1e-9)
    out["packet_ratio"] = out["orig_pkts"] / (out["resp_pkts"] + 1e-9)
    out["byte_ratio"] = out["orig_bytes"] / (out["resp_bytes"] + 1e-9)

    out["src_ip"] = out["src_ip"].astype(str)
    out["dst_ip"] = out["dst_ip"].astype(str)
    out["proto"] = out["proto"].astype(str).str.lower()

    src_is_private = out["src_ip"].apply(is_private_ip).astype(bool)
    dst_is_private = out["dst_ip"].apply(is_private_ip).astype(bool)
    out["is_outbound"] = (src_is_private & ~dst_is_private).astype(float)

    out = out.sort_values(["src_ip", "dst_ip", "proto", "ts"], kind="mergesort").reset_index(drop=True)
    grp = out.groupby(["src_ip", "dst_ip", "proto"], sort=False)
    out["iat"] = grp["ts"].diff().fillna(0.0).clip(lower=0.0)
    out["iat_delta"] = grp["iat"].diff().fillna(0.0)
    out["byte_delta"] = grp["orig_bytes"].diff().fillna(0.0)

    # Ensure all selected model features exist.
    missing_features = [c for c in FEATURE_NAMES if c not in out.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    cols = ["ts", "src_ip", "dst_ip", "proto"] + FEATURE_NAMES
    return out[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _load_flow_dataframe(path: str) -> pd.DataFrame:
    """Load raw flow file and return canonical dataframe for session building."""
    lower = path.lower()
    if lower.endswith(".binetflow"):
        return _standardize_columns(_read_binetflow(path))

    # Zeek conn.log detection by extension or the common marker in first line.
    if lower.endswith(".log") or "conn" in os.path.basename(lower):
        try:
            return process_conn_log(path, return_metadata=True)
        except Exception:
            # Fallback to generic parser for non-Zeek '.log' files.
            pass

    df = _read_generic_csv(path)
    return _standardize_columns(df)


def _build_labeled_sessions(paths: Iterable[str], label: int, min_flows: int, inactivity_timeout: float):
    sessions_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    masks_all: List[np.ndarray] = []

    for path in paths:
        print(f"[INFO] Processing file: {path}")
        df = _load_flow_dataframe(path)
        df = df.copy()
        df["label"] = int(label)

        sessions, labels, masks = build_sessions(
            df,
            label_col="label",
            inactivity_timeout=inactivity_timeout,
            min_flows=min_flows,
        )
        sessions_all.append(sessions)
        labels_all.append(labels)
        masks_all.append(masks)

    if not sessions_all:
        return np.empty((0, 20, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0, 20), dtype=bool)

    return np.concatenate(sessions_all, axis=0), np.concatenate(labels_all, axis=0), np.concatenate(masks_all, axis=0)


def build_mcfp_npz(
    c2_inputs: List[str],
    benign_inputs: Optional[List[str]],
    output_path: str,
    min_flows: int = 5,
    inactivity_timeout: float = 300.0,
) -> None:
    if not c2_inputs and not benign_inputs:
        raise ValueError("Provide at least one C2 or benign input file.")

    c2_sessions, c2_labels, c2_masks = _build_labeled_sessions(
        c2_inputs,
        LABEL_C2,
        min_flows=min_flows,
        inactivity_timeout=inactivity_timeout,
    )

    benign_sessions = np.empty((0, c2_sessions.shape[1] if c2_sessions.size else 20, c2_sessions.shape[2] if c2_sessions.size else len(FEATURE_NAMES)), dtype=np.float32)
    benign_labels = np.empty((0,), dtype=np.int64)
    benign_masks = np.empty((0, c2_masks.shape[1] if c2_masks.size else 20), dtype=bool)

    if benign_inputs:
        benign_sessions, benign_labels, benign_masks = _build_labeled_sessions(
            benign_inputs,
            LABEL_BENIGN,
            min_flows=min_flows,
            inactivity_timeout=inactivity_timeout,
        )

    X_parts = [arr for arr in [c2_sessions, benign_sessions] if arr.size > 0]
    y_parts = [arr for arr in [c2_labels, benign_labels] if arr.size > 0]
    m_parts = [arr for arr in [c2_masks, benign_masks] if arr.size > 0]

    if not X_parts:
        raise ValueError("No sessions were produced. Check inputs and min_flows.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    masks = np.concatenate(m_parts, axis=0)

    shuffle_idx = np.random.default_rng(42).permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    masks = masks[shuffle_idx]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, masks=masks, feature_names=np.array(FEATURE_NAMES))

    c2_count = int((y == LABEL_C2).sum())
    benign_count = int((y == LABEL_BENIGN).sum())
    print(f"[SUCCESS] Saved NPZ to {output_path}")
    print(f"[INFO] Sessions: total={len(y):,}, c2={c2_count:,}, benign={benign_count:,}")
    print(f"[INFO] Tensor shape: {X.shape} | Masks shape: {masks.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stratosphere MCFP sessions NPZ for zero-shot evaluation")
    parser.add_argument("--c2_inputs", nargs="*", default=[], help="List of malicious MCFP files")
    parser.add_argument("--benign_inputs", nargs="*", default=[], help="Optional list of benign flow files")
    parser.add_argument(
        "--output_path",
        default="data/processed/stratosphere_mcfp_sessions.npz",
        help="Output NPZ path",
    )
    parser.add_argument("--min_flows", type=int, default=5)
    parser.add_argument("--inactivity_timeout", type=float, default=300.0)
    args = parser.parse_args()

    build_mcfp_npz(
        c2_inputs=args.c2_inputs,
        benign_inputs=args.benign_inputs,
        output_path=args.output_path,
        min_flows=args.min_flows,
        inactivity_timeout=args.inactivity_timeout,
    )


if __name__ == "__main__":
    main()
