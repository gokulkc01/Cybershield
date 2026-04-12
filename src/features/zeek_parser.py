"""
zeek_parser.py
==============
Zeek conn.log → clean DataFrame with engineered features.

Three bugs fixed vs original version:
  1. Rolling IAT removed — IAT stats belong in session_builder, not here.
     Here we only compute raw per-flow IAT (time since previous flow to
     same dst from same src). Session-level stats are aggregated later.
  2. Sort-before-diff enforced — df is sorted BEFORE groupby so that
     diff() always operates on chronologically ordered rows.
  3. Feature list aligned with feature_config.FEATURE_NAMES exactly.
     tcp_flag_ratio and tls_present are removed. packet_ratio and
     byte_ratio are added. All 12 features match the config.

Pipeline position:
    PCAP → Zeek → conn.log → THIS FILE → session_builder.py
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from src.features.feature_config import (
    FEATURE_NAMES,
    FEATURE_DIM,
    ZEEK_REQUIRED_FIELDS,
    is_private_ip,
)


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def get_zeek_headers(file_path: str) -> List[str]:
    """
    Extract column names from the Zeek log header line.
    Zeek writes headers as: #fields\tcol1\tcol2\t...
    Must be called before pd.read_csv so we have column names ready.
    """
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#fields"):
                return line.strip().split("\t")[1:]
    raise ValueError(
        f"No '#fields' line found in {file_path}. "
        "Is this a valid Zeek conn.log?"
    )


def load_zeek_conn_log(file_path: str) -> pd.DataFrame:
    """
    Load Zeek conn.log into a DataFrame.
    Extracts column names from the #fields header, then skips all
    comment lines (those beginning with #) when reading data rows.
    """
    headers = get_zeek_headers(file_path)
    df = pd.read_csv(
        file_path,
        sep="\t",
        names=headers,
        comment="#",      # skip #fields, #types, #close, etc.
        low_memory=False,
    )
    return df


# ── Step 2: Clean ─────────────────────────────────────────────────────────────

def preprocess_zeek(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize and clean a raw Zeek DataFrame.

    - Replace Zeek's '-' sentinel with NaN then 0
    - Coerce numeric columns
    - Drop zero-packet rows (noise / incomplete captures)
    - Validate required columns are present
    """
    # Check for required columns
    missing = [c for c in ZEEK_REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(
            f"conn.log is missing required fields: {missing}\n"
            f"Present fields: {list(df.columns)}"
        )

    # Zeek uses '-' for missing values
    df = df.replace("-", np.nan)

    numeric_cols = ["ts", "duration", "orig_bytes", "resp_bytes",
                    "orig_pkts", "resp_pkts"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Ensure non-negative — Zeek sometimes writes negative durations for
    # incomplete flows captured mid-handshake
    df["duration"]   = df["duration"].clip(lower=0.0)
    df["orig_bytes"] = df["orig_bytes"].clip(lower=0.0)
    df["resp_bytes"] = df["resp_bytes"].clip(lower=0.0)
    df["orig_pkts"]  = df["orig_pkts"].clip(lower=1.0)   # minimum 1 to avoid /0
    df["resp_pkts"]  = df["resp_pkts"].clip(lower=0.0)

    # Drop rows with no packets — these are connection attempts with no data
    df = df[(df["orig_pkts"] + df["resp_pkts"]) > 0].reset_index(drop=True)

    return df


# ── Step 3: Engineer features ──────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 12 model features from Zeek columns.

    IAT NOTE — what this function computes vs what session_builder computes:
      Here:           raw per-flow IAT = time diff from previous flow in same
                      (src_ip, dst_ip, proto) group. This is the raw signal.
      session_builder: aggregates these raw IATs into session-level stats
                      (mean, std, cv) over all flows in the session.
    Do NOT compute rolling statistics here — that belongs in session_builder.
    """

    # ── BUG FIX: sort BEFORE groupby ──────────────────────────────────────────
    # If df is unsorted, groupby["ts"].diff() computes diff on arbitrary row
    # order, giving nonsense IAT values. Sort globally first.
    df = df.sort_values(by=["id.orig_h", "id.resp_h", "proto", "ts"],
                        kind="mergesort").reset_index(drop=True)

    # ── Volume features ───────────────────────────────────────────────────────
    total_pkts  = df["orig_pkts"] + df["resp_pkts"]
    total_bytes = df["orig_bytes"] + df["resp_bytes"]

    df["bytes_per_pkt"] = total_bytes / (total_pkts + 1e-9)
    df["packet_ratio"]  = df["orig_pkts"] / (df["resp_pkts"] + 1e-9)
    df["byte_ratio"]    = df["orig_bytes"] / (df["resp_bytes"] + 1e-9)

    # ── Directionality ────────────────────────────────────────────────────────
    df["is_outbound"] = df["id.orig_h"].apply(is_private_ip)

    # ── Raw per-flow IAT (time since previous flow, same connection path) ─────
    # This is computed per (src, dst, proto) group.
    # The result is a single number per row — used later in session_builder
    # to compute session-level iat_mean / iat_std / iat_cv.
    grp = df.groupby(["id.orig_h", "id.resp_h", "proto"], sort=False)
    df["iat"] = grp["ts"].diff().fillna(0.0)
    df["iat"] = df["iat"].clip(lower=0.0)   # diff can be negative if sort failed
    df["iat_delta"] = grp["iat"].diff().fillna(0.0)
    df["byte_delta"] = grp["orig_bytes"].diff().fillna(0.0)

    # ── Rename Zeek fields to standard names ─────────────────────────────────
    df = df.rename(columns={
        "id.orig_h": "src_ip",
        "id.resp_h": "dst_ip",
        "id.orig_p": "src_port",
        "id.resp_p": "dst_port",
    })

    return df


# ── Step 4: Final feature matrix ──────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select exactly the features listed in FEATURE_NAMES and sanitize.

    NOTE: iat_mean, iat_std, iat_cv are NOT computed here.
    They require session-level aggregation (across multiple flows).
    session_builder.py fills them in. This function just ensures the
    other 9 features are present and clean. The 3 IAT aggregate features
    will be 0 here and overwritten by session_builder.
    """
    # Add placeholder IAT aggregate columns (session_builder will fill these)
    for col in ["iat_mean", "iat_std", "iat_cv"]:
        if col not in df.columns:
            df[col] = 0.0

    # Validate all features present
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise KeyError(
            f"Features not found after engineering: {missing}\n"
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    df_final = df[FEATURE_NAMES].copy()

    # Sanitize: replace inf/nan with 0
    df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Clip extreme outliers that would dominate normalization
    # These caps are generous — only filter truly corrupted values
    df_final["orig_bytes"]   = df_final["orig_bytes"].clip(upper=1e9)
    df_final["resp_bytes"]   = df_final["resp_bytes"].clip(upper=1e9)
    df_final["bytes_per_pkt"]= df_final["bytes_per_pkt"].clip(upper=65535)
    df_final["packet_ratio"] = df_final["packet_ratio"].clip(upper=1000)
    df_final["byte_ratio"]   = df_final["byte_ratio"].clip(upper=1000)
    df_final["iat_mean"]     = df_final["iat_mean"].clip(upper=3600)
    df_final["iat_std"]      = df_final["iat_std"].clip(upper=3600)
    df_final["iat_cv"]       = df_final["iat_cv"].clip(upper=100)

    assert df_final.shape[1] == FEATURE_DIM, (
        f"Expected {FEATURE_DIM} features, got {df_final.shape[1]}"
    )
    return df_final


# ── Full pipeline ──────────────────────────────────────────────────────────────

def process_conn_log(
    file_path: str,
    return_metadata: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: conn.log → clean DataFrame with 12 features + metadata.

    Parameters
    ----------
    file_path       : path to Zeek conn.log file
    return_metadata : if True, also return src_ip, dst_ip, ts, proto columns
                      (needed by session_builder for grouping)

    Returns
    -------
    If return_metadata=False: DataFrame with exactly FEATURE_NAMES columns (12)
    If return_metadata=True : DataFrame with FEATURE_NAMES + metadata columns
    """
    print(f"Loading {file_path}...")
    df = load_zeek_conn_log(file_path)
    print(f"  Loaded {len(df):,} rows")

    df = preprocess_zeek(df)
    print(f"  After cleaning: {len(df):,} rows")

    df = engineer_features(df)

    if return_metadata:
        # Keep metadata columns alongside features for session_builder
        meta_cols = ["ts", "src_ip", "dst_ip", "src_port", "dst_port",
                     "proto", "iat"]
        available_meta = [c for c in meta_cols if c in df.columns]
        df_out = pd.concat([
            df[available_meta].reset_index(drop=True),
            build_feature_matrix(df).reset_index(drop=True),
        ], axis=1)
        return df_out

    return build_feature_matrix(df)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.features.zeek_parser <path/to/conn.log>")
        sys.exit(1)

    path = sys.argv[1]
    df = process_conn_log(path, return_metadata=True)
    print(f"\nOutput shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    print(f"\nFeature stats:")
    from src.features.feature_config import FEATURE_NAMES
    print(df[FEATURE_NAMES].describe().round(4).to_string())