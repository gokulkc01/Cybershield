"""
ctu13_processor.py
==================
Processes CTU-13 .binetflow into session-based dataset.
Pipeline:
CTU-13 CSV → Standardized DataFrame → Feature Engineering → Labeling → Session Builder → NPZ
"""

import pandas as pd
import numpy as np
import os
import sys

# Allow direct execution: `python src/data_loader/ctu13_processor.py`
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.features.feature_config import (
    FEATURE_NAMES,
    label_from_string,
    is_private_ip
)
from src.features.session_builder import build_sessions

# =========================================================
# STEP 1: LOAD + STANDARDIZE CTU-13
# =========================================================
def load_and_standardize_ctu13(file_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading CTU-13 data from {file_path}...")
    # low_memory=False prevents mixed-type warnings on large CSVs
    df = pd.read_csv(file_path, low_memory=False)

    print("[INFO] Standardizing columns...")

    # -----------------------------
    # 1. Timestamp conversion (Corrected)
    # -----------------------------
    # astype('int64') is the safe, modern Pandas way to get epoch nanoseconds
    df['ts'] = pd.to_datetime(df['StartTime'], errors='coerce').astype('int64') // 10**9

    # Drop any rows where timestamp parsing failed
    df = df.dropna(subset=['ts']).copy()

    # -----------------------------
    # 2. Rename columns to pipeline format
    # -----------------------------
    rename_map = {
        'SrcAddr': 'src_ip',
        'DstAddr': 'dst_ip',
        'Proto': 'proto',
        'Sport': 'src_port',
        'Dport': 'dst_port',
        'Dur': 'duration',
        'SrcBytes': 'orig_bytes'
    }
    df = df.rename(columns=rename_map)

    # -----------------------------
    # 3. Numeric conversion
    # -----------------------------
    numeric_cols = ['duration', 'orig_bytes', 'TotBytes', 'TotPkts']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # -----------------------------
    # 4. Derive missing fields (Corrected)
    # -----------------------------
    df['resp_bytes'] = (df['TotBytes'] - df['orig_bytes']).clip(lower=0)

    # We cannot hardcode resp_pkts to 1. Without Zeek's exact count, 
    # the safest statistical fallback is a split to preserve the total volume.
    df['orig_pkts'] = np.ceil(df['TotPkts'] / 2)
    df['resp_pkts'] = np.floor(df['TotPkts'] / 2)

    # -----------------------------
    # 5. Feature Engineering
    # -----------------------------
    total_pkts = df["orig_pkts"] + df["resp_pkts"]
    total_bytes = df["orig_bytes"] + df["resp_bytes"]

    df["bytes_per_pkt"] = total_bytes / (total_pkts + 1e-9)
    df["packet_ratio"]  = df["orig_pkts"] / (df["resp_pkts"] + 1e-9)
    df["byte_ratio"]    = df["orig_bytes"] / (df["resp_bytes"] + 1e-9)

    # Vectorized outbound logic (Massively faster than axis=1 apply)
    print("[INFO] Computing directionality...")
    src_is_private = df["src_ip"].apply(is_private_ip).astype(bool)
    dst_is_private = df["dst_ip"].apply(is_private_ip).astype(bool)
    
    # 1 if source is private AND destination is public, else 0
    df["is_outbound"] = (src_is_private & ~dst_is_private).astype(int)

    # -----------------------------
    # 6. Sort BEFORE IAT computation
    # -----------------------------
    print("[INFO] Sorting flows chronologically...")
    df = df.sort_values(
        by=["src_ip", "dst_ip", "proto", "ts"],
        kind="mergesort"
    ).reset_index(drop=True)

    # -----------------------------
    # 7. Compute IAT (per flow)
    # -----------------------------
    print("[INFO] Computing per-flow IAT...")
    grp = df.groupby(["src_ip", "dst_ip", "proto"], sort=False)
    df["iat"] = grp["ts"].diff().fillna(0.0).clip(lower=0.0)
    df["iat_delta"] = grp["iat"].diff().fillna(0.0)
    df["byte_delta"] = grp["orig_bytes"].diff().fillna(0.0)

    return df

# =========================================================
# STEP 2: LABELING
# =========================================================
def apply_ctu13_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Applying binary labels...")
    df['label'] = df['Label'].astype(str).apply(label_from_string)
    return df

# =========================================================
# STEP 3: PIPELINE EXECUTION
# =========================================================
def execute_pipeline(input_path: str, output_path: str):
    # Load + standardize
    df = load_and_standardize_ctu13(input_path)

    # Label
    df = apply_ctu13_labels(df)

    # -----------------------------
    # Feature validation
    # -----------------------------
    required_cols = ["ts", "src_ip", "dst_ip", "proto"] + FEATURE_NAMES + ["label"]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"[ERROR] Missing required columns: {missing}")

    df = df[required_cols].copy()

    # Neural networks hate NaN/Inf. Final sanitization pass.
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # -----------------------------
    # Build sessions
    # -----------------------------
    print("[INFO] Handing off to Session Builder...")
    sessions, labels, masks = build_sessions(df)

    # -----------------------------
    # Save dataset
    # -----------------------------
    print(f"[INFO] Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez(
        output_path,
        X=sessions,
        y=labels,
        masks=masks,
        feature_names=np.array(FEATURE_NAMES),
    )

    print(f"[SUCCESS] Saved {len(sessions)} sequences to disk.")

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    # Fixed the missing underscores in __name__ == "__main__"
    input_file = "data/raw/ctu13/capture20110810.binetflow"
    output_file = "data/processed/ctu13_c2_sessions.npz"

    if os.path.exists(input_file):
        execute_pipeline(input_file, output_file)
    else:
        print(f"[ERROR] File not found: {input_file}")
        print("Please run: wget -P data/raw/ctu13 https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset/1/capture20110810.binetflow")
        sys.exit(1)
