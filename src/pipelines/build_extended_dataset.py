"""Build extended_v1 dataset NPZ artifacts from normalized telemetry.

Usage:
    python -m src.pipelines.build_extended_dataset \\
        --conn-log  data/raw/ctu13/conn.log \\
        --ssl-log   data/raw/ctu13/ssl.log   \\
        --dns-log   data/raw/ctu13/dns.log   \\
        --out-dir   data/processed/extended/ctu13 \\
        --source    ctu13 \\
        --label     1

If ssl.log or dns.log are not available, those feature groups are
automatically zero-filled in the output tensors.

The output directory will contain:
    sessions_extended.npz   — (N, 20, 45) float32 tensors + labels + masks
    dataset_manifest.json   — sample counts, schema version, feature list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader.normalized_telemetry import (
    load_and_normalize_telemetry,
    normalize_telemetry_frame,
)
from src.features.extended_session_builder import (
    build_extended_sessions,
    save_extended_sessions,
)
from src.features.feature_config_extended import (
    FEATURE_DIM_EXTENDED,
    FEATURE_NAMES_EXTENDED,
    FEATURE_SCHEMA_VERSION,
)
from src.features.feature_config_experiment import SESSION_LEN
from src.features.zeek_parser import process_conn_log


def build_extended_dataset(
    conn_log_path: str,
    out_dir: str,
    *,
    ssl_log_path: str | None = None,
    dns_log_path: str | None = None,
    source_dataset: str | None = None,
    label: int | None = None,
    family: str = "",
    capture_id: str | None = None,
    min_flows: int = 1,
    inactivity_timeout: float = 300.0,
) -> dict:
    """Build extended_v1 NPZ from Zeek log files.

    Parameters
    ----------
    conn_log_path : Path to conn.log (required).
    out_dir : Output directory for NPZ and manifest.
    ssl_log_path : Path to ssl.log (optional — zero-fills TLS features).
    dns_log_path : Path to dns.log (optional — zero-fills DNS features).
    source_dataset : Dataset label (e.g. "ctu13").
    label : Binary label override (0=benign, 1=C2).
    family : C2 family name.
    capture_id : Capture identifier.
    min_flows : Minimum flows per session.
    inactivity_timeout : Session segmentation timeout.

    Returns
    -------
    dict with build statistics.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # --- 1. Load and normalize conn.log ---
    print(f"[1/4] Loading conn.log: {conn_log_path}")
    norm_df = load_and_normalize_telemetry(
        conn_log_path,
        source_dataset=source_dataset,
        label=label,
        family=family,
        capture_id=capture_id,
    )
    print(f"       → {len(norm_df):,} normalized flows")

    # --- 2. Load optional TLS context ---
    tls_df = None
    if ssl_log_path and os.path.exists(ssl_log_path):
        print(f"[2/4] Loading ssl.log: {ssl_log_path}")
        try:
            tls_df = _load_zeek_log(ssl_log_path)
            print(f"       → {len(tls_df):,} TLS records")
        except Exception as exc:
            print(f"       ⚠ Failed to load ssl.log: {exc}. TLS features will be zero-filled.")
            tls_df = None
    else:
        print("[2/4] No ssl.log provided — TLS features will be zero-filled.")

    # --- 3. Load optional DNS context ---
    dns_df = None
    if dns_log_path and os.path.exists(dns_log_path):
        print(f"[3/4] Loading dns.log: {dns_log_path}")
        try:
            dns_df = _load_zeek_log(dns_log_path)
            print(f"       → {len(dns_df):,} DNS records")
        except Exception as exc:
            print(f"       ⚠ Failed to load dns.log: {exc}. DNS features will be zero-filled.")
            dns_df = None
    else:
        print("[3/4] No dns.log provided — DNS features will be zero-filled.")

    # --- 4. Build extended sessions ---
    print("[4/4] Building extended_v1 sessions...")
    sessions, labels, masks = build_extended_sessions(
        norm_df,
        label_col="label",
        inactivity_timeout=inactivity_timeout,
        min_flows=min_flows,
        tls_df=tls_df,
        dns_df=dns_df,
    )

    # Save NPZ
    npz_path = os.path.join(out_dir, "sessions_extended.npz")
    save_extended_sessions(sessions, labels, masks, npz_path)

    labels_path = os.path.join(out_dir, "labels.csv")
    labels_df = pd.DataFrame({
        "session_index": np.arange(len(labels), dtype=np.int64),
        "label": labels.astype(np.int64),
    })
    labels_df.to_csv(labels_path, index=False)

    # Build manifest
    elapsed = time.time() - t0
    n_c2 = int((labels == 1).sum())
    n_benign = int((labels == 0).sum())

    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM_EXTENDED,
        "session_len": SESSION_LEN,
        "feature_names": list(FEATURE_NAMES_EXTENDED),
        "num_sessions": len(sessions),
        "num_c2": n_c2,
        "num_benign": n_benign,
        "c2_ratio": float(n_c2 / max(len(sessions), 1)),
        "source_dataset": source_dataset or "unknown",
        "family": family,
        "conn_log_path": str(conn_log_path),
        "ssl_log_path": str(ssl_log_path) if ssl_log_path else None,
        "dns_log_path": str(dns_log_path) if dns_log_path else None,
        "tls_features_available": tls_df is not None,
        "dns_features_available": dns_df is not None,
        "min_flows": min_flows,
        "inactivity_timeout": inactivity_timeout,
        "build_time_seconds": round(elapsed, 2),
        "npz_path": npz_path,
        "labels_path": labels_path,
    }

    manifest_path = os.path.join(out_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {manifest_path}")

    return manifest


def build_extended_dataset_from_dataframe(
    df: pd.DataFrame,
    out_dir: str,
    *,
    tls_df: pd.DataFrame | None = None,
    dns_df: pd.DataFrame | None = None,
    source_dataset: str = "unknown",
    min_flows: int = 1,
    inactivity_timeout: float = 300.0,
) -> dict:
    """Build extended_v1 NPZ from an already-normalized DataFrame.

    Use this when you have already called ``normalize_telemetry_frame()``
    or when building from an in-memory DataFrame (e.g. in tests or
    multi-source pipelines).
    """
    os.makedirs(out_dir, exist_ok=True)

    sessions, labels, masks = build_extended_sessions(
        df,
        label_col="label",
        inactivity_timeout=inactivity_timeout,
        min_flows=min_flows,
        tls_df=tls_df,
        dns_df=dns_df,
    )

    npz_path = os.path.join(out_dir, "sessions_extended.npz")
    save_extended_sessions(sessions, labels, masks, npz_path)

    labels_path = os.path.join(out_dir, "labels.csv")
    labels_df = pd.DataFrame({
        "session_index": np.arange(len(labels), dtype=np.int64),
        "label": labels.astype(np.int64),
    })
    labels_df.to_csv(labels_path, index=False)

    n_c2 = int((labels == 1).sum())
    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM_EXTENDED,
        "session_len": SESSION_LEN,
        "feature_names": list(FEATURE_NAMES_EXTENDED),
        "num_sessions": len(sessions),
        "num_c2": n_c2,
        "num_benign": int((labels == 0).sum()),
        "c2_ratio": float(n_c2 / max(len(sessions), 1)),
        "source_dataset": source_dataset,
        "tls_features_available": tls_df is not None,
        "dns_features_available": dns_df is not None,
        "npz_path": npz_path,
        "labels_path": labels_path,
    }

    manifest_path = os.path.join(out_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _load_zeek_log(path: str) -> pd.DataFrame:
    """Load a Zeek log file (ssl.log or dns.log) into a DataFrame."""
    # Try Zeek tab-separated format first
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Find header and data lines
    header_line = None
    data_lines = []
    for line in lines:
        if line.startswith("#fields"):
            header_line = line.strip()
        elif not line.startswith("#") and line.strip():
            data_lines.append(line.strip())

    if header_line and data_lines:
        fields = header_line.replace("#fields\t", "").replace("#fields ", "").split("\t")
        rows = [line.split("\t") for line in data_lines]
        df = pd.DataFrame(rows, columns=fields)
        # Replace Zeek "-" with NaN
        df = df.replace("-", np.nan)
        return df

    # Fallback: try CSV
    return pd.read_csv(path, sep=None, engine="python")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build extended_v1 dataset NPZ from Zeek logs"
    )
    parser.add_argument(
        "--conn-log", required=True,
        help="Path to conn.log (required)"
    )
    parser.add_argument(
        "--ssl-log", default=None,
        help="Path to ssl.log (optional)"
    )
    parser.add_argument(
        "--dns-log", default=None,
        help="Path to dns.log (optional)"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--source", default=None,
        help="Dataset source label"
    )
    parser.add_argument(
        "--label", type=int, default=None,
        help="Binary label (0=benign, 1=C2)"
    )
    parser.add_argument(
        "--family", default="",
        help="C2 family name"
    )
    parser.add_argument(
        "--min-flows", type=int, default=1,
        help="Minimum flows per session"
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Inactivity timeout for session segmentation"
    )

    args = parser.parse_args()
    manifest = build_extended_dataset(
        conn_log_path=args.conn_log,
        out_dir=args.out_dir,
        ssl_log_path=args.ssl_log,
        dns_log_path=args.dns_log,
        source_dataset=args.source,
        label=args.label,
        family=args.family,
        min_flows=args.min_flows,
        inactivity_timeout=args.timeout,
    )

    print(f"\n✅ Build complete: {manifest['num_sessions']:,} sessions "
          f"({manifest['schema_version']}, {manifest['feature_dim']}d)")


if __name__ == "__main__":
    main()
