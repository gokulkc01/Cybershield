"""UWF-ZeekData24 dataset processor for CyberShield v2.

Handles ingestion of the University of West Florida ZeekData24 dataset,
which provides labeled Zeek logs for MITRE ATT&CK techniques including
TA0011 (Command and Control) along with benign traffic.

Dataset structure (expected):
    uwf_zeekdata24/
    ├── benign/
    │   ├── conn.log
    │   ├── dns.log        (optional, for Phase 1 TLS/DNS features)
    │   ├── ssl.log        (optional)
    │   └── ...
    ├── TA0011/            (Command and Control)
    │   ├── <technique_dirs>/
    │   │   ├── conn.log
    │   │   └── ...
    │   └── ...
    └── labels.csv         (optional, for ground-truth mapping)

Usage:
    python -m src.data_loader.uwf_processor \\
        --data_dir data/raw/uwf_zeekdata24 \\
        --output_dir data/processed/uwf \\
        --min_flows 3
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.feature_config_v2 import (
    C2_POSITIVE_LABELS,
    DatasetSource,
    FEATURE_NAMES,
    LABEL_BENIGN,
    LABEL_C2,
    SessionMetadata,
)
from src.features.session_builder import build_sessions, save_sessions
from src.features.zeek_parser import process_conn_log


# ──────────────────────────────────────────────────────────────────────
# UWF directory scanning
# ──────────────────────────────────────────────────────────────────────

def discover_conn_logs(data_dir: str) -> Dict[str, List[str]]:
    """Scan UWF directory structure and classify conn.log files.

    Returns
    -------
    Dict with keys 'c2' and 'benign', each containing lists of
    absolute paths to conn.log files.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"UWF data directory not found: {data_dir}")

    result: Dict[str, List[str]] = {"c2": [], "benign": []}

    # Find all conn.log files recursively
    conn_logs = list(data_path.rglob("conn.log"))
    if not conn_logs:
        # Also try common alternative names
        conn_logs = list(data_path.rglob("conn*.log"))

    for log_path in sorted(conn_logs):
        # Classify based on parent directory structure
        rel_parts = log_path.relative_to(data_path).parts
        path_str = "/".join(rel_parts).lower()

        if _is_c2_path(path_str):
            result["c2"].append(str(log_path))
        elif _is_benign_path(path_str):
            result["benign"].append(str(log_path))
        else:
            # Unknown category — default to benign with warning
            print(f"[WARN] Unknown category for {log_path}, defaulting to benign")
            result["benign"].append(str(log_path))

    print(f"[INFO] Discovered {len(result['c2'])} C2 conn.logs, "
          f"{len(result['benign'])} benign conn.logs")
    return result


def _is_c2_path(path_str: str) -> bool:
    """Check if a path corresponds to C2 traffic based on naming conventions."""
    c2_indicators = [
        "ta0011", "command_and_control", "command-and-control",
        "c2", "c&c", "beacon", "implant",
        # MITRE ATT&CK sub-techniques under TA0011
        "t1071",  # Application Layer Protocol
        "t1132",  # Data Encoding
        "t1001",  # Data Obfuscation
        "t1568",  # Dynamic Resolution
        "t1573",  # Encrypted Channel
        "t1008",  # Fallback Channels
        "t1105",  # Ingress Tool Transfer
        "t1104",  # Multi-Stage Channels
        "t1095",  # Non-Application Layer Protocol
        "t1571",  # Non-Standard Port
        "t1572",  # Protocol Tunneling
        "t1090",  # Proxy
        "t1219",  # Remote Access Software
        "t1205",  # Traffic Signaling
        "t1102",  # Web Service
        # Common C2 framework names
        "sliver", "havoc", "cobalt", "metasploit", "meterpreter",
        "mythic", "brute_ratel", "covenant", "poshc2",
        # Malware families
        "emotet", "trickbot", "qakbot", "neris",
    ]
    return any(indicator in path_str for indicator in c2_indicators)


def _is_benign_path(path_str: str) -> bool:
    """Check if a path corresponds to benign traffic."""
    benign_indicators = [
        "benign", "normal", "legitimate", "clean", "baseline",
    ]
    return any(indicator in path_str for indicator in benign_indicators)


def _infer_c2_family(path_str: str) -> str:
    """Attempt to infer C2 family/technique from path."""
    families = {
        "sliver": "sliver",
        "havoc": "havoc",
        "cobalt": "cobalt_strike",
        "metasploit": "metasploit",
        "meterpreter": "metasploit",
        "mythic": "mythic",
        "brute_ratel": "brute_ratel",
        "covenant": "covenant",
        "poshc2": "poshc2",
        "emotet": "emotet",
        "trickbot": "trickbot",
        "qakbot": "qakbot",
    }
    path_lower = path_str.lower()
    for keyword, family in families.items():
        if keyword in path_lower:
            return family

    # Try MITRE technique ID
    import re
    tech_match = re.search(r"t\d{4}", path_lower)
    if tech_match:
        return f"mitre_{tech_match.group()}"

    return "unknown_c2"


# ──────────────────────────────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────────────────────────────

def process_uwf_conn_log(
    file_path: str,
    label: int,
    capture_id: str = "",
) -> pd.DataFrame:
    """Process a single UWF conn.log into a standardized flow DataFrame.

    Parameters
    ----------
    file_path : Path to conn.log file.
    label : LABEL_C2 or LABEL_BENIGN.
    capture_id : Identifier for this specific capture.

    Returns
    -------
    DataFrame with columns: ts, src_ip, dst_ip, proto, <FEATURE_NAMES>, label
    """
    try:
        df = process_conn_log(file_path, return_metadata=True)
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_path}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Add label column
    df["label"] = label

    # Add source tracking columns
    df["_source"] = DatasetSource.UWF_ZEEKDATA24.value
    df["_capture_id"] = capture_id

    return df


def build_uwf_sessions(
    data_dir: str,
    output_dir: str,
    min_flows: int = 3,
    inactivity_timeout: float = 300.0,
    max_benign_files: Optional[int] = None,
    max_c2_files: Optional[int] = None,
) -> Dict[str, str]:
    """Build session NPZ files from UWF-ZeekData24 dataset.

    Parameters
    ----------
    data_dir : Root directory of UWF-ZeekData24 dataset.
    output_dir : Directory to write output NPZ files.
    min_flows : Minimum flows per session.
    inactivity_timeout : Seconds of inactivity for session boundary.
    max_benign_files : Cap on number of benign files to process (None = all).
    max_c2_files : Cap on number of C2 files to process (None = all).

    Returns
    -------
    Dict mapping output type to file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Discover files
    discovered = discover_conn_logs(data_dir)
    c2_files = discovered["c2"]
    benign_files = discovered["benign"]

    if max_c2_files is not None:
        c2_files = c2_files[:max_c2_files]
    if max_benign_files is not None:
        benign_files = benign_files[:max_benign_files]

    outputs = {}

    # --- Process C2 traffic ---
    if c2_files:
        print(f"\n[PHASE] Processing {len(c2_files)} C2 conn.log files...")
        c2_dfs = []
        for path in tqdm(c2_files, desc="C2 files"):
            capture_id = _capture_id_from_path(path, data_dir)
            df = process_uwf_conn_log(path, LABEL_C2, capture_id=capture_id)
            if not df.empty:
                c2_dfs.append(df)

        if c2_dfs:
            c2_combined = pd.concat(c2_dfs, ignore_index=True)
            print(f"[INFO] Combined C2 flows: {len(c2_combined):,}")

            c2_sessions, c2_labels, c2_masks = build_sessions(
                c2_combined,
                label_col="label",
                inactivity_timeout=inactivity_timeout,
                min_flows=min_flows,
            )

            c2_path = os.path.join(output_dir, "uwf_c2_sessions.npz")
            save_sessions(c2_sessions, c2_labels, c2_masks, c2_path)
            outputs["c2"] = c2_path

    # --- Process benign traffic ---
    if benign_files:
        print(f"\n[PHASE] Processing {len(benign_files)} benign conn.log files...")
        benign_dfs = []
        for path in tqdm(benign_files, desc="Benign files"):
            capture_id = _capture_id_from_path(path, data_dir)
            df = process_uwf_conn_log(path, LABEL_BENIGN, capture_id=capture_id)
            if not df.empty:
                benign_dfs.append(df)

        if benign_dfs:
            benign_combined = pd.concat(benign_dfs, ignore_index=True)
            print(f"[INFO] Combined benign flows: {len(benign_combined):,}")

            b_sessions, b_labels, b_masks = build_sessions(
                benign_combined,
                label_col="label",
                inactivity_timeout=inactivity_timeout,
                min_flows=min_flows,
            )

            benign_path = os.path.join(output_dir, "uwf_benign_sessions.npz")
            save_sessions(b_sessions, b_labels, b_masks, benign_path)
            outputs["benign"] = benign_path

    # --- Combined output ---
    if "c2" in outputs and "benign" in outputs:
        from src.features.session_builder import load_sessions

        c2_s, c2_l, c2_m = load_sessions(outputs["c2"])
        b_s, b_l, b_m = load_sessions(outputs["benign"])

        X = np.concatenate([c2_s, b_s], axis=0)
        y = np.concatenate([c2_l, b_l], axis=0)
        masks = np.concatenate([c2_m, b_m], axis=0)

        # Shuffle
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(y))
        X, y, masks = X[idx], y[idx], masks[idx]

        combined_path = os.path.join(output_dir, "uwf_combined_sessions.npz")
        save_sessions(X, y, masks, combined_path)
        outputs["combined"] = combined_path

    print(f"\n[SUCCESS] UWF processing complete. Outputs: {outputs}")
    return outputs


def _capture_id_from_path(file_path: str, data_dir: str) -> str:
    """Generate a capture ID from the file's relative path."""
    try:
        rel = os.path.relpath(file_path, data_dir)
        # Use parent directory name(s) as capture ID
        parts = Path(rel).parts[:-1]  # Exclude the filename
        return "/".join(parts) if parts else "root"
    except ValueError:
        return os.path.basename(os.path.dirname(file_path))


# ──────────────────────────────────────────────────────────────────────
# Supplementary log support (Phase 1 preparation)
# ──────────────────────────────────────────────────────────────────────

def discover_supplementary_logs(data_dir: str) -> Dict[str, List[str]]:
    """Find dns.log and ssl.log files for future TLS/DNS feature extraction.

    Returns
    -------
    Dict with keys 'dns' and 'ssl', each containing lists of file paths.
    """
    data_path = Path(data_dir)
    return {
        "dns": sorted(str(p) for p in data_path.rglob("dns.log")),
        "ssl": sorted(str(p) for p in data_path.rglob("ssl.log")),
    }


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process UWF-ZeekData24 dataset for CyberShield v2"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root directory of UWF-ZeekData24 dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/uwf",
        help="Output directory for session NPZ files",
    )
    parser.add_argument("--min_flows", type=int, default=3)
    parser.add_argument("--inactivity_timeout", type=float, default=300.0)
    parser.add_argument(
        "--max_benign_files", type=int, default=None,
        help="Limit number of benign files to process",
    )
    parser.add_argument(
        "--max_c2_files", type=int, default=None,
        help="Limit number of C2 files to process",
    )
    args = parser.parse_args()

    build_uwf_sessions(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_flows=args.min_flows,
        inactivity_timeout=args.inactivity_timeout,
        max_benign_files=args.max_benign_files,
        max_c2_files=args.max_c2_files,
    )


if __name__ == "__main__":
    main()
