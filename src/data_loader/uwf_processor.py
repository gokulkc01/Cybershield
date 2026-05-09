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
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_loader.normalized_telemetry import (
    NORMALIZED_SCHEMA_COLUMNS,
    SessionArtifact,
    artifacts_to_arrays,
    build_session_artifacts,
    describe_pipeline,
    infer_label_and_family_from_path,
    infer_family_from_path,
    infer_source_dataset,
    load_and_normalize_telemetry,
    write_host_map_json,
    write_labels_csv,
)
from src.features.feature_config_v2 import DatasetSource, LABEL_BENIGN, LABEL_C2
from src.features.session_builder import save_sessions


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


SUPPORTED_INGESTION_SUFFIXES = {".parquet", ".pq", ".csv", ".tsv", ".jsonl", ".ndjson", ".log"}
SUPPLEMENTARY_LOG_NAMES = {"dns.log", "ssl.log"}


def _is_flow_like_file(path: Path) -> bool:
    """Return True for files that look like flow/session telemetry."""
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INGESTION_SUFFIXES:
        return False
    if name == "conn.log":
        return True
    if suffix in {".csv", ".tsv", ".jsonl", ".ndjson", ".parquet", ".pq"}:
        return True
    if suffix == ".log" and any(token in name for token in ("conn", "flow", "session", "traffic")):
        return True
    return False


def discover_raw_inputs(data_dir: str) -> Dict[str, List[str]]:
    """Discover flow-like telemetry files and supplementary Zeek logs."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"UWF data directory not found: {data_dir}")

    candidate_files: List[str] = []
    supplementary_logs: List[str] = []

    for path in sorted(data_path.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in SUPPLEMENTARY_LOG_NAMES:
            supplementary_logs.append(str(path))
            continue
        if _is_flow_like_file(path):
            candidate_files.append(str(path))

    print(
        f"[INFO] Discovered {len(candidate_files)} flow-like raw inputs and "
        f"{len(supplementary_logs)} supplementary logs"
    )
    return {"candidates": candidate_files, "supplementary": supplementary_logs}


def _write_schema_manifest(output_dir: str) -> str:
    manifest = {
        "schema_name": "CyberShield Unified Telemetry Schema",
        "description": (
            "Canonical telemetry schema used to normalize raw telemetry before "
            "host-centric sessionization."
        ),
        "pipeline": describe_pipeline(),
        "columns": list(NORMALIZED_SCHEMA_COLUMNS),
        "notes": [
            "timestamp is canonical; ts is a compatibility alias",
            "host_id is source-IP centric for longitudinal modeling",
            "labels.csv records session-level provenance and joins",
            "host_map.json summarizes host continuity and role continuity",
        ],
    }
    path = os.path.join(output_dir, "normalized_schema.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path

def process_uwf_conn_log(
    file_path: str,
    label: int,
    capture_id: str = "",
) -> pd.DataFrame:
    """Backward-compatible wrapper that now uses the flexible normalizer."""
    try:
        df = load_and_normalize_telemetry(
            file_path,
            source_dataset=DatasetSource.UWF_ZEEKDATA24.value,
            label=label,
            family=infer_family_from_path(file_path) if label == 1 else "",
            capture_id=capture_id,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to normalize {file_path}: {exc}")
        return pd.DataFrame()
    return df


def build_uwf_sessions(
    data_dir: str,
    output_dir: str,
    min_flows: int = 1,
    inactivity_timeout: float = 300.0,
    max_benign_files: Optional[int] = None,
    max_c2_files: Optional[int] = None,
) -> Dict[str, str]:
    """Build session NPZ files plus metadata sidecars from UWF-ZeekData24.

    Outputs:
    - uwf_c2_sessions.npz
    - uwf_benign_sessions.npz
    - uwf_combined_sessions.npz (if both classes exist)
    - labels.csv
    - host_map.json
    - normalized_schema.json
    """
    os.makedirs(output_dir, exist_ok=True)

    discovered = discover_raw_inputs(data_dir)
    raw_files = discovered["candidates"]
    if not raw_files:
        raise ValueError(f"No supported raw telemetry files found in {data_dir}")

    c2_files: List[str] = []
    benign_files: List[str] = []
    for file_path in raw_files:
        label, _family = infer_label_and_family_from_path(file_path)
        if label == LABEL_C2:
            c2_files.append(file_path)
        else:
            benign_files.append(file_path)

    if max_c2_files is not None:
        c2_files = c2_files[:max_c2_files]
    if max_benign_files is not None:
        benign_files = benign_files[:max_benign_files]

    outputs: Dict[str, str] = {}
    all_artifacts: List[SessionArtifact] = []
    host_map: Dict[str, Dict] = {}

    def _merge_host_map(target: Dict[str, Dict], source: Dict[str, Dict]) -> None:
        for host_id, payload in source.items():
            if host_id not in target:
                target[host_id] = payload
                continue
            merged = target[host_id]
            merged["source_datasets"] = sorted(set(merged.get("source_datasets", [])) | set(payload.get("source_datasets", [])))
            merged["source_paths"] = sorted(set(merged.get("source_paths", [])) | set(payload.get("source_paths", [])))
            merged["captures"] = sorted(set(merged.get("captures", [])) | set(payload.get("captures", [])))
            merged["labels"] = sorted(set(merged.get("labels", [])) | set(payload.get("labels", [])))
            merged["families"] = sorted(set(merged.get("families", [])) | set(payload.get("families", [])))
            merged["roles"] = sorted(set(merged.get("roles", [])) | set(payload.get("roles", [])))
            merged["session_count"] = int(merged.get("session_count", 0)) + int(payload.get("session_count", 0))
            merged["first_seen_ts"] = min(float(merged.get("first_seen_ts", 0.0)), float(payload.get("first_seen_ts", 0.0)))
            merged["last_seen_ts"] = max(float(merged.get("last_seen_ts", 0.0)), float(payload.get("last_seen_ts", 0.0)))

    def _process_group(paths: Sequence[str], group_name: str) -> List[SessionArtifact]:
        group_artifacts: List[SessionArtifact] = []
        if not paths:
            return group_artifacts

        print(f"\n[PHASE] Processing {len(paths)} {group_name} files...")
        for path in paths:
            try:
                normalized = load_and_normalize_telemetry(
                    path,
                    source_dataset=DatasetSource.UWF_ZEEKDATA24.value,
                    label=None,
                    family="",
                    capture_id=_capture_id_from_path(path, data_dir),
                )
            except Exception as exc:
                print(f"[ERROR] Skipping {path}: {exc}")
                continue

            artifacts, local_host_map = build_session_artifacts(
                normalized,
                label_col="label",
                inactivity_timeout=inactivity_timeout,
                min_flows=min_flows,
            )
            if not artifacts:
                print(f"[WARN] No sessions produced for {path}")
                continue

            group_artifacts.extend(artifacts)
            _merge_host_map(host_map, {host_id: entry.to_dict() for host_id, entry in local_host_map.items()})

            label_name = "C2" if group_name == "C2" else "benign"
            family_name = infer_family_from_path(path) if group_name == "C2" else "n/a"
            print(
                f"[INFO] {Path(path).name}: sessions={len(artifacts):,}, "
                f"label={label_name}, family={family_name}, source={infer_source_dataset(path)}"
            )

        return group_artifacts

    c2_artifacts = _process_group(c2_files, "C2")
    benign_artifacts = _process_group(benign_files, "benign")
    all_artifacts.extend(c2_artifacts)
    all_artifacts.extend(benign_artifacts)

    if not all_artifacts:
        raise ValueError("No sessions were produced. Check the raw inputs and min_flows setting.")

    def _save_group(artifacts: List[SessionArtifact], file_name: str) -> Optional[str]:
        if not artifacts:
            return None
        sessions, labels, masks = artifacts_to_arrays(artifacts)
        out_path = os.path.join(output_dir, file_name)
        save_sessions(sessions, labels, masks, out_path)
        return out_path

    c2_path = _save_group(c2_artifacts, "uwf_c2_sessions.npz")
    if c2_path:
        outputs["c2"] = c2_path

    benign_path = _save_group(benign_artifacts, "uwf_benign_sessions.npz")
    if benign_path:
        outputs["benign"] = benign_path

    if c2_artifacts and benign_artifacts:
        combined_artifacts = c2_artifacts + benign_artifacts
        rng = np.random.default_rng(42)
        order = rng.permutation(len(combined_artifacts))
        combined_artifacts = [combined_artifacts[i] for i in order]
        combined_path = _save_group(combined_artifacts, "uwf_combined_sessions.npz")
        if combined_path:
            outputs["combined"] = combined_path

    labels_csv_path = os.path.join(output_dir, "labels.csv")
    write_labels_csv(all_artifacts, labels_csv_path)
    outputs["labels_csv"] = labels_csv_path

    host_map_path = os.path.join(output_dir, "host_map.json")
    write_host_map_json(host_map, host_map_path)
    outputs["host_map"] = host_map_path

    schema_path = _write_schema_manifest(output_dir)
    outputs["schema_manifest"] = schema_path

    print(f"\n[INFO] Pipeline sketch: {describe_pipeline()}")
    print(f"[SUCCESS] UWF processing complete. Outputs: {outputs}")
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
    parser.add_argument("--min_flows", type=int, default=1)
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
