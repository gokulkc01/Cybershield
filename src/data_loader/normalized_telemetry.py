"""Unified telemetry normalization and metadata helpers for CyberShield.

This module defines the canonical normalized schema used by the ingestion
pipeline and provides helpers for:
- loading raw tabular telemetry from CSV / TSV / JSONL / Parquet / Zeek logs
- mapping dataset-specific aliases into a single schema
- producing session-level metadata artifacts
- writing `labels.csv` and `host_map.json` sidecars

Canonical normalized schema
---------------------------
The normalized dataframe is expected to carry at least:

    timestamp, ts, src_ip, dst_ip, proto, duration,
    orig_bytes, resp_bytes, orig_pkts, resp_pkts,
    src_port, dst_port, is_outbound,
    label, family, host_id, source_dataset,
    capture_id, source_path

The `timestamp` column is canonical. `ts` is kept as a compatibility alias
for downstream session builders that still expect `ts`.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features.feature_config_v2 import DatasetSource, LABEL_BENIGN, LABEL_C2, is_private_ip
from src.features.session_builder import segment_by_inactivity, session_to_tensor
from src.features.zeek_parser import process_conn_log
from src.features.feature_config_v2 import FEATURE_NAMES, SESSION_LEN, FEATURE_DIM


NORMALIZED_SCHEMA_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "ts",
    "src_ip",
    "dst_ip",
    "proto",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "bytes_per_pkt",
    "packet_ratio",
    "byte_ratio",
    "is_outbound",
    "src_port",
    "dst_port",
    "iat",
    "iat_delta",
    "byte_delta",
    "label",
    "family",
    "host_id",
    "source_dataset",
    "capture_id",
    "source_path",
)


_ALIASES: Dict[str, Tuple[str, ...]] = {
    "timestamp": (
        "ts",
        "time",
        "start_time",
        "starttime",
        "datetime",
        "date_time",
    ),
    "src_ip": ("id.orig_h", "srcaddr", "src", "source", "src_ip_addr", "src_ip_zeek"),
    "dst_ip": ("id.resp_h", "dstaddr", "dst", "destination", "dst_ip_addr", "dest_ip_zeek"),
    "proto": ("protocol", "proto_name"),
    "duration": ("dur", "flow_duration", "elapsed"),
    "orig_bytes": ("src_bytes", "origbytes", "bytes_sent", "totbytes_src"),
    "resp_bytes": ("dst_bytes", "respbytes", "bytes_recv", "totbytes_dst"),
    "orig_pkts": ("src_pkts", "origpkts", "packets_sent", "totpkts_src"),
    "resp_pkts": ("dst_pkts", "resppkts", "packets_recv", "totpkts_dst"),
    "src_port": ("sport", "srcport", "id.orig_p", "src_port_zeek"),
    "dst_port": ("dport", "dstport", "id.resp_p", "dest_port_zeek"),
    "label": ("class", "y", "label_binary"),
    "family": ("c2_family", "technique", "attack_family", "label_tactic", "label_technique"),
    "host_id": ("host", "src_host", "source_host"),
    "source_dataset": ("dataset", "source", "dataset_source"),
    "capture_id": ("capture", "scenario", "capture_name"),
    "source_path": ("path", "file_path", "origin_path"),
}


_C2_PATH_HINTS = (
    "ta0011",
    "command_and_control",
    "command-and-control",
    "c2",
    "c&c",
    "beacon",
    "implant",
    "sliver",
    "havoc",
    "cobalt",
    "metasploit",
    "meterpreter",
    "mythic",
    "brute_ratel",
    "covenant",
    "poshc2",
    "emotet",
    "trickbot",
    "qakbot",
    "neris",
)


@dataclass(frozen=True)
class SessionAnnotation:
    """Session-level metadata row for `labels.csv`."""

    sample_id: str
    host_id: str
    label: int
    family: str
    source_dataset: str
    capture_id: str
    source_path: str
    timestamp_start: float
    timestamp_end: float
    protocol: str
    dest_id: str
    flow_count: int


@dataclass
class HostMapEntry:
    """Aggregated host metadata for `host_map.json`."""

    host_id: str
    source_datasets: set[str] = field(default_factory=set)
    source_paths: set[str] = field(default_factory=set)
    captures: set[str] = field(default_factory=set)
    labels: set[int] = field(default_factory=set)
    families: set[str] = field(default_factory=set)
    session_count: int = 0
    first_seen_ts: float = math.inf
    last_seen_ts: float = 0.0
    roles: set[str] = field(default_factory=set)

    def ingest(self, ann: SessionAnnotation) -> None:
        self.source_datasets.add(ann.source_dataset)
        self.source_paths.add(ann.source_path)
        self.captures.add(ann.capture_id)
        self.labels.add(int(ann.label))
        if ann.family:
            self.families.add(ann.family)
        self.session_count += 1
        self.first_seen_ts = min(self.first_seen_ts, ann.timestamp_start)
        self.last_seen_ts = max(self.last_seen_ts, ann.timestamp_end)
        self.roles.add("source_host")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_id": self.host_id,
            "source_datasets": sorted(self.source_datasets),
            "source_paths": sorted(self.source_paths),
            "captures": sorted(self.captures),
            "labels": sorted(int(v) for v in self.labels),
            "families": sorted(self.families),
            "session_count": self.session_count,
            "first_seen_ts": 0.0 if math.isinf(self.first_seen_ts) else float(self.first_seen_ts),
            "last_seen_ts": float(self.last_seen_ts),
            "roles": sorted(self.roles),
        }


@dataclass(frozen=True)
class SessionArtifact:
    """Container for one session tensor and its provenance metadata."""

    sample_id: str
    host_id: str
    label: int
    family: str
    source_dataset: str
    capture_id: str
    source_path: str
    timestamp_start: float
    timestamp_end: float
    protocol: str
    dest_id: str
    flow_count: int
    tensor: np.ndarray
    mask: np.ndarray


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def infer_source_dataset(path: str) -> str:
    """Infer a dataset label from a path.

    This is intentionally conservative and used only when the caller does not
    provide an explicit source_dataset value.
    """
    lower = path.lower()
    if "uwf" in lower:
        return DatasetSource.UWF_ZEEKDATA24.value
    if "ctu13" in lower:
        return DatasetSource.CTU13.value
    if "mcfp" in lower:
        return DatasetSource.MCFP_STRATOSPHERE.value
    if "ciciot" in lower:
        return DatasetSource.CICIOT2023.value
    if "cicids" in lower:
        return DatasetSource.CICIDS2018.value
    return DatasetSource.UNKNOWN.value


def infer_label_and_family_from_path(path: str) -> Tuple[int, str]:
    """Infer a binary label and C2 family from a filesystem path."""
    lower = path.lower()
    if any(hint in lower for hint in _C2_PATH_HINTS):
        return LABEL_C2, infer_family_from_path(path)
    if any(hint in lower for hint in ("benign", "normal", "legitimate", "clean", "baseline")):
        return LABEL_BENIGN, ""
    return LABEL_BENIGN, ""


def infer_family_from_path(path: str) -> str:
    """Infer a family / technique label from a filesystem path."""
    lower = path.lower()
    patterns = (
        (r"cobalt[_\-]?strike", "cobalt_strike"),
        (r"brute[_\-]?ratel", "brute_ratel"),
        (r"meterpreter", "metasploit"),
        (r"sliver", "sliver"),
        (r"havoc", "havoc"),
        (r"metasploit", "metasploit"),
        (r"mythic", "mythic"),
        (r"covenant", "covenant"),
        (r"poshc2", "poshc2"),
        (r"emotet", "emotet"),
        (r"trickbot", "trickbot"),
        (r"qakbot", "qakbot"),
        (r"neris", "neris"),
        (r"rbot", "rbot"),
        (r"virut", "virut"),
        (r"menti", "menti"),
        (r"sogou", "sogou"),
        (r"murlo", "murlo"),
    )
    for pattern, family in patterns:
        if re.search(pattern, lower, flags=re.I):
            return family

    tech_match = re.search(r"t\d{4}", lower)
    if tech_match:
        return f"mitre_{tech_match.group()}"

    if "ctu13" in lower:
        return "ctu13_mixed"
    if "uwf" in lower:
        return "uwf_mixed"
    if "mcfp" in lower:
        return "mcfp_mixed"
    return "unknown"


def _load_tabular_frame(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    lower_name = Path(path).name.lower()

    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read parquet file {path}. Install pyarrow/fastparquet if needed."
            ) from exc

    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)

    if suffix in {".log", ".conn"} and "conn" in lower_name:
        return process_conn_log(path, return_metadata=True)

    # Use the Python engine for mixed delimiter / schema discovery, but avoid
    # `low_memory` because pandas does not support it with the Python engine.
    return pd.read_csv(path, sep=None, engine="python")


def _rename_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = {_normalize_key(col): col for col in df.columns}
    rename_map: Dict[str, str] = {}

    for canonical, aliases in _ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in (canonical, *aliases):
            normalized = _normalize_key(alias)
            if normalized in columns:
                rename_map[columns[normalized]] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _coerce_timestamp_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    out = pd.Series(np.zeros(len(series), dtype=np.float64), index=series.index)
    valid = parsed.notna()
    if valid.any():
        out.loc[valid] = parsed.loc[valid].astype("int64") / 10**9
    return out


def normalize_telemetry_frame(
    df: pd.DataFrame,
    *,
    source_path: str,
    source_dataset: Optional[str] = None,
    label: Optional[int] = None,
    family: str = "",
    capture_id: Optional[str] = None,
) -> pd.DataFrame:
    """Map a raw tabular telemetry frame into the unified CyberShield schema."""
    frame = df.copy()
    frame = _rename_alias_columns(frame)

    if "timestamp" not in frame.columns:
        raise ValueError(f"Missing timestamp-like column in {source_path}")
    for required in ("src_ip", "dst_ip", "proto"):
        if required not in frame.columns:
            raise ValueError(f"Missing required column '{required}' in {source_path}")

    frame["timestamp"] = _coerce_timestamp_column(frame["timestamp"])
    frame = frame.loc[frame["timestamp"].notna()].copy()
    frame["timestamp"] = frame["timestamp"].astype(float)
    frame["ts"] = frame["timestamp"]

    numeric_cols = [
        "duration",
        "orig_bytes",
        "resp_bytes",
        "orig_pkts",
        "resp_pkts",
        "src_port",
        "dst_port",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["duration"] = pd.to_numeric(frame.get("duration", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["orig_bytes"] = pd.to_numeric(frame.get("orig_bytes", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["resp_bytes"] = pd.to_numeric(frame.get("resp_bytes", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["orig_pkts"] = pd.to_numeric(frame.get("orig_pkts", 1.0), errors="coerce").fillna(1.0).clip(lower=1.0)
    frame["resp_pkts"] = pd.to_numeric(frame.get("resp_pkts", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    total_pkts = frame["orig_pkts"] + frame["resp_pkts"]
    total_bytes = frame["orig_bytes"] + frame["resp_bytes"]
    frame["bytes_per_pkt"] = total_bytes / (total_pkts + 1e-9)
    frame["packet_ratio"] = frame["orig_pkts"] / (frame["resp_pkts"] + 1e-9)
    frame["byte_ratio"] = frame["orig_bytes"] / (frame["resp_bytes"] + 1e-9)

    frame["src_ip"] = frame["src_ip"].astype(str)
    frame["dst_ip"] = frame["dst_ip"].astype(str)
    frame["proto"] = frame["proto"].astype(str).str.lower()
    frame["host_id"] = frame.get("host_id", frame["src_ip"]).astype(str)

    src_is_private = frame["src_ip"].apply(is_private_ip).astype(bool)
    dst_is_private = frame["dst_ip"].apply(is_private_ip).astype(bool)
    frame["is_outbound"] = (src_is_private & ~dst_is_private).astype(float)

    source_dataset = source_dataset or infer_source_dataset(source_path)

    # Prefer explicit dataset labels when they exist.
    if label is None:
        if "label_binary" in frame.columns:
            binary_values = pd.to_numeric(frame["label_binary"], errors="coerce")
            if binary_values.notna().any():
                label = int(binary_values.fillna(0).astype(int).mode().iloc[0])
        if label is None and "label_tactic" in frame.columns:
            label_tactic = frame["label_tactic"].astype(str).str.lower()
            if label_tactic.notna().any():
                tactic_text = label_tactic.mode().iloc[0]
                label = LABEL_C2 if tactic_text not in {"none", "benign", "normal", ""} else LABEL_BENIGN
        if label is None and "label_technique" in frame.columns:
            technique_text = frame["label_technique"].astype(str).str.lower().mode().iloc[0]
            label = LABEL_C2 if technique_text not in {"none", "", "nan"} else LABEL_BENIGN
        if label is None:
            inferred_label, _ = infer_label_and_family_from_path(source_path)
            label = inferred_label

    if not family:
        if "label_technique" in frame.columns:
            technique_text = str(frame["label_technique"].astype(str).mode().iloc[0]).strip().lower()
            if technique_text and technique_text not in {"none", "nan", ""}:
                family = f"mitre_{technique_text.lower()}" if technique_text.upper().startswith("T") else technique_text.replace(" ", "_")
        if not family and "label_tactic" in frame.columns:
            tactic_text = str(frame["label_tactic"].astype(str).mode().iloc[0]).strip().lower()
            if tactic_text and tactic_text not in {"none", "nan", ""}:
                family = tactic_text.replace(" ", "_")
        if not family:
            _, inferred_family = infer_label_and_family_from_path(source_path)
            family = inferred_family

    frame["label"] = int(label)
    frame["family"] = family if int(label) == LABEL_C2 else ""
    frame["source_dataset"] = source_dataset
    frame["capture_id"] = capture_id or Path(source_path).stem
    frame["source_path"] = source_path

    frame = frame.sort_values(["host_id", "dst_ip", "proto", "timestamp"], kind="mergesort").reset_index(drop=True)
    groups = frame.groupby(["host_id", "dst_ip", "proto"], sort=False)
    frame["iat"] = groups["timestamp"].diff().fillna(0.0).clip(lower=0.0)
    frame["iat_delta"] = groups["iat"].diff().fillna(0.0)
    frame["byte_delta"] = groups["orig_bytes"].diff().fillna(0.0)

    # Fill any missing expected columns so downstream schema is stable.
    for col in NORMALIZED_SCHEMA_COLUMNS:
        if col not in frame.columns:
            if col in {"label"}:
                frame[col] = int(label)
            elif col in {"family", "capture_id", "source_dataset", "source_path", "host_id"}:
                frame[col] = ""
            else:
                frame[col] = 0.0

    ordered = [col for col in NORMALIZED_SCHEMA_COLUMNS if col in frame.columns]
    extra_cols = [col for col in frame.columns if col not in ordered]
    return frame[ordered + extra_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def load_and_normalize_telemetry(
    path: str,
    *,
    source_dataset: Optional[str] = None,
    label: Optional[int] = None,
    family: str = "",
    capture_id: Optional[str] = None,
) -> pd.DataFrame:
    """Load a raw telemetry file and normalize it into the canonical schema."""
    raw = _load_tabular_frame(path)
    return normalize_telemetry_frame(
        raw,
        source_path=path,
        source_dataset=source_dataset,
        label=label,
        family=family,
        capture_id=capture_id,
    )


def build_session_artifacts(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    inactivity_timeout: float = 300.0,
    min_flows: int = 1,
) -> Tuple[List[SessionArtifact], Dict[str, HostMapEntry]]:
    """Convert a normalized dataframe into session artifacts and a host map."""
    required = {"timestamp", "ts", "host_id", "dst_ip", "proto", label_col, "source_dataset", "capture_id", "source_path", "family"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Normalized frame missing columns: {missing}")

    working = df.copy()
    working["timestamp"] = pd.to_numeric(working["timestamp"], errors="coerce").fillna(0.0)
    working = working.sort_values(["host_id", "dst_ip", "proto", "timestamp"], kind="mergesort").reset_index(drop=True)

    artifacts: List[SessionArtifact] = []
    host_map: Dict[str, HostMapEntry] = {}
    session_counter = 0

    groups = working.groupby(["host_id", "dst_ip", "proto"], sort=False)
    for (host_id, dst_ip, proto), group in groups:
        group = group.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        for sess_df in segment_by_inactivity(group, timeout=inactivity_timeout):
            if len(sess_df) < min_flows:
                continue

            tensor, mask = session_to_tensor(sess_df)
            sample_prefix = str(sess_df["capture_id"].mode().iloc[0]).replace(os.sep, "_").replace("/", "_")
            sample_id = f"{sample_prefix}_s{session_counter:08d}"
            session_counter += 1

            label = int(pd.to_numeric(sess_df[label_col], errors="coerce").fillna(LABEL_BENIGN).mode().iloc[0])
            family = str(sess_df["family"].mode().iloc[0]) if "family" in sess_df.columns else ""
            source_dataset = str(sess_df["source_dataset"].mode().iloc[0])
            capture_id = str(sess_df["capture_id"].mode().iloc[0])
            source_path = str(sess_df["source_path"].mode().iloc[0])
            start_ts = float(sess_df["timestamp"].iloc[0])
            end_ts = float(sess_df["timestamp"].iloc[-1])
            dest_id = f"{dst_ip}|{proto}"

            artifact = SessionArtifact(
                sample_id=sample_id,
                host_id=str(host_id),
                label=label,
                family=family if label == LABEL_C2 else "",
                source_dataset=source_dataset,
                capture_id=capture_id,
                source_path=source_path,
                timestamp_start=start_ts,
                timestamp_end=end_ts,
                protocol=str(proto),
                dest_id=dest_id,
                flow_count=len(sess_df),
                tensor=tensor,
                mask=mask,
            )
            artifacts.append(artifact)

            ann = SessionAnnotation(
                sample_id=sample_id,
                host_id=str(host_id),
                label=label,
                family=family if label == LABEL_C2 else "",
                source_dataset=source_dataset,
                capture_id=capture_id,
                source_path=source_path,
                timestamp_start=start_ts,
                timestamp_end=end_ts,
                protocol=str(proto),
                dest_id=dest_id,
                flow_count=len(sess_df),
            )
            if ann.host_id not in host_map:
                host_map[ann.host_id] = HostMapEntry(host_id=ann.host_id)
            host_map[ann.host_id].ingest(ann)

    return artifacts, host_map


def artifacts_to_arrays(artifacts: Sequence[SessionArtifact]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert session artifacts into session tensors, labels, and masks."""
    if not artifacts:
        return (
            np.empty((0, SESSION_LEN, FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, SESSION_LEN), dtype=bool),
        )

    sessions = np.stack([artifact.tensor for artifact in artifacts]).astype(np.float32)
    labels = np.array([artifact.label for artifact in artifacts], dtype=np.int64)
    masks = np.stack([artifact.mask for artifact in artifacts]).astype(bool)
    return sessions, labels, masks


def annotations_to_dataframe(artifacts: Sequence[SessionArtifact]) -> pd.DataFrame:
    """Convert session artifacts into a labels.csv-ready dataframe."""
    rows = [
        asdict(SessionAnnotation(
            sample_id=artifact.sample_id,
            host_id=artifact.host_id,
            label=artifact.label,
            family=artifact.family,
            source_dataset=artifact.source_dataset,
            capture_id=artifact.capture_id,
            source_path=artifact.source_path,
            timestamp_start=artifact.timestamp_start,
            timestamp_end=artifact.timestamp_end,
            protocol=artifact.protocol,
            dest_id=artifact.dest_id,
            flow_count=artifact.flow_count,
        ))
        for artifact in artifacts
    ]
    return pd.DataFrame(rows)


def write_labels_csv(artifacts: Sequence[SessionArtifact], path: str) -> None:
    """Write the session annotation manifest used for downstream joins."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    annotations_to_dataframe(artifacts).to_csv(path, index=False)


def write_host_map_json(host_map: Dict[str, HostMapEntry], path: str) -> None:
    """Write the host identity continuity map."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {}
    for host_id, entry in sorted(host_map.items()):
        if hasattr(entry, "to_dict"):
            payload[host_id] = entry.to_dict()
        elif isinstance(entry, dict):
            payload[host_id] = entry
        else:
            payload[host_id] = asdict(entry)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def describe_pipeline() -> str:
    """Return a short textual sketch of the ingestion pipeline."""
    return (
        "raw telemetry -> flexible ingestion -> schema normalization -> metadata generation -> "
        "sessionization -> host timelines -> behavioral features -> multi-signal fusion -> risk assessment"
    )
