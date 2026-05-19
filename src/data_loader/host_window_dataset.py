"""Host-aware window datasets for modern C2 research.

This module builds causal per-host windows for the research MVP:

current session + previous N sessions from the same real host.

The arrays saved here are intentionally separate from the production session
NPZ format.  They preserve host/timestamp/source metadata so evaluations can
prove whether a run used real host identity or synthetic smoke-test grouping.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from src.data_loader.split_utils import SplitConfig, SplitStrategy, split_dataset
from src.features.extended_session_builder import (
    _filter_context_for_session,
    _segment_by_inactivity,
    extended_session_to_tensor,
)
from src.features.feature_config_extended import (
    FEATURE_NAMES_EXTENDED,
    FEATURE_SCHEMA_VERSION,
    SESSION_LEN,
)
from src.features.feature_config_v2 import DatasetSource, SessionMetadata


HOST_AWARE_SCHEMA_VERSION = "host_aware_v1"
DEFAULT_HISTORY_SIZE = 32

HOST_FEATURE_NAMES = [
    "history_count",
    "history_unique_destinations",
    "history_duration_mean",
    "history_duration_std",
    "history_orig_bytes_mean",
    "history_resp_bytes_mean",
    "history_total_bytes_mean",
    "history_iat_mean",
    "history_iat_std",
    "history_gap_mean",
    "history_gap_std",
    "history_time_since_last",
    "history_age",
    "history_activity_entropy",
    "is_first_seen_host",
]


@dataclass(frozen=True)
class HostSessionRecord:
    """One session plus provenance needed for causal host-window building."""

    session: np.ndarray
    mask: np.ndarray
    label: int
    host_id: str
    dest_id: str
    timestamp: float
    source: str
    family: str = ""
    capture_id: str = ""
    original_index: int = 0
    host_identity_is_real: bool = True


@dataclass
class HostWindowArrays:
    current_sessions: np.ndarray
    current_masks: np.ndarray
    history_sessions: np.ndarray
    history_flow_masks: np.ndarray
    history_session_masks: np.ndarray
    host_features: np.ndarray
    labels: np.ndarray
    host_ids: np.ndarray
    dest_ids: np.ndarray
    timestamps: np.ndarray
    sources: np.ndarray
    families: np.ndarray
    capture_ids: np.ndarray
    original_indices: np.ndarray
    feature_names: tuple[str, ...]
    host_feature_names: tuple[str, ...]
    schema_version: str = HOST_AWARE_SCHEMA_VERSION
    session_schema_version: str = FEATURE_SCHEMA_VERSION
    history_size: int = DEFAULT_HISTORY_SIZE
    real_host_identity: bool = True

    @property
    def feature_dim(self) -> int:
        return int(self.current_sessions.shape[-1])

    @property
    def session_len(self) -> int:
        return int(self.current_sessions.shape[1])

    @property
    def host_feature_dim(self) -> int:
        return int(self.host_features.shape[-1])

    def copy_with(
        self,
        *,
        current_sessions: np.ndarray | None = None,
        history_sessions: np.ndarray | None = None,
    ) -> "HostWindowArrays":
        return HostWindowArrays(
            current_sessions=self.current_sessions if current_sessions is None else current_sessions,
            current_masks=self.current_masks,
            history_sessions=self.history_sessions if history_sessions is None else history_sessions,
            history_flow_masks=self.history_flow_masks,
            history_session_masks=self.history_session_masks,
            host_features=self.host_features,
            labels=self.labels,
            host_ids=self.host_ids,
            dest_ids=self.dest_ids,
            timestamps=self.timestamps,
            sources=self.sources,
            families=self.families,
            capture_ids=self.capture_ids,
            original_indices=self.original_indices,
            feature_names=self.feature_names,
            host_feature_names=self.host_feature_names,
            schema_version=self.schema_version,
            session_schema_version=self.session_schema_version,
            history_size=self.history_size,
            real_host_identity=self.real_host_identity,
        )


class HostWindowTorchDataset(torch.utils.data.Dataset):
    """PyTorch dataset for host-aware domain-adaptive training."""

    def __init__(
        self,
        arrays: HostWindowArrays,
        domain_to_index: dict[str, int] | None = None,
    ) -> None:
        self.arrays = arrays
        if domain_to_index is None:
            domains = sorted(str(source) for source in np.unique(arrays.sources))
            domain_to_index = {domain: idx for idx, domain in enumerate(domains)}
        self.domain_to_index = dict(domain_to_index)

        self.current_sessions = torch.tensor(arrays.current_sessions, dtype=torch.float32)
        self.current_padding_masks = torch.tensor(~arrays.current_masks.astype(bool), dtype=torch.bool)
        self.history_sessions = torch.tensor(arrays.history_sessions, dtype=torch.float32)
        self.history_flow_padding_masks = torch.tensor(~arrays.history_flow_masks.astype(bool), dtype=torch.bool)
        self.history_session_masks = torch.tensor(arrays.history_session_masks.astype(bool), dtype=torch.bool)
        self.host_features = torch.tensor(arrays.host_features, dtype=torch.float32)
        self.labels = torch.tensor(arrays.labels, dtype=torch.long)
        self.domain_ids = torch.tensor(
            [self.domain_to_index.get(str(source), 0) for source in arrays.sources],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        return (
            self.current_sessions[idx],
            self.current_padding_masks[idx],
            self.history_sessions[idx],
            self.history_flow_padding_masks[idx],
            self.history_session_masks[idx],
            self.host_features[idx],
            self.labels[idx],
            self.domain_ids[idx],
        )


def build_extended_host_session_records_from_dataframe(
    df: pd.DataFrame,
    *,
    source: str = "unknown",
    family: str = "",
    capture_id: str = "",
    label_col: str | None = "label",
    inactivity_timeout: float = 300.0,
    min_flows: int = 1,
    tls_df: pd.DataFrame | None = None,
    dns_df: pd.DataFrame | None = None,
) -> list[HostSessionRecord]:
    """Build extended_v1 session records with real host IDs from a flow DataFrame."""
    required = {"ts", "src_ip", "dst_ip", "proto"}
    if label_col:
        required.add(label_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns required for real host windows: {missing}")

    work = df.copy()
    work["ts"] = pd.to_numeric(work["ts"], errors="coerce").fillna(0.0)
    work = work.sort_values(["src_ip", "dst_ip", "proto", "ts"]).reset_index(drop=True)

    records: list[HostSessionRecord] = []
    for (src_ip, dst_ip, proto), group in work.groupby(["src_ip", "dst_ip", "proto"], sort=False):
        group = group.sort_values("ts").reset_index(drop=True)
        for session_df in _segment_by_inactivity(group, inactivity_timeout):
            if len(session_df) < min_flows:
                continue
            tls_context = _filter_context_for_session(
                tls_df, str(src_ip), str(dst_ip), session_df, context_type="tls"
            )
            dns_context = _filter_context_for_session(
                dns_df, str(src_ip), str(dst_ip), session_df, context_type="dns"
            )
            session, mask = extended_session_to_tensor(
                session_df,
                tls_context=tls_context,
                dns_context=dns_context,
            )
            label = (
                int(session_df[label_col].mode().iloc[0])
                if label_col and label_col in session_df.columns
                else 0
            )
            timestamp = float(pd.to_numeric(session_df["ts"], errors="coerce").fillna(0.0).iloc[0])
            records.append(
                HostSessionRecord(
                    session=session,
                    mask=mask,
                    label=label,
                    host_id=str(src_ip),
                    dest_id=str(dst_ip),
                    timestamp=timestamp,
                    source=source,
                    family=family if label == 1 else "",
                    capture_id=capture_id,
                    original_index=len(records),
                    host_identity_is_real=True,
                )
            )

    if not records:
        raise ValueError("No host session records built. Check input DataFrame and min_flows.")
    return records


def build_host_session_records_from_sessions(
    sessions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    metadata: Sequence[SessionMetadata],
    *,
    dest_ids: Sequence[str] | None = None,
    timestamps: Sequence[float] | None = None,
    require_real_host_ids: bool = True,
) -> list[HostSessionRecord]:
    """Build HostSessionRecord objects from existing session tensors and metadata.

    This path is useful for tests and transitional pipelines.  Final accuracy
    claims should use metadata with real source-host IDs.
    """
    if not (len(sessions) == len(labels) == len(masks) == len(metadata)):
        raise ValueError("sessions, labels, masks, and metadata must have equal length")

    records: list[HostSessionRecord] = []
    for i, meta in enumerate(metadata):
        host_id = str(meta.host_id or "")
        real_host = bool(host_id)
        if not host_id:
            if require_real_host_ids:
                raise ValueError(
                    "Missing real host_id in metadata. Use require_real_host_ids=False only for smoke tests."
                )
            host_id = f"synthetic_host_{i:08d}"
        dest_id = str(dest_ids[i]) if dest_ids is not None else "unknown_dest"
        timestamp = float(timestamps[i]) if timestamps is not None else float(meta.capture_start_ts or i)
        records.append(
            HostSessionRecord(
                session=sessions[i].astype(np.float32, copy=False),
                mask=masks[i].astype(bool, copy=False),
                label=int(labels[i]),
                host_id=host_id,
                dest_id=dest_id,
                timestamp=timestamp,
                source=meta.source.value,
                family=meta.c2_family if int(labels[i]) == 1 else "",
                capture_id=meta.capture_id,
                original_index=i,
                host_identity_is_real=real_host,
            )
        )
    return records


def build_host_windows(
    records: Sequence[HostSessionRecord],
    *,
    history_size: int = DEFAULT_HISTORY_SIZE,
    feature_names: Sequence[str] = FEATURE_NAMES_EXTENDED,
    session_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> HostWindowArrays:
    """Convert session records into causal host windows.

    History slots are right-aligned: if only one previous session exists, it is
    placed at index ``history_size - 1``.  ``history_session_masks`` marks real
    history slots with True.
    """
    if history_size < 1:
        raise ValueError("history_size must be >= 1")
    if not records:
        raise ValueError("Cannot build host windows from an empty record list")

    ordered = sorted(records, key=lambda item: (item.timestamp, item.original_index))
    n = len(ordered)
    session_len = int(ordered[0].session.shape[0])
    feature_dim = int(ordered[0].session.shape[1])

    current_sessions = np.zeros((n, session_len, feature_dim), dtype=np.float32)
    current_masks = np.zeros((n, session_len), dtype=bool)
    history_sessions = np.zeros((n, history_size, session_len, feature_dim), dtype=np.float32)
    history_flow_masks = np.zeros((n, history_size, session_len), dtype=bool)
    history_session_masks = np.zeros((n, history_size), dtype=bool)
    host_features = np.zeros((n, len(HOST_FEATURE_NAMES)), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)

    host_ids: list[str] = []
    dest_ids: list[str] = []
    timestamps: list[float] = []
    sources: list[str] = []
    families: list[str] = []
    capture_ids: list[str] = []
    original_indices: list[int] = []

    history_by_host: dict[str, list[HostSessionRecord]] = {}
    real_host_identity = all(record.host_identity_is_real for record in ordered)

    for row, record in enumerate(ordered):
        prior_history = history_by_host.setdefault(record.host_id, [])
        selected = prior_history[-history_size:]
        start = history_size - len(selected)
        for offset, prior in enumerate(selected, start=start):
            history_sessions[row, offset] = prior.session
            history_flow_masks[row, offset] = prior.mask
            history_session_masks[row, offset] = True

        current_sessions[row] = record.session
        current_masks[row] = record.mask
        host_features[row] = _summarize_history(selected, current_timestamp=record.timestamp)
        labels[row] = int(record.label)

        host_ids.append(record.host_id)
        dest_ids.append(record.dest_id)
        timestamps.append(record.timestamp)
        sources.append(record.source)
        families.append(record.family)
        capture_ids.append(record.capture_id)
        original_indices.append(record.original_index)

        prior_history.append(record)

    return HostWindowArrays(
        current_sessions=current_sessions,
        current_masks=current_masks,
        history_sessions=history_sessions,
        history_flow_masks=history_flow_masks,
        history_session_masks=history_session_masks,
        host_features=host_features,
        labels=labels,
        host_ids=np.asarray(host_ids),
        dest_ids=np.asarray(dest_ids),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        sources=np.asarray(sources),
        families=np.asarray(families),
        capture_ids=np.asarray(capture_ids),
        original_indices=np.asarray(original_indices, dtype=np.int64),
        feature_names=tuple(feature_names),
        host_feature_names=tuple(HOST_FEATURE_NAMES),
        session_schema_version=session_schema_version,
        history_size=history_size,
        real_host_identity=real_host_identity,
    )


def save_host_windows_npz(path: str | Path, arrays: HostWindowArrays) -> None:
    """Save host-window arrays in compressed NPZ format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        current_sessions=arrays.current_sessions.astype(np.float32),
        current_masks=arrays.current_masks.astype(bool),
        history_sessions=arrays.history_sessions.astype(np.float32),
        history_flow_masks=arrays.history_flow_masks.astype(bool),
        history_session_masks=arrays.history_session_masks.astype(bool),
        host_features=arrays.host_features.astype(np.float32),
        y=arrays.labels.astype(np.int64),
        host_ids=arrays.host_ids,
        dest_ids=arrays.dest_ids,
        timestamps=arrays.timestamps.astype(np.float64),
        sources=arrays.sources,
        families=arrays.families,
        capture_ids=arrays.capture_ids,
        original_indices=arrays.original_indices.astype(np.int64),
        feature_names=np.asarray(arrays.feature_names),
        host_feature_names=np.asarray(arrays.host_feature_names),
        schema_version=np.asarray(arrays.schema_version),
        session_schema_version=np.asarray(arrays.session_schema_version),
        history_size=np.asarray(arrays.history_size, dtype=np.int64),
        real_host_identity=np.asarray(arrays.real_host_identity, dtype=bool),
    )


def load_host_windows_npz(path: str | Path) -> HostWindowArrays:
    """Load host-window arrays saved by :func:`save_host_windows_npz`."""
    data = np.load(path, allow_pickle=True)
    try:
        schema_version = str(data["schema_version"]) if "schema_version" in data else "unknown"
        if schema_version != HOST_AWARE_SCHEMA_VERSION:
            raise ValueError(
                f"Host-window schema mismatch: expected {HOST_AWARE_SCHEMA_VERSION}, got {schema_version}"
            )
        labels = data["y"].astype(np.int64) if "y" in data else data["labels"].astype(np.int64)
        return HostWindowArrays(
            current_sessions=data["current_sessions"].astype(np.float32),
            current_masks=data["current_masks"].astype(bool),
            history_sessions=data["history_sessions"].astype(np.float32),
            history_flow_masks=data["history_flow_masks"].astype(bool),
            history_session_masks=data["history_session_masks"].astype(bool),
            host_features=data["host_features"].astype(np.float32),
            labels=labels,
            host_ids=data["host_ids"].astype(str),
            dest_ids=data["dest_ids"].astype(str),
            timestamps=data["timestamps"].astype(np.float64),
            sources=data["sources"].astype(str),
            families=data["families"].astype(str),
            capture_ids=data["capture_ids"].astype(str),
            original_indices=data["original_indices"].astype(np.int64),
            feature_names=tuple(str(name) for name in data["feature_names"].tolist()),
            host_feature_names=tuple(str(name) for name in data["host_feature_names"].tolist()),
            schema_version=schema_version,
            session_schema_version=str(data["session_schema_version"]),
            history_size=int(data["history_size"]),
            real_host_identity=bool(data["real_host_identity"]),
        )
    finally:
        data.close()


def split_records(
    records: Sequence[HostSessionRecord],
    *,
    split_strategy: str = "family_separated",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> dict[str, list[HostSessionRecord]]:
    """Split host records for leakage-resistant research evaluation."""
    if split_strategy == "host_separated":
        return _host_separated_split(records, val_fraction, test_fraction, random_seed)

    labels = np.asarray([record.label for record in records], dtype=np.int64)
    metadata = [_record_to_metadata(record) for record in records]
    config = SplitConfig(
        strategy=SplitStrategy(split_strategy),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    result = split_dataset(labels, metadata, config)
    return {
        "train": [records[i] for i in result.train_indices],
        "val": [records[i] for i in result.val_indices],
        "test": [records[i] for i in result.test_indices],
    }


def prepare_host_window_splits_from_dataframe(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    source: str = "unknown",
    family: str = "",
    capture_id: str = "",
    split_strategy: str = "family_separated",
    history_size: int = DEFAULT_HISTORY_SIZE,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
    label_col: str | None = "label",
    inactivity_timeout: float = 300.0,
    min_flows: int = 1,
    tls_df: pd.DataFrame | None = None,
    dns_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Build and save train/val/test host-window NPZ files from raw telemetry."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    records = build_extended_host_session_records_from_dataframe(
        df,
        source=source,
        family=family,
        capture_id=capture_id,
        label_col=label_col,
        inactivity_timeout=inactivity_timeout,
        min_flows=min_flows,
        tls_df=tls_df,
        dns_df=dns_df,
    )
    splits = split_records(
        records,
        split_strategy=split_strategy,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    outputs: dict[str, str] = {}
    summary: dict[str, object] = {
        "schema_version": HOST_AWARE_SCHEMA_VERSION,
        "session_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES_EXTENDED),
        "host_feature_names": list(HOST_FEATURE_NAMES),
        "history_size": history_size,
        "split_strategy": split_strategy,
        "real_host_identity": True,
        "splits": {},
    }

    for split_name, split_records_list in splits.items():
        arrays = build_host_windows(
            split_records_list,
            history_size=history_size,
            feature_names=FEATURE_NAMES_EXTENDED,
            session_schema_version=FEATURE_SCHEMA_VERSION,
        )
        path = out_root / f"{split_name}_host_windows.npz"
        save_host_windows_npz(path, arrays)
        outputs[f"{split_name}_host_windows"] = str(path)
        summary["splits"][split_name] = {
            "samples": int(len(arrays.labels)),
            "c2": int((arrays.labels == 1).sum()),
            "benign": int((arrays.labels == 0).sum()),
            "hosts": int(len(set(arrays.host_ids.tolist()))),
            "sources": sorted(set(str(x) for x in arrays.sources.tolist())),
            "families": sorted(set(str(x) for x in arrays.families.tolist() if str(x))),
        }

    metadata_path = out_root / "host_window_split_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs["metadata"] = str(metadata_path)
    return outputs


def validate_host_disjoint(*splits: HostWindowArrays) -> bool:
    """Return True if no host ID appears in more than one split."""
    seen: set[str] = set()
    for arrays in splits:
        hosts = set(str(host) for host in arrays.host_ids.tolist())
        if seen & hosts:
            return False
        seen.update(hosts)
    return True


def _record_to_metadata(record: HostSessionRecord) -> SessionMetadata:
    return SessionMetadata(
        source=_coerce_source(record.source),
        c2_family=record.family if int(record.label) == 1 else "",
        capture_id=record.capture_id,
        capture_start_ts=record.timestamp,
        host_id=record.host_id,
    )


def _coerce_source(source: str) -> DatasetSource:
    source_map = {item.value: item for item in DatasetSource}
    return source_map.get(str(source).lower(), DatasetSource.UNKNOWN)


def _host_separated_split(
    records: Sequence[HostSessionRecord],
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, list[HostSessionRecord]]:
    rng = np.random.default_rng(random_seed)
    host_to_records: dict[str, list[HostSessionRecord]] = {}
    for record in records:
        host_to_records.setdefault(record.host_id, []).append(record)

    hosts = np.asarray(sorted(host_to_records.keys()))
    if len(hosts) < 3:
        raise ValueError("host_separated split requires at least three hosts")

    shuffled_hosts = hosts[rng.permutation(len(hosts))]
    n_test = max(1, int(round(len(shuffled_hosts) * test_fraction)))
    n_val = max(1, int(round(len(shuffled_hosts) * val_fraction)))
    if n_test + n_val >= len(shuffled_hosts):
        n_test = 1
        n_val = 1

    test_hosts = set(shuffled_hosts[:n_test].tolist())
    val_hosts = set(shuffled_hosts[n_test:n_test + n_val].tolist())
    train_hosts = set(shuffled_hosts[n_test + n_val:].tolist())

    return {
        "train": [record for record in records if record.host_id in train_hosts],
        "val": [record for record in records if record.host_id in val_hosts],
        "test": [record for record in records if record.host_id in test_hosts],
    }


def _summarize_history(
    history: Sequence[HostSessionRecord],
    *,
    current_timestamp: float,
) -> np.ndarray:
    if not history:
        values = np.zeros(len(HOST_FEATURE_NAMES), dtype=np.float32)
        values[HOST_FEATURE_NAMES.index("is_first_seen_host")] = 1.0
        return values

    durations = np.asarray([_session_sum(record, "duration") for record in history], dtype=np.float32)
    orig_bytes = np.asarray([_session_sum(record, "orig_bytes") for record in history], dtype=np.float32)
    resp_bytes = np.asarray([_session_sum(record, "resp_bytes") for record in history], dtype=np.float32)
    iat_means = np.asarray([_session_mean(record, "iat") for record in history], dtype=np.float32)
    timestamps = np.asarray([record.timestamp for record in history], dtype=np.float64)
    gaps = np.diff(timestamps) if len(timestamps) > 1 else np.asarray([], dtype=np.float64)

    hours = np.asarray([int((ts // 3600) % 24) for ts in timestamps], dtype=np.int64)
    hour_counts = np.bincount(hours, minlength=24).astype(np.float64)
    probs = hour_counts / max(hour_counts.sum(), 1.0)
    non_zero_probs = probs[probs > 0]
    entropy = float(-(non_zero_probs * np.log2(non_zero_probs)).sum()) if len(non_zero_probs) else 0.0

    total_bytes = orig_bytes + resp_bytes
    return np.asarray(
        [
            float(len(history)),
            float(len({record.dest_id for record in history})),
            float(durations.mean()) if len(durations) else 0.0,
            float(durations.std()) if len(durations) > 1 else 0.0,
            float(orig_bytes.mean()) if len(orig_bytes) else 0.0,
            float(resp_bytes.mean()) if len(resp_bytes) else 0.0,
            float(total_bytes.mean()) if len(total_bytes) else 0.0,
            float(iat_means.mean()) if len(iat_means) else 0.0,
            float(iat_means.std()) if len(iat_means) > 1 else 0.0,
            float(gaps.mean()) if len(gaps) else 0.0,
            float(gaps.std()) if len(gaps) > 1 else 0.0,
            float(max(0.0, current_timestamp - float(timestamps[-1]))),
            float(max(0.0, current_timestamp - float(timestamps[0]))),
            entropy,
            0.0,
        ],
        dtype=np.float32,
    )


def _session_real_rows(record: HostSessionRecord) -> np.ndarray:
    return record.session[record.mask] if record.mask.any() else record.session[:1]


def _feature_index(record: HostSessionRecord, feature_name: str) -> int | None:
    try:
        return FEATURE_NAMES_EXTENDED.index(feature_name)
    except ValueError:
        if record.session.shape[-1] > 0 and feature_name == "orig_bytes":
            return 0
        return None


def _session_sum(record: HostSessionRecord, feature_name: str) -> float:
    idx = _feature_index(record, feature_name)
    if idx is None:
        return 0.0
    real = _session_real_rows(record)
    return float(real[:, idx].sum())


def _session_mean(record: HostSessionRecord, feature_name: str) -> float:
    idx = _feature_index(record, feature_name)
    if idx is None:
        return 0.0
    real = _session_real_rows(record)
    return float(real[:, idx].mean()) if len(real) else 0.0

