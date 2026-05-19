"""Inventory raw telemetry and build the first real host-aware C2 split.

This pipeline is intentionally conservative:

* Inventory every local raw/normalized source we know how to inspect.
* Build the first clean host-aware C2 split from raw Stratosphere MCFP
  BinetFlow files because they have real source-host identities and explicit
  botnet-vs-normal file labels.
* Keep UWF ZeekData24 in the inventory by default, but do not fold its
  defense-evasion/exfiltration/persistence labels into the C2 training split
  unless the caller opts in. Those labels are useful telemetry, but they are
  not clean TA0011 command-and-control labels in the local copy.

Outputs
-------
data/processed/host_aware_mvp/
  dataset_inventory.json
  dataset_inventory.md
  train_host_windows.npz
  val_host_windows.npz
  test_host_windows.npz
  host_window_split_metadata.json
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import re
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data_loader.host_window_dataset import (
    DEFAULT_HISTORY_SIZE,
    HOST_AWARE_SCHEMA_VERSION,
    HOST_FEATURE_NAMES,
    HostSessionRecord,
    build_extended_host_session_records_from_dataframe,
    build_host_windows,
    save_host_windows_npz,
)
from src.data_loader.stratosphere_mcfp_processor import _load_flow_dataframe
from src.features.feature_config import is_private_ip, label_from_string
from src.features.feature_config_extended import FEATURE_NAMES_EXTENDED, FEATURE_SCHEMA_VERSION
from src.features.feature_config_v2 import DatasetSource


CHUNK_SIZE = 200_000
DEFAULT_OUTPUT_DIR = Path("data/processed/host_aware_mvp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create dataset inventory and real host-aware train/val/test NPZ splits."
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Root raw telemetry directory")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--history-size", type=int, default=DEFAULT_HISTORY_SIZE)
    parser.add_argument(
        "--min-flows",
        type=int,
        default=5,
        help="Minimum flows per session. Default matches the existing MCFP session pipeline.",
    )
    parser.add_argument("--inactivity-timeout", type=float, default=300.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("mcfp", "ctu13", "uwf"),
        default=None,
        help="Sources to build into this split. Defaults to mcfp.",
    )
    parser.add_argument(
        "--include-ctu13",
        action="store_true",
        help="Include CTU-13 raw files in the built split. Inventory always includes them.",
    )
    parser.add_argument(
        "--include-uwf-attack-labels",
        action="store_true",
        help=(
            "Include UWF non-benign tactics as positive labels. Use only for "
            "pipeline experiments; the local labels are not clean C2 labels."
        ),
    )
    parser.add_argument(
        "--max-files-per-source",
        type=int,
        default=0,
        help="Optional debug cap per source; 0 means no cap.",
    )
    parser.add_argument(
        "--reuse-existing-inventory",
        action="store_true",
        help="Reuse dataset_inventory.json in the output directory when it already exists.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = out_dir / "dataset_inventory.json"
    if args.reuse_existing_inventory and inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        inventory = build_dataset_inventory(raw_dir)
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    markdown_path = out_dir / "dataset_inventory.md"
    markdown_path.write_text(render_inventory_markdown(inventory), encoding="utf-8")

    sources_to_build = set(args.sources or ["mcfp"])
    if args.include_ctu13:
        sources_to_build.add("ctu13")
    if args.include_uwf_attack_labels:
        sources_to_build.add("uwf")

    records: list[HostSessionRecord] = []
    selected_sources: list[str] = []
    label_semantics: list[str] = []

    if "mcfp" in sources_to_build:
        mcfp_records = build_mcfp_records(
            raw_dir / "mcfp",
            min_flows=args.min_flows,
            inactivity_timeout=args.inactivity_timeout,
            max_files=args.max_files_per_source,
        )
        records.extend(mcfp_records)
        selected_sources.append(DatasetSource.MCFP_STRATOSPHERE.value)
        label_semantics.append(
            "mcfp_stratosphere: file-level botnet-vs-normal labels from raw BinetFlow"
        )

    if "ctu13" in sources_to_build:
        ctu_records = build_ctu13_records(
            raw_dir,
            min_flows=args.min_flows,
            inactivity_timeout=args.inactivity_timeout,
            max_files=args.max_files_per_source,
        )
        records.extend(ctu_records)
        selected_sources.append(DatasetSource.CTU13.value)
        label_semantics.append("ctu13: parsed flow Label using C2/botnet keywords")

    if "uwf" in sources_to_build:
        uwf_records = build_uwf_records(
            raw_dir / "uwf_zeekdata24",
            min_flows=args.min_flows,
            inactivity_timeout=args.inactivity_timeout,
            max_files=args.max_files_per_source,
        )
        records.extend(uwf_records)
        selected_sources.append(DatasetSource.UWF_ZEEKDATA24.value)
        label_semantics.append(
            "uwf_zeekdata24: folder-level attack-vs-benign labels, not clean C2"
        )

    if not records:
        raise ValueError("No host-aware records were produced from selected sources.")

    records = renumber_records(records)
    splits = stratified_host_separated_split(
        records,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )
    outputs = save_split_artifacts(
        splits,
        out_dir,
        history_size=args.history_size,
        selected_sources=selected_sources,
        label_semantics=label_semantics,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
        min_flows=args.min_flows,
        inactivity_timeout=args.inactivity_timeout,
        inventory_path=inventory_path,
    )

    print(f"[SUCCESS] Inventory JSON: {inventory_path}")
    print(f"[SUCCESS] Inventory Markdown: {markdown_path}")
    print(f"[SUCCESS] Host-aware split metadata: {outputs['metadata']}")
    for split_name in ("train", "val", "test"):
        summary = outputs["splits"][split_name]
        print(
            "[INFO] "
            f"{split_name}: samples={summary['samples']:,}, "
            f"c2={summary['c2']:,}, benign={summary['benign']:,}, "
            f"hosts={summary['hosts']:,}"
        )


def build_dataset_inventory(raw_dir: Path) -> dict[str, object]:
    """Create a source-level and file-level inventory for local telemetry."""
    generated_at = datetime.now(timezone.utc).isoformat()
    sources = {
        "mcfp_stratosphere": inventory_mcfp(raw_dir / "mcfp"),
        "ctu13": inventory_ctu13(raw_dir),
        "uwf_zeekdata24": inventory_uwf(raw_dir / "uwf_zeekdata24"),
    }
    return {
        "generated_at_utc": generated_at,
        "raw_dir": str(raw_dir),
        "inventory_version": "host_aware_inventory_v1",
        "schema_notes": {
            "host_identity_requirement": "Final host-aware accuracy claims require real source host identity, normally src_ip.",
            "host_window_schema": HOST_AWARE_SCHEMA_VERSION,
            "session_schema": FEATURE_SCHEMA_VERSION,
            "history_size_default": DEFAULT_HISTORY_SIZE,
            "feature_names": list(FEATURE_NAMES_EXTENDED),
            "host_feature_names": list(HOST_FEATURE_NAMES),
        },
        "sources": sources,
    }


def inventory_mcfp(source_dir: Path) -> dict[str, object]:
    files = sorted(source_dir.glob("*.binetflow"))
    file_entries = []
    for path in files:
        assigned_label = "c2" if is_mcfp_c2_file(path) else "benign"
        entry = summarize_delimited_file(
            path,
            src_candidates=("SrcAddr",),
            dst_candidates=("DstAddr",),
            label_candidates=("Label",),
            extra_count_columns=("Proto", "State"),
        )
        entry.update(
            {
                "assigned_label": assigned_label,
                "label_source": "filename",
                "eligible_for_first_host_split": True,
            }
        )
        file_entries.append(entry)
    return summarize_source(
        name="mcfp_stratosphere",
        raw_type="Argus/BinetFlow flow CSV",
        description="Stratosphere Malware Capture Facility Project botnet and normal traffic.",
        label_semantics="File-level botnet-vs-normal labels inferred from CTU-Malware-Capture-Botnet and CTU-Normal filenames.",
        has_real_host_identity=True,
        dns_tls_context="not present as separate Zeek dns/ssl logs; DNS/TLS feature groups are zero-filled unless added later",
        split_eligibility="used_by_default_for_first_clean_c2_host_split",
        files=file_entries,
    )


def inventory_ctu13(raw_dir: Path) -> dict[str, object]:
    candidate_dirs = [raw_dir / "ctu13", raw_dir / "ctu13_extra"]
    seen: set[Path] = set()
    files: list[Path] = []
    for directory in candidate_dirs:
        if directory.exists():
            for path in sorted(directory.rglob("*.binetflow")):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(path)
    file_entries = []
    for path in files:
        entry = summarize_delimited_file(
            path,
            src_candidates=("SrcAddr",),
            dst_candidates=("DstAddr",),
            label_candidates=("Label",),
            extra_count_columns=("Proto", "State"),
            parse_binary_label=True,
        )
        entry.update(
            {
                "assigned_label": "mixed",
                "label_source": "flow Label column parsed with C2/botnet keywords",
                "eligible_for_first_host_split": False,
                "defer_reason": "larger legacy captures; keep for next family/source split after duplicate capture policy is chosen",
            }
        )
        file_entries.append(entry)
    return summarize_source(
        name="ctu13",
        raw_type="Argus/BinetFlow flow CSV",
        description="CTU-13 botnet scenarios with flow-level labels.",
        label_semantics="Flow Label parsed through existing C2/botnet keyword taxonomy.",
        has_real_host_identity=True,
        dns_tls_context="not present as separate Zeek dns/ssl logs in local raw copy",
        split_eligibility="inventory_only_by_default",
        files=file_entries,
    )


def inventory_uwf(source_dir: Path) -> dict[str, object]:
    files = sorted(source_dir.rglob("*.csv")) if source_dir.exists() else []
    file_entries = []
    for path in files:
        entry = summarize_delimited_file(
            path,
            src_candidates=("src_ip_zeek", "src_ip", "id.orig_h"),
            dst_candidates=("dest_ip_zeek", "dst_ip", "id.resp_h"),
            label_candidates=("label_tactic", "label_technique", "label_binary"),
            extra_count_columns=("proto", "service", "label_tactic", "label_technique"),
        )
        folder = path.parent.name.lower()
        entry.update(
            {
                "assigned_label": "benign" if folder == "benign" else "attack_positive",
                "label_source": "folder and label_tactic/label_technique columns",
                "eligible_for_first_host_split": False,
                "defer_reason": "local positives are attack tactics, not clean TA0011/C2 labels",
            }
        )
        file_entries.append(entry)
    return summarize_source(
        name="uwf_zeekdata24",
        raw_type="Zeek-derived normalized connection CSV",
        description="UWF ZeekData24 normalized conn-style telemetry with real source host fields.",
        label_semantics="Local folders are benign, defense_evasion, exfiltration, and persistence; positives are not clean C2 labels.",
        has_real_host_identity=True,
        dns_tls_context="connection-level service fields present; no separate local dns.log/ssl.log sidecar discovered",
        split_eligibility="inventory_only_by_default; opt-in as attack-vs-benign with --include-uwf-attack-labels",
        files=file_entries,
    )


def summarize_delimited_file(
    path: Path,
    *,
    src_candidates: Sequence[str],
    dst_candidates: Sequence[str],
    label_candidates: Sequence[str],
    extra_count_columns: Sequence[str] = (),
    parse_binary_label: bool = False,
) -> dict[str, object]:
    columns = read_header_columns(path)
    usecols = [
        col
        for col in set(src_candidates + dst_candidates + label_candidates + extra_count_columns)
        if col in columns
    ]
    src_col = first_present(columns, src_candidates)
    dst_col = first_present(columns, dst_candidates)

    rows = 0
    src_hosts: set[str] = set()
    dst_hosts: set[str] = set()
    valid_src_ip_rows = 0
    valid_dst_ip_rows = 0
    counters: dict[str, Counter[str]] = {col: Counter() for col in usecols if col not in {src_col, dst_col}}
    binary_labels = Counter()

    if usecols:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
            rows += int(len(chunk))
            if src_col and src_col in chunk.columns:
                src_values = chunk[src_col].dropna().astype(str)
                src_hosts.update(src_values.unique().tolist())
                valid_src_ip_rows += int(src_values.map(is_valid_ip).sum())
            if dst_col and dst_col in chunk.columns:
                dst_values = chunk[dst_col].dropna().astype(str)
                dst_hosts.update(dst_values.unique().tolist())
                valid_dst_ip_rows += int(dst_values.map(is_valid_ip).sum())
            for col, counter in counters.items():
                if col not in chunk.columns:
                    continue
                values = chunk[col].fillna("").astype(str)
                counter.update(values.tolist())
                if parse_binary_label and col == "Label":
                    binary_labels.update(
                        "c2" if int(label_from_string(value)) == 1 else "benign"
                        for value in values.tolist()
                    )
    else:
        rows = count_rows(path)

    entry = {
        "path": str(path),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "rows": int(rows),
        "columns": columns,
        "src_identity_column": src_col or "",
        "dest_identity_column": dst_col or "",
        "unique_src_hosts": int(len(src_hosts)),
        "unique_destinations": int(len(dst_hosts)),
        "valid_src_ip_rows": int(valid_src_ip_rows),
        "valid_dst_ip_rows": int(valid_dst_ip_rows),
        "has_real_host_identity": bool(src_col),
        "top_values": {
            col: counter_to_dict(counter, limit=12)
            for col, counter in counters.items()
            if counter
        },
    }
    if binary_labels:
        entry["parsed_binary_label_counts"] = counter_to_dict(binary_labels, limit=4)
    return entry


def summarize_source(
    *,
    name: str,
    raw_type: str,
    description: str,
    label_semantics: str,
    has_real_host_identity: bool,
    dns_tls_context: str,
    split_eligibility: str,
    files: Sequence[dict[str, object]],
) -> dict[str, object]:
    total_rows = sum(int(file_info.get("rows", 0)) for file_info in files)
    total_bytes = sum(int(file_info.get("bytes", 0)) for file_info in files)
    total_valid_src_rows = sum(int(file_info.get("valid_src_ip_rows", 0)) for file_info in files)
    unique_hosts = sum(int(file_info.get("unique_src_hosts", 0)) for file_info in files)
    labels = Counter(str(file_info.get("assigned_label", "unknown")) for file_info in files)
    return {
        "name": name,
        "description": description,
        "raw_type": raw_type,
        "label_semantics": label_semantics,
        "has_real_host_identity": has_real_host_identity,
        "dns_tls_context": dns_tls_context,
        "split_eligibility": split_eligibility,
        "file_count": len(files),
        "total_rows": int(total_rows),
        "total_bytes": int(total_bytes),
        "valid_src_ip_row_fraction": safe_fraction(total_valid_src_rows, total_rows),
        "sum_unique_src_hosts_by_file": int(unique_hosts),
        "assigned_label_file_counts": dict(labels),
        "files": list(files),
    }


def build_mcfp_records(
    source_dir: Path,
    *,
    min_flows: int,
    inactivity_timeout: float,
    max_files: int = 0,
) -> list[HostSessionRecord]:
    files = sorted(source_dir.glob("*.binetflow"))
    if max_files > 0:
        files = files[:max_files]
    records: list[HostSessionRecord] = []
    for path in files:
        label = 1 if is_mcfp_c2_file(path) else 0
        family = mcfp_family_from_filename(path) if label == 1 else ""
        df = _load_flow_dataframe(str(path))
        before_rows = len(df)
        df = filter_ip_flow_rows(df)
        if df.empty:
            print(f"[WARN] Skipping {path}: no IP flow rows after filtering {before_rows:,} rows")
            continue
        df = df.copy()
        df["label"] = int(label)
        file_records = build_extended_host_session_records_from_dataframe(
            df,
            source=DatasetSource.MCFP_STRATOSPHERE.value,
            family=family,
            capture_id=path.stem,
            label_col="label",
            inactivity_timeout=inactivity_timeout,
            min_flows=min_flows,
        )
        records.extend(file_records)
        print(
            f"[INFO] MCFP {path.name}: flows={len(df):,}/{before_rows:,}, "
            f"sessions={len(file_records):,}, label={'c2' if label else 'benign'}"
        )
    return records


def build_ctu13_records(
    raw_dir: Path,
    *,
    min_flows: int,
    inactivity_timeout: float,
    max_files: int = 0,
) -> list[HostSessionRecord]:
    from src.data_loader.ctu13_processor import apply_ctu13_labels, load_and_standardize_ctu13

    files = sorted((raw_dir / "ctu13").glob("*.binetflow"))
    extra_dir = raw_dir / "ctu13_extra"
    if extra_dir.exists():
        files.extend(sorted(extra_dir.glob("*.binetflow")))
    if max_files > 0:
        files = files[:max_files]

    records: list[HostSessionRecord] = []
    for path in files:
        df = load_and_standardize_ctu13(str(path))
        df = apply_ctu13_labels(df)
        before_rows = len(df)
        df = filter_ip_flow_rows(df)
        if df.empty:
            print(f"[WARN] Skipping {path}: no IP flow rows after filtering {before_rows:,} rows")
            continue
        file_records = build_extended_host_session_records_from_dataframe(
            df,
            source=DatasetSource.CTU13.value,
            family=f"ctu13_{path.stem}",
            capture_id=path.stem,
            label_col="label",
            inactivity_timeout=inactivity_timeout,
            min_flows=min_flows,
        )
        records.extend(file_records)
        print(f"[INFO] CTU13 {path.name}: flows={len(df):,}/{before_rows:,}, sessions={len(file_records):,}")
    return records


def build_uwf_records(
    source_dir: Path,
    *,
    min_flows: int,
    inactivity_timeout: float,
    max_files: int = 0,
) -> list[HostSessionRecord]:
    files = sorted(source_dir.rglob("*.csv"))
    if max_files > 0:
        files = files[:max_files]
    records: list[HostSessionRecord] = []
    for path in files:
        df = normalize_uwf_conn_csv(path)
        before_rows = len(df)
        df = filter_ip_flow_rows(df)
        if df.empty:
            print(f"[WARN] Skipping {path}: no IP flow rows after filtering {before_rows:,} rows")
            continue
        label = 0 if path.parent.name.lower() == "benign" else 1
        family = f"uwf_{path.parent.name.lower()}" if label else ""
        df = df.copy()
        df["label"] = int(label)
        file_records = build_extended_host_session_records_from_dataframe(
            df,
            source=DatasetSource.UWF_ZEEKDATA24.value,
            family=family,
            capture_id=f"{path.parent.name}_{path.stem}",
            label_col="label",
            inactivity_timeout=inactivity_timeout,
            min_flows=min_flows,
        )
        records.extend(file_records)
        print(
            f"[INFO] UWF {path.parent.name}/{path.name}: flows={len(df):,}/{before_rows:,}, "
            f"sessions={len(file_records):,}, label={'attack' if label else 'benign'}"
        )
    return records


def normalize_uwf_conn_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame()
    out["ts"] = pd.to_numeric(raw.get("ts", 0.0), errors="coerce").fillna(0.0)
    out["src_ip"] = raw.get("src_ip_zeek", raw.get("src_ip", "")).fillna("").astype(str)
    out["dst_ip"] = raw.get("dest_ip_zeek", raw.get("dst_ip", "")).fillna("").astype(str)
    out["proto"] = raw.get("proto", "").fillna("").astype(str).str.lower()
    out["src_port"] = pd.to_numeric(raw.get("src_port_zeek", 0), errors="coerce").fillna(0.0)
    out["dst_port"] = pd.to_numeric(raw.get("dest_port_zeek", 0), errors="coerce").fillna(0.0)
    out["duration"] = pd.to_numeric(raw.get("duration", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["orig_bytes"] = pd.to_numeric(raw.get("orig_bytes", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["resp_bytes"] = pd.to_numeric(raw.get("resp_bytes", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["orig_pkts"] = pd.to_numeric(raw.get("orig_pkts", 1), errors="coerce").fillna(1.0).clip(lower=1.0)
    out["resp_pkts"] = pd.to_numeric(raw.get("resp_pkts", 0), errors="coerce").fillna(0.0).clip(lower=0.0)

    total_pkts = out["orig_pkts"] + out["resp_pkts"]
    total_bytes = out["orig_bytes"] + out["resp_bytes"]
    out["bytes_per_pkt"] = total_bytes / (total_pkts + 1e-9)
    out["packet_ratio"] = out["orig_pkts"] / (out["resp_pkts"] + 1e-9)
    out["byte_ratio"] = out["orig_bytes"] / (out["resp_bytes"] + 1e-9)
    src_is_private = out["src_ip"].map(is_private_ip).astype(bool)
    dst_is_private = out["dst_ip"].map(is_private_ip).astype(bool)
    out["is_outbound"] = (src_is_private & ~dst_is_private).astype(float)

    out = out.sort_values(["src_ip", "dst_ip", "proto", "ts"], kind="mergesort").reset_index(drop=True)
    group = out.groupby(["src_ip", "dst_ip", "proto"], sort=False)
    out["iat"] = group["ts"].diff().fillna(0.0).clip(lower=0.0)
    out["iat_delta"] = group["iat"].diff().fillna(0.0)
    out["byte_delta"] = group["orig_bytes"].diff().fillna(0.0)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def stratified_host_separated_split(
    records: Sequence[HostSessionRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, list[HostSessionRecord]]:
    """Split by host while preserving at least some positive/benign host mix."""
    if not records:
        raise ValueError("Cannot split an empty record set")

    host_labels: dict[str, list[int]] = {}
    for record in records:
        host_labels.setdefault(record.host_id, []).append(int(record.label))

    positive_hosts = sorted(host for host, labels in host_labels.items() if any(labels))
    benign_hosts = sorted(host for host, labels in host_labels.items() if not any(labels))

    rng = np.random.default_rng(random_seed)
    assignment = {"train": set(), "val": set(), "test": set()}
    for host_group in (positive_hosts, benign_hosts):
        group_assignment = split_host_group(
            host_group,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            rng=rng,
        )
        for split_name, hosts in group_assignment.items():
            assignment[split_name].update(hosts)

    splits = {
        split_name: [record for record in records if record.host_id in hosts]
        for split_name, hosts in assignment.items()
    }
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError("Host-separated split produced an empty train/val/test split")
    return splits


def split_host_group(
    hosts: Sequence[str],
    *,
    val_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> dict[str, set[str]]:
    hosts_array = np.asarray(list(hosts))
    if len(hosts_array) == 0:
        return {"train": set(), "val": set(), "test": set()}
    shuffled = hosts_array[rng.permutation(len(hosts_array))]
    if len(shuffled) < 3:
        return {"train": set(shuffled.tolist()), "val": set(), "test": set()}

    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    if n_test + n_val >= len(shuffled):
        n_test = 1
        n_val = 1
    return {
        "test": set(shuffled[:n_test].tolist()),
        "val": set(shuffled[n_test:n_test + n_val].tolist()),
        "train": set(shuffled[n_test + n_val:].tolist()),
    }


def save_split_artifacts(
    splits: dict[str, list[HostSessionRecord]],
    out_dir: Path,
    *,
    history_size: int,
    selected_sources: Sequence[str],
    label_semantics: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
    min_flows: int,
    inactivity_timeout: float,
    inventory_path: Path,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, dict[str, object]] = {}
    split_paths: dict[str, str] = {}
    split_hosts: dict[str, set[str]] = {}

    for split_name in ("train", "val", "test"):
        records = splits[split_name]
        arrays = build_host_windows(
            records,
            history_size=history_size,
            feature_names=FEATURE_NAMES_EXTENDED,
            session_schema_version=FEATURE_SCHEMA_VERSION,
        )
        path = out_dir / f"{split_name}_host_windows.npz"
        save_host_windows_npz(path, arrays)
        split_paths[split_name] = str(path)
        split_hosts[split_name] = set(str(host) for host in arrays.host_ids.tolist())
        history_counts = arrays.history_session_masks.sum(axis=1) if len(arrays.labels) else np.asarray([])
        split_summaries[split_name] = {
            "path": str(path),
            "samples": int(len(arrays.labels)),
            "c2": int((arrays.labels == 1).sum()),
            "benign": int((arrays.labels == 0).sum()),
            "hosts": int(len(split_hosts[split_name])),
            "sources": sorted(set(str(value) for value in arrays.sources.tolist())),
            "families": sorted(set(str(value) for value in arrays.families.tolist() if str(value))),
            "captures": sorted(set(str(value) for value in arrays.capture_ids.tolist())),
            "first_timestamp": float(np.min(arrays.timestamps)) if len(arrays.timestamps) else None,
            "last_timestamp": float(np.max(arrays.timestamps)) if len(arrays.timestamps) else None,
            "mean_history_sessions": float(np.mean(history_counts)) if history_counts.size else 0.0,
            "max_history_sessions": int(np.max(history_counts)) if history_counts.size else 0,
            "real_host_identity": bool(arrays.real_host_identity),
        }

    evaluation_readiness = summarize_evaluation_readiness(split_summaries)
    metadata = {
        "schema_version": HOST_AWARE_SCHEMA_VERSION,
        "session_schema_version": FEATURE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_sources": list(selected_sources),
        "label_semantics": list(label_semantics),
        "split_strategy": "stratified_host_separated",
        "history_size": int(history_size),
        "min_flows": int(min_flows),
        "inactivity_timeout": float(inactivity_timeout),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "random_seed": int(random_seed),
        "feature_names": list(FEATURE_NAMES_EXTENDED),
        "host_feature_names": list(HOST_FEATURE_NAMES),
        "inventory_path": str(inventory_path),
        "split_integrity": {
            "host_disjoint": are_sets_pairwise_disjoint(split_hosts.values()),
            "real_host_identity": all(summary["real_host_identity"] for summary in split_summaries.values()),
            "no_npz_synthetic_host_keys": True,
        },
        "evaluation_readiness": evaluation_readiness,
        "splits": split_summaries,
    }
    metadata_path = out_dir / "host_window_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "metadata": str(metadata_path),
        "paths": split_paths,
        "splits": split_summaries,
    }


def render_inventory_markdown(inventory: dict[str, object]) -> str:
    lines = [
        "# Host-Aware Dataset Inventory",
        "",
        f"Generated UTC: `{inventory['generated_at_utc']}`",
        "",
        "This inventory separates clean C2 labels from broader attack labels so host-aware model claims stay honest.",
        "",
        "| Source | Files | Rows | Real host ID | Label semantics | Split eligibility |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    sources = inventory.get("sources", {})
    for source_name, source in sources.items():
        if not isinstance(source, dict):
            continue
        lines.append(
            "| "
            f"{source_name} | "
            f"{source.get('file_count', 0)} | "
            f"{source.get('total_rows', 0)} | "
            f"{source.get('has_real_host_identity', False)} | "
            f"{source.get('label_semantics', '')} | "
            f"{source.get('split_eligibility', '')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- MCFP is used by default for the first clean host-aware split because its local files are explicit botnet vs normal captures.",
            "- UWF ZeekData24 has the best local Zeek-style host fields, but the local positive folders are not clean C2/TA0011 labels.",
            "- CTU-13 remains eligible for the next family/source split after choosing a duplicate capture policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_evaluation_readiness(
    split_summaries: dict[str, dict[str, object]],
) -> dict[str, object]:
    budgets = (0.005, 0.015, 0.03)
    budget_rows: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for budget in budgets:
        min_benign = int(math.ceil(1.0 / budget))
        per_split = {}
        for split_name in ("train", "val", "test"):
            benign = int(split_summaries.get(split_name, {}).get("benign", 0))
            per_split[split_name] = {
                "benign_samples": benign,
                "minimum_for_one_false_positive_step": min_benign,
                "ready": benign >= min_benign,
            }
        budget_rows[str(budget)] = per_split

    for split_name in ("val", "test"):
        benign = int(split_summaries.get(split_name, {}).get("benign", 0))
        if benign < int(math.ceil(1.0 / 0.015)):
            warnings.append(
                f"{split_name} has only {benign} benign samples; recall-at-1.5%-FPR is not stable for this host-separated split."
            )
    if warnings:
        warnings.append(
            "This is a source-data limitation: benign sessions are concentrated in a small number of hosts."
        )
    return {
        "fpr_budget_checks": budget_rows,
        "warnings": warnings,
    }


def renumber_records(records: Sequence[HostSessionRecord]) -> list[HostSessionRecord]:
    return [replace(record, original_index=i) for i, record in enumerate(records)]


def filter_ip_flow_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "src_ip" not in df.columns or "dst_ip" not in df.columns:
        raise ValueError("Expected src_ip and dst_ip columns before host-aware filtering")
    src_valid = df["src_ip"].astype(str).map(is_valid_ip)
    dst_valid = df["dst_ip"].astype(str).map(is_valid_ip)
    return df.loc[src_valid & dst_valid].reset_index(drop=True)


def is_valid_ip(value: object) -> bool:
    try:
        ipaddress.ip_address(str(value))
        return True
    except ValueError:
        return False


def is_mcfp_c2_file(path: Path) -> bool:
    name = path.name.lower()
    return "malware" in name or "botnet" in name


def mcfp_family_from_filename(path: Path) -> str:
    match = re.search(r"botnet-(\d+)", path.stem.lower())
    if match:
        return f"mcfp_botnet_{match.group(1)}"
    return "mcfp_botnet"


def read_header_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().strip()
    return [column.strip() for column in header.split(",")] if header else []


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def counter_to_dict(counter: Counter[str], *, limit: int) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common(limit)}


def safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def are_sets_pairwise_disjoint(sets: Iterable[set[str]]) -> bool:
    seen: set[str] = set()
    for item_set in sets:
        if seen & item_set:
            return False
        seen.update(item_set)
    return True


if __name__ == "__main__":
    main()
