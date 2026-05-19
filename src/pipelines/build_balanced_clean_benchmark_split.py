"""Build a balanced, clean host-aware benchmark split.

Policy
------
Positive/C2 traffic:
  - MCFP botnet raw BinetFlow files
  - CTU-13 raw BinetFlow rows whose Label parses as C2/botnet

Benign traffic:
  - UWF ZeekData24 normalized benign connection CSVs only

This is intentionally separate from attack-vs-benign UWF experiments.  UWF
attack folders are not included because the local labels are not clean C2.

The output remains host-aware:
  - split by source host
  - build causal host windows from all records assigned to each split
  - optionally downsample current samples after window construction so class
    balance improves without destroying available host history context
"""

from __future__ import annotations

import argparse
import json
import math
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
    HostWindowArrays,
    build_extended_host_session_records_from_dataframe,
    build_host_windows,
    save_host_windows_npz,
)
from src.data_loader.normalized_telemetry import load_and_normalize_telemetry
from src.data_loader.stratosphere_mcfp_processor import _load_flow_dataframe, _standardize_columns
from src.features.feature_config import label_from_string
from src.features.feature_config_extended import FEATURE_NAMES_EXTENDED, FEATURE_SCHEMA_VERSION
from src.features.feature_config_v2 import DatasetSource
from src.pipelines.build_host_aware_mvp_dataset import (
    filter_ip_flow_rows,
    is_mcfp_c2_file,
    mcfp_family_from_filename,
    normalize_uwf_conn_csv,
)


CHUNK_SIZE = 250_000
MODERN_C2_SOURCE = "modern_c2"
MODERN_C2_FAMILIES = ("sliver", "havoc", "cobalt_strike", "mythic")
MODERN_C2_SUFFIXES = {".csv", ".tsv", ".log", ".conn", ".jsonl", ".ndjson", ".parquet", ".pq"}
SUPPLEMENTARY_LOG_NAMES = {"dns.log", "ssl.log", "x509.log", "http.log", "weird.log", "notice.log"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced clean host-aware C2 benchmark split")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed/host_aware_clean_balanced_benchmark")
    parser.add_argument("--history-size", type=int, default=DEFAULT_HISTORY_SIZE)
    parser.add_argument("--c2-min-flows", type=int, default=5)
    parser.add_argument("--benign-min-flows", type=int, default=1)
    parser.add_argument("--inactivity-timeout", type=float, default=300.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--min-eval-benign", type=int, default=200)
    parser.add_argument("--min-eval-positive", type=int, default=50)
    parser.add_argument("--max-class-ratio", type=float, default=1.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-ctu13", action="store_true")
    parser.add_argument("--skip-mcfp", action="store_true")
    parser.add_argument("--skip-modern-c2", action="store_true")
    parser.add_argument("--modern-c2-dir", default="data/raw/modern_c2")
    parser.add_argument(
        "--modern-families",
        default="sliver,havoc,cobalt_strike,mythic",
        help="Comma-separated modern C2 families to ingest from --modern-c2-dir",
    )
    parser.add_argument("--max-ctu-files", type=int, default=0, help="0 means all unique local CTU files")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_records: list[HostSessionRecord] = []
    benign_records: list[HostSessionRecord] = []
    source_summaries: list[dict[str, object]] = []

    if not args.skip_mcfp:
        records, summary = build_mcfp_c2_records(
            raw_dir / "mcfp",
            min_flows=args.c2_min_flows,
            inactivity_timeout=args.inactivity_timeout,
        )
        positive_records.extend(records)
        source_summaries.append(summary)

    if not args.skip_ctu13:
        records, summary = build_ctu13_c2_records(
            raw_dir,
            min_flows=args.c2_min_flows,
            inactivity_timeout=args.inactivity_timeout,
            max_files=args.max_ctu_files,
        )
        positive_records.extend(records)
        source_summaries.append(summary)

    modern_families = parse_family_list(args.modern_families)
    if not args.skip_modern_c2:
        records, summary = build_modern_c2_records(
            Path(args.modern_c2_dir),
            families=modern_families,
            min_flows=args.c2_min_flows,
            inactivity_timeout=args.inactivity_timeout,
        )
        positive_records.extend(records)
        source_summaries.append(summary)

    records, summary = build_uwf_benign_records(
        raw_dir / "uwf_zeekdata24" / "benign",
        min_flows=args.benign_min_flows,
        inactivity_timeout=args.inactivity_timeout,
    )
    benign_records.extend(records)
    source_summaries.append(summary)

    if not positive_records:
        raise ValueError("No C2-positive records were produced from MCFP/CTU.")
    if not benign_records:
        raise ValueError("No benign records were produced from UWF benign telemetry.")

    all_records = renumber_records(positive_records + benign_records)
    assigned_splits = split_records_by_label_and_host(
        all_records,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        min_eval_benign=args.min_eval_benign,
        min_eval_positive=args.min_eval_positive,
        random_seed=args.random_seed,
    )

    outputs: dict[str, str] = {}
    split_summaries: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    split_host_sets: dict[str, set[str]] = {}

    for split_name in ("train", "val", "test"):
        full_records = assigned_splits[split_name]
        full_arrays = build_host_windows(
            full_records,
            history_size=args.history_size,
            feature_names=FEATURE_NAMES_EXTENDED,
            session_schema_version=FEATURE_SCHEMA_VERSION,
        )
        selected_indices, balance_summary = select_balanced_current_indices(
            full_arrays,
            split_name=split_name,
            max_class_ratio=args.max_class_ratio,
            min_eval_benign=args.min_eval_benign,
            min_eval_positive=args.min_eval_positive,
            random_seed=args.random_seed,
        )
        arrays = subset_host_window_arrays(full_arrays, selected_indices)
        path = out_dir / f"{split_name}_host_windows.npz"
        save_host_windows_npz(path, arrays)
        outputs[split_name] = str(path)
        split_host_sets[split_name] = set(str(host) for host in arrays.host_ids.tolist())
        split_summaries[split_name] = summarize_arrays(arrays, full_arrays, balance_summary)

        if split_name in {"val", "test"}:
            if split_summaries[split_name]["benign"] < args.min_eval_benign:
                warnings.append(
                    f"{split_name} has {split_summaries[split_name]['benign']} benign samples, below requested {args.min_eval_benign}."
                )
            if split_summaries[split_name]["c2"] < args.min_eval_positive:
                warnings.append(
                    f"{split_name} has {split_summaries[split_name]['c2']} C2 samples, below requested {args.min_eval_positive}."
                )

    c2_sources = {
        str(summary["source"])
        for summary in source_summaries
        if summary.get("label") == "c2" and int(summary.get("sessions", 0)) > 0
    }
    for split_name in ("val", "test"):
        missing_sources = sorted(c2_sources - set(split_summaries[split_name]["c2_sources"]))
        if missing_sources:
            warnings.append(
                f"{split_name} C2 coverage is missing {missing_sources} because host-separated splitting keeps low-diversity C2 hosts out of that split."
            )

    for summary in source_summaries:
        if summary.get("source") == MODERN_C2_SOURCE:
            missing_families = list(summary.get("missing_families", []))
            if missing_families:
                warnings.append(
                    f"modern C2 telemetry missing families {missing_families}; add authorized Zeek/normalized files under {summary.get('root')}."
                )

    metadata = {
        "schema_version": HOST_AWARE_SCHEMA_VERSION,
        "session_schema_version": FEATURE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "positive_sources": [
                "mcfp_stratosphere botnet raw BinetFlow files",
                "ctu13 raw BinetFlow rows parsed as C2/botnet",
                "modern_c2 authorized Zeek/normalized telemetry under data/raw/modern_c2 when present",
            ],
            "benign_sources": ["uwf_zeekdata24 benign normalized-Zeek CSV"],
            "excluded_sources": [
                "uwf_zeekdata24 attack tactic folders because local positives are not clean C2 labels",
            ],
            "class_balance": f"majority current samples capped to about {args.max_class_ratio}:1 per split when possible",
            "host_separation": "source hosts are disjoint across train/val/test",
        },
        "settings": {
            "history_size": args.history_size,
            "c2_min_flows": args.c2_min_flows,
            "benign_min_flows": args.benign_min_flows,
            "inactivity_timeout": args.inactivity_timeout,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "min_eval_benign": args.min_eval_benign,
            "min_eval_positive": args.min_eval_positive,
            "max_class_ratio": args.max_class_ratio,
            "random_seed": args.random_seed,
            "modern_c2_dir": str(Path(args.modern_c2_dir)),
            "modern_families": modern_families,
        },
        "feature_names": list(FEATURE_NAMES_EXTENDED),
        "host_feature_names": list(HOST_FEATURE_NAMES),
        "source_summaries": source_summaries,
        "raw_record_counts": {
            "c2": len(positive_records),
            "benign": len(benign_records),
            "total": len(all_records),
        },
        "split_integrity": {
            "host_disjoint": are_sets_pairwise_disjoint(split_host_sets.values()),
            "real_host_identity": True,
            "label_policy_clean_c2_vs_benign": True,
        },
        "evaluation_readiness": {
            "fpr_budget_benign_requirements": {
                "0.005": 200,
                "0.015": 67,
                "0.03": 34,
            },
            "warnings": warnings,
        },
        "splits": split_summaries,
        "outputs": outputs,
    }
    metadata_path = out_dir / "clean_benchmark_split_metadata.json"
    metadata_path.write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
    benchmark_alias_path = out_dir / "host_window_split_metadata.json"
    benchmark_alias_path.write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
    markdown_path = out_dir / "clean_benchmark_split_metadata.md"
    markdown_path.write_text(render_markdown(metadata), encoding="utf-8")

    print(f"[SUCCESS] Saved clean balanced host-aware split to {out_dir}")
    print(f"[SUCCESS] Metadata JSON: {metadata_path}")
    print(f"[SUCCESS] Benchmark metadata alias: {benchmark_alias_path}")
    print(f"[SUCCESS] Metadata Markdown: {markdown_path}")
    for split_name, summary in split_summaries.items():
        print(
            f"[INFO] {split_name}: samples={summary['samples']:,}, "
            f"c2={summary['c2']:,}, benign={summary['benign']:,}, "
            f"hosts={summary['hosts']:,}, ratio={summary['class_ratio_majority_to_minority']:.2f}"
        )
    if warnings:
        print("[WARN] " + " | ".join(warnings))


def build_mcfp_c2_records(
    source_dir: Path,
    *,
    min_flows: int,
    inactivity_timeout: float,
) -> tuple[list[HostSessionRecord], dict[str, object]]:
    records: list[HostSessionRecord] = []
    files = [path for path in sorted(source_dir.glob("*.binetflow")) if is_mcfp_c2_file(path)]
    file_summaries: list[dict[str, object]] = []
    for path in files:
        df = filter_ip_flow_rows(_load_flow_dataframe(str(path)))
        df = df.copy()
        df["label"] = 1
        file_records = build_extended_host_session_records_from_dataframe(
            df,
            source=DatasetSource.MCFP_STRATOSPHERE.value,
            family=mcfp_family_from_filename(path),
            capture_id=path.stem,
            label_col="label",
            inactivity_timeout=inactivity_timeout,
            min_flows=min_flows,
        )
        records.extend(file_records)
        file_summaries.append(
            {
                "path": str(path),
                "flows": int(len(df)),
                "sessions": int(len(file_records)),
                "hosts": int(df["src_ip"].nunique()),
            }
        )
        print(f"[INFO] MCFP C2 {path.name}: flows={len(df):,}, sessions={len(file_records):,}")
    return records, {
        "source": DatasetSource.MCFP_STRATOSPHERE.value,
        "label": "c2",
        "files": file_summaries,
        "sessions": len(records),
    }


def build_ctu13_c2_records(
    raw_dir: Path,
    *,
    min_flows: int,
    inactivity_timeout: float,
    max_files: int,
) -> tuple[list[HostSessionRecord], dict[str, object]]:
    records: list[HostSessionRecord] = []
    files = unique_ctu13_files(raw_dir)
    if max_files > 0:
        files = files[:max_files]
    file_summaries: list[dict[str, object]] = []
    for path in files:
        df = load_ctu13_c2_dataframe(path)
        if df.empty:
            file_summaries.append({"path": str(path), "flows": 0, "sessions": 0, "hosts": 0})
            continue
        df = filter_ip_flow_rows(df)
        df = df.copy()
        df["label"] = 1
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
        file_summaries.append(
            {
                "path": str(path),
                "flows": int(len(df)),
                "sessions": int(len(file_records)),
                "hosts": int(df["src_ip"].nunique()),
            }
        )
        print(f"[INFO] CTU13 C2 {path.name}: flows={len(df):,}, sessions={len(file_records):,}")
    return records, {
        "source": DatasetSource.CTU13.value,
        "label": "c2",
        "files": file_summaries,
        "sessions": len(records),
    }


def build_modern_c2_records(
    modern_dir: Path,
    *,
    families: Sequence[str],
    min_flows: int,
    inactivity_timeout: float,
) -> tuple[list[HostSessionRecord], dict[str, object]]:
    """Load authorized modern C2 telemetry from Zeek/normalized files.

    Expected layout is intentionally simple and filesystem-label driven:

        data/raw/modern_c2/sliver/<capture>/conn.log
        data/raw/modern_c2/havoc/<capture>/flows.csv
        data/raw/modern_c2/cobalt_strike/<capture>/normalized.parquet
        data/raw/modern_c2/mythic/<capture>/conn.jsonl

    Files are treated as positive C2 only when their path contains one of the
    requested family names or common aliases.  Supplementary Zeek sidecars are
    ignored for now; conn/flow telemetry is the source of session windows.
    """
    canonical_families = tuple(canonical_modern_family(family) for family in families)
    summary: dict[str, object] = {
        "source": MODERN_C2_SOURCE,
        "label": "c2",
        "root": str(modern_dir),
        "expected_families": list(canonical_families),
        "present_families": [],
        "missing_families": list(canonical_families),
        "files": [],
        "sessions": 0,
    }
    if not modern_dir.exists():
        print(f"[WARN] Modern C2 directory not found: {modern_dir}")
        return [], summary

    files = discover_modern_c2_files(modern_dir, canonical_families)
    records: list[HostSessionRecord] = []
    file_summaries: list[dict[str, object]] = []
    present_families: set[str] = set()

    for path, family in files:
        try:
            df = load_and_normalize_telemetry(
                str(path),
                source_dataset=MODERN_C2_SOURCE,
                label=1,
                family=family,
                capture_id=modern_capture_id(modern_dir, path, family),
            )
            df = filter_ip_flow_rows(df)
            if df.empty:
                file_summaries.append(
                    {
                        "path": str(path),
                        "family": family,
                        "flows": 0,
                        "sessions": 0,
                        "hosts": 0,
                        "status": "skipped_empty_after_ip_filter",
                    }
                )
                continue
            file_records = build_extended_host_session_records_from_dataframe(
                df,
                source=MODERN_C2_SOURCE,
                family=family,
                capture_id=modern_capture_id(modern_dir, path, family),
                label_col="label",
                inactivity_timeout=inactivity_timeout,
                min_flows=min_flows,
            )
            records.extend(file_records)
            present_families.add(family)
            file_summaries.append(
                {
                    "path": str(path),
                    "family": family,
                    "flows": int(len(df)),
                    "sessions": int(len(file_records)),
                    "hosts": int(df["src_ip"].nunique()),
                    "status": "loaded",
                }
            )
            print(
                f"[INFO] Modern C2 {family} {path.name}: "
                f"flows={len(df):,}, sessions={len(file_records):,}"
            )
        except Exception as exc:
            file_summaries.append(
                {
                    "path": str(path),
                    "family": family,
                    "flows": 0,
                    "sessions": 0,
                    "hosts": 0,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[WARN] Failed to ingest modern C2 telemetry {path}: {exc}")

    summary["files"] = file_summaries
    summary["sessions"] = len(records)
    summary["present_families"] = sorted(present_families)
    summary["missing_families"] = sorted(set(canonical_families) - present_families)
    return records, summary


def discover_modern_c2_files(
    modern_dir: Path,
    families: Sequence[str],
) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for path in sorted(modern_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in SUPPLEMENTARY_LOG_NAMES:
            continue
        if path.suffix.lower() not in MODERN_C2_SUFFIXES:
            continue
        family = infer_modern_family_from_path(path, families)
        if family:
            out.append((path, family))
    return out


def infer_modern_family_from_path(path: Path, families: Sequence[str]) -> str | None:
    canonical_allowed = {canonical_modern_family(family) for family in families}
    text = str(path).lower().replace("\\", "/").replace("-", "_")
    aliases = {
        "sliver": ("sliver",),
        "havoc": ("havoc",),
        "cobalt_strike": ("cobalt_strike", "cobaltstrike", "cobalt/strike", "cobalt"),
        "mythic": ("mythic",),
    }
    for family, family_aliases in aliases.items():
        if family in canonical_allowed and any(alias in text for alias in family_aliases):
            return family
    return None


def canonical_modern_family(raw: str) -> str:
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"cobalt", "cobaltstrike"}:
        return "cobalt_strike"
    return normalized


def modern_capture_id(root: Path, path: Path, family: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts = [family, *relative.with_suffix("").parts]
    safe = "_".join(str(part).replace(" ", "_").replace("-", "_") for part in parts)
    return safe[:180]


def build_uwf_benign_records(
    benign_dir: Path,
    *,
    min_flows: int,
    inactivity_timeout: float,
) -> tuple[list[HostSessionRecord], dict[str, object]]:
    records: list[HostSessionRecord] = []
    file_summaries: list[dict[str, object]] = []
    files = sorted(benign_dir.glob("*.csv"))
    for path in files:
        df = filter_ip_flow_rows(normalize_uwf_conn_csv(path))
        df = df.copy()
        df["label"] = 0
        file_records = build_extended_host_session_records_from_dataframe(
            df,
            source=DatasetSource.UWF_ZEEKDATA24.value,
            family="",
            capture_id=f"uwf_benign_{path.stem}",
            label_col="label",
            inactivity_timeout=inactivity_timeout,
            min_flows=min_flows,
        )
        records.extend(file_records)
        file_summaries.append(
            {
                "path": str(path),
                "flows": int(len(df)),
                "sessions": int(len(file_records)),
                "hosts": int(df["src_ip"].nunique()),
            }
        )
        print(f"[INFO] UWF benign {path.name}: flows={len(df):,}, sessions={len(file_records):,}")
    return records, {
        "source": DatasetSource.UWF_ZEEKDATA24.value,
        "label": "benign",
        "files": file_summaries,
        "sessions": len(records),
    }


def unique_ctu13_files(raw_dir: Path) -> list[Path]:
    candidates = [
        raw_dir / "ctu13" / "capture20110810.binetflow",
        raw_dir / "ctu13_extra" / "capture20110811.binetflow",
        raw_dir / "ctu13_extra" / "capture20110815.binetflow",
    ]
    return [path for path in candidates if path.exists()]


def parse_family_list(raw_value: str) -> list[str]:
    families = [canonical_modern_family(part) for part in raw_value.split(",") if part.strip()]
    return families or list(MODERN_C2_FAMILIES)


def load_ctu13_c2_dataframe(path: Path) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    total = 0
    c2_rows = 0
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        total += int(len(chunk))
        labels = chunk["Label"].fillna("").astype(str).map(label_from_string).astype(bool)
        if labels.any():
            c2_chunk = chunk.loc[labels].copy()
            chunks.append(c2_chunk)
            c2_rows += int(len(c2_chunk))
    print(f"[INFO] CTU13 scan {path.name}: rows={total:,}, c2_rows={c2_rows:,}")
    if not chunks:
        return pd.DataFrame()
    raw_c2 = pd.concat(chunks, axis=0, ignore_index=True)
    return _standardize_columns(raw_c2)


def split_records_by_label_and_host(
    records: Sequence[HostSessionRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    min_eval_benign: int,
    min_eval_positive: int,
    random_seed: int,
) -> dict[str, list[HostSessionRecord]]:
    rng = np.random.default_rng(random_seed)
    splits = {"train": [], "val": [], "test": []}
    for label, min_eval in ((1, min_eval_positive), (0, min_eval_benign)):
        class_records = [record for record in records if int(record.label) == label]
        host_assignments = assign_hosts_by_count(
            class_records,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            min_eval=min_eval,
            rng=rng,
        )
        for split_name, hosts in host_assignments.items():
            splits[split_name].extend(record for record in class_records if record.host_id in hosts)
    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=lambda item: (item.timestamp, item.original_index))
    return splits


def assign_hosts_by_count(
    records: Sequence[HostSessionRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    min_eval: int,
    rng: np.random.Generator,
) -> dict[str, set[str]]:
    host_counts = Counter(record.host_id for record in records)
    hosts = list(host_counts)
    if len(hosts) < 3:
        raise ValueError(f"Need at least three hosts for host-separated split, got {len(hosts)}")
    total = sum(host_counts.values())
    test_target = min(max(int(round(total * test_fraction)), min_eval), max(1, total - 2))
    val_target = min(max(int(round(total * val_fraction)), min_eval), max(1, total - test_target - 1))

    shuffled_hosts = hosts[:]
    rng.shuffle(shuffled_hosts)
    shuffled_hosts.sort(key=lambda host: host_counts[host], reverse=True)

    test_hosts = pick_hosts_for_target(shuffled_hosts, host_counts, test_target)
    remaining = [host for host in shuffled_hosts if host not in test_hosts]
    val_hosts = pick_hosts_for_target(remaining, host_counts, val_target)
    train_hosts = set(hosts) - test_hosts - val_hosts
    if not train_hosts:
        # Move the smallest validation host back to train.
        smallest_val = min(val_hosts, key=lambda host: host_counts[host])
        val_hosts.remove(smallest_val)
        train_hosts.add(smallest_val)
    return {"train": train_hosts, "val": val_hosts, "test": test_hosts}


def pick_hosts_for_target(
    hosts: Sequence[str],
    host_counts: Counter[str],
    target: int,
) -> set[str]:
    selected: set[str] = set()
    selected_count = 0
    for host in hosts:
        if not selected:
            selected.add(host)
            selected_count += host_counts[host]
            continue
        if selected_count >= target:
            break
        selected.add(host)
        selected_count += host_counts[host]
    return selected


def select_balanced_current_indices(
    arrays: HostWindowArrays,
    *,
    split_name: str,
    max_class_ratio: float,
    min_eval_benign: int,
    min_eval_positive: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(random_seed + {"train": 0, "val": 17, "test": 31}[split_name])
    labels = arrays.labels.astype(int)
    benign_idx = np.where(labels == 0)[0]
    c2_idx = np.where(labels == 1)[0]
    keep_benign = len(benign_idx)
    keep_c2 = len(c2_idx)

    if keep_benign and keep_c2:
        if keep_benign > keep_c2 * max_class_ratio:
            min_keep = min_eval_benign if split_name in {"val", "test"} else 1
            keep_benign = min(keep_benign, max(int(math.ceil(keep_c2 * max_class_ratio)), min_keep))
        if keep_c2 > keep_benign * max_class_ratio:
            min_keep = min_eval_positive if split_name in {"val", "test"} else 1
            keep_c2 = min(keep_c2, max(int(math.ceil(keep_benign * max_class_ratio)), min_keep))

    selected_benign = sample_indices(benign_idx, keep_benign, rng)
    selected_c2 = sample_indices(c2_idx, keep_c2, rng)
    selected = np.sort(np.concatenate([selected_benign, selected_c2]).astype(np.int64))
    return selected, {
        "full_benign": int(len(benign_idx)),
        "full_c2": int(len(c2_idx)),
        "selected_benign": int(len(selected_benign)),
        "selected_c2": int(len(selected_c2)),
        "max_class_ratio": float(max_class_ratio),
    }


def sample_indices(indices: np.ndarray, keep: int, rng: np.random.Generator) -> np.ndarray:
    if keep >= len(indices):
        return indices.astype(np.int64)
    return rng.choice(indices, size=keep, replace=False).astype(np.int64)


def subset_host_window_arrays(arrays: HostWindowArrays, indices: np.ndarray) -> HostWindowArrays:
    return HostWindowArrays(
        current_sessions=arrays.current_sessions[indices],
        current_masks=arrays.current_masks[indices],
        history_sessions=arrays.history_sessions[indices],
        history_flow_masks=arrays.history_flow_masks[indices],
        history_session_masks=arrays.history_session_masks[indices],
        host_features=arrays.host_features[indices],
        labels=arrays.labels[indices],
        host_ids=arrays.host_ids[indices],
        dest_ids=arrays.dest_ids[indices],
        timestamps=arrays.timestamps[indices],
        sources=arrays.sources[indices],
        families=arrays.families[indices],
        capture_ids=arrays.capture_ids[indices],
        original_indices=arrays.original_indices[indices],
        feature_names=arrays.feature_names,
        host_feature_names=arrays.host_feature_names,
        schema_version=arrays.schema_version,
        session_schema_version=arrays.session_schema_version,
        history_size=arrays.history_size,
        real_host_identity=arrays.real_host_identity,
    )


def summarize_arrays(
    arrays: HostWindowArrays,
    full_arrays: HostWindowArrays,
    balance_summary: dict[str, object],
) -> dict[str, object]:
    benign = int((arrays.labels == 0).sum())
    c2 = int((arrays.labels == 1).sum())
    majority = max(benign, c2)
    minority = max(1, min(benign, c2))
    history_counts = arrays.history_session_masks.sum(axis=1) if len(arrays.labels) else np.asarray([])
    return {
        "samples": int(len(arrays.labels)),
        "c2": c2,
        "benign": benign,
        "hosts": int(len(set(str(host) for host in arrays.host_ids.tolist()))),
        "sources": sorted(set(str(source) for source in arrays.sources.tolist())),
        "c2_sources": sorted(set(str(source) for source in arrays.sources[arrays.labels == 1].tolist())),
        "benign_sources": sorted(set(str(source) for source in arrays.sources[arrays.labels == 0].tolist())),
        "families": sorted(set(str(family) for family in arrays.families.tolist() if str(family))),
        "class_ratio_majority_to_minority": float(majority / minority),
        "mean_history_sessions": float(np.mean(history_counts)) if history_counts.size else 0.0,
        "max_history_sessions": int(np.max(history_counts)) if history_counts.size else 0,
        "full_assigned_samples_before_balancing": int(len(full_arrays.labels)),
        "full_assigned_c2_before_balancing": int((full_arrays.labels == 1).sum()),
        "full_assigned_benign_before_balancing": int((full_arrays.labels == 0).sum()),
        "balance_summary": balance_summary,
    }


def renumber_records(records: Sequence[HostSessionRecord]) -> list[HostSessionRecord]:
    return [replace(record, original_index=i) for i, record in enumerate(records)]


def are_sets_pairwise_disjoint(sets: Iterable[set[str]]) -> bool:
    seen: set[str] = set()
    for item_set in sets:
        if seen & item_set:
            return False
        seen.update(item_set)
    return True


def render_markdown(metadata: dict[str, object]) -> str:
    lines = [
        "# Clean Balanced Host-Aware Benchmark Split",
        "",
        f"Generated UTC: `{metadata['generated_at_utc']}`",
        "",
        "## Policy",
        "",
        "- C2 positives: MCFP botnet files plus CTU-13 rows parsed as C2/botnet.",
        "- Benign: UWF ZeekData24 benign normalized-Zeek CSV only.",
        "- UWF attack-tactic folders are excluded because they are not clean C2 labels.",
        "- Splits are source-host separated.",
        "",
        "## Splits",
        "",
        "| Split | Samples | C2 | Benign | Hosts | Class Ratio | Sources |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split_name in ("train", "val", "test"):
        split = metadata["splits"][split_name]
        lines.append(
            f"| {split_name} | {split['samples']} | {split['c2']} | {split['benign']} | "
            f"{split['hosts']} | {split['class_ratio_majority_to_minority']:.2f} | "
            f"{', '.join(split['sources'])} |"
        )
    warnings = metadata["evaluation_readiness"]["warnings"]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.extend(["", "## Outputs", ""])
    for split_name, path in metadata["outputs"].items():
        lines.append(f"- {split_name}: `{path}`")
    return "\n".join(lines) + "\n"


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
