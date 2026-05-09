"""CyberShield v2 — Dataset builder with host-centric timeline support.

This is the true v2 builder. It produces TWO outputs per split:
  1. Session tensors (backward-compatible with v1 training)
  2. Host timelines (new: per-host behavioral histories for Phase 2+)

The key architectural shift: sessions are no longer anonymous floating tensors.
Every session is linked to a host identity, enabling longitudinal behavioral
analysis downstream.

Output structure:
    out_dir/
    ├── train_sessions.npz          # (N, 20, 12) session tensors
    ├── train_host_timelines.pkl    # Dict[host_id → HostTimeline]
    ├── val_sessions.npz
    ├── val_host_timelines.pkl
    ├── test_sessions.npz
    ├── test_host_timelines.pkl
    └── split_metadata.json         # Provenance + split diagnostics

Usage:
    python -m src.features.dataset_builder_v2 \\
        --sources data/processed/uwf/uwf_c2_sessions.npz:uwf_zeekdata24:c2 \\
                  data/processed/ctu13_c2_sessions.npz:ctu13:c2 \\
                  data/processed/uwf/uwf_benign_sessions.npz:uwf_zeekdata24:benign \\
        --split_strategy family_separated \\
        --out_dir data/processed/v2_mixed
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from src.features.feature_config_v2 import (
    DatasetSource,
    FEATURE_DIM,
    FEATURE_INDEX,
    FEATURE_NAMES,
    HostTimelineConfig,
    LABEL_BENIGN,
    LABEL_C2,
    SESSION_LEN,
    SessionMetadata,
    VAL_SPLIT,
    TEST_SPLIT,
)
from src.features.session_builder import load_sessions, save_sessions
from src.features.host_timeline import (
    HostTimeline,
    HostTimelineBuilder,
    SessionSummary,
)
from src.data_loader.split_utils import (
    SplitConfig,
    SplitStrategy,
    SplitResult,
    print_split_report,
    split_dataset,
)


# ──────────────────────────────────────────────────────────────────────
# Source specification parsing
# ──────────────────────────────────────────────────────────────────────

def parse_source_spec(spec: str) -> Tuple[str, DatasetSource, str]:
    """Parse a source specification string.

    Format: <npz_path>:<source_name>:<label_type>
    Example: data/processed/uwf/uwf_c2_sessions.npz:uwf_zeekdata24:c2

    Uses rsplit to handle Windows paths containing colons (e.g., D:\\path).
    """
    parts = spec.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid source spec '{spec}'. Expected format: "
            "<npz_path>:<source_name>:<label_type>"
        )
    npz_path, source_name, label_type = parts

    source_map = {s.value: s for s in DatasetSource}
    source = source_map.get(source_name.lower(), DatasetSource.UNKNOWN)

    label_type = label_type.lower()
    if label_type not in ("c2", "benign", "mixed"):
        raise ValueError(
            f"Invalid label_type '{label_type}'. Expected: c2, benign, or mixed"
        )

    return npz_path, source, label_type


# ──────────────────────────────────────────────────────────────────────
# Family inference
# ──────────────────────────────────────────────────────────────────────

# Ordered by specificity — more specific patterns first.
_FAMILY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"cobalt[_\-]?strike", re.I), "cobalt_strike"),
    (re.compile(r"brute[_\-]?ratel", re.I), "brute_ratel"),
    (re.compile(r"meterpreter", re.I), "metasploit"),
    (re.compile(r"sliver", re.I), "sliver"),
    (re.compile(r"havoc", re.I), "havoc"),
    (re.compile(r"metasploit", re.I), "metasploit"),
    (re.compile(r"mythic", re.I), "mythic"),
    (re.compile(r"covenant", re.I), "covenant"),
    (re.compile(r"poshc2", re.I), "poshc2"),
    (re.compile(r"emotet", re.I), "emotet"),
    (re.compile(r"trickbot", re.I), "trickbot"),
    (re.compile(r"qakbot", re.I), "qakbot"),
    (re.compile(r"neris", re.I), "neris"),
    (re.compile(r"rbot", re.I), "rbot"),
    (re.compile(r"virut", re.I), "virut"),
    (re.compile(r"menti", re.I), "menti"),
    (re.compile(r"sogou", re.I), "sogou"),
    (re.compile(r"murlo", re.I), "murlo"),
    # MITRE technique IDs
    (re.compile(r"(t1\d{3})", re.I), None),  # handled specially below
    # Generic dataset markers — only if nothing else matches
    (re.compile(r"ctu13", re.I), "ctu13_mixed"),
    (re.compile(r"uwf", re.I), "uwf_mixed"),
    (re.compile(r"mcfp", re.I), "mcfp_mixed"),
]


def _infer_family_from_path(path: str) -> str:
    """Infer C2 family from file path using regex patterns."""
    for pattern, family in _FAMILY_PATTERNS:
        match = pattern.search(path)
        if match:
            if family is None:
                # MITRE technique ID
                return f"mitre_{match.group(1).lower()}"
            return family
    return "unknown"


# ──────────────────────────────────────────────────────────────────────
# Host identity extraction from session tensors
# ──────────────────────────────────────────────────────────────────────

# Cached index lookups — never use raw ints for column access.
_FI = FEATURE_INDEX


def _extract_host_key(
    session: np.ndarray,
    mask: np.ndarray,
    metadata: SessionMetadata,
) -> str:
    """Derive a deterministic host fingerprint from a session tensor.

    Uses multiple behavioral signals (dst_port, src_port, byte_ratio,
    packet_ratio, outbound flag) to reduce collision rate compared to
    the single-signal approach.  Scoped by source+capture to prevent
    cross-dataset collisions.

    When raw flow data with real IPs is available, prefer
    HostTimelineBuilder.ingest_flow_dataframe() instead — it uses
    actual src_ip as host_id.

    All column accesses go through FEATURE_INDEX, not hardcoded ints.
    """
    real = session[mask] if mask.any() else session[:3]

    if len(real) == 0:
        return f"unknown_{metadata.capture_id}_{metadata.source.value}"

    # Multi-signal fingerprint via named feature indices
    dst_port_sig = int(np.median(real[:, _FI["dst_port"]]))
    src_port_sig = int(np.median(real[:, _FI["src_port"]]))
    byte_ratio_sig = int(np.median(real[:, _FI["byte_ratio"]]) * 100)
    pkt_ratio_sig = int(np.median(real[:, _FI["packet_ratio"]]) * 100)
    outbound_mode = int(np.round(real[:, _FI["is_outbound"]].mean()))

    return (
        f"{metadata.source.value}:{metadata.capture_id}"
        f":dp{dst_port_sig}:sp{src_port_sig}"
        f":br{byte_ratio_sig}:pr{pkt_ratio_sig}:o{outbound_mode}"
    )


# ──────────────────────────────────────────────────────────────────────
# Multi-source ingestion
# ──────────────────────────────────────────────────────────────────────

def load_multi_source(
    source_specs: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SessionMetadata]]:
    """Load and combine multiple NPZ sources with per-session metadata.

    Every session gets a SessionMetadata with host_id populated.

    Returns
    -------
    sessions : (N, SESSION_LEN, FEATURE_DIM)
    labels : (N,)
    masks : (N, SESSION_LEN)
    metadata : List[SessionMetadata] — one per session, host_id populated
    """
    all_sessions, all_labels, all_masks = [], [], []
    all_metadata: List[SessionMetadata] = []

    for spec in source_specs:
        npz_path, source, label_type = parse_source_spec(spec)
        print(f"[INFO] Loading {npz_path} (source={source.value}, type={label_type})")

        sessions, labels, masks = load_sessions(npz_path)

        if label_type == "c2":
            labels = np.ones(len(sessions), dtype=np.int64)
        elif label_type == "benign":
            labels = np.zeros(len(sessions), dtype=np.int64)

        assert sessions.shape[1:] == (SESSION_LEN, FEATURE_DIM), (
            f"Shape mismatch in {npz_path}: {sessions.shape}"
        )

        # Infer once per source; apply per-session only for positive labels.
        # This keeps mixed sources family-aware for family_separated splits.
        inferred_family = _infer_family_from_path(npz_path)
        capture_id = os.path.basename(npz_path)

        for i in range(len(sessions)):
            # Build metadata first (without host_id), then derive host key
            meta_partial = SessionMetadata(
                source=source,
                c2_family=inferred_family if labels[i] == LABEL_C2 else "",
                capture_id=capture_id,
            )
            host_key = _extract_host_key(sessions[i], masks[i], meta_partial)
            # Rebuild with host_id populated (frozen dataclass)
            meta = SessionMetadata(
                source=source,
                c2_family=inferred_family if labels[i] == LABEL_C2 else "",
                capture_id=capture_id,
                host_id=host_key,
            )
            all_metadata.append(meta)

        all_sessions.append(sessions)
        all_labels.append(labels)
        all_masks.append(masks)

        n_c2 = int((labels == LABEL_C2).sum())
        n_benign = int((labels == LABEL_BENIGN).sum())
        print(f"  → {len(sessions):,} sessions (C2: {n_c2:,}, Benign: {n_benign:,})")

    if not all_sessions:
        raise ValueError("No sessions loaded from any source")

    sessions = np.concatenate(all_sessions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Report host diversity
    unique_hosts = len({m.host_id for m in all_metadata})
    total_c2 = int((labels == LABEL_C2).sum())
    total_benign = int((labels == LABEL_BENIGN).sum())
    print(
        f"\n[TOTAL] {len(sessions):,} sessions | "
        f"C2: {total_c2:,} ({100*total_c2/len(sessions):.1f}%) | "
        f"Benign: {total_benign:,} | "
        f"Unique hosts: {unique_hosts:,}"
    )

    return sessions, labels, masks, all_metadata


# ──────────────────────────────────────────────────────────────────────
# Host timeline construction from session tensors
# ──────────────────────────────────────────────────────────────────────

def build_host_timelines_from_sessions(
    sessions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    metadata: List[SessionMetadata],
    config: Optional[HostTimelineConfig] = None,
) -> Dict[str, HostTimeline]:
    """Build per-host timelines from session tensors + metadata.

    This bridges the gap between NPZ session data (which loses IP identity)
    and the host-centric HostTimeline objects. Each session is assigned to
    its host_id from metadata and converted into a SessionSummary.

    Parameters
    ----------
    sessions : (N, SESSION_LEN, FEATURE_DIM) session tensors
    labels : (N,) labels
    masks : (N, SESSION_LEN) real-flow masks (True = real flow)
    metadata : Per-session metadata with host_id populated
    config : Host timeline configuration

    Returns
    -------
    Dict mapping host_id → HostTimeline
    """
    if config is None:
        config = HostTimelineConfig()

    timelines: Dict[str, HostTimeline] = {}

    for i in tqdm(range(len(sessions)), desc="Building host timelines", leave=False):
        meta = metadata[i]
        host_id = meta.host_id or f"unknown_host_{i}"

        if host_id not in timelines:
            timelines[host_id] = HostTimeline(host_id=host_id, config=config)

        # Single authoritative conversion — all index logic lives in
        # SessionSummary.from_tensor(), not scattered here.
        summary = SessionSummary.from_tensor(
            session=sessions[i],
            mask=masks[i],
            label=int(labels[i]),
            session_id=f"s{i:08d}",
            host_id=host_id,
            metadata=meta,
        )

        timelines[host_id].add_session(summary)

    n_with_baseline = sum(1 for t in timelines.values() if t.has_baseline)
    print(
        f"[HOST TIMELINES] {len(timelines):,} hosts | "
        f"{n_with_baseline:,} with baseline | "
        f"Total sessions: {len(sessions):,}"
    )

    return timelines


# ──────────────────────────────────────────────────────────────────────
# Timeline I/O
# ──────────────────────────────────────────────────────────────────────

def save_host_timelines(timelines: Dict[str, HostTimeline], path: str) -> None:
    """Serialize host timelines to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(timelines, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved {len(timelines):,} host timelines → {path}")


def load_host_timelines(path: str) -> Dict[str, HostTimeline]:
    """Load host timelines from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_timeline_summary(timelines: Dict[str, HostTimeline]) -> Dict:
    """Compute summary statistics for a set of host timelines."""
    if not timelines:
        return {"hosts": 0}
    session_counts = [t.num_sessions for t in timelines.values()]
    dest_counts = [t.unique_destinations for t in timelines.values()]
    return {
        "hosts": len(timelines),
        "total_sessions": sum(session_counts),
        "sessions_per_host_mean": float(np.mean(session_counts)),
        "sessions_per_host_median": float(np.median(session_counts)),
        "sessions_per_host_max": int(np.max(session_counts)),
        "hosts_with_baseline": sum(1 for t in timelines.values() if t.has_baseline),
        "unique_destinations_mean": float(np.mean(dest_counts)),
    }


# ──────────────────────────────────────────────────────────────────────
# Dataset building pipeline
# ──────────────────────────────────────────────────────────────────────

def build_v2_dataset(
    source_specs: List[str],
    out_dir: str,
    split_strategy: str = "family_separated",
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    random_seed: int = 42,
    c2_ratio: Optional[float] = None,
    held_out_families: Optional[str] = None,
    held_out_sources: Optional[str] = None,
    build_timelines: bool = True,
    timeline_config: Optional[HostTimelineConfig] = None,
) -> Dict[str, str]:
    """Full v2 dataset build pipeline.

    Produces per-split session NPZs AND host timeline pickles.

    Parameters
    ----------
    source_specs : Source spec strings (path:source:type).
    out_dir : Output directory.
    split_strategy : One of: random_stratified, family_separated,
                     source_separated, time_separated, zero_shot.
    val_split : Validation fraction.
    test_split : Test fraction.
    random_seed : Random seed.
    c2_ratio : Optional target C2 ratio (None = natural distribution).
    held_out_families : Comma-separated families to hold out for test.
    build_timelines : Whether to build host timelines (default True).
    timeline_config : Configuration for host timeline construction.

    Returns
    -------
    Dict mapping output type to file path.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load all sources with host-aware metadata
    sessions, labels, masks, metadata = load_multi_source(source_specs)

    # 2. Optionally rebalance C2 ratio
    if c2_ratio is not None:
        sessions, labels, masks, metadata = _rebalance(
            sessions, labels, masks, metadata, c2_ratio, random_seed
        )

    # 3. Configure and execute split
    strategy = SplitStrategy(split_strategy)
    config = SplitConfig(
        strategy=strategy,
        val_fraction=val_split,
        test_fraction=test_split,
        random_seed=random_seed,
    )
    if held_out_families:
        config.held_out_families = set(held_out_families.split(","))
    if held_out_sources:
        config.zero_shot_sources = set(held_out_sources.split(","))

    split_result = split_dataset(labels, metadata, config)
    print_split_report(split_result, labels)

    # 4. Save each split: session NPZ + host timelines
    outputs: Dict[str, str] = {}

    for split_name, indices in [
        ("train", split_result.train_indices),
        ("val", split_result.val_indices),
        ("test", split_result.test_indices),
    ]:
        # Session tensors (backward-compatible)
        npz_path = os.path.join(out_dir, f"{split_name}_sessions.npz")
        save_sessions(sessions[indices], labels[indices], masks[indices], npz_path)
        outputs[f"{split_name}_sessions"] = npz_path

        # Host timelines (new v2 output)
        if build_timelines:
            split_metadata = [metadata[i] for i in indices]
            timelines = build_host_timelines_from_sessions(
                sessions[indices],
                labels[indices],
                masks[indices],
                split_metadata,
                config=timeline_config,
            )
            tl_path = os.path.join(out_dir, f"{split_name}_host_timelines.pkl")
            save_host_timelines(timelines, tl_path)
            outputs[f"{split_name}_timelines"] = tl_path

    # 5. Save comprehensive split metadata
    meta_path = os.path.join(out_dir, "split_metadata.json")
    split_summary = split_result.summary()
    split_summary["source_specs"] = source_specs
    split_summary["build_timelines"] = build_timelines

    # Add host diversity stats per split
    host_stats = {}
    for split_name, indices in [
        ("train", split_result.train_indices),
        ("val", split_result.val_indices),
        ("test", split_result.test_indices),
    ]:
        split_hosts = {metadata[i].host_id for i in indices}
        host_stats[split_name] = {
            "unique_hosts": len(split_hosts),
            "sessions": len(indices),
        }
        if build_timelines:
            tl_key = f"{split_name}_timelines"
            if tl_key in outputs:
                tls = load_host_timelines(outputs[tl_key])
                host_stats[split_name]["timeline_summary"] = _get_timeline_summary(tls)
    split_summary["host_stats"] = host_stats

    with open(meta_path, "w") as f:
        json.dump(split_summary, f, indent=2, default=str)
    outputs["metadata"] = meta_path

    print(f"\n[SUCCESS] v2 dataset saved to {out_dir}/")
    for key, path in sorted(outputs.items()):
        print(f"  {key}: {path}")
    return outputs


# ──────────────────────────────────────────────────────────────────────
# Rebalancing
# ──────────────────────────────────────────────────────────────────────

def _rebalance(
    sessions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    metadata: List[SessionMetadata],
    target_c2_ratio: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SessionMetadata]]:
    """Rebalance C2/benign ratio by downsampling the majority class."""
    rng = np.random.default_rng(random_seed)

    c2_idx = np.where(labels == LABEL_C2)[0]
    benign_idx = np.where(labels == LABEL_BENIGN)[0]
    n_c2 = len(c2_idx)

    target_total = int(n_c2 / target_c2_ratio)
    target_benign = target_total - n_c2

    if target_benign <= 0 or target_benign > len(benign_idx):
        print(f"[WARN] Cannot achieve C2 ratio {target_c2_ratio:.2f} — using natural distribution")
        return sessions, labels, masks, metadata

    selected_benign = rng.choice(benign_idx, size=target_benign, replace=False)
    all_idx = np.concatenate([c2_idx, selected_benign])
    shuffle = rng.permutation(len(all_idx))
    all_idx = all_idx[shuffle]

    new_meta = [metadata[i] for i in all_idx]
    return sessions[all_idx], labels[all_idx], masks[all_idx], new_meta


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CyberShield v2 dataset builder — sessions + host timelines"
    )
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="Source specs: <npz_path>:<source_name>:<label_type> (repeatable)",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--split_strategy", default="family_separated",
        choices=["random_stratified", "family_separated", "source_separated",
                 "time_separated", "zero_shot"],
    )
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--test_split", type=float, default=TEST_SPLIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c2_ratio", type=float, default=None)
    parser.add_argument(
        "--held_out_families", default=None,
        help="Comma-separated C2 families to hold out for test",
    )
    parser.add_argument(
        "--held_out_sources", default=None,
        help="Comma-separated dataset sources to hold out for test",
    )
    parser.add_argument(
        "--no_timelines", action="store_true",
        help="Skip host timeline construction (session-only output)",
    )
    parser.add_argument(
        "--min_sessions_baseline", type=int, default=10,
        help="Min sessions per host before computing baselines",
    )
    args = parser.parse_args()

    tl_config = HostTimelineConfig(
        min_sessions_for_baseline=args.min_sessions_baseline,
    )

    build_v2_dataset(
        source_specs=args.sources,
        out_dir=args.out_dir,
        split_strategy=args.split_strategy,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed,
        c2_ratio=args.c2_ratio,
        held_out_families=args.held_out_families,
        held_out_sources=args.held_out_sources,
        build_timelines=not args.no_timelines,
        timeline_config=tl_config,
    )
