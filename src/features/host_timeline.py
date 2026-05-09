"""Host Timeline Infrastructure for CyberShield v2.

Builds and manages per-host behavioral histories from session data.
This is the core new abstraction in v2 — the system shifts from
isolated session classification to host-centric behavioral analysis.

Pipeline position:
    session_builder.py → THIS FILE → behavioral_features.py / host_profiler.py

Key concepts:
    HostTimeline:     Complete behavioral history for one host IP.
    SessionSummary:   Lightweight summary of one session (not the full tensor).
    PartnerHistory:   Relationship tracking between a host and one destination.
    RollingStatistics: Exponentially-decayed running stats for anomaly detection.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.feature_config_v2 import (
    FEATURE_NAMES,
    FEATURE_DIM,
    FEATURE_INDEX,
    HostTimelineConfig,
    SessionMetadata,
    DatasetSource,
    INACTIVITY_TIMEOUT,
)


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SessionSummary:
    """Lightweight summary of a single session for timeline tracking."""

    session_id: str                     # Unique identifier
    host_id: str                        # Source IP
    dest_id: str                        # Destination IP
    protocol: str                       # tcp/udp/icmp
    start_ts: float                     # First flow timestamp
    end_ts: float                       # Last flow timestamp
    duration: float                     # end_ts - start_ts
    num_flows: int                      # Number of flows in session
    label: int = 0                      # 0=benign, 1=C2

    # Aggregate statistics
    total_orig_bytes: float = 0.0
    total_resp_bytes: float = 0.0
    total_orig_pkts: float = 0.0
    total_resp_pkts: float = 0.0
    mean_iat: float = 0.0
    std_iat: float = 0.0
    cv_iat: float = 0.0                # Coefficient of variation
    mean_bytes_per_pkt: float = 0.0
    mean_packet_ratio: float = 0.0
    mean_byte_ratio: float = 0.0

    # Source metadata
    metadata: Optional[SessionMetadata] = None

    @property
    def byte_ratio_agg(self) -> float:
        """Originator / responder byte ratio for the full session."""
        if self.total_resp_bytes < 1e-9:
            return self.total_orig_bytes if self.total_orig_bytes > 0 else 0.0
        return self.total_orig_bytes / self.total_resp_bytes

    @classmethod
    def from_tensor(
        cls,
        session: np.ndarray,
        mask: np.ndarray,
        label: int,
        session_id: str,
        host_id: str,
        metadata: Optional[SessionMetadata] = None,
    ) -> "SessionSummary":
        """Construct a SessionSummary from a (SESSION_LEN, FEATURE_DIM) tensor.

        Uses FEATURE_INDEX for all column lookups — never hardcoded ints.
        This is the single authoritative tensor → summary conversion.

        Parameters
        ----------
        session : (SESSION_LEN, FEATURE_DIM) float32 tensor.
        mask : (SESSION_LEN,) bool — True for real flows, False for padding.
        label : 0 or 1.
        session_id : Unique session identifier.
        host_id : Host identifier (real IP or synthetic key).
        metadata : Optional provenance metadata.
        """
        _fi = FEATURE_INDEX  # shorthand
        real = session[mask] if mask.any() else session[:1]
        n_flows = int(mask.sum())

        # Aggregates via named indices
        orig_bytes_sum = float(real[:, _fi["orig_bytes"]].sum())
        resp_bytes_sum = float(real[:, _fi["resp_bytes"]].sum())
        orig_pkts_sum = float(real[:, _fi["orig_pkts"]].sum())
        resp_pkts_sum = float(real[:, _fi["resp_pkts"]].sum())
        mean_bpp = float(real[:, _fi["bytes_per_pkt"]].mean())
        mean_pkt_ratio = float(real[:, _fi["packet_ratio"]].mean())
        mean_byte_ratio = float(real[:, _fi["byte_ratio"]].mean())
        duration = float(real[:, _fi["duration"]].sum())

        iats = real[:, _fi["iat"]]
        mean_iat = float(iats.mean()) if len(iats) > 0 else 0.0
        std_iat = float(iats.std()) if len(iats) > 1 else 0.0
        cv_iat = std_iat / mean_iat if mean_iat > 1e-9 else 0.0

        # Destination key from dst_port median
        dst_port = int(np.median(real[:, _fi["dst_port"]])) if len(real) > 0 else 0
        dest_id = f"dest_port_{dst_port}"

        # Timestamps: use metadata if available, otherwise synthetic ordering
        ts = metadata.capture_start_ts if (metadata and metadata.capture_start_ts > 0) else 0.0
        start_ts = ts if ts > 0 else float(hash(session_id) % 1_000_000)
        end_ts = start_ts + duration

        return cls(
            session_id=session_id,
            host_id=host_id,
            dest_id=dest_id,
            protocol="tcp",
            start_ts=start_ts,
            end_ts=end_ts,
            duration=duration,
            num_flows=n_flows,
            label=label,
            total_orig_bytes=orig_bytes_sum,
            total_resp_bytes=resp_bytes_sum,
            total_orig_pkts=orig_pkts_sum,
            total_resp_pkts=resp_pkts_sum,
            mean_iat=mean_iat,
            std_iat=std_iat,
            cv_iat=cv_iat,
            mean_bytes_per_pkt=mean_bpp,
            mean_packet_ratio=mean_pkt_ratio,
            mean_byte_ratio=mean_byte_ratio,
            metadata=metadata,
        )


@dataclass
class PartnerHistory:
    """Tracks the relationship between a host and one destination."""

    dest_id: str
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    total_sessions: int = 0
    total_bytes_sent: float = 0.0
    total_bytes_recv: float = 0.0
    session_iats: List[float] = field(default_factory=list)  # Gaps between sessions

    @property
    def relationship_duration(self) -> float:
        """Total wall-clock duration of the relationship."""
        return max(0.0, self.last_seen_ts - self.first_seen_ts)

    @property
    def mean_session_iat(self) -> float:
        """Average inter-session arrival time."""
        if not self.session_iats:
            return 0.0
        return float(np.mean(self.session_iats))

    @property
    def session_regularity(self) -> float:
        """CV of inter-session times (lower = more regular)."""
        if len(self.session_iats) < 2:
            return 0.0
        mean = np.mean(self.session_iats)
        if mean < 1e-9:
            return 0.0
        return float(np.std(self.session_iats) / mean)


@dataclass
class RollingStatistics:
    """Exponentially-decayed running statistics for one metric.

    Uses Welford's online algorithm variant with exponential decay.
    """

    name: str
    decay: float = 0.95
    _count: int = 0
    _mean: float = 0.0
    _m2: float = 0.0       # Running sum of squared deviations
    _weight_sum: float = 0.0

    def update(self, value: float) -> None:
        """Add a new observation with exponential decay."""
        self._count += 1
        weight = 1.0

        # Apply decay to existing statistics
        self._weight_sum = self._weight_sum * self.decay + weight
        old_mean = self._mean
        self._mean = old_mean + (weight / self._weight_sum) * (value - old_mean)
        self._m2 = self._m2 * self.decay + weight * (value - old_mean) * (value - self._mean)

    @property
    def mean(self) -> float:
        return self._mean if self._count > 0 else 0.0

    @property
    def variance(self) -> float:
        if self._weight_sum < 1e-9:
            return 0.0
        return self._m2 / self._weight_sum

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    @property
    def count(self) -> int:
        return self._count

    def zscore(self, value: float) -> float:
        """Z-score of a new value against the running distribution."""
        if self._count < 2 or self.std < 1e-9:
            return 0.0
        return (value - self.mean) / self.std


@dataclass
class HostTimeline:
    """Complete behavioral history for one host IP.

    This is the central data structure in v2. Every host in the capture
    gets a timeline that tracks:
    - All sessions over time
    - Communication partner relationships
    - Behavioral rolling statistics
    - Activity rhythm (hour-of-day patterns)
    """

    host_id: str
    config: HostTimelineConfig = field(default_factory=HostTimelineConfig)

    # Session history (chronologically ordered)
    sessions: List[SessionSummary] = field(default_factory=list)

    # Partner tracking
    partners: Dict[str, PartnerHistory] = field(default_factory=dict)

    # Rolling behavioral statistics
    rolling_stats: Dict[str, RollingStatistics] = field(default_factory=dict)

    # Activity rhythm: count of sessions starting in each hour (0–23)
    hourly_activity: np.ndarray = field(
        default_factory=lambda: np.zeros(24, dtype=np.float64)
    )

    # Destination evolution tracking
    _known_destinations: set = field(default_factory=set)
    new_dest_timestamps: List[Tuple[float, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize rolling statistics trackers."""
        decay = self.config.baseline_decay_factor
        stat_names = [
            "session_byte_volume",
            "session_duration",
            "session_num_flows",
            "session_mean_iat",
            "session_cv_iat",
            "session_packet_ratio",
            "inter_session_gap",
        ]
        for name in stat_names:
            if name not in self.rolling_stats:
                self.rolling_stats[name] = RollingStatistics(name=name, decay=decay)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def has_baseline(self) -> bool:
        """Whether enough history exists to compute meaningful baselines."""
        return self.num_sessions >= self.config.min_sessions_for_baseline

    @property
    def unique_destinations(self) -> int:
        return len(self._known_destinations)

    @property
    def first_seen(self) -> float:
        if not self.sessions:
            return 0.0
        return self.sessions[0].start_ts

    @property
    def last_seen(self) -> float:
        if not self.sessions:
            return 0.0
        return self.sessions[-1].end_ts

    @property
    def active_duration(self) -> float:
        """Total wall-clock time this host has been observed."""
        return max(0.0, self.last_seen - self.first_seen)

    def add_session(self, summary: SessionSummary) -> Dict[str, float]:
        """Add a session to the timeline and return anomaly context.

        Returns a dict of deviation scores for the new session relative
        to this host's rolling history. These are used downstream by
        host_anomaly.py and host_aware_classifier.py.
        """
        deviations: Dict[str, float] = {}

        # --- Compute deviations BEFORE updating rolling stats ---
        if self.has_baseline:
            total_bytes = summary.total_orig_bytes + summary.total_resp_bytes
            deviations["byte_volume_zscore"] = self.rolling_stats[
                "session_byte_volume"
            ].zscore(total_bytes)
            deviations["duration_zscore"] = self.rolling_stats[
                "session_duration"
            ].zscore(summary.duration)
            deviations["num_flows_zscore"] = self.rolling_stats[
                "session_num_flows"
            ].zscore(summary.num_flows)
            deviations["mean_iat_zscore"] = self.rolling_stats[
                "session_mean_iat"
            ].zscore(summary.mean_iat)
            deviations["cv_iat_zscore"] = self.rolling_stats[
                "session_cv_iat"
            ].zscore(summary.cv_iat)

            # Inter-session gap
            if self.sessions:
                gap = summary.start_ts - self.sessions[-1].end_ts
                deviations["inter_session_gap_zscore"] = self.rolling_stats[
                    "inter_session_gap"
                ].zscore(gap)

            # New destination check
            deviations["is_new_destination"] = float(
                summary.dest_id not in self._known_destinations
            )

            # Idle-hour activity
            hour = _epoch_to_hour(summary.start_ts)
            deviations["is_idle_hour"] = float(
                hour in self.config.idle_hours
            )

        # --- Update rolling statistics ---
        total_bytes = summary.total_orig_bytes + summary.total_resp_bytes
        self.rolling_stats["session_byte_volume"].update(total_bytes)
        self.rolling_stats["session_duration"].update(summary.duration)
        self.rolling_stats["session_num_flows"].update(summary.num_flows)
        self.rolling_stats["session_mean_iat"].update(summary.mean_iat)
        self.rolling_stats["session_cv_iat"].update(summary.cv_iat)
        self.rolling_stats["session_packet_ratio"].update(summary.mean_packet_ratio)

        if self.sessions:
            gap = summary.start_ts - self.sessions[-1].end_ts
            self.rolling_stats["inter_session_gap"].update(gap)

        # --- Update partner tracking ---
        dest = summary.dest_id
        if dest not in self.partners:
            self.partners[dest] = PartnerHistory(
                dest_id=dest,
                first_seen_ts=summary.start_ts,
            )
        partner = self.partners[dest]

        # Record inter-session time for this partner
        if partner.total_sessions > 0:
            session_gap = summary.start_ts - partner.last_seen_ts
            partner.session_iats.append(session_gap)

        partner.last_seen_ts = summary.end_ts
        partner.total_sessions += 1
        partner.total_bytes_sent += summary.total_orig_bytes
        partner.total_bytes_recv += summary.total_resp_bytes

        # --- Update destination tracking ---
        if dest not in self._known_destinations:
            self._known_destinations.add(dest)
            self.new_dest_timestamps.append((summary.start_ts, dest))

        # --- Update hourly activity ---
        hour = _epoch_to_hour(summary.start_ts)
        if 0 <= hour < 24:
            self.hourly_activity[hour] += 1

        # --- Append to history (with cap) ---
        self.sessions.append(summary)
        if len(self.sessions) > self.config.max_history_sessions:
            self.sessions = self.sessions[-self.config.max_history_sessions:]

        return deviations

    def get_activity_entropy(self) -> float:
        """Shannon entropy of hourly activity distribution."""
        total = self.hourly_activity.sum()
        if total < 1:
            return 0.0
        probs = self.hourly_activity / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def get_destination_rate(self, window_seconds: float = 3600.0) -> float:
        """Rate of new destination appearances (per window)."""
        if not self.new_dest_timestamps or self.active_duration < 1:
            return 0.0
        n_windows = max(1.0, self.active_duration / window_seconds)
        return len(self.new_dest_timestamps) / n_windows

    def get_host_feature_vector(self) -> Dict[str, float]:
        """Extract host-level feature vector for downstream classifiers."""
        features = {}

        # Communication cadence
        inter_session_stat = self.rolling_stats.get("inter_session_gap")
        if inter_session_stat and inter_session_stat.count > 1:
            features["comm_cadence_cv"] = (
                inter_session_stat.std / inter_session_stat.mean
                if inter_session_stat.mean > 1e-9 else 0.0
            )
        else:
            features["comm_cadence_cv"] = 0.0

        # Session frequency
        if self.active_duration > 0:
            features["session_freq_avg"] = self.num_sessions / (
                self.active_duration / 3600.0
            )
        else:
            features["session_freq_avg"] = 0.0

        # Activity entropy
        features["active_hour_entropy"] = self.get_activity_entropy()

        # Destination diversity
        features["dest_diversity_rolling"] = float(self.unique_destinations)
        features["new_dest_rate"] = self.get_destination_rate()

        # Rolling Z-scores (most recent vs historical baseline)
        for stat_name in [
            "session_byte_volume",
            "session_mean_iat",
            "session_duration",
        ]:
            stat = self.rolling_stats.get(stat_name)
            short_name = stat_name.replace("session_", "")
            if stat and stat.count > 1:
                features[f"{short_name}_baseline_mean"] = stat.mean
                features[f"{short_name}_baseline_std"] = stat.std
            else:
                features[f"{short_name}_baseline_mean"] = 0.0
                features[f"{short_name}_baseline_std"] = 0.0

        return features

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/inspection."""
        return {
            "host_id": self.host_id,
            "num_sessions": self.num_sessions,
            "unique_destinations": self.unique_destinations,
            "active_duration": self.active_duration,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "activity_entropy": self.get_activity_entropy(),
            "top_partners": sorted(
                [
                    {
                        "dest": p.dest_id,
                        "sessions": p.total_sessions,
                        "duration": p.relationship_duration,
                        "regularity": p.session_regularity,
                    }
                    for p in self.partners.values()
                ],
                key=lambda x: x["sessions"],
                reverse=True,
            )[:10],
        }


# ──────────────────────────────────────────────────────────────────────
# Timeline Construction
# ──────────────────────────────────────────────────────────────────────

class HostTimelineBuilder:
    """Builds HostTimeline objects from flow DataFrames or session data.

    Typical usage:
        builder = HostTimelineBuilder()
        builder.ingest_flow_dataframe(df, source=DatasetSource.UWF_ZEEKDATA24)
        timelines = builder.build()
    """

    def __init__(self, config: Optional[HostTimelineConfig] = None) -> None:
        self.config = config or HostTimelineConfig()
        self._timelines: Dict[str, HostTimeline] = {}
        self._session_counter: int = 0

    @property
    def timelines(self) -> Dict[str, HostTimeline]:
        return self._timelines

    def get_timeline(self, host_id: str) -> Optional[HostTimeline]:
        return self._timelines.get(host_id)

    def ingest_flow_dataframe(
        self,
        df: pd.DataFrame,
        source: DatasetSource = DatasetSource.UNKNOWN,
        capture_id: str = "",
        label_col: Optional[str] = "label",
        inactivity_timeout: float = INACTIVITY_TIMEOUT,
        min_flows: int = 1,
    ) -> int:
        """Ingest a flow-level DataFrame and build session summaries into host timelines.

        Parameters
        ----------
        df : Flow-level DataFrame with columns: ts, src_ip, dst_ip, proto,
             and feature columns from FEATURE_NAMES.
        source : Dataset origin for metadata tracking.
        capture_id : Unique identifier for this capture.
        label_col : Column name for labels (None for unlabeled data).
        inactivity_timeout : Seconds of inactivity before new session.
        min_flows : Minimum flows to form a session.

        Returns
        -------
        Number of sessions ingested.
        """
        required = {"ts", "src_ip", "dst_ip", "proto"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        df = df.copy()
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0.0)
        df = df.sort_values(["src_ip", "dst_ip", "proto", "ts"]).reset_index(drop=True)

        sessions_ingested = 0
        groups = df.groupby(["src_ip", "dst_ip", "proto"], sort=False)

        for (src_ip, dst_ip, proto), group in tqdm(
            groups, desc="Building host timelines", leave=False, unit="group"
        ):
            group = group.sort_values("ts").reset_index(drop=True)
            session_dfs = _segment_by_inactivity(group, inactivity_timeout)

            for sess_df in session_dfs:
                if len(sess_df) < min_flows:
                    continue

                summary = _build_session_summary(
                    sess_df,
                    session_id=f"s{self._session_counter:08d}",
                    host_id=str(src_ip),
                    dest_id=str(dst_ip),
                    protocol=str(proto),
                    label_col=label_col,
                    source=source,
                    capture_id=capture_id,
                )
                self._session_counter += 1

                # Add to the source host's timeline
                host_id = str(src_ip)
                if host_id not in self._timelines:
                    self._timelines[host_id] = HostTimeline(
                        host_id=host_id, config=self.config
                    )
                self._timelines[host_id].add_session(summary)
                sessions_ingested += 1

        return sessions_ingested

    def build(self) -> Dict[str, HostTimeline]:
        """Return all constructed timelines."""
        return self._timelines

    def summary(self) -> Dict[str, Any]:
        """High-level summary of all host timelines."""
        if not self._timelines:
            return {"hosts": 0, "total_sessions": 0}

        session_counts = [t.num_sessions for t in self._timelines.values()]
        return {
            "hosts": len(self._timelines),
            "total_sessions": sum(session_counts),
            "sessions_per_host_mean": float(np.mean(session_counts)),
            "sessions_per_host_median": float(np.median(session_counts)),
            "sessions_per_host_max": int(np.max(session_counts)),
            "hosts_with_baseline": sum(
                1 for t in self._timelines.values() if t.has_baseline
            ),
            "total_unique_destinations": sum(
                t.unique_destinations for t in self._timelines.values()
            ),
        }


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _epoch_to_hour(ts: float) -> int:
    """Convert epoch timestamp to hour-of-day (0–23)."""
    import datetime
    try:
        dt = datetime.datetime.utcfromtimestamp(ts)
        return dt.hour
    except (OSError, OverflowError, ValueError):
        return 0


def _segment_by_inactivity(
    group: pd.DataFrame, timeout: float
) -> List[pd.DataFrame]:
    """Split a sorted flow group into sessions by inactivity gap."""
    if len(group) == 0:
        return []
    sessions: List[pd.DataFrame] = []
    current = [group.iloc[0]]
    for i in range(1, len(group)):
        gap = float(group.iloc[i]["ts"]) - float(group.iloc[i - 1]["ts"])
        if gap > timeout:
            sessions.append(pd.DataFrame(current).reset_index(drop=True))
            current = [group.iloc[i]]
        else:
            current.append(group.iloc[i])
    if current:
        sessions.append(pd.DataFrame(current).reset_index(drop=True))
    return sessions


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safe float conversion for mixed-type DataFrame values."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _build_session_summary(
    sess_df: pd.DataFrame,
    session_id: str,
    host_id: str,
    dest_id: str,
    protocol: str,
    label_col: Optional[str],
    source: DatasetSource,
    capture_id: str,
) -> SessionSummary:
    """Build a SessionSummary from a session DataFrame."""
    ts_vals = pd.to_numeric(sess_df["ts"], errors="coerce").fillna(0.0)
    start_ts = float(ts_vals.iloc[0])
    end_ts = float(ts_vals.iloc[-1])

    # Aggregate flow features
    orig_bytes = pd.to_numeric(sess_df.get("orig_bytes", 0), errors="coerce").fillna(0.0)
    resp_bytes = pd.to_numeric(sess_df.get("resp_bytes", 0), errors="coerce").fillna(0.0)
    orig_pkts = pd.to_numeric(sess_df.get("orig_pkts", 0), errors="coerce").fillna(0.0)
    resp_pkts = pd.to_numeric(sess_df.get("resp_pkts", 0), errors="coerce").fillna(0.0)

    # IAT statistics
    iats = pd.to_numeric(sess_df.get("iat", 0), errors="coerce").fillna(0.0).values
    mean_iat = float(np.mean(iats)) if len(iats) > 0 else 0.0
    std_iat = float(np.std(iats)) if len(iats) > 1 else 0.0
    cv_iat = std_iat / mean_iat if mean_iat > 1e-9 else 0.0

    # Packet/byte ratios
    bpp_vals = pd.to_numeric(sess_df.get("bytes_per_pkt", 0), errors="coerce").fillna(0.0)
    pkt_ratio_vals = pd.to_numeric(sess_df.get("packet_ratio", 0), errors="coerce").fillna(0.0)
    byte_ratio_vals = pd.to_numeric(sess_df.get("byte_ratio", 0), errors="coerce").fillna(0.0)

    # Label
    label = 0
    if label_col and label_col in sess_df.columns:
        label = int(sess_df[label_col].mode().iloc[0])

    # Metadata
    metadata = SessionMetadata(
        source=source,
        capture_id=capture_id,
        capture_start_ts=start_ts,
        host_id=host_id,
    )

    return SessionSummary(
        session_id=session_id,
        host_id=host_id,
        dest_id=dest_id,
        protocol=protocol,
        start_ts=start_ts,
        end_ts=end_ts,
        duration=max(0.0, end_ts - start_ts),
        num_flows=len(sess_df),
        label=label,
        total_orig_bytes=float(orig_bytes.sum()),
        total_resp_bytes=float(resp_bytes.sum()),
        total_orig_pkts=float(orig_pkts.sum()),
        total_resp_pkts=float(resp_pkts.sum()),
        mean_iat=mean_iat,
        std_iat=std_iat,
        cv_iat=cv_iat,
        mean_bytes_per_pkt=float(bpp_vals.mean()),
        mean_packet_ratio=float(pkt_ratio_vals.mean()),
        mean_byte_ratio=float(byte_ratio_vals.mean()),
        metadata=metadata,
    )
