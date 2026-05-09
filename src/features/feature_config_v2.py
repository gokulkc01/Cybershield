"""CyberShield v2 — Extended feature and label configuration.

Backward-compatible with v1 feature_config.py.  Adds:
  - v2 behavioral / entropy / persistence / TLS feature groups
  - Modern C2 family labels (Sliver, Havoc, Cobalt Strike malleable, etc.)
  - Source-metadata tracking constants
  - Relaxed FPR budget (0.015–0.03 range vs v1's 0.005)
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Set, Tuple

# ─────────────────────────────────────────────────────────────────────
# v1 backward compatibility — re-export everything the old config had
# ─────────────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "bytes_per_pkt",
    "packet_ratio",
    "byte_ratio",
    "is_outbound",
    "duration",
    "src_port",
    "dst_port",
    "iat",
]

FEATURE_DIM = len(FEATURE_NAMES)
assert FEATURE_DIM == 12, f"Expected 12 base features, got {FEATURE_DIM}"

DERIVATIVE_FEATURE_NAMES = ["iat_delta", "byte_delta"]

FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
ORIG_BYTES_IDX = FEATURE_INDEX["orig_bytes"]
IAT_IDX = FEATURE_INDEX["iat"]

SESSION_LEN = 20
INACTIVITY_TIMEOUT = 300

FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 1.0

LABEL_BENIGN = 0
LABEL_C2 = 1
LABEL_NAMES = {LABEL_BENIGN: "benign", LABEL_C2: "c2"}

ZEEK_REQUIRED_FIELDS = [
    "ts", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p",
    "proto", "duration", "orig_bytes", "resp_bytes",
    "orig_pkts", "resp_pkts", "conn_state", "history",
]


def is_private_ip(ip_str: str) -> int:
    try:
        return 1 if ipaddress.ip_address(ip_str).is_private else 0
    except ValueError:
        return 0


def label_from_string(raw_label: str) -> int:
    normalized = raw_label.strip().lower()
    if any(label in normalized for label in C2_POSITIVE_LABELS):
        return LABEL_C2
    return LABEL_BENIGN


# ─────────────────────────────────────────────────────────────────────
# v2 ADDITIONS
# ─────────────────────────────────────────────────────────────────────

# --- FPR Budget (relaxed for v2) ---
MAX_FPR_BUDGET_V1 = 0.005      # Original v1 target
MAX_FPR_BUDGET_LOW = 0.015     # v2 lower bound
MAX_FPR_BUDGET_HIGH = 0.03     # v2 upper bound
MAX_FPR_BUDGET = MAX_FPR_BUDGET_LOW  # Default operating point

# --- Split configuration ---
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# --- C2 Family Taxonomy ---
class C2Family(str, Enum):
    """Known C2 framework families for source-aware splitting."""
    # Legacy (v1)
    NERIS = "neris"
    RBOT = "rbot"
    VIRUT = "virut"
    MENTI = "menti"
    SOGOU = "sogou"
    MURLO = "murlo"
    EMOTET = "emotet"
    TRICKBOT = "trickbot"
    QAKBOT = "qakbot"
    # Modern frameworks (v2)
    SLIVER = "sliver"
    HAVOC = "havoc"
    COBALT_STRIKE = "cobalt_strike"
    MYTHIC = "mythic"
    BRUTE_RATEL = "brute_ratel"
    METASPLOIT = "metasploit"
    COVENANT = "covenant"
    POSHC2 = "poshc2"
    # Generic
    UNKNOWN_C2 = "unknown_c2"


# --- Expanded label sets ---
NON_C2_ATTACK_LABELS: Set[str] = {
    "ddos", "dos", "recon", "scanning", "bruteforce", "brute force",
    "web attack", "infiltration", "heartbleed", "bot-ddos", "mirai-ddos",
    "greip", "greeth", "udpplain", "syn", "udpflood",
}

C2_POSITIVE_LABELS: Set[str] = {
    # v1 labels
    "botnet", "c&c", "c2", "bot", "command and control",
    "neris", "rbot", "virut", "menti", "sogou", "murlo",
    "emotet", "trickbot", "qakbot",
    # v2 additions — modern C2 frameworks
    "sliver", "havoc", "cobalt strike", "cobaltstrike",
    "mythic", "brute ratel", "bruteratel",
    "metasploit", "meterpreter", "covenant", "poshc2",
    "beacon", "implant", "c2-beacon", "c2_beacon",
    "ta0011",  # MITRE ATT&CK tactic ID for C2
}


# --- Source / Dataset metadata ---
class DatasetSource(str, Enum):
    """Dataset origin for source-aware evaluation splits."""
    CTU13 = "ctu13"
    UWF_ZEEKDATA24 = "uwf_zeekdata24"
    MCFP_STRATOSPHERE = "mcfp_stratosphere"
    CICIDS2018 = "cicids2018"
    CICIOT2023 = "ciciot2023"
    CUSTOM_LAB = "custom_lab"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SessionMetadata:
    """Per-session provenance tracking for split integrity."""
    source: DatasetSource
    c2_family: str = ""            # Empty for benign
    capture_id: str = ""           # Unique capture/scenario identifier
    capture_start_ts: float = 0.0  # Start of capture window (epoch)
    host_id: str = ""              # Source IP (for host-timeline linkage)

    def split_key_family(self) -> str:
        """Key for family-separated splits."""
        return self.c2_family if self.c2_family else f"benign_{self.source.value}"

    def split_key_source(self) -> str:
        """Key for source-separated splits."""
        return self.source.value


# --- v2 Feature Groups ---

BEHAVIORAL_FEATURE_NAMES: List[str] = [
    # Temporal / stability
    "iat_cv",               # Coefficient of variation of IATs
    "iat_jitter",           # Normalized IAT std
    "burstiness_index",     # Fano factor
    "iat_skew",             # Skewness of IAT distribution
    "iat_kurtosis",         # Kurtosis of IAT distribution
    "session_regularity",   # Evenness of session spacing
    "activity_regularity",  # Regularity of activity windows
]

ENTROPY_FEATURE_NAMES: List[str] = [
    "timing_entropy",       # Shannon entropy of IAT bins
    "size_entropy",         # Entropy of packet size distribution
    "behavioral_entropy",   # Cross-session consistency
    "destination_entropy",  # Diversity of communication targets
]

PERSISTENCE_FEATURE_NAMES: List[str] = [
    "reconnect_time",       # Time to reconnect after session end
    "relationship_duration",# Total time communicating with this dest
    "session_consistency",  # Similarity of consecutive sessions to same dest
    "idle_period_activity", # Communication during off-hours
    "communication_longevity",  # How long the relationship has existed
]

TLS_FEATURE_NAMES: List[str] = [
    "has_tls",              # Whether TLS data is available
    "ja3_hash_rarity",      # Rarity score for JA3 fingerprint
    "tls_version_score",    # Encoding of TLS version
    "cert_chain_length",    # Certificate chain depth
    "sni_present",          # Whether SNI is set
    "self_signed_cert",     # Self-signed certificate indicator
]

# Combined v2 feature set (session-level enrichments)
V2_SESSION_FEATURES = (
    BEHAVIORAL_FEATURE_NAMES
    + ENTROPY_FEATURE_NAMES
    + PERSISTENCE_FEATURE_NAMES
    + TLS_FEATURE_NAMES
)

V2_FEATURE_DIM = FEATURE_DIM + len(V2_SESSION_FEATURES)

# --- Host-Timeline Feature Groups (computed per host, not per session) ---

HOST_TIMELINE_FEATURE_NAMES: List[str] = [
    "comm_cadence_cv",          # CV of communication cadence
    "session_freq_avg",         # Rolling average session frequency
    "active_hour_entropy",      # Entropy of active hour distribution
    "dest_diversity_rolling",   # Rolling unique destination count
    "new_dest_rate",            # Rate of new destination appearance
    "byte_volume_zscore",       # Z-score of current vs historical volume
    "iat_mean_zscore",          # Z-score of IAT mean vs host baseline
    "session_len_zscore",       # Z-score of session length vs baseline
]


# --- Host Timeline Configuration ---

@dataclass
class HostTimelineConfig:
    """Configuration for host timeline construction."""
    min_sessions_for_baseline: int = 10       # Minimum sessions before scoring
    rolling_window_seconds: float = 3600.0    # 1-hour rolling windows
    baseline_decay_factor: float = 0.95       # Exponential decay for rolling stats
    max_history_sessions: int = 1000          # Cap per-host session history
    idle_hours: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # 00:00–05:59 considered idle
    anomaly_zscore_threshold: float = 3.0     # Z-score threshold for anomalies


# --- Detector weights (Phase 3 defaults) ---

@dataclass
class FusionConfig:
    """Default weights for multi-signal fusion (Phase 3)."""
    host_anomaly_weight: float = 0.30
    behavioral_feature_weight: float = 0.25
    persistence_weight: float = 0.15
    transformer_ml_weight: float = 0.15
    tls_dns_weight: float = 0.10
    beaconing_weight: float = 0.05  # Intentionally low — secondary signal

    min_signals_for_high_risk: int = 3  # Require multiple aligned signals
    risk_threshold_high: float = 0.70
    risk_threshold_medium: float = 0.40
