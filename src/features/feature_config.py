"""
feature_config.py
=================
SINGLE SOURCE OF TRUTH for all feature definitions.
Import from here everywhere. Never hardcode feature lists elsewhere.

Dataset strategy (DO NOT CHANGE without updating this comment):
  C2 training source : CTU-13 scenarios 1-9  (real botnet C2 behavioral patterns)
  Benign source      : CICIoT2023 benign-only + CICIDS2018 benign-only
  Cross-dataset test : CTU-13 scenarios 10-13 — NEVER seen during training
  Real-world eval    : Stratosphere MCFP (Emotet 2020, Trickbot 2021, QakBot 2022)
  Adversarial data   : DReLAB / RL-generated

Why NOT training C2 on CICIoT2023:
  CICIoT2023 "botnet" = Mirai DDoS (high-volume, volumetric, noisy).
  Real APT C2 = low-and-slow HTTP callbacks, long sleep, structured timing.
  Training on Mirai teaches the wrong behavioral fingerprint entirely.

Why CTU-13 benign is NOT used for training:
  CTU-13 benign was captured in 2011 on a university network. Using it as
  the benign class would give the model a 2011 view of normal traffic and
  fail on any modern network. CICIoT2023 benign is from 2023 IoT devices
  and CICIDS2018 benign covers modern enterprise protocols.
"""

import ipaddress

# ════════════════════════════════════════════════════════════════════
# FEATURE NAMES  — 12 features derived from Zeek conn.log only
# Order matters: model input tensor columns follow this exact order.
# ════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "duration",       # flow duration in seconds
    "orig_bytes",     # bytes sent by originator
    "resp_bytes",     # bytes sent by responder
    "orig_pkts",      # packets sent by originator
    "resp_pkts",      # packets sent by responder
    "bytes_per_pkt",  # (orig_bytes + resp_bytes) / total_pkts — volume per pkt
    "packet_ratio",   # orig_pkts / resp_pkts — directionality asymmetry
    "byte_ratio",     # orig_bytes / resp_bytes — payload asymmetry
    "iat_mean",       # mean inter-arrival time across session (seconds)
    "iat_std",        # std of inter-arrival time across session
    "iat_cv",         # iat_std / iat_mean — beaconing regularity signal
    "is_outbound",    # 1 if orig IP is private (internal → external)
]

USE_DERIVATIVE_FEATURES = True

if USE_DERIVATIVE_FEATURES:
    # Keep FEATURE_DIM fixed at 12 by replacing two ratio features with temporal deltas.
    FEATURE_NAMES = [
        "duration",
        "orig_bytes",
        "resp_bytes",
        "orig_pkts",
        "resp_pkts",
        "bytes_per_pkt",
        "iat_delta",
        "byte_delta",
        "iat_mean",
        "iat_std",
        "iat_cv",
        "is_outbound",
    ]

FEATURE_DIM = len(FEATURE_NAMES)   # 12 — model input size, do not change
assert FEATURE_DIM == 12, f"Expected 12 features, got {FEATURE_DIM}"

# ════════════════════════════════════════════════════════════════════
# SESSION CONFIGURATION
# ════════════════════════════════════════════════════════════════════

SESSION_LEN         = 20    # flows per session (pad shorter, truncate longer)
INACTIVITY_TIMEOUT  = 300   # seconds of silence before starting a new session
                            # 60s is too short — splits 90s beacons in half
                            # 300s (5 min) keeps most C2 sessions intact

# ════════════════════════════════════════════════════════════════════
# DATASET ROLES  — never change these assignments mid-project
# ════════════════════════════════════════════════════════════════════

# C2 traffic source for training (real botnet behavioral patterns)
C2_TRAIN_SOURCE  = "ctu13_scenarios_1_9"

# Benign traffic source for training (modern, diverse)
BENIGN_TRAIN_SOURCE = "ciciot2023_benign"

# Validation split (from training data, stratified)
VAL_SPLIT = 0.15

# ════════════════════════════════════════════════════════════════════
# TRAINING DEFAULTS (IMBALANCE + OPERATING POINT)
# ════════════════════════════════════════════════════════════════════

FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 1.0
MAX_FPR_BUDGET = 0.005

# !! STRICTLY NEVER TRAIN OR VALIDATE ON THESE !!
CROSS_DATASET_TEST  = "ctu13_scenarios_10_13"
REALWORLD_TEST      = "stratosphere_mcfp"   # Emotet/Trickbot/QakBot

# ════════════════════════════════════════════════════════════════════
# LABEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════

LABEL_BENIGN  = 0
LABEL_C2      = 1
LABEL_NAMES   = {LABEL_BENIGN: "benign", LABEL_C2: "c2"}

# Non-C2 attack types that are relabeled as BENIGN (label=0) because
# they are not C2 and training on them as C2 would teach wrong patterns.
# DDoS, scanning, brute-force are volumetric/noisy — opposite of C2.
NON_C2_ATTACK_LABELS = {
    "ddos", "dos", "recon", "scanning", "bruteforce", "brute force",
    "web attack", "infiltration", "heartbleed", "bot-ddos", "mirai-ddos",
    "greip", "greeth", "udpplain", "syn", "udpflood",
}

# C2/botnet labels that map to LABEL_C2 = 1
C2_POSITIVE_LABELS = {
    "botnet", "c&c", "c2", "bot", "command and control",
    "neris", "rbot", "virut", "menti", "sogou", "murlo",
    "emotet", "trickbot", "qakbot",
}

# ════════════════════════════════════════════════════════════════════
# ZEEK FIELD MAPPING — columns extracted from conn.log
# ════════════════════════════════════════════════════════════════════

ZEEK_REQUIRED_FIELDS = [
    "ts",           # timestamp (epoch float)
    "id.orig_h",    # source IP
    "id.resp_h",    # destination IP
    "id.orig_p",    # source port
    "id.resp_p",    # destination port
    "proto",        # transport protocol (tcp/udp/icmp)
    "duration",     # flow duration
    "orig_bytes",   # bytes from originator
    "resp_bytes",   # bytes from responder
    "orig_pkts",    # packets from originator
    "resp_pkts",    # packets from responder
    "conn_state",   # connection state
    "history",      # TCP flag history string
]

# ════════════════════════════════════════════════════════════════════
# IP UTILITY
# ════════════════════════════════════════════════════════════════════

def is_private_ip(ip_str: str) -> int:
    """
    Return 1 if IP is private/internal, 0 otherwise.
    Uses Python's ipaddress module for correctness (handles edge cases
    like 127.x, 169.254.x, ::1 that prefix checks miss).
    Returns 0 for malformed or IPv6 addresses.
    """
    try:
        return 1 if ipaddress.ip_address(ip_str).is_private else 0
    except ValueError:
        return 0


def label_from_string(raw_label: str) -> int:
    """
    Map a raw dataset label string to binary C2 (1) or non-C2 (0).
    Called during dataset loading to enforce strict binary classification.
    """
    normalized = raw_label.strip().lower()
    if any(c in normalized for c in C2_POSITIVE_LABELS):
        return LABEL_C2
    return LABEL_BENIGN