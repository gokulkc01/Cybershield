"""Single source of truth for CyberShield feature and label metadata."""

from __future__ import annotations

import ipaddress

# Dataset strategy:
# - C2 training source: CTU-13 scenarios 1-9
# - Benign training source: CICIoT2023 benign-only + CICIDS2018 benign-only
# - Cross-dataset test: CTU-13 scenarios 10-13
# - Real-world evaluation: Stratosphere MCFP
# - Adversarial data: DReLAB / RL-generated

FEATURE_NAMES = [
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
assert FEATURE_DIM == 12, f"Expected 12 features, got {FEATURE_DIM}"

DERIVATIVE_FEATURE_NAMES = ["iat_delta", "byte_delta"]

FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
ORIG_BYTES_IDX = FEATURE_INDEX["orig_bytes"]
IAT_IDX = FEATURE_INDEX["iat"]

SESSION_LEN = 20
INACTIVITY_TIMEOUT = 300

C2_TRAIN_SOURCE = "ctu13_scenarios_1_9"
BENIGN_TRAIN_SOURCE = "ciciot2023_benign"
VAL_SPLIT = 0.15

FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 1.0
MAX_FPR_BUDGET = 0.005

CROSS_DATASET_TEST = "ctu13_scenarios_10_13"
REALWORLD_TEST = "stratosphere_mcfp"

LABEL_BENIGN = 0
LABEL_C2 = 1
LABEL_NAMES = {LABEL_BENIGN: "benign", LABEL_C2: "c2"}

NON_C2_ATTACK_LABELS = {
    "ddos",
    "dos",
    "recon",
    "scanning",
    "bruteforce",
    "brute force",
    "web attack",
    "infiltration",
    "heartbleed",
    "bot-ddos",
    "mirai-ddos",
    "greip",
    "greeth",
    "udpplain",
    "syn",
    "udpflood",
}

C2_POSITIVE_LABELS = {
    "botnet",
    "c&c",
    "c2",
    "bot",
    "command and control",
    "neris",
    "rbot",
    "virut",
    "menti",
    "sogou",
    "murlo",
    "emotet",
    "trickbot",
    "qakbot",
}

ZEEK_REQUIRED_FIELDS = [
    "ts",
    "id.orig_h",
    "id.resp_h",
    "id.orig_p",
    "id.resp_p",
    "proto",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "conn_state",
    "history",
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
