"""Extended feature schema for Phase 1 behavioral C2 experiments.

Schema ``extended_v1`` intentionally keeps all currently planned signals:

- 10 base session-flow behavior features
- 12 TLS behavior features
- 10 DNS behavior features
- 13 temporal recurrence/burst features
"""

from __future__ import annotations

from src.features.dns_features import DNSFeatureExtractor
from src.features.feature_config_experiment import FEATURE_NAMES as BASE_FEATURE_NAMES
from src.features.feature_config_experiment import SESSION_LEN
from src.features.temporal_features import TemporalStatsExtractor
from src.features.tls_features import TLSFeatureExtractor


SCHEMA_VERSION_EXTENDED = "extended_v1"
FEATURE_SCHEMA_VERSION = SCHEMA_VERSION_EXTENDED

BASE_FEATURE_NAMES_EXTENDED = list(BASE_FEATURE_NAMES)
TLS_FEATURE_NAMES_EXTENDED = list(TLSFeatureExtractor.TLS_FEATURES)
DNS_FEATURE_NAMES_EXTENDED = list(DNSFeatureExtractor.DNS_FEATURES)
TEMPORAL_FEATURE_NAMES_EXTENDED = list(TemporalStatsExtractor.TEMPORAL_FEATURES)

FEATURE_NAMES_EXTENDED = (
    BASE_FEATURE_NAMES_EXTENDED
    + TLS_FEATURE_NAMES_EXTENDED
    + DNS_FEATURE_NAMES_EXTENDED
    + TEMPORAL_FEATURE_NAMES_EXTENDED
)

FEATURE_DIM_EXTENDED = len(FEATURE_NAMES_EXTENDED)
assert FEATURE_DIM_EXTENDED == 45, f"Expected 45 extended_v1 features, got {FEATURE_DIM_EXTENDED}"

FEATURE_INDEX_EXTENDED = {name: idx for idx, name in enumerate(FEATURE_NAMES_EXTENDED)}

FEATURE_GROUPS_EXTENDED = {
    "base": BASE_FEATURE_NAMES_EXTENDED,
    "tls": TLS_FEATURE_NAMES_EXTENDED,
    "dns": DNS_FEATURE_NAMES_EXTENDED,
    "temporal": TEMPORAL_FEATURE_NAMES_EXTENDED,
}

FEATURE_GROUP_SLICES_EXTENDED = {}
_cursor = 0
for _group_name, _feature_names in FEATURE_GROUPS_EXTENDED.items():
    _next_cursor = _cursor + len(_feature_names)
    FEATURE_GROUP_SLICES_EXTENDED[_group_name] = slice(_cursor, _next_cursor)
    _cursor = _next_cursor

