from src.features.dns_features import DNSFeatureExtractor
from src.features.feature_config_experiment import FEATURE_NAMES as BASE_FEATURE_NAMES
from src.features.feature_config_extended import (
    DNS_FEATURE_NAMES_EXTENDED,
    FEATURE_DIM_EXTENDED,
    FEATURE_GROUP_SLICES_EXTENDED,
    FEATURE_INDEX_EXTENDED,
    FEATURE_NAMES_EXTENDED,
    FEATURE_SCHEMA_VERSION,
    SCHEMA_VERSION_EXTENDED,
    TEMPORAL_FEATURE_NAMES_EXTENDED,
    TLS_FEATURE_NAMES_EXTENDED,
)
from src.features.temporal_features import TemporalStatsExtractor
from src.features.tls_features import TLSFeatureExtractor


def test_extended_v1_feature_contract_is_pinned():
    assert SCHEMA_VERSION_EXTENDED == "extended_v1"
    assert FEATURE_SCHEMA_VERSION == "extended_v1"
    assert FEATURE_DIM_EXTENDED == 45
    assert len(FEATURE_NAMES_EXTENDED) == FEATURE_DIM_EXTENDED
    assert len(set(FEATURE_NAMES_EXTENDED)) == FEATURE_DIM_EXTENDED
    assert FEATURE_INDEX_EXTENDED == {name: idx for idx, name in enumerate(FEATURE_NAMES_EXTENDED)}


def test_extended_v1_groups_match_extractor_feature_order():
    assert FEATURE_NAMES_EXTENDED[:10] == list(BASE_FEATURE_NAMES)
    assert TLS_FEATURE_NAMES_EXTENDED == TLSFeatureExtractor.TLS_FEATURES
    assert DNS_FEATURE_NAMES_EXTENDED == DNSFeatureExtractor.DNS_FEATURES
    assert TEMPORAL_FEATURE_NAMES_EXTENDED == TemporalStatsExtractor.TEMPORAL_FEATURES


def test_extended_v1_group_slices_are_stable():
    assert FEATURE_NAMES_EXTENDED[FEATURE_GROUP_SLICES_EXTENDED["base"]] == list(BASE_FEATURE_NAMES)
    assert FEATURE_NAMES_EXTENDED[FEATURE_GROUP_SLICES_EXTENDED["tls"]] == TLSFeatureExtractor.TLS_FEATURES
    assert FEATURE_NAMES_EXTENDED[FEATURE_GROUP_SLICES_EXTENDED["dns"]] == DNSFeatureExtractor.DNS_FEATURES
    assert FEATURE_NAMES_EXTENDED[FEATURE_GROUP_SLICES_EXTENDED["temporal"]] == TemporalStatsExtractor.TEMPORAL_FEATURES
