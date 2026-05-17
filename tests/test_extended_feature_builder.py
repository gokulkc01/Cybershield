import numpy as np
import pandas as pd

from src.features.extended_feature_builder import ExtendedFeatureBuilder, build_extended_feature_vector
from src.features.feature_config_extended import (
    FEATURE_DIM_EXTENDED,
    FEATURE_GROUP_SLICES_EXTENDED,
    FEATURE_INDEX_EXTENDED,
    FEATURE_NAMES_EXTENDED,
)


def test_builder_zero_fills_missing_tls_dns_context():
    builder = ExtendedFeatureBuilder()
    session = {
        "orig_bytes": 1000,
        "resp_bytes": 250,
        "orig_pkts": 10,
        "resp_pkts": 5,
        "is_outbound": 1,
        "duration": 30,
        "iat": [10, 10, 10, 10],
    }

    vector = builder.build_feature_vector(session)

    assert vector.shape == (FEATURE_DIM_EXTENDED,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert vector[FEATURE_INDEX_EXTENDED["orig_bytes"]] == 1000
    assert vector[FEATURE_INDEX_EXTENDED["resp_bytes"]] == 250
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["bytes_per_pkt"]], 1250 / 15)
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["packet_ratio"]], 2.0)
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["byte_ratio"]], 4.0)
    assert np.all(vector[FEATURE_GROUP_SLICES_EXTENDED["tls"]] == 0.0)
    assert np.all(vector[FEATURE_GROUP_SLICES_EXTENDED["dns"]] == 0.0)
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["iat_mean"]], 10.0)
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["periodicity_confidence"]], 1.0)


def test_builder_includes_tls_dns_context_when_available():
    builder = ExtendedFeatureBuilder()
    session = {
        "bytes_in": 500,
        "bytes_out": 100,
        "packets_in": 5,
        "packets_out": 5,
        "duration": 3,
        "iat": [0.5, 1.0, 1.5],
    }
    tls_context = pd.DataFrame(
        {
            "version": ["TLSv1.3", "TLSv1.3"],
            "cipher": ["TLS_AES_128_GCM_SHA256", "TLS_AES_128_GCM_SHA256"],
            "next_protocol": ["h2", "http/1.1"],
            "ja3": ["771,4865-4866,0-11-10,29-23,0", "771,4865-4866,0-11-10,29-23,0"],
            "session_id": ["abc", "abc"],
            "cert_chain_fuids": ["cert-a", "cert-a"],
            "handshake_size": [512, 768],
            "cert_validity_days": [90, 90],
        }
    )
    dns_context = pd.DataFrame(
        {
            "query": ["a.example.com", "b.example.com", "xj92kq.biz"],
            "rcode_name": ["NOERROR", "NOERROR", "NXDOMAIN"],
            "qtype_name": ["A", "AAAA", "A"],
            "ttls": ["60,60", "60", "30"],
            "ts": [0.0, 0.5, 3.0],
        }
    )

    vector = builder.build_feature_vector(session, tls_context=tls_context, dns_context=dns_context)

    assert vector.shape == (FEATURE_DIM_EXTENDED,)
    assert np.isfinite(vector).all()
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["tls_version"]], 1.3)
    assert vector[FEATURE_INDEX_EXTENDED["tls_cipher_count"]] == 1.0
    assert vector[FEATURE_INDEX_EXTENDED["tls_alpn_count"]] == 2.0
    assert vector[FEATURE_INDEX_EXTENDED["dns_query_count"]] == 3.0
    assert vector[FEATURE_INDEX_EXTENDED["dns_unique_domains"]] == 3.0
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["dns_nxdomain_rate"]], 1 / 3)


def test_builder_accepts_session_dataframe_and_aggregates_base_features():
    builder = ExtendedFeatureBuilder()
    session_df = pd.DataFrame(
        {
            "orig_bytes": [100, 300],
            "resp_bytes": [50, 150],
            "orig_pkts": [10, 30],
            "resp_pkts": [5, 15],
            "duration": [1, 2],
            "is_outbound": [1, 1],
            "iat": [0, 3],
        }
    )

    vector = builder.build_feature_vector(session_df)

    assert vector[FEATURE_INDEX_EXTENDED["orig_bytes"]] == 400
    assert vector[FEATURE_INDEX_EXTENDED["resp_bytes"]] == 200
    assert vector[FEATURE_INDEX_EXTENDED["orig_pkts"]] == 40
    assert vector[FEATURE_INDEX_EXTENDED["resp_pkts"]] == 20
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["bytes_per_pkt"]], 10.0)
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["packet_ratio"]], 2.0)
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["byte_ratio"]], 2.0)
    assert vector[FEATURE_INDEX_EXTENDED["duration"]] == 3
    assert np.isclose(vector[FEATURE_INDEX_EXTENDED["iat"]], 1.5)


def test_builder_feature_dict_keys_match_schema():
    builder = ExtendedFeatureBuilder()

    features = builder.build_feature_dict(None)

    assert list(features) == FEATURE_NAMES_EXTENDED
    assert len(features) == FEATURE_DIM_EXTENDED
    assert all(value == 0.0 for value in features.values())


def test_convenience_builder_returns_extended_vector():
    vector = build_extended_feature_vector({"orig_bytes": 1, "resp_bytes": 2, "orig_pkts": 1, "resp_pkts": 1})

    assert vector.shape == (FEATURE_DIM_EXTENDED,)
    assert vector[FEATURE_INDEX_EXTENDED["orig_bytes"]] == 1
