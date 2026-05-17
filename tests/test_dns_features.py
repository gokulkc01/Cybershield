import numpy as np
import pandas as pd

from src.features.dns_features import DNSFeatureExtractor


def test_dns_feature_extraction_from_zeek_dns_log():
    extractor = DNSFeatureExtractor()
    dns_df = pd.DataFrame(
        {
            "ts": [0.0, 0.5, 3.0, 65.0],
            "query": [
                "google.com",
                "api.google.com",
                "xj29azq91k.biz",
                "xj29azq91k.biz",
            ],
            "qtype_name": ["A", "A", "AAAA", "A"],
            "rcode_name": ["NOERROR", "NOERROR", "NXDOMAIN", "NXDOMAIN"],
            "ttls": [[300], [300], [10], [20]],
        }
    )

    features = extractor.extract_from_zeek_dns_log(dns_df)

    assert set(features) == set(extractor.DNS_FEATURES)
    assert features["dns_query_count"] == 4
    assert features["dns_unique_domains"] == 3
    assert features["dns_nxdomain_rate"] == 0.5
    assert features["dns_query_type_entropy"] > 0
    assert features["dns_lookup_frequency"] > 0
    assert features["dns_rapid_fire_rate"] == 1 / 3
    assert features["dns_subdomain_depth_avg"] > 0
    assert features["dns_dga_score"] > 0


def test_dns_empty_features_are_finite_and_ordered():
    extractor = DNSFeatureExtractor()

    features = extractor.extract_from_zeek_dns_log(pd.DataFrame())
    vector = extractor.to_feature_vector(features)

    assert vector.shape == (len(extractor.DNS_FEATURES),)
    assert np.all(np.isfinite(vector))
    assert np.all(vector == 0.0)
