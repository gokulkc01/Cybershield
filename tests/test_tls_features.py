import numpy as np
import pandas as pd

from src.features.tls_features import TLSFeatureExtractor


def test_tls_feature_extraction_from_zeek_ssl_log():
    extractor = TLSFeatureExtractor()
    ssl_df = pd.DataFrame(
        {
            "version": ["TLSv1.2", "TLSv1.2", "TLSv1.3"],
            "cipher": ["ECDHE-RSA-AES128", "ECDHE-RSA-AES128", "TLS_AES_256_GCM_SHA384"],
            "ja3": ["771,1-2,10-11,23,0", "771,1-2,10-11,23,0", "772,3,13-14,29,0"],
            "session_id": ["a", "a", "b"],
            "cert_chain_fuids": ["cert-1", "cert-1", "cert-2"],
            "next_protocol": ["h2", "h2", "http/1.1"],
            "cert_validity_days": [90, 90, 30],
        }
    )

    features = extractor.extract_from_zeek_ssl_log(ssl_df)

    assert set(features) == set(extractor.TLS_FEATURES)
    assert features["tls_version"] == 1.2
    assert features["tls_version_diversity"] == 2
    assert features["tls_cipher_count"] == 2
    assert features["tls_extension_count"] == 2
    assert features["tls_session_reuse_rate"] == 1 / 3
    assert features["tls_cert_reuse_rate"] == 1 / 3
    assert features["tls_alpn_count"] == 2
    assert features["tls_cert_validity_days"] == 70


def test_tls_empty_features_are_finite_and_ordered():
    extractor = TLSFeatureExtractor()

    features = extractor.extract_from_zeek_ssl_log(pd.DataFrame())
    vector = extractor.to_feature_vector(features)

    assert vector.shape == (len(extractor.TLS_FEATURES),)
    assert np.all(np.isfinite(vector))
    assert np.all(vector == 0.0)
