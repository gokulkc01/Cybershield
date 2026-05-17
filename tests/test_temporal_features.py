import numpy as np
import pandas as pd

from src.features.temporal_features import TemporalStatsExtractor


def test_temporal_feature_extraction_for_regular_beacon():
    extractor = TemporalStatsExtractor()

    features = extractor.extract_from_session({"iat": np.ones(20)})

    assert set(features) == set(extractor.TEMPORAL_FEATURES)
    assert features["periodicity_confidence"] > 0.95
    assert features["recurrence_stability"] > 0.9
    assert features["iat_std"] == 0.0
    assert features["iat_p50"] == 1.0


def test_temporal_feature_extraction_for_irregular_timing():
    extractor = TemporalStatsExtractor()
    irregular_iats = np.array([0.1, 0.2, 0.15, 10.0, 30.0, 0.25, 0.3, 2.0])

    features = extractor.extract_from_session({"iat": irregular_iats})

    assert features["periodicity_confidence"] < 0.7
    assert features["iat_entropy"] > 0
    assert features["burst_count"] >= 1


def test_temporal_extractor_accepts_dataframe_timestamps():
    extractor = TemporalStatsExtractor()
    df = pd.DataFrame({"ts": [10.0, 11.0, 12.0, 13.0]})

    features = extractor.extract_from_session(df)

    assert features["iat_mean"] == 1.0
    assert features["periodicity_confidence"] > 0.95


def test_temporal_empty_features_are_finite_and_ordered():
    extractor = TemporalStatsExtractor()

    features = extractor.extract_from_session({})
    vector = extractor.to_feature_vector(features)

    assert vector.shape == (len(extractor.TEMPORAL_FEATURES),)
    assert np.all(np.isfinite(vector))
    assert np.all(vector == 0.0)
