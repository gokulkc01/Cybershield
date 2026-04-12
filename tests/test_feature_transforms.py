import numpy as np

from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.features.feature_config import FEATURE_INDEX


def test_apply_feature_transforms_log_scales_selected_features_and_keeps_padding_zero():
    x = np.zeros((1, 2, len(FEATURE_INDEX)), dtype=np.float32)
    x[0, 0, FEATURE_INDEX["orig_bytes"]] = 99.0
    x[0, 0, FEATURE_INDEX["iat"]] = 9.0
    masks = np.array([[True, False]], dtype=bool)

    transformed = apply_feature_transforms(
        x,
        masks,
        FeatureTransformConfig(log_scale_features=("orig_bytes", "iat"), ablate_features=()),
    )

    assert np.isclose(transformed[0, 0, FEATURE_INDEX["orig_bytes"]], np.log1p(99.0))
    assert np.isclose(transformed[0, 0, FEATURE_INDEX["iat"]], np.log1p(9.0))
    assert np.all(transformed[0, 1] == 0.0)


def test_apply_feature_transforms_ablation_zeroes_selected_features():
    x = np.ones((1, 1, len(FEATURE_INDEX)), dtype=np.float32)
    masks = np.array([[True]], dtype=bool)

    transformed = apply_feature_transforms(
        x,
        masks,
        FeatureTransformConfig(log_scale_features=(), ablate_features=("src_port", "dst_port", "is_outbound")),
    )

    assert transformed[0, 0, FEATURE_INDEX["src_port"]] == 0.0
    assert transformed[0, 0, FEATURE_INDEX["dst_port"]] == 0.0
    assert transformed[0, 0, FEATURE_INDEX["is_outbound"]] == 0.0
