import numpy as np

from src.data_loader.normalization import fit_feature_normalizer


def test_feature_normalizer_uses_only_real_flows_and_preserves_padding():
    x = np.zeros((2, 3, 2), dtype=np.float32)
    x[0, 0] = [1.0, 10.0]
    x[0, 1] = [3.0, 14.0]
    x[1, 0] = [5.0, 18.0]
    masks = np.array(
        [
            [True, True, False],
            [True, False, False],
        ],
        dtype=bool,
    )

    normalizer = fit_feature_normalizer(x, masks)
    transformed = normalizer.transform(x, masks)

    valid = transformed.reshape(-1, 2)[masks.reshape(-1)]
    assert np.allclose(valid.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(valid.std(axis=0), 1.0, atol=1e-6)
    assert np.all(transformed[~masks] == 0.0)
