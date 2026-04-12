import numpy as np
import pytest
import torch

from src.data_loader.npz_utils import load_session_npz
from src.features.feature_config import FEATURE_DIM, SESSION_LEN
from src.models.transformer import C2Transformer


def test_load_session_npz_normalizes_padding_mask_convention(tmp_path):
    x = np.zeros((2, SESSION_LEN, FEATURE_DIM), dtype=np.float32)
    x[0, :3, 0] = [10.0, 11.0, 12.0]
    x[1, :2, 0] = [20.0, 21.0]
    y = np.array([1, 0], dtype=np.int64)

    # Historical convention: True meant padding.
    stored_masks = np.array(
        [
            [False, False, False] + [True] * (SESSION_LEN - 3),
            [False, False] + [True] * (SESSION_LEN - 2),
        ],
        dtype=bool,
    )
    np.savez(tmp_path / "sessions.npz", X=x, y=y, masks=stored_masks)

    _, _, real_masks = load_session_npz(str(tmp_path / "sessions.npz"))
    expected = np.array(
        [
            [True, True, True] + [False] * (SESSION_LEN - 3),
            [True, True] + [False] * (SESSION_LEN - 2),
        ],
        dtype=bool,
    )
    assert np.array_equal(real_masks, expected)


def test_transformer_derivative_toggle_preserves_input_contract():
    model = C2Transformer(
        feature_dim=FEATURE_DIM,
        seq_len=SESSION_LEN,
        use_derivative_features=True,
    )
    assert model.input_projection.in_features == FEATURE_DIM + 2

    batch_x = torch.zeros(4, SESSION_LEN, FEATURE_DIM)
    batch_x[:, 0, 0] = 100.0
    batch_x[:, 1, 0] = 120.0
    batch_x[:, 1, -1] = 30.0
    batch_mask = torch.zeros(4, SESSION_LEN, dtype=torch.bool)

    logits = model(batch_x, batch_mask)
    assert logits.shape == (4,)


def test_evaluate_checkpoint_refuses_threshold_tuning_on_eval_data(tmp_path):
    checkpoint_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": C2Transformer().state_dict()}, checkpoint_path)

    x = np.zeros((2, SESSION_LEN, FEATURE_DIM), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    masks = np.zeros((2, SESSION_LEN), dtype=bool)
    masks[:, 0] = True
    np.savez(tmp_path / "eval.npz", X=x, y=y, masks=masks)

    from src.evaluation.evaluate_transformer import evaluate_checkpoint

    with pytest.raises(ValueError, match="missing a validation-derived operating threshold"):
        evaluate_checkpoint(str(checkpoint_path), str(tmp_path / "eval.npz"))
