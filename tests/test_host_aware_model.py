from __future__ import annotations

import torch

from src.models.host_aware_domain_adaptive_transformer import HostAwareDomainAdaptiveTransformer


def test_host_aware_model_forward_empty_and_full_history():
    batch_size = 3
    history_size = 4
    seq_len = 20
    feature_dim = 45
    host_feature_dim = 15

    model = HostAwareDomainAdaptiveTransformer(
        domains=["ctu13", "uwf_zeekdata24"],
        feature_dim=feature_dim,
        host_feature_dim=host_feature_dim,
        seq_len=seq_len,
        history_size=history_size,
        d_model=32,
        nhead=4,
        num_layers=1,
        host_num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )

    current = torch.zeros(batch_size, seq_len, feature_dim)
    current[:, 0, 0] = 1.0
    current_padding = torch.ones(batch_size, seq_len, dtype=torch.bool)
    current_padding[:, 0] = False

    history = torch.zeros(batch_size, history_size, seq_len, feature_dim)
    history_flow_padding = torch.ones(batch_size, history_size, seq_len, dtype=torch.bool)
    history_session_mask = torch.zeros(batch_size, history_size, dtype=torch.bool)
    host_features = torch.zeros(batch_size, host_feature_dim)
    domain_ids = torch.tensor([0, 1, 0], dtype=torch.long)

    empty_outputs = model(
        current,
        current_padding,
        history,
        history_flow_padding,
        history_session_mask,
        host_features,
        domain_ids,
        return_components=True,
    )
    assert empty_outputs["c2_logit"].shape == (batch_size,)
    assert empty_outputs["session_logit"].shape == (batch_size,)
    assert empty_outputs["host_logit"].shape == (batch_size,)
    assert torch.isfinite(empty_outputs["c2_logit"]).all()

    history[:, -2:, 0, 0] = 2.0
    history_flow_padding[:, -2:, 0] = False
    history_session_mask[:, -2:] = True
    full_logits = model(
        current,
        current_padding,
        history,
        history_flow_padding,
        history_session_mask,
        host_features,
        domain_ids,
    )
    assert full_logits.shape == (batch_size,)
    assert torch.isfinite(full_logits).all()

