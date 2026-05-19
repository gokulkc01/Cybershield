from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data_loader.host_window_dataset import (
    HOST_AWARE_SCHEMA_VERSION,
    HostSessionRecord,
    build_host_windows,
    save_host_windows_npz,
)
from src.features.feature_config_extended import (
    FEATURE_DIM_EXTENDED,
    FEATURE_NAMES_EXTENDED,
    FEATURE_SCHEMA_VERSION,
    SESSION_LEN,
)
from src.models.host_aware_domain_adaptive_transformer import HostAwareDomainAdaptiveTransformer
from src.pipelines.red_agent_host_aware_train import run_host_aware_red_agent_training


def _record(ts: float, label: int, value: float) -> HostSessionRecord:
    session = np.zeros((SESSION_LEN, FEATURE_DIM_EXTENDED), dtype=np.float32)
    session[0, 0] = value
    session[0, 2] = 1.0
    session[0, 8] = 1.0
    mask = np.zeros(SESSION_LEN, dtype=bool)
    mask[0] = True
    return HostSessionRecord(
        session=session,
        mask=mask,
        label=label,
        host_id="10.0.0.1",
        dest_id=f"192.0.2.{int(value)}",
        timestamp=ts,
        source="unit",
        family="unit_c2" if label else "",
        capture_id="unit",
        original_index=int(ts),
        host_identity_is_real=True,
    )


def _write_checkpoint(path: Path, history_size: int) -> None:
    model = HostAwareDomainAdaptiveTransformer(
        domains=["unit"],
        feature_dim=FEATURE_DIM_EXTENDED,
        host_feature_dim=15,
        seq_len=SESSION_LEN,
        history_size=history_size,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "host_aware_domain_adaptive_transformer",
            "domains": ["unit"],
            "domain_to_index": {"unit": 0},
            "schema_version": HOST_AWARE_SCHEMA_VERSION,
            "session_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES_EXTENDED),
            "host_feature_names": [f"h{i}" for i in range(15)],
            "feature_dim": FEATURE_DIM_EXTENDED,
            "host_feature_dim": 15,
            "session_len": SESSION_LEN,
            "history_size": history_size,
            "normalize_features": False,
            "normalize_host_features": False,
            "feature_transform_config": {"schema": FEATURE_SCHEMA_VERSION},
            "default_fpr_budget": 0.015,
            "validation_metrics_by_budget": {
                "0.0150": {
                    "threshold": 0.5,
                    "recall": 1.0,
                    "fpr": 0.0,
                    "f1": 1.0,
                    "feasible": True,
                }
            },
        },
        path,
    )


def test_host_aware_red_agent_training_smoke(tmp_path: Path):
    history_size = 2
    windows = build_host_windows(
        [
            _record(1.0, 1, 10.0),
            _record(2.0, 1, 20.0),
            _record(3.0, 0, 30.0),
        ],
        history_size=history_size,
        feature_names=FEATURE_NAMES_EXTENDED,
        session_schema_version=FEATURE_SCHEMA_VERSION,
    )
    host_npz = tmp_path / "train_host_windows.npz"
    save_host_windows_npz(host_npz, windows)
    checkpoint = tmp_path / "host_aware.pth"
    _write_checkpoint(checkpoint, history_size)

    summary = run_host_aware_red_agent_training(
        checkpoint_path=str(checkpoint),
        host_npz=str(host_npz),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        max_samples=2,
        device="cpu",
    )

    assert summary["samples"] == 2
    assert Path(summary["policy_checkpoint"]).exists()
    assert Path(summary["training_log"]).exists()
