from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_loader.npz_utils import detect_npz_schema, load_session_npz
from src.features.feature_config_extended import FEATURE_DIM_EXTENDED, FEATURE_SCHEMA_VERSION
from src.models.transformer import C2Transformer
from src.pipelines.build_extended_dataset import build_extended_dataset_from_dataframe


def test_extended_dataset_pipeline_end_to_end(tmp_path: Path):
    df = pd.DataFrame(
        {
            "ts": [0.0, 1.0, 2.0],
            "src_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.1"],
            "dst_ip": ["10.0.0.2", "10.0.0.2", "10.0.0.2"],
            "proto": ["tcp", "tcp", "tcp"],
            "label": [1, 1, 1],
            "orig_bytes": [120.0, 240.0, 360.0],
            "resp_bytes": [40.0, 80.0, 120.0],
            "orig_pkts": [12.0, 24.0, 36.0],
            "resp_pkts": [4.0, 8.0, 12.0],
            "duration": [1.0, 1.5, 2.0],
            "is_outbound": [1.0, 1.0, 1.0],
            "iat": [0.2, 0.3, 0.4],
        }
    )

    out_dir = tmp_path / "extended"
    manifest = build_extended_dataset_from_dataframe(
        df,
        str(out_dir),
        source_dataset="unit-test",
        min_flows=1,
        inactivity_timeout=300.0,
    )

    npz_path = out_dir / "sessions_extended.npz"
    labels_path = out_dir / "labels.csv"
    manifest_path = out_dir / "dataset_manifest.json"

    assert npz_path.exists()
    assert labels_path.exists()
    assert manifest_path.exists()
    assert manifest["schema_version"] == FEATURE_SCHEMA_VERSION
    assert manifest["feature_dim"] == FEATURE_DIM_EXTENDED

    schema = detect_npz_schema(str(npz_path))
    assert schema["schema_version"] == FEATURE_SCHEMA_VERSION
    assert schema["feature_dim"] == FEATURE_DIM_EXTENDED

    sequences, labels, masks = load_session_npz(str(npz_path))
    assert sequences.shape[1:] == (20, FEATURE_DIM_EXTENDED)
    assert labels.shape == (1,)
    assert masks.shape == (1, 20)
    assert np.isfinite(sequences).all()

    model = C2Transformer(feature_dim=FEATURE_DIM_EXTENDED, seq_len=20)
    padding_masks = torch.tensor(~masks, dtype=torch.bool)
    outputs = model(torch.tensor(sequences, dtype=torch.float32), padding_masks)

    assert outputs.shape == (1,)
    assert torch.isfinite(outputs).all()
