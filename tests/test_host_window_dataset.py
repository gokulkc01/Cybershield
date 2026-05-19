from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_loader.host_window_dataset import (
    HOST_AWARE_SCHEMA_VERSION,
    HOST_FEATURE_NAMES,
    HostSessionRecord,
    HostWindowTorchDataset,
    build_host_windows,
    load_host_windows_npz,
    save_host_windows_npz,
    split_records,
    validate_host_disjoint,
)
from src.features.feature_config_extended import FEATURE_DIM_EXTENDED, FEATURE_NAMES_EXTENDED, SESSION_LEN


def _record(host: str, ts: float, value: float, label: int = 0, source: str = "ctu13") -> HostSessionRecord:
    session = np.zeros((SESSION_LEN, FEATURE_DIM_EXTENDED), dtype=np.float32)
    session[0, 0] = value
    session[0, 8] = 1.0
    session[0, 9] = ts
    mask = np.zeros(SESSION_LEN, dtype=bool)
    mask[0] = True
    return HostSessionRecord(
        session=session,
        mask=mask,
        label=label,
        host_id=host,
        dest_id=f"dest-{int(value)}",
        timestamp=ts,
        source=source,
        family="unit_c2" if label else "",
        capture_id="unit",
        original_index=int(value),
        host_identity_is_real=True,
    )


def test_host_windows_are_causal_and_right_aligned():
    records = [
        _record("10.0.0.1", 3.0, 3.0),
        _record("10.0.0.1", 1.0, 1.0),
        _record("10.0.0.2", 2.0, 20.0),
        _record("10.0.0.1", 4.0, 4.0),
    ]

    windows = build_host_windows(records, history_size=2, feature_names=FEATURE_NAMES_EXTENDED)

    # Global output is chronological. First host session has no history.
    assert windows.host_ids.tolist()[0] == "10.0.0.1"
    assert windows.timestamps.tolist()[0] == 1.0
    assert not windows.history_session_masks[0].any()

    # The second 10.0.0.1 sample only sees the earlier value=1 session.
    host1_second = 2
    assert windows.host_ids.tolist()[host1_second] == "10.0.0.1"
    assert windows.history_session_masks[host1_second].tolist() == [False, True]
    assert windows.history_sessions[host1_second, 1, 0, 0] == 1.0

    # The third 10.0.0.1 sample sees values 1 and 3, never itself/future.
    host1_third = 3
    assert windows.history_session_masks[host1_third].tolist() == [True, True]
    assert windows.history_sessions[host1_third, 0, 0, 0] == 1.0
    assert windows.history_sessions[host1_third, 1, 0, 0] == 3.0
    assert windows.host_features[0, HOST_FEATURE_NAMES.index("is_first_seen_host")] == 1.0
    assert windows.host_features[host1_third, HOST_FEATURE_NAMES.index("history_count")] == 2.0


def test_host_window_npz_roundtrip_and_torch_dataset(tmp_path: Path):
    windows = build_host_windows(
        [_record("10.0.0.1", 1.0, 1.0), _record("10.0.0.1", 2.0, 2.0, label=1)],
        history_size=2,
        feature_names=FEATURE_NAMES_EXTENDED,
    )
    path = tmp_path / "host_windows.npz"
    save_host_windows_npz(path, windows)

    loaded = load_host_windows_npz(path)
    assert loaded.schema_version == HOST_AWARE_SCHEMA_VERSION
    assert loaded.current_sessions.shape == (2, SESSION_LEN, FEATURE_DIM_EXTENDED)
    assert loaded.history_sessions.shape == (2, 2, SESSION_LEN, FEATURE_DIM_EXTENDED)
    assert loaded.real_host_identity is True

    dataset = HostWindowTorchDataset(loaded, domain_to_index={"ctu13": 0})
    sample = dataset[1]
    assert sample[0].shape == (SESSION_LEN, FEATURE_DIM_EXTENDED)
    assert sample[2].shape == (2, SESSION_LEN, FEATURE_DIM_EXTENDED)
    assert sample[4].shape == (2,)
    assert int(sample[6]) == 1
    assert int(sample[7]) == 0


def test_host_separated_split_has_no_host_overlap():
    records = []
    for idx in range(6):
        records.append(_record(f"10.0.0.{idx}", float(idx), float(idx), label=idx % 2))

    splits = split_records(
        records,
        split_strategy="host_separated",
        val_fraction=0.2,
        test_fraction=0.2,
        random_seed=7,
    )
    arrays = {
        name: build_host_windows(split, history_size=1, feature_names=FEATURE_NAMES_EXTENDED)
        for name, split in splits.items()
    }

    assert validate_host_disjoint(arrays["train"], arrays["val"], arrays["test"])

