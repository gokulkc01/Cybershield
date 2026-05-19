from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipelines.build_balanced_clean_benchmark_split import (
    MODERN_C2_SOURCE,
    build_modern_c2_records,
    canonical_modern_family,
    discover_modern_c2_files,
)


def test_modern_c2_records_from_normalized_csv_aliases(tmp_path: Path):
    capture_dir = tmp_path / "modern_c2" / "sliver" / "capture_a"
    capture_dir.mkdir(parents=True)
    csv_path = capture_dir / "flows.csv"
    pd.DataFrame(
        {
            "ts": [100.0, 101.0, 102.0],
            "src_ip_zeek": ["10.10.1.5", "10.10.1.5", "10.10.1.5"],
            "dest_ip_zeek": ["203.0.113.10", "203.0.113.10", "203.0.113.10"],
            "proto": ["tcp", "tcp", "tcp"],
            "duration": [0.1, 0.1, 0.1],
            "orig_bytes": [120, 130, 140],
            "resp_bytes": [80, 90, 100],
            "orig_pkts": [2, 2, 2],
            "resp_pkts": [1, 1, 1],
            "src_port_zeek": [49152, 49152, 49152],
            "dest_port_zeek": [443, 443, 443],
        }
    ).to_csv(csv_path, index=False)

    records, summary = build_modern_c2_records(
        tmp_path / "modern_c2",
        families=["sliver", "havoc", "cobalt_strike", "mythic"],
        min_flows=1,
        inactivity_timeout=300.0,
    )

    assert summary["source"] == MODERN_C2_SOURCE
    assert summary["present_families"] == ["sliver"]
    assert "havoc" in summary["missing_families"]
    assert len(records) == 1
    assert records[0].source == MODERN_C2_SOURCE
    assert records[0].family == "sliver"
    assert records[0].label == 1
    assert records[0].host_id == "10.10.1.5"
    assert records[0].host_identity_is_real is True


def test_modern_c2_discovery_ignores_sidecars(tmp_path: Path):
    root = tmp_path / "modern_c2"
    (root / "cobalt-strike" / "cap").mkdir(parents=True)
    flow_path = root / "cobalt-strike" / "cap" / "conn.log"
    sidecar_path = root / "cobalt-strike" / "cap" / "dns.log"
    flow_path.write_text("#fields\tts\tid.orig_h\tid.resp_h\tid.orig_p\tid.resp_p\tproto\tduration\torig_bytes\tresp_bytes\torig_pkts\tresp_pkts\tconn_state\thistory\n")
    sidecar_path.write_text("")

    discovered = discover_modern_c2_files(root, ["cobalt_strike"])

    assert discovered == [(flow_path, "cobalt_strike")]
    assert canonical_modern_family("Cobalt-Strike") == "cobalt_strike"
