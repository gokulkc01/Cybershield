"""Tests for CyberShield v2 Phase 0 modules.

Validates:
  - feature_config_v2: backward compat + new feature groups
  - host_timeline: session ingestion, rolling stats, deviation scoring
  - split_utils: all split strategies, leakage validation
  - dataset_builder_v2: source parsing, family inference, host key extraction,
    host timeline construction from tensors, serialization roundtrip
"""

from __future__ import annotations

import math
import tempfile
import os
from typing import List

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────
# feature_config_v2 tests
# ──────────────────────────────────────────────────────────────────────

from src.features.feature_config_v2 import (
    FEATURE_NAMES, FEATURE_DIM, LABEL_BENIGN, LABEL_C2,
    SESSION_LEN,
    C2_POSITIVE_LABELS, C2Family, DatasetSource,
    SessionMetadata, HostTimelineConfig, FusionConfig,
    V2_SESSION_FEATURES, V2_FEATURE_DIM,
    BEHAVIORAL_FEATURE_NAMES, ENTROPY_FEATURE_NAMES,
    PERSISTENCE_FEATURE_NAMES, TLS_FEATURE_NAMES,
    HOST_TIMELINE_FEATURE_NAMES,
    MAX_FPR_BUDGET, MAX_FPR_BUDGET_LOW, MAX_FPR_BUDGET_HIGH,
    label_from_string, is_private_ip,
)


class TestFeatureConfigV2:
    """Test v2 feature configuration backward compatibility and extensions."""

    def test_v1_feature_names_preserved(self):
        """v1 features must remain in the same order."""
        expected = [
            "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
            "bytes_per_pkt", "packet_ratio", "byte_ratio", "is_outbound",
            "duration", "src_port", "dst_port", "iat",
        ]
        assert FEATURE_NAMES == expected

    def test_v1_feature_dim(self):
        assert FEATURE_DIM == 12

    def test_v2_feature_groups_exist(self):
        assert len(BEHAVIORAL_FEATURE_NAMES) > 0
        assert len(ENTROPY_FEATURE_NAMES) > 0
        assert len(PERSISTENCE_FEATURE_NAMES) > 0
        assert len(TLS_FEATURE_NAMES) > 0
        assert len(HOST_TIMELINE_FEATURE_NAMES) > 0

    def test_v2_combined_feature_dim(self):
        expected = FEATURE_DIM + len(V2_SESSION_FEATURES)
        assert V2_FEATURE_DIM == expected
        assert V2_FEATURE_DIM > FEATURE_DIM

    def test_modern_c2_labels(self):
        """Modern C2 frameworks should be in the label set."""
        for label in ["sliver", "havoc", "cobalt strike", "mythic"]:
            assert label in C2_POSITIVE_LABELS, f"Missing: {label}"

    def test_label_from_string_modern(self):
        assert label_from_string("sliver") == LABEL_C2
        assert label_from_string("Havoc C2 beacon") == LABEL_C2
        assert label_from_string("ta0011") == LABEL_C2
        assert label_from_string("normal web traffic") == LABEL_BENIGN

    def test_fpr_budget_relaxed(self):
        """v2 FPR budget should be less strict than v1."""
        assert MAX_FPR_BUDGET > 0.005
        assert MAX_FPR_BUDGET_LOW >= 0.015
        assert MAX_FPR_BUDGET_HIGH <= 0.03

    def test_c2_family_enum(self):
        assert C2Family.SLIVER.value == "sliver"
        assert C2Family.HAVOC.value == "havoc"

    def test_dataset_source_enum(self):
        assert DatasetSource.UWF_ZEEKDATA24.value == "uwf_zeekdata24"

    def test_session_metadata_split_keys(self):
        meta = SessionMetadata(
            source=DatasetSource.UWF_ZEEKDATA24,
            c2_family="sliver",
            capture_id="test_capture",
        )
        assert meta.split_key_family() == "sliver"
        assert meta.split_key_source() == "uwf_zeekdata24"

    def test_session_metadata_benign_split_key(self):
        meta = SessionMetadata(source=DatasetSource.CTU13)
        assert "benign" in meta.split_key_family()

    def test_is_private_ip(self):
        assert is_private_ip("192.168.1.1") == 1
        assert is_private_ip("10.0.0.1") == 1
        assert is_private_ip("8.8.8.8") == 0
        assert is_private_ip("invalid") == 0


# ──────────────────────────────────────────────────────────────────────
# host_timeline tests
# ──────────────────────────────────────────────────────────────────────

from src.features.host_timeline import (
    HostTimeline, HostTimelineBuilder, SessionSummary,
    PartnerHistory, RollingStatistics,
)


def _make_flow_df(n_flows: int = 50, host: str = "192.168.1.10",
                  dest: str = "1.2.3.4", start_ts: float = 1000.0,
                  label: int = 0) -> pd.DataFrame:
    """Create a synthetic flow DataFrame for testing."""
    rng = np.random.default_rng(42)
    ts = np.cumsum(rng.uniform(1, 30, n_flows)) + start_ts
    return pd.DataFrame({
        "ts": ts,
        "src_ip": host,
        "dst_ip": dest,
        "proto": "tcp",
        "orig_bytes": rng.uniform(100, 5000, n_flows),
        "resp_bytes": rng.uniform(50, 3000, n_flows),
        "orig_pkts": rng.integers(1, 20, n_flows).astype(float),
        "resp_pkts": rng.integers(0, 15, n_flows).astype(float),
        "bytes_per_pkt": rng.uniform(50, 500, n_flows),
        "packet_ratio": rng.uniform(0.5, 2.0, n_flows),
        "byte_ratio": rng.uniform(0.3, 3.0, n_flows),
        "is_outbound": 1.0,
        "duration": rng.uniform(0.1, 10, n_flows),
        "src_port": rng.integers(1024, 65535, n_flows).astype(float),
        "dst_port": 443.0,
        "iat": np.diff(ts, prepend=ts[0] - 5),
        "label": label,
    })


class TestRollingStatistics:

    def test_empty(self):
        stat = RollingStatistics(name="test")
        assert stat.mean == 0.0
        assert stat.std == 0.0
        assert stat.count == 0
        assert stat.zscore(5.0) == 0.0

    def test_single_update(self):
        stat = RollingStatistics(name="test")
        stat.update(10.0)
        assert stat.count == 1
        assert stat.mean == 10.0
        assert stat.zscore(10.0) == 0.0  # Not enough samples

    def test_multiple_updates(self):
        stat = RollingStatistics(name="test")
        for v in [10, 20, 30, 40, 50]:
            stat.update(v)
        assert stat.count == 5
        assert stat.mean > 0
        assert stat.std > 0

    def test_zscore_detects_outlier(self):
        stat = RollingStatistics(name="test")
        for v in range(100):
            stat.update(float(v))
        # A value far from the mean should have high z-score
        z = stat.zscore(1000.0)
        assert abs(z) > 3.0


class TestHostTimeline:

    def test_empty_timeline(self):
        tl = HostTimeline(host_id="192.168.1.10")
        assert tl.num_sessions == 0
        assert not tl.has_baseline
        assert tl.unique_destinations == 0

    def test_add_session(self):
        tl = HostTimeline(host_id="192.168.1.10")
        summary = SessionSummary(
            session_id="s001",
            host_id="192.168.1.10",
            dest_id="1.2.3.4",
            protocol="tcp",
            start_ts=1000.0,
            end_ts=1100.0,
            duration=100.0,
            num_flows=10,
            total_orig_bytes=5000.0,
            total_resp_bytes=3000.0,
            mean_iat=10.0,
            std_iat=2.0,
            cv_iat=0.2,
        )
        deviations = tl.add_session(summary)
        assert tl.num_sessions == 1
        assert tl.unique_destinations == 1
        assert "1.2.3.4" in tl.partners
        # No baseline yet → deviations should be empty
        assert len(deviations) == 0

    def test_baseline_after_min_sessions(self):
        config = HostTimelineConfig(min_sessions_for_baseline=5)
        tl = HostTimeline(host_id="192.168.1.10", config=config)

        for i in range(10):
            summary = SessionSummary(
                session_id=f"s{i:03d}",
                host_id="192.168.1.10",
                dest_id="1.2.3.4",
                protocol="tcp",
                start_ts=1000.0 + i * 120,
                end_ts=1100.0 + i * 120,
                duration=100.0,
                num_flows=10,
                total_orig_bytes=5000.0,
                total_resp_bytes=3000.0,
                mean_iat=10.0,
                std_iat=2.0,
                cv_iat=0.2,
            )
            deviations = tl.add_session(summary)

        assert tl.has_baseline
        # After baseline, deviations should be populated
        assert len(deviations) > 0
        assert "byte_volume_zscore" in deviations

    def test_new_destination_detection(self):
        config = HostTimelineConfig(min_sessions_for_baseline=2)
        tl = HostTimeline(host_id="192.168.1.10", config=config)

        # Add sessions to build baseline
        for i in range(3):
            tl.add_session(SessionSummary(
                session_id=f"s{i:03d}",
                host_id="192.168.1.10",
                dest_id="1.2.3.4",
                protocol="tcp",
                start_ts=1000.0 + i * 120,
                end_ts=1100.0 + i * 120,
                duration=100.0,
                num_flows=10,
            ))

        # New destination
        deviations = tl.add_session(SessionSummary(
            session_id="s003",
            host_id="192.168.1.10",
            dest_id="5.6.7.8",  # NEW destination
            protocol="tcp",
            start_ts=1400.0,
            end_ts=1500.0,
            duration=100.0,
            num_flows=10,
        ))

        assert deviations.get("is_new_destination") == 1.0

    def test_partner_tracking(self):
        tl = HostTimeline(host_id="192.168.1.10")
        for i in range(5):
            tl.add_session(SessionSummary(
                session_id=f"s{i:03d}",
                host_id="192.168.1.10",
                dest_id="1.2.3.4",
                protocol="tcp",
                start_ts=1000.0 + i * 600,
                end_ts=1100.0 + i * 600,
                duration=100.0,
                num_flows=5,
                total_orig_bytes=1000.0,
                total_resp_bytes=500.0,
            ))

        partner = tl.partners["1.2.3.4"]
        assert partner.total_sessions == 5
        assert partner.total_bytes_sent == 5000.0
        assert partner.relationship_duration > 0
        assert len(partner.session_iats) == 4  # 5 sessions = 4 gaps

    def test_host_feature_vector(self):
        tl = HostTimeline(host_id="192.168.1.10")
        for i in range(15):
            tl.add_session(SessionSummary(
                session_id=f"s{i:03d}",
                host_id="192.168.1.10",
                dest_id=f"1.2.3.{i % 3 + 1}",
                protocol="tcp",
                start_ts=1000.0 + i * 300,
                end_ts=1100.0 + i * 300,
                duration=100.0,
                num_flows=10,
                total_orig_bytes=5000.0,
                total_resp_bytes=3000.0,
                mean_iat=10.0,
                std_iat=2.0,
                cv_iat=0.2,
            ))

        features = tl.get_host_feature_vector()
        assert "comm_cadence_cv" in features
        assert "session_freq_avg" in features
        assert "active_hour_entropy" in features
        assert "dest_diversity_rolling" in features
        assert features["dest_diversity_rolling"] == 3.0

    def test_to_dict(self):
        tl = HostTimeline(host_id="192.168.1.10")
        tl.add_session(SessionSummary(
            session_id="s000",
            host_id="192.168.1.10",
            dest_id="1.2.3.4",
            protocol="tcp",
            start_ts=1000.0,
            end_ts=1100.0,
            duration=100.0,
            num_flows=10,
        ))
        d = tl.to_dict()
        assert d["host_id"] == "192.168.1.10"
        assert d["num_sessions"] == 1


class TestHostTimelineBuilder:

    def test_ingest_flow_dataframe(self):
        df = _make_flow_df(n_flows=50, host="192.168.1.10", dest="1.2.3.4")
        builder = HostTimelineBuilder()
        n_sessions = builder.ingest_flow_dataframe(df, source=DatasetSource.CTU13)

        assert n_sessions > 0
        timelines = builder.build()
        assert "192.168.1.10" in timelines
        assert timelines["192.168.1.10"].num_sessions == n_sessions

    def test_multiple_hosts(self):
        df1 = _make_flow_df(n_flows=30, host="192.168.1.10", dest="1.2.3.4")
        df2 = _make_flow_df(n_flows=30, host="192.168.1.20", dest="5.6.7.8",
                             start_ts=2000.0)
        df = pd.concat([df1, df2], ignore_index=True)

        builder = HostTimelineBuilder()
        builder.ingest_flow_dataframe(df)

        timelines = builder.build()
        assert len(timelines) >= 2

    def test_summary(self):
        df = _make_flow_df(n_flows=100)
        builder = HostTimelineBuilder()
        builder.ingest_flow_dataframe(df)

        summary = builder.summary()
        assert summary["hosts"] >= 1
        assert summary["total_sessions"] > 0


# ──────────────────────────────────────────────────────────────────────
# split_utils tests
# ──────────────────────────────────────────────────────────────────────

from src.data_loader.split_utils import (
    SplitConfig, SplitStrategy, split_dataset, print_split_report,
)


def _make_labeled_dataset(
    n_benign: int = 500,
    n_c2_per_family: int = 50,
    families: List[str] = None,
    sources: List[DatasetSource] = None,
) -> tuple:
    """Create synthetic labeled dataset with metadata."""
    if families is None:
        families = ["emotet", "trickbot", "sliver", "havoc"]
    if sources is None:
        sources = [DatasetSource.CTU13, DatasetSource.UWF_ZEEKDATA24]

    rng = np.random.default_rng(42)
    labels = []
    metadata = []

    # Benign samples
    for i in range(n_benign):
        labels.append(LABEL_BENIGN)
        source = sources[i % len(sources)]
        metadata.append(SessionMetadata(
            source=source,
            capture_start_ts=float(i * 10),
        ))

    # C2 samples
    for family in families:
        for i in range(n_c2_per_family):
            labels.append(LABEL_C2)
            source = sources[hash(family) % len(sources)]
            metadata.append(SessionMetadata(
                source=source,
                c2_family=family,
                capture_start_ts=float(i * 10 + 5000),
            ))

    labels = np.array(labels, dtype=np.int64)
    return labels, metadata


class TestSplitStrategies:

    def test_random_stratified(self):
        labels, metadata = _make_labeled_dataset()
        config = SplitConfig(strategy=SplitStrategy.RANDOM_STRATIFIED)
        result = split_dataset(labels, metadata, config)

        assert result.validate_no_leakage()
        assert len(result.train_indices) > 0
        assert len(result.val_indices) > 0
        assert len(result.test_indices) > 0

        total = len(result.train_indices) + len(result.val_indices) + len(result.test_indices)
        assert total == len(labels)

    def test_family_separated_no_overlap(self):
        labels, metadata = _make_labeled_dataset()
        config = SplitConfig(strategy=SplitStrategy.FAMILY_SEPARATED)
        result = split_dataset(labels, metadata, config)

        assert result.validate_no_leakage()

        # Verify no C2 family overlap between train and test
        train_c2_families = {
            metadata[i].c2_family
            for i in result.train_indices
            if labels[i] == LABEL_C2
        }
        test_c2_families = {
            metadata[i].c2_family
            for i in result.test_indices
            if labels[i] == LABEL_C2
        }
        overlap = train_c2_families & test_c2_families
        assert len(overlap) == 0, f"Family overlap found: {overlap}"

    def test_family_separated_explicit_holdout(self):
        labels, metadata = _make_labeled_dataset()
        config = SplitConfig(
            strategy=SplitStrategy.FAMILY_SEPARATED,
            held_out_families={"sliver", "havoc"},
        )
        result = split_dataset(labels, metadata, config)

        test_c2_families = {
            metadata[i].c2_family
            for i in result.test_indices
            if labels[i] == LABEL_C2
        }
        # Held-out families should be in test
        assert "sliver" in test_c2_families or "havoc" in test_c2_families

    def test_source_separated(self):
        labels, metadata = _make_labeled_dataset(
            sources=[DatasetSource.CTU13, DatasetSource.UWF_ZEEKDATA24, DatasetSource.MCFP_STRATOSPHERE]
        )
        config = SplitConfig(strategy=SplitStrategy.SOURCE_SEPARATED)
        result = split_dataset(labels, metadata, config)

        assert result.validate_no_leakage()

        train_sources = {metadata[i].source.value for i in result.train_indices}
        test_sources = {metadata[i].source.value for i in result.test_indices}
        assert len(train_sources & test_sources) == 0, "Source overlap found!"

    def test_time_separated(self):
        labels, metadata = _make_labeled_dataset()
        config = SplitConfig(strategy=SplitStrategy.TIME_SEPARATED)
        result = split_dataset(labels, metadata, config)

        assert result.validate_no_leakage()

        # Train timestamps should generally precede test timestamps
        train_ts = [metadata[i].capture_start_ts for i in result.train_indices]
        test_ts = [metadata[i].capture_start_ts for i in result.test_indices]
        assert np.mean(train_ts) < np.mean(test_ts)

    def test_zero_shot(self):
        labels, metadata = _make_labeled_dataset()
        config = SplitConfig(strategy=SplitStrategy.ZERO_SHOT)
        result = split_dataset(labels, metadata, config)

        assert result.validate_no_leakage()

        # Zero-shot: test families should not be in train
        train_c2_families = {
            metadata[i].c2_family
            for i in result.train_indices
            if labels[i] == LABEL_C2
        }
        test_c2_families = {
            metadata[i].c2_family
            for i in result.test_indices
            if labels[i] == LABEL_C2
        }
        overlap = train_c2_families & test_c2_families
        assert len(overlap) == 0, f"Zero-shot family overlap: {overlap}"

    def test_no_metadata_fallback(self):
        """Without metadata, should fall back to random stratified."""
        labels = np.array([0]*100 + [1]*50, dtype=np.int64)
        config = SplitConfig(strategy=SplitStrategy.FAMILY_SEPARATED)
        # Should warn and fall back
        with pytest.warns(UserWarning, match="requires metadata"):
            result = split_dataset(labels, metadata=None, config=config)
        assert result.strategy == SplitStrategy.RANDOM_STRATIFIED

    def test_split_report(self, capsys):
        labels, metadata = _make_labeled_dataset()
        config = SplitConfig(strategy=SplitStrategy.FAMILY_SEPARATED)
        result = split_dataset(labels, metadata, config)
        print_split_report(result, labels)
        captured = capsys.readouterr()
        assert "SPLIT REPORT" in captured.out
        assert "No leakage" in captured.out


# ──────────────────────────────────────────────────────────────────────
# dataset_builder_v2 tests
# ──────────────────────────────────────────────────────────────────────

from src.features.dataset_builder_v2 import (
    parse_source_spec,
    _infer_family_from_path,
    _extract_host_key,
    build_host_timelines_from_sessions,
    save_host_timelines,
    load_host_timelines,
)


class TestParseSourceSpec:
    """Test source spec parsing, including edge cases."""

    def test_basic_parse(self):
        path, source, label = parse_source_spec("data/file.npz:ctu13:c2")
        assert path == "data/file.npz"
        assert source == DatasetSource.CTU13
        assert label == "c2"

    def test_windows_path_with_drive_letter(self):
        """rsplit must handle D:\\path correctly."""
        path, source, label = parse_source_spec(
            "D:\\data\\processed\\uwf.npz:uwf_zeekdata24:benign"
        )
        assert path == "D:\\data\\processed\\uwf.npz"
        assert source == DatasetSource.UWF_ZEEKDATA24
        assert label == "benign"

    def test_unknown_source(self):
        path, source, label = parse_source_spec("file.npz:some_new_source:mixed")
        assert source == DatasetSource.UNKNOWN

    def test_invalid_label_type(self):
        with pytest.raises(ValueError, match="Invalid label_type"):
            parse_source_spec("file.npz:ctu13:attack")

    def test_insufficient_parts(self):
        with pytest.raises(ValueError, match="Invalid source spec"):
            parse_source_spec("file.npz:ctu13")


class TestFamilyInference:
    """Test regex-based family inference from file paths."""

    def test_modern_frameworks(self):
        assert _infer_family_from_path("/data/sliver_c2/sessions.npz") == "sliver"
        assert _infer_family_from_path("havoc_beacon_capture.npz") == "havoc"
        assert _infer_family_from_path("cobalt-strike/sessions.npz") == "cobalt_strike"
        assert _infer_family_from_path("cobalt_strike_data.npz") == "cobalt_strike"

    def test_legacy_families(self):
        assert _infer_family_from_path("neris_sessions.npz") == "neris"
        assert _infer_family_from_path("emotet/conn.npz") == "emotet"

    def test_case_insensitive(self):
        assert _infer_family_from_path("SLIVER_capture.npz") == "sliver"
        assert _infer_family_from_path("CobaltStrike.npz") == "cobalt_strike"

    def test_brute_ratel_variants(self):
        assert _infer_family_from_path("brute_ratel.npz") == "brute_ratel"
        assert _infer_family_from_path("bruteratel_c4.npz") == "brute_ratel"

    def test_unknown_path(self):
        assert _infer_family_from_path("totally_unknown_data.npz") == "unknown"


class TestHostKeyExtraction:
    """Test synthetic host key derivation from session tensors."""

    def _make_meta(self, source=DatasetSource.CTU13, capture="cap.npz"):
        return SessionMetadata(source=source, capture_id=capture)

    def test_deterministic(self):
        """Same session should always produce same host key."""
        rng = np.random.default_rng(99)
        session = rng.random((SESSION_LEN, FEATURE_DIM)).astype(np.float32)
        mask = np.ones(SESSION_LEN, dtype=bool)
        meta = self._make_meta()

        key1 = _extract_host_key(session, mask, meta)
        key2 = _extract_host_key(session, mask, meta)
        assert key1 == key2

    def test_different_sources_different_keys(self):
        """Sessions from different sources shouldn't collide."""
        rng = np.random.default_rng(99)
        session = rng.random((SESSION_LEN, FEATURE_DIM)).astype(np.float32)
        mask = np.ones(SESSION_LEN, dtype=bool)

        key1 = _extract_host_key(session, mask, self._make_meta(DatasetSource.CTU13))
        key2 = _extract_host_key(session, mask, self._make_meta(DatasetSource.UWF_ZEEKDATA24))
        assert key1 != key2

    def test_uses_multiple_signals(self):
        """Key should contain dst_port, src_port, byte_ratio, pkt_ratio, outbound."""
        rng = np.random.default_rng(42)
        session = rng.random((SESSION_LEN, FEATURE_DIM)).astype(np.float32)
        mask = np.ones(SESSION_LEN, dtype=bool)
        key = _extract_host_key(session, mask, self._make_meta())

        # Key format: source:capture:dp{...}:sp{...}:br{...}:pr{...}:o{...}
        assert ":dp" in key
        assert ":sp" in key
        assert ":br" in key
        assert ":pr" in key
        assert ":o" in key

    def test_empty_mask_fallback(self):
        """All-zero mask should produce a valid key, not crash."""
        session = np.zeros((SESSION_LEN, FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(SESSION_LEN, dtype=bool)
        key = _extract_host_key(session, mask, self._make_meta())
        assert len(key) > 0


class TestSessionSummaryFromTensor:
    """Test the factory method for converting tensors to SessionSummary."""

    def test_basic_conversion(self):
        rng = np.random.default_rng(42)
        session = rng.random((SESSION_LEN, FEATURE_DIM)).astype(np.float32)
        mask = np.ones(SESSION_LEN, dtype=bool)
        mask[15:] = False
        session[15:] = 0.0

        summary = SessionSummary.from_tensor(
            session=session, mask=mask, label=1,
            session_id="s001", host_id="test_host",
        )

        assert summary.session_id == "s001"
        assert summary.host_id == "test_host"
        assert summary.label == 1
        assert summary.num_flows == 15
        assert summary.total_orig_bytes > 0
        assert summary.duration > 0
        assert summary.dest_id.startswith("dest_port_")

    def test_uses_metadata_timestamp(self):
        rng = np.random.default_rng(42)
        session = rng.random((SESSION_LEN, FEATURE_DIM)).astype(np.float32)
        mask = np.ones(SESSION_LEN, dtype=bool)
        meta = SessionMetadata(
            source=DatasetSource.CTU13,
            capture_start_ts=1000000.0,
        )

        summary = SessionSummary.from_tensor(
            session=session, mask=mask, label=0,
            session_id="s002", host_id="host_a",
            metadata=meta,
        )

        assert summary.start_ts == 1000000.0
        assert summary.end_ts > summary.start_ts

    def test_all_values_finite(self):
        rng = np.random.default_rng(123)
        session = rng.random((SESSION_LEN, FEATURE_DIM)).astype(np.float32)
        mask = np.ones(SESSION_LEN, dtype=bool)

        summary = SessionSummary.from_tensor(
            session=session, mask=mask, label=0,
            session_id="s003", host_id="host_b",
        )

        for field_name in [
            "total_orig_bytes", "total_resp_bytes", "mean_iat",
            "std_iat", "cv_iat", "mean_bytes_per_pkt", "duration",
        ]:
            val = getattr(summary, field_name)
            assert np.isfinite(val), f"Non-finite {field_name}={val}"


class TestBuildHostTimelinesFromSessions:
    """Test the bridge between session tensors and HostTimeline objects."""

    def _make_synthetic_data(self, n=50):
        """Create synthetic session tensors with metadata."""
        rng = np.random.default_rng(42)
        sessions = rng.random((n, SESSION_LEN, FEATURE_DIM)).astype(np.float32)
        labels = np.array([0]*(n//2) + [1]*(n - n//2), dtype=np.int64)
        masks = np.ones((n, SESSION_LEN), dtype=bool)

        # Make first 5 flows real, rest padding for some sessions
        for i in range(n):
            n_real = rng.integers(3, SESSION_LEN + 1)
            masks[i, n_real:] = False
            sessions[i, n_real:] = 0.0

        metadata = []
        for i in range(n):
            metadata.append(SessionMetadata(
                source=DatasetSource.CTU13,
                c2_family="sliver" if labels[i] == 1 else "",
                capture_id="test_capture.npz",
                host_id=f"host_{i % 5}",  # 5 unique hosts
                capture_start_ts=float(i * 100),
            ))

        return sessions, labels, masks, metadata

    def test_builds_timelines(self):
        sessions, labels, masks, metadata = self._make_synthetic_data(50)
        config = HostTimelineConfig(min_sessions_for_baseline=3)
        timelines = build_host_timelines_from_sessions(
            sessions, labels, masks, metadata, config=config
        )

        # Should have 5 hosts (i % 5)
        assert len(timelines) == 5

        # Each host should have 10 sessions (50 / 5)
        for tl in timelines.values():
            assert tl.num_sessions == 10
            assert tl.has_baseline  # 10 > min_sessions_for_baseline=3

    def test_host_id_populated(self):
        sessions, labels, masks, metadata = self._make_synthetic_data(10)
        for meta in metadata:
            assert meta.host_id != "", "host_id must be populated"

    def test_timelines_have_sessions(self):
        sessions, labels, masks, metadata = self._make_synthetic_data(20)
        timelines = build_host_timelines_from_sessions(
            sessions, labels, masks, metadata
        )
        total_sessions = sum(tl.num_sessions for tl in timelines.values())
        assert total_sessions == 20

    def test_timeline_serialization_roundtrip(self, tmp_path):
        """Test save/load of host timelines."""
        sessions, labels, masks, metadata = self._make_synthetic_data(20)
        timelines = build_host_timelines_from_sessions(
            sessions, labels, masks, metadata
        )

        path = str(tmp_path / "timelines.pkl")
        save_host_timelines(timelines, path)
        loaded = load_host_timelines(path)

        assert set(loaded.keys()) == set(timelines.keys())
        for host_id in timelines:
            assert loaded[host_id].num_sessions == timelines[host_id].num_sessions

    def test_feature_vectors_from_built_timelines(self):
        """Timelines built from session tensors should produce valid features."""
        sessions, labels, masks, metadata = self._make_synthetic_data(50)
        config = HostTimelineConfig(min_sessions_for_baseline=3)
        timelines = build_host_timelines_from_sessions(
            sessions, labels, masks, metadata, config=config
        )

        for tl in timelines.values():
            features = tl.get_host_feature_vector()
            assert isinstance(features, dict)
            assert len(features) > 0
            for key, value in features.items():
                assert np.isfinite(value), f"Non-finite {key}={value} for {tl.host_id}"


# ──────────────────────────────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end integration tests for Phase 0 pipeline."""

    def test_flow_to_timeline_to_features(self):
        """Test the full flow: raw flows → host timeline → feature vector."""
        df = _make_flow_df(n_flows=100, host="192.168.1.10", dest="1.2.3.4")

        builder = HostTimelineBuilder(
            config=HostTimelineConfig(min_sessions_for_baseline=3)
        )
        n_sessions = builder.ingest_flow_dataframe(
            df, source=DatasetSource.CTU13
        )

        assert n_sessions > 0

        tl = builder.get_timeline("192.168.1.10")
        assert tl is not None

        features = tl.get_host_feature_vector()
        assert isinstance(features, dict)
        assert len(features) > 0

        for key, value in features.items():
            assert np.isfinite(value), f"Non-finite value for {key}: {value}"

    def test_session_tensor_to_timeline_to_features(self):
        """Test the NPZ path: session tensors → host timelines → features."""
        rng = np.random.default_rng(123)
        n = 30
        sessions = rng.random((n, SESSION_LEN, FEATURE_DIM)).astype(np.float32)
        labels = np.array([0]*15 + [1]*15, dtype=np.int64)
        masks = np.ones((n, SESSION_LEN), dtype=bool)

        metadata = [
            SessionMetadata(
                source=DatasetSource.CTU13,
                host_id=f"host_{i % 3}",
                capture_start_ts=float(i * 60),
            )
            for i in range(n)
        ]

        timelines = build_host_timelines_from_sessions(
            sessions, labels, masks, metadata,
            config=HostTimelineConfig(min_sessions_for_baseline=3),
        )

        assert len(timelines) == 3  # 3 unique hosts

        for tl in timelines.values():
            assert tl.num_sessions == 10  # 30 / 3
            assert tl.has_baseline
            features = tl.get_host_feature_vector()
            assert "comm_cadence_cv" in features
