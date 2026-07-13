"""Unit tests for the Phase 1 additions.

Covers the Fourier/autocorrelation beaconing detector, the host-context
ablation knobs on the host-aware trainer, and the phase1 baseline pipeline
helpers (orientation, pooling, evaluation report shape).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data_loader.host_window_dataset import HostWindowArrays
from src.models.beaconing_detector import (
    BeaconingConfig,
    BeaconingDetector,
    reconstruct_flow_times,
    score_event_times,
)
from src.pipelines.phase1_baselines import evaluate_method, masked_pool, orientation_sign
from src.training.train_host_aware_domain_adaptive import _apply_host_context_ablations


# ── Beaconing detector ────────────────────────────────────────────────────────


class TestScoreEventTimes:
    def test_perfect_beacon_scores_high(self):
        result = score_event_times(np.arange(100) * 60.0)
        assert result.score > 0.7
        assert result.cv_score == pytest.approx(1.0)
        assert result.autocorr_score == pytest.approx(1.0)

    def test_jittered_beacon_beats_poisson(self):
        rng = np.random.default_rng(7)
        beacon = np.cumsum(60.0 * (1.0 + 0.1 * rng.standard_normal(100)))
        poisson = np.cumsum(rng.exponential(60.0, 100))
        assert score_event_times(beacon).score > score_event_times(poisson).score

    def test_poisson_beats_bursty(self):
        rng = np.random.default_rng(7)
        poisson = np.cumsum(rng.exponential(60.0, 100))
        bursty = np.cumsum((rng.pareto(1.2, 100) + 0.1) * 10.0)
        assert score_event_times(poisson).score > score_event_times(bursty).score

    def test_too_few_events_scores_zero(self):
        result = score_event_times(np.arange(5) * 60.0)
        assert result.score == 0.0
        assert result.n_events == 5

    def test_unsorted_input_is_sorted(self):
        times = np.arange(50) * 30.0
        shuffled = np.random.default_rng(0).permutation(times)
        assert score_event_times(shuffled).score == score_event_times(times).score

    def test_score_bounded(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            times = np.cumsum(rng.exponential(10.0, 50))
            result = score_event_times(times)
            assert 0.0 <= result.score <= 1.0

    def test_simultaneous_events_do_not_crash(self):
        result = score_event_times(np.zeros(10))
        assert result.score == 0.0


class TestReconstructFlowTimes:
    def test_first_flow_at_session_timestamp(self):
        session = np.zeros((4, 10), dtype=np.float32)
        session[:, 2] = [999.0, 30.0, 30.0, 30.0]  # iat at index 2
        mask = np.ones(4, dtype=bool)
        times = reconstruct_flow_times(1000.0, session, mask, iat_index=2)
        # First flow's own IAT (gap to the previous session) is excluded.
        assert times.tolist() == [1000.0, 1030.0, 1060.0, 1090.0]

    def test_empty_mask_returns_session_timestamp(self):
        session = np.zeros((4, 10), dtype=np.float32)
        times = reconstruct_flow_times(500.0, session, np.zeros(4, dtype=bool), iat_index=2)
        assert times.tolist() == [500.0]

    def test_negative_iat_clipped(self):
        session = np.zeros((3, 5), dtype=np.float32)
        session[:, 0] = [0.0, -10.0, 5.0]
        mask = np.ones(3, dtype=bool)
        times = reconstruct_flow_times(0.0, session, mask, iat_index=0)
        assert times.tolist() == [0.0, 0.0, 5.0]


class TestBeaconingDetectorSessions:
    def _make_inputs(self, n_sessions: int = 8, period: float = 60.0):
        rng = np.random.default_rng(1)
        feature_names = ("orig_bytes", "iat")
        sessions = np.zeros((n_sessions, 5, 2), dtype=np.float32)
        masks = np.ones((n_sessions, 5), dtype=bool)
        sessions[:, :, 1] = period
        host_ids = np.asarray(["hostA"] * (n_sessions // 2) + ["hostB"] * (n_sessions - n_sessions // 2))
        dest_ids = np.asarray(["c2.example"] * (n_sessions // 2) + ["web.example"] * (n_sessions - n_sessions // 2))
        # hostA beacons every `period`; hostB sessions start at random times
        timestamps = np.concatenate(
            [
                np.arange(n_sessions // 2) * period * 5,
                np.sort(rng.uniform(0, 10000, n_sessions - n_sessions // 2)),
            ]
        )
        # hostB has irregular in-session iats
        sessions[n_sessions // 2 :, :, 1] = rng.exponential(period, (n_sessions - n_sessions // 2, 5))
        return sessions, masks, host_ids, dest_ids, timestamps, feature_names

    def test_scores_broadcast_per_pair(self):
        sessions, masks, host_ids, dest_ids, timestamps, feature_names = self._make_inputs()
        detector = BeaconingDetector(BeaconingConfig(min_events=6))
        scores = detector.score_sessions(
            sessions=sessions,
            masks=masks,
            host_ids=host_ids,
            dest_ids=dest_ids,
            timestamps=timestamps,
            feature_names=feature_names,
        )
        assert scores.shape == (len(sessions),)
        # all sessions of the same pair share one score
        assert len(set(scores[: len(sessions) // 2].tolist())) == 1
        # periodic pair outranks the irregular pair
        assert scores[0] > scores[-1]

    def test_missing_iat_feature_raises(self):
        sessions, masks, host_ids, dest_ids, timestamps, _ = self._make_inputs()
        detector = BeaconingDetector()
        with pytest.raises(ValueError, match="iat"):
            detector.score_sessions(
                sessions=sessions,
                masks=masks,
                host_ids=host_ids,
                dest_ids=dest_ids,
                timestamps=timestamps,
                feature_names=("orig_bytes", "resp_bytes"),
            )

    def test_length_mismatch_raises(self):
        sessions, masks, host_ids, dest_ids, timestamps, feature_names = self._make_inputs()
        detector = BeaconingDetector()
        with pytest.raises(ValueError, match="equal length"):
            detector.score_sessions(
                sessions=sessions,
                masks=masks,
                host_ids=host_ids[:-1],
                dest_ids=dest_ids,
                timestamps=timestamps,
                feature_names=feature_names,
            )


# ── Host-context ablation knobs ───────────────────────────────────────────────


def _make_host_arrays(n: int = 6, history_size: int = 8) -> HostWindowArrays:
    rng = np.random.default_rng(0)
    session_len, feature_dim, host_dim = 4, 3, 5
    return HostWindowArrays(
        current_sessions=rng.normal(size=(n, session_len, feature_dim)).astype(np.float32),
        current_masks=np.ones((n, session_len), dtype=bool),
        history_sessions=rng.normal(size=(n, history_size, session_len, feature_dim)).astype(np.float32),
        history_flow_masks=np.ones((n, history_size, session_len), dtype=bool),
        history_session_masks=np.ones((n, history_size), dtype=bool),
        host_features=rng.normal(size=(n, host_dim)).astype(np.float32),
        labels=np.zeros(n, dtype=np.int64),
        host_ids=np.asarray([f"h{i}" for i in range(n)]),
        dest_ids=np.asarray([f"d{i}" for i in range(n)]),
        timestamps=np.arange(n, dtype=np.float64),
        sources=np.asarray(["ctu13"] * n),
        families=np.asarray([""] * n),
        capture_ids=np.asarray(["c"] * n),
        original_indices=np.arange(n, dtype=np.int64),
        feature_names=("f0", "f1", "f2"),
        host_feature_names=("h0", "h1", "h2", "h3", "h4"),
        history_size=history_size,
    )


class TestHostContextAblations:
    def test_ablate_host_features_zeroes(self):
        arrays = _make_host_arrays()
        _apply_host_context_ablations(arrays, ablate_host_features=True, history_limit=None)
        assert not arrays.host_features.any()
        assert arrays.history_session_masks.all()  # history untouched

    def test_history_limit_keeps_most_recent(self):
        arrays = _make_host_arrays(history_size=8)
        _apply_host_context_ablations(arrays, ablate_host_features=False, history_limit=3)
        # oldest 5 slots masked and zeroed; most recent 3 (trailing) kept
        assert not arrays.history_session_masks[:, :5].any()
        assert arrays.history_session_masks[:, 5:].all()
        assert not arrays.history_sessions[:, :5].any()
        assert arrays.history_sessions[:, 5:].any()
        assert arrays.host_features.any()  # host features untouched

    def test_history_limit_zero_removes_all_history(self):
        arrays = _make_host_arrays()
        _apply_host_context_ablations(arrays, ablate_host_features=False, history_limit=0)
        assert not arrays.history_session_masks.any()
        assert not arrays.history_flow_masks.any()

    def test_history_limit_at_capacity_is_noop(self):
        arrays = _make_host_arrays(history_size=8)
        _apply_host_context_ablations(arrays, ablate_host_features=False, history_limit=8)
        assert arrays.history_session_masks.all()

    def test_negative_history_limit_raises(self):
        arrays = _make_host_arrays()
        with pytest.raises(ValueError, match="history_limit"):
            _apply_host_context_ablations(arrays, ablate_host_features=False, history_limit=-1)


# ── Phase 1 pipeline helpers ──────────────────────────────────────────────────


class TestOrientationSign:
    def test_positive_direction(self):
        labels = np.asarray([0, 0, 1, 1])
        assert orientation_sign(np.asarray([0.1, 0.2, 0.8, 0.9]), labels) == 1.0

    def test_negative_direction(self):
        labels = np.asarray([0, 0, 1, 1])
        assert orientation_sign(np.asarray([0.9, 0.8, 0.2, 0.1]), labels) == -1.0


class TestMaskedPool:
    def test_pooling_respects_mask(self):
        sequences = np.zeros((1, 3, 2), dtype=np.float32)
        sequences[0, 0] = [1.0, 10.0]
        sequences[0, 1] = [3.0, 30.0]
        sequences[0, 2] = [999.0, 999.0]  # padded, must be ignored
        masks = np.asarray([[True, True, False]])
        pooled = masked_pool(sequences, masks)
        assert pooled.shape == (1, 8)
        np.testing.assert_allclose(pooled[0, :2], [2.0, 20.0])  # mean block first
        np.testing.assert_allclose(pooled[0, 4:6], [1.0, 10.0])  # min
        np.testing.assert_allclose(pooled[0, 6:8], [3.0, 30.0])  # max

    def test_empty_session_pools_to_zero(self):
        sequences = np.ones((1, 3, 2), dtype=np.float32)
        masks = np.zeros((1, 3), dtype=bool)
        pooled = masked_pool(sequences, masks)
        np.testing.assert_allclose(pooled, 0.0)


class TestEvaluateMethod:
    def test_report_shape_and_frozen_thresholds(self):
        rng = np.random.default_rng(0)
        val_y = np.asarray([0] * 50 + [1] * 50)
        test_y = np.asarray([0] * 50 + [1] * 50)
        val_scores = np.concatenate([rng.uniform(0, 0.6, 50), rng.uniform(0.4, 1.0, 50)])
        test_scores = np.concatenate([rng.uniform(0, 0.6, 50), rng.uniform(0.4, 1.0, 50)])
        result = evaluate_method(
            "toy",
            val_y=val_y,
            val_scores=val_scores,
            test_y=test_y,
            test_scores=test_scores,
            budgets=(0.015, 0.03),
            default_fpr_budget=0.015,
            base_rates=(0.001, 0.01),
            n_bootstrap=50,
            seed=42,
        )
        assert result["model_name"] == "toy"
        assert set(result["validation_metrics_by_budget"]) == {"0.0150", "0.0300"}
        # test thresholds must equal the val-frozen ones
        for key in ("0.0150", "0.0300"):
            assert result["test_metrics_by_budget"][key]["threshold"] == (
                result["validation_metrics_by_budget"][key]["threshold"]
            )
        assert "prevalence_adjusted" in result["honest_test_report"]
        assert "bootstrap_cis" in result["honest_test_report"]
