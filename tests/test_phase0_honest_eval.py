"""Tests for the Phase 0 honest-evaluation harness.

Covers CTU-13 per-row labeling, capture-separated splitting, host label
purity, bootstrap CIs, base-rate adjusted metrics, and seed aggregation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.data_loader.host_window_dataset import HostSessionRecord
from src.evaluation.multifamily_metrics import (
    bootstrap_metric_ci,
    bootstrap_score_cis,
    prevalence_adjusted_metrics,
    seed_mean_ci,
)
from src.pipelines.build_balanced_clean_benchmark_split import (
    categorize_ctu13_label,
    enforce_host_label_purity,
    namespace_hosts_by_capture,
    parse_capture_split_spec,
    split_records_by_capture,
)


def make_record(
    host_id: str,
    label: int,
    capture_id: str = "cap_a",
    timestamp: float = 0.0,
    index: int = 0,
) -> HostSessionRecord:
    return HostSessionRecord(
        session=np.zeros((2, 3), dtype=np.float32),
        mask=np.ones(2, dtype=bool),
        label=label,
        host_id=host_id,
        dest_id="10.0.0.1",
        timestamp=timestamp,
        source="ctu13",
        family="fam" if label else "",
        capture_id=capture_id,
        original_index=index,
    )


class TestCategorizeCtu13Label:
    def test_from_botnet_is_c2(self):
        assert categorize_ctu13_label("flow=From-Botnet-V42-UDP-DNS") == "c2"

    def test_from_normal_is_normal(self):
        assert categorize_ctu13_label("flow=From-Normal-V42-Jist") == "normal"

    def test_plain_normal_is_normal(self):
        assert categorize_ctu13_label("flow=Normal-V42-HTTP-windowsupdate") == "normal"

    def test_background_is_background(self):
        assert categorize_ctu13_label("flow=Background-TCP-Established") == "background"

    def test_to_normal_is_background_because_source_unverified(self):
        assert categorize_ctu13_label("flow=To-Normal-V42-UDP-NTP-server") == "background"

    def test_to_background_is_background(self):
        assert categorize_ctu13_label("flow=To-Background-UDP-CVUT-DNS-Server") == "background"

    def test_empty_is_background(self):
        assert categorize_ctu13_label("") == "background"


class TestParseCaptureSplitSpec:
    def test_empty_spec_disables_capture_mode(self):
        assert parse_capture_split_spec("") == {}
        assert parse_capture_split_spec(None) == {}

    def test_valid_spec(self):
        spec = "train=cap_a,cap_b;val=cap_c;test=cap_d"
        assert parse_capture_split_spec(spec) == {
            "cap_a": "train",
            "cap_b": "train",
            "cap_c": "val",
            "cap_d": "test",
        }

    def test_missing_split_raises(self):
        with pytest.raises(ValueError, match="missing splits"):
            parse_capture_split_spec("train=cap_a;val=cap_b")

    def test_duplicate_capture_raises(self):
        with pytest.raises(ValueError, match="multiple splits"):
            parse_capture_split_spec("train=cap_a;val=cap_a;test=cap_b")

    def test_unknown_split_name_raises(self):
        with pytest.raises(ValueError, match="Unknown split name"):
            parse_capture_split_spec("train=cap_a;dev=cap_b;test=cap_c")


class TestHostLabelPurity:
    def test_pure_hosts_pass_through(self):
        records = [make_record("h1", 0), make_record("h2", 1)]
        kept, summary = enforce_host_label_purity(records)
        assert len(kept) == 2
        assert summary["conflicting_hosts"] == 0
        assert summary["dropped_records"] == 0

    def test_mixed_host_keeps_majority_label(self):
        records = [
            make_record("mixed", 1, index=0),
            make_record("mixed", 1, index=1),
            make_record("mixed", 0, index=2),
            make_record("clean", 0, index=3),
        ]
        kept, summary = enforce_host_label_purity(records)
        assert summary["conflicting_hosts"] == 1
        assert summary["dropped_records"] == 1
        mixed_labels = {int(r.label) for r in kept if r.host_id == "mixed"}
        assert mixed_labels == {1}


class TestCaptureSplit:
    def test_records_assigned_by_capture(self):
        records = [
            make_record("h1", 1, capture_id="cap_a", index=0),
            make_record("h2", 0, capture_id="cap_a", index=1),
            make_record("h3", 1, capture_id="cap_b", index=2),
            make_record("h4", 0, capture_id="cap_c", index=3),
        ]
        splits = split_records_by_capture(
            records, {"cap_a": "train", "cap_b": "val", "cap_c": "test"}
        )
        assert len(splits["train"]) == 2
        assert len(splits["val"]) == 1
        assert len(splits["test"]) == 1

    def test_unassigned_capture_raises(self):
        records = [make_record("h1", 1, capture_id="cap_x")]
        with pytest.raises(ValueError, match="no split assignment"):
            split_records_by_capture(records, {"cap_a": "train", "cap_b": "val", "cap_c": "test"})

    def test_empty_split_raises(self):
        records = [
            make_record("h1", 1, capture_id="cap_a"),
            make_record("h2", 0, capture_id="cap_b"),
        ]
        with pytest.raises(ValueError, match="received no records"):
            split_records_by_capture(records, {"cap_a": "train", "cap_b": "val", "cap_c": "test"})

    def test_namespacing_makes_same_ip_distinct_across_captures(self):
        records = [
            make_record("1.2.3.4", 1, capture_id="cap_a"),
            make_record("1.2.3.4", 1, capture_id="cap_b"),
        ]
        namespaced = namespace_hosts_by_capture(records)
        assert {r.host_id for r in namespaced} == {"cap_a:1.2.3.4", "cap_b:1.2.3.4"}


class TestBootstrapCI:
    def test_ci_contains_point_for_overlapping_classes(self):
        rng = np.random.default_rng(7)
        y = np.array([0] * 200 + [1] * 200)
        s = np.concatenate([rng.normal(0.4, 0.2, 200), rng.normal(0.6, 0.2, 200)])
        result = bootstrap_metric_ci(
            y, s, lambda t, p: float(roc_auc_score(t, p)), n_bootstrap=200, random_seed=3
        )
        assert result["ci_lower"] <= result["point"] <= result["ci_upper"]
        assert result["ci_lower"] < result["ci_upper"]

    def test_perfect_classifier_has_degenerate_ci_at_one(self):
        y = np.array([0] * 50 + [1] * 50)
        s = np.array([0.0] * 50 + [1.0] * 50)
        result = bootstrap_metric_ci(
            y, s, lambda t, p: float(roc_auc_score(t, p)), n_bootstrap=100, random_seed=3
        )
        assert result["point"] == 1.0
        assert result["ci_lower"] == 1.0
        assert result["ci_upper"] == 1.0

    def test_bootstrap_score_cis_keys(self):
        y = np.array([0] * 100 + [1] * 100)
        s = np.linspace(0, 1, 200)
        report = bootstrap_score_cis(y, s, fpr_budgets=(0.01, 0.05), n_bootstrap=50)
        assert set(report) == {"roc_auc", "pr_auc", "recall_at_fpr_0.0100", "recall_at_fpr_0.0500"}


class TestPrevalenceAdjustedMetrics:
    def test_perfect_classifier_keeps_adjusted_pr_auc_one(self):
        y = np.array([0] * 100 + [1] * 100)
        s = np.array([0.0] * 100 + [1.0] * 100)
        report = prevalence_adjusted_metrics(y, s, base_rates=(0.001,))
        assert report["0.001"]["adjusted_pr_auc"] == pytest.approx(1.0)

    def test_adjusted_precision_matches_bayes_formula(self):
        # 10% FPR at 100% TPR by construction: 10 of 100 benign share the
        # positive score band.
        y = np.array([0] * 100 + [1] * 100)
        s = np.array([0.0] * 90 + [1.0] * 10 + [1.0] * 100)
        report = prevalence_adjusted_metrics(y, s, base_rates=(0.01,), fpr_budgets=(0.5,))
        op = report["0.01"]["operating_points"]["0.5000"]
        pi, tpr, fpr = 0.01, 1.0, 0.1
        expected = pi * tpr / (pi * tpr + (1 - pi) * fpr)
        assert op["recall"] == pytest.approx(tpr)
        assert op["fpr"] == pytest.approx(fpr)
        assert op["adjusted_precision"] == pytest.approx(expected)

    def test_alert_volume_scales_with_fpr(self):
        y = np.array([0] * 100 + [1] * 100)
        s = np.array([0.0] * 90 + [1.0] * 10 + [1.0] * 100)
        report = prevalence_adjusted_metrics(y, s, base_rates=(0.01,), fpr_budgets=(0.5,))
        op = report["0.01"]["operating_points"]["0.5000"]
        assert op["alerts_per_1k_sessions"] == pytest.approx(1000 * (0.01 * 1.0 + 0.99 * 0.1))


class TestSeedMeanCI:
    def test_constant_values_zero_width(self):
        result = seed_mean_ci([0.9, 0.9, 0.9])
        assert result["mean"] == pytest.approx(0.9)
        assert result["ci_lower"] == pytest.approx(0.9)
        assert result["ci_upper"] == pytest.approx(0.9)

    def test_matches_manual_t_interval(self):
        values = [0.90, 0.92, 0.91, 0.93, 0.89]
        result = seed_mean_ci(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        half = 2.7764451051977987 * std / math.sqrt(5)  # t(0.975, df=4)
        assert result["mean"] == pytest.approx(mean)
        assert result["ci_lower"] == pytest.approx(mean - half)
        assert result["ci_upper"] == pytest.approx(mean + half)

    def test_single_value_degenerates_to_point(self):
        result = seed_mean_ci([0.5])
        assert result["mean"] == 0.5
        assert result["ci_lower"] == 0.5
        assert result["ci_upper"] == 0.5
        assert result["n"] == 1
