import json

from scripts.demo_red_agent_report import build_report, render_markdown


def test_red_agent_demo_report_builds_from_real_training_outputs(tmp_path):
    summary_path = tmp_path / "summary.json"
    log_path = tmp_path / "training.jsonl"

    rows = [
        {
            "epoch": 1,
            "mutation_type": "tls",
            "c2_probability_original": 0.9,
            "c2_probability_mutated": 0.4,
            "probability_delta": 0.5,
            "reward": 0.25,
            "realism_score": 0.8,
            "is_realistic": True,
            "constraint_violations": [],
        },
        {
            "epoch": 1,
            "mutation_type": "flow",
            "c2_probability_original": 0.7,
            "c2_probability_mutated": 0.8,
            "probability_delta": -0.1,
            "reward": -0.2,
            "realism_score": 0.6,
            "is_realistic": False,
            "constraint_violations": ["invalid duration"],
        },
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "detector_threshold": 0.5,
                "epochs": 1,
                "samples": 2,
                "training_log": str(log_path),
                "detector_checkpoint": "detector.pth",
                "host_npz": "host_windows.npz",
                "policy_checkpoint": "policy.pth",
            }
        ),
        encoding="utf-8",
    )

    report = build_report(summary_path)
    assert report.log_rows == 2
    assert report.threshold_evasions == 1
    assert report.threshold_evasion_rate == 0.5
    assert report.probability_drop_rate == 0.5
    assert report.constraint_violation_rate == 0.5
    assert report.mutation_counts == {"flow": 1, "tls": 1}

    markdown = render_markdown(report)
    assert "Threshold evasion rate" in markdown
    assert "50.0%" in markdown
