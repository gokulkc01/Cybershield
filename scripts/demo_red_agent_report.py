"""Generate a readable Red-Agent demo report from real training outputs.

This script turns the host-aware Red-Agent training summary and JSONL event log
into a concise demo artifact. It does not simulate results: every metric is
derived from the training files written by ``src.pipelines.red_agent_host_aware_train``.

Example:
    python scripts/demo_red_agent_report.py \
        --summary experiments/red_agent_host_aware_smoke/host_aware_phase1_summary.json \
        --out-md experiments/red_agent_host_aware_smoke/red_agent_demo_report.md \
        --out-json experiments/red_agent_host_aware_smoke/red_agent_demo_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "experiments" / "red_agent_host_aware_smoke" / "host_aware_phase1_summary.json"


@dataclass(frozen=True)
class RedAgentReport:
    summary_path: str
    training_log_path: str
    detector_checkpoint: str | None
    host_npz: str | None
    policy_checkpoint: str | None
    detector_threshold: float
    samples: int
    log_rows: int
    epochs: int
    mean_original_probability: float
    mean_mutated_probability: float
    mean_probability_delta: float
    median_probability_delta: float
    best_probability_drop: float
    worst_probability_increase: float
    probability_drop_rate: float
    threshold_evasion_rate: float
    threshold_evasions: int
    mean_reward: float
    mean_realism: float
    realistic_rate: float
    constraint_violation_rate: float
    mutation_counts: dict[str, int]
    mutation_summaries: dict[str, dict[str, float]]
    interpretation: str
    caveats: list[str]


def _resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected summary JSON object, got {type(data).__name__}")
    return data


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                warnings.append(f"Skipped malformed JSONL line {line_no}: {exc}")
                continue
            if not isinstance(parsed, dict):
                warnings.append(f"Skipped non-object JSONL line {line_no}")
                continue
            rows.append(parsed)

    if not rows:
        raise ValueError(f"No valid training rows found in {path}")
    return rows, warnings


def _number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def build_report(summary_path: Path, log_path: Path | None = None) -> RedAgentReport:
    summary = _load_json(summary_path)

    resolved_log = log_path
    if resolved_log is None:
        summary_log = summary.get("training_log")
        if not summary_log:
            raise ValueError("No --log-path provided and summary does not contain 'training_log'")
        resolved_log = _resolve_path(str(summary_log))

    rows, parse_warnings = _load_jsonl(resolved_log)
    threshold = float(summary.get("detector_threshold", 0.5))
    row_count = len(rows)

    original_probs = [_number(row, "c2_probability_original") for row in rows]
    mutated_probs = [_number(row, "c2_probability_mutated") for row in rows]

    # probability_delta in the training log is original_probability - mutated_probability.
    # Positive means the Red-Agent reduced detector confidence.
    deltas = [
        _number(row, "probability_delta", original - mutated)
        for row, original, mutated in zip(rows, original_probs, mutated_probs)
    ]
    rewards = [_number(row, "reward") for row in rows]
    realism_scores = [_number(row, "realism_score") for row in rows]

    dropped = sum(1 for delta in deltas if delta > 0)
    evasions = sum(
        1
        for original, mutated in zip(original_probs, mutated_probs)
        if original >= threshold and mutated < threshold
    )
    realistic = sum(1 for row in rows if bool(row.get("is_realistic")))
    violations = sum(1 for row in rows if row.get("constraint_violations"))

    mutation_counts = Counter(str(row.get("mutation_type", "unknown")) for row in rows)
    by_mutation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mutation[str(row.get("mutation_type", "unknown"))].append(row)

    mutation_summaries: dict[str, dict[str, float]] = {}
    for mutation, mutation_rows in sorted(by_mutation.items()):
        mutation_deltas = [
            _number(row, "probability_delta")
            for row in mutation_rows
        ]
        mutation_rewards = [
            _number(row, "reward")
            for row in mutation_rows
        ]
        mutation_summaries[mutation] = {
            "count": float(len(mutation_rows)),
            "mean_probability_delta": _mean(mutation_deltas),
            "mean_reward": _mean(mutation_rewards),
            "confidence_drop_rate": _rate(sum(1 for delta in mutation_deltas if delta > 0), len(mutation_rows)),
        }

    mean_delta = _mean(deltas)
    evasion_rate = _rate(evasions, row_count)
    drop_rate = _rate(dropped, row_count)
    mean_reward = _mean(rewards)
    mean_realism = _mean(realism_scores)
    violation_rate = _rate(violations, row_count)

    caveats = list(parse_warnings)
    if row_count < 25:
        caveats.append(
            "This is a small smoke run. Use it to demonstrate the workflow, not to claim stable adversarial performance."
        )
    if evasion_rate == 0:
        caveats.append(
            "No threshold evasion occurred in this run; the demo still shows detector-confidence movement and policy training output."
        )
    if violation_rate > 0:
        caveats.append(
            "Some mutations violated constraints; review those rows before presenting the run as behavior-preserving."
        )

    if evasion_rate > 0:
        interpretation = (
            "The Red-Agent found at least one mutation that crossed below the detector threshold, "
            "demonstrating concrete evasion pressure against the detector."
        )
    elif mean_delta > 0:
        interpretation = (
            "The Red-Agent reduced detector confidence on average, but this run did not cross the detection threshold."
        )
    else:
        interpretation = (
            "The Red-Agent did not reduce detector confidence on average in this short run; use this as a robustness smoke test "
            "and run more epochs/samples for a stronger adversarial evaluation."
        )

    return RedAgentReport(
        summary_path=str(summary_path),
        training_log_path=str(resolved_log),
        detector_checkpoint=summary.get("detector_checkpoint"),
        host_npz=summary.get("host_npz"),
        policy_checkpoint=summary.get("policy_checkpoint"),
        detector_threshold=threshold,
        samples=int(summary.get("samples", row_count)),
        log_rows=row_count,
        epochs=int(summary.get("epochs", max(int(_number(row, "epoch", 1)) for row in rows))),
        mean_original_probability=_mean(original_probs),
        mean_mutated_probability=_mean(mutated_probs),
        mean_probability_delta=mean_delta,
        median_probability_delta=_median(deltas),
        best_probability_drop=max(deltas) if deltas else 0.0,
        worst_probability_increase=min(deltas) if deltas else 0.0,
        probability_drop_rate=drop_rate,
        threshold_evasion_rate=evasion_rate,
        threshold_evasions=evasions,
        mean_reward=mean_reward,
        mean_realism=mean_realism,
        realistic_rate=_rate(realistic, row_count),
        constraint_violation_rate=violation_rate,
        mutation_counts=dict(sorted(mutation_counts.items())),
        mutation_summaries=mutation_summaries,
        interpretation=interpretation,
        caveats=caveats,
    )


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(report: RedAgentReport) -> str:
    lines = [
        "# CyberShield Red-Agent Demo Report",
        "",
        "## What This Demonstrates",
        "",
        (
            "The Red-Agent takes real host-aware C2 session windows, applies policy-selected mutations, "
            "and measures whether the trained detector's C2 probability changes. This report is generated "
            "from the actual training summary and JSONL event log."
        ),
        "",
        "## Inputs",
        "",
        f"- Detector checkpoint: `{report.detector_checkpoint}`",
        f"- Host-window NPZ: `{report.host_npz}`",
        f"- Training log: `{report.training_log_path}`",
        f"- Policy checkpoint: `{report.policy_checkpoint}`",
        f"- Detector threshold: `{report.detector_threshold:.6f}`",
        f"- Epochs: `{report.epochs}`",
        f"- Samples requested: `{report.samples}`",
        f"- Log rows analyzed: `{report.log_rows}`",
        "",
        "## Main Results",
        "",
        "| Metric | Value | Meaning |",
        "| --- | ---: | --- |",
        f"| Mean original C2 probability | {_num(report.mean_original_probability)} | Detector confidence before mutation |",
        f"| Mean mutated C2 probability | {_num(report.mean_mutated_probability)} | Detector confidence after mutation |",
        f"| Mean probability delta | {_num(report.mean_probability_delta)} | Positive means confidence dropped |",
        f"| Median probability delta | {_num(report.median_probability_delta)} | Typical confidence movement |",
        f"| Best probability drop | {_num(report.best_probability_drop)} | Strongest detector-confidence reduction |",
        f"| Worst probability increase | {_num(report.worst_probability_increase)} | Strongest adverse movement |",
        f"| Confidence drop rate | {_pct(report.probability_drop_rate)} | Share of rows where confidence decreased |",
        f"| Threshold evasion rate | {_pct(report.threshold_evasion_rate)} | Share crossing from detected to below threshold |",
        f"| Threshold evasions | {report.threshold_evasions} | Count of successful threshold crossings |",
        f"| Mean reward | {_num(report.mean_reward)} | PPO objective signal |",
        f"| Mean realism | {_num(report.mean_realism)} | Behavior-preservation score |",
        f"| Realistic mutation rate | {_pct(report.realistic_rate)} | Rows marked realistic |",
        f"| Constraint violation rate | {_pct(report.constraint_violation_rate)} | Rows with mutation constraint violations |",
        "",
        "## Mutation Breakdown",
        "",
        "| Mutation | Count | Mean Delta | Mean Reward | Drop Rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for mutation, count in report.mutation_counts.items():
        summary = report.mutation_summaries[mutation]
        lines.append(
            f"| {mutation} | {count} | {_num(summary['mean_probability_delta'])} | "
            f"{_num(summary['mean_reward'])} | {_pct(summary['confidence_drop_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Demo Interpretation",
            "",
            report.interpretation,
            "",
            "## How To Explain It Live",
            "",
            "1. Start with the detector threshold and original C2 probabilities.",
            "2. Explain that the Red-Agent chooses mutations such as timing, TLS, flow, or composite changes.",
            "3. Show the mutated probabilities and probability delta.",
            "4. If delta is positive, detector confidence went down.",
            "5. If threshold evasion is greater than zero, at least one sample crossed below the detector threshold.",
            "6. Use realism and constraint-violation metrics to explain whether mutations remained plausible.",
            "",
            "## Caveats",
            "",
        ]
    )

    if report.caveats:
        lines.extend(f"- {item}" for item in report.caveats)
    else:
        lines.append("- No caveats detected by the report generator.")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a real Red-Agent demo report from training outputs.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Path to host_aware_phase1_summary.json")
    parser.add_argument("--log-path", type=Path, default=None, help="Optional path to training JSONL log")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional Markdown report output path")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON report output path")
    args = parser.parse_args()

    summary_path = _resolve_path(args.summary)
    log_path = _resolve_path(args.log_path) if args.log_path else None
    report = build_report(summary_path, log_path)
    markdown = render_markdown(report)

    print(markdown)

    if args.out_md:
        out_md = _resolve_path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")

    if args.out_json:
        out_json = _resolve_path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
