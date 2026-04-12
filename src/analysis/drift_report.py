"""Feature drift analysis between two session datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from src.data_loader.split_utils import load_filtered_sessions
from src.features.feature_config import FEATURE_NAMES


@dataclass(frozen=True)
class FeatureDriftRow:
    feature: str
    source_mean: float
    target_mean: float
    source_std: float
    target_std: float
    source_median: float
    target_median: float
    standardized_mean_shift: float
    median_ratio: float
    source_p95: float
    target_p95: float


def _flatten_real_flows(sequences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    valid_rows = masks.reshape(-1)
    return sequences.reshape(-1, sequences.shape[-1])[valid_rows]


def _safe_ratio(numerator: float, denominator: float) -> float:
    denom = abs(denominator) if abs(denominator) > 1e-9 else 1e-9
    return float(numerator / denom)


def compute_feature_drift(
    source_npz: str,
    target_npz: str,
    min_flows: int = 5,
) -> list[FeatureDriftRow]:
    src_x, _, src_m = load_filtered_sessions(source_npz, min_flows=min_flows)
    tgt_x, _, tgt_m = load_filtered_sessions(target_npz, min_flows=min_flows)

    src = _flatten_real_flows(src_x, src_m)
    tgt = _flatten_real_flows(tgt_x, tgt_m)

    rows: list[FeatureDriftRow] = []
    for idx, feature in enumerate(FEATURE_NAMES):
        src_col = src[:, idx]
        tgt_col = tgt[:, idx]

        src_mean = float(np.mean(src_col))
        tgt_mean = float(np.mean(tgt_col))
        src_std = float(np.std(src_col))
        tgt_std = float(np.std(tgt_col))
        pooled_std = float(np.sqrt((src_std ** 2 + tgt_std ** 2) / 2.0) + 1e-9)
        standardized_mean_shift = abs(tgt_mean - src_mean) / pooled_std

        src_median = float(np.median(src_col))
        tgt_median = float(np.median(tgt_col))
        median_ratio = _safe_ratio(tgt_median + 1e-9, src_median + 1e-9)

        rows.append(
            FeatureDriftRow(
                feature=feature,
                source_mean=src_mean,
                target_mean=tgt_mean,
                source_std=src_std,
                target_std=tgt_std,
                source_median=src_median,
                target_median=tgt_median,
                standardized_mean_shift=float(standardized_mean_shift),
                median_ratio=float(median_ratio),
                source_p95=float(np.quantile(src_col, 0.95)),
                target_p95=float(np.quantile(tgt_col, 0.95)),
            )
        )

    rows.sort(key=lambda row: row.standardized_mean_shift, reverse=True)
    return rows


def format_report(rows: list[FeatureDriftRow], top_k: int = 12) -> str:
    lines = ["Top feature shifts:"]
    for row in rows[:top_k]:
        lines.append(
            f"{row.feature}: shift={row.standardized_mean_shift:.3f} | "
            f"mean {row.source_mean:.3f}->{row.target_mean:.3f} | "
            f"median {row.source_median:.3f}->{row.target_median:.3f} | "
            f"p95 {row.source_p95:.3f}->{row.target_p95:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare feature drift between two session NPZ datasets")
    parser.add_argument("--source_npz", required=True)
    parser.add_argument("--target_npz", required=True)
    parser.add_argument("--min_flows", type=int, default=5)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    rows = compute_feature_drift(args.source_npz, args.target_npz, min_flows=args.min_flows)
    print(format_report(rows))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump([asdict(row) for row in rows], handle, indent=2)


if __name__ == "__main__":
    main()
