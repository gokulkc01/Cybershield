"""Build an extended_v1 dataset from a labeled UGR'16-style tabular file.

The UGR'16 release is commonly distributed as a labeled CSV / tabular file.
This helper normalizes the common CTU-style flow columns, maps a user-selected
label column into binary labels, and then writes the 45-feature extended
dataset via the shared extended builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader.normalized_telemetry import _rename_alias_columns
from src.pipelines.build_extended_dataset import build_extended_dataset_from_dataframe


_BENIGN_TOKENS = {"benign", "normal", "legitimate", "clean", "background", "0", "false"}


def _read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def _coerce_timestamp(frame: pd.DataFrame) -> pd.Series:
    for candidate in ("timestamp", "ts", "StartTime", "start_time", "time", "datetime", "date_time"):
        if candidate in frame.columns:
            series = pd.to_datetime(frame[candidate], errors="coerce")
            if series.notna().any():
                return series.astype("int64") // 10**9
    raise ValueError("Could not find a timestamp column in the UGR'16 input")


def _map_labels(frame: pd.DataFrame, label_column: str) -> pd.Series:
    if label_column not in frame.columns:
        raise ValueError(f"Label column '{label_column}' not found in input file")

    values = frame[label_column]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).astype(int).clip(lower=0, upper=1)

    normalized = values.astype(str).str.strip().str.lower()
    return normalized.apply(lambda value: 0 if value in _BENIGN_TOKENS else 1).astype(int)


def _prepare_ugr16_frame(path: str, label_column: str, family_column: str | None) -> pd.DataFrame:
    raw = _read_table(path)
    frame = _rename_alias_columns(raw.copy())

    if "timestamp" not in frame.columns:
        frame["timestamp"] = _coerce_timestamp(raw)
    frame["ts"] = pd.to_numeric(frame["timestamp"], errors="coerce")

    if "src_ip" not in frame.columns or "dst_ip" not in frame.columns or "proto" not in frame.columns:
        missing = [name for name in ("src_ip", "dst_ip", "proto") if name not in frame.columns]
        raise ValueError(f"UGR'16 input is missing required columns after normalization: {missing}")

    frame["duration"] = pd.to_numeric(frame.get("duration", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["orig_bytes"] = pd.to_numeric(frame.get("orig_bytes", frame.get("bytes_sent", 0.0)), errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["resp_bytes"] = pd.to_numeric(frame.get("resp_bytes", frame.get("bytes_received", 0.0)), errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["orig_pkts"] = pd.to_numeric(frame.get("orig_pkts", 1.0), errors="coerce").fillna(1.0).clip(lower=1.0)
    frame["resp_pkts"] = pd.to_numeric(frame.get("resp_pkts", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["src_port"] = pd.to_numeric(frame.get("src_port", frame.get("src_port_zeek", 0.0)), errors="coerce").fillna(0.0)
    frame["dst_port"] = pd.to_numeric(frame.get("dst_port", frame.get("dest_port_zeek", 0.0)), errors="coerce").fillna(0.0)

    total_pkts = frame["orig_pkts"] + frame["resp_pkts"]
    total_bytes = frame["orig_bytes"] + frame["resp_bytes"]
    frame["bytes_per_pkt"] = total_bytes / (total_pkts + 1e-9)
    frame["packet_ratio"] = frame["orig_pkts"] / (frame["resp_pkts"] + 1e-9)
    frame["byte_ratio"] = frame["orig_bytes"] / (frame["resp_bytes"] + 1e-9)

    frame["src_ip"] = frame["src_ip"].astype(str)
    frame["dst_ip"] = frame["dst_ip"].astype(str)
    frame["proto"] = frame["proto"].astype(str).str.lower()

    label_values = _map_labels(raw, label_column)
    frame["label"] = label_values.to_numpy(dtype=np.int64)
    if family_column and family_column in raw.columns:
        frame["family"] = raw[family_column].astype(str)
    else:
        frame["family"] = np.where(frame["label"] == 1, "ugr16", "")

    frame["source_dataset"] = "ugr16"
    frame["capture_id"] = Path(path).stem
    frame["source_path"] = path
    frame["host_id"] = frame.get("host_id", frame["src_ip"]).astype(str)

    frame = frame.sort_values(["host_id", "dst_ip", "proto", "timestamp"], kind="mergesort").reset_index(drop=True)
    groups = frame.groupby(["host_id", "dst_ip", "proto"], sort=False)
    frame["iat"] = groups["timestamp"].diff().fillna(0.0).clip(lower=0.0)
    frame["iat_delta"] = groups["iat"].diff().fillna(0.0)
    frame["byte_delta"] = groups["orig_bytes"].diff().fillna(0.0)

    return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_ugr16_dataset(
    input_path: str,
    out_dir: str,
    *,
    label_column: str = "Label",
    family_column: str | None = None,
) -> dict:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"UGR'16 input file not found: {input_file}")

    frame = _prepare_ugr16_frame(str(input_file), label_column=label_column, family_column=family_column)
    manifest = build_extended_dataset_from_dataframe(
        frame,
        out_dir,
        source_dataset="ugr16",
        min_flows=1,
        inactivity_timeout=300.0,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an extended_v1 UGR'16 dataset from a labeled tabular file")
    parser.add_argument("--input", required=True, help="Path to the UGR'16 CSV / TSV / Parquet file")
    parser.add_argument(
        "--out-dir",
        default="data/processed/ugr16_extended",
        help="Output directory for sessions_extended.npz and dataset_manifest.json",
    )
    parser.add_argument(
        "--label-column",
        default="Label",
        help="Column that contains the benign/attack label values",
    )
    parser.add_argument(
        "--family-column",
        default=None,
        help="Optional column name with a family / technique label for C2 rows",
    )
    args = parser.parse_args()

    manifest = build_ugr16_dataset(
        args.input,
        args.out_dir,
        label_column=args.label_column,
        family_column=args.family_column,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()