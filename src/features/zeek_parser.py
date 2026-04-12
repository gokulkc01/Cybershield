"""Zeek conn.log to feature DataFrame conversion."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.features.feature_config import FEATURE_DIM, FEATURE_NAMES, ZEEK_REQUIRED_FIELDS, is_private_ip


def get_zeek_headers(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#fields"):
                return line.rstrip("\n").split("\t")[1:]
    raise ValueError(f"No '#fields' line found in {file_path}")


def load_zeek_conn_log(file_path: str) -> pd.DataFrame:
    headers = get_zeek_headers(file_path)
    return pd.read_csv(
        file_path,
        sep="\t",
        names=headers,
        comment="#",
        low_memory=False,
    )


def preprocess_zeek(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ZEEK_REQUIRED_FIELDS if column not in df.columns]
    if missing:
        raise ValueError(f"conn.log is missing required fields: {missing}")

    df = df.replace("-", np.nan).copy()

    numeric_cols = ["ts", "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["duration"] = df["duration"].clip(lower=0.0)
    df["orig_bytes"] = df["orig_bytes"].clip(lower=0.0)
    df["resp_bytes"] = df["resp_bytes"].clip(lower=0.0)
    df["orig_pkts"] = df["orig_pkts"].clip(lower=0.0)
    df["resp_pkts"] = df["resp_pkts"].clip(lower=0.0)

    df = df[(df["orig_pkts"] + df["resp_pkts"]) > 0].reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(
        by=["id.orig_h", "id.resp_h", "proto", "ts"],
        kind="mergesort",
    ).reset_index(drop=True)

    total_pkts = df["orig_pkts"] + df["resp_pkts"]
    total_bytes = df["orig_bytes"] + df["resp_bytes"]

    df["bytes_per_pkt"] = total_bytes / (total_pkts + 1e-9)
    df["packet_ratio"] = df["orig_pkts"] / (df["resp_pkts"] + 1e-9)
    df["byte_ratio"] = df["orig_bytes"] / (df["resp_bytes"] + 1e-9)
    src_is_private = df["id.orig_h"].apply(is_private_ip).astype(bool)
    dst_is_private = df["id.resp_h"].apply(is_private_ip).astype(bool)
    df["is_outbound"] = (src_is_private & ~dst_is_private).astype(float)

    group = df.groupby(["id.orig_h", "id.resp_h", "proto"], sort=False)
    df["iat"] = group["ts"].diff().fillna(0.0).clip(lower=0.0)
    df["iat_delta"] = group["iat"].diff().fillna(0.0)
    df["byte_delta"] = group["orig_bytes"].diff().fillna(0.0)

    df = df.rename(
        columns={
            "id.orig_h": "src_ip",
            "id.resp_h": "dst_ip",
            "id.orig_p": "src_port",
            "id.resp_p": "dst_port",
        }
    )
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_NAMES if column not in df.columns]
    if missing:
        raise KeyError(f"Missing engineered features: {missing}")

    feature_df = df[FEATURE_NAMES].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_df["orig_bytes"] = feature_df["orig_bytes"].clip(upper=1e9)
    feature_df["resp_bytes"] = feature_df["resp_bytes"].clip(upper=1e9)
    feature_df["bytes_per_pkt"] = feature_df["bytes_per_pkt"].clip(upper=65535)
    feature_df["packet_ratio"] = feature_df["packet_ratio"].clip(upper=1000)
    feature_df["byte_ratio"] = feature_df["byte_ratio"].clip(upper=1000)
    feature_df["duration"] = feature_df["duration"].clip(upper=3600)
    feature_df["src_port"] = feature_df["src_port"].clip(lower=0, upper=65535)
    feature_df["dst_port"] = feature_df["dst_port"].clip(lower=0, upper=65535)
    feature_df["iat"] = feature_df["iat"].clip(lower=0.0, upper=3600)

    assert feature_df.shape[1] == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {feature_df.shape[1]}"
    return feature_df


def process_conn_log(file_path: str, return_metadata: bool = False) -> pd.DataFrame:
    df = load_zeek_conn_log(file_path)
    df = preprocess_zeek(df)
    df = engineer_features(df)

    if return_metadata:
        metadata_columns = ["ts", "src_ip", "dst_ip", "proto"]
        metadata = df[[column for column in metadata_columns if column in df.columns]].reset_index(drop=True)
        features = build_feature_matrix(df).reset_index(drop=True)
        return pd.concat([metadata, features], axis=1)

    return build_feature_matrix(df)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.features.zeek_parser <path/to/conn.log>")
        raise SystemExit(1)

    frame = process_conn_log(sys.argv[1], return_metadata=True)
    print(f"Output shape: {frame.shape}")
    print(frame.head(3).to_string())
