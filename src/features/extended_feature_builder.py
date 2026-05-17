"""Feature assembly for the extended_v1 behavioral schema.

The builder combines base flow/session behavior with optional TLS and DNS
context plus derived temporal statistics into one stable 45-dimensional vector.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.features.dns_features import DNSFeatureExtractor
from src.features.feature_config_extended import (
    BASE_FEATURE_NAMES_EXTENDED,
    DNS_FEATURE_NAMES_EXTENDED,
    FEATURE_DIM_EXTENDED,
    FEATURE_NAMES_EXTENDED,
    FEATURE_SCHEMA_VERSION,
    TEMPORAL_FEATURE_NAMES_EXTENDED,
    TLS_FEATURE_NAMES_EXTENDED,
)
from src.features.feature_config_experiment import FEATURE_INDEX as BASE_FEATURE_INDEX
from src.features.temporal_features import TemporalStatsExtractor
from src.features.tls_features import TLSFeatureExtractor


class ExtendedFeatureBuilder:
    """Assemble one extended_v1 feature vector from available session context."""

    schema_version = FEATURE_SCHEMA_VERSION
    feature_names = FEATURE_NAMES_EXTENDED
    feature_dim = FEATURE_DIM_EXTENDED

    _EPSILON = 1e-9
    _ALIASES = {
        "orig_bytes": ("orig_bytes", "bytes_in", "src_bytes", "origbytes", "bytes_sent", "totbytes_src", "SrcBytes"),
        "resp_bytes": ("resp_bytes", "bytes_out", "dst_bytes", "respbytes", "bytes_recv", "totbytes_dst", "DstBytes"),
        "orig_pkts": ("orig_pkts", "packets_in", "src_pkts", "origpkts", "packets_sent", "totpkts_src", "SrcPkts"),
        "resp_pkts": ("resp_pkts", "packets_out", "dst_pkts", "resppkts", "packets_recv", "totpkts_dst", "DstPkts"),
        "bytes_per_pkt": ("bytes_per_pkt", "bytes_per_packet", "bpp"),
        "packet_ratio": ("packet_ratio", "pkt_ratio", "packets_ratio"),
        "byte_ratio": ("byte_ratio", "bytes_ratio"),
        "is_outbound": ("is_outbound", "outbound"),
        "duration": ("duration", "dur", "flow_duration", "elapsed"),
        "iat": ("iat", "inter_arrival", "interarrival", "inter_arrival_time"),
        "ts": ("ts", "timestamp", "time", "start_time", "starttime"),
        "timestamps": ("timestamps", "timestamp", "ts", "times"),
    }

    def __init__(
        self,
        *,
        tls_extractor: TLSFeatureExtractor | None = None,
        dns_extractor: DNSFeatureExtractor | None = None,
        temporal_extractor: TemporalStatsExtractor | None = None,
    ) -> None:
        self.tls_extractor = tls_extractor or TLSFeatureExtractor()
        self.dns_extractor = dns_extractor or DNSFeatureExtractor()
        self.temporal_extractor = temporal_extractor or TemporalStatsExtractor()

    def build_feature_vector(
        self,
        session: Mapping[str, Any] | pd.Series | pd.DataFrame | np.ndarray | None,
        *,
        tls_context: Any | None = None,
        dns_context: Any | None = None,
    ) -> np.ndarray:
        """Return an ordered ``extended_v1`` vector with shape ``(45,)``."""
        features = self.build_feature_dict(session, tls_context=tls_context, dns_context=dns_context)
        vector = np.array([features[name] for name in FEATURE_NAMES_EXTENDED], dtype=np.float32)
        return self._finite_array(vector)

    def build_feature_dict(
        self,
        session: Mapping[str, Any] | pd.Series | pd.DataFrame | np.ndarray | None,
        *,
        tls_context: Any | None = None,
        dns_context: Any | None = None,
    ) -> dict[str, float]:
        """Return all ``extended_v1`` features keyed by canonical feature name."""
        merged: dict[str, float] = {}
        merged.update(self.extract_base_features(session))
        merged.update(self._extract_tls_features(tls_context))
        merged.update(self._extract_dns_features(dns_context))
        merged.update(self._extract_temporal_features(session))
        return {name: self._finite_float(merged.get(name, 0.0)) for name in FEATURE_NAMES_EXTENDED}

    def extract_base_features(
        self,
        session: Mapping[str, Any] | pd.Series | pd.DataFrame | np.ndarray | None,
    ) -> dict[str, float]:
        """Extract the 10 base session behavior features."""
        if session is None:
            return self._empty_group(BASE_FEATURE_NAMES_EXTENDED)
        if isinstance(session, pd.DataFrame):
            return self._base_from_dataframe(session)
        if isinstance(session, pd.Series):
            return self._base_from_mapping(session.to_dict())
        if isinstance(session, Mapping):
            return self._base_from_mapping(session)

        try:
            return self._base_from_array(np.asarray(session, dtype=np.float64))
        except (TypeError, ValueError):
            return self._empty_group(BASE_FEATURE_NAMES_EXTENDED)

    def _base_from_mapping(self, session: Mapping[str, Any]) -> dict[str, float]:
        orig_bytes = self._mapping_numeric(session, "orig_bytes", aggregate="sum")
        resp_bytes = self._mapping_numeric(session, "resp_bytes", aggregate="sum")
        orig_pkts = self._mapping_numeric(session, "orig_pkts", aggregate="sum")
        resp_pkts = self._mapping_numeric(session, "resp_pkts", aggregate="sum")
        duration = self._mapping_numeric(session, "duration", aggregate="sum")
        is_outbound = self._mapping_numeric(session, "is_outbound", aggregate="mean")
        iat = self._mapping_numeric(session, "iat", aggregate="mean")

        bytes_per_pkt = self._mapping_numeric(session, "bytes_per_pkt", aggregate="mean", default=np.nan)
        packet_ratio = self._mapping_numeric(session, "packet_ratio", aggregate="mean", default=np.nan)
        byte_ratio = self._mapping_numeric(session, "byte_ratio", aggregate="mean", default=np.nan)

        total_bytes = orig_bytes + resp_bytes
        total_pkts = orig_pkts + resp_pkts
        if not math.isfinite(bytes_per_pkt):
            bytes_per_pkt = self._safe_ratio(total_bytes, total_pkts)
        if not math.isfinite(packet_ratio):
            packet_ratio = self._safe_ratio(orig_pkts, resp_pkts)
        if not math.isfinite(byte_ratio):
            byte_ratio = self._safe_ratio(orig_bytes, resp_bytes)

        return self._ordered_base(
            {
                "orig_bytes": orig_bytes,
                "resp_bytes": resp_bytes,
                "orig_pkts": orig_pkts,
                "resp_pkts": resp_pkts,
                "bytes_per_pkt": bytes_per_pkt,
                "packet_ratio": packet_ratio,
                "byte_ratio": byte_ratio,
                "is_outbound": is_outbound,
                "duration": duration,
                "iat": iat,
            }
        )

    def _base_from_dataframe(self, session_df: pd.DataFrame) -> dict[str, float]:
        if session_df.empty:
            return self._empty_group(BASE_FEATURE_NAMES_EXTENDED)

        orig_bytes = self._dataframe_numeric(session_df, "orig_bytes", aggregate="sum")
        resp_bytes = self._dataframe_numeric(session_df, "resp_bytes", aggregate="sum")
        orig_pkts = self._dataframe_numeric(session_df, "orig_pkts", aggregate="sum")
        resp_pkts = self._dataframe_numeric(session_df, "resp_pkts", aggregate="sum")
        duration = self._dataframe_numeric(session_df, "duration", aggregate="sum")
        is_outbound = self._dataframe_numeric(session_df, "is_outbound", aggregate="mean")
        iat = self._dataframe_iat_mean(session_df)

        total_bytes = orig_bytes + resp_bytes
        total_pkts = orig_pkts + resp_pkts

        bytes_per_pkt = self._safe_ratio(total_bytes, total_pkts)
        packet_ratio = self._safe_ratio(orig_pkts, resp_pkts)
        byte_ratio = self._safe_ratio(orig_bytes, resp_bytes)

        if total_pkts <= 0:
            bytes_per_pkt = self._dataframe_numeric(session_df, "bytes_per_pkt", aggregate="mean")
        if resp_pkts <= 0:
            packet_ratio = self._dataframe_numeric(session_df, "packet_ratio", aggregate="mean")
        if resp_bytes <= 0:
            byte_ratio = self._dataframe_numeric(session_df, "byte_ratio", aggregate="mean")

        return self._ordered_base(
            {
                "orig_bytes": orig_bytes,
                "resp_bytes": resp_bytes,
                "orig_pkts": orig_pkts,
                "resp_pkts": resp_pkts,
                "bytes_per_pkt": bytes_per_pkt,
                "packet_ratio": packet_ratio,
                "byte_ratio": byte_ratio,
                "is_outbound": is_outbound,
                "duration": duration,
                "iat": iat,
            }
        )

    def _base_from_array(self, session_array: np.ndarray) -> dict[str, float]:
        if session_array.ndim == 0:
            return self._empty_group(BASE_FEATURE_NAMES_EXTENDED)
        if session_array.ndim == 1:
            values = {
                name: self._finite_float(session_array[idx]) if idx < session_array.size else 0.0
                for idx, name in enumerate(BASE_FEATURE_NAMES_EXTENDED)
            }
            return self._ordered_base(values)
        if session_array.ndim != 2:
            return self._empty_group(BASE_FEATURE_NAMES_EXTENDED)

        matrix = np.where(np.isfinite(session_array), session_array, 0.0)
        real = matrix[np.any(np.abs(matrix) > 0.0, axis=1)]
        if real.size == 0:
            return self._empty_group(BASE_FEATURE_NAMES_EXTENDED)

        def col(name: str) -> np.ndarray:
            idx = BASE_FEATURE_INDEX[name]
            if idx >= real.shape[1]:
                return np.array([], dtype=np.float64)
            return real[:, idx].astype(np.float64)

        orig_bytes = float(np.sum(col("orig_bytes")))
        resp_bytes = float(np.sum(col("resp_bytes")))
        orig_pkts = float(np.sum(col("orig_pkts")))
        resp_pkts = float(np.sum(col("resp_pkts")))
        duration = float(np.sum(col("duration")))
        is_outbound_values = col("is_outbound")
        iat_values = col("iat")

        return self._ordered_base(
            {
                "orig_bytes": orig_bytes,
                "resp_bytes": resp_bytes,
                "orig_pkts": orig_pkts,
                "resp_pkts": resp_pkts,
                "bytes_per_pkt": self._safe_ratio(orig_bytes + resp_bytes, orig_pkts + resp_pkts),
                "packet_ratio": self._safe_ratio(orig_pkts, resp_pkts),
                "byte_ratio": self._safe_ratio(orig_bytes, resp_bytes),
                "is_outbound": float(np.mean(is_outbound_values)) if is_outbound_values.size else 0.0,
                "duration": duration,
                "iat": float(np.mean(iat_values)) if iat_values.size else 0.0,
            }
        )

    def _extract_tls_features(self, tls_context: Any | None) -> dict[str, float]:
        if tls_context is None:
            return self.tls_extractor.empty_features()
        tls_df = self._context_to_dataframe(tls_context)
        return self.tls_extractor.extract_from_zeek_ssl_log(tls_df)

    def _extract_dns_features(self, dns_context: Any | None) -> dict[str, float]:
        if dns_context is None:
            return self.dns_extractor.empty_features()
        dns_df = self._context_to_dataframe(dns_context)
        return self.dns_extractor.extract_from_zeek_dns_log(dns_df)

    def _extract_temporal_features(
        self,
        session: Mapping[str, Any] | pd.Series | pd.DataFrame | np.ndarray | None,
    ) -> dict[str, float]:
        if session is None:
            return self.temporal_extractor.empty_features()
        if isinstance(session, pd.DataFrame):
            if "iat" in session or "ts" in session:
                return self.temporal_extractor.extract_from_session(session)
            timestamp_col = self._resolve_column(session, self._ALIASES["timestamps"])
            if timestamp_col:
                temporal_df = session.rename(columns={timestamp_col: "ts"})
                return self.temporal_extractor.extract_from_session(temporal_df)
            return self.temporal_extractor.empty_features()
        if isinstance(session, pd.Series):
            return self._temporal_from_mapping(session.to_dict())
        if isinstance(session, Mapping):
            return self._temporal_from_mapping(session)

        try:
            return self._temporal_from_array(np.asarray(session, dtype=np.float64))
        except (TypeError, ValueError):
            return self.temporal_extractor.empty_features()

    def _temporal_from_mapping(self, session: Mapping[str, Any]) -> dict[str, float]:
        iat_value = self._lookup_mapping_value(session, self._ALIASES["iat"])
        if iat_value is not None:
            return self.temporal_extractor.extract_from_iat(self._numeric_array(iat_value))

        timestamps = self._lookup_mapping_value(session, self._ALIASES["timestamps"])
        if timestamps is not None:
            return self.temporal_extractor.extract_from_session({"timestamps": timestamps})

        return self.temporal_extractor.empty_features()

    def _temporal_from_array(self, session_array: np.ndarray) -> dict[str, float]:
        if session_array.ndim == 1:
            idx = BASE_FEATURE_INDEX["iat"]
            if idx < session_array.size:
                return self.temporal_extractor.extract_from_iat([session_array[idx]])
            return self.temporal_extractor.empty_features()
        if session_array.ndim != 2:
            return self.temporal_extractor.empty_features()

        idx = BASE_FEATURE_INDEX["iat"]
        if idx >= session_array.shape[1]:
            return self.temporal_extractor.empty_features()
        matrix = np.where(np.isfinite(session_array), session_array, 0.0)
        real = matrix[np.any(np.abs(matrix) > 0.0, axis=1)]
        if real.size == 0:
            return self.temporal_extractor.empty_features()
        return self.temporal_extractor.extract_from_iat(real[:, idx])

    def _context_to_dataframe(self, context: Any) -> pd.DataFrame | None:
        if context is None:
            return None
        if isinstance(context, pd.DataFrame):
            return context
        if isinstance(context, pd.Series):
            return context.to_frame().T
        if isinstance(context, Mapping):
            return pd.DataFrame([context])
        try:
            return pd.DataFrame(context)
        except (TypeError, ValueError):
            return None

    def _mapping_numeric(
        self,
        data: Mapping[str, Any],
        name: str,
        *,
        aggregate: str,
        default: float = 0.0,
    ) -> float:
        value = self._lookup_mapping_value(data, self._ALIASES[name])
        if value is None:
            return default
        return self._aggregate_numeric(value, aggregate=aggregate, default=default)

    def _dataframe_numeric(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        aggregate: str,
        default: float = 0.0,
    ) -> float:
        column = self._resolve_column(df, self._ALIASES[name])
        if column is None:
            return default
        return self._aggregate_numeric(df[column], aggregate=aggregate, default=default)

    def _dataframe_iat_mean(self, df: pd.DataFrame) -> float:
        iat_col = self._resolve_column(df, self._ALIASES["iat"])
        if iat_col is not None:
            return self._aggregate_numeric(df[iat_col], aggregate="mean")

        ts_col = self._resolve_column(df, self._ALIASES["timestamps"])
        if ts_col is None:
            return 0.0

        times = pd.to_numeric(df[ts_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if times.size < 2:
            return 0.0
        diffs = np.diff(np.sort(times))
        diffs = diffs[np.isfinite(diffs)]
        diffs = diffs[diffs >= 0.0]
        return float(np.mean(diffs)) if diffs.size else 0.0

    def _lookup_mapping_value(self, data: Mapping[str, Any], aliases: tuple[str, ...]) -> Any | None:
        for alias in aliases:
            if alias in data:
                return data[alias]
        normalized = {self._normalize_key(str(key)): key for key in data.keys()}
        for alias in aliases:
            key = normalized.get(self._normalize_key(alias))
            if key is not None:
                return data[key]
        return None

    def _resolve_column(self, df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            if alias in df.columns:
                return alias
        normalized = {self._normalize_key(str(column)): column for column in df.columns}
        for alias in aliases:
            column = normalized.get(self._normalize_key(alias))
            if column is not None:
                return str(column)
        return None

    def _aggregate_numeric(self, value: Any, *, aggregate: str, default: float = 0.0) -> float:
        values = self._numeric_array(value)
        if values.size == 0:
            return default
        if aggregate == "sum":
            return float(np.sum(values))
        if aggregate == "mean":
            return float(np.mean(values))
        raise ValueError(f"Unsupported aggregate: {aggregate}")

    def _numeric_array(self, value: Any) -> np.ndarray:
        if isinstance(value, pd.Series):
            raw = value
        elif isinstance(value, np.ndarray):
            raw = value.reshape(-1) if value.ndim else [value.item()]
        elif isinstance(value, (list, tuple, set)):
            raw = list(value)
        else:
            raw = [value]

        values = pd.to_numeric(pd.Series(raw), errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return np.array([], dtype=np.float64)
        return values.to_numpy(dtype=np.float64)

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        return self._finite_float(float(numerator) / (float(denominator) + self._EPSILON))

    def _ordered_base(self, values: Mapping[str, Any]) -> dict[str, float]:
        return {name: self._finite_float(values.get(name, 0.0)) for name in BASE_FEATURE_NAMES_EXTENDED}

    def _empty_group(self, names: list[str]) -> dict[str, float]:
        return {name: 0.0 for name in names}

    def _finite_array(self, vector: np.ndarray) -> np.ndarray:
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _finite_float(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0

    def _normalize_key(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def build_extended_feature_vector(
    session: Mapping[str, Any] | pd.Series | pd.DataFrame | np.ndarray | None,
    *,
    tls_context: Any | None = None,
    dns_context: Any | None = None,
) -> np.ndarray:
    """Convenience wrapper returning an ``extended_v1`` vector."""
    return ExtendedFeatureBuilder().build_feature_vector(
        session,
        tls_context=tls_context,
        dns_context=dns_context,
    )


def build_extended_feature_dict(
    session: Mapping[str, Any] | pd.Series | pd.DataFrame | np.ndarray | None,
    *,
    tls_context: Any | None = None,
    dns_context: Any | None = None,
) -> dict[str, float]:
    """Convenience wrapper returning ``extended_v1`` features by name."""
    return ExtendedFeatureBuilder().build_feature_dict(
        session,
        tls_context=tls_context,
        dns_context=dns_context,
    )


__all__ = [
    "ExtendedFeatureBuilder",
    "build_extended_feature_dict",
    "build_extended_feature_vector",
    "BASE_FEATURE_NAMES_EXTENDED",
    "TLS_FEATURE_NAMES_EXTENDED",
    "DNS_FEATURE_NAMES_EXTENDED",
    "TEMPORAL_FEATURE_NAMES_EXTENDED",
]
