"""TLS behavioral feature extraction from Zeek ssl.log data."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


class TLSFeatureExtractor:
    """Extract numeric TLS behavior features from Zeek SSL/TLS records."""

    TLS_FEATURES = [
        "tls_version",
        "tls_cipher_count",
        "tls_extension_count",
        "tls_supported_groups",
        "tls_alpn_count",
        "tls_session_reuse_rate",
        "tls_cert_reuse_rate",
        "tls_handshake_size",
        "tls_version_diversity",
        "tls_cipher_diversity",
        "tls_cert_validity_days",
        "tls_ja3_hash",
    ]

    _TLS_VERSION_MAP = {
        "ssl": 0.1,
        "sslv2": 0.2,
        "sslv3": 0.3,
        "tlsv10": 1.0,
        "tlsv1": 1.0,
        "tlsv1.0": 1.0,
        "tlsv11": 1.1,
        "tlsv1.1": 1.1,
        "tlsv12": 1.2,
        "tlsv1.2": 1.2,
        "tlsv13": 1.3,
        "tlsv1.3": 1.3,
    }

    def extract_from_zeek_ssl_log(self, ssl_df: pd.DataFrame | None) -> dict[str, float]:
        """Return all TLS feature values for a Zeek ssl.log-like dataframe."""
        if ssl_df is None or len(ssl_df) == 0:
            return self.empty_features()

        df = ssl_df.copy()
        features = self.empty_features()

        versions = self._clean_series(df.get("version"))
        ciphers = self._clean_series(df.get("cipher"))
        alpn = self._clean_series(df.get("next_protocol", df.get("alpn")))
        ja3 = self._clean_series(df.get("ja3", df.get("ja")))

        features["tls_version"] = self._mode_tls_version(versions)
        features["tls_version_diversity"] = float(versions.nunique())
        features["tls_cipher_count"] = float(ciphers.nunique())
        features["tls_cipher_diversity"] = float(ciphers.nunique() / max(len(df), 1))
        features["tls_extension_count"] = float(self._extract_extension_count(df, ja3))
        features["tls_supported_groups"] = float(self._extract_supported_groups(df))
        features["tls_alpn_count"] = float(alpn.nunique())
        features["tls_session_reuse_rate"] = self._compute_session_reuse_rate(df)
        features["tls_cert_reuse_rate"] = self._compute_cert_reuse_rate(df)
        features["tls_handshake_size"] = self._compute_handshake_size(df)
        features["tls_cert_validity_days"] = self._compute_cert_validity_days(df)
        features["tls_ja3_hash"] = self._hash_to_unit_interval(self._mode_text(ja3))

        return self._finite_features(features)

    def to_feature_vector(self, features: dict[str, float]) -> np.ndarray:
        """Convert a TLS feature dict to the canonical feature order."""
        return np.array([float(features.get(name, 0.0)) for name in self.TLS_FEATURES], dtype=np.float32)

    def empty_features(self) -> dict[str, float]:
        return {name: 0.0 for name in self.TLS_FEATURES}

    def _clean_series(self, series: pd.Series | None) -> pd.Series:
        if series is None:
            return pd.Series([], dtype=object)
        return series.replace("-", np.nan).dropna()

    def _mode_text(self, series: pd.Series) -> str:
        if series.empty:
            return ""
        modes = series.astype(str).mode()
        return str(modes.iloc[0]) if len(modes) else ""

    def _mode_tls_version(self, versions: pd.Series) -> float:
        mode = self._mode_text(versions).lower().replace("_", "").replace("-", "").replace(" ", "")
        return float(self._TLS_VERSION_MAP.get(mode, 0.0))

    def _extract_extension_count(self, df: pd.DataFrame, ja3: pd.Series) -> int:
        for column in ("extensions", "tls_extensions", "extension_list"):
            if column in df:
                values = self._clean_series(df[column])
                counts = [len(self._split_list_like(value)) for value in values]
                return int(round(float(np.mean(counts)))) if counts else 0

        # JA3 format: version,ciphers,extensions,elliptic_curves,point_formats
        counts = []
        for fingerprint in ja3.astype(str):
            parts = fingerprint.split(",")
            if len(parts) >= 3:
                counts.append(len([x for x in parts[2].split("-") if x]))
        return int(round(float(np.mean(counts)))) if counts else int(ja3.nunique())

    def _extract_supported_groups(self, df: pd.DataFrame) -> int:
        for column in ("supported_groups", "elliptic_curves", "curves"):
            if column in df:
                values = self._clean_series(df[column])
                groups: set[str] = set()
                for value in values:
                    groups.update(self._split_list_like(value))
                return len(groups)
        return 0

    def _compute_session_reuse_rate(self, df: pd.DataFrame) -> float:
        for boolean_col in ("resumed", "session_resumed"):
            if boolean_col in df:
                values = df[boolean_col].dropna()
                if len(values):
                    return float(values.astype(str).str.lower().isin({"true", "t", "1", "yes"}).mean())

        for id_col in ("session_id", "uid"):
            if id_col in df:
                ids = self._clean_series(df[id_col])
                if len(ids) >= 2:
                    reused = len(ids) - ids.astype(str).nunique()
                    return float(reused / len(ids))
        return 0.0

    def _compute_cert_reuse_rate(self, df: pd.DataFrame) -> float:
        for column in ("cert_chain_fuids", "server_cert_chain_fuids", "cert_hash", "server_cert_hash"):
            if column in df:
                certs = self._clean_series(df[column]).astype(str)
                if len(certs) >= 2:
                    reused = len(certs) - certs.nunique()
                    return float(reused / len(certs))
        return 0.0

    def _compute_handshake_size(self, df: pd.DataFrame) -> float:
        for column in ("handshake_size", "ssl_handshake_size", "request_bytes", "response_bytes"):
            if column in df:
                values = pd.to_numeric(df[column], errors="coerce").dropna()
                if len(values):
                    return float(values.mean())

        # Fallback proxy: longer cipher/SNI/cert metadata usually means richer handshakes.
        text_columns = [column for column in ("cipher", "server_name", "cert_chain_fuids") if column in df]
        if not text_columns:
            return 0.0
        lengths = []
        for column in text_columns:
            lengths.extend(df[column].dropna().astype(str).str.len().tolist())
        return float(np.mean(lengths)) if lengths else 0.0

    def _compute_cert_validity_days(self, df: pd.DataFrame) -> float:
        if "cert_validity_days" in df:
            values = pd.to_numeric(df["cert_validity_days"], errors="coerce").dropna()
            if len(values):
                return float(values.mean())

        start_col = self._first_present(df, ("cert_not_valid_before", "not_valid_before"))
        end_col = self._first_present(df, ("cert_not_valid_after", "not_valid_after"))
        if start_col and end_col:
            starts = pd.to_datetime(df[start_col], errors="coerce", utc=True)
            ends = pd.to_datetime(df[end_col], errors="coerce", utc=True)
            days = (ends - starts).dt.total_seconds() / 86400.0
            days = days.replace([np.inf, -np.inf], np.nan).dropna()
            if len(days):
                return float(days.mean())

        return 0.0

    def _first_present(self, df: pd.DataFrame, names: Iterable[str]) -> str | None:
        return next((name for name in names if name in df), None)

    def _split_list_like(self, value: object) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item)]
        text = str(value)
        return [part for part in re.split(r"[,;|\-\s]+", text) if part]

    def _hash_to_unit_interval(self, value: str) -> float:
        if not value:
            return 0.0
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)

    def _finite_features(self, features: dict[str, float]) -> dict[str, float]:
        out = {}
        for key in self.TLS_FEATURES:
            value = float(features.get(key, 0.0))
            out[key] = value if math.isfinite(value) else 0.0
        return out
