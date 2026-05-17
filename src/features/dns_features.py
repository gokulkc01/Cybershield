"""DNS behavioral feature extraction from Zeek dns.log data."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


class DNSFeatureExtractor:
    """Extract numeric DNS behavior features from Zeek DNS records."""

    DNS_FEATURES = [
        "dns_query_count",
        "dns_unique_domains",
        "dns_nxdomain_rate",
        "dns_dga_score",
        "dns_domain_age_avg",
        "dns_ttl_consistency",
        "dns_query_type_entropy",
        "dns_lookup_frequency",
        "dns_rapid_fire_rate",
        "dns_subdomain_depth_avg",
    ]

    _COMMON_TLDS = {"com", "org", "net", "edu", "gov", "co", "uk", "io", "ai"}
    _COMMON_DOMAINS = {
        "google.com",
        "microsoft.com",
        "amazon.com",
        "apple.com",
        "cloudflare.com",
        "akamai.com",
        "github.com",
        "facebook.com",
    }

    def extract_from_zeek_dns_log(self, dns_df: pd.DataFrame | None) -> dict[str, float]:
        """Return all DNS feature values for a Zeek dns.log-like dataframe."""
        if dns_df is None or len(dns_df) == 0:
            return self.empty_features()

        df = dns_df.copy()
        features = self.empty_features()
        queries = self._clean_domains(df.get("query"))

        features["dns_query_count"] = float(len(df))
        features["dns_unique_domains"] = float(len(set(queries)))
        features["dns_nxdomain_rate"] = self._compute_nxdomain_rate(df)
        features["dns_dga_score"] = self._compute_dga_score(queries)
        features["dns_domain_age_avg"] = self._estimate_domain_age(df, queries)
        features["dns_ttl_consistency"] = self._compute_ttl_consistency(df)
        features["dns_query_type_entropy"] = self._compute_query_type_entropy(df)
        features["dns_lookup_frequency"] = self._compute_lookup_frequency(df)
        features["dns_rapid_fire_rate"] = self._compute_rapid_fire_rate(df)
        features["dns_subdomain_depth_avg"] = self._compute_subdomain_depth(queries)

        return self._finite_features(features)

    def to_feature_vector(self, features: dict[str, float]) -> np.ndarray:
        """Convert a DNS feature dict to the canonical feature order."""
        return np.array([float(features.get(name, 0.0)) for name in self.DNS_FEATURES], dtype=np.float32)

    def empty_features(self) -> dict[str, float]:
        return {name: 0.0 for name in self.DNS_FEATURES}

    def _clean_domains(self, series: pd.Series | None) -> list[str]:
        if series is None:
            return []
        return [
            str(value).strip(".").lower()
            for value in series.replace("-", np.nan).dropna().tolist()
            if str(value).strip(".")
        ]

    def _compute_nxdomain_rate(self, df: pd.DataFrame) -> float:
        if "rcode" not in df and "rcode_name" not in df:
            return 0.0
        values = df.get("rcode_name", df.get("rcode")).dropna()
        if len(values) == 0:
            return 0.0
        text_values = values.astype(str).str.upper()
        numeric_values = pd.to_numeric(values, errors="coerce")
        hits = (text_values == "NXDOMAIN") | (numeric_values == 3)
        return float(hits.mean())

    def _compute_dga_score(self, domains: Iterable[str]) -> float:
        scores = [self._single_domain_dga_score(domain) for domain in domains]
        return float(np.mean(scores)) if scores else 0.0

    def _single_domain_dga_score(self, domain: str) -> float:
        labels = [part for part in domain.split(".") if part]
        if not labels:
            return 0.0
        sld = labels[-2] if len(labels) >= 2 else labels[0]
        tld = labels[-1]

        entropy_score = min(1.0, self._entropy(sld) / math.log2(36))
        length_score = min(1.0, max(0, len(sld) - 12) / 20.0)
        digit_score = sum(ch.isdigit() for ch in sld) / max(len(sld), 1)
        vowel_ratio = sum(ch in "aeiou" for ch in sld.lower()) / max(sum(ch.isalpha() for ch in sld), 1)
        vowel_anomaly = abs(vowel_ratio - 0.38) / 0.38
        consonant_runs = len(re.findall(r"[^aeiou\W\d_]{4,}", sld.lower()))
        uncommon_tld = 1.0 if tld not in self._COMMON_TLDS else 0.0

        score = (
            0.30 * entropy_score
            + 0.20 * length_score
            + 0.15 * digit_score
            + 0.15 * min(1.0, vowel_anomaly)
            + 0.10 * min(1.0, consonant_runs / 2.0)
            + 0.10 * uncommon_tld
        )
        return float(np.clip(score, 0.0, 1.0))

    def _entropy(self, text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        entropy = 0.0
        for count in counts.values():
            p = count / len(text)
            entropy -= p * math.log2(p)
        return float(entropy)

    def _estimate_domain_age(self, df: pd.DataFrame, domains: list[str]) -> float:
        if "domain_age_days" in df:
            values = pd.to_numeric(df["domain_age_days"], errors="coerce").dropna()
            if len(values):
                return float(values.mean())
        ages = [365.0 if self._registered_domain(domain) in self._COMMON_DOMAINS else 30.0 for domain in domains]
        return float(np.mean(ages)) if ages else 0.0

    def _compute_ttl_consistency(self, df: pd.DataFrame) -> float:
        if "ttls" not in df and "TTL" not in df:
            return 0.0
        column = "ttls" if "ttls" in df else "TTL"
        ttls: list[float] = []
        for value in df[column].dropna():
            ttls.extend(self._parse_numeric_list(value))
        return float(np.std(ttls)) if ttls else 0.0

    def _compute_query_type_entropy(self, df: pd.DataFrame) -> float:
        column = "qtype_name" if "qtype_name" in df else "qtype"
        if column not in df:
            return 0.0
        values = df[column].dropna().astype(str).tolist()
        return self._categorical_entropy(values)

    def _compute_lookup_frequency(self, df: pd.DataFrame) -> float:
        if "ts" not in df or len(df) < 2:
            return 0.0
        seconds = self._timestamp_span_seconds(df["ts"])
        minutes = max(seconds / 60.0, 1.0)
        return float(len(df) / minutes)

    def _compute_rapid_fire_rate(self, df: pd.DataFrame) -> float:
        if "ts" not in df or len(df) < 2:
            return 0.0
        times = np.sort(self._timestamps_to_seconds(df["ts"]))
        if len(times) < 2:
            return 0.0
        diffs = np.diff(times)
        return float(np.mean(diffs <= 1.0))

    def _compute_subdomain_depth(self, domains: Iterable[str]) -> float:
        depths = []
        for domain in domains:
            labels = [part for part in domain.split(".") if part]
            depths.append(max(0, len(labels) - 2))
        return float(np.mean(depths)) if depths else 0.0

    def _registered_domain(self, domain: str) -> str:
        labels = [part for part in domain.split(".") if part]
        return ".".join(labels[-2:]) if len(labels) >= 2 else domain

    def _parse_numeric_list(self, value: object) -> list[float]:
        if isinstance(value, (list, tuple, set, np.ndarray)):
            raw = value
        else:
            raw = re.split(r"[,;|\s]+", str(value).strip("[]"))
        numbers = []
        for item in raw:
            try:
                numbers.append(float(item))
            except (TypeError, ValueError):
                continue
        return numbers

    def _categorical_entropy(self, values: list[str]) -> float:
        if not values:
            return 0.0
        counts = Counter(values)
        entropy = 0.0
        for count in counts.values():
            p = count / len(values)
            entropy -= p * math.log2(p)
        return float(entropy)

    def _timestamp_span_seconds(self, series: pd.Series) -> float:
        values = self._timestamps_to_seconds(series)
        if len(values) < 2:
            return 0.0
        return float(np.nanmax(values) - np.nanmin(values))

    def _timestamps_to_seconds(self, series: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return numeric.dropna().to_numpy(dtype=float)
        datetimes = pd.to_datetime(series, errors="coerce", utc=True).dropna()
        return (datetimes.astype("int64") / 1e9).to_numpy(dtype=float)

    def _finite_features(self, features: dict[str, float]) -> dict[str, float]:
        out = {}
        for key in self.DNS_FEATURES:
            value = float(features.get(key, 0.0))
            out[key] = value if math.isfinite(value) else 0.0
        return out
