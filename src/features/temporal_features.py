"""Advanced temporal behavior feature extraction."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


class TemporalStatsExtractor:
    """Extract timing, burst, and recurrence features from session telemetry."""

    TEMPORAL_FEATURES = [
        "iat_entropy",
        "iat_autocorr",
        "iat_p25",
        "iat_p50",
        "iat_p75",
        "iat_p90",
        "iat_mean",
        "iat_std",
        "burst_count",
        "burst_avg_size",
        "burst_avg_spacing",
        "periodicity_confidence",
        "recurrence_stability",
    ]

    def extract_from_session(self, session_data: Mapping | pd.DataFrame | None) -> dict[str, float]:
        """Extract temporal features from a mapping or dataframe."""
        if session_data is None:
            return self.empty_features()

        if isinstance(session_data, pd.DataFrame):
            iats = self._iats_from_dataframe(session_data)
        elif "iat" in session_data:
            iats = np.asarray(session_data["iat"], dtype=np.float64)
        elif "timestamps" in session_data:
            iats = self._iats_from_timestamps(session_data["timestamps"])
        else:
            return self.empty_features()

        return self.extract_from_iat(iats)

    def extract_from_iat(self, iats: np.ndarray | list[float]) -> dict[str, float]:
        """Extract temporal features from inter-arrival times."""
        values = np.asarray(iats, dtype=np.float64)
        values = values[np.isfinite(values)]
        values = values[values >= 0.0]
        if values.size == 0:
            return self.empty_features()

        if values.size > 2:
            cap = np.percentile(values, 99)
            cleaned = values[values <= cap]
            values = cleaned if cleaned.size else values

        features = self.empty_features()
        features["iat_entropy"] = self._compute_entropy(values)
        features["iat_autocorr"] = self._compute_autocorrelation(values, lag=min(5, max(1, values.size // 4)))
        features["iat_p25"] = float(np.percentile(values, 25))
        features["iat_p50"] = float(np.percentile(values, 50))
        features["iat_p75"] = float(np.percentile(values, 75))
        features["iat_p90"] = float(np.percentile(values, 90))
        features["iat_mean"] = float(np.mean(values))
        features["iat_std"] = float(np.std(values))

        bursts = self._detect_bursts(values)
        burst_sizes = [len(burst) for burst in bursts]
        features["burst_count"] = float(len(bursts))
        features["burst_avg_size"] = float(np.mean(burst_sizes)) if burst_sizes else 0.0
        features["burst_avg_spacing"] = self._compute_burst_spacing(values, bursts)
        features["periodicity_confidence"] = self._compute_periodicity_confidence(values)
        features["recurrence_stability"] = self._compute_recurrence_stability(values)

        return self._finite_features(features)

    def to_feature_vector(self, features: dict[str, float]) -> np.ndarray:
        """Convert a temporal feature dict to the canonical feature order."""
        return np.array([float(features.get(name, 0.0)) for name in self.TEMPORAL_FEATURES], dtype=np.float32)

    def empty_features(self) -> dict[str, float]:
        return {name: 0.0 for name in self.TEMPORAL_FEATURES}

    def _iats_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        if "iat" in df:
            return pd.to_numeric(df["iat"], errors="coerce").dropna().to_numpy(dtype=float)
        if "ts" in df:
            return self._iats_from_timestamps(df["ts"])
        return np.array([], dtype=float)

    def _iats_from_timestamps(self, timestamps: object) -> np.ndarray:
        series = pd.Series(timestamps)
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            values = np.sort(numeric.dropna().to_numpy(dtype=float))
        else:
            datetimes = pd.to_datetime(series, errors="coerce", utc=True).dropna()
            values = np.sort((datetimes.astype("int64") / 1e9).to_numpy(dtype=float))
        return np.diff(values) if values.size >= 2 else np.array([], dtype=float)

    def _compute_entropy(self, iats: np.ndarray) -> float:
        if iats.size <= 1 or np.allclose(iats.min(), iats.max()):
            return 0.0
        upper = max(float(iats.max()), 1e-6)
        lower = max(float(iats[iats > 0].min()) if np.any(iats > 0) else 1e-6, 1e-6)
        bins = np.logspace(np.log10(lower), np.log10(upper + 1e-6), 20)
        hist, _ = np.histogram(np.clip(iats, lower, None), bins=np.unique(bins))
        hist = hist[hist > 0]
        if hist.size == 0:
            return 0.0
        p = hist / hist.sum()
        return float(-np.sum(p * np.log2(p)))

    def _compute_autocorrelation(self, iats: np.ndarray, lag: int = 1) -> float:
        if iats.size <= lag or lag <= 0:
            return 0.0
        centered = iats - np.mean(iats)
        denom = np.sum(centered * centered)
        if denom <= 0:
            return 0.0
        corr = np.sum(centered[:-lag] * centered[lag:]) / denom
        return float(np.clip(corr, -1.0, 1.0))

    def _detect_bursts(self, iats: np.ndarray) -> list[np.ndarray]:
        if iats.size < 2:
            return []
        threshold = np.percentile(iats, 35)
        bursts: list[list[float]] = []
        current: list[float] = []
        for value in iats:
            if value <= threshold:
                current.append(float(value))
            else:
                if len(current) >= 2:
                    bursts.append(current)
                current = []
        if len(current) >= 2:
            bursts.append(current)
        return [np.asarray(burst, dtype=float) for burst in bursts]

    def _compute_burst_spacing(self, iats: np.ndarray, bursts: list[np.ndarray]) -> float:
        if len(bursts) < 2:
            return 0.0
        spacings = []
        cursor = 0
        burst_positions = []
        threshold = np.percentile(iats, 35)
        while cursor < iats.size:
            if iats[cursor] <= threshold:
                start = cursor
                while cursor < iats.size and iats[cursor] <= threshold:
                    cursor += 1
                if cursor - start >= 2:
                    burst_positions.append((start, cursor))
            else:
                cursor += 1
        for (_, prev_end), (next_start, _) in zip(burst_positions, burst_positions[1:]):
            if next_start > prev_end:
                spacings.append(float(np.sum(iats[prev_end:next_start])))
        return float(np.mean(spacings)) if spacings else 0.0

    def _compute_periodicity_confidence(self, iats: np.ndarray) -> float:
        mean = float(np.mean(iats))
        if mean <= 0:
            return 0.0
        cv = float(np.std(iats) / mean)
        return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

    def _compute_recurrence_stability(self, iats: np.ndarray) -> float:
        if float(np.std(iats)) <= 1e-12:
            return 1.0
        periodicity = self._compute_periodicity_confidence(iats)
        lag1 = self._compute_autocorrelation(iats, lag=1)
        autocorr_score = (lag1 + 1.0) / 2.0
        if iats.size <= 2:
            autocorr_score = periodicity
        return float(np.clip(0.7 * periodicity + 0.3 * autocorr_score, 0.0, 1.0))

    def _finite_features(self, features: dict[str, float]) -> dict[str, float]:
        out = {}
        for key in self.TEMPORAL_FEATURES:
            value = float(features.get(key, 0.0))
            out[key] = value if math.isfinite(value) else 0.0
        return out
