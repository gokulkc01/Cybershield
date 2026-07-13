"""Classic Fourier/autocorrelation beaconing detector.

Non-learned reference baseline for the classic C2 timing signal: periodic
callbacks ("beacons") from an infected host to its C2 server. This is the
signal family used by RITA-style threat-hunting tools; it needs no labels
and no training - only flow timing.

Granularity: scores are computed per (host, destination) pair, the natural
unit for beacon analysis, then broadcast to every session of that pair so
that evaluation stays session-level and comparable with the learned models.
Event times are reconstructed from session start timestamps plus the raw
per-flow inter-arrival-time feature, so a slow beacon that the session
segmenter breaks into many single-flow sessions is still visible as one
periodic event series.

The score in [0, 1] averages three components over the pair's inter-event
intervals:

- ``cv_score``: 1 / (1 + coefficient of variation) - near 1 for
  metronome-like intervals, ~0.5 for Poisson arrivals, lower for bursts.
- ``autocorr_score``: peak positive autocorrelation of the (z-normalised)
  interval sequence - catches repeating jittered patterns; a zero-variance
  interval sequence is perfectly periodic and scores 1.
- ``fourier_score``: peak spectral density ratio of the binned event-rate
  series (bin width adapted to the median interval) - the classic
  periodogram view of beaconing.

Pairs with fewer than ``min_events`` events score 0.0: a timing detector
cannot assert periodicity from a handful of flows. Coverage is therefore an
honest output of the method, reported alongside its metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeaconingConfig:
    min_events: int = 6
    max_lag: int = 32
    min_bins: int = 32
    max_bins: int = 4096
    iat_feature: str = "iat"


@dataclass(frozen=True)
class PairScore:
    score: float
    cv_score: float
    autocorr_score: float
    fourier_score: float
    n_events: int


def _cv_score(intervals: np.ndarray) -> float:
    mean = float(np.mean(intervals))
    if mean <= 0.0:
        return 0.0
    cv = float(np.std(intervals)) / mean
    return 1.0 / (1.0 + cv)


def _autocorr_score(intervals: np.ndarray, max_lag: int) -> float:
    if float(np.mean(intervals)) <= 0.0:
        # All events simultaneous: a burst, not a beacon.
        return 0.0
    std = float(np.std(intervals))
    if std < 1e-9:
        # Zero jitter: perfectly periodic.
        return 1.0
    z = (intervals - float(np.mean(intervals))) / std
    n = len(z)
    best = 0.0
    for lag in range(1, min(max_lag, n - 2) + 1):
        acf = float(np.dot(z[:-lag], z[lag:])) / n
        best = max(best, acf)
    return min(best, 1.0)


def _fourier_score(event_times: np.ndarray, config: BeaconingConfig) -> float:
    span = float(event_times[-1] - event_times[0])
    if span <= 0.0:
        return 0.0
    intervals = np.diff(event_times)
    positive = intervals[intervals > 0]
    if len(positive) == 0:
        return 0.0
    # Bin at ~1/4 of the median interval so the beacon period spans >= 4 bins.
    bin_width = max(float(np.median(positive)) / 4.0, span / config.max_bins)
    n_bins = int(np.clip(np.ceil(span / bin_width), config.min_bins, config.max_bins))
    counts, _ = np.histogram(event_times, bins=n_bins)
    centered = counts - counts.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    power = power[1:]  # drop DC
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    return float(power.max()) / total


def score_event_times(event_times: np.ndarray, config: BeaconingConfig | None = None) -> PairScore:
    """Beacon score for one (host, destination) event-time series."""
    config = config or BeaconingConfig()
    times = np.sort(np.asarray(event_times, dtype=np.float64))
    n = int(len(times))
    if n < config.min_events:
        return PairScore(0.0, 0.0, 0.0, 0.0, n)
    intervals = np.diff(times)
    cv = _cv_score(intervals)
    autocorr = _autocorr_score(intervals, config.max_lag)
    fourier = _fourier_score(times, config)
    return PairScore(float((cv + autocorr + fourier) / 3.0), cv, autocorr, fourier, n)


def reconstruct_flow_times(
    session_timestamp: float,
    session: np.ndarray,
    mask: np.ndarray,
    iat_index: int,
) -> np.ndarray:
    """Absolute flow times for one session from its start time and per-flow IATs.

    The first flow sits at the session timestamp; flow k is offset by the
    cumulative IAT of flows 2..k (a flow's IAT is the gap to its predecessor,
    so the first flow's own IAT - the gap to the previous session - is not
    part of this session's timeline).
    """
    valid = session[np.asarray(mask, dtype=bool)]
    if len(valid) == 0:
        return np.asarray([float(session_timestamp)], dtype=np.float64)
    iats = np.clip(valid[:, iat_index].astype(np.float64), 0.0, None)
    offsets = np.concatenate([[0.0], np.cumsum(iats[1:])])
    return float(session_timestamp) + offsets


class BeaconingDetector:
    """Score sessions by the periodicity of their (host, destination) pair."""

    def __init__(self, config: BeaconingConfig | None = None):
        self.config = config or BeaconingConfig()

    def score_sessions(
        self,
        *,
        sessions: np.ndarray,
        masks: np.ndarray,
        host_ids: np.ndarray,
        dest_ids: np.ndarray,
        timestamps: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
    ) -> np.ndarray:
        """Per-session beacon scores in [0, 1] (pair score broadcast to sessions)."""
        feature_names = [str(name) for name in feature_names]
        if self.config.iat_feature not in feature_names:
            raise ValueError(f"Feature '{self.config.iat_feature}' not in feature_names")
        iat_index = feature_names.index(self.config.iat_feature)

        n = len(sessions)
        if not (len(masks) == len(host_ids) == len(dest_ids) == len(timestamps) == n):
            raise ValueError("sessions, masks, host_ids, dest_ids, timestamps must have equal length")

        pair_events: dict[tuple[str, str], list[np.ndarray]] = {}
        pair_of_session: list[tuple[str, str]] = []
        for i in range(n):
            key = (str(host_ids[i]), str(dest_ids[i]))
            pair_of_session.append(key)
            pair_events.setdefault(key, []).append(
                reconstruct_flow_times(float(timestamps[i]), sessions[i], masks[i], iat_index)
            )

        pair_scores = {
            key: score_event_times(np.concatenate(events), self.config)
            for key, events in pair_events.items()
        }
        return np.asarray([pair_scores[key].score for key in pair_of_session], dtype=np.float64)

    def pair_report(
        self,
        *,
        sessions: np.ndarray,
        masks: np.ndarray,
        host_ids: np.ndarray,
        dest_ids: np.ndarray,
        timestamps: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
    ) -> dict[tuple[str, str], PairScore]:
        """Full per-pair component scores (for analysis/debugging)."""
        feature_names = [str(name) for name in feature_names]
        iat_index = feature_names.index(self.config.iat_feature)
        pair_events: dict[tuple[str, str], list[np.ndarray]] = {}
        for i in range(len(sessions)):
            key = (str(host_ids[i]), str(dest_ids[i]))
            pair_events.setdefault(key, []).append(
                reconstruct_flow_times(float(timestamps[i]), sessions[i], masks[i], iat_index)
            )
        return {
            key: score_event_times(np.concatenate(events), self.config)
            for key, events in pair_events.items()
        }
