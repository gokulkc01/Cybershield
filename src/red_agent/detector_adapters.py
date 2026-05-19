"""Frozen detector adapters for Red-Agent training and evaluation.

The Red-Agent should optimize against the same validation-frozen operating
point used in detector benchmarks.  These adapters load existing CyberShield
checkpoints and expose positive-class C2 probabilities for session-only and
host-aware detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.data_loader.extended_feature_transforms import (
    ExtendedFeatureTransformConfig,
    apply_extended_feature_transforms,
)
from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.normalization import FeatureNormalizer
from src.features.feature_config import FEATURE_NAMES
from src.features.feature_config_extended import FEATURE_SCHEMA_VERSION as EXTENDED_SCHEMA_VERSION
from src.models.domain_adaptive_transformer import DomainAdaptiveC2Transformer
from src.models.host_aware_domain_adaptive_transformer import HostAwareDomainAdaptiveTransformer
from src.models.transformer import C2Transformer


@dataclass(frozen=True)
class DetectorScore:
    """Positive-class score from a frozen detector."""

    probability: float
    threshold: float

    @property
    def detected(self) -> bool:
        return self.probability >= self.threshold


class HostFeatureNormalizer:
    """Small inference-side normalizer for host summary features."""

    def __init__(self, payload: dict | None):
        self.enabled = bool(payload)
        self.mean = np.asarray(payload["mean"], dtype=np.float32) if payload else None
        self.std = np.asarray(payload["std"], dtype=np.float32) if payload else None

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return values.astype(np.float32, copy=True)
        assert self.mean is not None and self.std is not None
        return ((values.astype(np.float32) - self.mean) / self.std).astype(np.float32)


class FrozenDetectorAdapter:
    """Load a frozen detector checkpoint and expose C2 probability scoring."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        fpr_budget: float | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Detector checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model_type = str(self.checkpoint.get("model_type", "c2_transformer"))
        self.feature_names = tuple(str(name) for name in self.checkpoint.get("feature_names", FEATURE_NAMES))
        self.feature_dim = int(self.checkpoint.get("feature_dim", len(self.feature_names)))
        self.session_len = int(self.checkpoint.get("session_len", 20))
        self.schema_version = str(
            self.checkpoint.get(
                "session_schema_version",
                self.checkpoint.get("schema_version", ""),
            )
        )
        self.transform_config = self._load_transform_config()
        self.feature_normalizer = (
            FeatureNormalizer.from_checkpoint_dict(self.checkpoint["feature_normalizer"])
            if self.checkpoint.get("normalize_features", False)
            and self.checkpoint.get("feature_normalizer") is not None
            else None
        )
        self.host_feature_normalizer = HostFeatureNormalizer(
            self.checkpoint.get("host_feature_normalizer")
            if self.checkpoint.get("normalize_host_features", False)
            else None
        )
        self.threshold = self._resolve_threshold(fpr_budget)
        self.model = self._load_model().to(self.device)
        self.model.eval()

    def score_session(self, session: np.ndarray, mask: np.ndarray | None = None, *, domain_id: int = 0) -> DetectorScore:
        """Score a single session for C2 probability."""
        if self.model_type == "host_aware_domain_adaptive_transformer":
            history_size = int(self.checkpoint.get("history_size", 32))
            host_feature_dim = int(self.checkpoint.get("host_feature_dim", 15))
            history = np.zeros((history_size, self.session_len, self.feature_dim), dtype=np.float32)
            history_mask = np.zeros((history_size, self.session_len), dtype=bool)
            history_session_mask = np.zeros(history_size, dtype=bool)
            host_features = np.zeros(host_feature_dim, dtype=np.float32)
            return self.score_host_window(
                current_session=session,
                current_mask=mask,
                history_sessions=history,
                history_flow_masks=history_mask,
                history_session_masks=history_session_mask,
                host_features=host_features,
                domain_id=domain_id,
            )

        current_mask = self._coerce_mask(session, mask)
        processed = self._preprocess_sessions(
            np.asarray(session, dtype=np.float32)[np.newaxis, ...],
            current_mask[np.newaxis, ...],
        )
        padding_mask = torch.from_numpy(~current_mask[np.newaxis, ...]).to(self.device)
        features = torch.from_numpy(processed).to(self.device)
        with torch.no_grad():
            if self.model_type == "domain_adaptive_transformer":
                domain_ids = torch.full((1,), int(domain_id), dtype=torch.long, device=self.device)
                logits = self.model(features, padding_mask, domain_ids)
            else:
                logits = self.model(features, padding_mask)
            probability = torch.sigmoid(logits).item()
        return DetectorScore(probability=float(probability), threshold=self.threshold)

    def score_host_window(
        self,
        *,
        current_session: np.ndarray,
        current_mask: np.ndarray | None,
        history_sessions: np.ndarray,
        history_flow_masks: np.ndarray,
        history_session_masks: np.ndarray,
        host_features: np.ndarray,
        source: str | None = None,
        domain_id: int | None = None,
    ) -> DetectorScore:
        """Score a host-aware window while optionally replacing the current session."""
        if self.model_type != "host_aware_domain_adaptive_transformer":
            return self.score_session(current_session, current_mask, domain_id=domain_id or 0)

        current_mask = self._coerce_mask(current_session, current_mask)
        current = self._preprocess_sessions(
            np.asarray(current_session, dtype=np.float32)[np.newaxis, ...],
            current_mask[np.newaxis, ...],
        )

        history = np.asarray(history_sessions, dtype=np.float32)
        history_masks = np.asarray(history_flow_masks, dtype=bool)
        if history.ndim != 3:
            raise ValueError(f"history_sessions must have shape (H, T, D), got {history.shape}")
        history_processed = self._preprocess_history(history[np.newaxis, ...], history_masks[np.newaxis, ...])
        host_values = self.host_feature_normalizer.transform(
            np.asarray(host_features, dtype=np.float32)[np.newaxis, ...]
        )
        resolved_domain_id = self._resolve_domain_id(source=source, domain_id=domain_id)

        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(current).to(self.device),
                torch.from_numpy(~current_mask[np.newaxis, ...]).to(self.device),
                torch.from_numpy(history_processed).to(self.device),
                torch.from_numpy(~history_masks[np.newaxis, ...]).to(self.device),
                torch.from_numpy(np.asarray(history_session_masks, dtype=bool)[np.newaxis, ...]).to(self.device),
                torch.from_numpy(host_values).to(self.device),
                torch.full((1,), resolved_domain_id, dtype=torch.long, device=self.device),
            )
            probability = torch.sigmoid(logits).item()
        return DetectorScore(probability=float(probability), threshold=self.threshold)

    def _load_model(self) -> torch.nn.Module:
        if self.model_type == "host_aware_domain_adaptive_transformer":
            model = HostAwareDomainAdaptiveTransformer(
                domains=list(self.checkpoint["domains"]),
                feature_dim=self.feature_dim,
                host_feature_dim=int(self.checkpoint["host_feature_dim"]),
                seq_len=self.session_len,
                history_size=int(self.checkpoint["history_size"]),
            )
        elif self.model_type == "domain_adaptive_transformer":
            domains = self.checkpoint.get("domains")
            if not domains:
                domains = [str(self.checkpoint.get("domain_name", "session"))]
            model = DomainAdaptiveC2Transformer(
                domains=list(domains),
                feature_dim=self.feature_dim,
                seq_len=self.session_len,
                use_derivative_features=bool(self.checkpoint.get("use_derivative_features", False)),
            )
        else:
            model = C2Transformer(
                feature_dim=self.feature_dim,
                seq_len=self.session_len,
                use_derivative_features=bool(self.checkpoint.get("use_derivative_features", False)),
            )
        model.load_state_dict(self.checkpoint["model_state_dict"])
        return model

    def _load_transform_config(self) -> FeatureTransformConfig | ExtendedFeatureTransformConfig:
        payload = self.checkpoint.get("feature_transform_config")
        if self.schema_version == EXTENDED_SCHEMA_VERSION or (
            isinstance(payload, dict) and payload.get("schema") == EXTENDED_SCHEMA_VERSION
        ):
            return ExtendedFeatureTransformConfig.from_checkpoint_dict(payload)
        return FeatureTransformConfig.from_checkpoint_dict(payload)

    def _preprocess_sessions(self, sessions: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if self.schema_version == EXTENDED_SCHEMA_VERSION:
            processed = apply_extended_feature_transforms(
                sessions,
                masks,
                self.transform_config,  # type: ignore[arg-type]
            )
        else:
            processed = apply_feature_transforms(
                sessions,
                masks,
                self.transform_config,  # type: ignore[arg-type]
                feature_names=self.feature_names,
            )
        if self.feature_normalizer is not None:
            processed = self.feature_normalizer.transform(processed, masks)
        return processed.astype(np.float32)

    def _preprocess_history(self, history: np.ndarray, history_masks: np.ndarray) -> np.ndarray:
        batch_size, history_size, session_len, feature_dim = history.shape
        flat_history = history.reshape(batch_size * history_size, session_len, feature_dim)
        flat_masks = history_masks.reshape(batch_size * history_size, session_len)
        processed = self._preprocess_sessions(flat_history, flat_masks)
        return processed.reshape(batch_size, history_size, session_len, feature_dim)

    def _resolve_threshold(self, fpr_budget: float | None) -> float:
        metrics = self.checkpoint.get("validation_metrics_by_budget")
        if isinstance(metrics, dict) and metrics:
            if fpr_budget is None:
                fpr_budget = float(self.checkpoint.get("default_fpr_budget", 0.015))
            key = f"{float(fpr_budget):.4f}"
            if key in metrics and "threshold" in metrics[key]:
                return float(metrics[key]["threshold"])
            first_metrics = next(iter(metrics.values()))
            if isinstance(first_metrics, dict) and "threshold" in first_metrics:
                return float(first_metrics["threshold"])
        return float(self.checkpoint.get("optimal_threshold", 0.5))

    def _resolve_domain_id(self, *, source: str | None, domain_id: int | None) -> int:
        if domain_id is not None:
            return int(domain_id)
        domain_to_index = self.checkpoint.get("domain_to_index", {})
        if source is not None and source in domain_to_index:
            return int(domain_to_index[source])
        return 0

    def _coerce_mask(self, session: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        session_arr = np.asarray(session)
        if session_arr.shape != (self.session_len, self.feature_dim):
            raise ValueError(
                f"Expected session shape ({self.session_len}, {self.feature_dim}), got {session_arr.shape}"
            )
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != (self.session_len,):
                raise ValueError(f"Expected mask shape ({self.session_len},), got {mask_arr.shape}")
            return mask_arr
        return np.any(session_arr != 0.0, axis=1)


def score_many_sessions(
    adapter: FrozenDetectorAdapter,
    sessions: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> list[DetectorScore]:
    """Convenience helper used by tests and small diagnostics."""
    return [adapter.score_session(session, mask) for session, mask in zip(sessions, masks)]
