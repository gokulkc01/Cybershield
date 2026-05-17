"""Run baseline vs mutated evaluation using the existing CyberShield Transformer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_loader.extended_feature_transforms import (
    ExtendedFeatureTransformConfig,
    apply_extended_feature_transforms,
)
from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.normalization import FeatureNormalizer
from src.data_loader.npz_utils import detect_npz_schema, load_session_npz
from src.features.feature_config_experiment import FEATURE_DIM, FEATURE_NAMES, SESSION_LEN
from src.features.feature_config_extended import FEATURE_SCHEMA_VERSION as EXTENDED_SCHEMA_VERSION
from src.models.transformer import C2Transformer
from src.red_agent.behavioral_failure_analyzer import analyze_failures
from src.red_agent.robustness_metrics import (
    aggregate_by_mutation,
    compute_detection_metrics,
    compute_robustness_summary,
)


class _SessionDataset(Dataset):
    def __init__(self, sequences: np.ndarray, masks: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.padding_masks = torch.tensor(~masks, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.padding_masks[idx], self.labels[idx]


@dataclass(frozen=True)
class RunnerOutput:
    baseline_metrics: Dict[str, float]
    mutated_metrics: Dict[str, float]
    robustness_summary: Dict[str, float]
    mutation_breakdown: Dict[str, Dict[str, float]]
    failure_analysis: Dict[str, object]


def _load_transform_config(checkpoint: dict) -> FeatureTransformConfig | ExtendedFeatureTransformConfig:
    payload = checkpoint.get("feature_transform_config")
    if isinstance(payload, dict) and payload.get("schema") == EXTENDED_SCHEMA_VERSION:
        return ExtendedFeatureTransformConfig.from_checkpoint_dict(payload)
    return FeatureTransformConfig.from_checkpoint_dict(payload)


def _apply_transforms(
    sequences: np.ndarray,
    masks: np.ndarray,
    config: FeatureTransformConfig | ExtendedFeatureTransformConfig,
    schema_version: str,
    feature_names: tuple[str, ...] | None,
) -> np.ndarray:
    if schema_version == EXTENDED_SCHEMA_VERSION:
        return apply_extended_feature_transforms(sequences, masks, config)  # type: ignore[arg-type]
    return apply_feature_transforms(sequences, masks, config, feature_names=feature_names)


def _load_model_and_threshold(checkpoint_path: str, device: torch.device) -> tuple[torch.nn.Module, dict, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    threshold = float(checkpoint.get("optimal_threshold", 0.5))

    schema_version = str(checkpoint.get("schema_version", ""))
    if schema_version == EXTENDED_SCHEMA_VERSION:
        feature_dim = int(checkpoint.get("feature_dim", len(FEATURE_NAMES)))
        seq_len = int(checkpoint.get("session_len", SESSION_LEN))
    else:
        feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
        seq_len = int(checkpoint.get("session_len", SESSION_LEN))

    model = C2Transformer(
        feature_dim=feature_dim,
        seq_len=seq_len,
        use_derivative_features=bool(checkpoint.get("use_derivative_features", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, threshold


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    checkpoint: dict,
    sequences: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    transform_cfg = _load_transform_config(checkpoint)
    schema_version = str(checkpoint.get("schema_version", ""))
    x = _apply_transforms(sequences, masks, transform_cfg, schema_version, FEATURE_NAMES)

    if checkpoint.get("normalize_features", False):
        saved = checkpoint.get("feature_normalizer")
        if saved is None:
            raise ValueError("Checkpoint expects feature normalization but has no normalizer")
        normalizer = FeatureNormalizer.from_checkpoint_dict(saved)
        x = normalizer.transform(x, masks)

    ds = _SessionDataset(x, masks, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    start = time.perf_counter()
    for batch_x, batch_mask, batch_y in loader:
        logits = model(batch_x.to(device), batch_mask.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch_y.numpy())
    elapsed = (time.perf_counter() - start) * 1000.0
    latency_per_sample_ms = float(elapsed / max(1, len(ds)))

    return np.concatenate(probs), np.concatenate(ys), latency_per_sample_ms


def run_mutation_evaluation(
    checkpoint_path: str,
    baseline_npz: str,
    mutated_npz: str,
    metadata_jsonl: str,
    output_json: str,
    batch_size: int = 256,
    device: str | None = None,
) -> RunnerOutput:
    """Evaluate robustness by comparing baseline and mutated samples."""

    baseline_schema = detect_npz_schema(baseline_npz)
    mutated_schema = detect_npz_schema(mutated_npz)
    if baseline_schema["feature_names"] and mutated_schema["feature_names"] and tuple(baseline_schema["feature_names"]) != tuple(mutated_schema["feature_names"]):
        raise ValueError("Baseline and mutated NPZ feature schemas do not match.")

    expected_feature_names = tuple(baseline_schema["feature_names"]) if baseline_schema["feature_names"] else FEATURE_NAMES
    expected_session_len = int(baseline_schema["session_len"]) if baseline_schema["session_len"] else SESSION_LEN

    baseline_x, baseline_y, baseline_masks = load_session_npz(
        baseline_npz,
        expected_feature_names=expected_feature_names,
        expected_session_len=expected_session_len,
    )
    mutated_x, mutated_y, mutated_masks = load_session_npz(
        mutated_npz,
        expected_feature_names=expected_feature_names,
        expected_session_len=expected_session_len,
    )

    if baseline_x.shape != mutated_x.shape:
        raise ValueError(f"Shape mismatch baseline={baseline_x.shape}, mutated={mutated_x.shape}")

    rows: List[dict] = []
    with Path(metadata_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint, threshold = _load_model_and_threshold(checkpoint_path, device_obj)

    baseline_probs, baseline_targets, baseline_latency = _predict(
        model, checkpoint, baseline_x, baseline_masks, baseline_y, batch_size, device_obj
    )
    mutated_probs, mutated_targets, mutated_latency = _predict(
        model, checkpoint, mutated_x, mutated_masks, mutated_y, batch_size, device_obj
    )

    baseline_metrics = compute_detection_metrics(baseline_targets, baseline_probs, threshold)
    mutated_metrics = compute_detection_metrics(mutated_targets, mutated_probs, threshold)
    robustness_summary = compute_robustness_summary(
        baseline=baseline_metrics,
        mutated=mutated_metrics,
        baseline_latency_ms=baseline_latency,
        mutated_latency_ms=mutated_latency,
    )

    for i, row in enumerate(rows):
        prob = float(mutated_probs[i]) if i < len(mutated_probs) else float("nan")
        is_detected = bool(prob >= threshold) if np.isfinite(prob) else False
        row["evaluation_outcome"] = {
            "threshold": float(threshold),
            "mutated_score": prob,
            "is_detected": is_detected,
            "target_label": int(mutated_targets[i]) if i < len(mutated_targets) else None,
            "latency_per_sample_ms": mutated_latency,
        }

    mutation_breakdown = aggregate_by_mutation(rows)
    failure_analysis = analyze_failures(
        original_sequences=baseline_x,
        mutated_sequences=mutated_x,
        metadata_rows=rows,
        feature_names=list(FEATURE_NAMES),
    )

    payload = {
        "checkpoint_path": checkpoint_path,
        "threshold": float(threshold),
        "baseline_metrics": baseline_metrics,
        "mutated_metrics": mutated_metrics,
        "robustness_summary": robustness_summary,
        "mutation_breakdown": mutation_breakdown,
        "failure_analysis": failure_analysis,
        "n_metadata_rows": len(rows),
    }
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    Path(metadata_jsonl).write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )

    return RunnerOutput(
        baseline_metrics=baseline_metrics,
        mutated_metrics=mutated_metrics,
        robustness_summary=robustness_summary,
        mutation_breakdown=mutation_breakdown,
        failure_analysis=failure_analysis,
    )
