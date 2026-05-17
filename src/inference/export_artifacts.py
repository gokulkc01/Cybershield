"""Export model artifacts for inference: model checkpoint, normalizer, transforms, thresholds, and domain centroids.

Usage:
    python -m src.inference.export_artifacts --checkpoint experiments/domain_adaptive_sweep_20/best_transformer.pth --out_dir experiments/domain_adaptive_sweep_20/exported
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np

from src.data_loader.npz_utils import detect_npz_schema, load_session_npz
from src.data_loader.normalization import FeatureNormalizer


def compute_domain_centroids(mixed_npz, uwf_npz, expected_feature_names=None):
    # Load datasets and compute mean feature vectors across real flows per session,
    # then global centroid per domain.
    seq_m, y_m, masks_m = load_session_npz(mixed_npz, expected_feature_names=expected_feature_names)
    seq_u, y_u, masks_u = load_session_npz(uwf_npz, expected_feature_names=expected_feature_names)

    # Compute per-session mean over time axis (ignoring padding)
    def session_means(seqs, masks):
        real = masks.astype(bool)
        sums = (seqs * real[..., None]).sum(axis=1)
        counts = real.sum(axis=1).clip(min=1)[:, None]
        return sums / counts

    m_means = session_means(seq_m, masks_m)
    u_means = session_means(seq_u, masks_u)
    return m_means.mean(axis=0).tolist(), u_means.mean(axis=0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixed_npz", default="data/processed/extended_mixed_real_corrected/test_sessions.npz")
    parser.add_argument("--uwf_npz", default="data/processed/uwf_adaptation_split/test_sessions.npz")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Save checkpoint copy into exported dir
    ckpt_out = Path(args.out_dir) / "model_checkpoint.pth"
    torch.save(checkpoint, ckpt_out)

    # Save thresholds if present in checkpoint
    thresholds = {}
    if "mixed_val_threshold" in checkpoint:
        thresholds["mixed"] = float(checkpoint["mixed_val_threshold"]) 
    if "uwf_val_threshold" in checkpoint:
        thresholds["uwf"] = float(checkpoint["uwf_val_threshold"]) 
    with open(Path(args.out_dir) / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    # Save transform config and normalizer if present
    if "feature_transform_config" in checkpoint:
        with open(Path(args.out_dir) / "feature_transform_config.json", "w") as f:
            json.dump(checkpoint["feature_transform_config"], f, indent=2)
    if "feature_normalizer" in checkpoint and checkpoint["feature_normalizer"] is not None:
        fn = checkpoint["feature_normalizer"]
        # If normalizer is an object with to_checkpoint_dict, prefer that
        try:
            data = fn.to_checkpoint_dict()
        except Exception:
            # Fall back to JSON-safe conversion
            def _convert(o: Any):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, dict):
                    return {k: _convert(v) for k, v in o.items()}
                if isinstance(o, (list, tuple)):
                    return [_convert(x) for x in o]
                return o
            data = _convert(fn)
        with open(Path(args.out_dir) / "feature_normalizer.json", "w") as f:
            json.dump(data, f, indent=2)

    # Compute simple domain centroids (for domain detection)
    m_centroid, u_centroid = compute_domain_centroids(args.mixed_npz, args.uwf_npz, expected_feature_names=tuple(checkpoint.get("feature_names") or []))
    with open(Path(args.out_dir) / "domain_centroids.json", "w") as f:
        json.dump({"mixed": m_centroid, "uwf": u_centroid}, f, indent=2)

    meta = {
        "exported_checkpoint": str(ckpt_out),
        "thresholds": thresholds,
        "domain_centroids": str(Path(args.out_dir) / "domain_centroids.json"),
    }
    with open(Path(args.out_dir) / "export_manifest.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Export complete:", args.out_dir)


if __name__ == "__main__":
    main()
