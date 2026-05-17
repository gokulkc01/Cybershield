"""CLI inference for the exported domain-adaptive model with centroid-based domain detector.

Inputs can be:
 - an .npz file containing 'X'/'sessions' (batch inference), or
 - a single session JSON with 'features' array (single inference).

Usage examples:
  python -m src.inference.run_inference --export_dir experiments/domain_adaptive_sweep_20/exported --input data/processed/uwf_adaptation_split/test_sessions.npz --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data_loader.npz_utils import load_session_npz, detect_npz_schema
from src.models.domain_adaptive_transformer import DomainAdaptiveC2Transformer


def load_exported(export_dir: str):
    export_dir = Path(export_dir)
    ckpt = torch.load(export_dir / "model_checkpoint.pth", map_location="cpu", weights_only=False)
    thresholds = {}
    if (export_dir / "thresholds.json").exists():
        thresholds = json.loads((export_dir / "thresholds.json").read_text())
    centroids = json.loads((export_dir / "domain_centroids.json").read_text()) if (export_dir / "domain_centroids.json").exists() else None
    return ckpt, thresholds, centroids


def build_model(checkpoint: dict) -> DomainAdaptiveC2Transformer:
    """Build model from checkpoint and load weights. Returns model in eval mode."""
    domains = checkpoint.get("domains") or ["mixed", "uwf"]
    feature_dim = int(checkpoint.get("feature_dim", 10))
    seq_len = int(checkpoint.get("session_len", 20))
    model = DomainAdaptiveC2Transformer(
        domains=domains,
        feature_dim=feature_dim,
        seq_len=seq_len,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def detect_domain_by_centroid(session_vec: np.ndarray, centroids: dict[str, list[float]]) -> str:
    # cosine similarity
    best = None
    best_sim = -2.0
    for name, centroid in centroids.items():
        c = np.asarray(centroid, dtype=float)
        sim = float(np.dot(session_vec, c) / (np.linalg.norm(session_vec) * np.linalg.norm(c) + 1e-9))
        if sim > best_sim:
            best_sim = sim
            best = name
    return best


def detect_domains_batch(session_vecs: np.ndarray, centroids: dict[str, list[float]]) -> list[str]:
    """Vectorised domain detection for a batch of session vectors.
    
    Args:
        session_vecs: (N, D) array of mean-pooled session vectors
        centroids: mapping from domain name to centroid vector
    
    Returns:
        List of domain names, one per session
    """
    if centroids is None:
        return ["mixed"] * len(session_vecs)

    names = list(centroids.keys())
    centroid_matrix = np.array([centroids[n] for n in names], dtype=np.float64)  # (K, D)

    # Normalise
    vec_norms = np.linalg.norm(session_vecs, axis=1, keepdims=True).clip(min=1e-9)  # (N, 1)
    cent_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True).clip(min=1e-9)  # (K, 1)

    # Cosine similarity matrix: (N, K)
    similarities = (session_vecs / vec_norms) @ (centroid_matrix / cent_norms).T
    best_indices = similarities.argmax(axis=1)
    return [names[i] for i in best_indices]


def session_to_vector(session: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # simple mean pooling over real flows
    real = mask.astype(bool)
    sums = (session * real[..., None]).sum(axis=1)
    counts = real.sum(axis=1).clip(min=1)[:, None]
    return (sums / counts).mean(axis=0)


def sessions_to_vectors(sessions: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Vectorised mean pooling over real flows for all sessions.
    
    Args:
        sessions: (N, T, D) array
        masks: (N, T) boolean array (True = real flow)
    
    Returns:
        (N, D) array of mean-pooled vectors
    """
    real = masks.astype(bool)  # (N, T)
    sums = (sessions * real[..., None]).sum(axis=1)  # (N, D)
    counts = real.sum(axis=1, keepdims=True).clip(min=1)  # (N, 1)
    return sums / counts


def run_on_arrays(
    checkpoint: dict,
    thresholds: dict,
    centroids: dict,
    X: np.ndarray,
    masks: np.ndarray,
    batch_size: int = 512,
    model: DomainAdaptiveC2Transformer | None = None,
) -> list[dict]:
    """Run batched inference on pre-loaded numpy arrays.
    
    Works entirely in-memory — no file I/O required.
    
    Args:
        checkpoint: Loaded checkpoint dict
        thresholds: Domain threshold mapping
        centroids: Domain centroid mapping
        X: Session data array (N, T, D)
        masks: Real-flow masks (N, T)
        batch_size: Inference batch size (default 512)
        model: Optional pre-built model (avoids re-instantiation)
    
    Returns:
        List of prediction dicts with index, domain, prob, label
    """
    if model is None:
        model = build_model(checkpoint)

    domains = checkpoint.get("domains") or ["mixed", "uwf"]

    # --- Vectorised domain detection ---
    session_vecs = sessions_to_vectors(X, masks)
    domain_names = detect_domains_batch(session_vecs, centroids)

    # Map domain names to integer IDs
    domain_id_map = {name: idx for idx, name in enumerate(domains)}
    domain_ids = np.array([domain_id_map.get(d, 0) for d in domain_names], dtype=np.int64)

    # --- Batched model inference ---
    all_probs = np.empty(len(X), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            x_batch = torch.tensor(X[start:end], dtype=torch.float32)
            m_batch = torch.tensor(~masks[start:end], dtype=torch.bool)
            d_batch = torch.tensor(domain_ids[start:end], dtype=torch.long)
            logits = model(x_batch, m_batch, d_batch)
            all_probs[start:end] = torch.sigmoid(logits).cpu().numpy()

    # --- Build results ---
    results = []
    for i in range(len(X)):
        domain = domain_names[i]
        prob = float(all_probs[i])
        thresh = thresholds.get(domain)
        label = int(prob >= thresh) if thresh is not None else int(prob >= 0.5)
        results.append({"index": i, "domain": domain, "prob": prob, "label": label})

    return results


def run_on_npz(
    export_dir: str,
    checkpoint: dict,
    thresholds: dict,
    centroids: dict,
    npz_path: str,
    out_path: str,
    batch_size: int = 512,
    model: DomainAdaptiveC2Transformer | None = None,
):
    """Run batched inference on an NPZ file (CLI use).
    
    Args:
        export_dir: Path to exported model directory
        checkpoint: Loaded checkpoint dict
        thresholds: Domain threshold mapping
        centroids: Domain centroid mapping
        npz_path: Path to input NPZ file
        out_path: Path to write predictions JSON
        batch_size: Inference batch size (default 512)
        model: Optional pre-built model (avoids re-instantiation)
    """
    schema = detect_npz_schema(npz_path)
    X, y, masks = load_session_npz(npz_path, expected_feature_names=tuple(checkpoint.get("feature_names") or []))

    results = run_on_arrays(checkpoint, thresholds, centroids, X, masks, batch_size, model)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="inference_results.json")
    args = parser.parse_args()

    ckpt, thresholds, centroids = load_exported(args.export_dir)
    run_on_npz(args.export_dir, ckpt, thresholds, centroids, args.input, args.out)


if __name__ == "__main__":
    main()
