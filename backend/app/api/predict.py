"""
Domain-Adaptive Model Inference API.

Real-time C2 traffic detection using the domain-adaptive transformer.
Supports NPZ file uploads and single session JSON predictions.

Performance optimisations:
- Model loaded once at startup and cached globally
- CPU-bound inference offloaded to a thread pool (avoids blocking async loop)
- Batched inference instead of per-session loops
"""

import logging
import json
import asyncio
import traceback
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Default export directory
DEFAULT_EXPORT_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "domain_adaptive_sweep_20" / "exported"

# Global model cache — loaded once, reused forever
_model_cache: Dict[str, Any] = {}

# Thread pool for CPU-bound inference work so we don't block the async event loop
_inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")


def _get_or_load_model(export_dir: str):
    """Load model + thresholds + centroids once and cache them.
    
    Returns (model, thresholds, centroids) where model is already in eval mode.
    """
    if export_dir not in _model_cache:
        from src.inference.run_inference import load_exported, build_model
        logger.info(f"Loading model from {export_dir}...")
        ckpt, thresholds, centroids = load_exported(export_dir)
        model = build_model(ckpt)
        _model_cache[export_dir] = {
            "model": model,
            "ckpt": ckpt,
            "thresholds": thresholds,
            "centroids": centroids,
        }
        logger.info("✅ Model loaded and cached")
    return _model_cache[export_dir]


class SessionPayload(BaseModel):
    """Single session prediction payload."""
    session: list[list[float]]
    mask: list[int]
    domain: Optional[str] = None


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check if model is loaded and ready.
    
    Uses the global cache — does NOT reload the model each time.
    First call offloads model loading to thread pool to avoid blocking the event loop.
    """
    try:
        export_dir = str(DEFAULT_EXPORT_DIR)
        if not Path(export_dir).exists():
            raise HTTPException(status_code=503, detail=f"Export directory not found: {export_dir}")
        
        # Offload initial model loading to thread pool so we don't block the event loop
        loop = asyncio.get_running_loop()
        cached = await loop.run_in_executor(_inference_executor, _get_or_load_model, export_dir)
        thresholds = cached["thresholds"]
        return {
            "status": "ok",
            "model": "domain-adaptive-transformer",
            "domains": list(thresholds.keys()) if thresholds else ["mixed", "uwf"],
            "export_dir": export_dir,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")


def _run_inference_sync(file_bytes: bytes, filename: str, export_dir: str) -> Dict[str, Any]:
    """Synchronous inference worker — runs in thread pool.
    
    Loads NPZ from memory and runs batched model inference.
    No temp files are created (avoids Windows file locking issues).
    """
    import io
    import numpy as np
    from src.inference.run_inference import run_on_arrays
    from src.data_loader.npz_utils import load_session_npz_from_bytes

    logger.info(f"Starting inference on {filename} ({len(file_bytes)} bytes)")

    cached = _get_or_load_model(export_dir)
    model = cached["model"]
    ckpt = cached["ckpt"]
    thresholds = cached["thresholds"]
    centroids = cached["centroids"]

    # Load NPZ directly from bytes — no temp file needed
    X, y, masks = load_session_npz_from_bytes(
        file_bytes,
        expected_feature_names=tuple(ckpt.get("feature_names") or []),
    )

    # Run inference entirely in memory
    raw_preds = run_on_arrays(
        ckpt, thresholds, centroids, X, masks,
        batch_size=512, model=model,
    )

    # Transform to match frontend's expected Prediction interface:
    # { index, domain, probability, label (string), label_int (int) }
    preds = []
    for p in raw_preds:
        preds.append({
            "index": p["index"],
            "domain": p["domain"],
            "probability": p["prob"],
            "label": "C2" if p["label"] == 1 else "benign",
            "label_int": p["label"],
        })

    logger.info(f"Inference complete: {len(preds)} predictions")
    return {
        "status": "success",
        "n_predictions": len(preds),
        "predictions": preds,
    }


@router.post("/predict-file")
async def predict_file(file: UploadFile = File(...), export_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload NPZ file and get predictions for all sessions.
    
    Args:
        file: NPZ file containing session data
        export_dir: Optional path to exported model artifacts
    
    Returns:
        Dictionary with predictions for each session
    """
    export_dir = export_dir or str(DEFAULT_EXPORT_DIR)
    
    if not file.filename.endswith('.npz'):
        raise HTTPException(status_code=400, detail="Only .npz uploads supported")
    
    try:
        # Read file bytes in the async context
        contents = await file.read()
        
        # Offload CPU-bound inference to thread pool so the event loop stays responsive
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _inference_executor,
            _run_inference_sync,
            contents,
            file.filename,
            export_dir,
        )
        return result

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/predict-json")
async def predict_json(payload: SessionPayload, export_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Get prediction for a single session via JSON input.
    
    Args:
        payload: Session data with features and mask
        export_dir: Optional path to exported model artifacts
    
    Returns:
        Dictionary with domain, probability, and label
    """
    export_dir = export_dir or str(DEFAULT_EXPORT_DIR)
    
    try:
        from src.inference.run_inference import detect_domain_by_centroid, session_to_vector
        import numpy as np
        import torch
        
        cached = _get_or_load_model(export_dir)
        model = cached["model"]
        ckpt = cached["ckpt"]
        thresholds = cached["thresholds"]
        centroids = cached["centroids"]
        domains = ckpt.get("domains") or ["mixed", "uwf"]
        
        # Prepare tensors
        sess = np.asarray(payload.session, dtype=float)[None, ...]
        mask = np.asarray(payload.mask, dtype=int)[None, ...]
        
        # Domain detection
        vec = session_to_vector(sess, mask)
        domain = payload.domain or (detect_domain_by_centroid(vec, centroids) if centroids else 'mixed')
        
        # Inference (model already in eval mode from cache)
        x_t = torch.tensor(sess, dtype=torch.float32)
        m_t = torch.tensor(~mask, dtype=torch.bool)
        domain_id = domains.index(domain) if domain in domains else 0
        
        with torch.no_grad():
            logits = model(x_t, m_t, torch.tensor([domain_id]))
            prob = float(torch.sigmoid(logits).cpu().item())
        
        # Apply threshold
        thresh = thresholds.get(domain)
        label = int(prob >= thresh) if thresh is not None else int(prob >= 0.5)
        
        return {
            "status": "success",
            "domain": domain,
            "probability": prob,
            "label": "C2" if label else "benign",
            "label_int": label
        }
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
