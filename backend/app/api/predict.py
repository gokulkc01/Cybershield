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
import asyncio
import traceback
import io
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Default export directory
DEFAULT_EXPORT_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "domain_adaptive_sweep_20" / "exported"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SESSION_MODEL_CHECKPOINT = PROJECT_ROOT / "experiments" / "host_aware_model_benchmark_clean_balanced" / "c2_transformer" / "best_model.pth"
HOST_AWARE_CHECKPOINT = PROJECT_ROOT / "experiments" / "host_aware_uwf_attack_validation" / "best_host_aware_transformer.pth"

MODEL_OPTIONS = {
    "session": {
        "label": "Session Model",
        "input_format": "session_npz",
        "path": str(SESSION_MODEL_CHECKPOINT),
    },
    "domain_adaptive": {
        "label": "Domain-Adaptive Model",
        "input_format": "session_npz",
        "path": str(DEFAULT_EXPORT_DIR),
    },
    "host_aware": {
        "label": "Host-Aware Model",
        "input_format": "host_window_npz",
        "path": str(HOST_AWARE_CHECKPOINT),
    },
}

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


def _get_or_load_frozen_detector(checkpoint_path: str):
    """Load a non-exported detector checkpoint once and cache it."""
    cache_key = f"frozen::{checkpoint_path}"
    if cache_key not in _model_cache:
        from src.red_agent.detector_adapters import FrozenDetectorAdapter

        logger.info(f"Loading frozen detector from {checkpoint_path}...")
        _model_cache[cache_key] = FrozenDetectorAdapter(checkpoint_path, device="cpu")
        logger.info("Frozen detector loaded and cached")
    return _model_cache[cache_key]


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
            "available_models": MODEL_OPTIONS,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")


def _run_inference_sync(file_bytes: bytes, filename: str, model_type: str, export_dir: str) -> Dict[str, Any]:
    """Dispatch uploaded NPZ inference to the selected model family."""
    if model_type == "domain_adaptive":
        return _run_domain_adaptive_inference_sync(file_bytes, filename, export_dir)
    if model_type == "session":
        return _run_session_model_inference_sync(file_bytes, filename)
    if model_type == "host_aware":
        return _run_host_aware_inference_sync(file_bytes, filename)
    raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of: {', '.join(MODEL_OPTIONS)}")


def _run_domain_adaptive_inference_sync(file_bytes: bytes, filename: str, export_dir: str) -> Dict[str, Any]:
    """Synchronous inference worker — runs in thread pool.
    
    Loads NPZ from memory and runs batched model inference.
    No temp files are created (avoids Windows file locking issues).
    """
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

    labels = np.asarray(y, dtype=np.int64)

    # Transform to match frontend's expected Prediction interface:
    # { index, domain, probability, label (string), label_int (int) }
    preds = []
    for p in raw_preds:
        prediction = int(p["label"])
        target = int(labels[int(p["index"])])
        preds.append({
            "index": p["index"],
            "domain": p["domain"],
            "probability": p["prob"],
            "label": "C2" if prediction == 1 else "benign",
            "label_int": prediction,
            "ground_truth_label": "C2" if target == 1 else "benign",
            "ground_truth_label_int": target,
            "correct": bool(prediction == target),
        })

    logger.info(f"Inference complete: {len(preds)} predictions")
    return {
        "status": "success",
        "model_type": "domain_adaptive",
        "model_label": MODEL_OPTIONS["domain_adaptive"]["label"],
        "input_format": MODEL_OPTIONS["domain_adaptive"]["input_format"],
        "n_predictions": len(preds),
        "predictions": preds,
        "evaluation": _compute_binary_evaluation(preds),
    }


def _run_session_model_inference_sync(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Run the plain session Transformer checkpoint on a session NPZ."""
    import numpy as np
    import torch
    from src.data_loader.npz_utils import load_session_npz_from_bytes

    logger.info(f"Starting session-model inference on {filename} ({len(file_bytes)} bytes)")
    adapter = _get_or_load_frozen_detector(str(SESSION_MODEL_CHECKPOINT))
    X, y, masks = load_session_npz_from_bytes(
        file_bytes,
        expected_feature_names=adapter.feature_names,
        expected_session_len=adapter.session_len,
    )
    processed = adapter._preprocess_sessions(X, masks)
    probs = np.empty(len(X), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(X), 512):
            end = min(start + 512, len(X))
            logits = adapter.model(
                torch.from_numpy(processed[start:end]).to(adapter.device),
                torch.from_numpy(~masks[start:end]).to(adapter.device),
            )
            probs[start:end] = torch.sigmoid(logits).cpu().numpy()

    preds = _build_prediction_rows(
        probabilities=probs,
        labels=np.asarray(y, dtype=np.int64),
        threshold=float(adapter.threshold),
        domains=["session"] * len(probs),
    )
    return {
        "status": "success",
        "model_type": "session",
        "model_label": MODEL_OPTIONS["session"]["label"],
        "input_format": MODEL_OPTIONS["session"]["input_format"],
        "threshold": float(adapter.threshold),
        "n_predictions": len(preds),
        "predictions": preds,
        "evaluation": _compute_binary_evaluation(preds),
    }


def _run_host_aware_inference_sync(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Run the host-aware Transformer checkpoint on a host-window NPZ."""
    import numpy as np
    import torch
    from src.data_loader.host_window_dataset import HOST_AWARE_SCHEMA_VERSION

    logger.info(f"Starting host-aware inference on {filename} ({len(file_bytes)} bytes)")
    arrays = _load_host_windows_from_bytes(file_bytes)
    if arrays.schema_version != HOST_AWARE_SCHEMA_VERSION:
        raise ValueError(
            f"Host-aware model requires host-window NPZ schema {HOST_AWARE_SCHEMA_VERSION}, got {arrays.schema_version}"
        )

    adapter = _get_or_load_frozen_detector(str(HOST_AWARE_CHECKPOINT))
    if arrays.current_sessions.shape[2] != adapter.feature_dim:
        raise ValueError(
            f"Host-aware feature mismatch: model expects {adapter.feature_dim}, "
            f"uploaded file has {arrays.current_sessions.shape[2]}"
        )

    current = adapter._preprocess_sessions(arrays.current_sessions, arrays.current_masks)
    history = adapter._preprocess_history(arrays.history_sessions, arrays.history_flow_masks)
    host_features = adapter.host_feature_normalizer.transform(arrays.host_features)
    domain_ids = np.asarray(
        [adapter._resolve_domain_id(source=str(source), domain_id=None) for source in arrays.sources],
        dtype=np.int64,
    )
    probs = np.empty(len(arrays.labels), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(arrays.labels), 256):
            end = min(start + 256, len(arrays.labels))
            logits = adapter.model(
                torch.from_numpy(current[start:end]).to(adapter.device),
                torch.from_numpy(~arrays.current_masks[start:end]).to(adapter.device),
                torch.from_numpy(history[start:end]).to(adapter.device),
                torch.from_numpy(~arrays.history_flow_masks[start:end]).to(adapter.device),
                torch.from_numpy(arrays.history_session_masks[start:end].astype(bool)).to(adapter.device),
                torch.from_numpy(host_features[start:end]).to(adapter.device),
                torch.from_numpy(domain_ids[start:end]).to(adapter.device),
            )
            probs[start:end] = torch.sigmoid(logits).cpu().numpy()

    preds = _build_prediction_rows(
        probabilities=probs,
        labels=np.asarray(arrays.labels, dtype=np.int64),
        threshold=float(adapter.threshold),
        domains=[str(source) for source in arrays.sources],
    )
    for row, host_id, dest_id in zip(preds, arrays.host_ids, arrays.dest_ids):
        row["host_id"] = str(host_id)
        row["dest_id"] = str(dest_id)

    return {
        "status": "success",
        "model_type": "host_aware",
        "model_label": MODEL_OPTIONS["host_aware"]["label"],
        "input_format": MODEL_OPTIONS["host_aware"]["input_format"],
        "threshold": float(adapter.threshold),
        "n_predictions": len(preds),
        "predictions": preds,
        "evaluation": _compute_binary_evaluation(preds),
    }


def _load_host_windows_from_bytes(file_bytes: bytes):
    """Load HostWindowArrays from uploaded bytes without writing a temp file."""
    import numpy as np
    from src.data_loader.host_window_dataset import HostWindowArrays

    data = np.load(io.BytesIO(file_bytes), allow_pickle=True)
    try:
        labels = data["y"].astype(np.int64) if "y" in data else data["labels"].astype(np.int64)
        return HostWindowArrays(
            current_sessions=data["current_sessions"].astype(np.float32),
            current_masks=data["current_masks"].astype(bool),
            history_sessions=data["history_sessions"].astype(np.float32),
            history_flow_masks=data["history_flow_masks"].astype(bool),
            history_session_masks=data["history_session_masks"].astype(bool),
            host_features=data["host_features"].astype(np.float32),
            labels=labels,
            host_ids=data["host_ids"].astype(str),
            dest_ids=data["dest_ids"].astype(str),
            timestamps=data["timestamps"].astype(np.float64),
            sources=data["sources"].astype(str),
            families=data["families"].astype(str),
            capture_ids=data["capture_ids"].astype(str),
            original_indices=data["original_indices"].astype(np.int64),
            feature_names=tuple(str(name) for name in data["feature_names"].tolist()),
            host_feature_names=tuple(str(name) for name in data["host_feature_names"].tolist()),
            schema_version=_np_scalar_string(data["schema_version"]),
            session_schema_version=_np_scalar_string(data["session_schema_version"]),
            history_size=int(data["history_size"]),
            real_host_identity=bool(data["real_host_identity"]),
        )
    finally:
        data.close()


def _np_scalar_string(value: Any) -> str:
    return str(value.item() if hasattr(value, "item") else value)


def _build_prediction_rows(
    *,
    probabilities,
    labels,
    threshold: float,
    domains,
) -> list[dict[str, Any]]:
    rows = []
    for index, (probability, target, domain) in enumerate(zip(probabilities, labels, domains)):
        prediction = int(float(probability) >= threshold)
        target_int = int(target)
        rows.append(
            {
                "index": index,
                "domain": str(domain),
                "probability": float(probability),
                "threshold": float(threshold),
                "label": "C2" if prediction == 1 else "benign",
                "label_int": prediction,
                "ground_truth_label": "C2" if target_int == 1 else "benign",
                "ground_truth_label_int": target_int,
                "correct": bool(prediction == target_int),
            }
        )
    return rows


def _compute_binary_evaluation(predictions: list[dict[str, Any]]) -> Dict[str, Any]:
    """Compute binary classification metrics from prediction rows with labels."""
    if not predictions or "ground_truth_label_int" not in predictions[0]:
        return {"has_labels": False}

    y_true = [int(row["ground_truth_label_int"]) for row in predictions]
    y_pred = [int(row["label_int"]) for row in predictions]

    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    total = len(predictions)

    def safe_div(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "has_labels": True,
        "total": total,
        "correct": tp + tn,
        "incorrect": fp + fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
        "ground_truth": {
            "c2": tp + fn,
            "benign": tn + fp,
        },
        "predicted": {
            "c2": tp + fp,
            "benign": tn + fn,
        },
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
    }


@router.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    export_dir: Optional[str] = None,
    model_type: str = Query("domain_adaptive", description="Model to use: session, domain_adaptive, or host_aware"),
) -> Dict[str, Any]:
    """
    Upload NPZ file and get predictions for all sessions.
    
    Args:
        file: NPZ file containing session data
        export_dir: Optional path to exported model artifacts
    
    Returns:
        Dictionary with predictions for each session
    """
    export_dir = export_dir or str(DEFAULT_EXPORT_DIR)
    model_type = model_type.strip().lower()
    if model_type not in MODEL_OPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model_type '{model_type}'. Expected one of: {', '.join(MODEL_OPTIONS)}",
        )
    
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
            model_type,
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
