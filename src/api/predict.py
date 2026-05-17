"""FastAPI wrapper for the exported domain-adaptive inference artifacts.

Endpoints:
 - GET /health: health check
 - POST /predict-file: multipart file upload (.npz) -> returns predictions JSON
 - POST /predict-json: JSON body with `session` and `mask` arrays -> single prediction

Run with: `uvicorn src.api.predict:app --reload --port 8000`
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.run_inference import load_exported, run_on_npz, detect_domain_by_centroid, session_to_vector

app = FastAPI(
    title="CyberShield Inference API",
    description="Real-time C2 traffic detection using domain-adaptive transformer"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development/demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default export directory
DEFAULT_EXPORT_DIR = "experiments/domain_adaptive_sweep_20/exported"


class SessionPayload(BaseModel):
    session: list[list[float]]
    mask: list[int]
    domain: str | None = None


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Verify that we can load the exported artifacts
        ckpt, thresholds, centroids = load_exported(DEFAULT_EXPORT_DIR)
        return {
            "status": "ok",
            "model": "domain-adaptive-transformer",
            "domains": list(thresholds.keys()) if thresholds else ["mixed", "uwf"],
            "export_dir": DEFAULT_EXPORT_DIR
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")


@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    export_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Upload NPZ file and get predictions."""
    export_dir = export_dir or DEFAULT_EXPORT_DIR
    
    if not file.filename.endswith('.npz'):
        raise HTTPException(status_code=400, detail="Only .npz uploads supported for this endpoint")

    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / file.filename
            contents = await file.read()
            tmp_path.write_bytes(contents)

            ckpt, thresholds, centroids = load_exported(export_dir)
            out_path = Path(td) / "predictions.json"
            # run_on_npz will write predictions to out_path
            run_on_npz(export_dir, ckpt, thresholds, centroids, str(tmp_path), str(out_path))

            preds = json.loads(out_path.read_text())
            return {
                "status": "success",
                "n_predictions": len(preds),
                "predictions": preds
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict-json")
async def predict_json(
    payload: SessionPayload,
    export_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Get prediction for a single session (JSON input)."""
    export_dir = export_dir or DEFAULT_EXPORT_DIR
    
    try:
        ckpt, thresholds, centroids = load_exported(export_dir)

        # Convert session/mask to numpy arrays
        import numpy as np
        sess = np.asarray(payload.session, dtype=float)[None, ...]
        mask = np.asarray(payload.mask, dtype=int)[None, ...]

        # domain detection (unless domain supplied)
        vec = session_to_vector(sess, mask)
        domain = payload.domain or (detect_domain_by_centroid(vec, centroids) if centroids else 'mixed')

        # Lazy import model creation to avoid heavy startup cost
        from src.models.domain_adaptive_transformer import DomainAdaptiveC2Transformer
        domains = ckpt.get('domains') or ['mixed', 'uwf']
        model = DomainAdaptiveC2Transformer(domains=domains, feature_dim=int(ckpt.get('feature_dim', sess.shape[2])), seq_len=int(ckpt.get('session_len', sess.shape[1])))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        import torch
        x_t = torch.tensor(sess, dtype=torch.float32)
        m_t = torch.tensor(~mask, dtype=torch.bool)
        domain_id = domains.index(domain) if domain in domains else 0
        with torch.no_grad():
            logits = model(x_t, m_t, torch.tensor([domain_id]))
            prob = float(torch.sigmoid(logits).cpu().item())

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
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
