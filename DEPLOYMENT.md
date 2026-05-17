# CyberShield Deployment Guide

## Quick Start

### 1. Start the FastAPI Backend

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run the API server (default port 8000)
uvicorn src.api.predict:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

**Health Check:**
```bash
curl http://localhost:8000/health
```

### 2. Open the Frontend Demo

Open [docs/model-demo.html](model-demo.html) in your web browser:
- Drag and drop an NPZ file to analyze
- Or click "Browse Files" to select a file
- The backend will process it and display real-time predictions

**Requirements:**
- Modern web browser (Chrome, Firefox, Safari, Edge)
- NPZ file containing sessions in the expected format
- FastAPI backend running on the configured URL (default: `http://localhost:8000`)

## API Endpoints

### Health Check
```
GET /health
```
Returns status and model metadata.

**Example Response:**
```json
{
  "status": "ok",
  "model": "domain-adaptive-transformer",
  "domains": ["mixed", "uwf"],
  "export_dir": "experiments/domain_adaptive_sweep_20/exported"
}
```

### File Upload Inference
```
POST /predict-file
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): `.npz` file containing session data
- `export_dir` (optional): Path to exported model artifacts (default: `experiments/domain_adaptive_sweep_20/exported`)

**Example Response:**
```json
{
  "status": "success",
  "n_predictions": 1624,
  "predictions": [
    {
      "index": 0,
      "domain": "mixed",
      "probability": 0.9812,
      "label": "C2",
      "label_int": 1
    },
    ...
  ]
}
```

### JSON Session Inference
```
POST /predict-json
Content-Type: application/json
```

**Request Body:**
```json
{
  "session": [[f1, f2, ..., f45], [f1, f2, ..., f45], ...],
  "mask": [0, 0, 1, 1, ...],
  "domain": "uwf"
}
```

**Example Response:**
```json
{
  "status": "success",
  "domain": "uwf",
  "probability": 0.9812,
  "label": "C2",
  "label_int": 1
}
```

## Production Model

**Best Checkpoint:** `experiments/domain_adaptive_sweep_20/best_transformer.pth`

**Exported Artifacts:** `experiments/domain_adaptive_sweep_20/exported/`

Contents:
- `model_checkpoint.pth` - PyTorch model state dict
- `thresholds.json` - Per-domain classification thresholds
- `domain_centroids.json` - Session centroids for domain detection
- `feature_normalizer.json` - Feature normalization parameters
- `feature_transform_config.json` - Feature transformation config
- `export_manifest.json` - Metadata manifest

**Metrics (Multi-seed Validation):**
- Mixed domain test: AUC 0.9961±0.0011, F1 0.9067±0.0040
- UWF domain test: AUC 1.0, F1 0.9811±0.0094
- False Positive Rate: ~0.5% (calibrated)
- True Positive Rate: ~83% (mixed), 100% (UWF)

## Model Architecture

**Type:** Domain-Adaptive C2 Transformer

**Components:**
- Shared transformer backbone (encoder)
- Domain-specific classification heads
- Domain detector (centroid-based, cosine similarity)
- Per-domain calibration thresholds

**Input:**
- Session shape: (batch_size, 20 flows, 45 features)
- Padding mask: (batch_size, 20) boolean array

**Output:**
- Binary classification: C2 (1) or Benign (0)
- Probability: 0-1 range (model confidence)
- Domain: "mixed" or "uwf" (auto-detected or provided)

## Deprecated Models (Reference Only)

These models are no longer recommended for production but kept for reproducibility:

### Single-Head Baselines
- `experiments/extended_mixed_real_balanced_long/best_transformer.pth` - Mixed training (no UWF support)
- `experiments/ctu_mcfp_only_balanced/best_transformer.pth` - CTU+MCFP only
- `experiments/transformer_uwf_only/best_transformer.pth` - UWF only

### Continual Learning Attempts
- `experiments/continual_mixed_replay_uwf/best_transformer.pth` - Rehearsal+distillation (partial success)
- `experiments/ctu_mcfp_to_uwf_finetune/best_transformer.pth` - Fine-tuning approach

### Early Experimental Models
- `experiments/transformer/best_transformer.pth`
- `experiments/multifamily_generalization/strict_smoke/best_transformer.pth`

**Why Deprecated:**
- Single-head models cannot handle multi-domain scenarios simultaneously
- Continual learning approach showed calibration failures
- Early models lack proper per-domain thresholds
- Domain-adaptive model superior performance across all metrics

## Training Data

**Mixed Domain Training Set:**
- CTU-13 C2 flows: ~1,000 sessions
- MCFP C2 flows: ~1,500 sessions
- MCFP Benign flows: ~10,000 sessions
- Total: ~12,500 sessions

**UWF Adaptation Set:**
- UWF Benign flows: ~8,000 sessions
- UWF C2 flows (if available): included in test
- Train/Val/Test: 70/15/15 split

## Feature Schema

**Extended V1 (45 features):**
- 10 Base IP features (duration, bytes, packets, flags, etc.)
- 12 TLS features (cipher, version, extensions, etc.)
- 10 DNS features (queries, responses, TTL, etc.)
- 13 Temporal features (inter-arrival times, burstiness, etc.)

**Normalization:**
- Min-max scaling to [0, 1] range
- Per-feature statistics in `feature_normalizer.json`

## Troubleshooting

### "Model loading failed" on /health
- Verify export directory path exists
- Check model checkpoint file: `experiments/domain_adaptive_sweep_20/exported/model_checkpoint.pth`
- Ensure all export files are present (see "Production Model" section)

### "Inference failed" on predict-file
- Validate NPZ file format (must contain valid session data)
- Check feature count matches 45 (extended_v1 schema)
- Verify session length is 20 flows
- Enable debug logging in FastAPI (use `--reload` flag)

### Low confidence predictions
- May indicate domain drift or unusual session patterns
- Cross-check with manual inspection of raw flow data
- Consider retraining on recent data if distribution has shifted

### CORS errors in browser
- CORS middleware is enabled with `allow_origins=["*"]`
- If errors persist, check browser console for details
- Verify API URL matches exactly (e.g., `http://localhost:8000`)

## Development Commands

### Run API with Hot Reload
```bash
uvicorn src.api.predict:app --reload --port 8000
```

### Run API on Different Port
```bash
uvicorn src.api.predict:app --reload --port 9000
```

### Test API Locally
```bash
# Health check
python -c "
import requests
r = requests.get('http://localhost:8000/health')
print(r.json())
"

# Single session inference
python -c "
import requests
import json
payload = {
    'session': [[0.1]*45 for _ in range(20)],
    'mask': [0]*20,
    'domain': 'mixed'
}
r = requests.post('http://localhost:8000/predict-json', json=payload)
print(r.json())
"
```

## Performance Notes

- API startup: ~2-5 seconds (model loading)
- NPZ inference: ~50-200ms per file (depends on session count)
- Memory usage: ~2GB (loaded model + CUDA if available)
- Latency per single session: ~5-10ms

## Next Steps

1. **Containerization:** Create Docker image for deployment
2. **Authentication:** Add API key validation for production
3. **Rate Limiting:** Implement request throttling
4. **Monitoring:** Add Prometheus metrics and health monitoring
5. **Model Updates:** Plan retraining strategy for new data
6. **A/B Testing:** Compare with newer model versions in production

---

**Last Updated:** May 2026  
**Model Version:** Domain-Adaptive Sweep 20 (Seed Ensemble)  
**Status:** Production Ready ✅
