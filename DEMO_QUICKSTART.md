# CyberShield Model Demo - Quick Reference

## 🚀 Get Started in 3 Steps

### Step 1: Start the Backend API
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Start the FastAPI server
uvicorn src.api.predict:app --reload --port 8000
```

You should see:
```
Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Verify the API is Running
```powershell
# In a new terminal, test the health endpoint
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "model": "domain-adaptive-transformer",
  "domains": ["mixed", "uwf"],
  "export_dir": "experiments/domain_adaptive_sweep_20/exported"
}
```

### Step 3: Open the Frontend Demo
```
Open: docs/model-demo.html
Or: Right-click -> Open with Browser
```

## 📊 Using the Demo

### Upload & Analyze
1. **Drag and drop** an NPZ file onto the upload area
2. Or click **"Browse Files"** to select
3. Click **"🚀 Analyze File"** to start inference
4. Results display automatically

### What You'll See
- **Summary Statistics:**
  - Total sessions processed
  - Number of C2 detections vs benign
  - Detection rate (%)
  - Average model confidence
  
- **Detailed Predictions Table:**
  - Index: Session ID
  - Domain: auto-detected ("mixed" or "uwf")
  - Label: Classification result ("C2" or "Benign")
  - Probability: Confidence score (0-100%)

### API Configuration
The demo defaults to `http://localhost:8000`

To use a different API:
1. Paste the API URL in the **"API Endpoint"** field
2. Upload file and analyze

## 🔧 Testing with Sample Data

### Option 1: Use Existing Test Data
```powershell
# Run inference on existing UWF test set
python -c "
import numpy as np
npz = np.load('data/processed/uwf_adaptation_split/test_sessions.npz')
print('Sessions:', npz['sessions'].shape)
print('Labels:', npz['labels'].shape)
"
```

Then upload to demo and verify predictions.

### Option 2: Create a Small Test NPZ
```python
import numpy as np

# Create sample session (20 flows, 45 features)
session = np.random.randn(20, 45).astype(np.float32)
label = np.array([1], dtype=np.int64)  # 1 = C2, 0 = benign
mask = np.array([[0]*20], dtype=np.int64)

# Save as NPZ
np.savez('test_sample.npz', 
         sessions=session[None, ...],
         labels=label,
         masks=mask)

print("Created test_sample.npz")
```

Then drag `test_sample.npz` into the demo.

## 📈 Interpreting Results

### High Confidence C2 (e.g., 95%+)
- Likely malicious behavior detected
- Review raw flow data for confirmation
- Check domain (mixed vs UWF) for context

### Low Confidence Predictions (50-70%)
- Ambiguous behavior
- May indicate novel attack pattern
- Consider manual inspection

### Domain Detection
- **"mixed"** → Matches CTU/MCFP characteristics
- **"uwf"** → Matches UWF characteristics
- Auto-detection based on session features
- Influences which threshold is applied

## 🐛 Troubleshooting

### "Connection refused" or "ERR_CONNECTION_REFUSED"
- Is the backend running? Check terminal for `Uvicorn running...`
- Is the port correct? (default: 8000)
- Try: `curl http://localhost:8000/health`

### "HTTP 500 - Inference failed"
- Check NPZ file format (must contain sessions)
- Verify feature count (should be 45)
- Check session length (should be 20 flows)
- See terminal logs for detailed error

### No results displayed
- Check browser console (F12 → Console tab)
- Look for error messages in red
- Verify API response in Network tab

### Slow performance
- First request slower (~2-5s) due to model loading
- Subsequent requests faster (~200ms)
- Large NPZ files (1000+ sessions) take longer

## 💡 Tips & Tricks

### Batch Processing
- You can reuse the demo for multiple files
- Just upload another file and click analyze again
- Results update in the right panel

### Copy Results
- Results table shows first 100 predictions
- Full results in API response (browser console)
- Export to CSV (future enhancement)

### Performance Monitoring
- Watch the progress bar during analysis
- Check API logs in terminal
- Monitor memory usage with `nvidia-smi` (if GPU available)

## 📚 More Information

- **Deployment Guide:** See [DEPLOYMENT.md](../DEPLOYMENT.md)
- **Model Details:** See [MODEL_CLEANUP.md](../MODEL_CLEANUP.md)
- **API Docs:** Visit `http://localhost:8000/docs` (auto-generated Swagger UI)
- **Raw API Docs:** Visit `http://localhost:8000/redoc`

## 🎯 Next Steps

1. **Analyze your data:** Use the demo with your NPZ files
2. **Validate results:** Cross-check with manual inspection
3. **Deploy to production:** See DEPLOYMENT.md for hardening steps
4. **Monitor performance:** Set up logging and metrics collection
5. **Retrain periodically:** Plan model updates on new data

---

**Quick Command Reference:**

```powershell
# Start API
uvicorn src.api.predict:app --reload --port 8000

# Test API health
curl http://localhost:8000/health

# Test single prediction
curl -X POST http://localhost:8000/docs

# View API logs
# (watch terminal where uvicorn is running)

# Stop API
# (Ctrl+C in the terminal)
```

**Browser Shortcuts:**
- F12: Developer console (check for errors)
- Ctrl+Shift+N: Incognito mode (clear cache)
- Ctrl+L: Select address bar (copy API URL)

---

**Last Updated:** May 2026  
**Status:** Ready for Demo ✅
