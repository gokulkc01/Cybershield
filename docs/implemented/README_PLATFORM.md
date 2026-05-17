# CyberShield v2 - Behavioral Adversarial Robustness Intelligence Platform

**A complete full-stack demo platform for behavioral C2 detection and adversarial robustness evaluation.**

## 🎯 What is CyberShield v2?

CyberShield is **NOT** a simple malware classifier. It's a **behavioral adversarial robustness intelligence platform** that:

1. **Detects behavioral C2 patterns** using a session-centric Transformer
2. **Evaluates adversarial robustness** through deterministic mutation testing
3. **Visualizes behavioral resilience** in an interactive research platform
4. **Demonstrates cross-family generalization** (96.36% recall on unseen families)

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.9+
- Node.js 18+
- 2 GB RAM minimum

### 1. Start Backend
```bash
cd backend

# Install dependencies (first time only)
pip install -r requirements.txt

# Run server
python main.py
```
Backend runs at `http://localhost:8000`
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 2. Start Frontend
```bash
cd frontend

# Install dependencies (first time only)
npm install

# Run dev server
npm run dev
```
Frontend runs at `http://localhost:3000`

### 3. Open in Browser
Open http://localhost:3000 and explore the platform!

## 📋 Platform Features

### Dashboard 📊
- High-level metrics overview
- Total sessions, detection rate, robustness scores
- Risk distribution and mutation robustness charts

### Session Analysis 🔍
- Per-session behavioral inspection
- Risk scores and confidence metrics
- Searchable session table

### Mutation Lab ⭐ (Primary Differentiator)
**Interactive before/after robustness testing**
- Configure mutations (timing, TLS, flow-level)
- Adjust severity dynamically
- Visualize detection changes
- Analyze feature shifts

### Robustness Analytics 🛡️
- Research-grade metrics
- Mutation impact ranking
- Recall vs FPR comparison
- Behavioral invariance tracking

### Upload & Evaluation 📤
- Drag-and-drop file upload
- NPZ, JSONL, CSV, Zeek log support
- File metadata extraction

## 🏗️ Architecture

```
Frontend (Next.js/React/TypeScript)
    ↓ REST API
FastAPI Backend (Python async)
    ↓ Services
CyberShield ML Engine
    ├─ Transformer Detector
    ├─ Red-Agent Mutation Engine (11 operators)
    └─ Robustness Evaluator
    ↓ Storage
Artifacts (NPZ, JSON, JSONL)
```

## 📁 Key Files

**Backend**
- `backend/main.py` - FastAPI app entry point
- `backend/app/api/` - REST endpoints (detection, mutation, robustness, artifacts)
- `backend/app/services/` - Core logic (ML inference, mutations, evaluation)
- `backend/BACKEND_ARCHITECTURE.md` - Complete API documentation

**Frontend**
- `frontend/app/page.tsx` - Main app with navigation
- `frontend/components/pages/` - 5 demo pages
- `frontend/lib/api.ts` - API client
- `frontend/lib/store.ts` - Global state (Zustand)

## 🔌 API Endpoints

```
Detection:
  POST /api/v1/detection/session       - Analyze single session
  POST /api/v1/detection/batch         - Analyze batch
  POST /api/v1/detection/dataset       - Evaluate full dataset

Mutation:
  GET /api/v1/mutation/available       - List mutations
  POST /api/v1/mutation/apply          - Apply mutations
  POST /api/v1/mutation/evaluate       - Evaluate robustness

Robustness:
  GET /api/v1/robustness/report/{id}   - Get report
  GET /api/v1/robustness/metrics/{id}  - Get metrics
  POST /api/v1/robustness/failures     - Analyze failures

Artifacts:
  POST /api/v1/artifacts/upload        - Upload file
  GET /api/v1/artifacts/file/{id}      - Download file
  GET /api/v1/artifacts/list           - List files
```

See `backend/BACKEND_ARCHITECTURE.md` for complete documentation.

## 🧬 Available Mutations

CyberShield integrates **11 deterministic adversarial mutations**:

**Timing (3)**
- `timing_jitter` - Random communication delays
- `burst_callback` - Packet bursting
- `delayed_reconnect` - Extended reconnections

**Persistence (3)**
- `low_frequency_callback` - Reduced communication frequency
- `intermittent_communication` - Intermittent patterns
- `long_sleep` - Extended sleep periods

**TLS/Encryption (3)**
- `tls_padding` - TLS protocol padding
- `session_reuse_shape` - Vary session shapes
- `handshake_variation` - Modify handshakes

**Flow-level (2)**
- `packet_count_variation` - Vary packet counts
- `byte_distribution_shift` - Shift byte distributions

All mutations are **deterministic** (seeded for reproducibility).

## 📊 Demo Workflow

**Recommended user journey:**

1. **Open Dashboard** → See baseline metrics (96.36% recall)
2. **Visit Upload** → Upload test dataset
3. **Open Mutation Lab** → Configure mutations
4. **Adjust Severity** → Control mutation intensity
5. **Run Evaluation** → See robustness changes
6. **Analyze Results** → Understand behavioral resilience
7. **Visit Analytics** → Deep-dive research-grade metrics

## 🎨 Tech Stack

**Backend**
- FastAPI 0.104.1
- Python 3.9+
- PyTorch 2.1.1
- NumPy, SciPy, scikit-learn

**Frontend**
- Next.js 14
- React 18
- TypeScript 5
- Tailwind CSS 3
- Recharts (data visualization)
- Zustand (state management)
- Framer Motion (animations)

## 📚 Documentation

- **PLATFORM_IMPLEMENTATION_GUIDE.md** - Complete platform guide
- **backend/BACKEND_ARCHITECTURE.md** - API reference
- **RED_AGENT_QUICKSTART.md** - Mutation framework guide
- **RED_AGENT_QUICK_REFERENCE.md** - One-liners
- **RED_AGENT_V1_IMPLEMENTATION.md** - Technical design

## 🔍 Understanding the Differentiator

**The Mutation Lab is the PRIMARY differentiator.**

It visually demonstrates:
```
Original Session (96.36% detected as C2)
    ↓ Apply Mutation
Adversarially Mutated Session
    ↓ Re-run Detection
Result: 94.00% detected as C2
    ↓ Behavioral Shift Analysis
Feature changes: bytes_in +19KB, iat +177ms
    ↓ Robustness Score
Behavioral Invariance: 0.975 (resilient!)
```

This shows: **CyberShield maintains behavioral robustness under adversarial pressure.**

## ⚡ Performance

- **Detection**: ~100-1000 sessions/sec (CPU batch_size=256)
- **Mutation**: ~100-500 samples/sec
- **Evaluation**: ~50-200 samples/sec

GPU acceleration available (set `DEVICE=cuda` in backend config).

## 🛠️ Configuration

**Backend (.env or environment)**
```
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
CHECKPOINT_PATH=../experiments/multifamily_generalization/strict_smoke/best_transformer.pth
DEVICE=cpu  # or cuda
```

**Frontend (.env.local)**
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## 📦 Key Components

**ML Model**
- Session-centric Transformer (10 features, 20-step sequences)
- Trained on Neris + Kraken families
- 96.36% recall on unseen Conficker family
- Cross-family generalization validated

**Mutation Engine** (Red-Agent v1)
- 11 deterministic operators
- Seeded for reproducibility
- Per-sample metadata tracking
- Failure analysis integration

**Robustness Metrics**
- Detection recall/FPR under mutations
- Behavioral invariance score
- Feature shift analysis
- Fragile sample identification

## 🚨 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check port 8000 is available
lsof -i :8000  # Kill if needed: kill -9 <PID>
```

### Frontend won't start
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 18+
```

### API connection failed
- Ensure backend is running: http://localhost:8000/health
- Check frontend can reach backend (CORS enabled)
- Check both are on localhost:8000 and localhost:3000

## 🔗 Related Repositories

- Red-Agent v1: `src/red_agent/`
- ML Model: `src/models/transformer.py`
- Experiments: `experiments/multifamily_generalization/`
- Data: `data/processed/`

## 📝 Next Steps

1. **Test with real data**: Upload your dataset to the platform
2. **Interactive exploration**: Use Mutation Lab to test robustness
3. **Shadow deployment**: Evaluate on live network traffic
4. **User feedback**: Refine visualizations and workflows
5. **Production hardening**: Add authentication, database, etc.

See `NEXT_PHASE_PLAN.md` for detailed roadmap.

## 🤝 Contributing

To extend CyberShield:

1. **Add new page**: Create React component in `frontend/components/pages/`
2. **Add new API**: Create endpoint in `backend/app/api/`
3. **Add new mutation**: Register in `src/red_agent/mutation_registry.py`
4. **Add analytics**: Extend `backend/app/services/robustness.py`

## 📄 License

CyberShield v2 - Behavioral Adversarial Robustness Intelligence Platform

## 👨‍💻 Authors

Built by the Behavioral Cybersecurity Research Team

---

## 🎓 Core Message

> **CyberShield is fundamentally investigating which behavioral properties distinguish adversarial remote-control systems from legitimate enterprise automation under mutation pressure.**

The platform should continuously reinforce this scientific purpose through its visualizations, workflows, and metrics.

---

**Ready to explore behavioral adversarial robustness intelligence?**

Open http://localhost:3000 and start with the Dashboard!
