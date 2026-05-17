# CyberShield v2 - Architecture & System Design

## HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                            │
│                   (Frontend: Next.js/React/TypeScript)               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │Dashboard │  │ Analysis │  │Mutation  │  │Robustness│ ┌────────┐ │
│  │   📊     │  │   🔍     │  │Lab ⭐    │  │Analytics │ │ Upload │ │
│  │  Page    │  │  Page    │  │  Page    │  │   Page   │ │  📤    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ └────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Global State Management (Zustand)              │   │
│  │  - Dataset files  - Mutations  - Results  - UI state       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │        API Client (axios wrapper)                            │   │
│  │  Typed endpoint calls + error handling                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
              ↓↑  REST API (HTTP/JSON)
┌─────────────────────────────────────────────────────────────────────┐
│                      API LAYER                                       │
│                   (FastAPI Backend: Python)                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │  Detection API │  │  Mutation API  │  │ Robustness API │         │
│  │  (3 endpoints) │  │  (3 endpoints) │  │  (4 endpoints) │ ┌─────┐ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │ Art │ │
│                                                              │ fact│ │
│  Health Check + Root endpoint                               │ API │ │
│                                                              │(6)  │ │
└─────────────────────────────────────────────────────────────────────┘
              ↓↑  Service calls + Config
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVICES LAYER                                  │
│                    (Core Business Logic)                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ Detection Service│  │ Mutation Service │  │Robustness Service│   │
│  │                  │  │                  │  │                  │   │
│  │ • Load model     │  │ • Load registry  │  │ • Compare detect │   │
│  │ • Extract feat   │  │ • Apply mutations│  │ • Compute metrics│   │
│  │ • Batch infer    │  │ • Generate meta  │  │ • Analyze fails  │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         Artifact Service (File Management)                   │   │
│  │ • Upload files  • Store metadata  • Retrieve results         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
              ↓↑  File I/O + Model calls
┌─────────────────────────────────────────────────────────────────────┐
│                      ML ENGINE LAYER                                 │
│                   (Detection & Mutation)                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐              ┌──────────────────────┐    │
│  │ Transformer Detector │              │ Red-Agent v1         │    │
│  │                      │              │ Mutation Engine      │    │
│  │ • Checkpoint load    │              │                      │    │
│  │ • Feature extraction │              │ • 11 operators       │    │
│  │ • Inference (batch)  │              │ • Deterministic      │    │
│  │ • Risk scoring       │              │ • Metadata tracking  │    │
│  │ • Confidence est.    │              │ • Seeded mutations   │    │
│  └──────────────────────┘              └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
              ↓↑  Data I/O
┌─────────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                                   │
│                    (Artifact Persistence)                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   NPZ Files  │  │  JSON Files  │  │ JSONL Files  │               │
│  │  (Datasets)  │  │ (Reports)    │  │ (Metadata)   │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                       │
│  Filesystem Storage: ~/CyberShield/backend/artifacts/               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## DATA FLOW EXAMPLE: MUTATION LAB WORKFLOW

```
User opens Mutation Lab page
    ↓
Frontend loads (Zustand store initialized with empty state)
    ↓
User adds mutation: timing_jitter @ severity 0.3
    → Store updated: mutations = [{type: "timing_jitter", severity: 0.3}]
    ↓
User adds mutation: tls_padding @ severity 0.5
    → Store updated: mutations = [..., {type: "tls_padding", severity: 0.5}]
    ↓
User clicks "Run Evaluation" button
    ↓
Frontend calls API: POST /mutation/evaluate
    {
      baseline_npz_file_id: "file_123",
      mutated_npz_file_id: "file_456",
      checkpoint_name: "transformer"
    }
    ↓
Backend Mutation API receives request
    ↓
Service calls RobustnessService.evaluate_robustness()
    ↓
RobustnessService:
  1. Load baseline NPZ → Extract sessions
  2. Load mutated NPZ → Extract sessions
  3. Run batch inference on baseline (DetectionService)
  4. Run batch inference on mutated (DetectionService)
  5. Compare risk scores
  6. Compute metrics:
     - Baseline recall: 96.36%
     - Mutated recall: 94.00%
     - Recall degradation: -2.36%
     - Behavioral invariance: 0.975
     - FPR shift: +0.1%
  7. Analyze feature shifts
    ↓
Backend returns: RobustnessMetrics (JSON)
    ↓
Frontend updates store: robustnessResult = {...metrics...}
    ↓
Mutation Lab component re-renders with results
    ↓
User sees:
  - Before/After line chart
  - Baseline Recall: 96.36%
  - Mutated Recall: 94.00%
  - Behavioral Invariance: 0.975 (good!)
  - Feature shifts per feature
    ↓
User understands: "Mutations reduced detection by 2.36%, but behavioral
                  invariance is 0.975, so the detector is still resilient!"
```

---

## COMPONENT INTERACTION DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND (React)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Components          State              Utils              │
│  ─────────────       ──────              ─────              │
│  • Dashboard    ←→   Zustand    ←→  API Client             │
│  • Analysis     │    Store      │  • formatRiskScore       │
│  • MutationLab  │    ─────────  │  • getRiskBadgeColor     │
│  • Robustness   │    • dataset  │  • formatBytes           │
│  • Upload       │    • mutations│  • fileToBase64          │
│                 │    • results  │                          │
│  Navigation     │    • ui       │                          │
│                 └────────────────                          │
│                        ↓                                    │
└─────────────────────────────────────────────────────────────┘
              ↓ REST API calls
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Routes          Services           Schemas               │
│  ──────────      ─────────────      ────────              │
│  /detection  ←→  Detection    ←→  • SessionData           │
│  /mutation   │   Service      │    • DetectionResult      │
│  /robustness │   ─────────────     • MutationRequest      │
│  /artifacts  │   Mutation    ←→  • MutationResult         │
│              │   Service      │    • RobustnessMetrics    │
│  Config ──→ │   ─────────────     • FileUploadResponse    │
│              │   Robustness  ←→                           │
│              │   Service      │                           │
│              │   ─────────────                            │
│              │   Artifact    ←→                           │
│              │   Service      │                           │
│              └────────────────                            │
│                        ↓                                    │
└─────────────────────────────────────────────────────────────┘
              ↓ File I/O + Model calls
┌─────────────────────────────────────────────────────────────┐
│                   ML ENGINE + STORAGE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Transformer        Red-Agent          Artifacts           │
│  ──────────────     ──────────         ────────            │
│  • Model loaded     • 11 operators     • NPZ datasets      │
│  • Batch infer      • Seeded          • JSON reports      │
│  • Feature extract  • Deterministic   • JSONL metadata    │
│  • Risk scores      • Metadata        • Checksums         │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## REQUEST/RESPONSE FLOW: SINGLE SESSION ANALYSIS

```
Frontend Action:
  User enters session data and clicks "Analyze"
    ↓
Frontend (session-analysis.tsx):
  1. Collect 10 session features
  2. Call api.analyzeSession(sessionData)
    ↓
API Client (lib/api.ts):
  1. POST /api/v1/detection/session
  2. Body: SessionData (10 fields)
    ↓
Backend (app/api/detection.py):
  @router.post("/session")
  async def analyze_session(data: SessionData):
    ↓
Detection Service (app/services/detection.py):
  1. extract_features(data) → numpy array [10]
  2. infer_single(array) → (risk_score, is_suspicious, confidence)
    ↓
ML Engine (Transformer):
  1. Forward pass through model
  2. Sigmoid activation
  3. Return scalar [0, 1]
    ↓
Backend Response:
  {
    "session_id": "...",
    "risk_score": 0.96,
    "is_suspicious": true,
    "confidence": 0.98,
    "behavioral_features": {...},
    "timestamp": "2026-05-11T12:00:00Z"
  }
    ↓
Frontend receives DetectionResult (typed)
    ↓
Frontend (session-analysis.tsx):
  1. Update component state with results
  2. Display result in detail card
  3. Show risk badge (color-coded)
  4. Render confidence meter
    ↓
User sees: "Risk Score: 0.96 [SUSPICIOUS]"
```

---

## STATE MANAGEMENT FLOW: ZUSTAND STORE

```
┌────────────────────────────────────────────────────────┐
│              Zustand Store (lib/store.ts)               │
├────────────────────────────────────────────────────────┤
│                                                         │
│  AppState {                                             │
│    // Dataset Management                              │
│    baselineFileId: string | null                       │
│    mutatedFileId: string | null                        │
│    setBaselineFileId(id: string)                       │
│    setMutatedFileId(id: string)                        │
│                                                         │
│    // Mutation Configuration                          │
│    mutations: MutationConfig[]                         │
│    addMutation(config: MutationConfig)                 │
│    removeMutation(index: number)                       │
│    updateMutation(index: number, config)              │
│    clearMutations()                                    │
│                                                         │
│    // Results & Evaluation                             │
│    robustnessResult: RobustnessResult | null          │
│    setRobustnessResult(result: RobustnessResult)      │
│                                                         │
│    // UI State                                         │
│    isLoading: boolean                                  │
│    error: string | null                               │
│    setLoading(loading: boolean)                        │
│    setError(error: string | null)                      │
│    selectedPage: PageType                              │
│    setSelectedPage(page: PageType)                     │
│                                                         │
│    // Global Reset                                     │
│    reset()                                             │
│  }                                                      │
│                                                         │
└────────────────────────────────────────────────────────┘

How Components Use It:
  const store = useAppStore()
  const mutations = store.mutations
  store.addMutation({type: "timing_jitter", severity: 0.3})
  
Persist Across Pages:
  Upload page sets: baselineFileId
  Mutation Lab reads: baselineFileId
  Mutation Lab sets: robustnessResult
  Analytics page reads: robustnessResult
```

---

## SERVICE DEPENDENCY GRAPH

```
┌──────────────────────────────────────────────────┐
│         Frontend Components                       │
└──────────────────────────────────────────────────┘
              ↓ (API calls)
┌──────────────────────────────────────────────────┐
│         API Routes                                │
│  ┌─────────────────────────────────────────────┐ │
│  │ detection.py (3 endpoints)                  │ │
│  │  ├─ POST /session                           │ │
│  │  ├─ POST /batch                             │ │
│  │  └─ POST /dataset                           │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ mutation.py (3 endpoints)                   │ │
│  │  ├─ GET /available                          │ │
│  │  ├─ POST /apply                             │ │
│  │  └─ POST /evaluate ──→ robustness.py        │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ robustness.py (4 endpoints)                 │ │
│  │  ├─ GET /report/{id}                        │ │
│  │  ├─ GET /metrics/{id}                       │ │
│  │  ├─ POST /failures ──→ detection.py         │ │
│  │  └─ POST /compare ──→ detection.py          │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ artifacts.py (6 endpoints)                  │ │
│  │  ├─ POST /upload ──→ artifact.py            │ │
│  │  ├─ GET /file/{id}                          │ │
│  │  ├─ GET /metadata/{id}                      │ │
│  │  ├─ GET /list                               │ │
│  │  ├─ DELETE /file/{id}                       │ │
│  │  └─ POST /report                            │ │
│  └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
              ↓ (Service calls)
┌──────────────────────────────────────────────────┐
│         Services                                  │
│  ┌─────────────────────────────────────────────┐ │
│  │ detection.py                                │ │
│  │  ├─ initialize()                            │ │
│  │  ├─ extract_features()                      │ │
│  │  ├─ infer_single()                          │ │
│  │  └─ infer_batch()                           │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ mutation.py                                 │ │
│  │  ├─ initialize()                            │ │
│  │  ├─ list_mutations()                        │ │
│  │  ├─ apply_mutation()                        │ │
│  │  └─ batch_mutate()                          │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ robustness.py                               │ │
│  │  ├─ evaluate_robustness() ──→ detection    │ │
│  │  ├─ analyze_failures()                      │ │
│  │  ├─ _batch_predict()                        │ │
│  │  └─ _compute_recall()                       │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ artifact.py                                 │ │
│  │  ├─ upload_file()                           │ │
│  │  ├─ get_file()                              │ │
│  │  ├─ list_files()                            │ │
│  │  └─ save_report()                           │ │
│  └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
              ↓ (Model calls + File I/O)
┌──────────────────────────────────────────────────┐
│         ML Engine + Storage                       │
│  • Transformer Detector                           │
│  • Red-Agent Mutation Engine                      │
│  • Artifact Filesystem                            │
└──────────────────────────────────────────────────┘

Key Dependencies:
- mutation.py → detection.py (for comparing behavior)
- robustness.py → detection.py (for inference)
- All routes → artifact.py (for file management)
- All services → ML engine (for inference/mutation)
```

---

## MUTATION LAB VISUALIZATION FLOW

```
User Interface (React Component)
    ↓
Mutation Configuration Sidebar
  ├─ Add Mutation Button → Show available mutations dialog
  ├─ Remove Mutation Button → Filter from array
  ├─ Severity Slider → Update store with severity value
  └─ Run Evaluation Button → POST /mutation/evaluate
    ↓
Results Visualization (upon receiving response)
  ├─ MetricBox 1: "Baseline Recall: 96.36%"
  │  └─ Color: Emerald (good)
  │
  ├─ MetricBox 2: "Mutated Recall: 94.00%"
  │  └─ Color: Emerald (still good)
  │
  ├─ MetricBox 3: "Recall Degradation: -2.36%"
  │  └─ Color: Amber (warning, but expected)
  │
  ├─ MetricBox 4: "Behavioral Invariance: 0.975"
  │  └─ Color: Cyan (key metric!)
  │
  ├─ LineChart: "Before/After Detection"
  │  ├─ Green line: Baseline detection rate (96.36%)
  │  ├─ Red line: Mutated detection rate (94.00%)
  │  └─ Y-axis: Detection rate [%]
  │
  └─ FeatureShiftAnalysis: "Per-Feature Changes"
     ├─ Bar 1: bytes_in (+19KB) →
     ├─ Bar 2: bytes_out (-5KB) ←
     ├─ Bar 3: iat (+177ms) →
     └─ etc...
    ↓
User Interpretation:
  "The mutations reduced detection by 2.36%, but the
   behavioral invariance is still 0.975, which means
   the detector is resilient to these adversarial changes."
```

---

## ERROR HANDLING FLOW

```
Component Action
    ↓
Try/Catch Block
    ↓
  ├─ Success: Update store with results
  │    ├─ setLoading(false)
  │    ├─ setError(null)
  │    └─ setRobustnessResult(data)
  │
  └─ Error: Handle gracefully
       ├─ setLoading(false)
       ├─ setError(error.message)
       ├─ Toast notification (red)
       └─ Keep UI responsive
    ↓
Global Error Handler (Backend)
    ├─ Pydantic validation error → 422
    ├─ Not found error → 404
    ├─ Internal error → 500
    └─ Return JSON error response
    ↓
Frontend displays error message to user
```

---

## DEPLOYMENT ARCHITECTURE

```
Development:
  Frontend: localhost:3000
  Backend: localhost:8000
  
Production (Future):
  Frontend: Docker image → Container registry
  Backend: Docker image → Container registry
  Storage: Kubernetes persistent volumes
  Database: PostgreSQL/Cloud SQL
  CDN: Static assets
  Load Balancer: Distribute traffic
  Monitoring: Prometheus + Grafana
```

---

## TIMELINE & MILESTONES

```
Phase 1 - Backend APIs (✅ COMPLETE)
  Week 1: Services + Schemas + Endpoints
  Result: 15 REST endpoints, 4 services, full API docs

Phase 2 - Dashboard UI (✅ COMPLETE)
  Week 2: Components + Pages + Navigation
  Result: 5 interactive pages, Zustand state, Recharts

Phase 3 - Mutation Lab (✅ COMPLETE)
  Week 2-3: Interactive visualization + workflows
  Result: Centerpiece demonstration of robustness

Phase 4 - Polish (⏳ PENDING)
  Week 4: Animations, dark mode, report export
  Result: Production-ready UI/UX

Production Deployment (🎯 FUTURE)
  Shadow deployment, monitoring, hardening
```

---

**CyberShield v2 - Complete System Architecture Documented** ✅
