"""
CyberShield v2 Backend Architecture and API Documentation
===========================================================

OVERVIEW
--------

CyberShield v2 is a behavioral adversarial robustness intelligence platform
that operationalizes:

1. Session-centric Transformer behavioral detector
2. Deterministic adversarial mutation engine (Red-Agent v1)
3. Robustness evaluation pipeline
4. Failure analysis and analytics system

The backend exposes a comprehensive REST API that enables the frontend
to demonstrate behavioral robustness through interactive mutation testing.


SYSTEM ARCHITECTURE
-------------------

Frontend (Next.js/React)
    ↓ (REST API calls)
    │
FastAPI Backend (Python async)
    ├── /api/v1/detection      [Session inference, dataset analysis]
    ├── /api/v1/mutation       [Mutation generation, application]
    ├── /api/v1/robustness     [Metrics, failure analysis, comparison]
    └── /api/v1/artifacts      [Upload, retrieval, management]
    │
    ↓ (Service layer)
    │
CyberShield Services
    ├── DetectionService       [ML inference, model loading]
    ├── MutationService        [Red-Agent v1 framework integration]
    ├── RobustnessService      [Evaluation, analytics]
    └── ArtifactService        [File storage, metadata]
    │
    ↓ (Data layer)
    │
Artifacts Storage (Local filesystem → PostgreSQL/Parquet)
    ├── Datasets (NPZ format)
    ├── Metadata (JSONL format)
    ├── Reports (JSON format)
    └── Checkpoints (PT format)


CORE MODULES
============

1. DETECTION SERVICE (app/services/detection.py)
-------------------------------------------

Manages ML model inference for behavioral risk assessment.

Responsibilities:
- Load and initialize Transformer checkpoint
- Extract 10-feature behavioral schema from session data
- Run inference (single and batch)
- Produce risk scores [0, 1] and confidence metrics

Key Methods:
- initialize()                  - Load model and scaler
- extract_features()            - Extract 10 features from session
- infer_single()               - Single session inference
- infer_batch()                - Batch inference

Feature Schema (10 features):
  0. duration              - Session duration (seconds)
  1. bytes_in             - Inbound bytes
  2. bytes_out            - Outbound bytes
  3. packets_in           - Inbound packet count
  4. packets_out          - Outbound packet count
  5. protocol             - TCP/UDP (binary)
  6. src_port (normalized) - Source port [0, 1]
  7. dst_port (normalized) - Destination port [0, 1]
  8. timestamp            - Session start time
  9. reserved             - Reserved for future

Model Output:
- risk_score: Float [0, 1]  (0=benign, 1=C2)
- is_suspicious: Boolean     (risk_score > 0.5)
- confidence: Float [0, 1]   (model certainty)


2. MUTATION SERVICE (app/services/mutation.py)
----------------------------------------------

Integrates Red-Agent v1 adversarial mutation framework.

Responsibilities:
- Access mutation registry (11 registered operators)
- Apply deterministic mutations to session data
- Generate mutated datasets with metadata tracking
- Support per-mutation severity tuning

Available Mutations:
  Timing Mutations:
    - timing_jitter           - Add random delays
    - burst_callback          - Cluster packets into bursts
    - delayed_reconnect       - Extended reconnection delays

  Persistence Mutations:
    - low_frequency_callback  - Reduce communication frequency
    - intermittent_communication - Intermittent communication patterns
    - long_sleep             - Extended sleep periods

  TLS/Encryption Mutations:
    - tls_padding            - Add TLS padding
    - session_reuse_shape    - Vary TLS session shapes
    - handshake_variation    - Modify handshake patterns

  Flow Mutations:
    - packet_count_variation - Vary packet counts
    - byte_distribution_shift - Shift byte distributions

Key Methods:
- initialize()               - Load mutation registry
- list_mutations()          - List available operators
- apply_mutation()          - Apply single mutation
- batch_mutate()            - Mutate entire dataset

Determinism:
- All mutations seeded (seed parameter)
- Same config + seed = identical results
- Reproducible adversarial evaluation


3. ROBUSTNESS SERVICE (app/services/robustness.py)
---------------------------------------------------

Analyzes adversarial robustness and identifies behavioral failures.

Responsibilities:
- Compare baseline vs mutated detection performance
- Compute robustness metrics (recall, invariance, FPR shift)
- Identify fragile mutations and failure patterns
- Analyze feature shifts under mutation

Key Metrics:
- baseline_recall           - Detection recall on original data
- mutated_recall           - Detection recall on mutations
- recall_degradation       - Difference (negative = worse robustness)
- behavioral_invariance    - mutated_recall / baseline_recall
- fpr_shift                - False positive rate change
- fragile_count            - Samples detected in baseline but not mutated

Key Methods:
- evaluate_robustness()     - Full robustness evaluation
- analyze_failures()        - Identify failure patterns
- _analyze_feature_shift()  - Per-feature analysis

Output:
- Comprehensive robustness report with:
  * Per-mutation impact ranking
  * Feature shift analysis
  * Failure distributions
  * Behavioral invariance scores


4. ARTIFACT SERVICE (app/services/artifact.py)
------------------------------------------------

Manages persistent storage of datasets, results, and reports.

Responsibilities:
- Upload and store file artifacts
- Track metadata (creation time, type, related items)
- Retrieve artifacts by ID
- List and filter artifacts
- Save structured reports

Supported Formats:
- NPZ (numpy compressed)  - Datasets
- JSONL                   - Metadata, line-delimited
- JSON                    - Reports, results
- CSV                     - Results, analytics

Storage:
- Initial: Local filesystem (backend/artifacts/)
- Later: PostgreSQL, Parquet, Redis


API ENDPOINTS (OpenAPI/Swagger)
================================

BASE URL: http://localhost:8000/api/v1


DETECTION API (/detection)
---------------------------

POST /analysis/session
  Request:  SessionData (single session with 10 features)
  Response: DetectionResult (risk_score, confidence, explanation)
  Purpose:  Analyze single session
  
POST /analysis/batch
  Request:  SessionBatch (list of sessions)
  Response: BatchDetectionResult (per-session + summary)
  Purpose:  Batch inference with summary stats
  
POST /analysis/dataset
  Request:  DatasetEvaluationRequest (NPZ file ID, limits)
  Response: DatasetEvaluationResult (aggregate metrics)
  Purpose:  Full dataset evaluation

Example:
  curl -X POST http://localhost:8000/api/v1/detection/session \
    -H "Content-Type: application/json" \
    -d '{
      "session_id": "session_123",
      "duration": 45.2,
      "bytes_in": 50000,
      "bytes_out": 100000,
      "packets_in": 150,
      "packets_out": 200,
      "protocol": "TCP",
      "src_port": 12345,
      "dst_port": 443,
      "timestamp": 1620000000.0
    }'


MUTATION API (/mutation)
------------------------

GET /available
  Response: AvailableMutationsResponse
  Purpose:  List all available mutation operators
  
POST /apply
  Request:  MutationRequest
    - source_npz_file_id
    - mutations: List[MutationOperator]
    - base_seed, sample_limit
  Response: MutationResult
    - output_npz_file_id (mutated dataset)
    - metadata_file_id (JSONL with mutation details)
  Purpose:  Apply mutations to dataset
  
POST /evaluate
  Request:  MutationEvaluationRequest
    - baseline_npz_file_id
    - mutated_npz_file_id
  Response: MutationEvaluationResult
    - baseline_recall, mutated_recall
    - recall_degradation, fpr_shift
    - per_mutation_metrics, fragile_mutations
  Purpose:  Evaluate robustness against mutations

Example:
  curl -X POST http://localhost:8000/api/v1/mutation/apply \
    -H "Content-Type: application/json" \
    -d '{
      "source_npz_file_id": "file_abc123",
      "mutations": [
        {"mutation_type": "timing_jitter", "severity": 0.3},
        {"mutation_type": "tls_padding", "severity": 0.5}
      ],
      "base_seed": 42,
      "batch_size": 256
    }'


ROBUSTNESS API (/robustness)
-----------------------------

GET /report/{evaluation_id}
  Response: RobustnessReport
  Purpose:  Retrieve comprehensive robustness report
  
GET /metrics/{evaluation_id}
  Response: RobustnessMetrics
  Purpose:  Quick summary metrics
  
POST /failures
  Request:  Failure analysis request
  Response: FailureAnalysisResult
    - fragile_mutations
    - feature_shift_analysis
    - failure_patterns
  Purpose:  Analyze behavioral failures
  
POST /compare
  Request:  RobustnessComparisonRequest
  Response: Comparative analysis across mutation groups
  Purpose:  Compare robustness across mutation sets


ARTIFACTS API (/artifacts)
---------------------------

POST /upload
  Request:  File upload + metadata
  Response: FileUploadResponse (file_id, metadata)
  Purpose:  Upload dataset or artifact
  
GET /file/{file_id}
  Response: Binary file stream
  Purpose:  Download artifact
  
GET /metadata/{file_id}
  Response: ArtifactMetadata
  Purpose:  Get file metadata
  
GET /list
  Query:    file_type (npz, json, jsonl), limit, offset
  Response: ArtifactListResponse (paginated list)
  Purpose:  List artifacts with filtering
  
DELETE /file/{file_id}
  Response: Deletion confirmation
  Purpose:  Delete artifact
  
POST /report
  Request:  Report upload + metadata
  Response: ReportMetadata
  Purpose:  Save structured report


REQUEST/RESPONSE SCHEMAS
========================

See app/schemas/ directory for full Pydantic models:

detection.py
  - SessionData
  - SessionBatch
  - DetectionResult
  - BatchDetectionResult
  - DatasetEvaluationRequest/Result

mutation.py
  - MutationOperator
  - MutationRequest
  - MutationResult
  - MutationEvaluationRequest/Result

robustness.py
  - RobustnessReport
  - RobustnessMetrics
  - FailureAnalysisResult

artifact.py
  - FileUploadResponse
  - ArtifactMetadata
  - ReportMetadata


EXECUTION FLOW (Mutation Lab Example)
======================================

1. User uploads baseline dataset (NPZ)
   POST /api/v1/artifacts/upload
   → Returns file_id_baseline

2. User configures mutations (UI)
   - Select mutations: timing_jitter (0.3), tls_padding (0.5)
   - Base seed: 42

3. Backend applies mutations
   POST /api/v1/mutation/apply
   Request: {
     "source_npz_file_id": "file_id_baseline",
     "mutations": [
       {"mutation_type": "timing_jitter", "severity": 0.3},
       {"mutation_type": "tls_padding", "severity": 0.5}
     ]
   }
   → Returns file_id_mutated, metadata_id

4. Backend evaluates robustness
   POST /api/v1/mutation/evaluate
   Request: {
     "baseline_npz_file_id": "file_id_baseline",
     "mutated_npz_file_id": "file_id_mutated"
   }
   → Returns robustness metrics:
     * baseline_recall: 0.9636
     * mutated_recall: 0.9400
     * recall_degradation: 2.36%
     * behavioral_invariance: 0.975

5. Frontend visualizes results
   - Before/after risk score distribution
   - Feature shift heatmap
   - Fragile mutation ranking
   - Robustness invariance score

6. User inspects failures
   POST /api/v1/robustness/failures
   → Get failure analysis, fragile samples, feature shifts


INTEGRATION WITH RED-AGENT V1
==============================

The backend seamlessly integrates CyberShield's Red-Agent v1 framework:

1. Mutation Registry
   - All 11 mutations pre-registered
   - Accessible via MutationService.list_mutations()
   
2. Deterministic Execution
   - All mutations seeded for reproducibility
   - Same config + seed = identical results
   
3. Metadata Tracking
   - Each mutated sample tracked with:
     * mutation_type
     * severity
     * parameters
     * applied_at_step
     * original_label
     * outcome
   
4. Artifact Management
   - Mutated datasets saved as NPZ
   - Metadata saved as JSONL
   - Full lineage preserved


DEVELOPMENT SETUP
=================

Backend Installation:

1. Navigate to backend directory
   cd backend

2. Install dependencies
   pip install -r requirements.txt

3. Run server
   python main.py

4. Access API documentation
   http://localhost:8000/docs (Swagger UI)
   http://localhost:8000/redoc (ReDoc)

5. Example request
   curl -X GET http://localhost:8000/health


ENVIRONMENT CONFIGURATION
==========================

Set in .env file or OS environment:

API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

CHECKPOINT_PATH=../experiments/multifamily_generalization/strict_smoke/best_transformer.pth
SCALER_PATH=../experiments/baseline/scaler.joblib
DEVICE=cpu

DATA_DIR=../data/processed
ARTIFACTS_DIR=./artifacts

MUTATION_BASE_SEED=42
MUTATION_BATCH_SIZE=256

LOG_LEVEL=INFO


PERFORMANCE & OPTIMIZATION
===========================

Batch Processing:
- Detection: ~100-1000 sessions/sec (batch_size=256, CPU)
- Mutation: ~100-500 samples/sec (depends on mutation complexity)
- Evaluation: ~50-200 samples/sec (depends on dataset size)

Future Optimizations:
- GPU acceleration (CUDA for Transformer)
- Async mutation pipeline
- Result caching
- Database indexing
- Async workers (Celery)


TESTING
=======

Unit tests:
  pytest tests/backend/ -v

Integration tests:
  pytest tests/backend/integration/ -v

Example quick test:
  curl -X GET http://localhost:8000/health
  → {"status": "healthy", "service": "CyberShield v2 Backend", "version": "2.0.0"}


NEXT PHASE: FRONTEND IMPLEMENTATION
====================================

The backend is now ready for full-stack integration.

Next steps:
1. Set up Next.js frontend
2. Create Dashboard page (Phase 2)
3. Create Session Analysis page (Phase 2)
4. Create Mutation Lab (Phase 3) - PRIMARY DIFFERENTIATOR
5. Create Robustness Analytics (Phase 3)
6. Polish and animations (Phase 4)

Frontend connects to backend via REST APIs defined above.
All behavioral intelligence visualizations consume these endpoints.
