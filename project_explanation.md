# CyberShield — Complete Project Explanation (Research Paper Reference)

## 1. Abstract & Objective

CyberShield is a **behavioral adversarial robustness research platform for Command-and-Control (C2) network traffic detection**. It answers two questions:

1. **Detection**: Can a session-centric Transformer classifier reliably detect C2 traffic across multiple malware families and network environments?
2. **Robustness**: How resilient is the detector when an RL-trained adversary (Red-Agent) applies realistic behavioral mutations to evade detection?

The platform integrates a deep-learning detector, a PPO-based adversarial mutation agent, domain-constrained validity checking, a realism discriminator, and a full-stack web application for interactive analysis.

---

## 2. System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     FRONTEND (Next.js 14)                     │
│  Dashboard │ Session Analysis │ Mutation Lab │ Robustness     │
│  Upload    │ Zustand Store    │ Axios Client │ Recharts Plots │
└──────────────────────────┬────────────────────────────────────┘
                           │ REST API
┌──────────────────────────▼────────────────────────────────────┐
│                    BACKEND (FastAPI / Python)                  │
│  /detection  │  /mutation  │  /robustness  │  /artifacts      │
│  DetectionSvc│ MutationSvc │ RobustnessSvc │ ArtifactSvc      │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────┐
│                       ML ENGINE (PyTorch)                      │
│                                                                │
│  ┌─────────────────┐   ┌──────────────────────────────────┐   │
│  │   C2 DETECTOR    │   │         RED-AGENT (PPO)           │   │
│  │ C2Transformer    │   │ MutationPolicyNetwork             │   │
│  │ DomainAdaptive   │   │ Mutation Engines (4 types)        │   │
│  │ Baseline RF      │   │ BehavioralValidityConstraints     │   │
│  │ FocalLoss        │   │ RealismDiscriminator              │   │
│  │                  │   │ MultiObjectiveRewardEngine        │   │
│  │                  │   │ PPOTrainer (GAE + clipped obj.)   │   │
│  └─────────────────┘   └──────────────────────────────────┘   │
│                                                                │
│  DATA PIPELINE: CTU-13 / MCFP / UWF → SessionBuilder → NPZ    │
└────────────────────────────────────────────────────────────────┘
```

### 2.1 Frontend
- **Framework**: Next.js 14 + React 18 + TypeScript
- **State**: Zustand global store
- **Visualization**: Recharts
- **Pages**: Dashboard, Session Analysis, Upload, Mutation Lab, Robustness Analytics

### 2.2 Backend
- **Framework**: FastAPI (Python)
- **API Groups**: Detection, Mutation, Robustness, Artifacts, Predict
- **Entry**: `backend/main.py` → mounts routers at `/api/v1/*`

### 2.3 ML Engine
Located in `src/`, organized into: `models/`, `red_agent/`, `training/`, `features/`, `data_loader/`, `evaluation/`, `losses/`, `inference/`.

---

## 3. Datasets

### 3.1 CTU-13 (Primary Training Source)
- **Origin**: Czech Technical University, 13 botnet capture scenarios
- **Families**: Neris (Scenario 1), Kraken (Scenario 2), Conficker (Scenario 9), and others
- **Scale**: ~1.35M sessions from Scenario 1 alone (`ctu13_c2_sessions.npz` = 1.34 GB)
- **Format**: `.binetflow` CSV → processed by `ctu13_processor.py`
- **Class ratio**: Extreme imbalance (~0.6% C2)

### 3.2 Stratosphere MCFP (Zero-Shot Evaluation)
- **Origin**: Stratosphere IPS Project, Masaryk University
- **Families**: Emotet, Trickbot, Qakbot (real-world C2 captures)
- **Purpose**: Out-of-distribution evaluation; never used in training
- **Processor**: `stratosphere_mcfp_processor.py`

### 3.3 UWF-ZeekData24 (Domain Adaptation)
- **Origin**: University of West Florida
- **Content**: Labeled Zeek logs mapped to MITRE ATT&CK TA0011 (C2) plus benign
- **Families**: Sliver, Havoc, Cobalt Strike, Metasploit, and others
- **Purpose**: Cross-domain generalization testing
- **Processor**: `uwf_processor.py`

### 3.4 Canonical Data Schema
All datasets are converted to a unified NPZ format:
- `X`: shape `(N, 20, 12)` — N sessions, 20 flows per session, 12 features per flow
- `y`: shape `(N,)` — binary labels (0=benign, 1=C2)
- `masks`: shape `(N, 20)` — boolean; True = real flow, False = padding
- `feature_names`: the 12-feature vocabulary

### 3.5 Feature Vector (12 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `orig_bytes` | Bytes sent by originator |
| 1 | `resp_bytes` | Bytes sent by responder |
| 2 | `orig_pkts` | Packets sent by originator |
| 3 | `resp_pkts` | Packets sent by responder |
| 4 | `bytes_per_pkt` | Average bytes per packet |
| 5 | `packet_ratio` | orig_pkts / resp_pkts |
| 6 | `byte_ratio` | orig_bytes / resp_bytes |
| 7 | `is_outbound` | 1 if src is private, dst is public |
| 8 | `duration` | Connection duration (seconds) |
| 9 | `src_port` | Source port |
| 10 | `dst_port` | Destination port |
| 11 | `iat` | Inter-arrival time between flows |

**Derivative features** (optional): `iat_delta` and `byte_delta` are computed on-the-fly as sequential differences across flows within a session.

### 3.6 Session Construction Pipeline
```
Raw logs (CSV/Zeek) → Column standardization → Feature engineering
→ Chronological sort → IAT computation → Group by (src_ip, dst_ip, proto)
→ Segment by inactivity timeout (300s) → Truncate/pad to 20 flows
→ Build mask → Save as NPZ
```

---

## 4. Detector Architecture & Working

### 4.1 Model: C2Transformer

**Architecture class**: `src/models/transformer.py` → `C2Transformer`

```
Input: (B, 20, 12)      — batch of session tensors
       (B, 20) bool      — padding mask

Step 1: [Optional] Append derivative features → (B, 20, 14)
Step 2: Linear projection → (B, 20, d_model=64)
Step 3: LayerNorm + Dropout
Step 4: Prepend learnable [CLS] token → (B, 21, 64)
Step 5: Add learned positional encoding → (B, 21, 64)
Step 6: Transformer Encoder (3 layers, 4 heads, FFN=128, GELU) → (B, 21, 64)
Step 7: Final LayerNorm
Step 8: Extract [CLS] embedding → (B, 64)
Step 9: MLP head: Linear(64→32) → GELU → Dropout → Linear(32→1) → logit
Output: (B,) scalar logits
```

**Key design choices**:
- **CLS token aggregation** (BERT-style) rather than mean/max pooling — allows the model to learn a task-specific summary of all flow interactions
- **Padding mask** passed to `src_key_padding_mask` — ensures padded flows contribute zero attention
- **Xavier uniform initialization** for weight matrices; normal initialization (std=0.02) for positional encodings and CLS token
- **GELU activation** throughout (smoother gradient landscape than ReLU)

### 4.2 Domain-Adaptive Variant

**Architecture class**: `src/models/domain_adaptive_transformer.py` → `DomainAdaptiveC2Transformer`

Extends the base architecture with:
- **Shared backbone** (`TransformerBackbone`): identical to C2Transformer up to the CLS embedding
- **Per-domain classification heads** (`DomainHead`): each domain gets its own MLP + learnable calibration (scale, bias)
- **Domain routing**: a `domain_ids` tensor routes each sample through the appropriate head
- Designed for multi-task / continual learning across CTU-13 ("mixed") and UWF domains simultaneously

### 4.3 Baseline: Random Forest

**Implementation**: `src/models/baseline_rf.py`

- Flattens each session via mean/std/min/max pooling → 48-dimensional vector
- 200-tree RandomForest with balanced class weights
- Undersample benign to 10:1 ratio
- Threshold tuned on validation PR curve (not test set)
- Serves as the **official baseline** that the Transformer must beat

### 4.4 Loss Function: Focal Loss

**Implementation**: `src/losses/focal_loss.py`

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- Default: α=0.5, γ=1.0
- Addresses extreme class imbalance by down-weighting well-classified (easy) examples
- Computed on raw logits via numerically stable `binary_cross_entropy_with_logits`

### 4.5 Training Pipeline

**Script**: `src/training/train_transformer.py`

1. Load NPZ dataset; detect schema version (base 12-feature or extended 45-feature)
2. Apply feature transforms (log-scaling, ablation)
3. Fit per-feature normalizer on training split only
4. Create train/val/test DataLoaders (stratified split or separate NPZ files)
5. Optional: balanced sampling via `WeightedRandomSampler`
6. Optional: continual learning with replay buffer + teacher distillation
7. Train with AdamW (lr=1e-4, weight_decay=1e-3), gradient clipping (max_norm=1.0)
8. Validation: find threshold maximizing recall under FPR budget (≤0.005)
9. Early stopping on validation recall-under-budget (patience=8-10 epochs)
10. Blind test evaluation using frozen validation threshold
11. Save checkpoint with model weights, threshold, normalizer, and all hyperparameters

**Operating point policy**: The system does NOT simply maximize F1. It maximizes recall subject to FPR ≤ 0.5% — this is critical for operational deployment where false positive floods are unacceptable.

---

## 5. Red-Agent (RL Adversarial System) Architecture & Working

### 5.1 Objective
The Red-Agent learns to generate **realistic behavioral mutations** of C2 sessions that evade the detector while preserving operational plausibility.

### 5.2 Policy Network: MutationPolicyNetwork

**Implementation**: `src/red_agent/policy/__init__.py`

```
Input: session tensor (B, 20, 12) — flattened to (B, 240)

Encoder:  Linear(240 → 256) → ReLU → Linear(256 → 256) → ReLU → Dropout(0.2)

Mutation Head:  Linear(256 → 128) → ReLU → Linear(128 → 5)    → logits (5 mutation types)
Severity Head:  Linear(256 → 128) → ReLU → Linear(128 → 1) → Sigmoid   → mean ∈ [0,1]
                + learnable severity_std parameter (init=0.2)
Value Head:     Linear(256 → 128) → ReLU → Linear(128 → 1)    → V(s) estimate
```

**Action space** (hybrid discrete + continuous):
- **Discrete**: mutation type ∈ {timing, flow, TLS, composite, none} (Categorical distribution)
- **Continuous**: severity ∈ [0, 1] (Normal distribution with learned mean and std)

### 5.3 Mutation Engines (4 types)

Located in `src/red_agent/mutations/`:

| Engine | What it mutates | Examples |
|--------|----------------|----------|
| `TimingMutationEngine` | Temporal features | Jitter IAT, scale duration, add sleep intervals |
| `FlowMutationEngine` | Volume/count features | Scale bytes, adjust packet counts, modify ratios |
| `TLSMutationEngine` | Protocol metadata | Alter byte patterns, modify port distributions |
| `CompositeMutationEngine` | Chains multiple | Apply timing → flow → TLS in sequence |

### 5.4 Behavioral Validity Constraints

**Implementation**: `src/red_agent/validation/behavioral_validity_constraints.py`

Hard constraints that **cannot be violated** — enforcing physical and protocol realism:

- **Timing**: duration ∈ [0.1s, 30 days], IAT ∈ [1ms, 24h], jitter ≤ 50% of base interval
- **Flow**: packets ∈ [2, 1M], bytes ∈ [50, 10GB], byte/packet ratios ∈ [0.01, 100]
- **Protocol**: ports ∈ [1, 65535], TLS records ∈ [29, 16385] bytes
- **Statistical**: feature std ≥ 0.001 (prevents collapse to constants), variance ratio ≥ 10% of original

Constraint violations are counted and applied as **reward penalties**.

### 5.5 Realism Discriminator

**Implementation**: `src/red_agent/validation/realism_discriminator.py`

A **heuristic-based discriminator** (not learned) that scores mutations on 5 axes:

| Component (weight) | Method |
|---|---|
| Statistical plausibility (0.25) | Z-score vs. C2 training distribution |
| Distribution shift (0.25) | Kolmogorov-Smirnov test vs. original session |
| Behavioral coherence (0.20) | bytes_per_pkt consistency, is_outbound validity, non-negative packets |
| Correlation preservation (0.15) | Frobenius norm of correlation matrix difference |
| Protocol realism (0.15) | Packets-per-second range, port validity |

Output: scalar ∈ [0, 1]. Threshold for "realistic" = 0.6.

### 5.6 Multi-Objective Reward Engine

**Implementation**: `src/red_agent/training/reward_engine.py`

$$R = w_e \cdot R_{\text{evasion}} + w_r \cdot R_{\text{realism}} + w_f \cdot R_{\text{functionality}} + w_s \cdot R_{\text{stability}} + w_d \cdot R_{\text{diversity}} + P_{\text{constraint}}$$

| Component | Weight | Computation |
|-----------|--------|-------------|
| Evasion | 0.40 | Confidence drop + bonus for label flip (crossing 0.5 threshold) |
| Realism | 0.30 | Tiered: +bonus if score ≥ 0.7; penalty if < 0.4; harsh penalty if very low |
| Functionality | 0.15 | Preserves C2 operational properties |
| Stability | 0.10 | Prevents adversarial artifacts |
| Diversity | 0.05 | Exploration bonus to avoid repetitive exploits |
| Constraint penalty | unweighted | −0.5 per violation (capped at −2.0) |

### 5.7 PPO Trainer

**Implementation**: `src/red_agent/training/ppo_trainer.py`

Standard Proximal Policy Optimization with:
- **GAE** (Generalized Advantage Estimation): γ=0.99, λ=0.95
- **Clipped surrogate objective**: ε=0.2
- **Value function loss**: 0.5 × MSE(V(s), returns)
- **Entropy bonus**: coefficient=0.01
- **Gradient clipping**: max_norm=0.5
- **Optimizer**: Adam, lr=3e-4
- Mini-batch updates over 3 epochs per rollout

### 5.8 Orchestrator: End-to-End Pipeline

**Implementation**: `src/red_agent/orchestrator.py` → `RedAgentOrchestrator`

```
For each session:
  1. Select mutation engine based on policy action
  2. Apply mutation with chosen severity
  3. Run behavioral validity constraints → list violations
  4. Score realism via discriminator → scalar + component breakdown
  5. Run frozen detector on ORIGINAL session → confidence_original
  6. Run frozen detector on MUTATED session → confidence_mutated
  7. Compute reward via MultiObjectiveRewardEngine
  8. Return EvaluationResult(mutation_result, confidences, reward, components)
```

Batch evaluation supports mixed mutation types and random severities.

---

## 6. Data Flow (End-to-End)

### 6.1 Training Flow
```
Raw logs → Processor (CTU-13/MCFP/UWF) → SessionBuilder → NPZ
→ train_transformer.py: Load NPZ → Transform → Normalize → DataLoader
→ Training loop: FocalLoss + AdamW + threshold tuning under FPR budget
→ Checkpoint: model_state_dict + threshold + normalizer + config
```

### 6.2 Red-Agent Training Flow
```
Frozen detector checkpoint + CTU-13 NPZ
→ Sample session batch → Policy samples (mutation_type, severity)
→ Mutation engine applies → Validity check → Realism scoring
→ Detector inference (original + mutated) → Reward computation
→ PPO update (clipped surrogate + value loss + entropy)
→ Save policy checkpoint + JSONL logs + plots
```

### 6.3 Evaluation / Analysis Flow
```
Upload NPZ via frontend → Backend evaluates with detector → datasetEvaluationResult
→ Frontend stores in Zustand → Dashboard displays metrics
→ Mutation Lab: configure mutations → send to /robustness/red-agent/evaluate
→ Backend: orchestrator runs mutation → validity → realism → detection → reward
→ Return: baseline recall, mutated recall, evasion rate, realism score
→ Robustness Analytics: aggregate per-mutation rankings + comparison plots
```

---

## 7. Experimental Results

### 7.1 Random Forest Baseline (CTU-13)
- **Holdout F1 (tuned threshold)**: Established as official baseline
- Uses mean/std/min/max pooling → 48-feature flat vector
- 200 trees, balanced weights, 10:1 undersampling
- Top features: byte-related statistics dominate importance

### 7.2 Transformer Detector — CTU-13 Holdout
From `experiments/score_ctu_base.json`:
- **Threshold**: 0.361
- **Recall @ threshold**: 66.9%
- **FPR @ threshold**: 0.0014 (0.14%) — **well within 0.5% budget**
- **Positive mean score**: 0.647; Negative mean score: 0.038
- **Positive P50**: 0.885; Positive P95: 0.918
- Strong separation between C2 and benign score distributions

### 7.3 Zero-Shot Cross-Domain (MCFP)
From `test_results_multi_dataset.json` at threshold=0.799:

| Dataset | Recall | FPR | F1 | AUC |
|---------|--------|-----|-----|-----|
| MCFP Multi | **97.5%** | 0.90% | 0.987 | 0.995 |
| MCFP Multi + Benign | **97.5%** | 0.61% | 0.986 | 0.997 |
| Stratosphere MCFP | **97.5%** | 0.90% | 0.987 | 0.995 |

**Key finding**: The detector achieves near-perfect zero-shot generalization on MCFP C2 families (Emotet, Trickbot, Qakbot) — families never seen during training.

### 7.4 Cross-Family Generalization (CTU-13 Scenarios)
At same threshold=0.799:

| Scenario | Family | Recall | FPR | AUC |
|----------|--------|--------|-----|-----|
| Scenario 1 | Neris | 0.56% | 13.2% | 0.464 |
| Scenario 2 | Kraken | 0.21% | 10.2% | 0.493 |
| Scenario 9 | Conficker | 0.0% | 7.7% | 0.707 |

**Key finding**: The high threshold (0.799) tuned for one operating point produces very low recall on the same-dataset cross-scenario test. This reveals that **threshold calibration is domain-sensitive** — the detector's AUC for Conficker (0.707) shows the model does extract signal, but the operating point doesn't transfer. This motivated the domain-adaptive architecture.

### 7.5 Domain-Adaptive Transformer Results
The `DomainAdaptiveC2Transformer` with per-domain heads and calibration layers was trained on combined CTU-13 ("mixed") and UWF domains:
- Maintains performance on the mixed domain while adapting to UWF
- Per-domain thresholds prevent the cross-domain threshold transfer problem
- Weighted balanced sampling across domain × class combinations

### 7.6 Red-Agent Smoke Training Observations
From Phase 1 training runs:
- Different batch sizes and checkpoints produce different realism/evasion trade-offs
- Smaller batches → noisier gradients → different detector sensitivity profiles
- The realism discriminator and constraint penalties are **essential** — without them, the policy reward-hacks by producing unrealistic high-evasion mutations
- Composite mutations achieve higher evasion but more frequent constraint violations
- Timing mutations are most reliably realistic

---

## 8. Key Design Decisions & Rationale

### 8.1 Session-Centric vs. Host-Centric
The project initially used a session-centric representation (20 flows × 12 features). A confounded experiment suggested this was insufficient, but controlled analysis (documented in `README_RESEARCH_DIRECTION.md`) revealed the failure was due to **sample starvation and family overfitting**, not architectural limitations. The multi-family generalization experiment isolates training data diversity as the single variable.

### 8.2 FPR Budget Constraint
The system enforces **FPR ≤ 0.005** (0.5%) during threshold tuning. At 1M benign sessions/day, this caps false alerts at 5,000/day. This is a strict operational requirement, not just an academic metric.

### 8.3 Heuristic vs. Learned Discriminator
The realism discriminator uses statistical heuristics (z-scores, KS tests, correlation preservation) rather than a learned discriminator (GAN-style). This is deliberate: a learned discriminator could co-evolve with the policy and lose its grounding in domain physics.

### 8.4 Focal Loss for Extreme Imbalance
Standard BCE fails at 0.6% positive rate. Focal loss with γ=1.0 provides moderate focusing — enough to address imbalance without over-weighting noisy hard examples.

---

## 9. Project Structure

```
CyberShield/
├── backend/                    # FastAPI backend
│   ├── app/api/               # API routers (detection, mutation, robustness, artifacts)
│   ├── app/schemas/           # Pydantic request/response schemas
│   ├── app/services/          # Business logic services
│   └── main.py                # Application entry point
├── frontend/                   # Next.js 14 frontend
│   ├── app/                   # Pages and routing
│   ├── components/            # React components (Mutation Lab, etc.)
│   └── lib/                   # API client, store, utilities
├── src/                        # ML engine
│   ├── models/                # C2Transformer, DomainAdaptiveC2Transformer, baseline_rf
│   ├── red_agent/             # Adversarial mutation system
│   │   ├── mutations/         # TimingMutationEngine, FlowMutationEngine, TLS, Composite
│   │   ├── validation/        # BehavioralValidityConstraints, RealismDiscriminator
│   │   ├── training/          # MultiObjectiveRewardEngine, PPOTrainer
│   │   ├── policy/            # MutationPolicyNetwork
│   │   └── orchestrator.py    # Red-Agent coordination
│   ├── training/              # train_transformer.py, train_domain_adaptive_transformer.py
│   ├── features/              # Feature config, session builder, dataset builder
│   ├── data_loader/           # CTU-13, MCFP, UWF processors, NPZ utilities
│   ├── evaluation/            # Operating point, holdout evaluation, zero-shot eval
│   ├── losses/                # Focal loss
│   └── inference/             # Production inference pipeline
├── data/
│   ├── raw/                   # Original datasets (CTU-13, MCFP, UWF)
│   └── processed/             # NPZ files (canonical format)
├── experiments/                # Experiment outputs (37+ experiment directories)
└── docs/                       # Status notes, plans, reports
```

---

## 10. Current Limitations

1. **Threshold transfer**: A single threshold does not generalize across domains/families — the domain-adaptive architecture addresses this but adds complexity.
2. **Realism discriminator**: Heuristic-based; may not capture all aspects of behavioral realism for novel C2 families.
3. **Red-Agent policy**: Currently an MLP encoder — a Transformer-conditioned policy might capture flow-level patterns better.
4. **Feature set**: The 12-feature schema captures volume/timing/direction but lacks TLS handshake details, DNS behavior, and temporal persistence patterns. An extended 45-feature schema exists but is experimental.
5. **Mutation Lab fallback**: When backend raw tensors are unavailable, the frontend reconstructs from session summaries (approximation).
6. **Class imbalance**: 0.6% C2 ratio requires careful sampling, loss design, and threshold tuning.

---

## 11. Future Work (Prioritized)

### 11.1 Immediate
- **Sliver C2 dataset**: Collect/process Sliver C2 traffic in canonical `(B, 20, 12)` format; evaluate whether the current detector generalizes to modern C2 frameworks.
- **Complete multi-family experiment**: Train on Neris + Kraken, evaluate zero-shot on Conficker with ports ablated, to isolate whether data diversity solves generalization.

### 11.2 Short-Term
- **Full Red-Agent Phase 2**: Train the policy end-to-end with the PPO trainer on large batches; evaluate evasion rates and realism across mutation types.
- **Learned realism discriminator**: Train a discriminator on real vs. mutated sessions to complement the heuristic scorer.
- **Adversarial retraining**: Use Red-Agent-generated mutations to augment training data and harden the detector.

### 11.3 Long-Term
- **Host-centric modeling**: If session-level generalization proves insufficient, extend to host-level behavioral timelines with temporal persistence features.
- **Graph-based detection**: Model host communication graphs for structural C2 pattern detection.
- **Continual learning**: Expand the domain-adaptive architecture to incorporate new C2 families without catastrophic forgetting.
- **Deployment pipeline**: CI/CD with automated smoke tests, model versioning, and threshold re-calibration.

---

## 12. How to Interpret Results

| Metric Combination | Interpretation |
|---|---|
| High baseline recall + low mutated recall | Detector is vulnerable to that mutation type |
| High evasion + high realism | **Actionable robustness concern** — real operational risk |
| High evasion + low realism | Stress test only — not an operational threat |
| High FPR shift | Mutation causes detector to flag too many benign sessions |
| High behavioral invariance | Good — important behavioral constraints survived mutation |

**Rule**: Always use metrics together. A single number (recall, evasion rate) in isolation is misleading.

---

## 13. Reproducibility Notes

- **Random seeds**: All experiments use explicit seeds (default=42) for NumPy, PyTorch, and Python random.
- **Checkpoint contract**: Every saved checkpoint contains `model_state_dict`, `optimal_threshold`, `feature_normalizer`, `feature_transform_config`, and `schema_version` — sufficient to reproduce inference exactly.
- **NPZ schema**: All datasets store `X`, `y`, `masks`, and `feature_names` arrays.
- **FPR guard band**: Validation uses an explicit `fpr_guard_band` multiplier (default 1.0) to add margin between validation and test FPR.

---

## 14. Key References (Internal)

| Document | Purpose |
|----------|---------|
| `README.md` | Full project README with architecture and usage |
| `README_RESEARCH_DIRECTION.md` | Controlled generalization experiment design |
| `PROGRESS_CHECKLIST.md` | Implementation status tracking |
| `DOCUMENTATION_INDEX.md` | Index of all documentation |
| `experiments/` | All experiment outputs, configs, and plots |

---

*Document prepared for research paper reference. Last updated: May 2026.*
