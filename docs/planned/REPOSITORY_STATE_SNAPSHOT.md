# Repository State Snapshot (May 9, 2026)

**Purpose**: Document what exists from Phase 2 that's available for the new experiment

**Created**: May 9, 2026  
**Branch**: research/multifamily_generalization  
**Status**: ✅ Ready for Week 1 implementation

---

## Phase 2 Infrastructure (VALIDATED - Ready to Use)

### ✅ Existing Data Files

**Location**: data/processed/

| Path | Size | Description | Status |
|------|------|-------------|--------|
| v2_uwf/ | - | UWF train/val splits (22 C2 samples) | ✅ Verified |
| v2_mixed_session_only/ | - | CTU-13 mixed test set | ✅ Verified |
| ctu13_c2_sessions.npz | - | CTU-13 raw C2 sessions | ✅ Available |
| mcfp_multi_sessions.npz | - | MCFP C2 sessions | ✅ Available |
| benign_mix/ | - | Benign session data | ✅ Available |
| ctu13_multi/ | - | CTU-13 multi-family data | ✅ Available |

**Note**: UWF NOT used in new experiment (to prevent lab data confounding effect)

### ✅ Existing Core Classes

**Location**: src/

#### Data Loading (src/data_loader/)
```
├── npz_utils.py              # NPZ loading helpers ✅
├── torch_dataset.py          # C2SessionDataset class ✅
├── feature_transforms.py     # Feature extraction ⚠️ NEEDS UPDATES (port removal)
├── normalization.py          # Normalization/scaling ⚠️ NEEDS UPDATES
├── split_utils.py            # Data splitting utilities ✅
├── ctu13_processor.py        # CTU-13 parsing ✅
├── stratosphere_mcfp_processor.py  # MCFP parsing ✅
├── uwf_processor.py          # UWF parsing (not used in new exp)
└── normalized_telemetry.py   # Telemetry handling ✅
```

#### Models (src/models/)
```
├── transformer.py            # C2Transformer ✅ (UNCHANGED in new experiment)
│   - Features: seq_len=20, 12→10 features (will update)
│   - Embedding: 256D
│   - Heads: 4 attention heads
│   - Output: Binary classification (C2/Benign)
│   - Status: VALIDATED in Phase 2
│
└── baseline_rf.py            # Random forest baseline ✅ (reference only)
```

#### Training (src/training/)
```
├── train_transformer.py      # Main training loop ✅ COMPATIBLE
│   - Supports FPR-constrained threshold tuning
│   - Supports separate train/val NPZ paths
│   - Uses FocalLoss(alpha=0.5, gamma=1.0)
│   - Optimizer: AdamW(lr=1e-4, weight_decay=1e-3)
│   - Scheduler: ReduceLROnPlateau(patience=4)
│   - Status: TESTED, ready for reuse
│
└── [no other training modules]
```

#### Evaluation (src/evaluation/)
```
├── evaluate_transformer.py   # Per-fold validation ✅
├── evaluate_zero_shot.py     # Cross-dataset evaluation ✅
├── operating_point.py        # Threshold optimization ✅
└── [evaluation metrics, confusion matrix, ROC/PR curves] ✅
```

#### Features (src/features/)
```
├── feature_config.py         # Feature schema (12→10) ⚠️ NEEDS UPDATE
├── feature_config_v2.py      # Extended v2 schema ✅
├── zeek_parser.py            # Zeek log parsing ✅
├── session_builder.py        # Session construction ✅
├── host_timeline.py          # Host-level infrastructure (NOT used in new exp)
│   - Fully implemented from Phase 2
│   - Useful IF Outcome 2 (partial generalization) occurs
│   - On hold for now
└── dataset_builder*.py       # Session aggregation ✅
```

### ✅ Existing Test Suite

**Status**: 74 tests passing (baseline verified)

```
tests/
├── test_phase0.py              # 58 tests ✅ ALL PASSING
│   - Data contracts
│   - Feature extraction
│   - Session building
│   - Normalization
│   - Training operating points
│   - Evaluation metrics
│
├── test_data_contracts.py      # Data shape validation ✅
├── test_feature_transforms.py  # Feature computation ✅
├── test_evaluation_report.py   # Metrics computation ✅
├── test_normalization.py       # Z-score scaling ✅
├── test_training_operating_point.py  # Threshold tuning ✅
└── [16 other passing tests] ✅
```

### ✅ Existing Models

**Location**: experiments/

| Path | Type | Status | Use in New Exp? |
|------|------|--------|-----------------|
| transformer_uwf_only/ | C2Transformer (12 feat) | ✅ | Reference only |
| baseline/ | Random Forest baseline | ✅ | Comparison |
| transformer_focal_base/ | Variant (focal loss) | ✅ | Reference |
| transformer_deriv/ | Variants (derivative feats) | ✅ | Not used |
| transformer_ablate_ports_outbound/ | ✅ Ablation key result | ✅ | Validates port removal |

### ⚠️ Phase 2 Artifacts (For Reference)

```
docs/
├── implementation_summary_phase2.md     # Phase 2 recap
├── zero_shot_validation_report.md       # Previous results
├── phase3_*.md                          # ARCHIVED (invalid)
└── [new experiment docs]                # Fresh start
```

---

## What Needs to Be Built (Week 1-3)

### Week 1: Data Loading + Feature Updates

#### 1. CTU-13 Multi-Family Dataset Loading
```
✅ Required classes:
  - CTU13DatasetManager (load Scenarios 1, 2, 9)
  - Extract ground truth labels (C2 vs benign)
  - Compute per-family statistics
  - Create dataset manifest (JSON)
```

#### 2. Feature Schema Update (12 → 10)
```
✅ Updates:
  - Remove src_port, dst_port
  - Update feature_config.py assertions
  - Regenerate normalization on 10 features
  - Update test assertions
```

**Result**: Unified CTU-13 dataset with 300K+ C2 sessions (10 features)

---

### Week 2: Experiment Infrastructure

#### 1. FamilyAwareSplitter
```
✅ Required:
  - Prevent family leakage (train/test families must be different)
  - Generate split manifest (JSON with metadata)
  - Train: Scenarios 1, 2 (Neris + Kraken)
  - Test: Scenario 9 (Conficker zero-shot)
```

#### 2. Evaluation Pipeline
```
✅ Metrics to compute:
  - PR-AUC (primary metric for imbalanced data)
  - ROC-AUC (secondary)
  - Recall @ fixed FPR (1%, 2%, 5%)
  - Precision @ fixed Recall (70%, 80%, 90%)
  - Per-family confusion matrix
  - Error analysis (FN/FP clustering)
```

**Result**: Reproducible experiment setup with split manifest

---

### Week 3: Execution + Analysis

#### 1. Training
```
✅ Tasks:
  - Train on multi-family (Neris + Kraken) using existing train_transformer.py
  - Generate validation curves
  - Save best checkpoint
```

#### 2. Evaluation
```
✅ Tasks:
  - Evaluate on Conficker (zero-shot family)
  - Compute all metrics per family
  - Generate confusion matrices
  - Run error analysis
```

#### 3. Results Interpretation
```
✅ Apply decision logic:
  - Strong (recall ≥75%): Session-centric works
  - Partial (recall 50-75%): Host-centric justified
  - Failed (recall <50%): Major redesign needed
```

**Result**: Clear outcome + recommendation for next phase

---

## Existing Infrastructure to Reuse

### ✅ Can Use As-Is
- `src/models/transformer.py` (no changes)
- `src/training/train_transformer.py` (reuse training loop)
- `src/evaluation/evaluate_transformer.py` (reuse metrics)
- Test suite framework (add new tests)
- NPZ data format (reuse)
- Data splitting infrastructure (reuse)

### ⚠️ Needs Updates
- `src/features/feature_config.py` (remove ports)
- `src/data_loader/feature_transforms.py` (adapt for 10 features)
- `src/data_loader/normalization.py` (regenerate baselines)
- Test fixtures (update feature dim from 12 to 10)

### ❌ Won't Use
- UWF training data (confounds family effect)
- Host-timeline code (not needed for session-centric exp)
- Phase 3 planning docs (archived)

---

## Development Workflow

### Branch Strategy
```
main (stable, Phase 2 validated)
  └── research/multifamily_generalization (CURRENT)
      ├── Week 1 work (data loading + features)
      ├── Week 2 work (splits + evaluation)
      └── Week 3 work (training + analysis)
```

### Commit Strategy
```
Frequent commits:
  - "feat: update feature schema to 10 features"
  - "feat: add CTU-13 dataset loader"
  - "feat: implement FamilyAwareSplitter"
  - "feat: add evaluation metrics pipeline"
  - "experiment: train multi-family model"
  - "analysis: interpret results"
```

### Testing Strategy
```
After each change:
  - Run pytest tests/ -q
  - Verify 74 tests still pass
  - Add new tests for new code
  - Check no shape mismatches
```

---

## Key Files to Track

### Configuration Files
```
✅ src/features/feature_config.py
   Current: FEATURE_DIM = 12
   Target: FEATURE_DIM = 10
   Status: Ready to update Week 1, Day 4
```

### Data Manifests
```
To create:
  - experiments/multifamily_generalization/config.json
  - experiments/multifamily_generalization/split_manifest.json
  - experiments/multifamily_generalization/data_statistics.json
```

### Model Checkpoints
```
To generate:
  - experiments/multifamily_generalization/best_transformer.pth
  - experiments/multifamily_generalization/training_curves.json
  - experiments/multifamily_generalization/results.json
```

---

## Success Metrics

### Week 1 Complete When:
- [ ] CTU-13 Scenarios 1, 2, 9 loaded (300K+ sessions)
- [ ] Feature schema updated (10 features)
- [ ] All tests passing with new dimension
- [ ] Dataset statistics computed

### Week 2 Complete When:
- [ ] FamilyAwareSplitter working (no leakage verified)
- [ ] Evaluation pipeline complete (all metrics implemented)
- [ ] Split manifest created (JSON reproducibility)
- [ ] Ready to train

### Week 3 Complete When:
- [ ] Training completes successfully
- [ ] Evaluation generates all metrics
- [ ] Results interpreted using decision logic
- [ ] Next phase clearly planned

---

## Risk Factors

### ⚠️ Potential Issues

| Risk | Mitigation |
|------|-----------|
| Feature extraction changes shape | Run test suite after each change |
| Normalization parameters wrong | Compare old vs new stats |
| Family leakage in splits | Verify train/test families are disjoint |
| Training fails on new data | Monitor loss curves in real-time |
| Results ambiguous | Error analysis should identify patterns |

### ✅ Backup Plans

- **If data loading fails**: Verify CTU-13 format, use existing ctu13_c2_sessions.npz
- **If feature update breaks tests**: Revert to old schema, debug incrementally
- **If training won't converge**: Check learning rate, data distribution
- **If results unclear**: Extend Week 3 by one week for deeper analysis

---

## Timeline Summary

```
May 9 (Today):     ✅ Repository setup, feature schema documented
May 13-17 (Week 1): Data loading + feature updates
May 20-24 (Week 2): Experiment setup + reproducibility
May 27-31 (Week 3): Training + evaluation + analysis
```

---

## Quick Reference Commands

### Verify Baseline
```bash
pytest tests/ -q              # Should show: 74 passed
```

### Check Git Status
```bash
git status                    # Should show: clean, on feature branch
git branch -v                 # Should show: research/multifamily_generalization *
```

### Run Feature Extraction
```bash
python -m src.features.dataset_builder_v2 --config config.json
```

### Train Model
```bash
python -m src.training.train_transformer \
  --train_npz data/train.npz \
  --val_npz data/val.npz \
  --model_dir experiments/multifamily_generalization
```

---

## Next Actions (When Ready)

1. **Read this document** ← You are here
2. **Proceed to Week 1 Day 1**: Create CTU-13 data loader
3. **Follow IMMEDIATE_ACTION_PLAN.md** for daily tasks
4. **Track progress with PROGRESS_CHECKLIST.md**

---

**Status**: ✅ Repository ready, infrastructure documented  
**Next**: Week 1 implementation starting May 13

