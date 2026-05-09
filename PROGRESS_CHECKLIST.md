# Progress Tracking Checklist: Multifamily Generalization Experiment

**Date Started**: May 9, 2026 (Planning Phase)  
**Experiment Start**: Week 1 (May 13)  
**Expected Completion**: Week 3 (May 31)

## Status Snapshot (Updated May 10, 2026)

**Overall completion (engineering scaffolding + setup)**: ~55%  
**Overall completion (scientific experiment execution)**: ~20%

### Completed Since Start
- [x] Feature branch created and pushed (`research/multifamily_generalization`)
- [x] Environment setup validated (`.venv`, dependencies, tests passing)
- [x] Baseline test suite passing (74/74)
- [x] `FamilyAwareSplitter` implemented
- [x] CTU-13 multifamily loader scaffold implemented
- [x] Experiment 10-feature schema module implemented
- [x] Multifamily metrics module implemented (PR-AUC, ROC-AUC, recall@FPR, precision@recall)
- [x] Main training/eval pipeline wired to experiment schema via dedicated multifamily entrypoint
- [x] 12→10 feature conversion utility implemented
- [x] Available CTU processed NPZ files converted to 10-feature schema
- [x] Available-data multifamily manifest generated

### Current Blockers (Experiment Not Complete Yet)
- [ ] Raw CTU-13 scenario files for all required families are not present (currently only one raw capture file exists under `data/raw/ctu13`)
- [ ] Family-specific scenario NPZs for exact split target (`neris`, `kraken`, `conficker`) are missing
- [ ] True family-separated train/val/test execution has not run yet

### Immediate Next Executable Steps
- [ ] Generate scenario-specific NPZs for CTU-13 Scenario 1 (Neris), Scenario 2 (Kraken), Scenario 9 (Conficker)
- [ ] Build family-aware split manifest from those scenario-specific files
- [ ] Run multifamily training/evaluation pipeline on that split
- [ ] Fill Week 3 metrics + decision logic sections with real results

---

## Pre-Implementation Checklist (Week 1 - Before Day 1)

### Understanding
- [ ] Read README_RESEARCH_DIRECTION.md (30 min)
- [ ] Read DIRECTION_CHANGE.md (15 min)
- [ ] Skim research_plan_multifamily_generalization.md (20 min)
- [ ] Understand why Phase 3 was invalid
- [ ] Can explain: "One variable at a time" principle

### Repository Status
- [ ] Git working directory clean (`git status`)
- [ ] No uncommitted changes
- [ ] No stale branches

### Environment
- [ ] Python environment active (.venv)
- [ ] Dependencies installed
- [ ] Tests passing (`pytest tests/ -q`)

### Knowledge Gaps
- [ ] Understand CTU-13 scenarios (1, 2, 9)
- [ ] Know malware families (Neris, Kraken, Conficker)
- [ ] Know current feature schema (12 features, including src/dst_port)
- [ ] Know which features to remove (ports)

**Status**: ⏳ Not Started | 🟡 In Progress | ✅ Complete
- [ ] All items checked → Ready for Week 1

---

## Week 1: Data Preparation (May 13-17)

### Day 1: Setup & Verification

**Tasks**:
- [ ] Create feature branch: `git checkout -b research/multifamily_generalization`
- [ ] Document current state:
  - [ ] List current 12 features
  - [ ] Verify feature shapes
  - [ ] Check session count in existing data
- [ ] Read current dataset loading code
- [ ] Plan data loading architecture

**Verification**:
```bash
# Should work without errors
pytest tests/ -q
git status  # Clean
```

**Output**: Documented plan for multi-family loading

### Day 2-3: Multi-Family Dataset Loading

**CTU-13 Scenario 1 (Neris)**:
- [ ] Download/locate raw data
- [ ] Parse flow records
- [ ] Extract ground truth labels (C2 sessions identified)
- [ ] Compute statistics:
  - [ ] Total sessions: ___
  - [ ] C2 sessions: ___
  - [ ] Benign sessions: ___
- [ ] Verify format matches existing pipeline
- [ ] Document any format differences

**CTU-13 Scenario 2 (Kraken)**:
- [ ] Download/locate raw data
- [ ] Parse flow records
- [ ] Extract ground truth labels
- [ ] Compute statistics:
  - [ ] Total sessions: ___
  - [ ] C2 sessions: ___
  - [ ] Benign sessions: ___
- [ ] Cross-check format with Scenario 1
- [ ] Identify any inconsistencies

**CTU-13 Scenario 9 (Conficker)**:
- [ ] Download/locate raw data
- [ ] Parse flow records
- [ ] Extract ground truth labels
- [ ] Compute statistics:
  - [ ] Total sessions: ___
  - [ ] C2 sessions: ___
  - [ ] Benign sessions: ___
- [ ] Verify format consistency

**Consolidation**:
- [ ] Create unified dataset manifest (JSON):
  ```json
  {
    "scenario_1_neris": { "sessions": X, "c2": Y, "benign": Z },
    "scenario_2_kraken": { "sessions": X, "c2": Y, "benign": Z },
    "scenario_9_conficker": { "sessions": X, "c2": Y, "benign": Z },
    "total_sessions": X,
    "total_c2": Y,
    "total_benign": Z
  }
  ```
- [ ] Generate feature distribution report (mean, std per family)

**Output**: 300K+ sessions loaded, statistics computed

### Day 4: Feature Processing (Remove Ports)

**Current Feature Schema**:
- [ ] List current 12 features: `[ ____, ____, ..., ____ ]`
- [ ] Identify src_port and dst_port indices
- [ ] Document feature meanings

**New Feature Schema**:
- [ ] Remove src_port (feature index: ___)
- [ ] Remove dst_port (feature index: ___)
- [ ] Verify 10 remaining features
- [ ] Update feature config file
- [ ] Regenerate feature transforms

**Pipeline Update**:
- [ ] Update feature normalization (recompute on CTU data)
- [ ] Update z-score baselines
- [ ] Test feature pipeline on small subset
- [ ] Verify feature distributions reasonable

**Output**: 10-feature pipeline ready, normalization updated

### Day 5: First Data Run & Statistics

**Load All Data**:
- [ ] Load Scenarios 1, 2, 9
- [ ] Apply feature transforms (10 features)
- [ ] Generate unified dataset

**Statistics Report**:
```
Total sessions: ___
Total C2 sessions: ___
Total benign sessions: ___
C2 ratio: ___%

Per-family breakdown:
  Neris: ___ total, ___ C2
  Kraken: ___ total, ___ C2
  Conficker: ___ total, ___ C2

Feature statistics:
  Feature 1 (duration): mean=___, std=___
  Feature 2 (bytes): mean=___, std=___
  ... (all 10 features)

Class imbalance: ___:___ (C2:Benign)
```

**Output**: Comprehensive dataset ready for Week 2

**Week 1 Status**:
- [ ] All data loaded
- [ ] Feature schema updated
- [ ] Statistics computed
- [ ] Ready for Week 2

---

## Week 2: Experiment Setup (May 20-24)

### Day 1-2: Family-Aware Split Generator

**Implementation**:
- [ ] Create `FamilyAwareSplitter` class
- [ ] Implement logic:
  - [ ] Assign Scenarios 1, 2 to training
  - [ ] Assign Scenario 9 to test
  - [ ] Verify no family overlap
  - [ ] Preserve benign ratio

**Verification**:
```python
# These should return empty set
train_families = {s.family for s in train_sessions}
test_families = {s.family for s in test_sessions}
overlap = train_families & test_families
assert len(overlap) == 0  # ✅ Pass
```

**Output**: Split manifest
```json
{
  "split_name": "family_aware_v1",
  "random_seed": 42,
  "train_families": ["neris", "kraken"],
  "test_families": ["conficker"],
  "train_sessions": ___,
  "test_sessions": ___,
  "train_c2": ___,
  "test_c2": ___,
  "benign_ratio_train": 0.15,
  "benign_ratio_test": 0.15
}
```

### Day 3: Model Training Setup

**Training Configuration**:
- [ ] Create config file: `experiments/multifamily_generalization/config.json`
  ```json
  {
    "experiment_name": "multifamily_generalization",
    "date": "2026-05-09",
    "train_families": ["neris", "kraken"],
    "test_families": ["conficker"],
    "features_removed": ["src_port", "dst_port"],
    "model_architecture": "transformer_unchanged",
    "random_seed": 42,
    "batch_size": 64,
    "epochs": 50,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "early_stopping": true,
    "patience": 5
  }
  ```

**Model Verification**:
- [ ] Load existing Transformer model
- [ ] Verify architecture unchanged
- [ ] Verify input shape: (batch, 20, 12) → (batch, 20, 10) with port removal
- [ ] Test forward pass on dummy data

**Training Loop Update**:
- [ ] Add per-family metric logging
- [ ] Add checkpointing (save best model)
- [ ] Verify reproducibility (fixed seed)

**Output**: Training setup complete, ready to train

### Day 4-5: Evaluation Pipeline

**Metrics Implementation**:
- [ ] PR-AUC computation
- [ ] Recall @ 1%, 2%, 5% FPR
- [ ] Precision @ 70%, 80%, 90% Recall
- [ ] ROC-AUC
- [ ] Calibration metrics

**Per-Family Analysis**:
- [ ] Confusion matrix per family
- [ ] Recall per family
- [ ] Precision per family
- [ ] PR-AUC per family

**Error Analysis**:
- [ ] False negative clustering
  - [ ] Group by family
  - [ ] Group by protocol
  - [ ] Group by duration
  - [ ] Identify patterns
- [ ] False positive clustering
  - [ ] Group by benign app type
  - [ ] Group by periodicity
  - [ ] Identify confusion sources

**Output**: Comprehensive evaluation pipeline ready

**Week 2 Status**:
- [ ] Split generator working
- [ ] No family leakage verified
- [ ] Training config ready
- [ ] Metrics pipeline complete
- [ ] Ready for Week 3 training

---

## Week 3: Execution & Analysis (May 27-31)

### Day 1-2: Model Training

**Training**:
- [ ] Start training on (Neris + Kraken)
- [ ] Monitor:
  - [ ] Training loss: ___
  - [ ] Validation loss: ___
  - [ ] Validation AUC: ___
  - [ ] Validation recall: ___
- [ ] Check for convergence
- [ ] Save best checkpoint

**Curves to Generate**:
- [ ] Training loss over epochs
- [ ] Validation loss over epochs
- [ ] Validation metrics over epochs
- [ ] Learning curve (sample efficiency)

**Output**: Trained model checkpoint

### Day 3: Evaluation on Held-Out Family

**Evaluate on Conficker (Zero-Shot)**:
- [ ] Load best model
- [ ] Run inference on test set
- [ ] Compute all metrics:
  - [ ] PR-AUC: ___
  - [ ] ROC-AUC: ___
  - [ ] Recall @ 1% FPR: ___%
  - [ ] Recall @ 2% FPR: ___%
  - [ ] Recall @ 5% FPR: ___%
  - [ ] Precision @ 70% Recall: ___%
  - [ ] Precision @ 80% Recall: ___%

**Per-Family Results**:
```
| Family     | In Training? | Recall | Precision | PR-AUC | Notes |
|------------|--------------|--------|-----------|--------|-------|
| Neris      | Yes          | ___%   | ___%      | ___    |       |
| Kraken     | Yes          | ___%   | ___%      | ___    |       |
| Conficker  | No           | ___%   | ___%      | ___    | KEY   |
```

**Output**: Comprehensive evaluation results

### Day 4: Error Analysis

**False Negatives**:
- [ ] Total FN count: ___
- [ ] FN per family:
  - [ ] Neris: ___
  - [ ] Kraken: ___
  - [ ] Conficker: ___
- [ ] Common characteristics:
  - [ ] Protocol distribution
  - [ ] Duration distribution
  - [ ] Bytes distribution
  - [ ] Packet patterns
- [ ] Top 10 FN clusters

**False Positives**:
- [ ] Total FP count: ___
- [ ] Common benign applications triggering FP
- [ ] Protocol distribution
- [ ] Periodicity patterns
- [ ] Top 10 FP clusters

**Output**: Detailed error analysis report

### Day 5: Interpretation & Decision

**Apply Decision Logic**:

**If Recall ≥ 75% on Conficker** → Strong Generalization
- [ ] Session-level behavioral invariants validated
- [ ] No architectural redesign needed
- [ ] Next: Deployment preparation
- [ ] Document: "Session-centric sufficient for multi-family"

**If Recall 50-75% on Conficker** → Partial Generalization
- [ ] Session-level partially sufficient
- [ ] Missing information identified in error analysis
- [ ] Next: Plan host-centric redesign (NOW JUSTIFIED)
- [ ] Document: "Session-level limited, temporal context needed"

**If Recall <50% on Conficker** → Complete Failure
- [ ] Session-level fundamentally insufficient
- [ ] Multi-variable redesign necessary
- [ ] Next: Major architecture change
- [ ] Document: "Need temporal/graph modeling"

**Final Report**:
- [ ] Summary of findings
- [ ] Key metrics
- [ ] Error analysis highlights
- [ ] Decision: Next phase direction
- [ ] Recommendations

**Output**: Final experiment report + next phase plan

**Week 3 Status**:
- [ ] Training complete
- [ ] Evaluation complete
- [ ] Error analysis complete
- [ ] Results interpreted
- [ ] Next phase planned

---

## Post-Experiment Checklist

### Documentation
- [ ] Experiment config saved
- [ ] All metrics computed
- [ ] Error analysis complete
- [ ] Results written up
- [ ] Decision logic applied

### Reproducibility
- [ ] Split manifest saved
- [ ] Model checkpoint saved
- [ ] Training curves saved
- [ ] Random seed documented
- [ ] Feature schema documented

### Next Phase Planning
- [ ] Outcome clearly identified (Strong/Partial/Complete)
- [ ] Next steps documented
- [ ] Timeline estimated
- [ ] Resources allocated

### Code Quality
- [ ] All code committed
- [ ] Tests updated
- [ ] Documentation updated
- [ ] Code reviewed

---

## Success Criteria Verification

✅ **Experiment Successful If**:
- [ ] All three families loaded correctly
- [ ] Feature schema updated (ports removed)
- [ ] Family-aware split implemented (no leakage)
- [ ] Model trained without errors
- [ ] All metrics computed
- [ ] Error analysis identifies patterns
- [ ] Decision outcome clear (1, 2, or 3)
- [ ] Next phase planned

❌ **Experiment Failed If**:
- [ ] Family leakage detected
- [ ] Metrics incomplete
- [ ] Results ambiguous
- [ ] Unknown failures

---

## Notes & Observations

**Week 1 Notes**:
```
[Space for observations]




```

**Week 2 Notes**:
```
[Space for observations]




```

**Week 3 Notes**:
```
[Space for observations]




```

**Key Insights**:
```
[Space for capturing key findings]




```

---

## Final Status

**Experiment Complete On**: [Date] ___________

**Outcome**: [ ] Strong | [ ] Partial | [ ] Complete

**Next Phase**: _________________________________

**Created By**: _________________________________

**Reviewed By**: _________________________________

---

This checklist should be printed and tracked weekly. ✅ items as completed.

