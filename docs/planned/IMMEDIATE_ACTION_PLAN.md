# Immediate Action Plan: Controlled Generalization Experiment

**Date**: May 9, 2026  
**Priority**: 🔴 CRITICAL  
**Objective**: Set up controlled multi-family C2 generalization experiment

---

## Current Status

✅ **Good News**: Codebase is clean
- No host-centric code contamination
- Session-centric pipeline intact
- No half-finished temporal modeling

❌ **Issue Identified**: Previous Phase 3 planning invalid
- Experiment was confounded (multiple variables changed)
- Wrong architectural conclusion drawn
- Must restart with proper scientific design

---

## What Needs to Happen (Next 3 Weeks)

### Week 1: Data Preparation (May 13-17)

#### Day 1: Setup & Verification
- [ ] Create feature branch: `git checkout -b research/multifamily_generalization`
- [ ] Verify git status clean
- [ ] Review current session-centric pipeline
- [ ] Check existing feature schema (document current 12 features)

#### Day 2-3: Multi-Family Dataset Loading
- [ ] CTU-13 Scenario 1 (Neris botnet) → load and parse
  - [ ] Verify sessions count (target: ~50K+)
  - [ ] Extract ground truth labels
  - [ ] Compute statistics
  
- [ ] CTU-13 Scenario 2 (Kraken botnet) → load and parse
  - [ ] Verify sessions count (target: ~100K+)
  - [ ] Consistency check with Scenario 1 format
  
- [ ] CTU-13 Scenario 9 (Conficker botnet) → load and parse
  - [ ] Verify sessions count (target: ~150K+)
  - [ ] Same format, new family

- [ ] Create consolidated dataset manifest
  ```json
  {
    "scenario_1": { "family": "neris", "sessions": 50000, "c2_sessions": 45000 },
    "scenario_2": { "family": "kraken", "sessions": 100000, "c2_sessions": 92000 },
    "scenario_9": { "family": "conficker", "sessions": 150000, "c2_sessions": 140000 }
  }
  ```

#### Day 4: Feature Processing (Remove Ports)
- [ ] Current feature schema: 12 features (list them)
  - [ ] src_port ← REMOVE
  - [ ] dst_port ← REMOVE
  - [ ] Keep: duration, bytes, packets, cadence, direction, etc.

- [ ] Update feature schema to 10 features
- [ ] Update feature transforms for new schema
- [ ] Regenerate normalization baselines (on combined CTU data)
- [ ] Verify feature distributions reasonable

#### Day 5: First Data Run
- [ ] Load all three scenarios
- [ ] Convert to unified format
- [ ] Run feature transformation pipeline
- [ ] Generate feature statistics report
  ```
  Total sessions: 300,000
  C2 sessions: 277,000
  Benign sessions: 23,000
  Class imbalance: 92:8
  Feature means: [...]
  Feature stds: [...]
  ```

### Week 2: Experiment Setup (May 20-24)

#### Day 1-2: Family-Aware Split Generator
- [ ] Create `FamilyAwareSplitter` class
  ```python
  splitter = FamilyAwareSplitter(
      train_families=['neris', 'kraken'],
      test_families=['conficker'],
      benign_ratio=0.15,
      random_seed=42
  )
  train_indices, test_indices = splitter.split(all_sessions)
  ```
  
- [ ] Verify no family leakage
  ```python
  train_families = {s.family for s in train_sessions}
  test_families = {s.family for s in test_sessions}
  assert len(train_families & test_families) == 0  # No overlap
  ```

- [ ] Generate split manifest (reproducible)
- [ ] Compute per-family counts for train and test

#### Day 3: Model Training Setup
- [ ] Ensure Transformer unchanged (verify architecture identical)
- [ ] Create training config:
  ```json
  {
    "experiment_name": "multifamily_generalization",
    "train_families": ["neris", "kraken"],
    "test_families": ["conficker"],
    "features_removed": ["src_port", "dst_port"],
    "model_architecture": "transformer_unchanged",
    "random_seed": 42,
    "batch_size": 64,
    "epochs": 50,
    "optimizer": "AdamW",
    "learning_rate": 1e-4
  }
  ```

- [ ] Update training loop to log per-family metrics
- [ ] Add checkpointing

#### Day 4-5: Evaluation Pipeline
- [ ] Implement metric computation:
  - [ ] PR-AUC (per-family and overall)
  - [ ] Recall @ fixed FPR (1%, 2%, 5%)
  - [ ] Precision @ fixed Recall (70%, 80%, 90%)
  - [ ] ROC-AUC
  - [ ] Calibration metrics

- [ ] Implement per-family breakdown
- [ ] Implement error analysis:
  - [ ] False negative clustering
  - [ ] False positive clustering
  - [ ] Family-specific failure modes

### Week 3: Execution & Analysis (May 27-31)

#### Day 1-2: Run Experiment
- [ ] Train model on Neris + Kraken
- [ ] Generate validation curves (loss, metrics)
- [ ] Save best checkpoint

#### Day 3-4: Evaluation
- [ ] Evaluate on Conficker (zero-shot family)
- [ ] Generate per-family confusion matrix
- [ ] Compute all metrics
- [ ] Run error analysis

#### Day 5: Interpretation
- [ ] Apply decision logic:
  - [ ] Strong generalization? → No architectural redesign needed
  - [ ] Partial generalization? → Host-centric justified
  - [ ] Complete failure? → Major redesign required
- [ ] Document findings
- [ ] Plan next phase

---

## Deliverables by Week

### Week 1 Deliverables
```
✅ Multi-family dataset loaded (300K+ sessions)
✅ Feature schema updated (10 features, no ports)
✅ Dataset statistics report
✅ Feature distributions verified
✅ Setup complete for Week 2
```

### Week 2 Deliverables
```
✅ Family-aware split generator implemented
✅ Split manifest (reproducible)
✅ Training config snapshot
✅ Evaluation pipeline ready
✅ Error analysis tooling ready
```

### Week 3 Deliverables
```
✅ Model trained on multi-family data
✅ Per-family evaluation complete
✅ Detailed error analysis
✅ Decision logic applied
✅ Results interpretation + next steps
```

---

## Critical Don'ts During Experiment

- ❌ **Do NOT** include UWF in training (lab data confounds family effect)
- ❌ **Do NOT** change model architecture
- ❌ **Do NOT** change features mid-experiment
- ❌ **Do NOT** skip family-aware splits
- ❌ **Do NOT** use accuracy metric
- ❌ **Do NOT** assume results will show X (let data speak)

---

## Expected Outcomes & Branches

### Outcome 1: Strong Generalization (Target)
```
Recall ≥ 75% across all families
Conclusion: Session-centric works
Next: Minor tuning + deployment

Branch after this outcome:
  - No architectural redesign
  - Consider deployment path
```

### Outcome 2: Partial Generalization (Possible)
```
Recall 50-75% with systematic gaps
Conclusion: Session-level partially sufficient
Next: Host-centric becomes justified

Branch after this outcome:
  - Plan Phase 3 host-centric redesign (NOW with scientific justification)
  - Use this as baseline to validate Phase 3 improvements
```

### Outcome 3: Complete Failure (Unlikely)
```
Recall <50% even with diverse training
Conclusion: Session-level fundamentally insufficient
Next: Major architectural redesign

Branch after this outcome:
  - Multi-variable redesign justified
  - Host-centric + temporal sequences
  - Possibly graph modeling
```

---

## Metrics to Compute (Week 2-3)

### Primary Metrics (Imbalanced Data)
```
✅ PR-AUC (Precision-Recall Area Under Curve)
✅ Recall @ 1% FPR
✅ Recall @ 2% FPR
✅ Recall @ 5% FPR
✅ Precision @ 70% Recall
✅ Precision @ 80% Recall
```

### Secondary Metrics
```
✅ ROC-AUC
✅ Calibration error
✅ Brier score
```

### Per-Family Breakdown
```
| Family     | Seen in Train? | Recall | Precision | PR-AUC | Notes |
|------------|----------------|--------|-----------|--------|-------|
| Neris      | Yes            | ?%     | ?%        | ?      |       |
| Kraken     | Yes            | ?%     | ?%        | ?      |       |
| Conficker  | No (zero-shot) | ?%     | ?%        | ?      | Key metric |
```

---

## Success Criteria

**Experiment is successful if:**
1. ✅ All three families load correctly
2. ✅ Feature schema change (remove ports) works
3. ✅ Family-aware split prevents leakage
4. ✅ Model trains without errors
5. ✅ All metrics computed
6. ✅ Error analysis identifies patterns
7. ✅ Decision logic applied unambiguously

**We don't care about the outcome (Outcome 1/2/3), just that the experiment is clean.**

---

## Risk Mitigation

| Risk | How to Prevent |
|------|----------------|
| Family leakage in splits | Verify no overlap programmatically |
| Port removal breaks pipeline | Test on small subset first |
| Model doesn't converge | Check learning rate, gradient flow |
| Out of memory | Reduce batch size, process in chunks |
| Metrics not computed | Implement robust error handling |

---

## File Organization

```
docs/
├── research_plan_multifamily_generalization.md    ← Main reference
├── DIRECTION_CHANGE.md                            ← Status of invalid Phase 3
├── IMMEDIATE_ACTION_PLAN.md                       ← THIS DOCUMENT
│
└── [ARCHIVE - DO NOT USE]
    ├── phase3_index.md                            ❌ INVALID
    ├── phase3_quickstart_guide.md                 ❌ INVALID
    ├── phase3_scientific_justification.md         ❌ INVALID
    ├── phase3_architectural_redesign_plan.md      ❌ INVALID
    ├── phase3_implementation_skeleton.md          ❌ INVALID
    ├── phase3_implementation_roadmap.md           ❌ INVALID
    └── phase3_status_summary.md                   ❌ INVALID

src/
├── data_loader/          ← May need: family-aware split generator
├── features/             ← Update: remove src_port, dst_port
├── models/               ← NO CHANGES
└── training/             ← Update: log per-family metrics

experiments/multifamily_generalization/
├── config.json           ← Experiment metadata
├── train_curves.json     ← Training history
├── results.json          ← Final metrics
├── error_analysis.json   ← False negative/positive breakdown
└── best_model.pth        ← Trained checkpoint
```

---

## Git Strategy

```bash
# Create clean feature branch
git checkout -b research/multifamily_generalization

# Commit frequently
git add src/data_loader/family_splitter.py
git commit -m "Add family-aware split generator"

# After Week 1
git commit -m "Complete: Multi-family dataset loading + feature schema update"

# After Week 2
git commit -m "Complete: Family-aware splits + evaluation pipeline"

# After Week 3
git commit -m "Complete: Controlled generalization experiment [Outcome: X]"

# Then merge or PR based on results
```

---

## Summary

### Old Path (INVALID)
```
UWF experiment fails → Assume architecture broken → Redesign architecture
[Result: Wasted effort, still doesn't work]
```

### New Path (SCIENTIFIC)
```
Run controlled multi-family experiment
↓
Measure cross-family generalization
↓
Data answers: "Does session-centric work?"
↓
If YES: No redesign needed
If PARTIAL: Host-centric justified
If NO: Major redesign needed
```

### Next Step
Read: `docs/research_plan_multifamily_generalization.md`

Then start Week 1 Day 1.

---

**Status**: Ready to begin controlled experiment  
**Expected Duration**: 3 weeks  
**Principle**: Measure first, redesign only if needed

