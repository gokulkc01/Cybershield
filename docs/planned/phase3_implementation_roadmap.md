# Phase 3: Implementation Roadmap

**Status**: Ready for implementation  
**Timeline**: 4 weeks (5 days/week)  
**Priority**: CRITICAL - Fixes zero-shot generalization failure

---

## Document Index

| Document | Purpose | Audience |
|----------|---------|----------|
| [phase3_architectural_redesign_plan.md](phase3_architectural_redesign_plan.md) | Full architectural vision, new components, design rationale | Architects, Tech Leads |
| [phase3_implementation_skeleton.md](phase3_implementation_skeleton.md) | Concrete code structure, Python skeletons, migration path | Implementers, Code Review |
| [phase3_scientific_justification.md](phase3_scientific_justification.md) | Why session-centric failed, why host-centric will work | Data Scientists, Validators |
| phase3_implementation_roadmap.md ← YOU ARE HERE | Weekly breakdown, task assignments, success criteria | Project Managers, Implementers |

---

## Week 1: Foundation (BehavioralWindow + HostBehavioralDataset)

### Goal
Implement the core data representation layer that converts host timelines into training samples.

### Tasks

#### Day 1: Behavioral Window Implementation (4 hrs)
- [ ] Create `src/features/behavioral_window.py`
  - [ ] `BehavioralWindow` dataclass
  - [ ] Properties: `duration`, `session_count`, `regularity`, `persistence_score`
  - [ ] `BehavioralWindowBuilder` class
  - [ ] Window extraction from timelines
- [ ] Unit tests: `tests/test_behavioral_window.py`
  - [ ] Test window chronological ordering
  - [ ] Test window sliding with various parameters
  - [ ] Validate regularit y computation
  - [ ] Verify persistence_score calculation

**Deliverable**: Behavioral windows can be extracted from any host timeline

#### Day 2: Temporal Dataset Implementation (4 hrs)
- [ ] Create `src/data_loader/temporal_dataset.py`
  - [ ] `HostBehavioralDataset` class
  - [ ] `__getitem__` returns (X, mask, y) correctly shaped
  - [ ] `create_temporal_dataloaders()` function
  - [ ] Proper masking for variable-length sequences
- [ ] Unit tests: `tests/test_temporal_dataset.py`
  - [ ] Verify tensor shapes: (batch, seq_len, 20, 12)
  - [ ] Test masking behavior
  - [ ] Verify padding works correctly
  - [ ] Check normalization per window

**Deliverable**: PyTorch dataloaders can iterate over behavioral windows

#### Day 3: Integration with Existing Timelines (3 hrs)
- [ ] Load existing `train_host_timelines.pkl` from `data/processed/v2_*`
- [ ] Extract behavioral windows
- [ ] Verify no data leakage, no temporal leakage
- [ ] Create test script: `scripts/test_window_extraction.py`
  - Loads UWF timelines
  - Extracts 10,000 windows
  - Prints statistics: window distribution, regularity distribution
  - Outputs sample windows to JSON for inspection

**Deliverable**: Window extraction works on real data

#### Day 4-5: Validation & Documentation (4 hrs)
- [ ] Run full test suite on new components
- [ ] Verify backward compatibility with existing code
- [ ] Document window extraction process
- [ ] Create walkthrough: "How to extract behavioral windows"

**Deliverable**: Week 1 foundation is robust, tested, documented

### Success Criteria (Week 1)
- ✅ `BehavioralWindow` objects can be created and introspected
- ✅ `BehavioralWindowBuilder.build_windows()` extracts >10K windows from UWF
- ✅ `HostBehavioralDataset` produces correctly-shaped tensors
- ✅ Dataloaders iterate without errors
- ✅ Chronological ordering is preserved
- ✅ All unit tests pass

### Risk Mitigation (Week 1)
| Risk | Mitigation |
|------|-----------|
| Window extraction is slow | Use NumPy operations, not loops for sorting |
| Memory issues with large windows | Lazy load session tensors, don't cache everything |
| Incorrect mask computation | Write explicit test cases for masking edge cases |
| Data leakage between splits | Verify window host_ids don't cross train/val/test |

---

## Week 2: Model Adaptation (Temporal Transformer)

### Goal
Modify the existing Transformer to accept and process temporal sequences.

### Tasks

#### Day 1: Model Forward Pass Adaptation (3 hrs)
- [ ] Modify `src/models/transformer.py`
  - [ ] Accept input shape: (batch, seq_len, 20, 12)
  - [ ] Process with masking support
  - [ ] Options:
    - Option A: Flatten seq_len * 20 as sequence of flows (simpler)
    - Option B: Add session-level aggregation, then temporal attention (better)
  - [ ] Start with Option A for fast iteration
- [ ] Update forward docstring
- [ ] Create test: dummy batch of shape (8, 50, 20, 12)

**Deliverable**: Model accepts temporal input shape

#### Day 2: Masking & Variable-Length Support (3 hrs)
- [ ] Implement attention masking for padded sequences
- [ ] Ensure masked positions don't contribute to gradients
- [ ] Test with variable batch masks:
  - Batch[0]: full 50 sessions
  - Batch[1]: only 10 sessions + 40 padding
- [ ] Verify loss/gradients compute correctly

**Deliverable**: Model handles variable-length sequences

#### Day 3: Model Training Integration (3 hrs)
- [ ] Skeleton `src/training/train_temporal_behavioral.py`
  - [ ] Load timelines
  - [ ] Build behavioral windows
  - [ ] Create dataloaders
  - [ ] Training loop adapted for temporal batches
- [ ] Update batch processing:
  - Loop over (batch_X, batch_mask, batch_y)
  - Pass mask to model
  - Compute loss only on real (non-padded) sequences

**Deliverable**: Training loop can process behavioral windows

#### Day 4: Hyperparameter Tuning (2 hrs)
- [ ] Determine good max_seq_len value
  - [ ] Histogram of window sizes in UWF
  - [ ] Target: capture 95% of windows without excessive padding
- [ ] Adjust batch size for temporal sequences (may need to decrease)
- [ ] Test learning rates with temporal input

**Deliverable**: Hyperparameters set for temporal training

#### Day 5: Validation & Integration (3 hrs)
- [ ] Ensure backward compatibility: old session-centric model still works
- [ ] Create end-to-end test script
  - Train on 100 UWF behavioral windows
  - Evaluate on 100 CTU-13 behavioral windows
  - Compare predictions before/after temporal model
- [ ] Document changes to model architecture

**Deliverable**: Temporal model is ready for real training

### Success Criteria (Week 2)
- ✅ Model accepts (batch, seq_len, 20, 12) shape
- ✅ Masking works correctly for variable-length sequences
- ✅ Loss computation handles masked values properly
- ✅ Gradient flow is correct
- ✅ Training loop completes without errors
- ✅ Inference generates predictions

### Risk Mitigation (Week 2)
| Risk | Mitigation |
|------|-----------|
| Model doesn't learn from temporal info | Add ablation: compare attention patterns to session-centric |
| Sequence length mismatch causes crashes | Write shape assertion in model forward pass |
| Memory OOM with temporal sequences | Reduce batch size or max_seq_len |
| Model regresses vs. session-centric | Track both architectures' validation curves |

---

## Week 3: Training & Evaluation (Zero-Shot Validation)

### Goal
Train host-centric model on UWF and evaluate zero-shot on CTU-13/MCFP.

### Tasks

#### Day 1: Data Preparation (2 hrs)
- [ ] Load/verify UWF timelines exist
- [ ] Load/verify CTU-13/MCFP test timelines exist
- [ ] Extract behavioral windows:
  - [ ] UWF: train/val windows
  - [ ] CTU-13/MCFP: test windows (no tuning)
- [ ] Create window statistics report

**Deliverable**: All required windows ready for training

#### Day 2-3: Model Training (6 hrs)
- [ ] Run `train_temporal_behavioral.py` on UWF windows
  - [ ] Track training/validation loss and metrics
  - [ ] Early stopping based on validation AUC
  - [ ] Optimal threshold tuning on validation set
  - [ ] Save checkpoint with embedded threshold
- [ ] Expected results (conservative):
  - Validation recall: 85%+
  - Validation AUC: 0.98+

**Deliverable**: Trained temporal model saved

#### Day 4: Zero-Shot Evaluation (3 hrs)
- [ ] Load trained model + optimal threshold
- [ ] Evaluate on CTU-13 windows (no threshold tuning)
- [ ] Evaluate on MCFP windows (no threshold tuning)
- [ ] Generate detailed report:
  - Per-source performance (CTU-13 vs. MCFP)
  - Confusion matrix
  - Recall @ various FPR targets
  - Predicted probabilities distribution
- [ ] **CRITICAL**: Compare to session-centric baseline
  - Session-centric recall: 0.19%
  - Target temporal recall: 75%+

**Deliverable**: Zero-shot evaluation report generated

#### Day 5: Analysis & Documentation (3 hrs)
- [ ] Analyze which windows/families temporal model detects
- [ ] Investigate any remaining false negatives
- [ ] Compare attention maps vs. session-centric model
- [ ] Document results and lessons learned
- [ ] Create visualization: Recall progression across phases

**Deliverable**: Comprehensive evaluation analysis

### Success Criteria (Week 3)
- ✅ Model trains on UWF windows: Recall 85%+ on validation
- ✅ Zero-shot CTU-13 recall: 75%+ (vs. 0.19% baseline)
- ✅ Zero-shot MCFP recall: 70%+ (vs. 0.19% baseline)
- ✅ FPR @ target recall: <2% (operational threshold)
- ✅ Recall improvement: >400x vs. session-centric

### Acceptable Falls-Back (If not met)
- If recall < 75%: Investigate with feature analysis tool
- If FPR too high: Adjust threshold via operating point analysis
- If model training diverges: Check attention masking

### Risk Mitigation (Week 3)
| Risk | Mitigation |
|------|-----------|
| Zero-shot recall still low (<50%) | Debug: Which windows/features help? |
| Training loss doesn't decrease | Check learning rate, gradient flow, input normalization |
| Overfitting to UWF | Increase regularization, add dropout, check window parameters |
| Evaluation crashes on large test set | Reduce batch size, process in chunks |

---

## Week 4: Integration, Optimization & Documentation

### Goal
Integrate host-centric training into main pipeline, optimize for production, document fully.

### Tasks

#### Day 1: Pipeline Integration (3 hrs)
- [ ] Create orchestrator: `src/pipelines/train_host_centric_zero_shot.py`
  - [ ] Analogous to `train_and_eval_zero_shot.py` but for temporal
  - [ ] Unified interface: load timelines, train, evaluate
- [ ] Add to runnable scripts: `scripts/run_phase3_zero_shot.py`
- [ ] Update CI/CD to run new pipeline

**Deliverable**: Temporal training integrated into main pipeline

#### Day 2: Performance Optimization (3 hrs)
- [ ] Profile: which components are slowest?
- [ ] Optimize:
  - [ ] Window extraction speed (use NumPy bulk operations)
  - [ ] DataLoader speed (increase num_workers if available)
  - [ ] Model forward pass (check for redundant operations)
- [ ] Document performance improvements

**Deliverable**: Pipeline runs efficiently end-to-end

#### Day 3: Comparison & Ablation (4 hrs)
- [ ] Side-by-side comparison:
  - Session-centric (baseline)
  - Host-centric with persistence features
  - Host-centric with full longitudinal features
- [ ] Ablation studies:
  - Without regularity: does persistence alone help?
  - Without destination stability: what's left?
  - Varying window size: 10 vs. 30 vs. 50 sessions
- [ ] Document findings

**Deliverable**: Comprehensive ablation report

#### Day 4: Documentation (4 hrs)
- [ ] Update README with temporal modeling explanation
- [ ] Create tutorial: "Training your own host-centric model"
- [ ] API documentation for new classes
- [ ] Architecture diagram: session-centric → host-centric
- [ ] FAQ: Why temporal? When to use window size X?

**Deliverable**: Complete documentation for Phase 3

#### Day 5: Testing & Validation (3 hrs)
- [ ] Full integration test suite
- [ ] Verify no regressions in existing functionality
- [ ] Test edge cases:
  - Empty timelines
  - Single-session windows
  - All-benign windows
  - All-C2 windows
- [ ] Performance benchmarks

**Deliverable**: All tests pass, no regressions

### Success Criteria (Week 4)
- ✅ Temporal pipeline fully integrated
- ✅ Documentation complete and accessible
- ✅ Performance acceptable for production
- ✅ No regressions in existing functionality
- ✅ Team can run temporal training end-to-end

### Risk Mitigation (Week 4)
| Risk | Mitigation |
|------|-----------|
| Integration breaks existing code | Create new files, don't modify existing |
| Performance still slow | Profile, identify bottleneck, optimize |
| Documentation unclear | Have teammate read and ask questions |

---

## Success Metrics Summary

### Phase 3 Success = Meeting ALL criteria:

```
PRIMARY METRICS (Non-negotiable):
  ✅ Zero-shot recall on CTU-13:     75%+ (vs. 0.19% baseline)
  ✅ Zero-shot recall on MCFP:       70%+ (vs. 0.19% baseline)
  ✅ FPR @ target recall:             <2%
  ✅ Improvement factor:              >400x vs. session-centric

SECONDARY METRICS (Indicators of success):
  ✅ Validation AUC:                  0.95+
  ✅ Per-host detection accuracy:     85%+
  ✅ Cross-family generalization:     Uniform performance

OPERATIONAL METRICS (Production readiness):
  ✅ Training time:                   <4 hours per source
  ✅ Inference time:                  <1 sec per host
  ✅ Memory usage:                    <8GB (CPU feasible)
  ✅ Code coverage:                   >80% (all new modules)

QUALITY METRICS (Engineering standards):
  ✅ All tests passing:               100%
  ✅ No regressions:                  0 failures in existing tests
  ✅ Documentation complete:          Every class/function documented
  ✅ Code review approval:            2+ approvals per PR
```

---

## Weekly Progress Check-In Template

### End of Week 1 Check-In
```
Date: [Friday]

✅ Completed:
- [ ] BehavioralWindow class implemented
- [ ] HostBehavioralDataset working
- [ ] 10K+ windows extracted from UWF
- [ ] All unit tests passing

🔴 Issues encountered:
- [List any blockers, with current status]

⏭️ Ready for Week 2:
- [ ] Confirm: Window extraction complete
- [ ] Confirm: Model adaptation can proceed
```

---

## File Checklist

### Files to Create (Phase 3)
```
src/features/behavioral_window.py         (Week 1)
src/data_loader/temporal_dataset.py       (Week 1)
src/training/train_temporal_behavioral.py (Week 2-3)
src/pipelines/train_host_centric_zero_shot.py (Week 4)

tests/test_behavioral_window.py           (Week 1)
tests/test_temporal_dataset.py            (Week 1)
tests/test_temporal_pipeline.py           (Week 3)

scripts/test_window_extraction.py         (Week 1)
scripts/run_phase3_zero_shot.py           (Week 4)

docs/phase3_weekly_progress.md            (Ongoing)
```

### Files to Modify
```
src/models/transformer.py                 (Week 2)
                                         - Accept (batch, seq_len, 20, 12) input
                                         - Add masking support
                                         - No breaking changes to interface
```

### Documents to Create
```
docs/phase3_architectural_redesign_plan.md       ✅ (CREATED)
docs/phase3_implementation_skeleton.md           ✅ (CREATED)
docs/phase3_scientific_justification.md          ✅ (CREATED)
docs/phase3_implementation_roadmap.md            ✅ (THIS DOCUMENT)
docs/phase3_weekly_progress.md                   (Week 1 onwards)
```

---

## Exit Criteria: Phase 3 Complete

**Phase 3 is DONE when:**

1. ✅ Behavioral windows can be extracted from host timelines
2. ✅ Temporal model trains on UWF windows with >85% validation recall
3. ✅ Zero-shot validation shows >400x improvement (75%+ recall on CTU-13/MCFP)
4. ✅ All new code is tested, documented, integrated
5. ✅ No regressions in existing session-centric functionality
6. ✅ Team can run full pipeline end-to-end

**Upon completion:**
- Transition to Phase 4: Feature Engineering + Longitudinal Analysis
- Begin integration of persistence, partner concentration, behavioral drift

---

## Dependencies & Prerequisites

### Already Available
- ✅ Host timelines saved in `data/processed/v2_*`
- ✅ SessionSummary objects with metadata
- ✅ Existing Transformer model
- ✅ Test data with ground truth labels
- ✅ Evaluation infrastructure

### Must Prepare
- [ ] Confirm timeline loading works
- [ ] Verify no temporal leakage in splits
- [ ] Check session tensor lookup availability

---

## Resource Allocation

| Component | Effort | Owner | Week |
|-----------|--------|-------|------|
| BehavioralWindow | 8h | [Dev1] | 1 |
| HostBehavioralDataset | 8h | [Dev2] | 1 |
| Model Adaptation | 8h | [Dev1] | 2 |
| Temporal Training Loop | 8h | [Dev2] | 2 |
| Zero-Shot Training | 12h | [Dev1] | 3 |
| Evaluation & Analysis | 8h | [Dev2] | 3 |
| Pipeline Integration | 8h | [Dev1] | 4 |
| Documentation | 8h | [Dev1+Dev2] | 4 |
| Testing & QA | 12h | [QA] | All |

**Total Effort**: ~80 developer-hours (2 people, 4 weeks)

---

## How to Use This Roadmap

1. **Print this document** and put on team wall
2. **Weekly standup**: Review progress against this roadmap
3. **Daily**: Check current week tasks
4. **Risk review**: Evaluate mitigations weekly
5. **Upon completion**: Archive as Phase 3 completion record

---

## Next Action

**Start Date**: [Monday of next week]
**Week 1 Lead**: [Developer assignment]
**Week 1 Goal**: Have BehavioralWindow + HostBehavioralDataset ready

**Before starting Week 1:**
- [ ] Read all Phase 3 documents in this folder
- [ ] Verify prerequisites (timelines exist, loads correctly)
- [ ] Set up development environment
- [ ] Create feature branch: `feature/phase3-temporal-modeling`

---

**END OF ROADMAP**

