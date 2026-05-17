# Phase 1 Quick Checklist: Enhanced Features (Print & Use)

**Timeline**: 6 weeks (May 13 - June 24, 2026)  
**Goal**: Add 30 new features, validate cross-dataset

---

## ✅ WEEK 1: Feature Extractors

### Monday-Tuesday: TLS Extractor
- [ ] Create `src/features/tls_features.py`
- [ ] Implement `TLSFeatureExtractor` class
- [ ] Add 12 TLS features
- [ ] Create `tests/test_tls_features.py`
- [ ] Run tests: `pytest tests/test_tls_features.py -v`
- [ ] All tests passing ✅

### Tuesday-Wednesday: DNS Extractor
- [ ] Create `src/features/dns_features.py`
- [ ] Implement `DNSFeatureExtractor` class
- [ ] Add 8 DNS features (including DGA score)
- [ ] Create `tests/test_dns_features.py`
- [ ] Run tests: `pytest tests/test_dns_features.py -v`
- [ ] All tests passing ✅

### Wednesday-Thursday: Temporal Extractor
- [ ] Create `src/features/temporal_features.py`
- [ ] Implement `TemporalStatsExtractor` class
- [ ] Add 10 temporal features (IAT entropy, burst analysis, periodicity)
- [ ] Create `tests/test_temporal_features.py`
- [ ] Run tests: `pytest tests/test_temporal_features.py -v`
- [ ] All tests passing ✅

### Friday: Integration & Validation
- [ ] Update `src/data_loader/feature_transforms.py` with `extract_extended_40_features()`
- [ ] Update `src/features/feature_config.py` with new schema
- [ ] Run integration test: can load 40-feature dataset
- [ ] Verify feature extraction works end-to-end
- [ ] All tests passing ✅

---

## ✅ WEEK 2: Feature Pipeline & Model Training

### Monday-Tuesday: Schema Update
- [ ] Add FEATURE_NAMES_EXTENDED to feature_config.py (40 features)
- [ ] Add FEATURE_DIM_EXTENDED = 40
- [ ] Create backward compatibility layer (10-feature model still works)
- [ ] Update backend detection service to support 40 features
- [ ] Run all tests: `pytest tests/ -v`
- [ ] All passing ✅

### Wednesday-Thursday: Model Training
- [ ] Train Transformer on 40-feature CTU-13 dataset
- [ ] Verify model converges normally
- [ ] Measure recall, AUC, FPR
- [ ] Compare against 10-feature baseline (should be ≥ baseline)
- [ ] Save checkpoint: `experiments/transformer_40_features/best_model.pth`
- [ ] Document results ✅

### Friday: Documentation
- [ ] Document all 40 features with meanings
- [ ] Create feature importance baseline (will use later)
- [ ] Create feature extraction README
- [ ] Update main README with Phase 1 status
- [ ] Commit to git with message "Phase 1 Week 1-2: Feature extractors complete"

---

## ✅ WEEK 3: Dataset Preparation

### Monday: UGR'16 Dataset
- [ ] Download UGR'16 dataset (or obtain Zeek logs)
- [ ] Create `src/pipelines/prepare_ugr16.py`
- [ ] Parse UGR'16 Zeek logs → NPZ format
- [ ] Extract sessions with 40 features
- [ ] Save: `data/processed/ugr16_sessions_40feat.npz`
- [ ] Verify: load and inspect dataset
- [ ] Record stats: number of sessions, C2/benign split

### Tuesday: CICIDS2017 Dataset
- [ ] Download CICIDS2017 dataset (or Zeek logs)
- [ ] Create `src/pipelines/prepare_cicids.py`
- [ ] Parse CICIDS2017 → NPZ format
- [ ] Extract sessions with 40 features
- [ ] Save: `data/processed/cicids2017_sessions_40feat.npz`
- [ ] Record stats

### Wednesday: MAWI Archive
- [ ] Download MAWI samples
- [ ] Create `src/pipelines/prepare_mawi.py`
- [ ] Parse MAWI → NPZ format (sample, not full archive)
- [ ] Extract 100K+ benign sessions
- [ ] Save: `data/processed/mawi_benign_sample_40feat.npz`
- [ ] Record stats

### Thursday-Friday: Dataset Validation
- [ ] Verify all datasets load without error
- [ ] Check feature distributions are reasonable
- [ ] Verify no NaN or inf values
- [ ] Commit dataset scripts to git
- [ ] Document dataset sizes and sources

---

## ✅ WEEK 4: Cross-Dataset Training

### Monday-Tuesday: CTU → UGR
- [ ] Train model on CTU-13 (original split)
- [ ] Evaluate on UGR'16 test set
- [ ] Measure: recall, AUC, FPR, precision
- [ ] Save results: `experiments/cross_dataset/ctu_train_ugr_test.json`
- [ ] Expected recall: ≥ 60% (will improve with Phase 2-3)

### Tuesday-Wednesday: UGR → CTU
- [ ] Train model on UGR'16
- [ ] Evaluate on CTU-13 Conficker (unseen family)
- [ ] Measure: recall, AUC, FPR, precision
- [ ] Save results: `experiments/cross_dataset/ugr_train_ctu_test.json`

### Wednesday-Thursday: CICIDS → CTU
- [ ] Train model on CICIDS2017
- [ ] Evaluate on CTU-13 Conficker
- [ ] Save results: `experiments/cross_dataset/cicids_train_ctu_test.json`

### Friday: Analysis
- [ ] Compile cross-dataset results table
- [ ] Calculate average cross-dataset recall
- [ ] Document findings: which datasets work well together?
- [ ] Create visualization: heatmap of train→test recall

---

## ✅ WEEK 5: Feature Importance Analysis

### Monday-Wednesday: Ablation Study
- [ ] Train model without TLS features (30 features)
  - Measure recall → record
- [ ] Train model without DNS features (32 features)
  - Measure recall → record
- [ ] Train model without temporal features (30 features)
  - Measure recall → record
- [ ] Train model with only TLS (12 features)
  - Measure recall → record
- [ ] Create table: ablation results

### Thursday-Friday: Baseline Comparisons
- [ ] Train XGBoost on 40 features
  - Measure AUC on test set
- [ ] Train Random Forest on 40 features
  - Measure AUC on test set
- [ ] Train LSTM on 40 features
  - Measure AUC on test set
- [ ] Create comparison table: Transformer vs. baselines
- [ ] Transformer should win by ≥5% AUC

---

## ✅ WEEK 6: Documentation & Publication

### Monday-Tuesday: Comprehensive Report
- [ ] Write Phase 1 results document
- [ ] Include: methodology, datasets, features, results
- [ ] Include: ablation studies
- [ ] Include: baseline comparisons
- [ ] Include: cross-dataset validation
- [ ] Include: limitations and next steps
- [ ] Create: `docs/phase1_results.md`

### Wednesday-Thursday: Publication Prep
- [ ] Create tables and figures for paper
- [ ] Document reproducibility (seeds, hyperparameters)
- [ ] Create: `PHASE_1_COMPLETE.md`
- [ ] Update main README with Phase 1 results

### Friday: Finalization
- [ ] Review all code for quality
- [ ] Update test coverage (aim for >90%)
- [ ] Commit to git: "Phase 1 Complete"
- [ ] Create GitHub release tag: `v0.2.0-phase1`
- [ ] Celebrate! 🎉

---

## Success Criteria (End of Week 6)

### Feature Engineering ✅
- [ ] TLS features extracted correctly
- [ ] DNS features extracted correctly
- [ ] Temporal features extracted correctly
- [ ] All 40 features in unified format
- [ ] Backward compatibility maintained

### Dataset Validation ✅
- [ ] UGR'16 dataset prepared
- [ ] CICIDS2017 dataset prepared
- [ ] MAWI dataset prepared
- [ ] All datasets load without errors
- [ ] Feature distributions reasonable

### Model Performance ✅
- [ ] CTU model recalls ≥96% (baseline maintained)
- [ ] Cross-dataset recall ≥70% on average
- [ ] Feature importance understood
- [ ] Transformer beats baselines by ≥5% AUC

### Documentation ✅
- [ ] Phase 1 report completed
- [ ] All results reproducible
- [ ] Code commented and clean
- [ ] Tests passing (>90% coverage)
- [ ] Ready for Phase 2 start

---

## Daily Standup Template

Use this daily:

```
DONE (Yesterday):
- [ ] TLS feature extraction (40% complete)

IN PROGRESS (Today):
- [ ] DNS feature extraction

BLOCKERS:
- [ ] None

NEXT:
- [ ] Temporal feature extraction
```

---

## Metrics to Track

Track these throughout Phase 1:

```
Week 1:
  - Feature extractors: TLS ✅ DNS ✅ Temporal ✅
  - Tests passing: 12/12 ✅
  
Week 2:
  - 40-feature model recall: 96.3%
  - vs 10-feature baseline: 96.36%
  - Improvement: +0.06% (expected, adding features doesn't always help)
  
Week 3:
  - UGR'16 prepared: 500K sessions ✅
  - CICIDS prepared: 1M sessions ✅
  - MAWI prepared: 100K sessions ✅
  
Week 4:
  - CTU→UGR recall: 65%
  - UGR→CTU recall: 68%
  - CICIDS→CTU recall: 62%
  - Average: 65% (target: ≥70% by end of Phase 1)
  
Week 5:
  - TLS ablation impact: +2.1% recall
  - DNS ablation impact: +1.8% recall
  - Temporal ablation impact: +3.2% recall
  - Transformer vs XGBoost: Transformer AUC +7.2%
  
Week 6:
  - Phase 1 report: DONE ✅
  - Code coverage: 92%
  - All tests: PASS ✅
```

---

## Risk Mitigation

If something fails:

**"Model doesn't train with 40 features"**
- Solution: Check for NaN/inf in features, reduce feature dimension
- Fallback: Use 30 features (drop least important)

**"Cross-dataset recall drops to <50%"**
- Solution: Features might be dataset-specific (like ports)
- Fallback: Increase regularization, use only top-10 features

**"Datasets fail to prepare"**
- Solution: Check Zeek log format, may need preprocessing
- Fallback: Use simpler datasets (just CTU-13, skip UGR/CICIDS)

**"Baseline models are too slow"**
- Solution: Reduce feature dimension or sample size
- Fallback: Skip full baseline comparisons, do partial comparison

---

## Phase 1 Success = Phase 2 Unlock

Once Phase 1 is ✅:
- You understand which features matter
- You have validated cross-dataset generalization
- You have confidence in 40-feature architecture
- You're ready to add host-level reasoning (Phase 2)

**Phase 1 → Phase 2**: Start Phase 2 Week 1 work

---

## Contacts & Resources

### Documentation
- Strategic roadmap: [STRATEGIC_ROADMAP_2026.md](STRATEGIC_ROADMAP_2026.md)
- Implementation guide: [PHASE_1_IMPLEMENTATION_GUIDE.md](PHASE_1_IMPLEMENTATION_GUIDE.md)
- Kickoff doc: [IMPLEMENTATION_KICKOFF.md](IMPLEMENTATION_KICKOFF.md)

### Code References
- TLS code template: In PHASE_1_IMPLEMENTATION_GUIDE.md Week 1 Task 1.2
- DNS code template: In PHASE_1_IMPLEMENTATION_GUIDE.md Week 1 Task 1.3
- Temporal code template: In PHASE_1_IMPLEMENTATION_GUIDE.md Week 1 Task 1.4

### Key Commands
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_tls_features.py::test_tls_feature_extraction -v

# Train model
python src/training/train_transformer.py --features 40

# Cross-dataset evaluation
python src/pipelines/cross_dataset_train_eval.py
```

---

**Print this checklist. Check off each item. Measure progress daily. Phase 1 done in 6 weeks. Then Phase 2. Let's go! 🚀**

