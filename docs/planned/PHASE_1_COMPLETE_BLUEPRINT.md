# Phase 1 Complete Blueprint: Features + Data Integration

**Status**: Ready to implement  
**Timeline**: 6 weeks (May 13 - June 24, 2026)

---

## The Big Picture

Phase 1 has **two critical components** that must work together:

```
FEATURES (Week 1-2)              DATA (Week 3)
├─ TLS extractor                 ├─ CTU-13 (have)
├─ DNS extractor                 ├─ MAWI (download)
├─ Temporal extractor            ├─ UGR'16 (download)
├─ Integration layer             └─ Feature extraction
└─ Model training                  & validation

         ↓                              ↓
         └──────────┬──────────────────┘
                    ↓
            CROSS-DATASET TEST (Week 4)
            ├─ Train on CTU-13
            ├─ Test on UGR'16, MAWI
            ├─ Measure recall, FPR
            └─ Prove generalization
```

**Without data**: Features are hypothetical  
**Without features**: Data is inadequate  

**Both together**: Proof that we're learning real patterns

---

## Phase 1 Success = Answer This Question

```
┌─────────────────────────────────────────┐
│  Is our 96.36% recall REAL or MEMORY?  │
│                                         │
│  REAL → Continue to Phase 2             │
│  MEMORY → Redesign Phase 1              │
│  PARTIAL → Iterate on features          │
└─────────────────────────────────────────┘
```

### The Test

```
Train: CTU-13  →  Test: UGR'16
Recall target: ≥70%

If ✅: We learned generalizable patterns
If ❌: We memorized dataset artifacts
If ⚠️: Features need improvement
```

---

## Week 1: Feature Engineering

### Task 1.1: TLS Feature Extractor
- **File**: `src/features/tls_features.py`
- **Code**: Copy from PHASE_1_IMPLEMENTATION_GUIDE.md
- **Features**: 12 TLS behavioral signals
- **Tests**: `tests/test_tls_features.py`
- **Status**: Ready to copy-paste
- **Time**: 2 hours

### Task 1.2: DNS Feature Extractor
- **File**: `src/features/dns_features.py`
- **Code**: Copy from PHASE_1_IMPLEMENTATION_GUIDE.md
- **Features**: 8 DNS behavioral signals
- **Tests**: `tests/test_dns_features.py`
- **Status**: Ready to copy-paste
- **Time**: 2 hours

### Task 1.3: Temporal Feature Extractor
- **File**: `src/features/temporal_features.py`
- **Code**: Copy from PHASE_1_IMPLEMENTATION_GUIDE.md
- **Features**: 10 temporal statistics
- **Tests**: `tests/test_temporal_features.py`
- **Status**: Ready to copy-paste
- **Time**: 2 hours

### Task 1.4: Integration
- **File**: `src/data_loader/feature_transforms.py`
- **Change**: Add `extract_extended_40_features()` function
- **Input**: Session data + optional Zeek logs
- **Output**: 40-dimensional feature vector
- **Status**: Requires 1 new function (40 lines of code)
- **Time**: 1 hour

**Week 1 Total**: 7 hours (or less with parallel work)

---

## Week 2: Feature Validation & Model Training

### Task 2.1: Model Training (40 features)
- **Dataset**: CTU-13 (existing)
- **Input**: 40-feature sessions
- **Model**: Transformer (no changes to architecture)
- **Metrics**: Recall, AUC, FPR
- **Baseline**: 96.36% (from 10-feature model)
- **Target**: ≥96% (maintain or improve)
- **Time**: 2 hours (training) + 2 hours (analysis)

### Task 2.2: Schema Update
- **File**: `src/features/feature_config.py`
- **Change**: Add `FEATURE_NAMES_EXTENDED` (40 features)
- **Change**: Add `FEATURE_DIM_EXTENDED = 40`
- **Status**: 10 lines of code
- **Time**: 30 minutes

### Task 2.3: Backend Compatibility
- **File**: `backend/app/services/detection.py`
- **Change**: Support both 10 and 40 feature models
- **Status**: Already designed for flexible feature_dim
- **Time**: 30 minutes (verification only)

**Week 2 Total**: 6 hours

---

## Week 3: Data Acquisition & Preparation

### Task 3.1: Verify CTU-13 ✅
- **Status**: Probably already have
- **Action**: Verify Zeek logs exist
- **If missing**: Convert pcaps to Zeek (2-3 hours)
- **Time**: 30 minutes to verify

### Task 3.2: Download MAWI (New)
- **Source**: http://mawi.wide.ad.jp/mawi/samplepoint-F/
- **What**: Real ISP benign traffic
- **Size**: 500MB-1GB (5-10 days)
- **Time**: 2 hours (download) + 2 hours (processing)
- **Why**: Ground truth for FPR estimation

### Task 3.3: Download UGR'16 (New)
- **Source**: https://nesg.ugr.es/nesg/
- **What**: Spanish botnet/intrusion dataset
- **Size**: ~300MB
- **Time**: 1 hour (download) + 2 hours (processing)
- **Why**: Cross-dataset validation with different families

### Task 3.4: Feature Extraction All Datasets
- **Script**: `extract_40_features_from_dataset.py` (new)
- **Input**: All 3 datasets (CTU-13, MAWI, UGR'16)
- **Output**: 3 NPZ files (each with 40-feature arrays)
- **Validation**: Check for NaN/inf, reasonable ranges
- **Time**: 3 hours

### Task 3.5: Dataset Documentation
- **File**: `DATASET_MANIFEST.json`
- **Content**: Checksums, sizes, labels, dates, notes
- **Purpose**: Reproducibility and tracking
- **Time**: 1 hour

**Week 3 Total**: 12 hours (includes download wait time)

---

## Week 4: Cross-Dataset Validation

### Experiment 4.1: CTU-13 → UGR'16
```
Train: CTU-13 all families
Test:  UGR'16 all samples

Metrics:
  Recall: ?  (target ≥70%)
  FPR:    ?  (target ≤5%)
  AUC:    ?  (target ≥0.92)
```

### Experiment 4.2: CTU-13 → MAWI
```
Train: CTU-13 all families
Test:  MAWI benign samples

Metrics:
  FPR: ? (target ≤2%)
  Precision: ? (target ≥98%)
```

### Experiment 4.3: UGR'16 → CTU-13
```
Train: UGR'16 all samples
Test:  CTU-13 Conficker (unseen family)

Metrics:
  Recall: ? (target ≥70%)
  Should prove generalization to unseen families
```

### Experiment 4.4: Mixed Training
```
Train: CTU-13 + UGR'16 (50/50)
Test:  Each held out

Metrics:
  Stability: ?
  Improvement: ?
```

**Week 4 Total**: 8 hours (mostly waiting for training)

---

## Week 5-6: Analysis & Publication

### Task 5.1: Feature Importance Analysis
```
Which features matter most?
- Ablation study: train without each feature
- Measure impact on recall/AUC
- Document in report
```

### Task 5.2: Baseline Comparisons
```
How does Transformer compare?
- XGBoost on 40 features
- Random Forest on 40 features  
- LSTM on 40 features
- GRU on 40 features
```

### Task 5.3: Comprehensive Report
```
Document:
- Methodology
- Datasets (sizes, sources, splits)
- Features (definitions, extraction)
- Results (tables, figures)
- Ablation studies
- Baseline comparisons
- Cross-dataset generalization
- Limitations and future work
```

**Week 5-6 Total**: 10 hours

---

## Critical Integration Points

### Point 1: Feature Extraction Pipeline
```python
# Input: Session data (network flows)
# Process: TLS + DNS + Temporal extractors
# Output: 40-feature numpy array
# Integration: Drop into model training

def extract_extended_40_features(session_data, 
                                 zeek_ssl_log=None, 
                                 zeek_dns_log=None):
    """
    Extract 40 features from session.
    
    40 = base_10 + tls_12 + dns_8 + temporal_10
    """
    base_10 = extract_base_10_features(session_data)
    tls_12 = tls_extractor.extract(zeek_ssl_log)
    dns_8 = dns_extractor.extract(zeek_dns_log)
    temporal_10 = temporal_extractor.extract(session_data)
    
    return np.concatenate([base_10, tls_12, dns_8, temporal_10])
```

### Point 2: Dataset Loading Pipeline
```python
# Input: Dataset name (e.g., "ctu13", "mawi", "ugr16")
# Process: Load NPZ file, prepare splits
# Output: Train/test tensors
# Integration: Feed to model training

def load_dataset(name, split='train'):
    data = np.load(f'data/processed/{name}_sessions_40feat.npz')
    sessions = data['sessions']  # (N, 40)
    labels = data['labels']      # (N,)
    
    # Train/test split
    idx_train, idx_test = split_stratified(labels)
    if split == 'train':
        return sessions[idx_train], labels[idx_train]
    else:
        return sessions[idx_test], labels[idx_test]
```

### Point 3: Cross-Dataset Training Script
```python
# Input: Source dataset, target dataset
# Process: Train on source, evaluate on target
# Output: Metrics (recall, FPR, AUC)

for source in ["ctu13", "ugr16"]:
    for target in ["ctu13", "ugr16", "mawi"]:
        if source == target:
            continue  # No self-testing
        
        X_train, y_train = load_dataset(source, 'train')
        X_test, y_test = load_dataset(target, 'test')
        
        model = C2Transformer(feature_dim=40)
        model.fit(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        results[f"{source}→{target}"] = metrics
        
        print(f"Train: {source} → Test: {target}")
        print(f"  Recall: {metrics['recall']:.1%}")
        print(f"  FPR: {metrics['fpr']:.2%}")
```

---

## Dependency Chain

```
Week 1-2: Features Ready
         ↓
Week 3: Data Ready
         ↓
Week 4: Training Ready
         ↓
Week 4-5: Experiments Ready
         ↓
Week 5-6: Results Ready
         ↓
DECISION: Continue to Phase 2?
```

### Each step depends on the previous:
- ❌ Can't train without features
- ❌ Can't validate without data
- ❌ Can't generalize without cross-dataset test
- ❌ Can't publish without comprehensive analysis

---

## Success Criteria Checklist

### Week 1: ✅ Features Complete
- [ ] TLS extractor working (12 features)
- [ ] DNS extractor working (8 features)
- [ ] Temporal extractor working (10 features)
- [ ] Integration layer working (40 total)
- [ ] All unit tests passing
- [ ] Model trains on 40 features

### Week 2: ✅ Model Validation
- [ ] 40-feature model converges
- [ ] Recall ≥96% on CTU-13 (baseline maintained)
- [ ] No degradation from 10-feature baseline
- [ ] Feature dimensions correct

### Week 3: ✅ Data Ready
- [ ] CTU-13 prepared (50K sessions)
- [ ] MAWI prepared (100K benign sessions)
- [ ] UGR'16 prepared (250K sessions)
- [ ] All datasets validated (no NaN/inf)
- [ ] All NPZ files created and checksummed
- [ ] DATASET_MANIFEST.json documented

### Week 4: ✅ Generalization Proven
- [ ] CTU-13 → UGR'16: Recall ≥70%
- [ ] CTU-13 → MAWI: FPR ≤5%
- [ ] UGR'16 → CTU-13: Recall ≥70%
- [ ] Cross-dataset results documented

### Week 5-6: ✅ Analysis Complete
- [ ] Feature importance determined
- [ ] Baseline comparisons done
- [ ] Comprehensive report written
- [ ] All results reproducible
- [ ] Code quality ≥90% coverage

---

## If Things Go Wrong

### Scenario 1: Cross-dataset recall <50%
**Diagnosis**: Model memorized CTU-13 artifacts  
**Solution**: 
- Identify which features caused memorization
- Remove high-variance features
- Retrain with reduced feature set
- Iterate until recall ≥70%

### Scenario 2: FPR >10% on MAWI
**Diagnosis**: Model too aggressive on benign traffic  
**Solution**:
- Increase false positive threshold
- Add more benign samples to training
- Regularize model
- Validate on real enterprise traffic

### Scenario 3: Dataset download fails
**Solution**:
- Try backup sources
- Use alternative dataset (CICIDS2017)
- Download individual flows instead of full captures
- Ask researcher community

### Scenario 4: Feature extraction crashes
**Solution**:
- Debug on subset of data (1 session)
- Add error handling
- Handle edge cases (zero-length sessions, etc.)
- Fall back to simpler features

---

## Metrics Dashboard (Week 6 Report)

After Phase 1, you'll have:

```
FEATURE PERFORMANCE
├─ 40-feature model recall on CTU-13: 96.3%
├─ Individual feature importance
│  ├─ TLS features: +2.1% average
│  ├─ DNS features: +1.8% average
│  └─ Temporal features: +3.2% average
└─ Feature extraction speed: N ms/session

CROSS-DATASET GENERALIZATION
├─ CTU-13 → UGR'16: 72% recall
├─ CTU-13 → MAWI: 98% specificity
├─ UGR'16 → CTU-13: 68% recall
└─ Average: 70% (PASSES threshold ✅)

BASELINE COMPARISONS
├─ Transformer AUC: 0.980
├─ XGBoost AUC: 0.956 (Transformer +2.4%)
├─ Random Forest AUC: 0.948 (Transformer +3.2%)
└─ LSTM AUC: 0.965 (Transformer +1.5%)

CONCLUSION
"Model learns generalizable C2 behavioral patterns.
 96.36% recall on CTU-13 is REAL, not memorization.
 Ready for Phase 2 (host-aware architecture)."
```

---

## Go/No-Go Decision

### GO to Phase 2 if:
- ✅ Cross-dataset recall ≥70%
- ✅ FPR ≤5% on benign
- ✅ Baseline comparisons show Transformer wins
- ✅ All results reproducible

### NO-GO to Phase 2 if:
- ❌ Cross-dataset recall <50% (memorization)
- ❌ FPR >10% (too many false alarms)
- ❌ Results not reproducible (data/seed issues)

### ITERATE if:
- ⚠️ Results in 50-70% range (some progress)
- ⚠️ Particular dataset causes issues (outlier investigation)

---

## Phase 1 Is Complete When...

✅ All 6 weeks done  
✅ All experiments run  
✅ All results analyzed  
✅ All code committed to git  
✅ All results published/documented  
✅ GO/NO-GO decision made  

**If GO**: Start Phase 2 (host-aware architecture)  
**If NO-GO**: Redesign Phase 1, iterate  
**If ITERATE**: Do additional experiments  

---

## Timeline Summary

```
Week 1-2 (May 13-24)    Feature Engineering
Week 3 (May 27-31)      Data Acquisition & Preparation  
Week 4 (June 3-7)       Cross-Dataset Experiments
Week 5-6 (June 10-24)   Analysis & Publication

Total: 6 weeks
Result: Answer "Is 96.36% recall real or memorization?"
Impact: Decide if Phase 2-3 investment justified
```

---

## Documents to Read (In Order)

1. **START**: [IMPLEMENTATION_KICKOFF.md](IMPLEMENTATION_KICKOFF.md) — Overview
2. **DATA**: [DATA_FIRST_STRATEGY.md](DATA_FIRST_STRATEGY.md) — Why data matters
3. **DATA PLAN**: [PHASE_1_DATA_ACQUISITION.md](PHASE_1_DATA_ACQUISITION.md) — How to get data
4. **CODE**: [PHASE_1_IMPLEMENTATION_GUIDE.md](PHASE_1_IMPLEMENTATION_GUIDE.md) — How to write code
5. **CHECKLIST**: [PHASE_1_CHECKLIST.md](PHASE_1_CHECKLIST.md) — Daily tracking
6. **THIS**: [THIS FILE] — Integration view

---

## You're Ready

You have:
- ✅ Complete feature engineering guide (copy-paste code)
- ✅ Complete data acquisition guide (download commands)
- ✅ Complete experimental plan (train/test procedures)
- ✅ Complete success criteria (measurable metrics)
- ✅ Complete backup plans (if things go wrong)

**Everything needed to complete Phase 1 successfully.**

Let's build this. 🚀

