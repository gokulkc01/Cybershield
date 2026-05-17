# Data-First Strategy: Why This Matters Before Any Code

**Date**: May 13, 2026  
**Status**: CRITICAL - Read before starting Phase 1 implementation

---

## The Problem We're Solving

You just asked the right question: **"Everything we build is limited by whatever data we feed it."**

### Current Situation
```
Model Performance on CTU-13: 96.36% recall

But is this:
  ✅ REAL (learned true C2 behavioral patterns), or
  ❌ MEMORIZATION (learned dataset-specific artifacts)?
```

We don't know yet. This is the fundamental question Phase 1 answers.

---

## The Data-First Principle

### Traditional ML Workflow (Wrong)
```
1. Have data
2. Build model
3. Evaluate
4. Ship (hope it works in production)
```

**Problem**: If data is wrong, everything downstream is wrong.

### Data-First Workflow (Right)
```
1. Validate data quality
2. Validate data representativeness
3. THEN build models
4. Validate cross-dataset generalization
5. ONLY THEN ship
```

**Benefit**: Know exactly what your model learned (patterns vs. artifacts)

---

## The CyberShield Data Challenge

### What We Know ✅
- CTU-13: High quality C2 labels, 96.36% recall
- MAWI: Real benign ISP traffic (no malware)
- UGR'16: Different malware families, academic standard

### What We DON'T Know ❌
- Is 96.36% real generalization or memorization?
- How does model perform on other families?
- What's the FPR on real production traffic?
- Can we maintain performance if C2 evolves?

### How We Answer These Questions
1. **Extract 40 new features** (TLS, DNS, temporal) — Week 1-2
2. **Get 3 independent datasets** (CTU-13, MAWI, UGR'16) — Week 3
3. **Cross-dataset validation** (train on one, test on other) — Week 4
4. **Measure generalization** (recall ≥70% cross-dataset = real learning)

---

## The Data Roadmap

### Phase 1 (Right Now - May 2026)

**Goal**: Answer "Is our 96.36% recall real?"

**Data Strategy**:
1. CTU-13 (primary - have)
2. MAWI (benign baseline - download)
3. UGR'16 (cross-validation - download)

**Success Metric**:
- Cross-dataset recall ≥70% (if <50%, we were memorizing)
- FPR ≤5% on benign (production-ready)

**Timeline**: Week 3-4

### Phase 2 (Q3 2026)

**Goal**: Add host-level reasoning

**Data Strategy**:
- Same 3 datasets
- NEW requirement: Host-level labels (which sessions = same host)
- Need to reconstruct host timelines

**Implication**: Data preparation more complex (multi-session linking)

### Phase 3 (Q4 2026)

**Goal**: Foundation model with pretraining

**Data Strategy**:
- CTU-13 + UGR'16 (labeled, fine-tuning)
- MAWI full archive (unlabeled, pretraining)
- Need millions of unlabeled sessions

**Implication**: Storage and compute requirements increase 10x

### Phase 4 (Q1 2027)

**Goal**: Production deployment

**Data Strategy**:
- Live network traffic (real deployment)
- Ground truth from other detection methods
- Automated feedback loop

**Implication**: Data acquisition is continuous, not one-time

### Phase 5 (2027+)

**Goal**: Transfer to other detection tasks

**Data Strategy**:
- Domain-specific datasets (DGA, botnet, exfil)
- Foundation model as starting point
- Task-specific fine-tuning

---

## Your Week 3 Data Acquisition Tasks

### Task 1: Verify CTU-13 Exists ✅
**Status**: Probably already done
```bash
ls /data/raw/CTU-13/*/zeek_logs/
# Should see: conn.log, dns.log, ssl.log
```

**If missing**: Convert pcaps to Zeek logs (2-3 hours)

### Task 2: Download MAWI (New) ⚠️
**Timeline**: 2-3 hours
```bash
# Real ISP traffic (pure benign)
mkdir -p /data/raw/mawi/
cd /data/raw/mawi/

# Download 5-10 days (500MB-1GB total)
wget http://mawi.wide.ad.jp/mawi/samplepoint-F/2026-05-10/*.gz
wget http://mawi.wide.ad.jp/mawi/samplepoint-F/2026-05-09/*.gz
# ... etc
```

**Why**: Need real benign baseline to estimate FPR on production traffic

### Task 3: Download UGR'16 (New) ⚠️
**Timeline**: 1-2 hours
```bash
# Independent dataset from different source (Spain)
mkdir -p /data/raw/ugr16/
cd /data/raw/ugr16/

# Download from https://nesg.ugr.es/nesg/
wget <direct-link-to-ugr16>
```

**Why**: Different malware families prove generalization to unseen threats

### Task 4: Extract 40 Features ✅
**Timeline**: 2-3 hours
```bash
# Use code from Week 1 feature extractors
python src/data_loader/feature_transforms.py \
  --dataset ctu13 \
  --output data/processed/ctu13_sessions_40feat.npz

python src/data_loader/feature_transforms.py \
  --dataset mawi \
  --output data/processed/mawi_sessions_40feat.npz

python src/data_loader/feature_transforms.py \
  --dataset ugr16 \
  --output data/processed/ugr16_sessions_40feat.npz
```

### Task 5: Validate All Datasets ✅
**Timeline**: 1 hour
```bash
# Check for NaN, inf, reasonable ranges
python src/validation/validate_dataset.py \
  data/processed/ctu13_sessions_40feat.npz

python src/validation/validate_dataset.py \
  data/processed/mawi_sessions_40feat.npz

python src/validation/validate_dataset.py \
  data/processed/ugr16_sessions_40feat.npz
```

**Expected output**:
```
ctu13_sessions_40feat.npz:
  Shape: (50000, 40)
  Features valid: ✅
  Labels: benign=35000, C2=15000
  C2 ratio: 30%

mawi_sessions_40feat.npz:
  Shape: (100000, 40)
  Features valid: ✅
  Labels: benign=100000, C2=0
  C2 ratio: 0%

ugr16_sessions_40feat.npz:
  Shape: (250000, 40)
  Features valid: ✅
  Labels: benign=200000, C2=50000
  C2 ratio: 20%
```

---

## Critical Success Factor

### Before you train ANYTHING:

1. ✅ Do you have 3 independent datasets?
2. ✅ Have you validated them (no NaN, reasonable ranges)?
3. ✅ Do you understand what each dataset is (source, labels, size)?
4. ✅ Can you explain why cross-dataset validation proves real learning?

**If you can't answer these, don't proceed to training yet.**

---

## Documentation to Read

1. **START HERE**: [IMPLEMENTATION_KICKOFF.md](IMPLEMENTATION_KICKOFF.md)
   - Overview of entire 6-month plan
   - Decisions you're making
   - Success criteria

2. **DATA STRATEGY**: [DATA_STRATEGY.md](DATA_STRATEGY.md)
   - Complete data landscape
   - Why each dataset matters
   - How to validate data quality
   - Data governance

3. **DATA ACQUISITION**: [PHASE_1_DATA_ACQUISITION.md](PHASE_1_DATA_ACQUISITION.md)
   - Week-by-week task schedule
   - Download links and commands
   - Troubleshooting procedures
   - What to do if datasets unavailable

4. **IMPLEMENTATION GUIDE**: [PHASE_1_IMPLEMENTATION_GUIDE.md](PHASE_1_IMPLEMENTATION_GUIDE.md)
   - Feature extractor code (Week 1)
   - How to use it (Week 2-3)
   - Integration points (Week 4+)

5. **CHECKLIST**: [PHASE_1_CHECKLIST.md](PHASE_1_CHECKLIST.md)
   - Print this
   - Check off daily
   - Track progress

---

## The Sequential Path

You have **two parallel workstreams**:

### Workstream A: Feature Engineering (Week 1-2)
```
Monday-Tuesday:    TLS feature extractor
Wednesday-Friday:  DNS + Temporal extractors
Next week:         Integration + model training
```

**Owner**: ML Engineer  
**Dependency**: None (can start immediately)

### Workstream B: Data Acquisition (Week 3)
```
Monday-Tuesday:    Download datasets (background process)
Wednesday:         Process to Zeek format
Thursday-Friday:   Extract features + validate
```

**Owner**: Data Engineer  
**Dependency**: Zeek installed, disk space available (~50GB)

### Convergence Point: Week 4
```
Train on CTU-13, test on UGR'16, MAWI
Measure: Does recall stay ≥70%?
```

**Owner**: ML Engineer  
**Dependency**: Both workstreams complete

---

## Risk Mitigation

### Risk 1: MAWI Download Fails
**Mitigation**: Have 2-3 backup sources
- Primary: MAWI official (http://mawi.wide.ad.jp/)
- Backup 1: Google Drive (if shared by researcher)
- Backup 2: Use CICIDS2017 benign subset

### Risk 2: Datasets Don't Align
**Mitigation**: Use only compatible datasets
- Best case: All 3 work (CTU-13, MAWI, UGR'16)
- Fallback: Use 2 (CTU-13 + one other)
- Worst case: Use 1 (CTU-13 internal validation)

### Risk 3: Feature Extraction Fails
**Mitigation**: 
- Incremental testing (extract 1 session first)
- Error handling in code
- Fallback to simpler features if needed

### Risk 4: Validation Fails (NaN/inf)
**Mitigation**:
- Debug feature extractors
- Handle edge cases (zero-length sessions, etc.)
- Drop problematic sessions if necessary

---

## Decision Framework

### When you encounter a problem, ask:

**Q1: Is this a data problem or a code problem?**
- Data: Check dataset source, format, validation
- Code: Debug feature extractors

**Q2: Can I workaround it or do I need to fix it?**
- Workaround: Drop problematic samples, use simpler features
- Fix: Debug and resolve root cause

**Q3: Does this block Phase 1 or just delay it?**
- Blocks: Escalate, get help
- Delays: Document, note for Phase 2

---

## The Bottom Line

### Our 96.36% Recall on CTU-13 is Meaningless Until...

...we prove it generalizes to:
1. ✅ Different malware families (UGR'16 has different families)
2. ✅ Different network environments (MAWI is real ISP traffic)
3. ✅ Different feature scales (if CTU-13 ports are 1-65535 but UGR'16 ports are 0-1, model needs to handle it)

### Cross-Dataset Validation is the Test

```
If recall ≥ 70% on UGR'16:  → We learned real patterns ✅
If recall < 50% on UGR'16:  → We memorized artifacts ❌
If recall 50-70% on UGR'16: → We learned some patterns, some artifacts (iterate)
```

**This is how we know if Phase 2-3 investment is justified.**

---

## Action: This Week

### By EOW (May 24):

- [ ] Read all 4 documents above
- [ ] Verify CTU-13 data exists
- [ ] Start MAWI download (let it run overnight)
- [ ] Download UGR'16
- [ ] Extract 40 features from all datasets
- [ ] Validate (no NaN/inf, reasonable ranges)
- [ ] Document dataset sizes in DATASET_MANIFEST.json
- [ ] Commit to git: "Phase 1 Week 3: Datasets prepared"

### Week 4: Start Cross-Dataset Training
- Train on CTU-13, test on UGR'16
- Measure recall, FPR, AUC
- Document results

### The Question We're Answering
**"Is our 96.36% recall real or memorization?"**

If cross-dataset recall ≥70%, the answer is **real** ✅  
If cross-dataset recall <50%, the answer is **memorization** ❌

---

## Next Steps (After You Read This)

1. **Open [PHASE_1_DATA_ACQUISITION.md](PHASE_1_DATA_ACQUISITION.md)**
   - Detailed week-by-week schedule
   - Download commands
   - Validation procedures

2. **Open [DATA_STRATEGY.md](DATA_STRATEGY.md)**
   - Understanding data landscape
   - Why each dataset matters
   - Long-term data strategy

3. **Start Week 1 feature engineering** (parallel with data acquisition)
   - [PHASE_1_IMPLEMENTATION_GUIDE.md](PHASE_1_IMPLEMENTATION_GUIDE.md) Week 1 section

4. **Plan Week 3 data acquisition**
   - Identify which datasets to download first
   - Estimate storage needs
   - Setup download locations

---

## Final Thought

**Most ML projects fail because of data, not models.**

Models are easy. Data is hard.

You asked the right question today. Let's get the data right first, then everything else is straightforward.

We've got this. 🎯

