# Phase 1 Week 3: Data Gate Analysis & Action Plan

**Current Status**: Phase 1 Weeks 1-2 COMPLETE (Extended Schema Implemented)  
**Blocker**: Data acquisition path unclear due to format mismatch  
**Timeline Impact**: 3-5 day delay if not resolved quickly

---

## 📊 Current Data State

### Existing CTU-13 Data
| File | Schema | Features | Status |
|------|--------|----------|--------|
| `ctu13_c2_sessions.npz` | base_v1 | 12 | Legacy baseline |
| `ctu13_scenario1_neris.npz` | experiment_v1 | 10 | Experiment variant |
| `data/processed/v2_mixed_session_only/train_sessions.npz` | base_v1 | 12 | 8,041 training sessions |

### What We Need
```
Extended Phase 1 Dataset:
├── CTU-13 extended_v1     (45 features, verified format)
├── MAWI benign            (45 features, real ISP traffic)
└── UGR'16 cross-test      (45 features, unseen families)
```

### The Gap
- **Extended extractors**: ✅ Implemented (TLS, DNS, Temporal)
- **Build pipeline**: ✅ Ready (build_extended_dataset.py)
- **Raw data format**: ❌ Binetflow only (no Zeek logs)
- **Zeek converter**: ❌ Missing (would convert binetflow → conn.log)

---

## 🚧 The Data Gate

```
Current:  binetflow (compact Zeek flow format)
Required: Zeek logs (conn.log, ssl.log, dns.log)
Problem:  No converter available in codebase
```

### Existing Raw Data
```
data/raw/
├── ctu13/
│   └── capture20110810.binetflow        (368 MB)
├── ctu13_extra/
│   ├── capture20110811.binetflow        (235 MB)
│   ├── capture20110815.binetflow        (146 MB)
│   └── CTU-13-Dataset.tar.bz2           (1.9 GB - unexplored)
└── mcfp/
    ├── CTU-Malware-Capture-Botnet-264-1.binetflow
    ├── CTU-Malware-Capture-Botnet-265-1.binetflow
    └── ... (8 more MCFP binetflow files)
```

---

## 🎯 Recommended Path Forward

### **OPTION A: Fast-Track with New Datasets** (RECOMMENDED)
**Time**: 24-48 hours  
**Risk**: Low

1. **Immediately start MAWI download** (can run overnight)
   ```bash
   # These are standard gzip'd pcap files from a reputable source
   # No format conversion needed - can process directly
   ```

2. **Download UGR'16** (parallel)
   ```bash
   # Spanish research dataset, will have better documentation
   ```

3. **Parallel: Create binetflow loader** (optional backup)
   ```python
   # Parse existing binetflow files if MAWI/UGR'16 fail
   # Binetflow format is parseable but undocumented
   ```

4. **Validate on MAWI first** (proof of concept)
   - Process MAWI → extract 45 features
   - Validate output NPZ matches schema
   - Confirm training pipeline works

5. **Then process UGR'16** (cross-validation)
   - If MAWI worked, UGR'16 will work
   - Fast-track CTU-13 rebuild after validating new datasets

**Advantages**:
- MAWI & UGR'16 have official documentation
- No custom converters needed
- Can skip binetflow issues entirely
- Get fresh, clean data for cross-dataset validation

**Disadvantages**:
- Loses CTU-13 momentum
- Must download ~50-100GB
- Adds 2-3 days for download + processing

---

### **OPTION B: Rebuild with Binetflow Loader** 
**Time**: 4-6 hours  
**Risk**: Medium (format uncertainty)

Create a custom binetflow parser:
```python
# Option B1: Use zeek-cut to convert binetflow → conn.log
zeek-cut -r capture.binetflow > conn.log

# Option B2: Parse binetflow directly
def load_binetflow(path):
    """Parse compact binetflow format"""
    # binetflow format: semicolon-delimited,
    # each line = one flow
    # Can deduce schema from Zeek documentation
```

**Advantages**:
- Uses existing CTU-13 data
- Faster than downloading new data
- Keeps current timeline

**Disadvantages**:
- Undocumented format (need reverse engineering)
- Might lose TLS/DNS context (binetflow is conn-only)
- No validation that data is correct
- Still need MAWI/UGR'16 eventually anyway

---

### **OPTION C: Validate Training Pipeline with Base Features** 
**Time**: 2-3 hours  
**Risk**: Very Low (zero new data)

**Approach**: 
1. Use existing base_v1 NPZ files (we already have these)
2. Train Transformer on base_v1 format
3. Measure baseline performance
4. Then overlay extended features once data is ready

**Code change needed**:
```python
# train_transformer.py is already schema-aware
# Just run training on existing v2_mixed_session_only/train_sessions.npz
python -m src.training.train_transformer \
  --train-data data/processed/v2_mixed_session_only/train_sessions.npz \
  --epochs 10 \
  --output experiments/phase1_baseline/
```

**Result**: Validate entire pipeline works before week 3 data arrives

**Advantages**:
- Zero risk (all code already works)
- Validates training pipeline immediately
- Can run in parallel with data acquisition
- Gives us baseline metrics

**Disadvantages**:
- Doesn't extract new features yet
- Still limited to 12 features
- Doesn't answer generalization question

---

## 🏗️ Recommended Execution Plan

### **Immediate (Next 2 hours)**

1. **Validate Training Pipeline** (OPTION C - parallelize)
   ```bash
   cd d:\CyberShield
   & .venv\Scripts\python.exe -m src.training.train_transformer \
     --train-data data/processed/v2_mixed_session_only/train_sessions.npz \
     --epochs 2 \
     --output experiments/phase1_baseline/
   ```
   **Goal**: Ensure schema-aware pipeline works end-to-end

2. **Start MAWI Download** (OPTION A - overnight)
   ```bash
   # Prepare MAWI download script
   # Can run on background job or separate terminal
   ```

### **Today Evening**

3. **Monitor downloads**
   - Check MAWI progress
   - Start UGR'16 if bandwidth permits

### **Tomorrow (Day 2)**

4. **Process First Dataset**
   - If MAWI arrived: process it to extended NPZ
   - Validate schema and data quality
   - Confirm training works with 45 features

5. **Decide on CTU-13**
   - If MAWI/UGR'16 validation succeeds: rebuild CTU-13 with binetflow loader
   - If issues arise: skip CTU-13 for now, focus on two new datasets

### **Week 3 Completion**

- [ ] 2-3 datasets in extended_v1 format
- [ ] All validated (no NaN/inf, schema correct)
- [ ] DATASET_MANIFEST.json created
- [ ] Ready for Week 4 cross-dataset validation

---

## 📋 Decision Matrix

| Option | Effort | Risk | Timeline | Includes CTU-13 |
|--------|--------|------|----------|-----------------|
| **A: New Datasets** | 20 hrs | Low | +5 days | ❌ Later |
| **B: Binetflow Loader** | 4 hrs | Medium | +1 day | ✅ Yes |
| **C: Base Features** | 2 hrs | None | Now | ❌ No |
| **A+B Combined** | 24 hrs | Low | +5 days | ✅ Yes |

---

## ✅ Immediate Action

**RECOMMENDED**: Execute **C (NOW) + A (overnight)** in parallel

```bash
# Terminal 1: Validate pipeline with existing data (runs in 10 min)
cd d:\CyberShield
& .venv\Scripts\python.exe -m src.training.train_transformer \
  --train-data data/processed/v2_mixed_session_only/train_sessions.npz \
  --epochs 2

# Terminal 2: Start MAWI download (runs overnight)
# [Detailed download script in next section]
```

**Goal**: 
- By tomorrow morning: Pipeline validated + MAWI data arriving
- By end of week: 2-3 datasets ready in extended_v1 format
- Week 4: Cross-dataset validation proves generalization (or doesn't)

---

## 🔄 Contingency

If MAWI download fails:
→ Fall back to OPTION B (binetflow loader) for CTU-13  
→ Download UGR'16 from academic source (more stable)  
→ Process both in extended_v1 format  

If all downloads fail:
→ Use OPTION C + run synthetic data validation  
→ Extend timeline by 1 week  
→ Re-prioritize by Friday end-of-day

---

**Next Message**: Detailed MAWI download and processing commands
