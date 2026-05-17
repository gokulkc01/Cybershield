# Phase 1 Week 3: Data Acquisition Action Plan

**Timeline**: May 20-24, 2026 (parallel with feature extractors from Week 1-2)

---

## Critical Fact

**All ML is limited by training data. You can have the best model architecture in the world, but if your training data is garbage, your model will be garbage.**

Our current situation:
- ✅ We have CTU-13 (proven, clean, 96.36% recall)
- ❌ We don't know if this is real generalization or memorization
- ❌ Cross-dataset testing will reveal the truth

This week: **Get the data we need to answer the generalization question.**

---

## Three Datasets Required for Phase 1

### Dataset 1: CTU-13 (Primary - Already Have ✅)

**Status**: Already in `/data/raw/CTU-13/`

**What we need**:
- [ ] Verify Zeek logs exist (conn.log, ssl.log, dns.log)
- [ ] Count total sessions
- [ ] Verify labels (which sessions are C2?)
- [ ] Extract to 40-feature NPZ format

**Command to check**:
```bash
ls -lh /data/raw/CTU-13/*/zeek_logs/
# Should see: conn.log, dns.log, ssl.log, http.log, etc.

# Count sessions
wc -l /data/raw/CTU-13/*/zeek_logs/conn.log
```

**Action**: If Zeek logs don't exist, we need to convert pcaps to Zeek:
```bash
# Assumes pcaps are in /data/raw/CTU-13/*/
for pcap in /data/raw/CTU-13/*/capture*.pcap; do
  DIR=$(dirname "$pcap")
  zeek -r "$pcap" -l "$DIR/zeek_logs" local
done
```

**Expected outcome**:
- 50K-100K sessions
- ~30% labeled as C2, ~70% benign

---

### Dataset 2: MAWI (Benign Baseline - NEW)

**What**: Real ISP traffic from Japan (pure benign, no malware)

**Why we need it**: 
- Our model must have ≥95% specificity on real benign traffic
- Academic datasets can have false labels
- Real ISP traffic = ground truth for benign

**Where to get**:
```bash
# Download 5-10 recent days of traffic
# Each day ~100-200MB

mkdir -p /data/raw/mawi/
cd /data/raw/mawi/

# Download command (replace dates as needed)
wget -r -l 1 -nd "http://mawi.wide.ad.jp/mawi/samplepoint-F/2026-05-10/"
wget -r -l 1 -nd "http://mawi.wide.ad.jp/mawi/samplepoint-F/2026-05-09/"
wget -r -l 1 -nd "http://mawi.wide.ad.jp/mawi/samplepoint-F/2026-05-08/"
wget -r -l 1 -nd "http://mawi.wide.ad.jp/mawi/samplepoint-F/2026-05-07/"
wget -r -l 1 -nd "http://mawi.wide.ad.jp/mawi/samplepoint-F/2026-05-06/"

# You should have 5-10 .gz files, each 100-200MB
ls -lh *.gz
```

**Processing steps**:
```bash
# 1. Decompress
gunzip *.gz

# 2. Convert pcaps to Zeek logs
mkdir zeek_logs/
for pcap in *.pcap; do
  zeek -r "$pcap" -l zeek_logs/ local
done

# 3. Merge Zeek logs
cat zeek_logs/conn.log.* > zeek_logs/conn.log.all
cat zeek_logs/dns.log.* > zeek_logs/dns.log.all
cat zeek_logs/ssl.log.* > zeek_logs/ssl.log.all
```

**Expected outcome**:
- 50K-500K sessions (depends on traffic volume)
- 100% benign (no C2 expected)
- Real-world traffic patterns (important for FPR estimation)

**Timeline**: 
- Download: 1-2 hours (depends on internet speed)
- Processing: 2-4 hours (Zeek processing takes time)
- Feature extraction: 1-2 hours

---

### Dataset 3: UGR'16 (Cross-Validation - NEW)

**What**: Spanish research dataset with labeled botnet traffic

**Why we need it**: 
- Independent dataset from different source (Spain vs Czech)
- Botnet families different from CTU-13
- Proves generalization to unseen families

**Where to get**:
```bash
mkdir -p /data/raw/ugr16/
cd /data/raw/ugr16/

# Option A: Direct download from UC3M
# Visit: https://nesg.ugr.es/nesg/
# Look for: UGR'16 dataset link
# Download and extract

# Option B: If direct link available:
wget https://nesg.ugr.es/datasets/ugr16_full.tar.gz
tar -xzf ugr16_full.tar.gz

# You should have CSV files with netflow data
ls -lh *.csv
```

**Files to expect**:
- UGR16_Normalized.csv (or similar)
  - Columns: flow_id, src_ip, src_port, dst_ip, dst_port, protocol, duration, bytes_sent, bytes_received, etc.
  - Labels: attack/benign

**Processing steps**:
```bash
# Convert NetFlow CSV to session format
python src/pipelines/prepare_ugr16.py \
  --input UGR16_Normalized.csv \
  --output zeek_logs/sessions.json
```

**Expected outcome**:
- 200K-500K total flows/sessions
- ~20-30% labeled as botnet/attack, ~70-80% benign
- Flows with full 5-tuple (src_ip, src_port, dst_ip, dst_port, protocol)

**Timeline**:
- Download: 30 min (300MB file, depending on connection)
- Processing: 2-3 hours (CSV parsing + feature extraction)

---

## Week 3 Task Schedule

### Monday (May 20)
**Morning** (2 hours): Download datasets
- [ ] Start MAWI download (background process)
- [ ] Download UGR'16 (fast, ~300MB)
- [ ] Verify CTU-13 Zeek logs exist

**Afternoon** (2 hours): Verify data integrity
- [ ] Check file sizes reasonable
- [ ] Spot-check a few packets (zeek can read pcaps)
- [ ] Create `/data/processed/` directory structure

### Tuesday (May 21)
**Morning** (3 hours): Process MAWI (Zeek conversion)
- [ ] Decompress MAWI pcaps
- [ ] Run Zeek on all pcaps
- [ ] Merge Zeek logs
- [ ] Check output log files exist

**Afternoon** (2 hours): Process UGR'16 (NetFlow parsing)
- [ ] Parse UGR'16 CSV
- [ ] Convert to internal session format
- [ ] Verify session counts reasonable

### Wednesday (May 22)
**Morning** (3 hours): Extract 40 features
- [ ] Write feature extraction script (if not already done from Week 1)
- [ ] Extract features from CTU-13
- [ ] Extract features from MAWI
- [ ] Extract features from UGR'16

**Afternoon** (2 hours): Create NPZ files
- [ ] Convert feature arrays to NPZ format
- [ ] Add metadata (session IDs, labels, timestamps)
- [ ] Create DATASET_MANIFEST.json

### Thursday (May 23)
**Morning** (2 hours): Validate datasets
- [ ] Run validation script on all 3 datasets
- [ ] Check for NaN/inf values
- [ ] Verify label distributions
- [ ] Check feature ranges are reasonable

**Afternoon** (2 hours): Document and commit
- [ ] Create dataset README
- [ ] Calculate checksums (md5)
- [ ] Commit to git with message "Phase 1 Week 3: Datasets prepared"

### Friday (May 24)
**Morning** (2 hours): Ready for training
- [ ] Verify cross-dataset training is possible
- [ ] Ensure all 3 datasets load without errors
- [ ] Test train/test split logic

**Afternoon** (1 hour): Prepare for Week 4
- [ ] Review cross-dataset experiment plan
- [ ] Set up experiment tracking

---

## Expected File Structure After Week 3

```
/data/
├── raw/
│   ├── CTU-13/
│   │   ├── capture1.pcap (existing)
│   │   ├── zeek_logs/ (Zeek processed)
│   │   │   ├── conn.log
│   │   │   ├── dns.log
│   │   │   ├── ssl.log
│   │   │   └── ...
│   │   └── labels.json (C2 annotations)
│   │
│   ├── mawi/
│   │   ├── 2026-05-10.pcap (downloaded)
│   │   ├── 2026-05-09.pcap
│   │   ├── ...
│   │   └── zeek_logs/
│   │       ├── conn.log
│   │       ├── dns.log
│   │       └── ssl.log
│   │
│   └── ugr16/
│       ├── UGR16_Normalized.csv (downloaded)
│       └── converted/
│           ├── conn_equivalent.log
│           └── sessions.json
│
└── processed/
    ├── ctu13_sessions_40feat.npz        ✅ NEW
    ├── ctu13_sessions_40feat.npz.md5
    ├── mawi_benign_40feat.npz           ✅ NEW
    ├── mawi_benign_40feat.npz.md5
    ├── ugr16_sessions_40feat.npz        ✅ NEW
    ├── ugr16_sessions_40feat.npz.md5
    └── DATASET_MANIFEST.json
```

---

## Critical Data Quality Checks

### Check 1: No Missing Values
```python
import numpy as np

data = np.load('data/processed/ctu13_sessions_40feat.npz')
sessions = data['sessions']

# Should be True
assert not np.isnan(sessions).any(), "NaN found in features!"
assert not np.isinf(sessions).any(), "Inf found in features!"
```

### Check 2: Reasonable Ranges
```python
# All features should be in expected ranges
print(f"Min: {sessions.min()} | Max: {sessions.max()}")

# Duration (feature 0): should be 0-86400 seconds
assert sessions[:, 0].min() >= 0
assert sessions[:, 0].max() <= 86400 * 365

# Bytes (features 1-2): should be 0-many
assert sessions[:, 1].min() >= 0
assert sessions[:, 2].min() >= 0
```

### Check 3: Label Balance
```python
# Load labels
labels = data['labels']

# Should have mix of 0 (benign) and 1 (C2)
print(f"Benign: {(labels == 0).sum()}")
print(f"C2: {(labels == 1).sum()}")
print(f"C2 ratio: {(labels == 1).sum() / len(labels):.1%}")

# Sanity check: not 100% benign or 100% C2
assert 0.1 <= (labels == 1).sum() / len(labels) <= 0.9
```

### Check 4: No Duplicates
```python
# Sessions should be unique (or at least mostly unique)
unique_count = len(np.unique(sessions, axis=0))
total_count = len(sessions)

print(f"Unique: {unique_count} / {total_count}")
assert unique_count / total_count > 0.9  # >90% unique
```

---

## Data Troubleshooting

### Problem: MAWI download is slow
**Solution**: 
- Download fewer days (3-5 instead of 10)
- Download in parallel: `wget -P /data/raw/mawi/ <url1> & wget -P /data/raw/mawi/ <url2> & ...`
- Download overnight (set up batch job)

### Problem: Zeek not installed
**Solution**:
```bash
# Install Zeek on Ubuntu/Debian
sudo apt-get install zeek

# Or on Windows via WSL:
wsl sudo apt-get install zeek

# Or use Docker:
docker pull zeek/zeek:latest
```

### Problem: UGR'16 download link broken
**Solution**:
- Email UC3M asking for direct link
- Check archived versions on different mirrors
- Fallback: Use only CTU-13 + MAWI for Phase 1 (still valuable cross-validation)

### Problem: Feature extraction fails (NaN values)
**Solution**:
- Check Zeek logs for format issues
- Add error handling to feature extractors
- Fallback: Drop problematic sessions

### Problem: Datasets are huge (multi-GB)
**Solution**:
- Sample to subset (first 1M flows)
- Use parallel processing
- Increase storage capacity

---

## Success Criteria for Week 3

By end of Friday:

- [ ] **CTU-13**: Ready to use, 40 features extracted, ≥50K sessions
- [ ] **MAWI**: Downloaded, processed, 40 features extracted, ≥50K benign sessions
- [ ] **UGR'16**: Downloaded, processed, 40 features extracted, ≥50K sessions
- [ ] **All NPZ files**: Created, validated (no NaN/inf), checksummed
- [ ] **Metadata**: DATASET_MANIFEST.json created with all dataset info
- [ ] **Documentation**: Dataset README with source, processing steps, statistics
- [ ] **Ready for training**: Week 4 can immediately start cross-dataset experiments

---

## Key Questions Before Starting

**Q: Do we already have CTU-13 Zeek logs?**
- Check: `ls /data/raw/CTU-13/*/zeek_logs/conn.log`
- If exists: Great, move to feature extraction
- If not exists: Need to convert pcaps with Zeek

**Q: Is MAWI download available from UC3M?**
- Check: Browse http://mawi.wide.ad.jp/mawi/samplepoint-F/
- Should see directories for each date
- Download 5-10 recent directories

**Q: Do we have UGR'16 access?**
- Check: https://nesg.ugr.es/nesg/
- May need to request access (academic license)
- Fallback: Use CICIDS2017 (less ideal but available)

---

## Contingency Plans

**If MAWI unavailable**:
- Use CICIDS2017 instead (less ideal - synthetic, but available)
- Or use only CTU-13 (not ideal - only one dataset for cross-validation)

**If UGR'16 unavailable**:
- Use CICIDS2017 for cross-validation
- Or use only CTU-13 for internal validation

**If both unavailable**:
- Do internal cross-validation: CTU-13 train on malware family 1, test on family 2
- Not as strong as external dataset, but still informative

---

## After Week 3: Week 4 Plan

Once datasets ready, Week 4 does cross-dataset training:

```
Train: CTU-13  →  Test: UGR'16     (measure: recall ≥70%?)
Train: CTU-13  →  Test: MAWI       (measure: FPR ≤5%?)
Train: UGR'16  →  Test: CTU-13     (measure: recall ≥70%?)
Train: Mixed   →  Test: Each       (measure: stability)
```

This tells us: **Is our 96.36% recall real or memorization?**

---

## Remember

**Garbage in → Garbage out**

Our model is only as good as our data. 

Spend the time here. Get the data right. Then everything else is easier.

Let's go. 🎯

