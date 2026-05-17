# CyberShield Data Strategy: The Foundation

**Principle**: All ML performance is bounded by data quality, representativeness, and scale.

Our current 96.36% recall on CTU-13 could be:
- ✅ Real (model learned true C2 behavioral patterns), OR
- ❌ Memorization (model learned dataset artifacts: specific ports, IPs, time periods)

**Phase 1 goal**: Answer this question through cross-dataset validation.

---

## 1. Current Data Landscape

### What We Have ✅

**CTU-13** (Primary research dataset)
- **Source**: Stratosphere Lab (https://www.stratosphereips.org)
- **What**: Captured botnet traffic (2011 era)
- **Format**: Zeek logs (conn.log, ssl.log, dns.log, http.log, etc.)
- **Size**: ~40GB of pcaps → processed into sessions
- **Families**: Neris, Kraken, Conficker, Rbot, Nsis-ay, Virut, Emotet-like, etc.
- **Quality**: ✅ Excellent - manually labeled, verified, academic standard
- **Recency**: ❌ Old (2011) - C2 has evolved significantly
- **Our Status**: Already have and use this

**MAWI Archive** (Benign baseline)
- **Source**: Monami Topology Lab (http://mawi.wide.ad.jp/mawi/)
- **What**: Real ISP traffic (Japan, periodic captures)
- **Format**: Pcaps → need to convert to Zeek
- **Size**: 100TB+ archive, but samples are available
- **Families**: No malware (pure benign traffic)
- **Quality**: ✅ Real-world data (but no malware ground truth)
- **Recency**: ✅ Ongoing (latest captures available)
- **Use Case**: Benign baseline, negative class training
- **Our Status**: Need to prepare samples

### What We Can Get ⚠️

**UGR'16** (Spanish research dataset)
- **Source**: UC3M (https://nesg.ugr.es/nesg/)
- **What**: Network intrusion detection dataset
- **Format**: NetFlow data → need conversion to sessions
- **Size**: ~1 million flows
- **Families**: Various botnets, but mixed with general intrusions
- **Quality**: ⚠️ Academic, but botnet labels may be approximate
- **Recency**: ❌ 2016
- **Our Status**: Publicly available, need to download and process

**CICIDS2017** (Canadian research dataset)
- **Source**: Canadian Institute Cybersecurity (https://www.unb.ca/cic/datasets/)
- **What**: Intrusion detection dataset
- **Format**: Exported features + pcaps
- **Size**: ~2.8M flows
- **Families**: Many attack types including botnets
- **Quality**: ⚠️ Mostly lab-generated (attacks on isolated testbed)
- **Recency**: ❌ 2017
- **Limitation**: Synthetic attacks ≠ real C2
- **Our Status**: Available, but synthetic nature is problematic

**UNSW-NB15** (Australian research dataset)
- **Source**: UNSW Sydney
- **What**: Intrusion detection dataset
- **Format**: Exported features
- **Size**: ~2.5M flows
- **Limitation**: Pre-extracted features (can't re-engineer with our schema)
- **Our Status**: Avoid - features locked in

### What We DON'T Have ❌

**Modern Real C2 Traffic** (2022-2025)
- Modern C2 uses: TLS/HTTPS, DNS fast-flux, decentralized C&C
- Public datasets are old (2011-2017)
- Why it matters: C2 techniques evolve rapidly
- Solution: Phase 4 (production deployment will capture real modern C2)

**Production Network Baselines**
- Enterprise traffic patterns ≠ academic datasets
- Why it matters: FPR will increase in production
- Solution: Phase 4 (shadow deployment on real networks)

**Mobile/IoT Botnet Traffic**
- Most datasets focus on server/PC traffic
- Why it matters: Modern botnets target mobile, IoT
- Solution: Phase 5 (specialized models for different device types)

---

## 2. Dataset Acquisition Strategy

### Tier 1: Get Started NOW (Phase 1)

**CTU-13** ✅
- Already have: `/data/raw/CTU-13/`
- Status: Ready to use
- Action: Verify existing NPZ files are correct format
- Backup source: https://www.stratosphereips.org/datasets-ctupapers

**MAWI Samples** (New - start Week 3)
- Where to get: http://mawi.wide.ad.jp/mawi/
- How to download:
  ```bash
  # Download latest samples (e.g., 2026-05-10)
  wget http://mawi.wide.ad.jp/mawi/samplepoint-F/2026-05-10/
  
  # Download ~5-10 recent days (each ~100MB)
  # This gives 500MB-1GB of benign traffic
  ```
- Processing: Convert pcaps → Zeek logs → feature extraction
- Expected sessions: 100K+ benign sessions
- Timeline: 1-2 days to download + process

**UGR'16** (New - start Week 3)
- Where to get: https://nesg.ugr.es/nesg/ (direct links available)
- File size: ~500MB
- Processing: NetFlow → session format → feature extraction
- Expected botnet sessions: ~50K labeled C2
- Expected benign sessions: ~200K
- Timeline: 1 day to download + process

**CICIDS2017** (Optional - start Week 3, use only if UGR has issues)
- Where to get: https://www.unb.ca/cic/datasets/
- Files to download: `GeneratedLabelledFlows.csv` (~400MB)
- Limitation: Pre-extracted features, may not align with our schema
- Timeline: 1 day to evaluate feasibility
- Decision: Use only if it doesn't require re-engineering features

### Tier 2: Enhance After Phase 1

**Zeek Botnet Exercises**
- Where: https://www.stamus-networks.com/blog/open-source-iocs
- What: Curated botnet C2 communication samples
- Format: Pcaps → Zeek logs
- Use: Validate feature extractors on known C2 behavior

**Archive.org Security Datasets**
- Search: botnet, C2, malware traffic samples
- Evaluate: Quality and representativeness
- Use: Augment training data

### Tier 3: Production Data (Phase 4+)

**Live Network Capture** (Shadow deployment)
- Deploy inline on real network (Zeek or Suricata)
- Capture all traffic (or sampled)
- Labels: Come from other detection methods + human verification
- Use: Retrain model with production data

**Bug Bounty / Threat Feeds**
- Integrate threat intelligence feeds
- Get latest indicators of compromise (IOCs)
- Use: Ground truth for model improvement

---

## 3. Data Processing Pipeline

### Current State
```
CTU-13 pcaps
    ↓
Zeek logs (conn.log, ssl.log, dns.log, http.log)
    ↓
Feature extraction (10 features currently)
    ↓
NPZ format (sessions, labels, metadata)
    ↓
Training
```

### Phase 1 Target
```
Multiple sources (CTU-13, MAWI, UGR'16, CICIDS2017)
    ↓
Zeek logs (standardized format)
    ↓
Feature extraction (40 features)
    ↓
NPZ format (sessions, labels, metadata)
    ↓
Cross-dataset validation
```

### Implementation Steps (Week 3-4)

**Step 1: Create Dataset Processor**
```python
# src/pipelines/dataset_processor.py

class DatasetProcessor:
    """
    Unified pipeline for converting various data sources
    to standardized feature format
    """
    
    def process_zeek_logs(self, conn_log, ssl_log, dns_log):
        """
        Input: Zeek log files
        Output: Session dataset with 40 features
        """
        sessions = parse_conn_log(conn_log)  # sessions = list of dicts
        sessions = enrich_with_ssl(sessions, ssl_log)
        sessions = enrich_with_dns(sessions, dns_log)
        features = extract_40_features(sessions)  # 40-dim arrays
        return features, labels
    
    def process_netflow(self, netflow_file):
        """
        Input: NetFlow data (UGR'16)
        Output: Session dataset with 40 features
        """
        # Convert NetFlow → Zeek-equivalent format
        # Then run through extract_40_features
        pass
    
    def process_cicids(self, csv_file):
        """
        Input: CICIDS2017 CSV
        Output: Session dataset with 40 features
        """
        # Validate if CSV features can map to our 40 features
        # If not, reject and use only Zeek-based datasets
        pass

class DatasetValidator:
    """
    Validate dataset quality and representativeness
    """
    
    def check_feature_distributions(self, dataset):
        """Ensure features are reasonable, no NaN/inf"""
        pass
    
    def check_label_balance(self, dataset):
        """Check C2/benign ratio"""
        pass
    
    def check_temporal_coverage(self, dataset):
        """Ensure time periods cover diverse conditions"""
        pass
```

**Step 2: Create Dataset Configs**
```yaml
# data/configs/datasets.yaml

datasets:
  ctu13:
    name: "CTU-13 Botnet"
    source: /data/raw/CTU-13/
    format: zeek_logs
    zeek_logs:
      conn: conn.log
      ssl: ssl.log
      dns: dns.log
    output: data/processed/ctu13_sessions_40feat.npz
    expected_sessions: 50000
    expected_c2_ratio: 0.3
    
  mawi:
    name: "MAWI ISP Traffic"
    source: /data/raw/mawi/
    format: pcap
    pcap_source: http://mawi.wide.ad.jp/mawi/samplepoint-F/2026-*/
    output: data/processed/mawi_benign_40feat.npz
    expected_sessions: 100000
    expected_c2_ratio: 0.0  # benign only
    
  ugr16:
    name: "UGR'16 Network Intrusion"
    source: /data/raw/ugr16/
    format: netflow
    output: data/processed/ugr16_sessions_40feat.npz
    expected_sessions: 250000
    expected_c2_ratio: 0.2
    
  cicids2017:
    name: "CICIDS2017 (Synthetic)"
    source: /data/raw/cicids2017/
    format: csv
    output: data/processed/cicids2017_sessions_40feat.npz
    expected_sessions: 2800000
    expected_c2_ratio: 0.15
    use_only_if: ["ugr16 not available"]
```

**Step 3: Implement Zeek Log Parser**
```python
# src/data_loader/zeek_parser.py

def parse_zeek_logs(conn_log, ssl_log, dns_log):
    """
    Parse Zeek logs and create session objects
    """
    # Read conn.log - this defines sessions
    sessions = {}
    with open(conn_log) as f:
        for line in f:
            if line.startswith('#'): continue
            
            # Parse columns: ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p, ...
            src_ip, src_port, dst_ip, dst_port, proto, duration, ...
            
            # Create session key
            session_key = (src_ip, src_port, dst_ip, dst_port)
            
            # Add to sessions dict
            sessions[session_key] = {
                'timestamp': ts,
                'uid': uid,
                'duration': duration,
                'bytes_in': bytes_in,
                'bytes_out': bytes_out,
                ...
            }
    
    # Enrich with SSL features (from ssl.log)
    with open(ssl_log) as f:
        for line in f:
            if line.startswith('#'): continue
            
            # Extract TLS info
            uid, version, cipher, ...
            
            # Match to session via uid
            sessions[uid]['tls_version'] = version
            sessions[uid]['tls_cipher'] = cipher
            ...
    
    # Enrich with DNS features (from dns.log)
    with open(dns_log) as f:
        for line in f:
            if line.startswith('#'): continue
            
            # Extract DNS query info
            # For same src_ip host, aggregate
            
    return sessions

def extract_40_features_from_session(session_dict):
    """
    Convert session dict to 40-feature array
    """
    features = np.zeros(40)
    
    # Base 10 features
    features[0] = session_dict['duration']
    features[1] = session_dict['bytes_in']
    ...
    
    # TLS 12 features
    tls_extractor = TLSFeatureExtractor()
    tls_features = tls_extractor.extract_from_session(session_dict)
    features[10:22] = tls_features
    
    # DNS 8 features
    dns_extractor = DNSFeatureExtractor()
    dns_features = dns_extractor.extract_from_session(session_dict)
    features[22:30] = dns_features
    
    # Temporal 10 features
    temporal_extractor = TemporalStatsExtractor()
    temporal_features = temporal_extractor.extract_from_session(session_dict)
    features[30:40] = temporal_features
    
    return features
```

---

## 4. Dataset Quality Checklist

For each dataset, verify:

- [ ] **Completeness**: No missing values in critical fields
- [ ] **Correctness**: Labels are accurate (especially C2/benign split)
- [ ] **Consistency**: Format matches schema (same fields, types, ranges)
- [ ] **Coverage**: Time periods diverse (morning, evening, weekends)
- [ ] **Scale**: Sufficient samples (at least 10K per category)
- [ ] **Reproducibility**: Same version reproducible (use checksums)
- [ ] **Provenance**: Source documented and accessible
- [ ] **Ethics**: Legal to use (check licensing)

**Automated Validation Script:**
```python
# src/validation/validate_dataset.py

def validate_dataset(dataset_path):
    """
    Load dataset and run validation checks
    """
    data = np.load(dataset_path)
    sessions = data['sessions']
    labels = data['labels']
    
    print(f"Shape: {sessions.shape}")
    print(f"Labels: {np.bincount(labels)}")
    
    # Check for NaN/inf
    assert np.isfinite(sessions).all(), "NaN or inf found"
    
    # Check feature ranges
    assert sessions.min() >= -1000, "Features too small"
    assert sessions.max() <= 1e10, "Features too large"
    
    # Check label distribution
    ratio = labels.sum() / len(labels)
    print(f"C2 ratio: {ratio:.2%}")
    assert 0.1 <= ratio <= 0.5, "Unexpected C2 ratio"
    
    # Check for duplicate sessions
    unique_sessions = len(np.unique(sessions, axis=0))
    print(f"Unique sessions: {unique_sessions} / {len(sessions)}")
    
    return True
```

---

## 5. Dataset Versioning & Reproducibility

### Version Control
```
data/processed/
├── ctu13_sessions_40feat_v1.0.npz    (May 2026)
├── ctu13_sessions_40feat_v1.0.npz.md5 (checksum)
├── mawi_benign_40feat_v1.0.npz
├── ugr16_sessions_40feat_v1.0.npz
└── DATASET_MANIFEST.json
```

**DATASET_MANIFEST.json:**
```json
{
  "datasets": {
    "ctu13_v1.0": {
      "version": "1.0",
      "created": "2026-05-13",
      "source": "https://www.stratosphereips.org/",
      "processing_script": "src/pipelines/prepare_ctu13.py",
      "processing_date": "2026-05-13",
      "num_sessions": 50000,
      "num_features": 40,
      "num_c2": 15000,
      "num_benign": 35000,
      "feature_schema": "v1_extended",
      "zeek_version": "5.0",
      "md5": "abc123def456...",
      "notes": "Includes Neris, Kraken, Conficker families"
    },
    "mawi_v1.0": {
      "version": "1.0",
      "created": "2026-05-20",
      "source": "http://mawi.wide.ad.jp/",
      "capture_dates": ["2026-04-15", "2026-04-16", ...],
      "num_sessions": 100000,
      "num_features": 40,
      "num_c2": 0,
      "num_benign": 100000,
      ...
    }
  }
}
```

### Reproducibility Steps
1. Use fixed random seeds (seed=42 for train/test split)
2. Document Zeek version used (may affect log format)
3. Version feature extraction code (tag in git)
4. Store checksums for all datasets
5. Document train/test splits explicitly

---

## 6. Data Acquisition Timeline

### Week 3 (May 20-24, 2026)

**Monday-Tuesday**: Download and verify datasets
- [ ] Download MAWI samples (5GB)
- [ ] Download UGR'16 (500MB)
- [ ] Optionally download CICIDS2017 (400MB)
- [ ] Verify checksums
- [ ] Store in `/data/raw/`

**Wednesday-Thursday**: Convert to Zeek format (if needed)
- [ ] MAWI: Convert pcaps → Zeek logs
  ```bash
  # If not already Zeek logs:
  zeek -r mawi_sample.pcap -l logs local
  ```
- [ ] UGR'16: Convert NetFlow → session format
- [ ] CICIDS2017: Evaluate if usable

**Friday**: Initial feature extraction
- [ ] Extract 40 features from each dataset
- [ ] Create NPZ files
- [ ] Run validation script
- [ ] Document dataset sizes

### Week 4-6: Parallel processing
- Continue dataset preparation while training models
- Background task: Process large CICIDS2017 if needed
- Prepare remaining datasets for cross-dataset validation

---

## 7. Critical Data Decisions for Phase 1

### Decision 1: Which datasets to use?
**Answer: CTU-13 primary, augment with MAWI + UGR'16**
- CTU-13: Best quality C2 labels
- MAWI: Best quality benign baseline (real ISP traffic)
- UGR'16: Academic botnet dataset for cross-validation
- CICIDS2017: Only if UGR'16 not sufficient (synthetic nature is risky)

### Decision 2: How to handle label imbalance?
**Answer: Keep natural distribution in validation, stratified sampling in training**
```
Natural distribution: 20-30% C2, 70-80% benign
Training: Stratified sampling (train on natural distribution)
Validation: Test on natural distribution (realistic FPR)
```

### Decision 3: How to handle Zeek log format differences?
**Answer: Standardize to common schema**
- Different Zeek versions may have slightly different field names
- Create mapping layer: normalize all logs to common format
- Document version used (Zeek 5.0+)

### Decision 4: How to validate cross-dataset generalization?
**Answer: Train on one dataset, test on another (4-6 combinations)**
```
Train: CTU-13      Test: UGR'16        (recall: target ≥70%)
Train: CTU-13      Test: MAWI benign   (FPR: target ≤5%)
Train: UGR'16      Test: CTU-13        (recall: target ≥70%)
Train: UGR'16      Test: MAWI benign   (FPR: target ≤5%)
Train: CTU+UGR     Test: CICIDS (optional)
```

### Decision 5: How to handle concept drift over time?
**Answer: Phase 1 doesn't address it, Phase 4 will**
- Phase 1: Static datasets (all labeled upfront)
- Phase 4: Continual learning (handle drift with monthly retraining)

---

## 8. Data Quality vs. Quantity Trade-off

### High Quality, Smaller Scale
- **Example**: CTU-13 (50K sessions, excellent labels)
- **Advantage**: Can train reliable model on smaller data
- **Disadvantage**: May not capture full range of behaviors
- **Best for**: Phase 1 (understand what works)

### Lower Quality, Larger Scale
- **Example**: CICIDS2017 (2.8M flows, synthetic)
- **Advantage**: More diverse samples, handle edge cases
- **Disadvantage**: Synthetic ≠ real, may learn wrong patterns
- **Best for**: Phase 3 (foundation model pretraining on unlabeled data)

### Recommendation
**Phase 1**: High quality > quantity
- Start with CTU-13 (proven, clean labels)
- Add MAWI (real benign baseline)
- Add UGR'16 (cross-validation)
- Skip CICIDS2017 (synthetic risk)

**Phase 3**: Low quality OK for pretraining
- Use CICIDS2017, MAWI, other unlabeled data
- Fine-tune on high-quality labeled data (CTU-13, UGR'16)

**Phase 4**: Live data only
- Deploy and collect real production traffic
- This becomes the gold standard

---

## 9. Data Governance & Documentation

### What to Document
1. **Source**: Where did the data come from?
2. **Collection**: How was it collected? What period?
3. **Processing**: What transformations applied?
4. **Labels**: How were C2/benign labels assigned?
5. **Statistics**: Session counts, feature distributions, time periods
6. **Quality**: Any known issues or limitations?
7. **License**: Legal status, can we publish results?

### Template for Each Dataset
```markdown
# Dataset: CTU-13

## Source
- **URL**: https://www.stratosphereips.org/datasets-ctupapers
- **Citation**: Stratosphere Lab, Garcia et al.
- **License**: Creative Commons (check specific terms)

## Collection
- **Dates**: Sept-Dec 2011
- **Method**: Passive network capture (Zeek)
- **Network**: Dionaea honeypots

## Ground Truth
- **Method**: Manual labeling + community review
- **Families**: Neris, Kraken, Conficker, Rbot, Nsis-ay, Virut, etc.
- **Confidence**: High (academic standard)

## Statistics
- Sessions: 50K
- C2: 15K (30%)
- Benign: 35K (70%)
- Time span: 90 days
- Feature schema: Extended 40-feature

## Known Issues
- Old (2011 era) - C2 has evolved
- Limited to captured families
- No encrypted communication inspection (TLS content)

## Files
- ctu13_sessions_40feat_v1.0.npz
- ctu13_sessions_40feat_v1.0.npz.md5
```

---

## 10. Data Roadmap: Phases 1-5

### Phase 1 (Now)
- **Data**: CTU-13, MAWI, UGR'16
- **Purpose**: Validate feature engineering
- **Quality**: High (academic standard)
- **Scale**: ~100K-500K total sessions

### Phase 2 (After Phase 1)
- **Data**: Same as Phase 1 + CICIDS2017 (optional)
- **Purpose**: Validate host-aware architecture
- **New requirement**: Host-level labels (which sessions belong to same host)
- **Scale**: Same

### Phase 3 (After Phase 2)
- **Data**: CTU-13 + MAWI (now used for pretraining)
- **Purpose**: Foundation model with self-supervised pretraining
- **New requirement**: Large unlabeled dataset (MAWI full archive)
- **Scale**: 1M+ benign sessions for pretraining

### Phase 4 (Deployment)
- **Data**: Live production network traffic
- **Purpose**: Real-world validation
- **Ground truth**: Other detection methods + manual review
- **Scale**: Continuous, streaming

### Phase 5 (Generalization)
- **Data**: Domain-specific datasets (DGA, botnet, exfil)
- **Purpose**: Transfer learning to other tasks
- **Strategy**: Use Phase 3 foundation model as starting point
- **Scale**: Varies by task

---

## 11. FAQs: Data Questions

**Q: Can we use synthetic data?**
A: Not for Phase 1-4. Phase 3 can use unlabeled synthetic data for pretraining, but only after Phase 1 proves real data works.

**Q: What if UGR'16 download fails?**
A: Fallback: Use CICIDS2017 (more available online). It's synthetic but still useful for cross-dataset validation. Or use only CTU-13 + MAWI.

**Q: How do we know labels are correct?**
A: CTU-13: Academic gold standard (verified). MAWI: No labels (benign ISP traffic assumed clean). UGR'16: Academic labels (reasonable). If unsure, test only on CTU-13 in Phase 1.

**Q: Can we use IP reputation (like Shodan)?**
A: Not recommended. IPs change, are reused, not stable feature. Stick with behavioral features.

**Q: What about encrypted DNS (DoH) and encrypted SNI?**
A: Not in current datasets. Phase 4 (production) will encounter this. For Phase 1-3, assume standard DNS/TLS.

---

## 12. Next Steps: This Week

### Action Items (Week 3)

**Monday-Tuesday** (2 hours):
- [ ] Create `/data/raw/` structure
- [ ] Start MAWI download (run in background)
- [ ] Download UGR'16 (fast, ~500MB)

**Wednesday-Thursday** (4 hours):
- [ ] Create dataset processor class
- [ ] Write Zeek parser
- [ ] Test on CTU-13 existing data

**Friday** (2 hours):
- [ ] Extract 40 features from MAWI + UGR'16
- [ ] Validate datasets (no NaN/inf, correct shapes)
- [ ] Document dataset sizes and characteristics

**By end of Week 3**:
- [ ] 3 datasets ready (CTU-13 refreshed, MAWI new, UGR'16 new)
- [ ] All converted to 40-feature NPZ format
- [ ] Ready for cross-dataset training in Week 4

---

## Summary: Data is the Foundation

**Remember**: Our 96.36% recall means nothing if it's memorization.

**Phase 1 data strategy**:
1. ✅ Use proven datasets (CTU-13 high quality)
2. ✅ Add real benign baseline (MAWI ISP traffic)
3. ✅ Cross-validate (UGR'16 independent dataset)
4. ✅ Prove >70% cross-dataset recall (rule out memorization)
5. ✅ Document everything (reproducibility)

**Success metric**: If cross-dataset recall ≥70%, we're learning real patterns. If <50%, we're memorizing artifacts.

Let's get the data right first. Everything else depends on it. 🎯

