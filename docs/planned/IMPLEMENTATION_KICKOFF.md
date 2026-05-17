# CyberShield Evolution Plan: Complete Overview

## What We're Building

Transform CyberShield from a **session-centric behavioral classifier** into a **hierarchical behavioral reasoning system** for modern C2 detection.

**Key Principle**: Learn *behavioral invariants of malicious coordination*, not *dataset-specific artifacts*.

---

## Why This Matters

### Current State (✅ Complete)
- **Model**: Session-centric Transformer
- **Performance**: 96.36% recall on cross-family test (Neris+Kraken → Conficker)
- **Limitation**: Only sees one session at a time
- **Problem**: Misses multi-session coordination patterns, host-level persistence

### Future State (🚀 Vision)
- **Model**: Hierarchical Transformer with host memory + self-supervised pretraining
- **Performance Target**: ≥97% on cross-dataset, ≥94% under adversarial mutations
- **Capability**: Learns coordination patterns across sessions, hosts, and time
- **Impact**: Foundation model for all network anomaly detection tasks

---

## The 5-Phase Evolution

```
Phase 0 (DONE)        → 96.36% cross-family recall ✅
Phase 1 (6 weeks)     → Enhanced features + cross-dataset validation
Phase 2 (8 weeks)     → Host-aware architecture + temporal memory
Phase 3 (10 weeks)    → Hierarchical Transformer + self-supervised pretraining
Phase 4 (ongoing)     → Production deployment + continual learning
Phase 5 (2027+)       → Foundation models for network behavior
```

---

## Phase 1: Enhanced Features & Cross-Dataset Validation (Q2 2026, 6 weeks)

### What
Add **30 new features** from three critical domains:
1. **TLS Behavioral** (12 features): JA3, cipher diversity, cert reuse, extension ordering
2. **DNS Behavioral** (8 features): Query cadence, NXDOMAIN rate, DGA probability, domain reuse
3. **Temporal Statistics** (10 features): IAT entropy, burst morphology, periodicity, autocorrelation

**New feature set**: 40 total features (up from 10)

### Why
- Current 10 features missing encrypted traffic signals (TLS)
- Missing infrastructure reconnaissance patterns (DNS)
- Missing timing sophistication metrics (temporal stats)
- These are what modern C2 actually uses (everything is encrypted)

### Timeline

| Week | Task |
|------|------|
| Week 1 | Implement TLS, DNS, temporal extractors |
| Week 2 | Integrate features, update schemas, tests |
| Week 3 | Prepare UGR'16, CICIDS2017, MAWI datasets |
| Week 4 | Cross-dataset training (CTU→UGR, UGR→CTU, etc.) |
| Week 5-6 | Feature importance analysis, baseline comparisons |

### Success Criteria
- ✅ 40-feature model trains without degradation
- ✅ Cross-dataset recall ≥70% (CTU trained, UGR tested)
- ✅ TLS/DNS/temporal features show importance
- ✅ Transformer outperforms baseline models

### Immediate Next Steps (Starting NOW)

1. **Copy the feature extraction code** from `PHASE_1_IMPLEMENTATION_GUIDE.md`
2. **Create three new files**:
   - `src/features/tls_features.py`
   - `src/features/dns_features.py`
   - `src/features/temporal_features.py`
3. **Run tests**: Ensure all extractors work
4. **Integrate**: Update feature pipeline to use 40 features
5. **Test on CTU-13**: Confirm model trains with 40 features

---

## Phase 2: Host-Aware Architecture & Temporal Memory (Q3 2026, 8 weeks)

### What
Add **host-level context** to the model:
- Reconstruct host timelines (all sessions from one host, chronologically ordered)
- Extract multi-session features (persistence, continuity, coordination)
- Modify Transformer to use **host memory mechanism**

**New features**: 15-20 multi-session features  
**New model**: C2TransformerV2HostAware

### Why
- Single-session model is blind to coordination patterns
- Real C2 uses multiple sessions: callback sessions, data exfil, command staging
- Host timeline reveals: callback regularity, persistence, infection continuity

### Timeline
8 weeks (after Phase 1)

### Success Criteria
- ✅ HostTimeline class functional
- ✅ Host-aware model recall ≥97.5% on CTU-13
- ✅ Cross-dataset recall ≥72% (improvement over Phase 1)
- ✅ Inference latency still < 100ms with host context

---

## Phase 3: Hierarchical Transformer & Self-Supervised Pretraining (Q4 2026, 10 weeks)

### What
Build a **foundation model for network behavior**:
1. **Self-supervised pretraining** on massive unlabeled traffic (MAWI, UGR'16)
   - Masked flow prediction (like BERT)
   - Temporal consistency learning
   - Next-flow prediction
   
2. **Hierarchical architecture**: Packet → Flow → Session → Host
   - Each level learns its own encoder
   - Higher levels inherit representations from lower levels
   
3. **Fine-tune** on labeled C2 data
4. **Adversarial train** against Red-Agent mutations

### Why
- Supervised learning alone underutilizes data
- Pretraining on unlabeled traffic teaches "network grammar"
- Hierarchical structure captures natural causality
- Foundation model enables transfer to DGA, botnet, exfil detection

### Timeline
10 weeks (after Phase 2)

### Success Criteria
- ✅ Unsupervised pretraining runs to convergence
- ✅ Fine-tuned recall ≥97% on CTU-13
- ✅ Cross-dataset recall ≥75%
- ✅ Model maintains ≥94% recall under Red-Agent mutations
- ✅ Foundation encoder transfers to other tasks

---

## Phase 4: Production Deployment & Continual Adaptation (Q1-Q2 2027, ongoing)

### What
**Deploy to shadow mode** (no alerts, just logging):
1. Run inference on live network traffic
2. Collect ground truth labels
3. Monitor performance vs. offline baseline
4. Implement continual learning (monthly retraining)
5. Detect concept drift

### Why
- Lab performance (96%) ≠ production performance
- Real networks have different traffic patterns
- New malware families evolve continuously
- Adaptive systems stay relevant

### Timeline
Ongoing (starts after Phase 3)

### Success Criteria
- ✅ Live recall ≥75%
- ✅ Live FPR ≤5%
- ✅ System uptime ≥99%
- ✅ Inference latency < 100ms (p95)
- ✅ Monthly retraining automated

---

## Phase 5: Foundation Models for Network Behavior (2027+)

### Vision
Use the same pretrained encoder for:
- **DGA Detection**: Detect domains with anomalous lookup patterns
- **Botnet Detection**: Detect C&C traffic, secondary C2, proxy networks
- **Data Exfiltration**: Detect anomalous volume/persistence in data transfers
- **Intrusion Detection**: General-purpose network anomaly detection

### Impact
Shift from "malware classifier" to "network behavior foundation model"

---

## Resource Checklist

### Team
- [ ] 2-3 ML Engineers (data loading, model training, evaluation)
- [ ] 1-2 Data Engineers (dataset prep, feature extraction, Zeek log processing)
- [ ] 1 Security Researcher (domain expertise, threat modeling)
- [ ] 1 DevOps Engineer (deployment, monitoring, SIEM integration)

### Infrastructure
- [ ] GPU compute (1x A100 or 2x V100)
- [ ] 500GB storage (datasets + checkpoints)
- [ ] 16+ CPU cores (data preprocessing)
- [ ] Monitoring stack (ELK or equivalent)

### Data Sources
- [ ] ✅ CTU-13 (already have)
- [ ] ⚠️ UGR'16 (need to prepare)
- [ ] ⚠️ CICIDS2017 (need to prepare)
- [ ] ⚠️ MAWI archive (need samples)
- [ ] ⚠️ Enterprise traffic (synthetic or real)

---

## Documentation Files

### Strategic Level
- **STRATEGIC_ROADMAP_2026.md** (this is your master plan)
  - Phases 1-5 detailed
  - Success criteria for each phase
  - Risk mitigation strategies

### Implementation Level
- **PHASE_1_IMPLEMENTATION_GUIDE.md** (code-ready guide)
  - Week-by-week tasks
  - Copy-paste code examples
  - Command sequences

### Reference
- **README.md** (what was built)
- **EXECUTIVE_SUMMARY.md** (why it matters)
- **ARCHITECTURE_GUIDE.md** (how it works)

---

## How to Use This Plan

### Week 1-2: Phase 1 Setup
1. Read **STRATEGIC_ROADMAP_2026.md** (Phase 1 section)
2. Read **PHASE_1_IMPLEMENTATION_GUIDE.md** (Week 1 section)
3. Copy code from Week 1 tasks
4. Implement TLS, DNS, temporal extractors
5. Run tests

### Week 3-6: Phase 1 Execution
Follow Week 2-6 in **PHASE_1_IMPLEMENTATION_GUIDE.md**

### Week 7+: Phase 2 Planning
Read **STRATEGIC_ROADMAP_2026.md** (Phase 2 section)

### Publication
Each phase generates:
- Ablation study
- Cross-dataset benchmark
- Baseline comparisons
- Architecture improvements

---

## Key Decisions You're Making NOW

✅ **Decision 1**: Architecture stays session-centric initially
- Phase 1-2: Enhance current architecture
- Phase 3: Upgrade to hierarchical
- Rationale: Proven 96% recall validates baseline, enhance incrementally

✅ **Decision 2**: Features are domain-specific (TLS, DNS, temporal)
- NOT payload-based (won't work on encrypted C2)
- NOT IP/domain reputation (too easy to evade)
- Rationale: Modern C2 uses TLS+HTTPS, signature detection fails

✅ **Decision 3**: Multi-dataset validation is critical
- Cross-dataset recall ≥70% is passing bar
- Single dataset ≥97% is expected (memorization possible)
- Rationale: Proves real generalization, not overfitting

✅ **Decision 4**: Self-supervised pretraining is Phase 3 (not Phase 1)
- Phase 1-2: Supervised learning to get baseline improvements
- Phase 3: Add pretraining on unlabeled traffic
- Rationale: Baseline improvements easier to debug than pretraining issues

✅ **Decision 5**: Production deployment is Phase 4 (not Phase 1-3)
- Phases 1-3: Research & validation on static datasets
- Phase 4: Real-world validation with live traffic
- Rationale: Research-grade accuracy ≠ production-ready

---

## Expected Outcomes

### After Phase 1 (6 weeks)
- ✅ 40-feature model validated
- ✅ Cross-dataset recall ≥70%
- ✅ Feature importance understood
- ✅ Baseline comparisons show Transformer value

### After Phase 2 (14 weeks)
- ✅ Host-aware reasoning working
- ✅ Cross-dataset recall ≥72%
- ✅ Multi-session patterns recognized
- ✅ Inference latency acceptable

### After Phase 3 (24 weeks)
- ✅ Foundation model trained
- ✅ Cross-dataset recall ≥75%
- ✅ Robustness to mutations ≥94%
- ✅ Adversarial resilience proven

### After Phase 4 (ongoing)
- ✅ Production deployment
- ✅ Live recall ≥75%
- ✅ Continual learning automated
- ✅ Real-world validation complete

### After Phase 5 (2027)
- ✅ Transfer to other detection tasks
- ✅ Foundation model for network security
- ✅ Multiple applications of same encoder

---

## START HERE: Phase 1, Week 1

### Today's Task (4-6 hours)

1. **Read**:
   - STRATEGIC_ROADMAP_2026.md (Phase 1 section) — 15 min
   - PHASE_1_IMPLEMENTATION_GUIDE.md (Week 1 section) — 15 min

2. **Create files**:
   - Copy `src/features/tls_features.py` code into new file
   - Copy `src/features/dns_features.py` code into new file
   - Copy `src/features/temporal_features.py` code into new file

3. **Test**:
   - Run: `pytest tests/test_tls_features.py -v`
   - Run: `pytest tests/test_dns_features.py -v`
   - Run: `pytest tests/test_temporal_features.py -v`
   - All should pass ✅

4. **Integrate**:
   - Update `src/data_loader/feature_transforms.py` with the extended feature extraction code
   - Run: `pytest tests/test_feature_transforms.py -v`

5. **Validate**:
   - Train model on 40 features (CTU-13)
   - Verify recall ≥ 96% (should be same or better)
   - If so, Phase 1 Week 1 ✅

### This Week's Deliverables
- [ ] 3 new feature extractors coded
- [ ] All unit tests passing
- [ ] Model trains on 40 features
- [ ] Recall ≥ 96% verified

---

## Questions to Ask Yourself

Before starting each phase, ask:

**Phase 1**:
- [ ] Do I understand why TLS+DNS+temporal features matter?
- [ ] Can I explain why ports were causing memorization?
- [ ] Do I know the difference between 10 and 40 feature models?

**Phase 2**:
- [ ] Can I explain host-level context?
- [ ] Do I understand multi-session patterns?
- [ ] Can I build a HostTimeline class?

**Phase 3**:
- [ ] Do I understand self-supervised pretraining?
- [ ] Can I explain why hierarchical architecture helps?
- [ ] What is foundation model transfer?

**Phase 4**:
- [ ] What is concept drift?
- [ ] How do I monitor live performance?
- [ ] What triggers a model retraining?

If you can't answer these, re-read the relevant section.

---

## Final Words

This is a **6-month research & engineering roadmap** to build a world-class C2 detection system.

The journey:
1. **Month 1** (Phase 1): Validate enhanced features work better
2. **Month 2** (Phase 2): Add host-level reasoning
3. **Month 3** (Phase 3): Build foundation model capabilities
4. **Month 4+** (Phase 4): Deploy to production
5. **2027** (Phase 5): Transfer to other tasks

The philosophy:
- **One variable at a time** (avoid confounded experiments)
- **Validate before advancing** (success criteria matter)
- **Research-grade first** (then production-grade)
- **Benchmark constantly** (prove improvement, not assumed)
- **Document everything** (publish findings)

**This is not a simple classifier. This is a behavioral reasoning system for modern C2 detection.**

Good luck. Let's build something amazing. 🚀

