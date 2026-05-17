# Post-Experiment: Next Phase Planning

**Decision**: Strong Generalization → **PROCEED TO SHADOW MODE DEPLOYMENT**

**Date**: May 10, 2026  
**Outcome**: Recall 96.36% on unseen Conficker family exceeds 75% threshold  
**Recommendation**: Session-centric approach validated; NO architectural redesign needed

---

## Phase 4: Shadow Mode Deployment & Live Validation (Weeks 1-4)

### Objectives
1. Deploy trained model to production monitoring (shadow mode)
2. Collect live C2 traffic metrics and ground truth labels
3. Validate model performance on real-world network data
4. Establish operational decision thresholds
5. Identify error patterns requiring future refinement

### Success Criteria
- ✅ Recall ≥ 75% on live traffic
- ✅ False Positive Rate ≤ 5% acceptable for operational threshold
- ✅ Model deployment latency < 100ms per session
- ✅ Zero crashes or errors in production
- ✅ Ground truth labels obtained for >80% of detected anomalies

### Timeline
- **Week 1**: Deploy infrastructure, model integration, monitoring setup
- **Week 2**: Collect ~1-2 weeks of traffic data, initial metrics
- **Week 3**: Error analysis, threshold optimization, feedback loops
- **Week 4**: Finalize operational procedures, documentation, go/no-go decision

### Workstreams

#### 1. Deployment Infrastructure (Week 1)
**Tasks**:
- [ ] Create shadow-mode inference pipeline
  - Load model checkpoint
  - Implement session windowing from live Zeek logs
  - Batch inference (64 sessions per batch)
  - Score caching and TTL
- [ ] Set up monitoring and metrics collection
  - Log all session scores
  - Store high-confidence predictions
  - Collect ground truth labels (security team input)
- [ ] Create alerting (optional for shadow mode)
  - Alert on high-confidence C2 detections
  - Log to SIEM for human review
- [ ] Performance testing
  - Latency benchmarking
  - Throughput testing (sessions/sec)
  - Memory profiling

**Deliverables**:
- `src/inference/shadow_mode_pipeline.py`
- `src/monitoring/metrics_collector.py`
- Deployment configuration (ports, batch settings, etc.)

#### 2. Data Collection (Weeks 2-3)
**Tasks**:
- [ ] Run shadow mode continuously on live traffic
- [ ] Collect model outputs:
  - Session-level scores
  - Decision thresholds applied
  - Per-family predictions (if multi-family routing available)
- [ ] Gather ground truth labels
  - Work with security team for known C2 incident labels
  - Document detection accuracy of labeled incidents
  - Collect false positive reports from analysts
- [ ] Aggregate metrics
  - Daily recall/FPR snapshots
  - Distribution of scores (histogram)
  - Error samples for analysis

**Deliverables**:
- Live metrics dashboard (CSV or JSON logs)
- Ground truth label set (~50-100 labeled sessions minimum)
- Anomaly report (high-score sessions for analyst review)

#### 3. Validation & Analysis (Week 3)
**Tasks**:
- [ ] Compute live metrics with ground truth
  - Recall, FPR, precision, F1 on labeled data
  - Compare to offline test results (96.36% recall)
  - Identify drift or domain shifts
- [ ] Error analysis on live data
  - False negatives: cluster by session characteristics
  - False positives: identify common benign patterns
  - Correlation with network changes or new apps
- [ ] Threshold optimization
  - Analyze decision boundary sensitivity
  - Recommend operational thresholds for different risk levels
- [ ] Document findings
  - Comparison: offline vs. live performance
  - Identified domain shifts or challenges
  - Recommendations for model refinement

**Deliverables**:
- Live validation report (metrics + error analysis)
- Recommended operational thresholds
- List of identified domain shifts

#### 4. Operational Procedures (Week 4)
**Tasks**:
- [ ] Finalize deployment checklist
- [ ] Create runbook for model updates
- [ ] Set up alerts and escalation procedures
- [ ] Document known limitations (e.g., FN clusters)
- [ ] Plan for model retraining (quarterly, on new malware families)
- [ ] Go/no-go decision criteria

**Deliverables**:
- Deployment checklist
- Operational runbook
- Alert rules and thresholds
- Model retraining plan

---

## Phase 5: Expanded Validation (Optional, Weeks 5-8)

**Trigger**: If live validation (Phase 4) shows recall ≥ 75% and FPR ≤ 5%

### Objectives
- Test on additional malware families (Emotet, TrickBot, Trickbot, etc. if available)
- Evaluate on diverse network environments (enterprise, ISP, mobile)
- Assess robustness to adversarial perturbations
- Identify edge cases and limitations

### Timeline
- 4 weeks (parallel to Phase 4 if resources available)

### Key Experiments
1. **Multi-Family Validation**: Retrain on 2 families, test on 3+ held-out families
2. **Network Diversity**: Evaluate on different network types and protocols
3. **Adversarial Robustness**: Test robustness to feature noise, timing shifts
4. **Scalability**: Evaluate on 100M+ session datasets

---

## Phase 6: Enhancement & Refinement (Optional, Weeks 9+)

**Trigger**: If error analysis or new threats emerge during Phase 4-5

### Potential Research Directions
1. **Host-Centric Augmentation** (if session-level hits plateau)
   - Combine with host temporal patterns
   - Implement ensemble methods
   - Test on multi-session C2 chains

2. **Feature Augmentation**
   - Add encryption indicators (TLS fingerprints)
   - Include geolocation and ASN information
   - Temporal features (session inter-arrival time)

3. **Domain Adaptation**
   - Fine-tune on new malware families (transfer learning)
   - Adversarial domain adaptation for new networks
   - Few-shot learning for emerging threats

---

## Resource Requirements

### Phase 4 (Immediate)
- **Engineering**: 1 FTE (deployment + monitoring)
- **Security/Ops**: 0.5 FTE (ground truth labeling, analysis)
- **Infrastructure**: Zeek deployment, Kafka/logging, monitoring dashboards
- **Compute**: CPU-based inference (no GPU required), ~2-4 cores, 8 GB RAM

### Phase 5 (Optional)
- **Engineering**: 0.5 FTE (experiment coordination)
- **Compute**: Dataset acquisition and analysis
- **Datasets**: Public malware capture collections (Stratosphere, UNB UNSW, ISOT, etc.)

---

## Decision Gates

### Gate 1: After Phase 4 (Week 4)
- **Go**: Live recall ≥ 75%, FPR ≤ 5% → Deploy to production
- **Go with monitoring**: 60-75% recall, FPR 5-10% → Deploy with human review
- **Hold**: Recall < 60% → Investigate domain shift, consider architectural changes

### Gate 2: After Phase 5 (Week 8)
- **Go**: Expanded validation shows consistent performance → Prod rollout
- **Iterate**: Edge cases identified → Plan enhancements
- **Redesign**: Major failure on new families → Trigger Phase 6 research

---

## Known Limitations & Future Work

### Current Limitations
1. **Session-only**: Cannot detect multi-session C2 chains or long-term persistence
2. **Real-time latency**: No live streaming; requires session completion for inference
3. **Benign size**: Test set has only 186 benign samples (limited FPR confidence)
4. **Malware diversity**: Validated on 3 families; generalization to 10+ families unknown

### Future Work (Candidates)
1. Temporal/multi-session modeling (Phase 6 research)
2. Encrypted payload analysis (if TLS fingerprints available)
3. Active learning from analyst feedback
4. Few-shot learning for zero-day families

---

## Success Metrics & Milestones

| Milestone | Timeline | Success Criteria | Owner |
|-----------|----------|------------------|-------|
| **Shadow Deploy** | Week 1 | Deployment latency < 100ms, 0 errors | Eng |
| **Data Collection** | Weeks 2-3 | 1M+ sessions scored, 50+ labeled | Ops/Sec |
| **Live Validation** | Week 3 | Recall ≥ 75%, report completed | Eng/Sec |
| **Go/No-Go Decision** | Week 4 | Decision documented | Management |
| **Production Deploy** | Week 5+ | Live detections operational | Ops |

---

## References

- **Offline Results**: EXPERIMENT_RESULTS_STRICT_MULTIFAMILY.md
- **Model Config**: experiments/multifamily_generalization/config.json
- **Reproducibility**: REPRODUCIBILITY.md
- **Decision Logic**: PROGRESS_CHECKLIST.md (Week 3)

---

**Document Created**: May 10, 2026 | **Next Review**: Week 4 (Post-Phase 4 Go/No-Go)
