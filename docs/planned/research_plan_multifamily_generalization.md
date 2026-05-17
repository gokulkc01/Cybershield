# Research Plan: Multi-Family C2 Generalization Experiment

**Date**: May 9, 2026  
**Research Question**: Do session-level behavioral flow features capture malware-family-invariant C2 behavior when trained on sufficient multi-family diversity?  
**Status**: 🔴 CRITICAL METHODOLOGICAL CORRECTION - Previous conclusions invalid

---

## Executive Summary

**Previous Conclusion Was Wrong**

The Phase 3 host-centric architectural redesign was based on a **confounded experiment**:
- UWF training: 22 C2 samples (17 after split)
- Extreme sample starvation
- Single malware family
- Lab-controlled traffic only
- 526x recall collapse on held-out sources

**Premature Attribution**: Claimed session-centric architecture failed, when actually:
- Could be feature failure
- Could be representation failure  
- Could be sample starvation failure
- Multiple confounded variables

**Scientific Principle**: **Change ONE major variable at a time**

**This Experiment Changes**: Dataset diversity (multi-family C2 training)  
**This Experiment Keeps Constant**: Features, architecture, representation

---

## Research Question

### Primary Question
**Does session-level behavioral flow features capture malware-family-invariant C2 behavior when trained on sufficient multi-family C2 diversity?**

### What We're Testing
✅ Can session-centric features generalize across C2 families with adequate training data?  
✅ Do behavioral flow invariants (duration, bytes, cadence) work across Conficker, Alureon, etc.?  
✅ Is the previous failure attributable to data starvation, not architecture?

### What We're NOT Testing (Yet)
❌ Tiny-data learning viability  
❌ Host-centric superiority  
❌ Temporal aggregation benefits  
❌ Graph-based modeling  
❌ Multi-session sequences

**Architecture remains unchanged until this question is answered.**

---

## Experimental Design

### Phase 1: Multi-Family Training Dataset

**Goal**: Dramatically increase C2 sample count + behavioral diversity

**Data Sources**:
```
CTU-13 Scenario 1:  Neris botnet
  - Family: Neris/Conficker variant
  - Duration: 7 days
  - Estimated sessions: ~50K+
  - Benign baseline: Yes

CTU-13 Scenario 2:  Kraken botnet
  - Family: Kraken/Botnet
  - Duration: 9 days
  - Estimated sessions: ~100K+
  - Benign baseline: Yes

CTU-13 Scenario 9:  Conficker botnet
  - Family: Conficker
  - Duration: 5 days
  - Estimated sessions: ~150K+
  - Benign baseline: Yes

Total C2 diversity: 3 distinct malware families
Total sessions: ~300K+
Benign sessions: ~50K+
```

**Do NOT include UWF-ZeekData24 in training**

Reason:
- Lab-controlled traffic fundamentally different from wild captures
- Would confound with family diversity effect
- Can use optionally for auxiliary testing later

**Dataset Statistics to Compute**:
- Per-family C2 session count
- Per-family duration distribution
- Per-family protocol distribution
- Per-family port distribution (before removal)
- Class imbalance ratio
- Feature distribution by family
- Temporal distribution by family

### Phase 2: Feature Set Refinement

**Current Behavioral Features** (RETAIN):
```
✅ duration            (session duration in seconds)
✅ orig_bytes          (bytes from originator)
✅ resp_bytes          (bytes from responder)
✅ orig_pkts           (packets from originator)
✅ resp_pkts           (packets from responder)
✅ bytes_per_pkt       (aggregate statistics)
✅ packet_ratio        (directional balance)
✅ byte_ratio          (directional balance)
✅ iat (inter-arrival time) (timing invariant)
✅ Protocol type       (TCP/UDP/ICMP)
✅ Timing cadence      (beacon-relevant statistics)
✅ Flow direction      (is_outbound)
```

**Remove These Features**:
```
❌ src_port           (family-specific memorization likely)
❌ dst_port           (family-specific memorization likely)
```

**Rationale for Port Removal**:
- Previous ablation suggested ports caused family-specific overfitting
- Conficker uses port 445, Kraken uses different ports
- Model likely memorized "port X = family Y"
- Removing forces focus on behavioral invariants
- Behavioral invariants should be: persistence, duration, cadence, symmetry

**Do NOT redesign features yet**:
- No host aggregation
- No temporal smoothing
- No graph relationships
- Pure session-level flow statistics

**Feature processing pipeline**:
1. Load raw netflow/Zeek logs
2. Normalize to 12-feature schema (minus ports)
3. Apply z-score normalization per-source
4. Generate training tensors

### Phase 3: Architecture (UNCHANGED)

**Current Model**: Transformer on session-level flows

**Keep Exactly As Is**:
```
✅ Input shape: (batch, 20, 12)  [20 flows per session, 12 features]
✅ Embedding dimension: 256
✅ Attention heads: 4
✅ Feed-forward dim: 512
✅ Dropout: 0.1
✅ Output: (batch, 1) risk score

NO changes to model internals.
NO architectural experimentation during this phase.
```

**Allowed Changes** (Bug fixes, stability):
- Logging improvements
- Reproducibility (fixed seeds)
- Documentation
- Training stability monitoring
- Gradient checking

**Forbidden Changes**:
- Input representation changes
- Attention mechanism changes
- Output head changes
- Host timeline integration
- Multi-session aggregation

---

## Phase 4: Data Splitting Strategy

### Strict Family-Aware Evaluation

**Principle**: **No malware family appears in both train and test**

**Split Strategy 1**: Family-based
```
Training:   CTU-13 Scenario 1 (Neris) + Scenario 2 (Kraken)
Test:       CTU-13 Scenario 9 (Conficker)  [never seen Conficker during training]

Metrics:
- Zero-shot family generalization (Conficker)
- Cross-family behavioral invariants
```

**Split Strategy 2**: Source-based (if available)
```
Training:   CTU-13 Scenarios 1, 2, 9 combined
Test:       MCFP dataset (different C2 variants)

Metrics:
- Cross-dataset generalization
- Absolute ground truth on wild-captured malware
```

**Contamination Prevention**:
- [ ] No session overlap (check flow 5-tuples)
- [ ] No temporal overlap within same host
- [ ] No host IP reuse across splits
- [ ] No protocol port reuse for fingerprinting
- [ ] Family tags must be explicitly verified

**Reproducibility**:
- [ ] Split manifests saved as JSON
  ```json
  {
    "split_name": "ctu_family_aware_v1",
    "train_families": ["neris", "kraken"],
    "test_families": ["conficker"],
    "train_sessions": 250000,
    "test_sessions": 150000,
    "benign_ratio": 0.15,
    "random_seed": 42,
    "feature_hash": "sha256_of_feature_columns"
  }
  ```
- [ ] Dataset hashes for integrity verification
- [ ] Experiment metadata logged

---

## Phase 5: Evaluation Metrics

### Primary Metrics (C2 Detection is Imbalanced)

**Why Accuracy is Wrong**:
```
Assume: 1% of test sessions are C2
Model says: "Everything is benign"
Accuracy: 99%
Reality: Model is useless (0% recall)
```

**Use These Instead**:

| Metric | Why | Interpretation |
|--------|-----|-----------------|
| **PR-AUC** | Primary for imbalanced data | Area under precision-recall curve |
| **Recall @ FPR** | Operational relevance | Can we detect C2 at acceptable false-alarm rate? |
| **Precision @ Recall** | Triage effectiveness | How many alerts are real? |
| **ROC-AUC** | Secondary, baseline | Discriminatory ability |
| **Calibration** | Confidence quality | Are predicted probabilities reliable? |

**Per-Family Breakdown**:
```
Metric: Recall per malware family
Why: Shows which families generalize vs. which don't

Example output:
  Neris (train family):       87% recall
  Kraken (train family):      85% recall
  Conficker (zero-shot):      62% recall  ← Generalization signal

This isolates family-specific failure modes.
```

**Operational Point**:
```
Operating point: "Recall ≥ 70% at FPR < 2%"
Can model achieve this cross-family?
```

---

## Phase 6: Failure Analysis

### Detailed Error Analysis Infrastructure

**For Each False Negative** (missed C2 session):

Collect:
```
- Malware family
- Session protocol (TCP, UDP, ICMP)
- Session duration (seconds)
- Bytes sent/received
- Packet count asymmetry
- Flow direction (inbound vs. outbound)
- Timing cadence (inter-arrival patterns)
- Port used (for analysis, even if removed from features)
- Is_outbound flag
- HTTPS/encryption indicators
- Beacon-like behavior presence

Example FN analysis:
  Family: Conficker
  Protocol: TCP 445 (SMB)
  Duration: 12 seconds
  Bytes: 5KB → 2KB (client → server asymmetry)
  Cadence: Regular 4-hour intervals
  Question: Why didn't model detect persistence pattern?
  Hypothesis: Port removal lost the "port 445 = C2" signal
  Action: Investigate whether removal helps or hurts
```

**For Each False Positive** (wrongly flagged benign):

Collect:
```
- Destination IP (is it a known CDN/service?)
- Application/service (HTTP, DNS, NTP?)
- Frequency (is it periodic?)
- Byte patterns (symmetric or asymmetric?)
- Protocol (TLS/UDP/other?)
- Is_outbound flag
- Duration

Example FP analysis:
  Service: DNS query
  Duration: 0.5 seconds
  Bytes: 0.1KB (typical for DNS)
  Frequency: Every 30 minutes (periodic)
  Question: Why flagged as C2?
  Hypothesis: Periodic traffic looks like beaconing
  Action: Better distinguish legitimate periodic traffic
```

### Structured Error Reports

**Error Distribution by Family**:
```python
{
  "conficker": {
    "false_negatives": 15000,
    "false_positives": 500,
    "recall": 0.62,
    "precision": 0.95,
    "top_fn_reasons": [
      "duration_too_short",
      "encrypted_payload",
      "atypical_ports"
    ]
  },
  "kraken": {
    "false_negatives": 2000,
    "false_positives": 200,
    "recall": 0.87,
    "precision": 0.97
  }
}
```

**Error Clusters**:
- Group false negatives by shared characteristics
- Identify systematic failure modes
- Distinguish random noise from systematic gaps

---

## Phase 7: Decision Logic

### Experimental Outcomes & Next Steps

**Outcome 1: Strong Cross-Family Generalization**
```
Evidence:
  - Recall ≥ 75% on all test families
  - Consistent performance across Neris, Kraken, Conficker
  - No major false positive inflation

Conclusion:
  Session-level behavioral invariants ARE family-invariant
  Transformation to host-centric NOT justified by data

Next step:
  - Investigate minor family-specific gaps
  - Consider modest feature engineering (NOT architectural redesign)
  - Deploy session-centric model with confidence
```

**Outcome 2: Partial Generalization with Clear Patterns**
```
Evidence:
  - Recall 50-75% cross-family
  - Systematic gaps by protocol or timing characteristics
  - Can be improved but not by simple tuning

Conclusion:
  Session-level features capture part of invariants
  Missing temporal/context information
  Architecture redesign MAY be justified

Next step:
  - Detailed failure analysis by protocol/timing
  - Consider host-centric modeling for persistence detection
  - Hypothesis: single sessions miss beaconing patterns
  - Test: can temporal sequences help?
```

**Outcome 3: Complete Cross-Family Failure**
```
Evidence:
  - Recall <50% even with diverse training
  - Per-family recall varies wildly (20% to 80%)
  - No improvement with larger datasets

Conclusion:
  Session-level flow representation fundamentally insufficient
  Multiple malware families use incompatible evasion strategies
  Architecture redesign is scientifically justified

Next step:
  - Investigate: which families fail and why?
  - Consider host-centric sequences (temporal context)
  - Consider graph-based relationships
  - Consider protocol-specific models
```

**Decision Rule**:
```
We DO NOT assume Outcome C in advance.
We measure Outcome 1 or 2, THEN decide.
We do NOT speculate about architecture deficiencies.
```

---

## Implementation Tasks

### Task 1: Clean Up Contamination
- [ ] Review current codebase for host-centric fragments
- [ ] Remove any half-finished host timeline training code
- [ ] Ensure session-centric pipeline is pristine
- [ ] Create clean feature branch: `research/multifamily_generalization`

### Task 2: Build Dataset Construction Pipeline
- [ ] Define CTU-13 scenario loading
  - [ ] Parse Scenario 1 (Neris): raw → normalized
  - [ ] Parse Scenario 2 (Kraken): raw → normalized
  - [ ] Parse Scenario 9 (Conficker): raw → normalized
- [ ] Consolidate into unified format
- [ ] Generate feature statistics (per-family distributions)
- [ ] Compute class imbalance report

### Task 3: Feature Processing (Remove Ports)
- [ ] Update feature schema: 12 features → 10 features (remove src_port, dst_port)
- [ ] Regenerate feature transforms for new schema
- [ ] Regenerate normalization baselines
- [ ] Verify feature distributions are reasonable

### Task 4: Family-Aware Split Generator
- [ ] Implement `FamilyAwareSplitter` class
  ```python
  split = FamilyAwareSplitter(
      train_families=['neris', 'kraken'],
      test_families=['conficker'],
      benign_ratio=0.15
  )
  train_sessions, test_sessions = split.create_split(all_sessions)
  ```
- [ ] Create split manifest (reproducible)
- [ ] Verify no family leakage
- [ ] Compute per-family counts

### Task 5: Model Training Pipeline
- [ ] Ensure Transformer unchanged
- [ ] Update training loop to:
  - [ ] Track loss by malware family
  - [ ] Log per-family validation metrics
  - [ ] Save checkpoints at best validation performance
- [ ] Add reproducibility: fixed seeds, config snapshots

### Task 6: Evaluation Pipeline
- [ ] Compute primary metrics: PR-AUC, Recall, Precision, FPR
- [ ] Generate per-family breakdown
- [ ] Implement error analysis:
  - [ ] False negative clustering
  - [ ] False positive clustering
  - [ ] Family-specific failure modes
- [ ] Create confusion analysis reports

### Task 7: Experiment Reproducibility
- [ ] Config snapshots for every run
- [ ] Dataset hashes (SHA256 of all session tensors)
- [ ] Random seeds logged explicitly
- [ ] Experiment metadata:
  ```json
  {
    "date": "2026-05-09",
    "question": "multifamily_generalization",
    "training_families": ["neris", "kraken"],
    "test_families": ["conficker"],
    "train_sessions": 250000,
    "test_sessions": 150000,
    "features": ["duration", "bytes", "packets", "cadence", "..."],
    "features_removed": ["src_port", "dst_port"],
    "random_seed": 42,
    "model_checkpoint": "path/to/model.pth"
  }
  ```

### Task 8: Metrics Dashboard
- [ ] Create comprehensive results table:
  ```
  | Family     | Train? | Recall | Precision | PR-AUC | FPR @ 70% recall |
  |------------|--------|--------|-----------|--------|------------------|
  | Neris      | Yes    | 87%    | 96%       | 0.94   | 1.2%             |
  | Kraken     | Yes    | 85%    | 95%       | 0.92   | 1.5%             |
  | Conficker  | No     | 62%    | 94%       | 0.81   | 2.8%             |
  ```
- [ ] Visualization: ROC curves per family
- [ ] Visualization: PR curves per family
- [ ] Visualization: Feature importance (if available from model)

---

## Success Criteria

### Minimum Success (Question Answered)
- [ ] Multi-family dataset loaded and verified
- [ ] Model trains without errors
- [ ] Cross-family evaluation completes
- [ ] Clear outcome: which of [Strong, Partial, Complete] generalization?

### Strong Success (Question Validated)
- [ ] Cross-family recall ≥ 70%
- [ ] Per-family performance consistent
- [ ] False positive rate < 2% @ target recall
- [ ] Error analysis shows systematic patterns (not random)

### Excellent Success (Ready for Deployment)
- [ ] Cross-family recall ≥ 80%
- [ ] FPR < 1.5% @ target recall
- [ ] Model generalizes across:
  - Seen families (Neris, Kraken)
  - Unseen families (Conficker zero-shot)
  - Potentially new datasets (MCFP if available)

---

## Timeline

### Week 1 (May 13-17)
- [ ] **Day 1**: Dataset loading infrastructure
- [ ] **Day 2**: Feature processing (remove ports)
- [ ] **Day 3**: Multi-family split generator
- [ ] **Day 4**: Model training setup
- [ ] **Day 5**: Initial training run + results review

### Week 2 (May 20-24)
- [ ] **Day 1**: Evaluation pipeline
- [ ] **Day 2**: Error analysis infrastructure
- [ ] **Day 3**: Per-family analysis
- [ ] **Day 4**: Metrics dashboard
- [ ] **Day 5**: Results interpretation

### Week 3 (May 27-31)
- [ ] Reproducibility verification
- [ ] Final results documentation
- [ ] Decision logic applied
- [ ] Next phase planning (based on outcomes)

---

## Critical Warnings

### ⚠️ Do NOT:
- ❌ Skip family-aware splitting (will leak)
- ❌ Include UWF in training without explicit justification
- ❌ Change architecture before results
- ❌ Make feature engineering changes mid-experiment
- ❌ Run multiple experiments without tracking configs
- ❌ Assume results will show host-centric is needed
- ❌ Use accuracy as a metric

### ✅ DO:
- ✅ Track every experiment with metadata
- ✅ Preserve intermediate results
- ✅ Create detailed error reports
- ✅ Let data answer the question
- ✅ Be prepared for any outcome
- ✅ Question your assumptions
- ✅ Use appropriate metrics for imbalanced data

---

## Expected Outcomes & Paths Forward

### If Strong Generalization (Outcome 1)
- Session-centric features work
- No architectural redesign needed
- Focus: minor tuning + deployment
- Next: Production validation

### If Partial Generalization (Outcome 2)
- Session-level captures some invariants
- Host-centric modeling becomes justified
- Next: Implement Phase 3 host-centric redesign
- With solid scientific foundation

### If Complete Failure (Outcome 3)
- Session-level insufficient
- Redesign to temporal sequences (host-centric)
- Add graph-based relationships
- Investigate protocol-specific models

**Key Point**: We measure first, then decide. Not the other way around.

---

## Conclusion

This experiment answers a single, precise question:

**"Do behavioral flow-level invariants generalize across malware families when trained on sufficient multi-family diversity?"**

The answer will:
1. Validate or invalidate session-centric approach
2. Justify or eliminate host-centric redesign
3. Identify family-specific gaps (if any)
4. Provide data-driven basis for next phase

**We proceed scientifically. We measure before we redesign.**

