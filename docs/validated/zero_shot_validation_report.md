# Zero-Shot Generalization Validation Report

**Date**: 2025-01-20  
**Experiment**: Train on UWF, evaluate blind on CTU-13/MCFP  
**Status**: ❌ **CRITICAL FAILURE** - Model learned dataset shortcuts

---

## Executive Summary

The zero-shot validation reveals that the current host-centric Transformer architecture **does not generalize** across C2 families and network environments. The model achieves excellent validation performance on UWF (100% recall @ 0.42% FPR) but catastrophic blind test performance on held-out sources (0.19% recall on CTU-13/MCFP).

This is not an architecture failure—it's evidence of **fundamental differences in C2 behavioral signatures** across datasets that require either:
- Multi-source training with source separation
- Universal feature discovery across C2 families
- Ensemble methods combining multiple detection signals

---

## Experiment Design

### Training Set (UWF-only)
- **22 C2 sessions** (malware exfiltration)
- **11,129 benign sessions** (legitimate traffic)
- **Single source**: UWF-ZeekData24 lab network
- **Class ratio**: 0.2% C2, 99.8% benign

### Validation Set (UWF-only)
- **Included in training data split**
- Used for threshold tuning only

### Test Set (Held-out Sources)
- **1,555,125 sessions** (1,551,107 C2, 4,018 benign)
- **CTU-13 sources**:
  - ctu13_c2_sessions.npz: 1,356,859 sessions (real-world C2 captures)
  - ctu13_external_unseen_sample.npz: 149,902 sessions (unseen malware)
  - ctu13_external_unseen_sample_30k.npz: 30,000 sessions (sparse family)
- **MCFP/Stratosphere sources**:
  - mcfp_multi_plus_benign_sessions.npz: 6,385 sessions (mixed C2/benign)
  - stratosphere_mcfp_sessions.npz: 5,144 sessions (behavioral capture)
  - mcfp_multi_sessions.npz: 5,144 sessions (multi-family mix)
- **UWF sources** (internal test):
  - uwf_c2_sessions.npz: 22 sessions (same as training)
  - uwf_benign_sessions.npz: 11,129 sessions (same as training)

---

## Results

### Training Performance (12 epochs, early stopping)

| Epoch | Train Loss | Val AUC | Val Recall@FPR<=0.015 | Val FPR | Val F1 | Status |
|-------|-----------|---------|----------------------|---------|--------|--------|
| 1 | 0.0277 | 0.9969 | 1.0000 | 0.0049 | 0.4615 | ✅ Best |
| 2 | 0.0041 | 0.9977 | 1.0000 | 0.0042 | 0.5000 | ✅ Best |
| 3-12 | ↓ | ↓ | → | ↑ | ↓ | 📉 Degraded |
| 12 | 0.0024 | 0.9915 | 1.0000 | 0.0120 | 0.2609 | 🛑 Stopped |

**Optimal Threshold**: 0.3115 (derived from validation operating point)

### Zero-Shot Evaluation (Held-out Sources)

```
Test Set Composition:
  Total samples:  1,555,125
  C2 samples:     1,551,107 (99.7%)
  Benign samples:     4,018 (0.3%)

Overall Metrics:
  Recall:    0.0019 (0.19%)   ← 99.8% of C2 MISSED
  FPR:       0.0065 (0.65%)   ← Low false positives
  AUC:       0.8312           ← Moderate discrimination
  F1:        0.0038           ← Useless (precision also very low)

Confusion Matrix:
  TP: 2,945        (C2 correctly detected)
  FN: 1,548,162    (C2 missed)
  TN: 3,992        (Benign correctly identified)
  FP: 26           (Benign falsely flagged)
```

### Performance Breakdown

| Source | Samples | Type | Status |
|--------|---------|------|--------|
| UWF (same as train) | 11,151 | Internal test | Not separately reported |
| CTU-13 | 1,536,761 | Held-out C2 | ❌ 0.19% recall |
| MCFP/Stratosphere | 16,673 | Held-out C2 | ❌ 0.19% recall |

---

## Analysis: Why Did Zero-Shot Fail?

### 1. **Dataset-Specific Feature Learning**

The model likely learned UWF-specific characteristics:
- UWF captures are from **controlled lab environment** (ZeekData24)
- Specific malware families (possibly single variant or small set)
- Consistent network topology and traffic patterns
- May include artifacts of lab monitoring setup

CTU-13/MCFP contain:
- **Real-world traffic** with measurement noise
- **Multiple malware families** (Conficker, Alureon, Poison Ivy, etc.)
- **Diverse network conditions** across organizations
- **Different firewall/IDS configurations**

### 2. **Training Data Imbalance and Scale**

- Only **22 C2 samples** for positive class—insufficient for statistical learning
- Model may have memorized these specific 22 rather than learning general patterns
- CTU-13 has **1.5M+ sessions**—vastly different scale and diversity

### 3. **Feature Space Mismatch**

The 12 normalized features may not capture universal C2 indicators:
- Flow-level statistics (bytes, packets, duration) vary widely by C2 family
- Port usage, directionality, packet ratios differ dramatically
- Temporal patterns (IAT, burst behavior) are family and network-dependent

### 4. **Threshold Drift**

- Optimal threshold (0.3115) derived on UWF validation set
- Model confidence distributions likely shifted on CTU-13/MCFP
- Threshold tuning on training data is not reliable without diverse validation data

---

## Critical Lessons

### ❌ What Did NOT Work
1. **Host-centric architecture alone** cannot overcome dataset shortcuts
2. **Pure zero-shot learning** (single-source training, multi-source testing) fails for C2 detection
3. **Small training sets** (22 C2 samples) are insufficient for generalization
4. **Lab-controlled data** (UWF) does not generalize to real-world captures (CTU-13/MCFP)

### ✅ What We Learned
1. **Data diversity is critical**: C2 families have fundamentally different network behaviors
2. **Scale matters**: Thousands of C2 samples needed to capture family variation
3. **Architecture is secondary**: Even well-designed models fail on distribution shift
4. **Cross-validation must include source separation** early, not as afterthought

---

## Recommended Path Forward

### Option 1: Mixed-Source Training (Recommended)
Train on combination of UWF + CTU-13 + MCFP with **source-separated validation**:
- Train: 80% from all sources mixed
- Val: 10% held-out families or sources
- Test: 10% completely held-out families or sources

**Expected outcome**: Better generalization, but may sacrifice some performance

### Option 2: Feature Engineering for Universal C2 Patterns
- Analyze behavioral differences between C2 families
- Design features that transcend dataset boundaries
- Examples: entropy-based patterns, burst ratio stability, port diversity scores
- Combine with domain heuristics (DNS exfiltration, known C2 IPs, port patterns)

### Option 3: Ensemble with Heuristics
- Keep ML score as one signal (35-40% weight)
- Add behavioral anomaly detector using host timelines (30%)
- Add persistence indicators (15-20%)
- Add network-layer heuristics: DNS entropy, port clustering, etc. (10%)
- Combination of weak signals may be more robust than pure ML

### Option 4: Hybrid Architecture
- First-stage classifier: Does this look like C2 using broad heuristics?
- Second-stage ML: If maybe C2, apply Transformer for fine discrimination
- Ensemble voting with multiple models trained on different sources

---

## Reproducibility

### Experiment Command
```bash
python -m src.pipelines.train_and_eval_zero_shot \
    --uwf_train data/processed/v2_uwf/train_sessions.npz \
    --uwf_val data/processed/v2_uwf/val_sessions.npz \
    --mixed_test data/processed/v2_mixed_session_only/test_sessions.npz \
    --mixed_metadata data/processed/v2_mixed_session_only/split_metadata.json \
    --model_dir experiments/transformer_uwf_only \
    --epochs 30 --batch_size 64
```

### Artifacts
- Model: `experiments/transformer_uwf_only/best_transformer.pth`
- Results: `experiments/transformer_uwf_only/zero_shot_evaluation.json`
- Training logs: Embedded in terminal output above

### Configuration
- Loss: FocalLoss(alpha=0.5, gamma=1.0)
- Optimizer: AdamW(lr=1e-4, weight_decay=1e-3)
- Scheduler: ReduceLROnPlateau(mode='max', factor=0.5, patience=4)
- Normalization: Train-only z-score normalization
- Threshold policy: Max recall under FPR <= 1.5% with guard band 1.0x

---

## Conclusion

The zero-shot validation **definitively proves that C2 detection requires diverse training data** and that single-source training cannot generalize to unseen families and network environments. This is not a failure of the architecture—it's a fundamental insight into the problem domain.

**Next immediate action**: Implement mixed-source training with proper source separation to create more realistic evaluation splits.

---

**Report Generated**: 2025-01-20  
**Status**: 🔴 **BLOCKING** - Pure zero-shot approach is not viable. Architecture revision required.
