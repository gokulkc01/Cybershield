# Feature Schema Update: 12 → 10 Features (May 9, 2026)

**Purpose**: Document feature schema change for controlled multi-family generalization experiment

**Reason**: Previous ablation study showed src_port and dst_port cause family-specific memorization (model learns port patterns unique to each C2 family rather than generalizable behavioral invariants)

---

## Current Schema (Phase 2)

**Total Features**: 12  
**Session Length**: 20 flows per session  
**Feature Dimension**: (batch, 20, 12)

### Feature List with Indices

| Idx | Feature Name | Type | Description | Keep? |
|-----|--------------|------|-------------|-------|
| 0 | orig_bytes | int64 | Bytes sent by origin | ✅ YES |
| 1 | resp_bytes | int64 | Bytes sent by responder | ✅ YES |
| 2 | orig_pkts | int64 | Packets sent by origin | ✅ YES |
| 3 | resp_pkts | int64 | Packets sent by responder | ✅ YES |
| 4 | bytes_per_pkt | float32 | Average bytes per packet | ✅ YES |
| 5 | packet_ratio | float32 | Ratio of packets orig:resp | ✅ YES |
| 6 | byte_ratio | float32 | Ratio of bytes orig:resp | ✅ YES |
| 7 | is_outbound | int32 | 1 if outbound, 0 if inbound | ✅ YES |
| 8 | duration | float32 | Flow duration in seconds | ✅ YES |
| 9 | src_port | int32 | Source port number | ❌ REMOVE |
| 10 | dst_port | int32 | Destination port number | ❌ REMOVE |
| 11 | iat | float32 | Inter-arrival time in seconds | ✅ YES |

**Total After Removal**: 10 features

---

## New Schema (Week 1 Onwards)

**Total Features**: 10  
**Session Length**: 20 flows per session (unchanged)  
**Feature Dimension**: (batch, 20, 10)

### Updated Feature List with New Indices

| New Idx | Old Idx | Feature Name | Type | Description |
|---------|---------|--------------|------|-------------|
| 0 | 0 | orig_bytes | int64 | Bytes sent by origin |
| 1 | 1 | resp_bytes | int64 | Bytes sent by responder |
| 2 | 2 | orig_pkts | int64 | Packets sent by origin |
| 3 | 3 | resp_pkts | int64 | Packets sent by responder |
| 4 | 4 | bytes_per_pkt | float32 | Average bytes per packet |
| 5 | 5 | packet_ratio | float32 | Ratio of packets orig:resp |
| 6 | 6 | byte_ratio | float32 | Ratio of bytes orig:resp |
| 7 | 7 | is_outbound | int32 | 1 if outbound, 0 if inbound |
| 8 | 8 | duration | float32 | Flow duration in seconds |
| 9 | 11 | iat | float32 | Inter-arrival time in seconds |

---

## What Gets Changed

### Files That Need Updates

1. **src/features/feature_config.py**
   - Remove "src_port" and "dst_port" from FEATURE_NAMES
   - Update FEATURE_DIM assertion (12 → 10)
   - Update feature indices

2. **src/data_loader/feature_transforms.py**
   - Update feature extraction logic
   - Remove port extraction
   - Update normalization indices

3. **src/data_loader/normalization.py**
   - Recompute mean/std on 10-feature format
   - Update z-score baseline computation
   - Regenerate on CTU-13 multi-family data

4. **src/models/transformer.py**
   - Input shape validation: 12 → 10
   - No architecture changes (keep 4 attention heads, embedding_dim=256, etc.)
   - Test forward pass with new dimension

5. **src/data_loader/torch_dataset.py**
   - Update feature dimension check
   - Verify masking logic still works

6. **tests/test_*.py**
   - Update feature dimension assertions
   - Update test data shapes

### Files That Don't Change

- ✅ src/models/transformer.py (architecture unchanged)
- ✅ src/training/train_transformer.py (training loop unchanged)
- ✅ src/evaluation/evaluate_transformer.py (metrics unchanged)
- ✅ Session-centric representation (keep as-is)

---

## Rationale for Port Removal

### Previous Ablation Study Findings
```
Model with all 12 features (including ports):
  - UWF (seen): 100% recall on training families
  - CTU-13 (unseen): 0.19% recall on new families
  - Conclusion: Model memorized port patterns specific to UWF

Hypothesis:
  - Each malware family uses distinctive port patterns
  - Model learned "Neris uses X ports, Kraken uses Y ports"
  - When tested on Conficker, port patterns don't match
  - Result: Massive generalization failure
```

### Why This Helps Generalization
```
Port numbers are:
  ✅ Family-specific (changes across malware families)
  ✅ Attacker-controlled (can be randomized)
  ❌ Not behavioral invariant (not generalizable)

Behavioral invariants (what we keep):
  ✅ Traffic patterns (bytes, packets, duration)
  ✅ Flow characteristics (byte ratio, packet ratio)
  ✅ Timing (inter-arrival time, duration)
  ✅ Direction (outbound vs inbound)
  ✅ Load distribution (bytes_per_pkt, packet_ratio)
```

---

## Implementation Schedule

### Week 1: Feature Update
- **Day 1**: Document current schema (this file)
- **Days 2-3**: Load CTU-13 data without ports
- **Day 4**: Update feature extraction pipeline
- **Day 5**: Regenerate normalization on CTU-13

### Week 2: Verification
- Test forward pass with new dimensions
- Verify no data leakage
- Compare with old schema results

### Week 3: Training
- Train on 10-feature multi-family data
- Evaluate on zero-shot family
- Compare results with old schema

---

## Backward Compatibility

⚠️ **This change breaks compatibility with Phase 2 models**:
- Old checkpoints trained on 12 features won't work with 10-feature pipeline
- New checkpoints will only work with 10-feature data
- **Action**: Save Phase 2 models as "baseline_12_features" for reference

---

## Verification Checklist

### Day 1 (May 13)
- [ ] Read this document
- [ ] Understand why ports are removed
- [ ] List the 10 features in new order

### Days 2-4 (May 13-15)
- [ ] Load CTU-13 without ports
- [ ] Verify feature shapes: (N, 20, 10)
- [ ] Compute feature statistics
- [ ] Check no NaN/Inf values

### Days 5+ (May 15+)
- [ ] Regenerate normalization on CTU-13
- [ ] Update all feature_config.py assertions
- [ ] Update test data shapes
- [ ] Run test suite: all tests should pass

---

## Success Criteria

✅ **Feature update successful if**:
- All tests pass with new dimension
- No data shape mismatches
- Normalization computed on 10 features
- Model trains without shape errors
- Evaluation metrics computed correctly

❌ **Feature update failed if**:
- Shape mismatches during training
- Unknown NaN values in features
- Normalization breaks
- Tests fail on dimension assertions

---

## References

- **Previous ablation study**: experiments/ablation_smoke/ 
- **Phase 2 baseline**: experiments/baseline/
- **Decision rationale**: research_plan_multifamily_generalization.md
- **Implementation guide**: IMMEDIATE_ACTION_PLAN.md (Week 1, Days 4-5)

---

**Status**: ✅ Documented, ready for implementation on May 13  
**Created**: May 9, 2026  
**Implements**: research_plan_multifamily_generalization.md § Feature Schema Update

