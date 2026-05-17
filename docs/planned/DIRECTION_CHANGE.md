# 🔴 CRITICAL: Research Direction Correction

**Date**: May 9, 2026  
**Status**: Phase 3 Planning INVALIDATED | New Research Plan ACTIVE  
**Scientific Review**: Conclusion was premature - experiment was confounded

---

## What Happened

### Phase 3 Planning Documents (MARKED INVALID)
The following 7 documents created earlier are **NO LONGER VALID**:

```
docs/phase3_index.md                          ❌ INVALID
docs/phase3_quickstart_guide.md                ❌ INVALID
docs/phase3_scientific_justification.md        ❌ INVALID
docs/phase3_architectural_redesign_plan.md     ❌ INVALID
docs/phase3_implementation_skeleton.md         ❌ INVALID
docs/phase3_implementation_roadmap.md          ❌ INVALID
docs/phase3_status_summary.md                  ❌ INVALID
```

### Why They're Invalid

**Root Cause**: Confounded experimental design in zero-shot validation

```
Previous Experiment:
  Training data: UWF (22 C2 samples, 17 after split)
  Test data: CTU-13/MCFP (1.55M samples, 3 malware families)
  Result: 0.19% recall (526x failure)
  
Variables Changed Simultaneously:
  ✗ Dataset (UWF → CTU-13/MCFP)
  ✗ Sample count (22 → 1.55M)
  ✗ Family diversity (1 lab environment → 3 real families)
  ✗ Traffic distribution (lab → wild captures)
  
Attribution Error:
  Claimed: "Session-centric architecture fails"
  Actual possibilities:
    - Sample starvation (22 training samples is tiny)
    - Family memorization (model never saw CTU families)
    - Feature failure (not architecture failure)
    - Representation failure (not architecture failure)
```

**Scientific Principle Violated**: "Change ONE major variable at a time"

---

## What We're Doing Now

### New Research Plan: Multi-Family C2 Generalization

**Location**: `docs/research_plan_multifamily_generalization.md`

**Core Question**: 
"Do session-level behavioral flow features capture malware-family-invariant C2 behavior when trained on sufficient multi-family diversity?"

**Experimental Design**:
```
Phase 1: Build multi-family training dataset
         CTU-13 Scenario 1 (Neris) + Scenario 2 (Kraken) + Scenario 9 (Conficker)
         → ~300K+ C2 sessions, 3 distinct families
         → NO UWF (lab data confounds family effect)

Phase 2: Remove ports from features (likely family-specific memorization)
         Keep behavioral invariants: duration, bytes, cadence, symmetry

Phase 3: Keep Transformer architecture UNCHANGED
         Only variable we change is training data diversity

Phase 4: Strict family-aware evaluation
         Train on Neris + Kraken
         Test on Conficker (zero-shot family generalization)

Phase 5: Use appropriate metrics for imbalanced data
         Primary: PR-AUC, Recall, Precision, FPR (NOT accuracy)

Phase 6: Detailed error analysis
         Identify systematic failures by family and behavior

Phase 7: Decision logic based on outcomes
         Strong generalization → session-centric validated
         Partial generalization → host-centric justified
         Complete failure → major redesign needed
```

**Key Principle**: We measure first, then decide. Not speculative.

---

## Why This Matters

### Previous Approach (WRONG)
```
"Model fails on cross-family validation"
↓
"Therefore, architecture must change"
↓
Redesign to host-centric [Months of work]
↓
Find that it doesn't help either [Data starvation was the real issue]
```

### New Approach (CORRECT)
```
"Model fails on cross-family validation"
↓
"Isolate why: data starvation? family memorization? features?"
↓
"Fix ONE variable at a time"
↓
"Measure results with proper metrics"
↓
"Make architectural decision based on data"
```

---

## Immediate Tasks

### Task 1: Clean Repository
- [ ] Review codebase for any host-centric fragments
- [ ] Remove unfinished migration code
- [ ] Ensure session-centric pipeline is pristine
- [ ] Create clean branch: `research/multifamily_generalization`

### Task 2: Build Multi-Family Dataset
- [ ] Load CTU-13 Scenario 1 (Neris)
- [ ] Load CTU-13 Scenario 2 (Kraken)
- [ ] Load CTU-13 Scenario 9 (Conficker)
- [ ] Consolidate into unified format with family labels

### Task 3: Feature Refinement
- [ ] Remove src_port and dst_port from feature schema
- [ ] Keep behavioral invariants: duration, bytes, packets, cadence, etc.
- [ ] Update all feature processing pipelines
- [ ] Regenerate normalization baselines

### Task 4: Family-Aware Split Generator
- [ ] Create split that prevents family leakage
- [ ] Train on: Neris + Kraken
- [ ] Test on: Conficker (true zero-shot family)
- [ ] Generate split manifest for reproducibility

### Task 5: Run Controlled Experiment
- [ ] Train model on multi-family dataset (300K+ C2 samples)
- [ ] Evaluate with proper metrics (PR-AUC, Recall, Precision, FPR)
- [ ] Generate per-family analysis
- [ ] Create detailed error reports

### Task 6: Interpret Results
- [ ] Measure generalization across families
- [ ] Identify systematic failure modes (if any)
- [ ] Apply decision logic (Strong/Partial/Complete)
- [ ] Plan next phase based on outcomes

---

## What Changes, What Doesn't

### Changes
```
✅ Training data: UWF → CTU-13 multi-family (~300K C2 samples)
✅ Features: 12 features → 10 features (remove ports)
✅ Evaluation strategy: family-aware splits + per-family metrics
✅ Error analysis: detailed failure breakdown by family
```

### Doesn't Change
```
✅ Transformer architecture: IDENTICAL
✅ Session-centric representation: UNCHANGED
✅ Feature processing pipeline: SAME (just different features)
✅ Model size/hyperparameters: SAME
```

---

## Success Criteria

### We'll Know It's Working When:

**Outcome 1: Strong Generalization** ✅ Session-centric validated
```
- Cross-family recall ≥ 75%
- Conficker zero-shot recall ≥ 70%
- Consistent performance across all families
- Conclusion: No architectural redesign needed
```

**Outcome 2: Partial Generalization** ⚠️ Session-centric limited
```
- Cross-family recall 50-75%
- Systematic gaps by protocol/timing
- Clear failure patterns (not random)
- Conclusion: Host-centric modeling justified
```

**Outcome 3: Complete Failure** ❌ Session-centric insufficient
```
- Cross-family recall <50%
- Per-family recall varies wildly
- No improvement with larger data
- Conclusion: Major redesign needed
```

---

## Critical Rules

### DO NOT
- ❌ Include UWF in training (without specific hypothesis)
- ❌ Change architecture before results
- ❌ Make feature changes mid-experiment
- ❌ Skip family-aware splits (will leak)
- ❌ Use accuracy as a metric
- ❌ Assume host-centric is needed

### DO
- ✅ Track every experiment with metadata
- ✅ Use PR-AUC, Recall, Precision, FPR (proper metrics)
- ✅ Create detailed error reports
- ✅ Let data answer the question
- ✅ Be prepared for any outcome
- ✅ Question your assumptions

---

## Timeline

```
Week 1 (May 13-17):     Dataset + feature processing + initial training
Week 2 (May 20-24):     Evaluation + error analysis
Week 3 (May 27-31):     Results interpretation + decision logic
```

---

## How to Proceed

### Read First
- [docs/research_plan_multifamily_generalization.md](research_plan_multifamily_generalization.md)

### Then Start
- Task 1: Clean repository
- Task 2: Multi-family dataset loading
- Task 3: Feature removal (ports)

### Don't Read (Outdated)
- ❌ phase3_index.md
- ❌ phase3_quickstart_guide.md
- ❌ phase3_scientific_justification.md
- ❌ phase3_architectural_redesign_plan.md
- ❌ phase3_implementation_skeleton.md
- ❌ phase3_implementation_roadmap.md
- ❌ phase3_status_summary.md

---

## Archive Note

The Phase 3 planning documents represent a common research mistake: **premature conclusion from confounded experiment**.

They're preserved for historical reference showing:
1. How confounded variables lead to wrong conclusions
2. Why experimental isolation is critical
3. The importance of changing one variable at a time
4. How to recover from methodological errors

**Lesson**: Always validate assumptions before architectural redesign.

---

## Summary

**Old Direction**: "Session-centric failed → redesign to host-centric"  
**Status**: ❌ INVALID (based on confounded experiment)

**New Direction**: "Did session-centric actually fail, or was it sample starvation?"  
**Status**: ✅ ACTIVE (controlled scientific experiment)

**Outcome**: Measure cross-family generalization with sufficient training data  
**Decision**: Architecture change only if experiment shows it's needed

**Start**: Read research_plan_multifamily_generalization.md today

