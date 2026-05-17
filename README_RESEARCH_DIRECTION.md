# README: Controlled Generalization Experiment (May 9, 2026)

**STATUS**: ✅ **Research direction corrected - ready to proceed**

---

## Quick Summary

**What Changed?**
- Previous Phase 3 planning was based on a **confounded experiment**
- UWF experiment changed multiple variables simultaneously
- Conclusion "session-centric failed" was scientifically premature
- **Action**: Revert to proper scientific method

**New Direction?**
- Run controlled experiment with ONE variable changing: **training data diversity**
- Train on multi-family C2 (Neris + Kraken: ~250K samples)
- Test on unseen family (Conficker: zero-shot evaluation)
- Keep architecture unchanged (session-centric Transformer)
- Keep features mostly unchanged (remove ports that cause memorization)
- **Question**: Do behavioral flow invariants generalize across families?

**Timeline?**
- Week 1 (May 13-17): Data preparation + feature updates
- Week 2 (May 20-24): Split generator + evaluation pipeline
- Week 3 (May 27-31): Train + evaluate + interpret results

---

## What You Need to Know

### The Problem (Previous Approach)

```
Training data:  22 C2 samples from UWF (single family, lab traffic)
Test data:      1.55M C2 samples from CTU-13/MCFP (3 families, wild traffic)
Result:         0.19% recall (526x failure)

What Changed Simultaneously:
  ✗ Dataset (UWF → CTU-13)
  ✗ Sample count (22 → 1.55M)
  ✗ Family diversity (1 family → 3 families)
  ✗ Traffic source (lab → wild captures)

What We Concluded:
  "Architecture must change" → Began Phase 3 host-centric redesign

What Was Actually Wrong:
  Could be any of:
  - Sample starvation (22 training samples is tiny)
  - Family overfitting (only saw one family)
  - Feature failure (not architecture failure)
  - Representation failure (not architecture failure)
  
Scientific Error:
  Changed multiple variables, then blamed one cause
```

### The Solution (New Approach)

```
Training data:  ~300K C2 samples from CTU-13 (3 families, controlled split)
                  - Neris (Scenario 1)
                  - Kraken (Scenario 2)
                  - Both in training

Test data:      ~150K C2 samples from CTU-13 (1 family, held-out)
                  - Conficker (Scenario 9)
                  - Never seen during training

What Changes:
  ✅ Dataset diversity (single family → multi-family)
  ✅ Sample count (22 → 300K+)
  ✅ Features (remove ports that cause memorization)

What Stays Same:
  ✅ Transformer architecture
  ✅ Session-centric representation
  ✅ Feature processing pipeline

Question Answered:
  "Do session-level behavioral features generalize across C2 families?"
  
Decision Logic:
  Strong generalization → Session-centric works ✓
  Partial generalization → Host-centric needed
  Complete failure → Major redesign needed
```

---

## Key Documents

| Document | Purpose | Read When |
|----------|---------|-----------|
| [docs/planned/research_plan_multifamily_generalization.md](docs/planned/research_plan_multifamily_generalization.md) | Full experimental design | Starting implementation |
| [docs/planned/IMMEDIATE_ACTION_PLAN.md](docs/planned/IMMEDIATE_ACTION_PLAN.md) | Week-by-week tasks | Planning your work |
| [docs/planned/DIRECTION_CHANGE.md](docs/planned/DIRECTION_CHANGE.md) | Why Phase 3 was invalid | Understanding the correction |
| ~~phase3_*.md~~ | **DEPRECATED - Do not use** | Reference only (historical) |

---

## What to Do Now

### Step 1: Understand (30 minutes)
- Read this README
- Read [docs/planned/DIRECTION_CHANGE.md](docs/planned/DIRECTION_CHANGE.md)
- Skim [docs/planned/research_plan_multifamily_generalization.md](docs/planned/research_plan_multifamily_generalization.md)

### Step 2: Setup (15 minutes)
```bash
cd d:\CyberShield
git checkout -b research/multifamily_generalization
git status  # Should be clean
```

### Step 3: Execute (3 weeks)
Follow [docs/planned/IMMEDIATE_ACTION_PLAN.md](docs/planned/IMMEDIATE_ACTION_PLAN.md):
- Week 1: Load multi-family dataset + remove ports
- Week 2: Family-aware splits + evaluation
- Week 3: Train + evaluate + interpret

---

## Scientific Principles (Non-Negotiable)

### 1. Change ONE Variable at a Time
```
✅ This experiment changes: training data diversity
❌ This experiment does NOT change: architecture, representation, model type
```

### 2. Proper Metrics for Imbalanced Data
```
✅ Use: PR-AUC, Recall, Precision, FPR
❌ Don't use: Accuracy
```

### 3. No Family Leakage
```
✅ Train on: Neris + Kraken (Scenarios 1, 2)
✅ Test on: Conficker (Scenario 9) — never seen before
❌ Never: Mix families in train and test
```

### 4. Data Drives Decisions
```
✅ Let experiment results guide next phase
❌ Don't assume results ahead of time
```

---

## Expected Outcomes & Next Steps

### If Strong Generalization (Recall ≥ 75% on Conficker)
```
Evidence: Session-level features capture behavioral invariants
Conclusion: Architecture is fine
Next Phase: Minor tuning, deployment prep
Host-centric redesign: NOT needed
```

### If Partial Generalization (Recall 50-75%)
```
Evidence: Session-level partly sufficient
Conclusion: Missing temporal/context information
Next Phase: Implement host-centric redesign (NOW with justification)
Host-centric redesign: Justified by data
```

### If Complete Failure (Recall <50%)
```
Evidence: Session-level fundamentally insufficient
Conclusion: Major rethinking needed
Next Phase: Comprehensive redesign
Host-centric redesign: Necessary, plus graph modeling?
```

**Key**: We don't know outcome yet. Data will tell us.

---

## What Changes from Phase 3 Planning

| Aspect | Phase 3 (Invalid) | New Approach |
|--------|-------------------|--------------|
| **Status** | ❌ INVALID | ✅ Active |
| **Question** | "Does architecture work?" | "Do features generalize?" |
| **Method** | Architectural redesign | Controlled experiment |
| **Variables Changed** | Multiple (confounded) | One (training diversity) |
| **Training Data** | 22 UWF samples | 300K+ CTU-13 multi-family |
| **Architecture** | Changed to host-centric | Unchanged (isolated variable) |
| **Timeline** | 4 weeks (wrong) | 3 weeks (proper) |
| **Outcome** | Premature conclusion | Data-driven decision |

---

## Why This Matters

### Common Research Mistake
```
Experiment fails
↓
Assume problem is X
↓
Redesign to fix X
↓
Find out problem was actually Y
↓
Wasted effort
```

### Correct Approach
```
Experiment fails
↓
Isolate which variable caused failure
↓
Change one variable at a time
↓
Measure each change
↓
Fix based on evidence
↓
Efficient progress
```

**This experiment follows the correct approach.**

---

## Important Notes

- **UWF is NOT used for training** (lab data confounds family effect)
- **Architecture stays identical** (we're testing features, not architecture)
- **Ports are removed** (previous ablation showed they cause memorization)
- **Metrics matter** (PR-AUC, not accuracy)
- **Error analysis required** (understand failure modes by family)
- **Results will guide next phase** (we don't assume outcome)

---

## Quick Reference: Files to Read

```
START HERE:
  1. This README (you are here)
  2. docs/planned/DIRECTION_CHANGE.md (why Phase 3 is invalid)

THEN:
  3. docs/planned/research_plan_multifamily_generalization.md (full design)
  4. docs/planned/IMMEDIATE_ACTION_PLAN.md (task breakdown)

DO NOT READ (deprecated):
  ✗ phase3_index.md
  ✗ phase3_quickstart_guide.md
  ✗ phase3_scientific_justification.md
  ✗ phase3_architectural_redesign_plan.md
  ✗ phase3_implementation_skeleton.md
  ✗ phase3_implementation_roadmap.md
  ✗ phase3_status_summary.md
```

---

## Questions to Ask Yourself

**Before starting implementation:**
- [ ] Do I understand why Phase 3 was invalid?
- [ ] Can I explain the confounded variables?
- [ ] Do I know what ONE variable we're changing?
- [ ] Can I list the three malware families?
- [ ] Do I know why ports are removed?
- [ ] Do I understand family-aware splits?
- [ ] Can I list the primary metrics (not accuracy)?

**If you answer NO to any, re-read the relevant document.**

---

## Timeline at a Glance

```
Week 1 (May 13-17)
  Mon-Fri: Load datasets, update features
  Goal: 300K+ multi-family dataset ready

Week 2 (May 20-24)
  Mon-Fri: Split generator, evaluation pipeline
  Goal: Reproducible experiment setup

Week 3 (May 27-31)
  Mon-Wed: Train model
  Thu: Evaluate on held-out family
  Fri: Interpret results + next steps

Post-experiment:
  Make architectural decision based on outcomes
  Begin Phase 4 (TBD based on results)
```

---

## Success Criteria

✅ **Experiment is successful if:**
- All three CTU-13 scenarios load correctly
- Feature schema updated (ports removed)
- Family-aware splits prevent leakage
- Model trains without errors
- All metrics computed
- Error analysis completes
- Results clearly indicate one of [Strong/Partial/Complete] generalization

❌ **Experiment failed if:**
- Data leakage between families
- Inconsistent metrics
- Unknown failure modes
- Ambiguous results

---

## Final Thought

> "We don't assume architectural failure. We measure it."

The previous Phase 3 planning was well-intentioned but premature. This experiment corrects course by isolating variables properly.

After 3 weeks, data will tell us whether to pursue host-centric modeling or focus on deployment.

**Let's get the science right.** 🔬

---

**Created**: May 9, 2026  
**Status**: ✅ Ready for Week 1 kickoff  
**Next Step**: Read docs/planned/DIRECTION_CHANGE.md

