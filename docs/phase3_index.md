# CyberShield Phase 3: Complete Planning Package

**Date**: May 9, 2026  
**Status**: ✅ Planning Complete | ⏳ Ready for Implementation  
**Objective**: Fix zero-shot generalization failure via host-centric longitudinal modeling

---

## 📚 What's Been Created

This folder now contains a **complete Phase 3 planning suite** with 5 integrated documents:

### Quick Navigation

| Document | Size | Read Time | Start Here If You... |
|----------|------|-----------|----------------------|
| **[phase3_quickstart_guide.md](phase3_quickstart_guide.md)** | 6 KB | 10 min | 🆕 Want the TL;DR / starting today |
| [phase3_scientific_justification.md](phase3_scientific_justification.md) | 8 KB | 15 min | Want to understand WHY |
| [phase3_architectural_redesign_plan.md](phase3_architectural_redesign_plan.md) | 12 KB | 20 min | Want to understand WHAT |
| [phase3_implementation_skeleton.md](phase3_implementation_skeleton.md) | 20 KB | 25 min | Want to understand HOW |
| [phase3_implementation_roadmap.md](phase3_implementation_roadmap.md) | 14 KB | 15 min | Want to understand WHEN |

---

## The Problem in 30 Seconds

**Zero-shot validation failure:**
```
Train on: UWF (22 C2 samples)
Test on: CTU-13/MCFP (1.55M samples, unseen families)

Results:
  ✅ Validation (UWF): 100% recall
  ❌ Test (CTU-13/MCFP): 0.19% recall  
  
Issue: 526x recall collapse = model learned UWF shortcuts, not universal patterns
```

---

## The Solution in 30 Seconds

**Architectural redesign: Session-centric → Host-centric**

| Aspect | Current (❌) | Proposed (✅) |
|--------|-------------|----------------|
| **Training Unit** | Individual session | Host's behavioral window (30 sessions) |
| **Input Shape** | (N, 20, 12) | (N, seq_len, 20, 12) |
| **What Model Sees** | Isolated flow statistics | Temporal host evolution |
| **C2 Signal** | Local byte patterns | Persistence + regularity |
| **Cross-family? | No (UWF-specific) | Yes (universal C2 properties) |

**Expected improvement**: 0.19% → 75%+ recall (400x+)

---

## Why This Will Work

### Modern C2 Reveals Itself Over Time

```
Session 1: "Looks like TLS connection"
Session 2: "Looks like admin SSH"
Session 3: "Looks like Windows Update"
Session N: "Looks like benign traffic"

Each session: ⚠️ Inconclusive (50% benign-like)

But: SAME destination, SAME pattern, EVERY 4 hours
    → Pattern over time: ✅ OBVIOUS C2 (99% malicious)
```

### Universal Across C2 Families
- **Conficker** (CTU-13): Beacons to same server regularly ✓
- **Alureon** (CTU-13): Persistent reconnection pattern ✓
- **Sliver** (modern C2): Heartbeat callbacks ✓
- **Any remote-control malware**: Must be persistent ✓

**These properties are invariant—not dataset-specific artifacts.**

---

## What Gets Built

### 4 Core Components (Phase 3.1-3.3)

```python
# 1. Behavioral Window Extraction
from src.features.behavioral_window import BehavioralWindowBuilder
windows = BehavioralWindowBuilder().build_windows(timelines, window_size=30)
# Output: List[BehavioralWindow] with 10K+ chronological windows

# 2. Temporal Dataset
from src.data_loader.temporal_dataset import HostBehavioralDataset
dataset = HostBehavioralDataset(windows, session_lookup)
X, mask, y = dataset[0]  # (50, 20, 12), (50,), 1

# 3. Modified Model
# Forward pass: accepts (batch, seq_len, 20, 12) instead of (batch, 20, 12)

# 4. Temporal Training Loop
from src.training.train_temporal_behavioral import train_temporal_behavioral_model
train_temporal_behavioral_model(...)
# Output: Model trained on host-centric longitudinal behavior
```

---

## Reading Path by Role

### 👨‍💼 Project Manager / Tech Lead
1. Read: [phase3_quickstart_guide.md](phase3_quickstart_guide.md) (10 min)
2. Skim: [phase3_implementation_roadmap.md](phase3_implementation_roadmap.md) (15 min)
3. Review: Success metrics and timeline
4. **Action**: Allocate 2 devs for 4 weeks, start Monday

### 👨‍🔬 Data Scientist / Researcher
1. Read: [phase3_scientific_justification.md](phase3_scientific_justification.md) (15 min)
2. Read: [phase3_architectural_redesign_plan.md](phase3_architectural_redesign_plan.md) (20 min)
3. **Action**: Review C2 properties, validate assumptions with domain knowledge

### 👨‍💻 Developer (Implementation)
1. Read: [phase3_quickstart_guide.md](phase3_quickstart_guide.md) (10 min)
2. Read: [phase3_implementation_skeleton.md](phase3_implementation_skeleton.md) (25 min)
3. Follow: [phase3_implementation_roadmap.md](phase3_implementation_roadmap.md) Week 1 tasks
4. **Action**: Start with BehavioralWindow class, follow skeleton

### 🔍 QA / Code Reviewer
1. Read: [phase3_implementation_skeleton.md](phase3_implementation_skeleton.md) - Testing section
2. Read: [phase3_implementation_roadmap.md](phase3_implementation_roadmap.md) - Success criteria
3. **Action**: Create test plan, review checklist

---

## Key Metrics to Track

### Primary Success Metrics (Must Achieve)
```
✅ Zero-shot recall on CTU-13:      75%+ (vs. 0.19% baseline)
✅ Zero-shot recall on MCFP:        70%+ (vs. 0.19% baseline)
✅ FPR @ target recall:             <2%
✅ Improvement factor:              >400x vs. baseline
```

### Secondary Indicators
```
✅ Validation AUC:                  0.95+
✅ Per-host detection accuracy:     85%+
✅ Cross-family generalization:     Uniform performance
✅ Training time:                   <4 hours/source
```

---

## Timeline Overview

### Phase 3 Execution (4 weeks)

| Week | Focus | Deliverable | Go/No-Go |
|------|-------|-------------|----------|
| **1** | Foundation | BehavioralWindow + HostBehavioralDataset | Windows extract properly |
| **2** | Model | Temporal Transformer + training loop | Model accepts (batch, seq, 20, 12) |
| **3** | Validation | Zero-shot training & evaluation | Recall >75% on CTU-13/MCFP |
| **4** | Integration | Pipeline + docs + optimization | Production-ready code |

### How to Use the Roadmap
- 📋 Weekly standup: Check roadmap against progress
- 🎯 Daily tasks: Current week's checklist
- ⚠️ Risk mitigation: Review weekly mitigations
- ✅ Completion: Archive as Phase 3 record

---

## Critical Design Principles (DON'T SKIP)

### 1. Chronological Order is Sacred
- Windows must preserve session timestamp order
- No shuffling, no reordering, ever
- This is why it works—temporal context is everything

### 2. No Future Information
- Window at time T cannot see data from time T+1
- Otherwise: easy to overfit but fails on real deployment
- Enforce strict temporal separation in splits

### 3. Masking for Variable Length
- Not all windows have 30 sessions
- Pad to max_seq_len, use mask to hide padding
- Model must only attend to real sessions

### 4. No Train/Test Leakage
- Same host can appear in train AND test (different time windows)
- But temporal windows must not overlap
- Verify: window_end_ts < next_window_start_ts

---

## Files to Create

### Definite (Week 1-3)
```
src/features/behavioral_window.py           ← BehavioralWindow + Builder
src/data_loader/temporal_dataset.py         ← HostBehavioralDataset + DataLoaders
src/training/train_temporal_behavioral.py   ← Training orchestrator
src/pipelines/train_host_centric_zero_shot.py  ← Main pipeline
```

### Test Files
```
tests/test_behavioral_window.py
tests/test_temporal_dataset.py
tests/test_temporal_pipeline.py
```

### Reference (Read, Don't Modify)
```
src/features/host_timeline.py               ✅ Already exists
src/models/transformer.py                   (Minor modification)
```

---

## Prerequisites Check

Before starting implementation:

```bash
# ✅ Do these exist?
ls -lh data/processed/v2_uwf/train_host_timelines.pkl
ls -lh data/processed/v2_uwf/val_host_timelines.pkl
ls -lh data/processed/v2_mixed_session_only/test_sessions.npz

# ✅ Can you load them?
python -c "import pickle; pickle.load(open('data/processed/v2_uwf/train_host_timelines.pkl', 'rb')); print('✅ Timelines load')"

# ✅ Are tests passing?
pytest tests/ -q

# ✅ Is git ready?
git status  # Clean working directory?
```

---

## Getting Started (Right Now)

### Step 1: Understand the Problem (15 min)
```
Read: phase3_scientific_justification.md
Goal: Know WHY we're doing this
```

### Step 2: Understand the Solution (20 min)
```
Read: phase3_architectural_redesign_plan.md
Goal: Know WHAT we're building
```

### Step 3: Understand the Implementation (25 min)
```
Skim: phase3_implementation_skeleton.md (focus on BehavioralWindow)
Goal: Know HOW to code it
```

### Step 4: Plan the Work (15 min)
```
Read: phase3_implementation_roadmap.md - Week 1 section
Goal: Know WHEN and IN WHAT ORDER
```

### Step 5: Create Feature Branch
```bash
git checkout -b feature/phase3-temporal-modeling
git push -u origin feature/phase3-temporal-modeling
```

### Step 6: Start Implementation
```bash
# Create empty file
touch src/features/behavioral_window.py

# Copy code skeleton from implementation_skeleton.md
# Start with BehavioralWindow class
```

---

## Document Cross-References

### From Scientific Justification
- **Key claim**: C2 is a longitudinal problem
- **Evidence**: CTU-13 Conficker beacons to same IP every 4 hours
- **Implication**: Session-level statistics insufficient
- **Link to**: Architectural redesign shows new components needed

### From Architectural Redesign
- **Component 1**: BehavioralWindow (extracts temporal structure)
- **Component 2**: HostBehavioralDataset (PyTorch integration)
- **Component 3**: LongitudinalFeatureExtractor (persistence metrics)
- **Component 4**: Train loop (orchestrates everything)
- **Link to**: Implementation skeleton shows exact code

### From Implementation Skeleton
- **BehavioralWindow**: Full code, 150+ lines, ready to copy
- **HostBehavioralDataset**: Full code, 120+ lines, ready to copy
- **Migration path**: Shows how to evolve from session-centric
- **Link to**: Roadmap shows when/how to test each component

### From Implementation Roadmap
- **Week 1**: Foundation layer (BehavioralWindow + Dataset)
- **Week 2**: Model adaptation (temporal transformer)
- **Week 3**: Zero-shot validation (train & evaluate)
- **Week 4**: Integration (pipeline + docs)
- **Link to**: Quickstart shows how to start Week 1

### From Quickstart
- **Problem**: 0.19% recall on held-out sources
- **Solution**: Host-centric longitudinal model
- **Success**: 75%+ recall (300x+ improvement)
- **Starting point**: Week 1 tasks from roadmap

---

## FAQ: "Why Do I Need 5 Documents?"

| Question | Document | Answer |
|----------|----------|--------|
| What's the high-level idea? | Scientific Justification | C2 reveals itself over time |
| What components do I build? | Architectural Redesign | 4 components with specific roles |
| What does the code look like? | Implementation Skeleton | Full Python skeletons ready to copy |
| When do I do it? | Implementation Roadmap | Week-by-week breakdown |
| Which one do I read first? | Quickstart Guide | Start here, links to others |

---

## Success: What It Looks Like

### After Week 1
- ✅ Behavioral windows extracted from timelines
- ✅ 10K+ windows available for training
- ✅ PyTorch dataloaders working
- ✅ All tests passing

### After Week 2
- ✅ Model accepts (batch, seq_len, 20, 12) input
- ✅ Training loop processes temporal batches
- ✅ Dummy training step completes

### After Week 3
- ✅ Model trained on UWF windows: 85%+ validation recall
- ✅ Zero-shot evaluation complete: 75%+ CTU-13 recall
- ✅ 400x+ improvement over baseline

### After Week 4
- ✅ Everything integrated and documented
- ✅ Team can run full pipeline independently
- ✅ Codebase ready for production

---

## Contact Points & Decisions

### Decision 1: Window Size
**Default**: 30 sessions/window  
**Why**: Balances temporal context with computational efficiency  
**Tunable**: Can vary 10-50 based on early results

### Decision 2: Sequence Length (max_seq_len)
**Default**: 50 sessions (pad to this)  
**Why**: Covers 95% of windows without excess padding  
**Tunable**: Adjust based on histogram of actual window sizes

### Decision 3: Model Architecture
**Start with**: Flatten seq_len × 20 flows as one sequence (Option A)  
**Why**: Simpler, faster to implement  
**Upgrade to**: Hierarchical attention later if needed (Option B)

### Decision 4: Longitudinal Features
**Phase 3**: Start with persistence + regularity  
**Phase 4**: Add partner concentration, behavioral drift, anomaly features  
**Why**: Prioritize simple features first, validate assumptions

---

## Next Steps (Rank Order)

### 🔴 Critical Path
1. ✅ Create planning documents (DONE - you are here)
2. ⏳ Implement BehavioralWindow class (Week 1)
3. ⏳ Implement HostBehavioralDataset class (Week 1)
4. ⏳ Run zero-shot validation with temporal model (Week 3)

### 🟡 Important but Not Blocking
- Performance optimization (Week 4)
- Comprehensive ablations (Week 4)
- Documentation/writeup (Week 4)

### 🟢 Nice to Have (Post-Phase 3)
- Hierarchical attention model
- Advanced longitudinal features
- Cross-dataset benchmarking

---

## How to Update These Documents

If you discover something during implementation:

1. **Bug in skeleton**: Update `phase3_implementation_skeleton.md`
2. **Timeline wrong**: Update `phase3_implementation_roadmap.md`
3. **New insight**: Add to `phase3_scientific_justification.md`
4. **Lessons learned**: Create `phase3_weekly_progress.md`

**Keep docs in sync with reality—they're your roadmap.**

---

## Final Checklist Before Starting

- [ ] Read phase3_quickstart_guide.md (10 min)
- [ ] Read phase3_scientific_justification.md (15 min)
- [ ] Skim phase3_implementation_skeleton.md - BehavioralWindow section (10 min)
- [ ] Verify prerequisites: Timelines exist and load? (5 min)
- [ ] Create feature branch: `feature/phase3-temporal-modeling` (2 min)
- [ ] Create empty files for Week 1 components (2 min)
- [ ] Bookmark these 5 documents for reference (1 min)
- [ ] Schedule team standup to review roadmap (5 min)

**Total prep time: ~50 minutes**

**Then: Start Week 1 implementation** 🚀

---

## Archive & Reference

**These documents will become:**
- Onboarding material for new team members
- Reference for Phase 4+ architectural decisions
- Case study on architectural redesign for C2 detection
- Foundation for follow-up work (cross-family validation, feature engineering, etc.)

**Keep them in `docs/` folder for permanent reference.**

---

## Your Role Right Now

- 🔵 **If you're a developer**: Start Phase 3.1 (Week 1) following the skeleton
- 🟠 **If you're a manager**: Allocate resources, schedule kickoff, review roadmap
- 🟡 **If you're a researcher**: Validate scientific assumptions, design ablations
- 🟣 **If you're a reviewer**: Prepare test plan, acceptance criteria

---

**Everything you need is in this folder. Questions? Reference the documents. Ready to code? Follow the skeleton. Ready to test? Follow the roadmap. Ready to understand? Read the justification.**

**You've got this. Phase 3 begins now.** 💪

---

*Phase 3 Planning Package*  
*Created: May 9, 2026*  
*Status: ✅ Complete and Ready for Implementation*

