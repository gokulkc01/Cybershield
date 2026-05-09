# Phase 3: Quick-Start Guide

**Status**: ✅ Complete Planning Phase | ⏳ Ready for Implementation  
**Date**: May 9, 2026  
**Your Next Action**: Implement Week 1 (BehavioralWindow extraction)

---

## What Happened & Why You're Here

### The Problem
- ❌ **Zero-shot recall**: 0.19% (trained on UWF, tested on CTU-13/MCFP)
- ✅ **Validation recall**: 100% (perfect on UWF training data)
- **Diagnosis**: Session-centric model learned UWF-specific shortcuts, not universal C2 patterns

### The Solution
- Redesign from **session classification** → **host-centric longitudinal modeling**
- Current: X.shape = (N, 20, 12) — one session per sample
- New: X.shape = (N, seq_len, 20, 12) — one host's behavioral window per sample
- **Expected**: Zero-shot recall 75%+ (300x+ improvement)

### Why It Will Work
Modern C2 reveals itself through persistent behavioral patterns over time:
- Same destination repeatedly (Conficker: same C2 server for days)
- Regular reconnection intervals (beaconing every 4 hours)
- Consistent byte patterns (session profiles stable)

**These properties are universal across C2 families—not UWF-specific.**

---

## You Now Have 4 Planning Documents

### 📋 Read These IN ORDER

| # | Document | Read Time | Purpose | Key Takeaway |
|---|----------|-----------|---------|--------------|
| 1 | [phase3_scientific_justification.md](phase3_scientific_justification.md) | 15 min | **Why** this works | C2 is a longitudinal problem, not session-level |
| 2 | [phase3_architectural_redesign_plan.md](phase3_architectural_redesign_plan.md) | 20 min | **What** needs to change | 4 new components: Windows, Dataset, Extractor, Training loop |
| 3 | [phase3_implementation_skeleton.md](phase3_implementation_skeleton.md) | 25 min | **How** to implement it | Concrete Python code structure and full skeletons |
| 4 | [phase3_implementation_roadmap.md](phase3_implementation_roadmap.md) | 15 min | **When** to do it | Week-by-week breakdown, success criteria |

**Total reading time: ~75 minutes**

---

## The Architecture in 60 Seconds

### Current (Session-Centric - ❌ FAILS)
```
Training Data:
  Session 1: [features] → Label: C2
  Session 2: [features] → Label: Benign
  ...
  Session 8000: [features] → Label: C2

Model learns: "What local flow features look like C2?"
  → Perfect on UWF (learned UWF statistics)
  → Fails on CTU-13 (different family statistics)
```

### New (Host-Centric - ✅ SUCCEEDS)
```
Training Data:
  Host A's 30-day behavior:
    - Session 1 (Day 1, 10:00 AM)
    - Session 2 (Day 1, 2:00 PM)
    - Session 3 (Day 1, 6:00 PM)
    ... → Label: C2 (persistent, regular pattern)
  
  Host B's 30-day behavior:
    - Session 1 (Day 1, random time)
    - Session 2 (Day 3, different time)
    - Session 3 (Day 5, different dest)
    ... → Label: Benign (irregular, varied)

Model learns: "What host patterns are C2?"
  → Works on UWF (learned behavioral invariants)
  → Works on CTU-13 (same invariants apply)
  → Will work on unknown malware (C2 requires persistence)
```

---

## 4 Core Components You'll Build

### 1️⃣ BehavioralWindow (`src/features/behavioral_window.py`)
**What**: Extract rolling windows from host timelines  
**Why**: Convert temporal host data into training samples  
**Input**: `Dict[host_id → HostTimeline]`  
**Output**: `List[BehavioralWindow]` (chronologically ordered)  
**Key metric**: `regularity` (CV of inter-session gaps, 0-1 scale)

### 2️⃣ HostBehavioralDataset (`src/data_loader/temporal_dataset.py`)
**What**: PyTorch Dataset for behavioral windows  
**Why**: Dataloaders need to iterate over temporal sequences  
**Input**: `List[BehavioralWindow]` + session tensor lookup  
**Output**: Batches of shape (batch, seq_len, 20, 12)  
**Key feature**: Masking for variable-length windows

### 3️⃣ Temporal Model (modify `src/models/transformer.py`)
**What**: Update forward pass to handle (batch, seq_len, 20, 12)  
**Why**: Model must see chronological session sequences  
**Change**: Accept masking, process sequence dimension  
**Effort**: ~30 lines of code

### 4️⃣ Training Loop (`src/training/train_temporal_behavioral.py`)
**What**: New training orchestrator using behavioral windows  
**Why**: Coordinates everything: load timelines → extract windows → train  
**Process**: Load timelines → build windows → create dataloaders → train  
**Output**: Trained model with zero-shot validation results

---

## Your Week 1 Task: Foundation Layer

### What You'll Do
1. Create `src/features/behavioral_window.py`
   - Implement `BehavioralWindow` dataclass
   - Implement `BehavioralWindowBuilder` class
   - Extract windows from timelines (preserving chronological order)

2. Create `src/data_loader/temporal_dataset.py`
   - Implement `HostBehavioralDataset` (PyTorch Dataset)
   - Handle variable-length sequences with masking
   - Create dataloaders

3. Verify on real data
   - Load UWF timelines
   - Extract 10,000 windows
   - Inspect window statistics

### Success Looks Like
```python
# Load timelines
timelines = load_host_timelines("data/processed/v2_uwf/train_host_timelines.pkl")

# Extract windows
windows = BehavioralWindowBuilder().build_windows(timelines, window_size=30)
# Returns: List[BehavioralWindow] with 10,000+ windows

# Create dataset
dataset = HostBehavioralDataset(windows, session_lookup)

# Get one sample
X, mask, y = dataset[0]
print(X.shape)      # (50, 20, 12) — chronological session sequence
print(mask.shape)   # (50,) — which positions are real vs. padding
print(y)            # 1 (C2) or 0 (benign)
```

### Time Estimate
- Implementation: 6-8 hours
- Testing: 2-3 hours
- Total: ~10 hours (1.5 days)

---

## Key Files Already Exist (Don't Reinvent)

✅ **You can use these directly:**
- `src/features/host_timeline.py` — HostTimeline, SessionSummary classes
- `src/features/dataset_builder_v2.py` — load_multi_source, build_host_timelines_from_sessions
- `src/data_loader/torch_dataset.py` — Reference for PyTorch patterns
- `data/processed/v2_uwf/train_host_timelines.pkl` — Existing host timelines

✅ **These have what you need:**
- Session timelines with chronological ordering
- SessionSummary with metadata (start_ts, end_ts, label, etc.)
- Examples of PyTorch Dataset patterns

---

## Critical Design Principles (Don't Skip!)

### 1. Chronological Order is EVERYTHING
```python
# ✅ CORRECT: Sessions in time order
window.session_summaries = [
    SessionSummary(start_ts=1000),
    SessionSummary(start_ts=2000),
    SessionSummary(start_ts=3000),
]

# ❌ WRONG: Random order
window.session_summaries = [
    SessionSummary(start_ts=2000),
    SessionSummary(start_ts=1000),  # Out of order!
    SessionSummary(start_ts=3000),
]
```

### 2. No Future Information
```python
# ✅ CORRECT: Window uses sessions up to time T
window_start_ts = 1000
window_end_ts = 5000
sessions = [s for s in host.sessions if s.start_ts <= 5000]

# ❌ WRONG: Window sees future sessions
sessions = [s for s in host.sessions if s.start_ts <= 100000]  # Cheating!
```

### 3. Masking for Variable Length
```python
# ✅ CORRECT: Pad to max_seq_len, use mask
X = np.zeros((50, 20, 12))  # 50-session capacity
X[0:5] = real_sessions      # Real sessions
mask = [True]*5 + [False]*45  # Only 5 real, 45 padding

# ❌ WRONG: Different lengths per batch
batch_X = [np.array((..., 20, 12)), np.array((..., 20, 12))]  # Different shapes!
```

---

## Common Pitfalls & How to Avoid Them

| Pitfall | Impact | Prevention |
|---------|--------|-----------|
| Sessions shuffled in window | Loses temporal context | Sort by start_ts explicitly |
| Masking not passed to model | Model trains on padding | Pass mask through forward() |
| No temporal leakage check | Overfitting | Verify train/val/test host_ids don't overlap |
| Variable window sizes cause crashes | Training breaks | Pad to max_seq_len, use mask |
| Regularity computation fails on empty windows | NaN handling | Check window.session_count > 1 |

---

## Testing Checklist

Before claiming Week 1 is done:

```python
# Test 1: Windows are chronological
for window in windows:
    for i in range(1, len(window.session_summaries)):
        assert window.session_summaries[i].start_ts >= window.session_summaries[i-1].start_ts

# Test 2: Regularrity computed correctly
window = windows[0]
assert 0 <= window.regularity < float('inf'), f"Bad regularity: {window.regularity}"

# Test 3: Dataset returns correct shapes
dataset = HostBehavioralDataset(windows, lookup)
X, mask, y = dataset[0]
assert X.shape == (50, 20, 12), f"Wrong X shape: {X.shape}"
assert mask.shape == (50,), f"Wrong mask shape: {mask.shape}"
assert y in [0, 1], f"Wrong label: {y}"

# Test 4: Masking works on variable windows
for i in range(len(dataset)):
    X, mask, y = dataset[i]
    real_count = mask.sum()
    assert real_count > 0, "All-padding window!"
    assert real_count <= 50, "Overflow window!"

# Test 5: No temporal leakage
train_hosts = {w.host_id for w in train_windows}
test_hosts = {w.host_id for w in test_windows}
assert len(train_hosts & test_hosts) == 0, "Host appears in both train and test!"
```

---

## Success Metrics for Week 1

✅ **You'll know it's working when:**

1. **BehavioralWindow** extracts 10,000+ windows from UWF timelines
2. **Regularity** metric varies across windows (not all 0 or all NaN)
3. **Dataloaders** iterate without crashing
4. **Tensor shapes** are (batch, 50, 20, 12) consistently
5. **Masking** correctly identifies real vs. padding sessions
6. **All unit tests** pass

❌ **Red flags:**
- Windows not in chronological order
- NaN values in regularity
- Incorrect tensor shapes
- Crashes during dataloader iteration
- Tests failing

---

## Implementation Order (Dependency Graph)

```
1. BehavioralWindow class
   ↓
2. BehavioralWindowBuilder class
   ↓ (requires timelines)
3. Unit tests for both
   ↓ (should pass)
4. HostBehavioralDataset class
   ↓
5. create_temporal_dataloaders() function
   ↓
6. Unit tests for dataset
   ↓ (should pass)
7. Integration test: Load real timelines → extract windows → iterate dataloaders
```

**This order ensures each layer is tested before the next depends on it.**

---

## Quick Reference: File Locations

```
Implementation files to create:
  src/features/behavioral_window.py          ← Week 1
  src/data_loader/temporal_dataset.py        ← Week 1

Testing files to create:
  tests/test_behavioral_window.py            ← Week 1
  tests/test_temporal_dataset.py             ← Week 1

Reference files (read, don't modify):
  src/features/host_timeline.py              ✅ Exists
  src/features/dataset_builder_v2.py         ✅ Exists
  src/data_loader/torch_dataset.py           ✅ Exists (for patterns)

Data files:
  data/processed/v2_uwf/train_host_timelines.pkl
  data/processed/v2_uwf/val_host_timelines.pkl
  data/processed/v2_mixed_session_only/test_sessions.npz
```

---

## Questions to Ask Before Starting

1. **Are host timelines available?**
   ```bash
   ls -lh data/processed/v2_uwf/*timelines.pkl
   ```
   Expected: Files exist, >1MB each

2. **Can you load a timeline?**
   ```python
   import pickle
   with open('data/processed/v2_uwf/train_host_timelines.pkl', 'rb') as f:
       timelines = pickle.load(f)
   print(len(timelines), "hosts loaded")
   ```
   Expected: >1000 hosts

3. **Do SessionSummary objects have timestamps?**
   ```python
   host_id = list(timelines.keys())[0]
   timeline = timelines[host_id]
   first_session = timeline.sessions_received[0]
   print(first_session.start_ts, first_session.end_ts)
   ```
   Expected: Numeric timestamps (Unix epoch)

---

## Debugging Checklist If Stuck

| Problem | Debug Step |
|---------|-----------|
| Import errors | Verify file path matches Python imports |
| NaN in regularity | Check: do sessions have timestamps? |
| Tensor shape mismatch | Print X.shape, mask.shape, y.type for sanity |
| Dataloader hangs | Check: is your dataset finite? (test on tiny dataset) |
| Mask all False | Check: are there real sessions in the window? |

---

## What Success Means

After Week 1, you should be able to run:

```python
from src.features.behavioral_window import BehavioralWindowBuilder
from src.data_loader.temporal_dataset import create_temporal_dataloaders
import pickle

# Load timelines
with open('data/processed/v2_uwf/train_host_timelines.pkl', 'rb') as f:
    timelines = pickle.load(f)

# Extract windows
windows = BehavioralWindowBuilder().build_windows(timelines)
print(f"Extracted {len(windows)} behavioral windows")

# Create dataloaders
train_loader, val_loader, _ = create_temporal_dataloaders(
    windows[:1000], windows[1000:], None, session_lookup
)

# Iterate
for batch_X, batch_mask, batch_y in train_loader:
    print(f"Batch shapes: X={batch_X.shape}, mask={batch_mask.shape}, y={batch_y.shape}")
    print(f"Sample regularity: {windows[0].regularity:.3f}")
    break  # Just verify it works

print("✅ Week 1 complete!")
```

**That's your end-of-week celebration.**

---

## Next: Week 2 Preview

Once Week 1 is complete, Week 2 involves:
- Modify `src/models/transformer.py` to accept (batch, seq_len, 20, 12) input
- Add masking support to model forward pass
- Create training loop scaffolding
- Run a dummy training step to verify shapes flow correctly

**But that's next week.** This week: **build the foundation.**

---

## Let's Go! 🚀

**Your next action (right now):**

1. ✅ Read [phase3_scientific_justification.md](phase3_scientific_justification.md) (15 min)
2. ✅ Read [phase3_architectural_redesign_plan.md](phase3_architectural_redesign_plan.md) (20 min)
3. ✅ Skim [phase3_implementation_skeleton.md](phase3_implementation_skeleton.md) for BehavioralWindow code (10 min)
4. ✅ Create feature branch: `git checkout -b feature/phase3-temporal-modeling`
5. ✅ Create file: `src/features/behavioral_window.py` (empty for now)
6. ✅ Start implementing: Copy skeleton from implementation_skeleton.md

---

**You've got this. The science is sound. The infrastructure is ready. Let's fix the zero-shot problem.** 💪

