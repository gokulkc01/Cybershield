# CyberShield v2 Architectural Transition: Session-Centric → Host-Centric Longitudinal

**Date**: May 9, 2026  
**Phase**: 3 - Core Architecture Redesign  
**Status**: DESIGN PHASE (pre-implementation)

---

## Executive Summary

CyberShield v2 has host timeline infrastructure built but is **still training on isolated sessions**, not on longitudinal behavioral sequences. This architectural mismatch explains:
- ❌ 0.19% recall on held-out CTU-13/MCFP sources (zero-shot failure)
- ✅ 100% recall on isolated UWF sessions (dataset-specific overfitting)

**The fix**: Redesign the training pipeline to use `HostTimeline` objects as training samples, not individual sessions.

---

## Current State Analysis

### ✅ What Exists (Infrastructure)

1. **Host Timeline Construction** (`src/features/host_timeline.py`)
   - `HostTimeline` class: complete behavioral history per host
   - `SessionSummary`: lightweight session metadata
   - `PartnerHistory`: relationship tracking between host and destination
   - `RollingStatistics`: exponential-decay behavioral metrics
   - Chronological session ordering preserved
   - ~1.2M hosts tracked with 1.55M sessions in mixed benchmark

2. **Timeline Generation** (`src/features/dataset_builder_v2.py`)
   - `build_host_timelines_from_sessions()`: Creates Dict[host_id → HostTimeline]
   - Metadata preservation: host_map.json, labels.csv
   - Per-host aggregation and partner tracking
   - Baseline deviation scoring

3. **Timeline Storage**
   - `train_host_timelines.pkl`, `val_host_timelines.pkl`, `test_host_timelines.pkl`
   - Already saved in `data/processed/v2_uwf/` and `data/processed/v2_mixed/`
   - Available but **not used for training**

### ❌ What's Missing (Training Layer)

1. **Behavioral Window Generation**
   - No rolling window extraction from timelines
   - No chronological sequence construction
   - No temporal continuity enforcement

2. **Timeline-Aware Dataset Classes**
   - Current: `C2SessionDataset` → one sample = one session
   - Needed: `HostBehavioralDataset` → one sample = behavioral window over time

3. **Temporal Modeling**
   - Current: Transformer doesn't see temporal host context
   - Needed: Model receives chronological session sequences for one host

4. **Longitudinal Feature Engineering**
   - Missing: Persistence detection
   - Missing: Reconnect pattern analysis
   - Missing: Behavioral drift tracking
   - Missing: Communication stability scoring

---

## Root Cause Analysis: Why Zero-Shot Failed

### Session-Centric Training (Current)

```
Session_A (UWF, Exfiltration)
  ↓
Normalize features: [bytes, packets, duration, ports, ...]
  ↓
Transformer (no context)
  ↓
Label: "C2" based on local statistics

Learned pattern:
  - UWF has high bytes/pkt ratio
  - UWF has specific port patterns
  - UWF has particular duration distributions
```

**When tested on CTU-13 (different malware families)**:
- CTU-13 Conficker: different byte patterns
- CTU-13 Alureon: different port usage
- CTU-13 poisoned traffic: different timing
→ **Model has never seen these families' local patterns** → Recall: 0.19%

### Host-Centric Longitudinal Training (Proposed)

```
Host_X Timeline over 24 hours:
  Session_1 (start_ts=T0)
  Session_2 (start_ts=T1)
  Session_3 (start_ts=T2)
  ... (chronological)
  ↓
Extract behavioral window: [S1, S2, S3, S4, S5]
  ↓
Per-session features + longitudinal context:
  - Persistence (how many sessions in 24h?)
  - Regularity (are reconnections predictable?)
  - Communication partner (is it always the same dest?)
  - Behavioral consistency (do metrics stay stable?)
  - Abnormality (do baselines spike?)
  ↓
Transformer (sees temporal host evolution)
  ↓
Label: "C2" based on persistence structure
```

**When tested on CTU-13**:
- CTU-13 Conficker: also shows persistence + regular beaconing
- CTU-13 Alureon: also shows behavioral consistency
- CTU-13 poisoned traffic: also shows abnormal patterns
→ **Model has learned universal C2 behavioral invariants** → Recall: 85%+ (projected)

---

## Key Architectural Differences

| Aspect | Session-Centric (Current) | Host-Centric Longitudinal (Proposed) |
|--------|--------------------------|-------------------------------------|
| **Input** | One session tensor | Chronological sequence of sessions |
| **Shape** | (N, 20, 12) | (N_hosts, seq_len, 20, 12) |
| **Time** | Ignored/implicit | Explicit timestamps between sessions |
| **Context** | None | Full host behavioral history |
| **Unit** | Flow aggregate | Host behavioral evolution |
| **C2 Signature** | Local statistics | Persistence + regularity |
| **Generalization** | Dataset-specific | Across C2 families |

---

## New Data Pipeline

### Stage 1: Timeline Construction (✅ DONE)
```
Raw telemetry
  ↓ (normalization)
Sessions
  ↓ (grouping by host_id)
Host Timelines (Dict[host_id → HostTimeline])
  ↓
Save: timelines.pkl
```

### Stage 2: Behavioral Window Extraction (❌ NEW)
```
Host Timelines
  ↓ (sliding window over chronological sessions)
Behavioral Windows
  ↓
Window Features:
  - window_id
  - host_id
  - label (from window sessions)
  - session_sequence (ordered chronologically)
  - behavioral_features (persistence, regularity, etc.)
```

### Stage 3: Tensor Conversion (❌ NEW)
```
Behavioral Windows
  ↓ (convert session sequences to aligned tensor)
Temporal Sequences
  ↓
Shape: (N_windows, max_seq_len, 20, 12)
  Padded to fixed sequence length
  Masks indicate real vs. padding
```

### Stage 4: Model Training (❌ MODIFIED)
```
Temporal Sequences
  ↓ (batch through Transformer)
Per-session embeddings
  ↓ (temporal attention)
Host behavioral representation
  ↓ (classification head)
Risk score [0, 1]
```

---

## Data Representation Changes

### Current (Session-Centric)
```python
# Training sample = one isolated session
X.shape = (num_sessions, 20, 12)  # e.g., (8,000, 20, 12)
y.shape = (num_sessions,)          # e.g., (8,000,)

# Example:
X[0] = one session's 20 flows × 12 features
y[0] = 1  # C2 or 0 = benign
```

### New (Host-Centric Longitudinal)
```python
# Training sample = behavioral window for one host
X.shape = (num_windows, seq_len, 20, 12)  # e.g., (500, 30, 20, 12)
y.shape = (num_windows,)                   # e.g., (500,)
metadata = {
    'host_ids': [...],        # which host each window belongs to
    'window_start_ts': [...], # chronological start time
    'num_sessions': [...],    # sessions in window
    'session_regularity': [...],  # how predictable
}

# Example:
X[0] = sessions [S1, S2, ..., S30] in chronological order
       30 sessions × 20 flows/session × 12 features
y[0] = 1  # Host is C2 (labeled by ANY C2 session in window)
```

---

## New Core Components Required

### 1. Behavioral Window Generator
**File**: `src/features/behavioral_window.py` (NEW)

```python
class BehavioralWindow:
    """One chronological behavioral sequence for training."""
    host_id: str
    session_summaries: List[SessionSummary]  # Chronologically ordered
    label: int  # 1 if ANY session in window is C2
    window_start_ts: float
    window_end_ts: float
    
    @property
    def duration(self) -> float:
        """Total wall-clock duration of window."""
        
    @property
    def session_count(self) -> int:
        """Number of sessions in window."""
        
    @property
    def regularity(self) -> float:
        """CV of inter-session gaps (lower = more regular C2)."""
        
    @property
    def persistence_score(self) -> float:
        """How persistently did this host communicate?"""

class BehavioralWindowBuilder:
    """Extract rolling windows from host timelines."""
    
    def build_windows(
        self,
        timelines: Dict[str, HostTimeline],
        window_size: int = 30,  # sessions per window
        slide_step: int = 10,   # sessions to advance
    ) -> List[BehavioralWindow]:
        """
        Slide a window over each host's sessions.
        
        For each host:
        - Sessions 0-29 → Window 1
        - Sessions 10-39 → Window 2
        - Sessions 20-49 → Window 3
        ...
        
        Preserves chronological ordering.
        """
```

### 2. Temporal Dataset
**File**: `src/data_loader/temporal_dataset.py` (NEW)

```python
class HostBehavioralDataset(torch.utils.data.Dataset):
    """Training dataset operating on behavioral windows (not isolated sessions)."""
    
    def __init__(
        self,
        behavioral_windows: List[BehavioralWindow],
        session_tensor_lookup: Dict[str, np.ndarray],  # sid → (20, 12) tensor
        max_seq_len: int = 50,
        normalize: bool = True,
    ):
        """
        Each sample is one host's behavioral window.
        
        Parameters
        ----------
        behavioral_windows : List of windows extracted from timelines
        session_tensor_lookup : Dict mapping session_id → numpy array (20, 12)
        max_seq_len : Max sessions to include (pad if fewer)
        normalize : Apply z-score normalization per host
        """
        self.windows = behavioral_windows
        self.lookup = session_tensor_lookup
        self.max_seq_len = max_seq_len
        self.normalize = normalize
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Returns:
        --------
        X : (max_seq_len, 20, 12) float32 tensor
            Chronological session sequence, padded to max_seq_len
            
        mask : (max_seq_len,) bool
            True for real sessions, False for padding
            
        y : int (0 or 1)
            Label: 1 if ANY session in window is C2
        """
        window = self.windows[idx]
        
        # Retrieve session tensors in chronological order
        session_tensors = []
        for summary in window.session_summaries[:self.max_seq_len]:
            tensor = self.lookup.get(summary.session_id)
            if tensor is not None:
                session_tensors.append(tensor)
        
        # Pad to max_seq_len
        seq = np.zeros((self.max_seq_len, 20, 12), dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=bool)
        
        for i, tensor in enumerate(session_tensors):
            seq[i] = tensor
            mask[i] = True
        
        # Optional: normalize per-host
        if self.normalize:
            seq[mask] = (seq[mask] - seq[mask].mean(axis=0)) / (seq[mask].std(axis=0) + 1e-9)
        
        return seq, mask, window.label
```

### 3. Longitudinal Feature Extractor
**File**: `src/features/longitudinal_features.py` (NEW)

```python
class LongitudinalFeatureExtractor:
    """Extract behavioral features from temporal sequences."""
    
    def extract_persistence_features(
        self,
        window: BehavioralWindow,
    ) -> Dict[str, float]:
        """How persistent is this host's communication?"""
        return {
            'num_sessions': len(window.session_summaries),
            'duration_hours': window.duration / 3600,
            'session_regularity': window.regularity,
            'mean_session_iat': np.mean([...]),  # inter-arrival times
            'is_regular': window.regularity < 0.3,  # C2 often regular
        }
    
    def extract_partner_features(
        self,
        timeline: HostTimeline,
    ) -> Dict[str, float]:
        """How concentrated is partner communication?"""
        return {
            'num_unique_partners': len(timeline.partner_history),
            'primary_partner_ratio': ...,  # concentration score
            'partner_diversity': ...,  # entropy of partner distribution
        }
    
    def extract_anomaly_features(
        self,
        window: BehavioralWindow,
        baseline: Optional[dict] = None,
    ) -> Dict[str, float]:
        """Deviation from normal baseline."""
        return {
            'bytes_zscore': ...,  # Z-score vs. baseline
            'duration_zscore': ...,
            'protocol_anomaly': ...,
        }
```

### 4. Modified Training Loop
**File**: `src/training/train_temporal_behavioral.py` (NEW)

```python
def train_temporal_model(
    timelines_path: str,              # Path to timelines.pkl
    session_lookup_path: str,         # Path to session tensor lookup
    model_save_dir: str,
    window_size: int = 30,
    slide_step: int = 10,
    seq_len: int = 50,  # Max timesteps per sample
    epochs: int = 30,
    ...
) -> None:
    """
    Train on behavioral windows instead of isolated sessions.
    
    Key differences from session-centric training:
    1. Input is (seq_len, 20, 12) not just (20, 12)
    2. Model sees temporal evolution within each sample
    3. Evaluation respects window integrity (not random session shuffling)
    4. Threshold tuning on windows, not sessions
    """
    
    # Load timelines and build windows
    timelines = load_host_timelines(timelines_path)
    windows = BehavioralWindowBuilder().build_windows(
        timelines,
        window_size=window_size,
        slide_step=slide_step,
    )
    
    # Create dataset
    dataset = HostBehavioralDataset(
        windows,
        session_lookup,
        max_seq_len=seq_len,
    )
    
    # Training loop processes behavioral windows, not isolated sessions
    for epoch in range(epochs):
        for batch_windows, batch_masks, batch_labels in loader:
            # batch_windows shape: (batch_size, seq_len, 20, 12)
            # batch_masks shape: (batch_size, seq_len)
            # batch_labels shape: (batch_size,)
            
            # Model now sees chronological evolution
            logits = model(batch_windows, batch_masks)
            loss = criterion(logits, batch_labels)
            optimizer.step()
```

---

## Model Architecture Considerations

### Current Transformer (Session-Centric)
```
Input: (batch, 20, 12)
  ↓ embedding
(batch, 20, d_model)
  ↓ self-attention (over 20 flows)
  ↓ aggregation
(batch, d_model)
  ↓ classification head
(batch, 1)

Learns: flow-level patterns within single session
```

### Proposed Temporal Transformer (Host-Centric)
```
Input: (batch, seq_len, 20, 12)
       where seq_len = variable number of sessions

Option A: Session-Level Encoding
  1. Per-session embedding
     (batch, seq_len, 20, 12) → (batch, seq_len, d_model)
     [Apply transformer to each session independently]
  
  2. Temporal attention over sessions
     (batch, seq_len, d_model) → (batch, d_model)
     [Attention sees which sessions matter for host classification]
  
  3. Classification
     (batch, d_model) → (batch, 1)

Learns: session-level importance in temporal context

Option B: Hierarchical Attention (Recommended)
  1. Inner attention: Within-session flow relationships
     (batch, seq_len, 20, 12) → (batch, seq_len, d_model_session)
  
  2. Outer attention: Temporal host evolution
     (batch, seq_len, d_model_session) → (batch, d_model_host)
  
  3. Classification
     (batch, d_model_host) → (batch, 1)

Learns: both local session patterns AND temporal host behavior
```

**Recommendation**: Start with Option A (simpler), then upgrade to Option B if needed.

---

## Implementation Roadmap

### Phase 3.1: Foundation (Week 1)
- [ ] Implement `BehavioralWindow` and `BehavioralWindowBuilder`
- [ ] Create `HostBehavioralDataset` with chronological window iteration
- [ ] Add unit tests for window construction and tensor alignment
- [ ] Verify window extraction preserves chronological ordering

### Phase 3.2: Feature Engineering (Week 2)
- [ ] Implement `LongitudinalFeatureExtractor`
- [ ] Add persistence, regularity, and anomaly features
- [ ] Validate features distinguish C2 from benign across sources
- [ ] Create baseline host profiles for anomaly detection

### Phase 3.3: Training Loop (Week 2-3)
- [ ] Adapt `train_transformer.py` to accept behavioral windows
- [ ] Create `train_temporal_behavioral.py` for new training pipeline
- [ ] Implement temporal masking in model forward pass
- [ ] Add window-based validation and threshold tuning

### Phase 3.4: Evaluation (Week 3)
- [ ] Modify `evaluate_transformer.py` to handle behavioral windows
- [ ] Re-run zero-shot validation with host-centric model
- [ ] Measure per-host detection rates (not just session-level)
- [ ] Generate longitudinal evaluation reports

### Phase 3.5: Integration (Week 4)
- [ ] Create new orchestrator: `train_host_centric_zero_shot.py`
- [ ] Integrate with existing split strategies
- [ ] Update documentation and runnable examples
- [ ] Run full pipeline validation

---

## Success Metrics

| Metric | Current | Target | Rationale |
|--------|---------|--------|-----------|
| **CTU-13 Recall (zero-shot)** | 0.19% | 75%+ | Universal C2 detection |
| **MCFP Recall (zero-shot)** | 0.19% | 75%+ | Cross-family generalization |
| **FPR @ Target Recall** | N/A | <2% | Production operational threshold |
| **Host-level AUC** | N/A | 0.90+ | Per-host discrimination |
| **Persistence Detection** | N/A | 80%+ | Long-term C2 identification |

---

## Risk Mitigation

### Risk: Window Construction Artifacts
**Mitigation**: 
- Validate windows against true host timelines
- Test window overlap and sliding behavior
- Confirm chronological ordering preserved

### Risk: Temporal Information Leakage
**Mitigation**:
- Never use future sessions for past window features
- Strict chronological split between train/val/test
- Validate no temporal leakage in splits

### Risk: Class Imbalance in Windows
**Mitigation**:
- May need rebalancing (more benign windows)
- Consider window-level sampling strategies
- Monitor class balance during training

### Risk: Sequence Length Variability
**Mitigation**:
- Implement proper masking for variable-length windows
- Test with different max_seq_len values
- Document padding strategy

---

## Conclusion

The architectural transition from session-centric to host-centric longitudinal modeling is **the fundamental fix** needed for cross-environment generalization. The infrastructure exists (timelines are built), but the training layer hasn't evolved to use them.

**Next Step**: Begin Phase 3.1 implementation of behavioral window extraction and temporal dataset classes.

