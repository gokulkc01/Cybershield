# Phase 3 Implementation Skeleton: Host-Centric Longitudinal Architecture

**Status**: Pre-implementation code architecture  
**Purpose**: Concrete code structure for the transition  
**Reference**: See `docs/phase3_architectural_redesign_plan.md` for conceptual design

---

## File Structure

```
src/
├── features/
│   ├── behavioral_window.py           (NEW - window extraction)
│   ├── longitudinal_features.py       (NEW - temporal feature engineering)
│   └── host_timeline.py               (EXISTS - used by windows)
│
├── data_loader/
│   ├── temporal_dataset.py            (NEW - behavioral window dataset)
│   ├── torch_dataset.py               (EXISTS - for reference/migration)
│   └── npz_utils.py                   (EXISTS - used for lookups)
│
├── training/
│   ├── train_temporal_behavioral.py   (NEW - behavioral window training)
│   └── train_transformer.py           (EXISTS - to integrate with)
│
├── models/
│   └── transformer.py                 (EXISTS - may need temporal extension)
│
└── pipelines/
    ├── train_host_centric_zero_shot.py  (NEW - orchestrator)
    └── train_and_eval_zero_shot.py      (EXISTS - session-centric reference)

tests/
└── test_temporal_pipeline.py          (NEW - behavioral validation)
```

---

## Component 1: Behavioral Window Extraction

### File: `src/features/behavioral_window.py`

```python
"""
Behavioral window extraction from host timelines.

Converts HostTimeline objects into training samples (BehavioralWindow)
that preserve chronological ordering and temporal context.

Pipeline:
    HostTimeline (Dict[host_id → sessions over time])
        ↓
    BehavioralWindow (rolling windows of sessions)
        ↓
    Training samples with temporal structure
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np
from src.features.host_timeline import HostTimeline, SessionSummary


@dataclass
class BehavioralWindow:
    """One chronological behavioral sequence for one host.
    
    CRITICAL: Session ordering is preserved chronologically.
    This is the unit of training, not individual sessions.
    """
    
    window_id: str                          # unique identifier
    host_id: str                            # source host
    session_summaries: List[SessionSummary] # chronologically ordered
    label: int                              # 1 if ANY session is C2
    
    # Temporal metadata
    window_start_ts: float                  # earliest session start
    window_end_ts: float                    # latest session end
    source_dataset: str                     # uwf / ctu13 / mcfp
    
    # Derived properties (lazily computed)
    _regularity: Optional[float] = field(default=None, repr=False)
    _persistence: Optional[float] = field(default=None, repr=False)
    
    @property
    def duration(self) -> float:
        """Wall-clock duration of entire window."""
        return max(0.0, self.window_end_ts - self.window_start_ts)
    
    @property
    def session_count(self) -> int:
        """Number of sessions in this window."""
        return len(self.session_summaries)
    
    @property
    def regularity(self) -> float:
        """
        Coefficient of variation of inter-session gaps.
        
        Lower = more regular (typical for C2 beaconing)
        Higher = irregular (typical for benign)
        
        Range: [0, ∞)
        - 0-0.3: Very regular (strong C2 signal)
        - 0.3-1.0: Somewhat regular
        - 1.0+: Highly irregular
        """
        if self._regularity is not None:
            return self._regularity
        
        if len(self.session_summaries) < 2:
            self._regularity = float('nan')
            return self._regularity
        
        gaps = []
        for i in range(1, len(self.session_summaries)):
            gap = (self.session_summaries[i].start_ts - 
                   self.session_summaries[i-1].end_ts)
            if gap >= 0:
                gaps.append(gap)
        
        if not gaps or len(gaps) < 2:
            self._regularity = float('nan')
            return self._regularity
        
        gaps_arr = np.array(gaps)
        mean_gap = np.mean(gaps_arr)
        
        if mean_gap < 1e-9:
            self._regularity = 0.0
            return self._regularity
        
        cv = float(np.std(gaps_arr) / mean_gap)
        self._regularity = cv
        return cv
    
    @property
    def persistence_score(self) -> float:
        """
        Quantifies persistence of communication.
        
        Combines:
        - Duration: longer = higher score
        - Session count: more sessions = higher score
        - Regularity: regular = higher score (C2 pattern)
        
        Returns: [0, 1] approximate probability this is persistent C2
        """
        if self._persistence is not None:
            return self._persistence
        
        # Duration contribution (normalized to 1 day = 0.25)
        duration_days = self.duration / 86400.0
        duration_score = min(1.0, duration_days / 1.0)
        
        # Session count contribution (normalized to 100 = 1.0)
        session_score = min(1.0, self.session_count / 100.0)
        
        # Regularity contribution (inverted CV)
        if np.isnan(self.regularity) or self.regularity > 5.0:
            regularity_score = 0.0
        else:
            regularity_score = max(0.0, 1.0 - self.regularity / 5.0)
        
        # Weighted combination
        persistence = (
            0.3 * duration_score +
            0.3 * session_score +
            0.4 * regularity_score
        )
        
        self._persistence = float(persistence)
        return self._persistence
    
    @property
    def is_labeled_c2(self) -> bool:
        """Any session in this window labeled as C2."""
        return self.label == 1
    
    @property
    def c2_fraction(self) -> float:
        """Fraction of sessions in window labeled as C2."""
        if self.session_count == 0:
            return 0.0
        c2_count = sum(1 for s in self.session_summaries if s.label == 1)
        return c2_count / self.session_count


class BehavioralWindowBuilder:
    """Extract rolling behavioral windows from host timelines.
    
    Key principle:
    - Windows slide chronologically
    - Session order is NEVER shuffled
    - Each window is one training sample
    - Windows may overlap (controlled by slide_step)
    """
    
    def build_windows(
        self,
        timelines: Dict[str, HostTimeline],
        window_size: int = 30,
        slide_step: int = 10,
        min_sessions_per_window: int = 1,
    ) -> List[BehavioralWindow]:
        """
        Extract behavioral windows from all hosts.
        
        Parameters
        ----------
        timelines : Dict mapping host_id → HostTimeline
        window_size : Number of sessions per window
        slide_step : Sessions to advance between windows
        min_sessions_per_window : Discard windows with fewer sessions
        
        Returns
        -------
        List of BehavioralWindow objects, chronologically ordered within each window
        
        Algorithm
        ---------
        For each host:
            sessions = sorted(host.sessions, by=start_ts)
            for i in range(0, len(sessions) - window_size, slide_step):
                window = sessions[i : i + window_size]
                yield BehavioralWindow(window)
        
        Example (window_size=3, slide_step=1):
            Host_A has sessions: [S0, S1, S2, S3, S4, S5]
            Window 0: [S0, S1, S2]
            Window 1: [S1, S2, S3]
            Window 2: [S2, S3, S4]
            Window 3: [S3, S4, S5]
        """
        windows = []
        window_counter = 0
        
        for host_id, timeline in timelines.items():
            # Get host's sessions in chronological order
            sessions_sorted = sorted(
                timeline.sessions_received,  # from HostTimeline.sessions_received
                key=lambda s: s.start_ts,
            )
            
            if len(sessions_sorted) < window_size:
                # Skip hosts with insufficient sessions
                continue
            
            # Slide window across sessions
            for start_idx in range(
                0,
                len(sessions_sorted) - window_size + 1,
                slide_step,
            ):
                end_idx = start_idx + window_size
                window_sessions = sessions_sorted[start_idx:end_idx]
                
                # Determine label: 1 if ANY session is C2
                label = max(s.label for s in window_sessions)
                
                # Create window
                window = BehavioralWindow(
                    window_id=f"{host_id}_w{window_counter}",
                    host_id=host_id,
                    session_summaries=window_sessions,
                    label=label,
                    window_start_ts=window_sessions[0].start_ts,
                    window_end_ts=window_sessions[-1].end_ts,
                    source_dataset=timeline.source_dataset,
                )
                
                if len(window_sessions) >= min_sessions_per_window:
                    windows.append(window)
                    window_counter += 1
        
        return windows
    
    def build_windows_with_metadata(
        self,
        timelines: Dict[str, HostTimeline],
        metadata_dict: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[List[BehavioralWindow], Dict]:
        """
        Build windows and preserve provenance metadata.
        
        Returns
        -------
        (windows, metadata_dict) where metadata_dict contains:
            - 'num_hosts': int
            - 'num_windows': int
            - 'avg_window_size': float
            - 'c2_fraction': float (fraction of windows with C2)
            - 'regular_windows': int (regularity < 0.3)
        """
        windows = self.build_windows(timelines, **kwargs)
        
        metadata = {
            'num_hosts': len(timelines),
            'num_windows': len(windows),
            'avg_window_size': np.mean([w.session_count for w in windows]),
            'c2_fraction': np.mean([1 if w.is_labeled_c2 else 0 for w in windows]),
            'regular_windows': sum(1 for w in windows if w.regularity < 0.3),
            'window_builder_config': {
                'window_size': kwargs.get('window_size', 30),
                'slide_step': kwargs.get('slide_step', 10),
            }
        }
        
        return windows, metadata


# Utility functions for window analysis

def analyze_window_properties(windows: List[BehavioralWindow]) -> Dict:
    """Statistics about extracted windows."""
    regularity_vals = [w.regularity for w in windows if not np.isnan(w.regularity)]
    persistence_vals = [w.persistence_score for w in windows]
    
    return {
        'num_windows': len(windows),
        'c2_windows': sum(1 for w in windows if w.is_labeled_c2),
        'benign_windows': sum(1 for w in windows if not w.is_labeled_c2),
        'avg_sessions_per_window': np.mean([w.session_count for w in windows]),
        'avg_regularity': np.mean(regularity_vals) if regularity_vals else np.nan,
        'avg_persistence': np.mean(persistence_vals),
        'persistent_c2_windows': sum(
            1 for w in windows 
            if w.is_labeled_c2 and w.persistence_score > 0.6
        ),
    }
```

---

## Component 2: Temporal Dataset Class

### File: `src/data_loader/temporal_dataset.py`

```python
"""
PyTorch dataset for behavioral windows (temporal sequences).

Unlike C2SessionDataset (one sample = one session),
HostBehavioralDataset treats one sample = chronological window of sessions.

Key difference:
    C2SessionDataset: X.shape = (num_sessions, 20, 12)
    HostBehavioralDataset: X.shape = (num_windows, seq_len, 20, 12)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.features.behavioral_window import BehavioralWindow
from src.data_loader.normalization import FeatureNormalizer


class HostBehavioralDataset(Dataset):
    """Training dataset operating on behavioral windows."""
    
    def __init__(
        self,
        behavioral_windows: List[BehavioralWindow],
        session_tensor_lookup: Dict[str, np.ndarray],
        max_seq_len: int = 50,
        normalize_per_window: bool = True,
        feature_normalizer: Optional[FeatureNormalizer] = None,
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        behavioral_windows : List of BehavioralWindow objects
        session_tensor_lookup : Dict mapping session_id → (20, 12) numpy array
                                Each session is pre-processed flow tensor.
        max_seq_len : Maximum number of sessions per window (padded if fewer)
        normalize_per_window : Normalize each window independently
        feature_normalizer : Optional pre-fit normalizer for global normalization
        
        Important:
        - session_tensor_lookup contains pre-computed session tensors
        - Never reconstruct tensors at runtime
        - Sessions should be normalized/transformed before lookup creation
        """
        self.windows = behavioral_windows
        self.lookup = session_tensor_lookup
        self.max_seq_len = max_seq_len
        self.normalize_per_window = normalize_per_window
        self.normalizer = feature_normalizer
        
        # Validation
        assert max_seq_len > 0, "max_seq_len must be positive"
        assert len(self.windows) > 0, "No windows provided"
        assert len(self.lookup) > 0, "No session lookup provided"
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Get one behavioral window.
        
        Returns
        -------
        X : (max_seq_len, 20, 12) float32
            Chronological sessions, padded to max_seq_len
            X[0] = earliest session
            X[-1] = latest session (or padding)
            
        mask : (max_seq_len,) bool
            True for real sessions, False for padding
            
        y : int (0 or 1)
            Label for this window
        """
        window = self.windows[idx]
        
        # Initialize output tensor
        X = np.zeros((self.max_seq_len, 20, 12), dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=bool)
        
        # Populate tensor with session sequences
        num_filled = 0
        for i, session_summary in enumerate(window.session_summaries):
            if i >= self.max_seq_len:
                break
            
            # Look up pre-computed session tensor
            session_tensor = self.lookup.get(session_summary.session_id)
            if session_tensor is None:
                # Missing session tensor - skip or raise?
                # For now, skip (maintains data integrity)
                continue
            
            X[i] = session_tensor
            mask[i] = True
            num_filled += 1
        
        # Normalization
        if num_filled > 0:
            if self.normalize_per_window:
                # Normalize each window independently (z-score on real sessions)
                real_data = X[mask]
                if len(real_data) > 1:
                    mean = real_data.mean(axis=0)
                    std = real_data.std(axis=0)
                    std = np.where(std < 1e-9, 1.0, std)  # Avoid division by 0
                    X[mask] = (real_data - mean) / std
            
            elif self.normalizer is not None:
                # Apply pre-fit global normalizer
                X = self.normalizer.transform(X, mask)
        
        return X, mask, window.label


def create_temporal_dataloaders(
    train_windows: List[BehavioralWindow],
    val_windows: List[BehavioralWindow],
    test_windows: Optional[List[BehavioralWindow]],
    session_lookup: Dict[str, np.ndarray],
    batch_size: int = 32,
    max_seq_len: int = 50,
    normalize: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch dataloaders for temporal behavioral training.
    
    Parameters
    ----------
    train_windows, val_windows, test_windows : List[BehavioralWindow]
    session_lookup : Session tensor dictionary
    batch_size : Batch size for training
    max_seq_len : Max sessions per window
    normalize : Apply per-window normalization
    num_workers : Number of dataloader workers
    
    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    
    train_dataset = HostBehavioralDataset(
        train_windows,
        session_lookup,
        max_seq_len=max_seq_len,
        normalize_per_window=normalize,
    )
    
    val_dataset = HostBehavioralDataset(
        val_windows,
        session_lookup,
        max_seq_len=max_seq_len,
        normalize_per_window=normalize,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = None
    if test_windows is not None:
        test_dataset = HostBehavioralDataset(
            test_windows,
            session_lookup,
            max_seq_len=max_seq_len,
            normalize_per_window=normalize,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    return train_loader, val_loader, test_loader
```

---

## Component 3: Integration with Training

### File: `src/training/train_temporal_behavioral.py` (Skeleton)

```python
"""
Train Transformer on behavioral windows (temporal sequences).

This replaces session-centric training with host-centric training.
Key difference:
- Input model receives (batch, seq_len, 20, 12) not (batch, 20, 12)
- Model must handle variable-length sequences
- Threshold tuning respects window integrity
"""

import torch
from src.data_loader.temporal_dataset import create_temporal_dataloaders
from src.features.behavioral_window import BehavioralWindowBuilder
from src.features.host_timeline import load_host_timelines


def train_temporal_behavioral_model(
    train_timelines_path: str,
    val_timelines_path: str,
    test_timelines_path: Optional[str] = None,
    session_lookup_path: str,        # Dict[session_id → (20, 12) tensor]
    model_save_dir: str = "experiments/transformer_temporal",
    window_size: int = 30,
    slide_step: int = 10,
    seq_len: int = 50,
    batch_size: int = 32,
    epochs: int = 50,
    **kwargs
) -> None:
    """
    Train model on behavioral windows.
    
    This is the new entry point for Phase 3 training.
    """
    
    # Load timelines
    train_timelines = load_host_timelines(train_timelines_path)
    val_timelines = load_host_timelines(val_timelines_path)
    test_timelines = (
        load_host_timelines(test_timelines_path) 
        if test_timelines_path else None
    )
    
    # Build behavioral windows
    builder = BehavioralWindowBuilder()
    train_windows = builder.build_windows(
        train_timelines,
        window_size=window_size,
        slide_step=slide_step,
    )
    val_windows = builder.build_windows(
        val_timelines,
        window_size=window_size,
        slide_step=slide_step,
    )
    test_windows = (
        builder.build_windows(
            test_timelines,
            window_size=window_size,
            slide_step=slide_step,
        ) if test_timelines else None
    )
    
    # Load session lookup
    session_lookup = np.load(session_lookup_path, allow_pickle=True)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        train_windows,
        val_windows,
        test_windows,
        session_lookup,
        batch_size=batch_size,
        max_seq_len=seq_len,
    )
    
    # Training loop (similar to session-centric but processes windows)
    for epoch in range(epochs):
        for batch_X, batch_mask, batch_y in train_loader:
            # batch_X: (batch, seq_len, 20, 12)
            # batch_mask: (batch, seq_len)
            # batch_y: (batch,)
            
            # Forward pass through temporal model
            logits = model(batch_X, batch_mask)
            # ... compute loss, backprop, etc.
```

---

## Migration Path

### From Session-Centric to Host-Centric

```python
# OLD (Session-Centric)
from src.data_loader.torch_dataset import create_dataloaders
train_loader, val_loader, test_loader, ... = create_dataloaders(npz_path)

# For each batch_X in train_loader:
#   batch_X.shape = (batch, 20, 12)  # One session per sample
#   model processes in isolation


# NEW (Host-Centric Longitudinal)
from src.features.behavioral_window import BehavioralWindowBuilder
from src.data_loader.temporal_dataset import create_temporal_dataloaders

timelines = load_host_timelines("timelines.pkl")
windows = BehavioralWindowBuilder().build_windows(timelines)
train_loader, val_loader, test_loader = create_temporal_dataloaders(
    windows_train, windows_val, windows_test, session_lookup
)

# For each batch_X in train_loader:
#   batch_X.shape = (batch, seq_len, 20, 12)  # Behavioral windows
#   model sees temporal evolution of each host
```

---

## Key Design Principles

1. **Chronological Integrity**: Session order is NEVER shuffled within windows
2. **Temporal Context**: Model always sees sessions in time order
3. **No Future Leakage**: Windows never include future information
4. **Flexible Window Size**: Supports different window sizes for different experiments
5. **Metadata Preservation**: Source dataset, host IDs, timestamps preserved
6. **Scalability**: Handles variable-length windows efficiently

---

## Next Steps

1. Implement `BehavioralWindow` and `BehavioralWindowBuilder`
2. Implement `HostBehavioralDataset` with proper masking
3. Test window extraction on existing timelines
4. Adapt model forward pass to accept (batch, seq_len, 20, 12) input
5. Create new training orchestrator
6. Validate on zero-shot benchmark

