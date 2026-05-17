# Reproducibility & Artifact Documentation

**Date**: May 10, 2026  
**Experiment**: Strict Multifamily Generalization  
**Status**: ✅ COMPLETE

## Reproducibility Information

### Random Seed
- **Fixed Seed**: 42
- **Framework**: PyTorch
- **Impact**: Ensures deterministic model initialization and training

### Feature Schema
- **Config File**: `src/features/feature_config_experiment.py`
- **Feature Count**: 10
- **Feature Names** (in order):
  1. orig_bytes
  2. resp_bytes
  3. orig_pkts
  4. resp_pkts
  5. bytes_per_pkt
  6. packet_ratio
  7. byte_ratio
  8. is_outbound
  9. duration
  10. iat
- **Removed Features**: src_port (index 10), dst_port (index 11)
- **Session Length**: 20 flows per session
- **Normalization**: Z-score (computed on training set)

### Model Checkpoint
- **Path**: `experiments/multifamily_generalization/strict_smoke/best_transformer.pth`
- **Size**: 0.439 MB
- **Format**: PyTorch state_dict
- **Architecture**: Transformer (2 layers, 4 heads, 64 dim, dropout=0.1)
- **Epoch Saved**: 1 (best validation AUC: 0.9949)

### Split Manifest
- **Path**: `data/processed/experiment_10f_splits_strict/split_summary.json`
- **Contents**:
  - Train family list: ["neris", "kraken"]
  - Test family list: ["conficker"]
  - Train session count: 1,895,329
  - Val session count: 334,470
  - Test session count: 374,187
  - Family overlap verification: ZERO

### Data Files
**NPZ Files** (session collections, 10-feature format):
- Train: `data/processed/experiment_10f_splits_strict/train_sessions.npz` (1.9 MB)
- Val: `data/processed/experiment_10f_splits_strict/val_sessions.npz` (334 KB)
- Test: `data/processed/experiment_10f_splits_strict/test_sessions.npz` (374 KB)

**Scenario NPZs** (source data, 10-feature format):
- Neris: `data/processed/ctu13_scenario1_neris.npz`
- Kraken: `data/processed/ctu13_scenario2_kraken.npz`
- Conficker: `data/processed/ctu13_scenario9_conficker.npz`

### Environment
- **Python Version**: 3.9+
- **PyTorch Version**: 2.0+
- **Device**: CPU
- **OS**: Windows / Linux (path-agnostic)
- **Dependencies**: See requirements.txt

### Execution Steps (To Reproduce)

```bash
# 1. Activate environment
source .venv/Scripts/activate

# 2. Build scenario NPZs (if not present)
python -m src.pipelines.build_ctu13_scenarios \
  --manifest experiments/multifamily_generalization/ctu13_raw_sources.json \
  --out_sources_manifest experiments/multifamily_generalization/ctu13_sources_generated.json

# 3. Prepare family-aware splits
python -m src.pipelines.prepare_multifamily_splits \
  --sources_manifest experiments/multifamily_generalization/ctu13_sources_generated.json \
  --out_dir data/processed/experiment_10f_splits_strict \
  --train_families neris,kraken \
  --test_families conficker

# 4. Train and evaluate
python -m src.pipelines.train_and_eval_multifamily \
  --train_npz data/processed/experiment_10f_splits_strict/train_sessions.npz \
  --val_npz data/processed/experiment_10f_splits_strict/val_sessions.npz \
  --test_npz data/processed/experiment_10f_splits_strict/test_sessions.npz \
  --split_metadata data/processed/experiment_10f_splits_strict/split_summary.json \
  --model_dir experiments/multifamily_generalization/strict_smoke \
  --epochs 1 \
  --batch_size 64
```

### Expected Output
- Model checkpoint: `experiments/multifamily_generalization/strict_smoke/best_transformer.pth`
- Evaluation metrics in terminal (or logs)
- JSON results: `experiments/multifamily_generalization/strict_smoke/eval_results.json`

## Key Verification Checks

✅ **No Family Leakage**
- Train families: {neris, kraken}
- Test families: {conficker}
- Intersection: {} (empty)

✅ **Feature Schema Consistency**
- All NPZs verified to have exactly 10 features
- Feature names match across all files
- Ports successfully removed

✅ **Random Seed Documentation**
- Seed 42 used throughout
- Reproducible model initialization

✅ **Model Architecture Unchanged**
- Original Transformer preserved
- No new layers or components added
- Pure feature engineering experiment

## Dependency Tree

```
Input: Raw CTU-13 captures (binetflow format)
  ↓
build_ctu13_scenarios.py
  ↓
Scenario NPZs (ctu13_scenario*.npz) [10-feature, converted]
  ↓
prepare_multifamily_splits.py
  ↓
Train/Val/Test splits (experiment_10f_splits_strict/)
  ↓
train_and_eval_multifamily.py
  ↓
Trained model + Evaluation results
```

---

**Document Created**: May 10, 2026 | **Status**: READY FOR REPRODUCIBILITY
