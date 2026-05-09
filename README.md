# CyberShield

Session-level C2 detection research pipeline.

Current branch: `research/multifamily_generalization`
Current status: baseline verified, controlled generalization experiment ready.

## Quick Start

### 1) Open terminal in project root

Git Bash:
```bash
cd /d/CyberShield
source .venv/Scripts/activate
```

PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& d:\CyberShield\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn torch joblib pytest matplotlib seaborn
```

### 3) Run tests

```bash
python -m pytest tests/ -q
```

Expected: all tests pass.

## Run Commands

### Evaluate existing zero-shot model

```bash
python -m src.evaluation.evaluate_zero_shot \
  --model_path experiments/transformer_uwf_only/best_transformer.pth \
  --test_npz data/processed/v2_mixed_session_only/test_sessions.npz \
  --split_metadata data/processed/v2_mixed_session_only/split_metadata.json
```

### Train + evaluate pipeline (example)

```bash
python -m src.pipelines.train_and_eval_zero_shot \
  --uwf_train data/processed/v2_uwf/train_sessions.npz \
  --uwf_val data/processed/v2_uwf/val_sessions.npz \
  --mixed_test data/processed/v2_mixed_session_only/test_sessions.npz \
  --mixed_metadata data/processed/v2_mixed_session_only/split_metadata.json \
  --model_dir experiments/transformer_uwf_only \
  --epochs 30 \
  --batch_size 64
```

## Current Branch Workflow

Research work is being developed on:

```bash
git checkout research/multifamily_generalization
```

## Notes

- Use PR-AUC / recall / precision / FPR for imbalanced evaluation.
- Avoid using accuracy as a primary metric.
- Current controlled experiment direction is documented in:
  - docs/research_plan_multifamily_generalization.md
  - docs/IMMEDIATE_ACTION_PLAN.md
  - docs/PROGRESS_CHECKLIST.md

## Verified Results

- `pytest tests/ -q`: 74 passed
- Example training/evaluation smoke run: completed on CPU
- Zero-shot smoke result on mixed test set: recall 0.0024, FPR 0.0055, AUC 0.8952
