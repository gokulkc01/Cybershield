# CyberShield Project Status

Date: 2026-04-13

## Summary

The project has reached an important diagnostic stage. The training and evaluation pipeline now works end to end, the CLI-based training entry point has been fixed, and a repeatable ablation workflow exists. The results show that the system can achieve strong in-domain performance, but the real problem is cross-domain generalization. In other words, the model is currently good at learning the training corpus, but it is not yet reliable on a truly unseen external dataset.

The low false-positive budget is not the root cause of the weakness. It is a deployment constraint that reveals whether the model has learned portable C2 behavior. The external tests show that the current representation still depends too much on dataset-specific signals.

## What Has Been Done So Far

### 1. Training entrypoint was fixed

The training script now correctly honors command-line arguments instead of silently using hardcoded defaults.

Implemented in:
- [src/training/train_transformer.py](../src/training/train_transformer.py)

This made it possible to actually train on the updated dataset with the intended feature ablations and loss settings.

### 2. Port-ablation training was made reproducible

Feature masking is now applied consistently through the existing preprocessing pipeline. The system can train with:
- no ablation
- `src_port` and `dst_port` removed
- `is_outbound` removed
- all three removed together

Implemented through:
- [src/data_loader/feature_transforms.py](../src/data_loader/feature_transforms.py)
- [src/training/ablation_runner.py](../src/training/ablation_runner.py)

### 3. A new ablation runner was added

A dedicated runner now trains multiple variants and evaluates them on the same internal holdout and the same external benchmark. This was added so feature decisions are based on direct evidence rather than manual one-off experiments.

Implemented in:
- [src/training/ablation_runner.py](../src/training/ablation_runner.py)

### 4. A richer benign mix was created

A benign sample was extracted from the CTU-13 capture and merged with the existing MCFP benign data. This gave the model more benign diversity than the previous single-source setup.

Artifacts created:
- [data/processed/benign_mix/ctu13_benign_sample_sessions.npz](../data/processed/benign_mix/ctu13_benign_sample_sessions.npz)
- [data/processed/mcfp_multi_plus_benign_sessions.npz](../data/processed/mcfp_multi_plus_benign_sessions.npz)

### 5. Internal and external evaluation paths were validated

The evaluation code can now score both:
- internal holdout split
- frozen external dataset

Relevant files:
- [src/evaluation/evaluate_holdout.py](../src/evaluation/evaluate_holdout.py)
- [src/evaluation/evaluate_transformer.py](../src/evaluation/evaluate_transformer.py)

### 6. External benchmark overlap was checked

The existing Stratosphere MCFP NPZ was found to overlap almost completely with the training corpus, so it is not a true external benchmark. A deduplicated external CTU sample was built and used instead.

External benchmark artifacts:
- [data/processed/ctu13_external_unseen_sample.npz](../data/processed/ctu13_external_unseen_sample.npz)
- [data/processed/ctu13_external_unseen_sample_30k.npz](../data/processed/ctu13_external_unseen_sample_30k.npz)

### 7. The evaluation pipeline now exposes the real failure mode

The model can look very strong on in-domain holdout, but it still collapses on a truly unseen external slice. That confirmed the main issue is domain shift, not just threshold selection.

## What We Have Learned

### In-domain behavior

On the combined in-domain dataset, the model can reach strong recall and low FPR after training and threshold selection. That means the architecture and training loop are not broken.

### External behavior

On the deduplicated unseen CTU sample, the model’s external recall and AUC were poor, and the false-positive rate was too high for the deployment budget. That means the current representation is not robust enough to transfer across traffic sources.

### Feature bias

The ablation study showed that removing ports and directionality changes the behavior, but no feature-only variant solved the external problem.

Observed pattern:
- baseline: best internal holdout behavior, still weak externally
- drop ports: slightly better internal budget behavior, still weak externally
- drop is_outbound: not enough
- drop ports and is_outbound: highest external recall among the tested variants, but FPR became unacceptable

Interpretation:
- ports and directionality matter
- but the problem is broader than a single feature
- external generalization still needs better data coverage and split strategy

## Current Objectives

### Primary objective

Build a C2 detector that keeps false positives under the 0.005 budget while remaining useful on unseen traffic.

### Secondary objectives

1. Improve external recall on unseen traffic.
2. Reduce dependence on dataset-specific artifacts like fixed ports or traffic direction.
3. Make validation representative of the deployment problem.
4. Keep the evaluation process leakage-free and reproducible.

### Scientific objective

Demonstrate that the model learns portable behavioral C2 signals rather than shortcuts tied to one dataset or one family of malware.

## What Still Needs To Be Achieved

### 1. Better source diversity

The training mixture still needs more realistic variety, especially on the benign side and across C2 families. The current data is not broad enough to support strong external transfer.

### 2. Source-aware split strategy

The current split strategy is not yet enough to ensure the validation set matches the intended external domain. We need splits that better reflect deployment conditions.

### 3. Stronger external benchmark

The present external CTU sample is useful, but we still need a more robust external evaluation setup that stays clearly separated from training data and better represents the deployment environment.

### 4. Calibration after representation improves

Threshold calibration is worth revisiting, but only after the representation itself becomes more portable. Threshold tuning alone will not fix the current external gap.

### 5. Final feature decision

We still need to decide whether the best long-term feature policy is:
- keep ports removed permanently
- keep `is_outbound`
- remove `is_outbound` as well
- or move to a richer feature policy after more data diversity is added

### 6. Final model comparison

The ablation runner exists, but the next step is to use it for full-length runs and decide based on the completed JSON comparison report.

## Next Plan

### Immediate next step

Run the full ablation study with a realistic number of epochs and use the generated comparison report as the source of truth for feature selection.

### After that

1. Keep the feature variant that gives the best external tradeoff.
2. Expand the training data with more benign sources and more C2 families.
3. Rebuild the internal validation split so it is closer to the intended external domain.
4. Re-evaluate on the unseen external benchmark.
5. Only then revisit calibration or architecture changes if the external FPR is still too high.

### Decision rule

Accept a configuration only if:
- external recall improves materially
- external FPR stays below 0.005
- the result holds on a truly unseen benchmark

## Current Best Reading Of The Problem

The model is not failing because the FPR budget is too strict. The budget is exposing a generalization problem. The next engineering win will come from better data coverage and more realistic splits, not from more threshold tuning alone.

## Key Files

- [src/training/train_transformer.py](../src/training/train_transformer.py)
- [src/training/ablation_runner.py](../src/training/ablation_runner.py)
- [src/data_loader/feature_transforms.py](../src/data_loader/feature_transforms.py)
- [src/data_loader/split_utils.py](../src/data_loader/split_utils.py)
- [src/evaluation/evaluate_holdout.py](../src/evaluation/evaluate_holdout.py)
- [src/evaluation/evaluate_transformer.py](../src/evaluation/evaluate_transformer.py)
- [src/evaluation/operating_point.py](../src/evaluation/operating_point.py)
- [src/features/feature_config.py](../src/features/feature_config.py)

## Current Best Artifacts

- [experiments/ablation_results/ablation_report.json](../experiments/ablation_results/ablation_report.json)
- [experiments/ablation_results/](../experiments/ablation_results/)
- [data/processed/mcfp_multi_plus_benign_sessions.npz](../data/processed/mcfp_multi_plus_benign_sessions.npz)
- [data/processed/ctu13_external_unseen_sample_30k.npz](../data/processed/ctu13_external_unseen_sample_30k.npz)

## Bottom Line

The pipeline is working, the ablation workflow is working, and the current results have identified the real blocker: external generalization. The next plan is to improve source diversity and split realism, then rerun the same external benchmark under the same evaluation rules.
