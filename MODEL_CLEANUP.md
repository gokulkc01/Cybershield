# Model Cleanup & Deprecation Guide

## Overview

This document categorizes all trained models in the `experiments/` directory as either **ACTIVE** (production use) or **DEPRECATED** (reference/historical only).

## Active Models (Keep These)

### Production Model ✅
- **Path:** `experiments/domain_adaptive_sweep_20/`
- **Export:** `experiments/domain_adaptive_sweep_20/exported/`
- **Status:** PRIMARY PRODUCTION MODEL
- **Description:** Multi-seed domain-adaptive transformer (seeds: 11, 42, 1337)
- **Performance:**
  - Mixed domain: AUC 0.9961, F1 0.9067, FPR 0.0049
  - UWF domain: AUC 1.0, F1 0.9811
  - Validated across 3 independent seeds with consistent results
- **Key Features:**
  - Shared backbone architecture
  - Domain-specific classification heads
  - Automatic domain detection (centroid-based)
  - Per-domain operating thresholds
- **Use Case:** All production inference, new deployments
- **API:** Default model loaded by `src/api/predict.py`

### Multi-Seed Checkpoints (Reference)
- `experiments/domain_adaptive_sweep_seed_11/best_transformer.pth`
- `experiments/domain_adaptive_sweep_seed_42/best_transformer.pth`
- `experiments/domain_adaptive_sweep_seed_1337/best_transformer.pth`
- **Status:** VALIDATED ENSEMBLE MEMBERS
- **Use Case:** Cross-validation, ensemble methods, reproducibility verification
- **Note:** Individual seed results confirm reproducibility of main model

## Deprecated Models (Reference Only)

### Single-Head Baselines ⚠️
| Model | Path | Issue | Notes |
|-------|------|-------|-------|
| Mixed training | `extended_mixed_real_balanced_long/` | No UWF support | Single domain, cannot generalize |
| CTU+MCFP only | `ctu_mcfp_only_balanced/` | No UWF support | Fails on unseen UWF (AUC 0.71) |
| UWF only | `transformer_uwf_only/` | No mixed domain | Cannot generalize to CTU/MCFP |

**Why Deprecated:**
- Single classification head cannot simultaneously support multiple domains
- Cross-dataset generalization fails (demonstrated by 0.71 AUC on UWF test)
- No per-domain threshold calibration
- Cannot auto-detect incoming domain

**Historical Value:**
- Demonstrate generalization failure (CTU+MCFP → UWF drops to 0.71 AUC)
- Establish baseline for multi-task improvement

### Continual Learning Attempts ⚠️
| Model | Path | Approach | Result |
|-------|------|----------|--------|
| Rehearsal+Distill | `continual_mixed_replay_uwf/` | Replay buffer + KD | Calibration failed (inf threshold) |
| Fine-tuning | `ctu_mcfp_to_uwf_finetune/` | Sequential fine-tune | Lost mixed domain performance |

**Why Deprecated:**
- Rehearsal approach showed threshold calibration instability
- Fine-tuning preserved UWF but could not recover mixed domain
- Multi-task approach (multi-head) proved superior
- Calibration strategy insufficient for online learning

**Lessons Learned:**
- Simple rehearsal insufficient for multi-domain adaptation
- Fine-tuning alone causes catastrophic forgetting
- Explicit per-domain heads needed for stable calibration

### Early Experimental Models ⚠️
- `experiments/transformer/best_transformer.pth`
- `experiments/multifamily_generalization/strict_smoke/best_transformer.pth`
- `experiments/phase1_baseline_validation/`
- `experiments/extended_mixed_real/`
- `experiments/extended_mixed_real_balanced/`

**Status:** OUTDATED - RESEARCH ONLY
**Reason:** Early prototype iterations, suboptimal architecture, poor metrics
**Keep:** Yes (for reproducibility/audit trail)
**Use:** Historical reference only

### Experiment-Specific Models ⚠️
| Directory | Purpose | Status |
|-----------|---------|--------|
| `ablation_results/` | Feature ablation studies | EXPERIMENTAL |
| `transformer_ablate_*` | Port/payload ablations | EXPERIMENTAL |
| `transformer_focal_*` | Focal loss variants | EXPERIMENTAL |
| `transformer_quick/` | Quick smoke test | SMOKE TEST |
| `transformer_tune/` | Hyperparameter tuning | EXPERIMENTAL |
| `red_agent_*` | Red agent research | RESEARCH |

**Status:** All DEPRECATED
**Reason:** Specialized one-off experiments
**Use:** Reference for methodology only
**Recommendation:** Archive to `old_experiments/` if space becomes critical

## Storage Recommendations

### Essential (Keep at Full Fidelity)
```
experiments/domain_adaptive_sweep_20/
├── best_transformer.pth (PRODUCTION)
├── exported/
│   ├── model_checkpoint.pth
│   ├── thresholds.json
│   ├── domain_centroids.json
│   └── feature_normalizer.json
└── training_log.json
```
**Rationale:** Production model and export artifacts

### Historical (Keep for Reference)
```
experiments/domain_adaptive_sweep_seed_*/  
experiments/extended_mixed_real_balanced_long/
experiments/ctu_mcfp_only_balanced/
experiments/transformer_uwf_only/
experiments/continual_mixed_replay_uwf/
experiments/ctu_mcfp_to_uwf_finetune/
```
**Rationale:** Validate reproducibility, demonstrate generalization failure, lessons learned

### Optional Archive (Move if Space Critical)
```
experiments/ablation_results/
experiments/transformer_ablate_*/
experiments/transformer_focal_*/
experiments/transformer_quick/
experiments/transformer_tune/
experiments/red_agent_*/
experiments/phase1_baseline_validation/
experiments/extended_mixed_real/
```
**Rationale:** Research/experiment data, not needed for production or primary validation

## Deprecation Timeline

| Date | Action | Models |
|------|--------|--------|
| Phase 2 End | Single-head training ends | Extended mixed, CTU+MCFP, UWF only |
| Phase 3 Mid | Continual learning abandoned | Rehearsal+distill, fine-tuning |
| Phase 3 End | Multi-task production ready | Domain-adaptive sweep 20 promoted |
| Phase 4 (Future) | Archive old experiments | ablation_results, transformer_tune, etc. |

## Migration Checklist

If migrating from old model to domain-adaptive:

- [ ] Update API `DEFAULT_EXPORT_DIR` (currently: `experiments/domain_adaptive_sweep_20/exported`)
- [ ] Test `/health` endpoint to verify model loading
- [ ] Test `/predict-file` with sample NPZ file
- [ ] Verify domain detection matches expected sessions
- [ ] Update documentation links
- [ ] Run A/B test vs. old model (optional)
- [ ] Retire old model API endpoint

## Frequently Asked Questions

**Q: Can I still use the single-head mixed model?**
A: Not recommended. It will fail on UWF data (AUC 0.71). The domain-adaptive model handles both domains.

**Q: Why keep the old models?**
A: Reproducibility, audit trail, and historical validation. They demonstrate why multi-task architecture was necessary.

**Q: What if I need the old API behavior?**
A: Set `export_dir="experiments/extended_mixed_real_balanced_long/exported"` in API call (if exported).

**Q: How much space do old models use?**
A: ~500MB total for all checkpoints. Not critical unless severely space-constrained.

**Q: Can I delete the old models?**
A: After 6+ months in production and validation that new model performs better, yes. Recommend archiving to backup first.

## Audit Trail

| Epoch | Decision | Reason |
|-------|----------|--------|
| Mixed only | Baseline established | Single domain works well (AUC 0.9999) |
| CTU+MCFP | Cross-dataset tested | Generalization failure discovered (0.71 AUC) |
| UWF fine-tune | Adaptation strategy 1 | Recovered UWF but lost mixed domain |
| Rehearsal+distill | Adaptation strategy 2 | Calibration failed (inf threshold) |
| Multi-task heads | Adaptation strategy 3 | ✅ SUCCESS - Both domains supported |
| Domain-adaptive sweep | Production model | 3-seed validation confirms reproducibility |

---

**Model Status Summary:**

| Category | Count | Recommendation |
|----------|-------|-----------------|
| 🟢 Active | 1 | Use for all new deployments |
| 🟡 Validated Seeds | 3 | Keep for reproducibility proof |
| 🔴 Deprecated | ~40 | Archive or delete after 6 months |
| 📦 Total Space | ~2.5GB | Consider archiving old_experiments/ |

**Last Updated:** May 2026  
**Next Review:** August 2026
