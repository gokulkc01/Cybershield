# Phase 0 — Honest Baseline on the Same-Provenance CTU-13 Split

**Status:** COMPLETE — gate **PASSED with a documented caveat** (see §6)
**Date:** 2026-07-12
**Plan:** `docs/planned/research_plan_v2_modern_c2.md` (Phase 0)
**Supersedes:** all numbers derived from `experiments/host_aware_model_benchmark_clean_balanced/` (provenance-confounded, AUC = 1.0000 for every model — see `memory/cybershield-eval-confound.md`).

---

## 1. What Phase 0 set out to do

The original benchmark labeled *all* benign from UWF-ZeekData24 and *all* C2 from CTU-13/MCFP, so "C2 vs benign" was perfectly aligned with "which capture the data came from." Every model — including the plain baseline — scored AUC = 1.0000, which measures provenance recognition, not C2 detection. Phase 0 breaks that confound using CTU-13's own within-capture `Normal` traffic as benign, and builds an evaluation harness that reports believable numbers (multi-seed CIs, bootstrap CIs, base-rate-adjusted metrics).

**Gate:** baseline test AUC must drop well below 1.0 on the same-provenance split, with a real above-chance signal and non-degenerate CIs.

## 2. Harness changes (code)

| Change | Where |
|---|---|
| CTU-13 per-row labels: `From-Botnet` → C2, `From-Normal`/`Normal` → benign; `Background` and destination-derived `To-*` rows excluded | `src/pipelines/build_balanced_clean_benchmark_split.py` (`categorize_ctu13_label`, `load_ctu13_labeled_dataframe`, `build_ctu13_records`) |
| CTU benign routed into `benign_records`; `--skip-uwf-benign` for a CTU-only same-provenance split | same file, `main()` |
| `--split-by-capture` mode with per-capture host namespacing + host-label-purity enforcement | same file (`split_records_by_capture`, `namespace_hosts_by_capture`, `enforce_host_label_purity`) |
| Bootstrap CIs (stratified percentile), prevalence-adjusted precision / PR-AUC / alerts-per-1k-sessions, t-distribution seed aggregation, one-stop `honest_test_report` | `src/evaluation/multifamily_metrics.py` |
| Multi-seed benchmark loop (default seeds 42–46), real training (epochs 30, patience 5 — replacing the `epochs=2` smoke config), mean ± 95% CI aggregation | `src/pipelines/benchmark_host_aware_models.py` |
| Host-aware trainer now exposes `test_scores`/`test_labels` for bootstrap evaluation | `src/training/train_host_aware_domain_adaptive.py` |
| 27 unit tests for all of the above | `tests/test_phase0_honest_eval.py` |

## 3. The split — design decisions forced by the data

**Discovery: a pooled host-separated split is impossible for the C2 class.** All three local CTU-13 captures contain exactly **one botnet source host, and it is the same IP in all three** (`147.32.84.165`). The plan's assumption of splitting C2 hosts across train/val/test cannot hold (the splitter needs ≥3 hosts per class).

**Resolution: capture-separated splits.** Each split takes *all* labeled traffic (benign **and** C2) from one capture. Within every split, both classes share identical provenance, so capture-identity features cannot separate the classes — this is the strongest structural form of the provenance break. Host IDs are namespaced per capture (`capture:ip`), keeping splits host-disjoint.

**Second discovery (caught by the gate — see §5): asymmetric session policy was a label shortcut.** The pipeline originally built C2 sessions with `min_flows=5` but benign with `min_flows=1`, so any session with <5 flows was benign *by construction* (70–87% of benign sessions). Session flow-count alone scored AUC 0.88–0.93. Fixed with a symmetric policy (`min_flows=1` for both classes); after the fix, flow-count alone scores *below chance* (AUC 0.30–0.45).

### Split manifest

Source: `data/processed/host_aware_ctu13_same_provenance/` (metadata: `clean_benchmark_split_metadata.json`)

| Split | Capture | Scenario family | C2 sessions | Benign sessions | Hosts |
|---|---|---|---:|---:|---:|
| train | capture20110810 | Neris | 2,676¹ | 1,784 | 11 |
| val | capture20110811 | Neris (same family, different day) | 560¹ | 373 | 7 |
| test | capture20110815 | **Rbot (unseen family)** | 718 | 1,538 | 11 |

¹ train/val C2 capped by the 1.5:1 balancing policy (train from 3,097 full; val from 1,327 full). Test keeps **all** benign (no balancing) for maximum FPR resolution — test class ratio ≈ 2.1:1.

Raw row provenance (per-capture): 0810 = 40,961 C2 / 30,267 Normal / 2.75M Background excluded; 0811 = 20,941 / 9,084 / 1.78M; 0815 = 2,580 / 25,219 / 1.09M.

Integrity checks (recorded in metadata): `host_disjoint: true`, `conflicting_hosts: 0`, all splits single-source (`ctu13`), real host identity.

**Axes:** val = same-family / cross-capture generalization; test = **cross-family** / cross-capture. These are reported separately below.

## 4. Multi-seed baseline results (5 seeds, mean [95% CI])

Benchmark: `experiments/host_aware_benchmark_ctu13_same_provenance/comparison_report.{json,md}`. Training: 30 epochs max, early stopping patience 5, thresholds frozen on val per FPR budget.

### Test axis (unseen family Rbot, 718 C2 / 1,538 benign)

| Model | Test ROC-AUC | Test PR-AUC | Recall@1.5%FPR | Realized FPR |
|---|---|---|---|---|
| c2_transformer (baseline) | **0.9670 [0.9640, 0.9701]** | 0.9271 [0.9177, 0.9366] | 0.4448 [0.3977, 0.4920] | 0.0074 |
| domain_adaptive_transformer | 0.9717 [0.9685, 0.9748] | 0.9421 [0.9170, 0.9672] | 0.3256 [0.1136, 0.5377] | 0.0038 |
| host_aware_domain_adaptive | 0.9902 [0.9831, 0.9974] | 0.9875 [0.9819, 0.9932] | 0.9114 [0.9091, 0.9137] | 0.0001 |

### Validation axis (same family Neris, different capture day)

| Model | Val ROC-AUC | Val Recall@1.5%FPR |
|---|---|---|
| c2_transformer | 0.9027 [0.8951, 0.9103] | 0.6743 [0.6038, 0.7448] |
| domain_adaptive_transformer | 0.8974 [0.8813, 0.9135] | 0.4971 [0.1294, 0.8649] |
| host_aware_domain_adaptive | 0.9950 [0.9912, 0.9987] | 0.9857 [0.9758, 0.9956] |

### Base-rate (deployment-prevalence) view — the sobering table

Prevalence-adjusted PR-AUC (Bayes-reweighted precision over the ROC sweep), mean [95% CI]:

| Model | π = 0.1% | π = 1% |
|---|---|---|
| c2_transformer | **0.0600 [0.0327, 0.0874]** | 0.3061 [0.2485, 0.3636] |
| domain_adaptive_transformer | 0.1379 [−0.0255, 0.3014]² | 0.4072 [0.2237, 0.5907] |
| host_aware_domain_adaptive | 0.9182 [0.9125, 0.9240] | 0.9365 [0.9299, 0.9431] |

² t-interval on 5 seeds; bounds may exceed [0, 1]. The wide CIs for the domain-adaptive model reflect genuine seed instability at low-FPR operating points.

Concretely, at the 1.5% FPR operating point the *baseline* delivers ~3% precision at π = 0.1% (≈15 alerts per 1k sessions, nearly all false) — the original "99.91% recall at 0.00% FPR" framing does not survive contact with a realistic base rate.

## 5. Shortcut probes (why we believe the numbers)

Run on the final symmetric split, train→test:

| Probe | Test AUC | Reading |
|---|---:|---|
| Session flow count (pre-fix split) | 0.8754 | **Label artifact** — asymmetric `min_flows`; fixed |
| Session flow count (final split) | 0.2980 | Artifact gone (below chance) |
| Single feature: `resp_bytes` | 0.9681 | Trivial byte-volume signal — genuine behavior of 2011 botnets |
| Single feature: `orig_bytes` / `resp_pkts` / `orig_pkts` | 0.966 / 0.964 / 0.957 | same |
| Logistic regression on mean-pooled features | 0.9323 | linear signal ceiling |

The transformer baseline (0.9670) performs at roughly the level of a *single byte-count feature* (0.9681). The remaining separability is not evaluation leakage — provenance is structurally excluded within each split — it is the well-documented fact that classic loud botnets (Rbot: IRC C2, tiny chatty flows, UDP floods) are nearly separable from normal browsing by volume statistics alone.

## 6. Gate verdict

**PASSED, with one caveat.**

- ✅ The degenerate pattern is gone: baseline test AUC fell from exactly **1.0000 → 0.9670 [0.9640, 0.9701]**; recall@1.5%FPR from ~99.9% → **44%**; models are now statistically distinguishable (host-aware separates from both session baselines with non-overlapping CIs on recall).
- ✅ CIs are non-degenerate; results are stable across 5 seeds (per-seed baseline AUC range 0.9645–0.9701).
- ✅ The gate mechanism itself worked: it caught a *second* confound (session-length labeling artifact) that would otherwise have shipped.
- ⚠️ **Caveat:** baseline AUC (0.967) sits above the plan's "< ~0.95" target. The shortcut probes attribute this to intrinsic byte-volume separability of 2011-era botnet traffic (a single feature achieves 0.968), not to residual leakage. We treat this as *era-easiness*, which is precisely the thesis motivation: modern encrypted C2 (Sliver/Havoc/Mythic over TLS with jitter) does not present this trivial volume signature — hence the Phase 2 testbed.
- ⚠️ Watch item: the host-aware model saturates the same-family axis (val AUC 0.9950, seed-42 run hit 1.0000 during training). Not a gate violation (test axis is the meaningful one), but host-level regularities deserve scrutiny in Phase 1 ablations before crediting the architecture.

## 7. Reproduction

```bash
# 1. Build the same-provenance split (capture-separated, symmetric session policy)
python -m src.pipelines.build_balanced_clean_benchmark_split \
  --skip-mcfp --skip-modern-c2 --skip-uwf-benign \
  --split-by-capture "train=capture20110810;val=capture20110811;test=capture20110815" \
  --min-eval-benign 1600 --c2-min-flows 1 --benign-min-flows 1 \
  --out-dir data/processed/host_aware_ctu13_same_provenance

# 2. Multi-seed benchmark (3 models x 5 seeds, real training)
python -m src.pipelines.benchmark_host_aware_models \
  --host_split_dir data/processed/host_aware_ctu13_same_provenance \
  --out_dir experiments/host_aware_benchmark_ctu13_same_provenance \
  --seeds 42,43,44,45,46 --epochs 30 --patience 5

# 3. Unit tests for the harness
python -m pytest tests/test_phase0_honest_eval.py -q
```

The pre-fix (asymmetric `min_flows`) split can be reproduced by dropping the two `--*-min-flows` flags; its diagnostic numbers are in §5.

## 8. Implications for the next phases

1. **Phase 1 (baselines):** add RandomForest and the Fourier/autocorrelation beaconing detector on this split; include the `resp_bytes` single-feature detector and the linear probe as *mandatory floor baselines* — any proposed model must be compared against them, not only against other transformers.
2. **Phase 1 (analysis):** investigate the host-aware model's val-axis saturation (ablate host features / history length) before attributing its test-axis advantage (0.99 AUC, 91% recall@1.5%) to architecture.
3. **Phase 2 (testbed):** this result quantifies the motivation — 2011 botnets are ~solved by byte counts; the modern-C2 dataset must demonstrate that this trivial signal disappears (the provenance sanity control in the plan's Phase 2 gate).
4. **Reporting standard going forward:** every result table must carry multi-seed mean [95% CI], PR-AUC, and the π = 0.1%/1% base-rate view, per the plan's non-negotiables.
