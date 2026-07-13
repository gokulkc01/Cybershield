# CyberShield v2 — Research Plan: Paper-Ready Modern C2 Detection

**Status:** planned (canonical, supersedes the invalidated `phase3_*` cluster)
**Created:** 2026-07-11
**Owner:** Gokul K C (MSRIT) + team
**Execution model:** one phase per working session. Each phase has a hard **gate** — do not proceed until it passes.

---

## Context (why this plan exists)

The original CyberShield paper reports 99.91% recall at 0.00% FPR with AUC = 1.0000 for *all three* models. Verified against `experiments/host_aware_model_benchmark_clean_balanced/comparison_report.json`, those numbers are **not trustworthy**:

- **Provenance confound:** all benign came from UWF-ZeekData24; all C2 from CTU-13 + MCFP. "C2 vs benign" is aligned with "which capture/era," so models can win by recognizing provenance. This is why AUC is exactly 1.0 for even the plain baseline.
- **Not zero-shot family:** test families (`mcfp_botnet_264/345`) also appear in train. The split is host-separated, same-family — not leave-one-family-out.
- **Threshold artifact:** the 89.97% → 99.91% "gain" is threshold placement on one tiny split (baseline AUC is also 1.0), not a representational advantage.
- **Smoke runs as final:** all runs used `epochs=2`; the Red-Agent "100 episodes / 2% evasion" claim maps to a logged run of `epochs=1, batches=2, samples=4, evasion=0.0`.

**Goal:** a research-grade system that accurately detects C2 in *modern* environments, and a paper that survives expert review. **Target:** top-tier (USENIX Security / NDSS / CCS) as north star; de-risk via a milestone ladder (workshop WOOT/AISec/DLS → mid-tier RAID/DIMVA/ACSAC/EuroS&P → top-tier). Structure work so a strong mid-tier paper falls out even if top-tier slips.

---

## Thesis & contributions

**"Detecting modern encrypted C2 under honest and adversarial evaluation."**

- **C1 — Dataset:** a released, same-provenance modern-C2 testbed dataset (Sliver/Havoc/Mythic + realistic same-environment benign, Zeek logs). Literature scan found no equivalent public dataset — genuine whitespace.
- **C2 — Method:** a host-behavior-anchored detector fusing beaconing + JA4/TLS + temporal host context, with a self-supervised benign-behavior component for unseen-family generalization. Differentiated from GESR (graph benign-only) by host-temporal modeling + modern data.
- **C3 — Robustness:** a problem-space *adaptive* adversary (functionality-preserving) **and** adversarial hardening that measurably improves robust recall — engaging the "evasion is impractical against NIDS" debate via cost-to-evade curves.

Self-supervision is a *tool* for C2/C3, **not** the headline (that lane is crowded + contested — see `memory/cybershield-novelty-landscape.md`).

---

## Non-negotiable principles (apply to every phase)

1. **Same-provenance labels** — benign and C2 from the same captures/environment.
2. **True leave-one-family-out**, reported separately from in-distribution.
3. **Realistic base rate** — evaluate at C2 prevalence ~0.1–1%; report PR-AUC, precision@recall, alerts/day — not balanced-set recall@one-threshold.
4. **Multi-seed with 95% CIs** and significance tests (bootstrap / McNemar).
5. **Adaptive attacks** for any robustness claim (white-box, attacker knows the defense).
6. **Follow the Arp et al. "Dos and Don'ts of ML in Computer Security" checklist** and cite it.

**Required reading before writing the paper:** Arp et al. *Dos and Don'ts*; *Debunking Representation Learning for Encrypted Traffic Classification* (ACM 2025); *Evasion Attacks Remain Impractical against ML-NIDS* (arXiv 2306.05494); *GESR* (arXiv 2605.07536, nearest method neighbor).

---

## Phase plan (one session each; gates are mandatory)

### Phase 0 — Honest evaluation harness + same-provenance CTU-13 split  ✅ COMPLETE (2026-07-12)
**Outcome:** gate PASSED with documented caveat — see `docs/validated/phase0_honest_baseline.md`. Baseline test AUC dropped 1.0000 → 0.9670 [0.9640, 0.9701] (5 seeds). Two deviations from the tasks below, forced by the data: (1) all three CTU captures share a single botnet source IP, so the split is **capture-separated** (each split holds benign+C2 from the same capture — strongest provenance break) rather than pooled host-separated; (2) the gate caught a second confound — asymmetric `min_flows` (5 for C2, 1 for benign) made session length a label rule (flow count alone: AUC 0.88–0.93) — fixed with a symmetric policy. Residual baseline AUC ≈ 0.97 is attributable to trivial byte-volume separability of 2011 botnets (`resp_bytes` alone: 0.968), which is the Phase 2 motivation.
**Objective:** break the provenance confound using data we already have, and build an evaluation that reports believable numbers.

**Feasibility confirmed:** CTU-13 raw captures contain `Normal` (clean benign) + `Background` (uncertain) flows in the *same files* as `From-Botnet` (e.g. `data/raw/ctu13/capture20110810.binetflow`: ~29k botnet / ~22k normal / ~1.9M background). The code currently discards them.

**Tasks (surgical):**
- Fix `src/pipelines/build_balanced_clean_benchmark_split.py`:
  - `load_ctu13_c2_dataframe` (~L586-601): remove the C2-only filter at ~L592-594; keep `Normal` rows and assign per-row `label = label_from_string(...)`.
  - `build_ctu13_c2_records` (~L327-371): remove hardcoded `df["label"]=1` (L348); route resulting benign CTU windows into `benign_records`.
  - **Use `Normal` as benign; exclude `Background`** from the labeled set (it's uncertain) — or hold it out as an unlabeled pool for later self-supervision.
  - Reuse as-is: `split_records_by_label_and_host` (L604-628), `select_balanced_current_indices` (L682-715), `label_from_string` (`src/features/feature_config.py` L116-120), and the host-window builder `build_extended_host_session_records_from_dataframe` (`src/data_loader/host_window_dataset.py` L183).
- **Build (MISSING) in `src/evaluation/multifamily_metrics.py`:** bootstrap confidence intervals; base-rate / imbalanced re-weighted evaluation. (PR-AUC, `recall_at_fpr`, `precision_at_recall`, `compute_threshold_confusion` already exist — reuse.)
- **Build (MISSING) in `src/pipelines/benchmark_host_aware_models.py`:** a multi-seed loop (currently single seed at L107-111) that aggregates mean ± 95% CI across ≥5 seeds. Train for real (not `epochs=2`).

**Gate:** on the same-provenance split, **baseline test AUC drops well below 1.0** (target < ~0.95) and a real, above-chance signal survives with non-degenerate CIs. If AUC stays ~1.0, the split is still leaking — stop and fix before anything else.

**Deliverable:** `docs/validated/phase0_honest_baseline.md` with the new split manifest + multi-seed baseline table (mean ± CI).

---

### Phase 1 — Baselines on the honest harness  ✅ COMPLETE (2026-07-12; ablation sweep finishing in background)
**Outcome:** gate PASSED — see `docs/validated/phase1_baselines.md`. Headline: **the floors match the deep models.** A single `resp_bytes` feature ties the c2_transformer on test AUC (0.9681 vs 0.9670); a RandomForest ties the host-aware transformer (0.9902 both), on pure volume statistics. The new Fourier/autocorrelation beaconing detector scores **below chance** (test AUC 0.224) — benign update/keep-alive traffic is more periodic than 2011 botnet C2, and C2 sessions concentrate in low-event pairs. Watch item resolved: the host-aware val saturation is a **host-identity shortcut** (logreg on 15 host summary features alone: val AUC 0.9964; every host in every split is label-pure — one infected host per capture), so the host-aware base-rate advantage is discounted pending the host-context ablation sweep (`experiments/phase1_host_aware_ablations/`). New Phase 2 requirement derived: testbed must break host-label purity (multiple/partially infected hosts) in addition to volume separability.
**Objective:** establish credible reference numbers.
**Tasks:** run (a) RandomForest (`src/models/baseline_rf.py`), (b) a **new Fourier/autocorrelation beaconing detector** (classic C2 timing signal — currently absent), (c) the existing C2Transformer — all multi-seed with CIs on the Phase 0 split.
**Gate:** differences between methods are believable and statistically significant; no method is trivially perfect.
**Deliverable:** baseline comparison table + PR curves.

---

### Phase 2 — Modern-C2 testbed dataset (long pole — start capture early, in parallel)
**Objective:** the C1 contribution — same-provenance modern C2.
**Tasks:** stand up isolated lab; run **Sliver / Havoc / Mythic / Metasploit-over-HTTPS**; emulate Cobalt Strike malleable-profile behaviors (jitter, sleep, domain fronting). Run **realistic concurrent benign** (browser automation, OS/app updates, streaming, sync) across many hosts so "busy vs idle" isn't a shortcut. Capture with **Zeek** (`conn`, `ssl`/`x509`, `dns` logs → JA4/cert/DNS features). Vary beacon config (sleep 30s–24h, jitter 0–50%, HTTP/HTTPS/DNS). Label by construction.
**Gate:** a provenance **sanity control** — a model trained to predict *source/provenance* must NOT trivially predict the *label*; if it does, benign realism is insufficient — fix before use.
**Deliverable:** released dataset + datasheet; integrated via existing Zeek path in `host_window_dataset.py`.

---

### Phase 3 — Method: host-behavior-anchored + self-supervised detector
**Objective:** the C2 contribution.
**Tasks:**
- Extend features to **JA4/JA4S/JARM** + cert/DNS (the extended 45-dim schema already has TLS/DNS slots — upgrade JA3→JA4).
- Add a **self-supervised benign-behavior model** (deep SVDD / masked-flow pretraining on abundant benign) → score C2 as deviation; fuse with the supervised host-aware head.
- Extend `src/training/ablation_runner.py` (currently feature-group only, `_make_variants` L41-47) with **history-length H** and **architecture-component** ablations.
- Evaluate on four **separately reported** axes: in-distribution; **leave-one-family-out** (`FamilyAwareSplitter` + `SplitStrategy.FAMILY_SEPARATED`, `family_splitter.py` L73/L102-110); **cross-environment** (train CTU → test UWF/testbed, report the drop honestly); **temporal**.
**Gate:** beats baselines on **unseen families**, not just in-distribution, with CIs.
**Deliverable:** method + ablation + multi-axis results.

---

### Phase 4 — Adversarial robustness (problem-space + adaptive + hardening)
**Objective:** the C3 contribution.
**Tasks:**
- Reframe the Red-Agent as a **problem-space** adversary (mutations map to real, replayable, functionality-preserving traffic).
- **Build (MISSING): a white-box / gradient (adaptive) attack.** Today the detector is black-box only (`FrozenDetectorAdapter.score_host_window`; the only `.backward()` in `ppo_trainer.py` L181 trains the policy, not gradients through the detector).
- **Adversarial training / hardening** — retrain with adversarial examples; show robust recall improves.
- Run at **real scale** (thousands of mutations, multi-seed) — reuse `orchestrator.batch_evaluate_mutations`, `red_agent_robustness.run_robustness_pipeline`, `threshold_evasion_rate` (`red_agent_host_aware_train.py` L166-171); replace the 4-sample smoke config.
- Report **cost-to-evade** curves (evasion vs perturbation budget), not a single number.
**Gate:** robustness result holds under a white-box adaptive attacker.
**Deliverable:** robustness section with adaptive-attack + hardening results.

---

### Phase 5 — Write, release, red-team
**Objective:** submission.
**Tasks:** write from scratch (honest limitations); release dataset + repro pack (splits, seeds, configs); internal red-team review against the Arp et al. checklist; clean the documentation sprawl (fix broken links, archive invalidated `phase3_*`, remove stale root entry-point docs) so the repo reads as a research record, not a product pitch.
**Gate:** internal review finds no confound / leakage / overclaim.

---

## Infrastructure inventory (reuse vs build)

| Need | Status | Location |
|---|---|---|
| ROC-AUC, PR-AUC, recall@FPR, precision@recall, confusion, per-family | **EXISTS** | `src/evaluation/multifamily_metrics.py` |
| FPR-budget threshold selection | **EXISTS** | `src/evaluation/operating_point.py::find_threshold_under_fpr_budget` |
| Family-separated / LOFO splitting | **EXISTS** | `src/data_loader/family_splitter.py::FamilyAwareSplitter` |
| Host-window build (benign-capable) | **EXISTS** | `src/data_loader/host_window_dataset.py::build_extended_host_session_records_from_dataframe` |
| Per-row CTU label parsing | **EXISTS** | `feature_config.py::label_from_string`; `ctu13_processor.py::apply_ctu13_labels` |
| Base-rate / imbalanced evaluation | **BUILD** | add to `multifamily_metrics.py` |
| Bootstrap confidence intervals | **BUILD** | add to `multifamily_metrics.py` |
| Multi-seed sweep + CI aggregation | **BUILD** | `benchmark_host_aware_models.py` (single-seed today) |
| Fourier/autocorrelation beacon baseline | **BUILD (new)** | `src/models/` |
| History-length H + architecture ablations | **BUILD** | extend `ablation_runner.py::_make_variants` |
| Self-supervised / anomaly detector | **BUILD (new)** | `src/models/` |
| White-box / adaptive attack | **BUILD** | red-agent is black-box only today |
| JA4/JA4S upgrade | **BUILD** | extended feature builder (has JA3 today) |

---

## Context-saving / tooling
- **graphify** installed as a Claude Code skill → regenerate the code knowledge graph after each phase so future sessions load structure without re-exploring.
- **Memory files** (`~/.claude/projects/d--CyberShield/memory/`): `cybershield-research-direction`, `cybershield-eval-confound`, `cybershield-novelty-landscape`.
- **This document** is the cross-session anchor — update the phase statuses as they complete.

## Risks & mitigations
- *Confound survives Phase 0* → the whole premise is wrong; cheap to discover early (that's why Phase 0 is first).
- *Testbed benign not realistic* → provenance sanity control (Phase 2 gate) catches it.
- *Self-supervision underperforms* → keep supervised + LOFO as the fallback contribution.
- *Top-tier slips* → milestone ladder means a mid-tier paper is always in reach.
