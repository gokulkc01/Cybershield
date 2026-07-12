MODEL DETAILED ARCHITECTURE — Session Transformer, Domain-Adaptive Transformer, Host-Aware Transformer, and Red-Agent (RL)

Purpose
-------
This document provides a deep technical reference for the core model families in CyberShield: the session Transformer (C2Transformer), the domain-adaptive Transformer, the host-aware domain-adaptive Transformer, and the Red-Agent (RL) mutation policy. It explains model internals, input data contracts, training procedures, evaluation, artifacts, and operational considerations. Source pointers to code and scripts are included for reproducibility.

Top-level references (source locations)
--------------------------------------
- Session Transformer source: `src/models/transformer.py`
- Domain-adaptive transformer: `src/models/domain_adaptive_transformer.py`
- Host-aware transformer: `src/models/host_aware_domain_adaptive_transformer.py`
- Red-Agent components: `src/red_agent/` (policy, mutation_registry, orchestrator)
- Training pipelines and example scripts:
  - `src/pipelines/red_agent_host_aware_train.py`
  - `scripts/train_host_aware_red_agent.py`
  - `scripts/demo_red_agent_cli.py`
  - `scripts/demo_red_agent_report.py`
- Demo/Report: `experiments/red_agent_host_aware_smoke/*` (checkpoints, JSONL logs, summary.json)

Schema & dataset contract (high-level)
--------------------------------------
CyberShield uses both session-level and host-history inputs. The host-aware flow expects `.npz` files that include arrays for: current-session windows, host-history windows (causal sequences), labels, and feature names. Typical keys (names vary by pipeline) include:
- `session_window` (float32): per-sample session tensor shaped [N, T_session, F_session]
- `host_history` (float32): per-sample causal host-history shaped [N, T_history, F_host]
- `labels` (int/float): per-sample binary/score labels [N]
- `feature_names` (object/str list): names of features

Note: exact array names and shapes depend on the dataset builder step: see `src.pipelines.build_host_aware_mvp_dataset` usage in README and scripts. If you need exact shapes, open the `.npz` produced under `data/processed`.

Common evaluation metrics
-------------------------
- Recall/TPR at fixed FPR (C2 detection tasks focus on high recall while constraining false positive rates)
- ROC AUC
- Precision/Recall / F1
- Evasion Rate (for Red-Agent): percentage of mutated samples whose detector score falls below detection threshold
- Mean probability delta: average change in detector confidence caused by mutation
- Realism / Constraint violation rate (for Red-Agent): fraction of mutations that violate realism constraints or host-sanity checks

1) Session (C2) Transformer (C2Transformer)
-------------------------------------------
Overview
- Purpose: Learn to detect C2 behavior from a time-window of session telemetry. Acts as the primary detector for session-only inputs.
- Business value: Strong per-session discrimination for suspicious/benign flows; used as first-stage filter.

Core architecture
- Backbone: Transformer encoder applied across session time steps (temporal tokens).
- Tokenization: Each time-step is a vector of engineered features (flows, bytes, packet counts, ports, TLS flags, durations, etc.). This is the session `T_session x F_session` matrix.
- Positional encoding: either learned or sinusoidal positional embeddings added to tokens.
- Encoder blocks: Multi-head self-attention (MHSA) layers followed by feed-forward MLP blocks, layernorm, and residual connections. Typical hyperparameters used in the codebase:
  - hidden_dim (d_model): 128–512 (varies by experiment)
  - n_heads: 4–8
  - n_layers: 2–6
  - ff_dim: 4*d_model
- Head: final pooled representation (CLS or mean pooling) passed to a calibrated head that outputs a C2 score (logit), optionally with temperature scaling or Platt calibration (domain-adaptive heads handle per-domain calibration).

Input preprocessing
- Session windows are normalized per-feature using dataset-level statistics (mean/std) computed during dataset build.
- Categorical features (ports, TLS versions, etc.) encoded as one-hot or embedded features prior to concatenation into token vectors.

Training procedure
- Script: `python -m src.training.train_transformer` (see README and `src/training/train_transformer.py` implementations)
- Loss: Binary cross-entropy / focal loss variants (experiments include focal loss in `transformer_focal_*` experiments).
- Optimizer: Adam or AdamW
- Learning rate: common example `3e-4` (see `DEPLOYMENT.md` and experiment sweep files)
- Batch size: 128 (varies by experiment)
- Scheduler: optional step LR or ReduceLROnPlateau
- Epochs: 5–50 (depends on dataset size and early stopping)
- Regularization: weight decay, dropout in Transformer blocks
- Mixed precision: optional via AMP
- Logging: per-epoch metrics saved to `experiments/<run>/` and best checkpoint saved as `best_transformer.pth`

Evaluation & deployment
- Calibrate thresholds using validation split to achieve operating point (FPR threshold).
- Checkpoints are loaded via `torch.load()`; detectors constructed from `C2Transformer` class in `src/models/transformer.py`
- Example evaluation run: `python -m src.evaluation.evaluate_transformer --checkpoint experiments/transformer/best_transformer.pth --test_npz data/processed/..`

Important implementation notes
- The repo contains variants: `transformer_focal_*`, `transformer_ablate_*` for ablation studies, and `transformer_uwf_only` for dataset-specific runs.
- Keep deterministic seeds when comparing experimental results (see experiment scripts and model sweep pipelines).

2) Domain-Adaptive Transformer
------------------------------
Overview
- Purpose: A shared Transformer backbone trained across multiple domains (network families / datasets) with per-domain calibration heads. The goal is to improve generalization across deployment environments by learning shared representations and per-domain specialization.
- Business value: Robust multi-domain detectors; enables single model serving multiple environments with per-host-domain calibration.

Architecture
- Backbone: Same Transformer encoder as Session Transformer (`src/models/domain_adaptive_transformer.py` builds on `transformer.py` concepts) but the head is replaced with a set of domain-specific heads.
- Domain heads: Small MLPs taking backbone pooled representation and outputting domain-calibrated logits. Heads can be temperature-scaled or use domain-specific bias terms.
- Training modes:
  - Joint training: backbone parameters updated from mixed-domain batches with domain labels provided per sample.
  - Fine-tune mode: load backbone from a pre-trained session Transformer and train domain heads on smaller domain-labeled datasets.

Input & domain signals
- Input: session windows as above. Additionally, dataset or domain id per sample (e.g., `domain_id`) used for routing to the head.
- Domain metadata: keep mapping of domain name -> head index in experiment metadata.

Training procedure
- Script: `python -m src.training.train_domain_adaptive_transformer` (README references)
- Loss: aggregate cross-entropy or domain-weighted loss to prevent dominant-domain collapse
- Batch composition: domain-balanced batches or sampling strategies (mixing ratios)
- Hyperparameters: similar to session transformer, but recommended to perform domain sweep for learning rate and mixing coefficients

Evaluation & deployment
- Evaluate per-domain ROC/Recall at FPR
- For deployment, pick a single backbone artifact and store per-domain calibration parameters in a sidecar store or configuration file. The `DEPLOYMENT.md` lists best checkpoints like `experiments/domain_adaptive_sweep_20/best_transformer.pth`.

3) Host-Aware Domain-Adaptive Transformer (Host-Aware Transformer)
-----------------------------------------------------------------
Overview
- Purpose: Extend the domain-adaptive Transformer to incorporate causal host history. Instead of only considering the current session window, the model also consumes a sequence of previous session encodings or host-history tokens to better detect distributed/low-and-slow C2 that unfolds across sessions on the same host.
- Business value: Improves detection of sophisticated C2 that distributes evidence across time and hosts by using host-context.

Architecture
- Two-stream encoder:
  - Session stream: same Transformer encoder applied to current session tokens.
  - Host-history stream: encoder (often a smaller Transformer or causal transformer) applied to the host's previous session embeddings or raw host-history windows.
- Fusion layer: outputs of session and host-history encoders are fused via concatenation or cross-attention; the fused representation is passed to domain-adaptive heads.
- Causality: host-history encoder processes historical windows in causal order to avoid information leakage from future host activity.
- Implementation: `src/models/host_aware_domain_adaptive_transformer.py` is the source-of-truth.

Input contract
- `host_npz` must contain both `session_window` and `host_history` arrays. Host history length (number of past windows) is configurable at data-build time. Typical contract:
  - `session_window`: shape [N, T_session, F_session]
  - `host_history`: shape [N, T_history, F_history] or [N, T_history, D] where host-history is pre-embedded

Training procedure
- Script: `python -m src.pipelines.red_agent_host_aware_train` (also `train_host_aware_red_agent.py` reference for smaller setups)
- Loss and optimizer: same as above (BCE/CE variants)
- Data pipeline: host-aware dataset builder constructs paired (session, host_history) examples and writes `.npz` files referenced by `--host_npz` flag in the pipeline
- Hardware: host-aware inputs increase compute and memory; tune `batch_size` and consider gradient accumulation
- Regularization: consider temporal dropout, masking of historical tokens for robustness

Evaluation & artifacts
- Save artifacts under `experiments/host_aware_*/`:
  - `policy` artifacts for Red-Agent (if training RL afterwards)
  - `best_host_aware_transformer.pth` checkpoint for detection
  - `host_aware_phase1_summary.json` per-phase summary used by demo/report scripts

Known limitations
- Host-aware pipelines require host history data; building such datasets is heavier and the repo contains notes about missing `host_window_dataset.py` or important host-aware builder modules (PRODUCTION_READINESS_AUDIT references)

4) Red-Agent (RL) — Mutation Policy (PPO-based)
-----------------------------------------------
Overview
- Purpose: A research Red-Agent that learns to propose realistic feature-space mutations to evade detectors (the Transformer's detector). It is not an on-network attacker; it operates in feature space for robustness evaluation.
- Business value: Quantify model robustness, produce mutation strategies used for adversarial testing, and identify weak spots in detectors.

Core components
- Mutation registry (`src/red_agent/mutation_registry.py`): defines atomic mutation operators (timing, flow, TLS, persistence, composites). Each operator transforms session feature vectors under realism constraints.
- Detector adapter (`src/red_agent/detector_adapters.py` / `FrozenDetectorAdapter`): wraps a trained detector (e.g., host-aware Transformer) so the RL agent receives detector scores as part of the reward computation.
- Policy network (`src/red_agent/policy.py` / `MutationPolicyNetwork`): maps session features to a distribution over discrete mutation types and continuous severity parameters. Typical architecture: MLP or small Transformer + heads for discrete action logits, severity mean, and value head for baseline (actor-critic design).
- PPO trainer (`src/red_agent/training/ppo_trainer.py`): implements Proximal Policy Optimization to update policy network using collected rollouts.
- Orchestrator (`src/red_agent/orchestrator.py`): applies policy-selected mutations to session samples, enforces constraints, and routes mutated samples to the detector adapter to evaluate rewards.

Action space & realism
- Discrete mutation types: e.g., timing shift, port change, TLS flag toggle, payload obfuscation. Each defined in `src/red_agent/mutations/`.
- Continuous parameters: severity, scale of mutation; policy may produce real-valued severity which is clipped and rounded according to mutation contract.
- Realism constraints: rules that reject impossible mutations (e.g., cannot increase bytes without adding packets) — enforced by mutation engine and reflected in reward as penalty when violated.

Reward function
- Multi-objective reward combining:
  - Evasion reward: positive when detector confidence reduces or sample crosses below operating threshold
  - Realism reward: positive when mutations are realistic, penalize constraint violation
  - Compactness / stealth penalty: penalize high-severity or unrealistic mutations
- Reward engine implemented in `src/red_agent/training/reward_engine.py` (multi-objective weighting configurable)

Training procedure (host-aware Red-Agent example)
- Script: `python -m src.pipelines.red_agent_host_aware_train` (pipeline) or `scripts/train_host_aware_red_agent.py` for smaller runs
- Example CLI:
  python -m src.pipelines.red_agent_host_aware_train \
    --checkpoint_path experiments\host_aware_domain_adaptive\best_host_aware_transformer.pth \
    --host_npz data\processed\host_aware_clean_balanced_benchmark\train_host_windows.npz \
    --output_dir experiments\red_agent_host_aware_phase1 \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 3e-4

- High-level steps:
  1. Load detector checkpoint into FrozenDetectorAdapter
  2. Initialize MutationPolicyNetwork (e.g., input_dim=feature_count, hidden_dim=256, num_mutations=M)
  3. For each episode / epoch:
     - Sample batch of session+host examples
     - Policy selects mutation action (mut_type, severity)
     - Orchestrator applies mutation (subject to constraints)
     - Mutated sample scored by detector; compute reward
     - Store transitions in buffer
     - Run PPO update steps (policy loss, value loss, entropy)
     - Log metrics to JSONL and periodically checkpoint
  4. After training: save `policy.pth`, `training_log.jsonl`, and `summary.json`

PPO details & hyperparameters
- Policy architecture: small MLP or shallow Transformer followed by heads
- Value head: scalar baseline for advantage estimation
- Loss composition: clipped surrogate loss (PPO), value MSE loss, entropy bonus
- Typical hyperparameters used in scripts:
  - learning_rate: 3e-4
  - batch_size (rollout): 16–256
  - epochs: 3–100 (depending on compute)
  - gamma (discount): 0.99
  - lam (lambda for GAE): 0.95
  - clip_eps: 0.2
  - entropy_coef: 0.01
  - value_coef: 0.5
  - num_mutations: small discrete set (e.g., 5)
- Implementation specifics live in `src/red_agent/training/ppo_trainer.py`.

Logging and demo artifacts
- Training logs: newline-delimited JSON events (JSONL) saved per step/epoch (used by `scripts/demo_red_agent_report.py`)
- Summary JSON: aggregated metrics saved as `host_aware_phase1_summary.json` (fields include mean_reward, mean_realism, threshold_evasion_rate, mutation_summaries)
- Demo generation: `scripts/demo_red_agent_report.py` converts logs+summary into markdown and JSON used by the frontend demo endpoint.

Safety and ethics
- Red-Agent is a research tool for offline robustness evaluation. It should not be used to synthesize live network attacks. Enforce access control and logging; never publish raw mutation policies without red-team approvals.

Operational considerations & compute
- Host-aware training is memory intensive. Use GPU(s) and tune `batch_size` and gradient accumulation.
- For reproducibility, capture seed, environment (Python, PyTorch versions), and experiment config in `experiments/<run>/params.json`.

Debugging checklists
- Detector loads: run `python -c "import torch; torch.load('experiments/.../best_transformer.pth')"` to verify checkpoint loads
- Dataset sanity: inspect `.npz` keys and shapes:
  ```python
  import numpy as np
  data = np.load('data/processed/host_aware_mvp/train_host_windows.npz')
  print(data.files)
  print(data['session_window'].shape)
  ```
- Run smoke training with small epochs and batch sizes to surface issues quickly.

Recommended hyperparameter defaults (starting point)
- Detector (Transformer): lr=3e-4, batch_size=128, epochs=10, d_model=256, n_heads=8, n_layers=4
- Domain-adaptive: same backbone, domain-balanced sampling, temperature calibration per domain
- Host-aware: reduce batch_size by factor 2, d_model=256, ensure host_history length <= 8–16 tokens
- Red-Agent (PPO): lr=3e-4, batch_size=32 (rollout), gamma=0.99, clip_eps=0.2, hidden_dim=256, num_mutations=5

Artifacts and naming conventions
- Detector checkpoint: `experiments/.../best_transformer.pth` or `best_host_aware_transformer.pth`
- Red-Agent policy: `host_aware_phase1_policy.pth` or `policy.pth`
- Training log: `*_training_log.jsonl`
- Summary: `*_summary.json`

References to scripts and examples
- Train session transformer: see README section "Train Session Transformer" and `src/training/train_transformer.py`
- Train domain-adaptive: README "Train Domain-Adaptive Transformer" and `src/training/train_domain_adaptive_transformer.py`
- Train host-aware: README "Full Host-Aware Red-Agent Pipeline" examples
- Train Red-Agent PPO: `scripts/train_host_aware_red_agent.py` and pipeline `src.pipelines.red_agent_host_aware_train`
- Demo CLI: `scripts/demo_red_agent_cli.py` — useful interactive walkthrough combining policy, orchestrator, and FrozenDetectorAdapter (also prints sanity checks and performs small PPO update steps)

Appendix A — Example commands
----------------------------
Start backend dev server:
```
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
Start frontend dev server:
```
cd frontend
npm run dev
```
Quick host-aware smoke training (example):
```
.\.venv\Scripts\Activate.ps1
python -m src.pipelines.red_agent_host_aware_train \
  --checkpoint_path experiments\host_aware_domain_adaptive\best_host_aware_transformer.pth \
  --host_npz data\processed\host_aware_mvp\test_host_windows.npz \
  --output_dir experiments\red_agent_host_aware_smoke \
  --epochs 3 \
  --batch_size 16
```
Generate demo report from a completed run:
```
python scripts/demo_red_agent_report.py --summary experiments/red_agent_host_aware_smoke/host_aware_phase1_summary.json --out-md experiments/red_agent_host_aware_smoke/red_agent_demo_report.md --out-json experiments/red_agent_host_aware_smoke/red_agent_demo_report.json
```

Appendix B — Where to fill exact numbers
----------------------------------------
- Exact model dimensions (d_model, number of layers) live in `src/models/*.py` files — inspect constructors for defaults
- Exact dataset shapes and NPZ key names produced by builders: inspect `src/pipelines/*` and `src/data_loader/*`
- Hyperparameter sweeps and experiment configs: `experiments/*/params.json` or experiment launch scripts; search for `--learning_rate`, `--batch_size`, etc.

End of document.
