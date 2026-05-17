# CyberShield v2

CyberShield is a behavioral adversarial robustness platform for C2 detection research.

It combines a session-centric Transformer detector, a FastAPI backend, and a Next.js frontend to answer one practical question:

> How robust is the detector when behavioral traffic is mutated in realistic ways, and what does that mean for whether a host is compromised or C2 traffic exists?

This repository now includes:

- A session-level detection pipeline for CTU-13 / related processed datasets.
- A backend API for detection, mutation, robustness, and artifact management.
- A frontend with dashboard, session analysis, upload, robustness analytics, and Mutation Lab pages.
- A Red-Agent adversarial mutation system for realism-constrained robustness testing.
- Training, evaluation, smoke-test, and visualization scripts.

Current branch direction: `research/multifamily_generalization`

Current state: backend/frontend integrated, Mutation Lab wired to Red-Agent evaluation, dataset evaluation now returns raw tensors and masks, and the frontend prefers backend-provided data while still falling back to reconstructed tensors when needed.

## What This Project Is For

CyberShield is not a generic dashboard and not an offensive tool.

It is a research platform for:

- Detecting suspicious network sessions.
- Explaining why a session or host looks compromised.
- Testing how mutations affect detector confidence and recall.
- Measuring whether adversarial changes remain realistic.
- Deciding whether a session-centric detector is enough or whether broader host-centric / temporal modeling is needed.

The primary user-facing question is whether C2 traffic exists and which host looks compromised, and the research question is how reliably the detector holds up under controlled mutation pressure.

## What Has Been Built So Far

### Backend

The backend is a FastAPI application with modular services and typed schemas.

Implemented API areas:

- Detection endpoints for single-session, batch, and dataset evaluation.
- Mutation endpoints for listing, applying, and evaluating mutations.
- Robustness endpoints for reports, metrics, failure analysis, and comparison.
- Artifact endpoints for upload, download, metadata, listing, deletion, and report persistence.

Core backend services:

- Detection service for loading the Transformer detector and running inference.
- Mutation service for Red-Agent integration.
- Robustness service for baseline-vs-mutated comparisons.
- Artifact service for file persistence and retrieval.

### Frontend

The frontend is a Next.js 14 + React 18 + TypeScript app with:

- Dashboard page for high-level metrics.
- Session Analysis page for inspecting individual sessions.
- Mutation Lab page for adversarial testing and before/after comparison.
- Robustness Analytics page for research-grade mutation analysis.
- Upload page for dataset and artifact handling.

Shared frontend infrastructure:

- Zustand store for global state.
- Typed Axios client for backend calls.
- Recharts for visualizations.
- Tailwind CSS for UI styling.

### Red-Agent

The Red-Agent is a domain-constrained adversarial behavioral mutation system.

It includes:

- Mutation engines for timing, flow, TLS, and composite mutations.
- Hard behavioral validity constraints.
- A realism discriminator to avoid unrealistic or reward-hacked mutations.
- A multi-objective reward engine.
- PPO policy training.
- An orchestrator that coordinates mutation → validity → realism → detector inference → reward.

## What We Have Done During This Work

This repository changed from a basic detector demo into a full behavioral adversarial robustness platform.

Recent work includes:

- Fixed backend startup/import issues around `fastapi` and project root resolution.
- Designed and implemented the next-generation Red-Agent architecture.
- Added mutation validity checks and a realism discriminator.
- Added a reward engine and PPO trainer.
- Implemented the Phase 1 training pipeline for Red-Agent against a frozen detector.
- Ran smoke training configurations and compared outputs.
- Generated plots from training logs.
- Extended the backend with a Red-Agent evaluation endpoint.
- Extended the frontend API client to call that endpoint.
- Wired Mutation Lab to send actual tensors instead of placeholder zero arrays.
- Added support for backend-returned raw session tensors and masks.
- Updated Mutation Lab to use backend-provided metrics when available.
- Added multiple editable mutation parameters instead of a single severity-only control.
- Removed hardcoded demo metric fallbacks from the main user flow.

## How To Read The UI

### Dashboard

The Dashboard gives a compact operational view:

- Total sessions observed.
- Suspicious or C2-like sessions.
- Detection rate.
- Behavioral robustness / invariance summary.

Use it to understand whether the dataset contains enough suspicious activity to justify deeper inspection.

### Session Analysis

The Session Analysis page lets you inspect one session at a time.

It helps answer:

- Is this traffic suspicious?
- How confident is the detector?
- Which behavioral features drive the score?
- Does the session resemble C2 activity?

### Upload

The Upload page loads datasets and sends them to the backend for evaluation.

This is usually the starting point for a real analysis session.

### Mutation Lab

This is the centerpiece page.

It is used to:

- Configure mutations.
- Adjust severity and mutation-specific parameters.
- Run Red-Agent evaluation.
- Compare baseline vs mutated detection.
- Inspect robustness metrics.

This page is now wired to use backend tensors when available, and otherwise reconstructs session tensors from session summaries as a fallback.

### Robustness Analytics

This page summarizes how each mutation affects:

- Recall.
- False positive rate.
- Behavioral invariance.
- Fragile samples.
- Mutation impact ranking.

It is the high-level analysis view for deciding which mutations matter.

## What The Main Terms Mean

These terms are the ones most users should understand first.

- Baseline Recall: percent of malicious sessions detected before mutation.
- Mutated Recall: percent detected after mutation.
- Recall Degradation: how much detection drops after mutation.
- Mean Reward: Red-Agent objective value after realism, evasion, and constraint terms are combined.
- Mean Realism: how plausible the mutated session looks.
- Evasion Rate: how often the mutation caused the detector to miss the sample.
- Behavioral Invariance: how well important behavioral constraints survived mutation.
- FPR Shift: how much the false positive rate changed.
- Feature Shift Analysis: which behavioral features moved the most.

How to interpret them:

- High baseline recall and low mutated recall means the model is vulnerable to that mutation.
- High evasion with low realism means the mutation is likely not realistic enough to matter operationally.
- High evasion with high realism means the mutation is a real robustness concern.
- High FPR shift means the detector may start flagging too many benign sessions.
- High behavioral invariance is good; low invariance means the mutation broke important constraints.

## What The User Should Conclude

Use the metrics together, not in isolation.

If a mutation causes recall to drop but realism remains low, the result is mostly a stress test and not a practical risk.

If a mutation preserves realism and drops recall, the detector is genuinely brittle for that behavior.

If behavioral invariance remains high while evasion rises, that is the most interesting case for adversarial robustness research.

The practical conclusion should be:

- Is this mutation realistic?
- Did it reduce detector confidence or recall?
- Did it create false positives elsewhere?
- Does the host still look behaviorally plausible?

That is the real robustness story.

## Data Contract And Pipeline Invariants

These are important implementation rules remembered during the work:

- Keep a single mask convention across the pipeline.
- NPZ schema should be consistent: `X`, `y`, `masks`.
- Preserve the model tensor contract `(B, 20, 12)` unless doing an explicit ablation.
- Threshold tuning must use validation metadata, not test labels.
- Sparse capture sets like UWF-ZeekData24 may need `min_flows=1` to avoid dropping valid C2 activity.

## Architecture Overview

```text
Frontend (Next.js / React / TypeScript)
  - Dashboard
  - Session Analysis
  - Mutation Lab
  - Robustness Analytics
  - Upload
  - Zustand store
  - Axios API client

Backend (FastAPI / Python)
  - Detection API
  - Mutation API
  - Robustness API
  - Artifact API
  - Health/root endpoints

ML Engine
  - Session-centric Transformer detector
  - Red-Agent mutation engines
  - PPO policy trainer
  - Reward engine
  - Realism discriminator

Storage
  - NPZ datasets
  - JSON reports
  - JSONL logs
  - Model checkpoints
```

## Mutation Lab Data Flow

The Mutation Lab flow is now:

1. Load and analyze a dataset on the Upload page.
2. Store `datasetEvaluationResult` in the frontend store.
3. Open Mutation Lab.
4. Build mutation requests from the current mutation list.
5. Use backend raw tensors and masks if they are present.
6. Otherwise reconstruct tensors from session summaries.
7. Send the request to `/robustness/red-agent/evaluate`.
8. Render returned metrics if available.
9. Avoid pretending missing data is a real measurement.

## Red-Agent Training Flow

The training pipeline that was implemented and tested works like this:

1. Load a frozen detector checkpoint.
2. Sample session tensors from the CTU-13 processed data.
3. Apply mutation engines.
4. Validate the mutation against hard constraints.
5. Score realism.
6. Run the detector on original and mutated samples.
7. Compute reward.
8. Update PPO policy.
9. Save policy checkpoint and logs.
10. Generate summaries and plots for comparison.

## Validation And Results So Far

Validation and smoke tests completed during development included:

- Python import verification for backend startup.
- Frontend TypeScript compilation.
- Red-Agent Phase 1 smoke training runs.
- Log comparisons across runs.
- Plot generation for reward, realism, violations, and confidence deltas.
- Frontend-to-backend wiring for Red-Agent evaluation.

Observed smoke-training artifacts included:

- Policy checkpoints.
- JSONL training logs.
- Epoch summaries.
- Plot images for comparison.

Key finding from the smoke comparisons:

- Different batch sizes and checkpoints change realism, evasion rate, and constraint violation behavior.
- Smaller batches can produce different detector sensitivity profiles.
- Realism and constraint compliance remain as important as evasion.

## Recent User-Facing Fixes

These are the specific issues addressed in the frontend/backend integration:

- Removed the dummy all-zero session tensors from Mutation Lab.
- Added backend-provided session tensors and masks to dataset evaluation results.
- Preferred backend metrics when they exist.
- Avoided hardcoded demo recall values in the main display path.
- Added editable mutation parameters so one mutation block can carry multiple controls.
- Hid feature-shift visuals unless the backend actually returns feature-shift data.
- Improved the Mutation Lab request so it can operate on real session data.

## Current Limitations And Caveats

This is the honest status of the platform.

- Some display fields may still be partially synthetic if the backend does not return the specific metric requested.
- Fallback tensor reconstruction is an approximation when raw tensors are unavailable.
- Mutation Lab is only as faithful as the dataset analysis input it receives.
- The best evaluation happens when the backend returns full raw tensors from dataset analysis.
- Some documentation in the repo reflects earlier research directions and should be treated as historical unless explicitly marked current.

## Future Plan

The next major research direction is to train and evaluate the detector on **Sliver C2 traffic**.

Why this matters:

- Sliver is a real-world C2 framework with behavior that can differ from the CTU-13 families used so far.
- Training on Sliver traffic should help answer whether the current session-centric feature set generalizes beyond the datasets already studied.
- It creates a stronger next-step benchmark for robustness, mutation testing, and compromise detection.

What I plan to do next:

- Build or collect a Sliver C2 dataset in the same `(B, 20, 12)` session format, or document the exact feature contract if the feature space changes.
- Train the detector on Sliver traffic and compare performance against the existing CTU-13-based baselines.
- Re-run Mutation Lab and Red-Agent evaluation on Sliver-derived sessions.
- Compare baseline recall, mutated recall, realism, and behavioral invariance across CTU-13 vs Sliver.
- Decide whether the current architecture still holds or whether the Sliver results justify a broader redesign.

Expected outcome:

- If the detector generalizes well to Sliver, that strengthens confidence in the current session-centric design.
- If performance drops sharply, Sliver becomes the next evidence point for broader behavioral, temporal, or host-centric modeling.

## Project Structure

```text
backend/
  app/api/
  app/schemas/
  app/services/
  main.py

frontend/
  app/
  components/
  lib/
  styles/

src/
  analysis/
  data_loader/
  evaluation/
  features/
  losses/
  models/
  red_agent/
  pipelines/
  training/

data/
  raw/
  processed/

experiments/
  baseline/
  transformer/
  red_agent_phase1_*/

docs/
  status notes, plans, reports, and summaries
```

## How To Run

### Backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Tests

```bash
python -m pytest tests/ -q
cd frontend
npx tsc --noEmit
```

## Important Files

- [Mutation Lab](frontend/components/pages/mutation-lab.tsx)
- [API client](frontend/lib/api.ts)
- [Global store](frontend/lib/store.ts)
- [Detection API](backend/app/api/detection.py)
- [Detection schema](backend/app/schemas/detection.py)
- [Feature contract](src/features/feature_config.py)
- [Red-Agent docs](RED-AGENT.md)
- [Architecture guide](ARCHITECTURE_GUIDE.md)

## Where To Read More

If you want the deepest supporting material, these docs already exist in the repo:

- `ARCHITECTURE_GUIDE.md`
- `RED-AGENT.md`
- `FINAL_DELIVERY_REPORT.md`
- `FINAL_SUMMARY.md`
- `docs/implemented/IMPLEMENTATION_COMPLETE.md`
- `COMPLETION_SUMMARY.md`
- `docs/implemented/EXECUTIVE_SUMMARY.md`
- `PROGRESS_CHECKLIST.md`
- `README_RESEARCH_DIRECTION.md`
- `docs/planned/NEXT_PHASE_PLAN.md`

## What Matters Most Right Now

The current goal is to keep the platform honest:

- Real tensors when available.
- Explicit fallbacks when not.
- No hidden demo data pretending to be measurement.
- Mutation controls that actually expose research parameters.
- Metrics that are interpretable instead of decorative.

That is the difference between a demo UI and a usable research tool.

## Final Status

CyberShield v2 is now a full behavioral adversarial robustness platform for C2 research.

It has:

- Working backend APIs.
- Working frontend pages.
- A Red-Agent mutation and training stack.
- Real dataset evaluation flow.
- Robustness visualization and analysis.
- Supporting documentation and experimental artifacts.

The next work should focus on stronger data fidelity, more realistic mutation modeling, and continued end-to-end validation.
