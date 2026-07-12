# CyberShield Production Readiness Audit

Audit date: 2026-06-09  
Auditor stance: senior software architect, staff engineer, DevOps engineer, technical product lead  
Repository: `D:\CyberShield`

## 1. Executive Summary

### What This Project Does

CyberShield is a defensive C2 detection and adversarial robustness research
platform. It combines:

- Python ML research code for flow/session feature extraction, Transformer
  models, domain-adaptive models, host-aware models, evaluation, and data
  pipelines.
- A Red-Agent framework that mutates feature-space telemetry to measure model
  robustness.
- A FastAPI backend exposing detection, artifact upload, mutation,
  robustness, and model prediction endpoints.
- A Next.js frontend for demo workflows: upload data, inspect sessions,
  evaluate mutations, and visualize robustness.

The useful core idea is strong: defensive behavioral C2 model evaluation under
controlled feature-space mutation. The repository state, however, is not a
production candidate. It is a research/demo MVP mixed with generated files,
experimental artifacts, duplicate docs, partially broken host-aware code, and
missing operational foundations.

### Current Maturity Level

Current maturity: **MVP / research prototype**.

It is not Beta, Production Candidate, or Production Ready.

Reason:

- Backend and frontend have meaningful implemented surfaces.
- Lint passes for the frontend.
- A top-level README now explains the system.
- Core tests do not pass because `src/data_loader/host_window_dataset.py` is
  empty.
- Frontend type-check fails.
- Frontend test docs describe tests that are not present.
- Artifact storage contains huge local runtime data.
- There is no CI/CD, containerization, auth, monitoring, retention policy,
  database, object storage, model registry, or reproducible environment lock.
- Several API paths are placeholders or demo-only.

### Overall Architecture Assessment

The repository currently contains three products in one:

1. Research ML workbench.
2. Demo web application.
3. Planned production inference/robustness platform.

Those concerns are not cleanly separated. The result is high cognitive load,
unclear ownership, and fragile workflows. The ML/research layer has useful
work, but the app layer should not be productionized until the repository is
split into explicit packages, generated artifacts are removed, runtime storage
is externalized, and the broken host-aware contract is restored.

### Biggest Risks

| Risk | Severity | Why It Matters |
| --- | --- | --- |
| `src/data_loader/host_window_dataset.py` is zero bytes | Critical | Host-aware training, host-aware data building, Red-Agent host-aware training, and tests fail at import time. |
| Frontend type-check fails | Critical | The web app cannot be considered build-clean. |
| `backend/artifacts/` contains huge uploaded data | Critical | Runtime artifacts are mixed into the repo; one file is 1.34 GB. |
| `.gitignore` can hide `src/models/` | Critical | The broad `models/` ignore pattern hides source files from normal `rg --files` views and may hide them from git depending on tracking state. |
| No production storage/auth/quota | High | Upload endpoints accept large data into local disk with JSON metadata and no auth. |
| Unsafe artifact loading patterns | High | `torch.load(weights_only=False)`, `np.load(... allow_pickle=True)`, `joblib.load`, and pickle usage are only safe for trusted artifacts. |
| Duplicate/obsolete API surfaces | High | `src/api/predict.py` duplicates backend prediction logic and enables wildcard CORS. |
| Placeholder robustness endpoints | High | Some user-visible endpoints return static/sample data or unimplemented comparison results. |
| Documentation says "production-ready" while code is not | High | Misleads users and future maintainers. |
| No CI/CD or deployment boundary | High | No repeatable build/test/release path. |

### Estimated Effort to Reach Production Readiness

Assuming the goal is a single production-grade inference and robustness web
application, not a fully polished research platform:

- Repository triage and cleanup: 2-4 days.
- Restore broken host-aware module or explicitly remove host-aware workflows:
  2-5 days.
- Make tests/type-check/build green: 3-7 days.
- Stabilize backend APIs, storage, auth, and model config: 2-3 weeks.
- Containerization, CI/CD, observability, deployment docs: 1-2 weeks.
- Security review, load testing, model cards, runbooks: 1-2 weeks.

Estimated effort: **6-10 weeks** for a production candidate.

If host-aware training and Red-Agent training are in production scope:
**10-14 weeks**.

## Evidence Collected

Commands and checks run during audit:

| Check | Result |
| --- | --- |
| Repository file inventory with `rg --files` | About 278 non-ignored files visible. |
| Top-level workspace listing | Many ignored/generated directories exist: `.venv`, `venv`, `.tmp_pytest_*`, `data`, `experiments`, frontend `node_modules`, `.next`, coverage. |
| Frontend lint | `npm run lint` passed. |
| Frontend type-check | `npm run type-check` failed: `tsconfig.json(11,31): error TS5103: Invalid value for '--ignoreDeprecations'.` |
| Python smoke test through system Python | Failed because system Python lacks pytest. |
| Python smoke test through `.venv` | Failed collecting `tests/test_host_window_dataset.py`: cannot import `HOST_AWARE_SCHEMA_VERSION` from empty `host_window_dataset.py`. |
| Host-window source file | `src/data_loader/host_window_dataset.py` length is `0`. |
| Frontend tests directory | Contains only `frontend/__tests__/README.md`; referenced test file is absent. |
| Backend artifacts size scan | `backend/artifacts/` contains many NPZ files, including a 1.34 GB upload. |
| Ignore rule check | `.gitignore` has broad `models/`; `rg --files` hides `src/models`, while `rg --files -u src/models` reveals model source files. |

## 2. Repository Audit

### Major Folder and File Audit

| Path | Purpose | Used? | Risk | Action |
| --- | --- | --- | --- | --- |
| `README.md` | Main project guide | Yes | Low | Keep, but update after cleanup to match final structure. |
| `PRODUCTION_READINESS_AUDIT.md` | This audit | Yes | Low | Keep until roadmap execution is complete. |
| `.gitignore` | Ignore generated files | Yes | High | Refactor. Add `backend/artifacts/`, frontend generated paths, pytest temp dirs. Replace broad `models/` with root-scoped `/models/` or artifact-specific rules. |
| `pytest.ini` | Python test config | Yes | Medium | Keep. Add cache dir override away from denied `.pytest_cache` or fix permissions. |
| `backend/` | FastAPI app | Yes | High | Keep and refactor for production boundaries. |
| `backend/main.py` | API entry point | Yes | Medium | Keep. Change launch/import layout for package-safe deployment. |
| `backend/app/api/` | REST routes | Yes | High | Keep but harden. Remove placeholders, add auth/deps, request limits, real persistence. |
| `backend/app/services/` | Business logic | Yes | High | Keep but split ML inference, artifact storage, mutation, robustness into testable modules. |
| `backend/app/schemas/` | Pydantic contracts | Yes | Medium | Keep. Tighten validation and schema versions. |
| `backend/app/core/config.py` | Environment config | Yes | Medium | Keep but replace ad hoc settings class with Pydantic Settings. |
| `backend/requirements.txt` | Backend runtime deps | Yes | Medium | Split into `requirements-api.txt`, `requirements-ml.txt`, `requirements-dev.txt` or move to `pyproject.toml`. |
| `backend/artifacts/` | Runtime uploads | Yes at runtime | Critical | Remove from repo workspace, add ignore rule, replace with object storage or dedicated local data dir. |
| `src/` | Research/ML package | Yes | High | Keep but restructure into package boundaries. |
| `src/models/` | Model definitions | Yes | Critical | Keep. Fix ignore rule hiding this directory. |
| `src/data_loader/` | Data IO and transformations | Yes | Critical | Keep and repair `host_window_dataset.py`; remove stale duplicate loaders after consolidation. |
| `src/features/` | Feature extraction/schemas | Yes | Medium | Keep. Consolidate v1/v2/extended schema docs. |
| `src/training/` | Training entry points | Research | Medium | Keep as research package, not production app dependency. |
| `src/evaluation/` | Evaluation utilities | Research/API support | Medium | Keep core metrics; archive one-off scripts if not covered by tests. |
| `src/inference/` | Exported inference support | Yes | High | Keep and make this the only inference library used by backend. |
| `src/api/predict.py` | Standalone FastAPI prediction app | Maybe historical | High | Remove or archive. Duplicates `backend/app/api/predict.py` and has wildcard CORS. |
| `src/red_agent/` | Mutation and robustness engine | Yes | Medium | Keep core. Separate demo/training-only pieces from production-safe evaluator. |
| `src/pipelines/` | Data/train/eval pipeline scripts | Research | High | Keep only reproducible pipelines; archive experimental/obsolete one-offs. |
| `src/analysis/` | Score/drift diagnostics | Research | Low | Keep under `tools/analysis` or `research/analysis`. |
| `src/losses/` | Focal loss | Yes | Low | Keep. |
| `scripts/` | Demos and plot scripts | Mixed | Medium | Keep selected scripts under `tools/`; archive demos not needed for prod. |
| `tests/` | Python tests | Yes | High | Keep. Restore failing contracts and add API integration tests. |
| `frontend/` | Next.js app | Yes | High | Keep but fix type-check, remove generated files, add real tests. |
| `frontend/app/` | Next app shell | Yes | Medium | Keep. |
| `frontend/components/` | UI pages/layout | Yes | Medium | Keep. Refactor demo-only data paths into typed API hooks. |
| `frontend/lib/` | API client/store/utils | Yes | Medium | Keep. Remove unused imports and add generated API types. |
| `frontend/styles/` | Global CSS | Yes | Low | Keep. |
| `frontend/__tests__/` | Test docs only | No actual tests | High | Add tests or remove misleading README. |
| `frontend/coverage/` | Generated coverage report | No | Medium | Delete and ignore. |
| `frontend/.next/` | Generated Next build | No | Medium | Delete and ignore. |
| `frontend/.swc/` | Generated compiler cache | No | Low | Delete and ignore. |
| `frontend/node_modules/` | Installed packages | No source | Medium | Delete from repo workspace when packaging; keep ignored. |
| `frontend/tsconfig.tsbuildinfo` | Generated TypeScript cache | No | Low | Delete and ignore. |
| `docs/` | Documentation library | Mixed | High | Consolidate heavily. Keep few canonical docs; archive historical planning. |
| `docs/figures/` | Report figures | Useful for papers | Low | Move under `docs/research/figures` or archive with reports. |
| `docs/model-demo.html` | Static demo artifact | Probably obsolete | Medium | Archive or delete if frontend supersedes it. |
| `data/` | Raw/processed datasets | Runtime/research | High | Keep outside git and outside app repo. Manage through data registry. |
| `experiments/` | Model/training outputs | Research artifacts | High | Keep outside git. Promote only one blessed export through model registry. |
| `notebooks/` | Notebook folder | Empty in current listing | Low | Delete if empty; otherwise archive research notebooks. |
| `.venv/`, `venv/` | Local virtual environments | No | Medium | Delete from repo workspace; ignore. |
| `.pytest_cache/`, `.pytest_tmp`, `.tmp_pytest*` | Test caches/temp dirs | No | Medium | Delete after fixing permissions; ignore all temp patterns. |
| `results.json` | Large root output, about 19 MB | No | High | Delete or archive under external experiment artifacts. |
| `test_results_multi_dataset.json` | Root test output | No | Medium | Delete/archive. |
| `test_multi_dataset.py` | Root ad hoc test/script | Maybe | Medium | Move into `tests/` if valid; otherwise archive/delete. |
| `check_*.py` root files | One-off data inspection scripts | Maybe historical | Medium | Move to `tools/data_checks/` if still useful; otherwise delete. |
| `monitor_build.py` | One-off helper | Unknown | Low | Move to `tools/` or delete if unused. |
| `tmp_compare_logs.py` | Temporary helper | No | Medium | Delete. |

### Root Markdown and Miscellaneous Files

| Path | Purpose | Used? | Risk | Action |
| --- | --- | --- | --- | --- |
| `START_HERE.md` | Old onboarding | Redundant | Medium | Merge into README then archive/delete. |
| `QUICK_REFERENCE.md` | Old quick reference | Redundant | Medium | Merge into README or `docs/runbook.md`. |
| `DEMO_QUICKSTART.md` | Demo instructions | Useful | Medium | Merge into `docs/runbook.md`. |
| `DEPLOYMENT.md` | Deployment notes | Useful but incomplete | High | Replace with real `docs/operations/deployment.md`. |
| `DOCUMENTATION_INDEX.md` | Doc index | Useful temporarily | Medium | Replace after docs cleanup. |
| `PLATFORM_IMPLEMENTATION_GUIDE.md` | Long implementation guide | Useful but redundant | Medium | Merge critical parts into architecture/API docs; archive rest. |
| `FRONTEND_DEMO_SUMMARY.md` | Frontend demo summary | Historical | Low | Archive. |
| `IMPLEMENTATION_COMPLETE_FRONTEND.md` | Completion summary | Historical | Low | Archive. |
| `MODEL_CLEANUP.md` | Experiment/model cleanup notes | Useful | Medium | Merge into model registry/model card; archive original. |
| `PHASE_1_DATA_GATE_ANALYSIS.md` | Research planning/status | Historical | Low | Archive. |
| `PROGRESS_CHECKLIST.md` | Planning checklist | Historical | Low | Archive. |
| `README_RESEARCH_DIRECTION.md` | Research correction narrative | Historical/useful | Low | Archive under `docs/archive/research_direction/`. |
| `project_explanation.md` | Long project explanation | Redundant | Medium | Merge essential content into README/architecture; archive. |

## 3. Documentation Audit

The documentation set is too large, repetitive, and self-contradictory. Many
documents claim "production-ready" while live verification shows broken tests,
failed type-check, placeholder endpoints, and runtime artifacts in the repo.
That erodes trust.

### Keep

Keep these as canonical or near-canonical docs after edits:

- `README.md` - canonical entry point.
- `PRODUCTION_READINESS_AUDIT.md` - current execution plan.
- `DEPLOYMENT.md` - keep content only if rewritten into a real deployment
  runbook.
- `MODEL_CLEANUP.md` - keep content only as source for a real model registry.
- `frontend/TESTING.md` - keep after correcting missing test references.
- `docs/validated/REPRODUCIBILITY.md` - keep as research reproducibility base.
- `docs/validated/EXPERIMENT_RESULTS_STRICT_MULTIFAMILY.md` - keep as a
  validated result record.
- `docs/validated/EXPERIMENT_SUMMARY.md` - keep if it is the concise current
  result summary.
- `docs/implemented/BACKEND_ARCHITECTURE.md` - keep only after updating to
  match actual backend and removing overclaims.
- `docs/implemented/ARCHITECTURE_GUIDE.md` - keep only after encoding cleanup
  and consolidation.
- `docs/status/README.md` - keep temporarily until docs are reorganized.

### Merge

Merge these into fewer professional docs:

- `START_HERE.md`, `QUICK_REFERENCE.md`, `DEMO_QUICKSTART.md`,
  `docs/implemented/README_PLATFORM.md` -> merge into `README.md` and
  `docs/runbook/local-development.md`.
- `PLATFORM_IMPLEMENTATION_GUIDE.md`,
  `docs/implemented/IMPLEMENTATION_COMPLETE.md`,
  `docs/implemented/EXECUTIVE_SUMMARY.md`,
  `docs/implemented/FINAL_DELIVERY_REPORT.md`,
  `IMPLEMENTATION_COMPLETE_FRONTEND.md`,
  `FRONTEND_DEMO_SUMMARY.md` -> merge useful facts into
  `docs/architecture.md` and `docs/product-overview.md`; archive the rest.
- `MODEL_CLEANUP.md`, `DEPLOYMENT.md`, and deployment/model sections in
  README -> merge into `docs/model-registry.md` and
  `docs/operations/deployment.md`.
- `PHASE_1_DATA_GATE_ANALYSIS.md`,
  `docs/planned/DATA_STRATEGY.md`,
  `docs/planned/DATA_FIRST_STRATEGY.md`,
  `docs/planned/PHASE_1_DATA_ACQUISITION.md` -> merge into one
  `docs/research/data-strategy.md`.
- `docs/validated/project_status_v2_2026-05-08.md`,
  `docs/validated/project_status_2026-04-13.md`,
  `docs/status/*.md` -> merge into one `docs/status.md`.

### Archive

Archive historical planning under `docs/archive/`:

- `README_RESEARCH_DIRECTION.md`
- `PROGRESS_CHECKLIST.md`
- `PHASE_1_DATA_GATE_ANALYSIS.md`
- `project_explanation.md`
- `docs/validated/FINAL_SUMMARY.md`
- `docs/validated/COMPLETION_SUMMARY.md`
- `docs/validated/POST_EXPERIMENT_SUMMARY.md`
- `docs/validated/implementation_summary_phase2.md`
- `docs/validated/phase3_status_summary.md`
- `docs/validated/zero_shot_validation_report.md`
- `docs/planned/DIRECTION_CHANGE.md`
- `docs/planned/IMPLEMENTATION_KICKOFF.md`
- `docs/planned/IMMEDIATE_ACTION_PLAN.md`
- `docs/planned/NEXT_PHASE_PLAN.md`
- `docs/planned/research_plan_multifamily_generalization.md`
- `docs/planned/STRATEGIC_ROADMAP_2026.md`
- `docs/planned/REPOSITORY_STATE_SNAPSHOT.md`
- `docs/planned/START_HERE_PHASE_1.md`
- `docs/planned/PHASE_1_READY_TO_EXECUTE.md`
- `docs/planned/PHASE_1_IMPLEMENTATION_GUIDE.md`
- `docs/planned/PHASE_1_DOCUMENT_LIBRARY.md`
- `docs/planned/PHASE_1_COMPLETE_BLUEPRINT.md`
- `docs/planned/PHASE_1_CHECKLIST.md`
- `docs/planned/FEATURE_SCHEMA_CHANGE.md`
- `docs/planned/DOCUMENTATION_ARCHITECTURE.md`
- `docs/planned/phase3_*`

Archive does not mean worthless. It means not part of the production docs
surface.

### Delete

Delete if no external deliverable requires them:

- `docs/model-demo.html` if the Next.js `Model Demo` page supersedes it.
- `frontend/coverage/` generated HTML/XML/LCOV outputs.
- Root generated JSON outputs: `results.json`, `test_results_multi_dataset.json`.
- Root temporary helper: `tmp_compare_logs.py`.
- Empty `notebooks/` if truly empty.
- Generated caches and build outputs: `.pytest_cache/`, `.pytest_tmp/`,
  `.tmp_pytest*`, `.venv/`, `venv/`, `frontend/.next/`, `frontend/.swc/`,
  `frontend/tsconfig.tsbuildinfo`.

### Minimum Documentation Required for Production

Keep the production docs set small:

```text
README.md
CHANGELOG.md
CONTRIBUTING.md
SECURITY.md
docs/
  architecture.md
  api.md
  local-development.md
  operations/
    deployment.md
    runbook.md
    incident-response.md
  model/
    model-card.md
    data-card.md
    model-registry.md
  research/
    reproducibility.md
    validated-results.md
  adr/
    0001-repository-boundaries.md
    0002-artifact-storage.md
```

## 4. Architecture Review

### Strengths

- Clear useful domain: C2 behavioral detection and robustness evaluation.
- Backend has conventional route/schema/service layers.
- Frontend has a simple shell, reusable store, and a typed API client.
- ML code has explicit model classes and data/evaluation/training separation.
- Tests exist for important research contracts: feature schemas, transforms,
  operating point selection, Red-Agent reward, TLS/DNS/temporal features.

### Architectural Smells

| Smell | Severity | Example | Recommendation |
| --- | --- | --- | --- |
| Research, demo, and production code are mixed | High | `src/pipelines`, `backend`, `frontend`, root scripts all live as one product | Split into packages and define production import boundary. |
| Broken central abstraction | Critical | `host_window_dataset.py` is empty but widely imported | Restore immediately or remove all host-aware workflows until restored. |
| Runtime artifact storage in app repo | Critical | `backend/artifacts` contains uploaded NPZs | Move to object storage or external local storage path. |
| Duplicate inference API | High | `src/api/predict.py` and `backend/app/api/predict.py` | Keep one backend route; delete/archive standalone API. |
| Placeholder APIs | High | Robustness report/metrics return fixed values; compare endpoint placeholder | Replace with persistence-backed real results or remove endpoints. |
| Model/data schema drift | High | Base schema says 12 features; backend single-session service extracts 10 features; extended schema has 45 | Create one schema registry with explicit model compatibility. |
| Hidden source due ignore rules | Critical | `models/` ignore hides `src/models` from `rg --files` | Fix `.gitignore` immediately. |
| Local filesystem storage abstraction too weak | High | `.metadata.json`, no locks, no quota, no retention | Use object storage + DB metadata or at least SQLite + configured data dir. |
| Unsafe artifact loading | High | Pickle-capable loaders on configurable paths | Restrict loaders to trusted model registry; never load arbitrary user artifacts with pickle. |
| Frontend API and backend contracts hand-maintained | Medium | Manual `any` request shapes in frontend | Generate types from OpenAPI or share schema definitions. |

### Missing Abstractions

- Model registry abstraction: named model, version, schema, threshold, device,
  artifact URI, checksum, model card.
- Artifact storage abstraction: upload stream, retention, quota, metadata DB,
  object store path, content scanning, lifecycle policy.
- Job/execution abstraction: mutation and dataset evaluation are long-running
  and should not block request threads.
- Auth and authorization: API keys or user identity are absent.
- Observability: request ids, structured logs, metrics, traces, health/readiness
  checks.
- API error model: consistent domain errors vs generic `500` string detail.
- Dataset schema validator: validate NPZ keys, dtypes, shapes, feature names,
  masks, and max sizes before storage.

### Premature Complexity

- Too many planning docs and phase documents for the current maturity.
- Multiple dataset builders and schema variants without a canonical support
  matrix.
- Separate standalone prediction API after integrated backend route exists.
- Red-Agent training surface exposed next to app demo without a clear product
  boundary.

## 5. Code Quality Review

### Critical Findings

| Finding | Evidence | Impact | Fix |
| --- | --- | --- | --- |
| Empty host-window module | `src/data_loader/host_window_dataset.py` length 0; tests fail import | Host-aware workflows broken | Restore from history or reimplement to satisfy tests. |
| Frontend type-check fails | `ignoreDeprecations: "6.0"` invalid | Build not clean | Remove/fix option to valid TypeScript value. |
| Model source may be ignored | `.gitignore` has `models/`; `rg --files` hides `src/models` | Source may be untracked/missed by tools | Replace ignore pattern with `/models/` or `/artifacts/models/`. |
| Runtime artifacts in repo | `backend/artifacts` contains 1.34 GB NPZ and duplicates | Repo/workspace bloat, data leakage risk | Delete from repo workspace, add ignore, externalize storage. |
| Robustness service likely mishandles 3D session tensors | `_batch_predict` treats `row` as scalar feature vector | Runtime failure or incorrect metrics | Use tensor inference path with masks; add tests. |
| Detection mask padding bug risk | `_predict_from_sequences` pads a 2D mask with 3D pad widths on feature mismatch | Runtime error on feature-dim mismatch | Remove mask padding; only feature tensor needs feature padding/truncation. |

### High Findings

| Finding | Impact | Fix |
| --- | --- | --- |
| Placeholder robustness endpoints | Users get misleading static metrics | Remove or implement persistence-backed reports. |
| Artifact metadata feature count bug | For `sessions` tensors, feature count uses `shape[1]` instead of `shape[-1]` | Wrong UI/API metadata | Fix and test NPZ metadata extraction. |
| Mutation output overwrites fixed filenames before upload | Race/collision risk under concurrent requests | Write to unique temp/output path per request. |
| Artifact service reads whole files into memory | Large uploads/downloads can exhaust memory | Stream uploads/downloads and enforce size limits. |
| No auth | Anyone can upload/score/delete artifacts in deployed service | Add API auth before any shared environment. |
| Unsafe CORS in standalone API | `src/api/predict.py` allows all origins | Remove standalone API or lock CORS. |
| `allow_pickle=True` and `torch.load(weights_only=False)` used broadly | Unsafe for untrusted artifacts | Restrict to trusted registry; use safer formats/checksums. |
| Frontend tests absent | No regression safety for UI workflows | Add real tests or remove test claims. |
| Root scripts and outputs mixed with source | Maintenance noise | Move scripts to `tools/` and outputs to external artifacts. |

### Medium Findings

- Backend uses direct singleton-style service instances in route modules.
- Backend configuration is a hand-written class instead of typed settings.
- API errors mostly collapse to generic `500` with raw exception messages.
- CORS allows all methods/headers and credentials for several origins.
- Frontend has unused dependencies or dependencies in wrong sections:
  `class-variance-authority` and `framer-motion` appear unused; `typescript`,
  `tailwindcss`, `postcss`, `autoprefixer`, and `@types/*` belong in dev deps.
- Frontend API client uses `any` for most request/response shapes.
- `frontend/__tests__/README.md` documents absent tests.
- Documentation contains encoding artifacts and overclaims.

### Low Findings

- Several unused imports, for example `AxiosError` in `frontend/lib/api.ts`.
- Some comments are stale, such as "NEW FIX" in model code.
- Root utility names are inconsistent: `check_*`, `monitor_build.py`,
  `tmp_compare_logs.py`.
- Some docs use old line counts/status claims instead of current verified state.

## 6. Dependency Audit

This audit did not run live vulnerability scanners. Run `pip-audit`,
`safety`, `npm audit`, or a SCA tool in CI before production decisions.

### Backend Dependencies

Current `backend/requirements.txt`:

```text
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
numpy>=1.26.0
scipy>=1.14.0
torch>=2.2.0
scikit-learn>=1.3.2
joblib==1.3.2
aiofiles==23.2.1
python-dotenv==1.0.0
```

Findings:

- Mixed strict pins and broad ranges reduce reproducibility.
- Research scripts import packages not listed here, notably `pandas` and
  plotting dependencies.
- Torch is a broad `>=` dependency, which can unexpectedly pull incompatible
  versions.
- No development dependency set for pytest, ruff, mypy, black, pip-audit, etc.
- No lock file for Python.

Recommendations:

- Move to `pyproject.toml` with dependency groups:
  - `api`
  - `ml`
  - `research`
  - `dev`
- Pin production dependencies through a lock process.
- Separate API image dependencies from training/research dependencies.
- Add `pip-audit` or equivalent to CI.

### Frontend Dependencies

Findings:

- `class-variance-authority` and `framer-motion` appear unused in source.
- `typescript`, `tailwindcss`, `postcss`, `autoprefixer`, and `@types/*` are
  listed under production dependencies but should generally be dev deps for a
  Next app.
- Package ranges use `^`, while production builds should rely on lock file.
- `npm run type-check` currently fails.
- `npm run lint` passes.

Recommendations:

- Fix type-check first.
- Move build-only dependencies to devDependencies.
- Remove unused dependencies after verifying with dependency tooling.
- Add `npm audit` to CI.

## 7. Production Readiness Audit

### Infrastructure

| Area | Current State | Readiness |
| --- | --- | --- |
| Deployment | Local start docs only; no Dockerfile, Compose, Helm, or deployment manifests visible | Not ready |
| CI/CD | No visible workflow files | Not ready |
| Environment management | Basic env vars through `python-dotenv` | Partial |
| Secrets handling | No secret manager; no auth secrets | Not ready |
| Artifact storage | Local filesystem under backend | Not ready |
| Model registry | Informal paths under `experiments/` | Not ready |

### Reliability

| Area | Current State | Readiness |
| --- | --- | --- |
| Logging | Basic Python logging | Partial |
| Monitoring | None | Not ready |
| Error handling | Global generic exception handler, route try/except | Partial |
| Recovery | No job retry/resume model | Not ready |
| Long-running jobs | Synchronous request path | Not ready |
| Storage durability | Local JSON metadata | Not ready |

### Testing

| Area | Current State | Readiness |
| --- | --- | --- |
| Python unit tests | Present but currently fail collection for host-window path | Not ready |
| Backend API tests | Sparse/absent | Not ready |
| Frontend unit tests | Documentation only; no actual test file visible | Not ready |
| E2E tests | None visible | Not ready |
| Type checking | Frontend fails | Not ready |
| Coverage | Generated frontend coverage exists, but no current green evidence | Not ready |

### Performance

| Area | Risk |
| --- | --- |
| Large uploads | Files are read fully into memory. |
| Dataset evaluation | Can return full session tensors in JSON, very expensive. |
| Mutation/evaluation | CPU-heavy work runs synchronously in API request path. |
| Artifact listing | JSON metadata file becomes bottleneck and corruption risk. |
| Model loading | Some paths load models lazily but without readiness gating or lifecycle management. |

### Readiness Score

Production Readiness Score: **31/100**.

Interpretation:

- Useful MVP and research platform.
- Not a production candidate.
- Needs cleanup, build/test repair, storage/auth, deployment, and observability.

## 8. Repository Cleanup Plan

### Delete Immediately

Do not delete blindly if any file is needed for an external deliverable, but
these should not remain in the production repository:

- `backend/artifacts/`
- `frontend/coverage/`
- `frontend/.next/`
- `frontend/.swc/`
- `frontend/tsconfig.tsbuildinfo`
- `.pytest_cache/`
- `.pytest_tmp/`
- `.tmp_pytest*`
- `.venv/`
- `venv/`
- `results.json`
- `test_results_multi_dataset.json`
- `tmp_compare_logs.py`
- Empty `notebooks/`

### Archive

Move to `docs/archive/` or external research archive:

- Historical planning docs under `docs/planned/`.
- Completion/status docs that overclaim production readiness.
- Root planning docs after merging essential content.
- Old experiment outputs under `experiments/` that are not the blessed model.

### Merge

- Merge quickstart docs into README and local development runbook.
- Merge architecture docs into one accurate `docs/architecture.md`.
- Merge deployment/model cleanup into operations and model registry docs.
- Merge data strategy docs into one research data strategy doc.

### Restructure Directories

Proposed production-friendly structure:

```text
CyberShield/
  README.md
  pyproject.toml
  package.json or frontend/package.json
  .gitignore
  .github/
    workflows/
      ci.yml
  backend/
    cybershield_api/
      __init__.py
      main.py
      api/
      core/
      schemas/
      services/
      storage/
      models/
    tests/
  ml/
    cybershield_ml/
      data/
      features/
      models/
      inference/
      evaluation/
      red_agent/
    tests/
  pipelines/
    data/
    training/
    evaluation/
  frontend/
    app/
    components/
    lib/
    styles/
    tests/
  tools/
    data_checks/
    plotting/
  docs/
    architecture.md
    api.md
    local-development.md
    operations/
    model/
    research/
    adr/
    archive/
```

If keeping one Python package is preferred, use:

```text
src/cybershield/
  api/
  ml/
  data/
  red_agent/
  pipelines/
```

But do not keep production API code importing ad hoc research scripts.

### Naming Inconsistencies

- `src/api/predict.py` vs `backend/app/api/predict.py`: keep one.
- `feature_config.py`, `feature_config_v2.py`,
  `feature_config_experiment.py`, `feature_config_extended.py`: create schema
  registry and deprecate legacy configs explicitly.
- Root `check_*.py`: move to `tools/data_checks/`.
- Multiple "phase" docs: archive, not production docs.

## 9. Development Roadmap

### Phase 1 - Critical Fixes

| Task | Impact | Effort | Priority | Dependencies |
| --- | --- | --- | --- | --- |
| Restore/reimplement `host_window_dataset.py` or remove host-aware scope | Unblocks tests and host-aware workflows | M | P0 | Decide whether host-aware is in MVP scope |
| Fix frontend `tsconfig.json` type-check failure | Restores build confidence | S | P0 | None |
| Fix `.gitignore` to stop hiding `src/models` and ignore `backend/artifacts` | Prevents source loss and artifact bloat | S | P0 | None |
| Remove generated/runtime artifacts from repo workspace | Reduces bloat and leakage risk | M | P0 | Confirm no required deliverable uses them |
| Add minimal CI for lint/type-check/Python tests | Stops regression | M | P0 | Tests must collect |
| Replace placeholder robustness endpoints or hide them | Stops misleading API behavior | M | P0 | Decide product contract |

### Phase 2 - Stabilization

| Task | Impact | Effort | Priority | Dependencies |
| --- | --- | --- | --- | --- |
| Add backend API integration tests | Validates real app behavior | M | P1 | Stable API contracts |
| Add real frontend tests for Mutation Lab, Upload, Model Demo | Prevents demo regressions | M | P1 | Type-check fixed |
| Split dependencies into prod/dev/research | Reproducible installs | M | P1 | Decide packaging |
| Implement NPZ schema validation | Prevents bad uploads | M | P1 | Canonical schema registry |
| Fix artifact service metadata and streaming | Handles large files safely | M | P1 | Storage decision |
| Consolidate prediction APIs | Reduces duplicate maintenance | S | P1 | Backend prediction path chosen |

### Phase 3 - Production Readiness

| Task | Impact | Effort | Priority | Dependencies |
| --- | --- | --- | --- | --- |
| Add auth and authorization | Required for deployment | M | P1 | Product security model |
| Replace local artifact storage with object storage + DB metadata | Durable production storage | L | P1 | Infra target |
| Add job queue for long-running mutation/evaluation | Reliability and UX | L | P1 | Storage and API contracts |
| Add model registry/model card/checksum validation | Safe model operations | M | P1 | Model selection |
| Containerize backend/frontend | Repeatable deploy | M | P1 | Dependency split |
| Add structured logging, metrics, tracing, health/readiness | Operability | M | P1 | Deployment target |
| Write runbooks and incident response docs | Production operations | M | P2 | Infra design |

### Phase 4 - Scale and Optimization

| Task | Impact | Effort | Priority | Dependencies |
| --- | --- | --- | --- | --- |
| Batch inference service optimization | Higher throughput | M/L | P2 | Baseline performance tests |
| GPU/CPU deployment profiles | Cost/performance tuning | M | P2 | Model registry |
| Dataset/result pagination and streaming | Large dataset UX | M | P2 | Storage refactor |
| Drift monitoring and model retraining workflow | Sustained quality | L | P2 | Production telemetry |
| E2E browser tests in CI | Confidence for releases | M | P2 | Stable UI |

## 10. First Actions

### First 10 Actions I Would Perform

1. Freeze scope: decide whether production MVP includes host-aware workflows or
   only domain-adaptive inference plus mutation evaluation.
2. Fix `.gitignore`: add `backend/artifacts/`, frontend generated dirs, pytest
   temp dirs; correct broad `models/` ignore.
3. Remove generated/runtime artifacts from the working repository folder.
4. Restore or reimplement `src/data_loader/host_window_dataset.py`.
5. Fix `frontend/tsconfig.json` and make `npm run type-check` pass.
6. Make `.venv` pytest collect and run at least the core test suite.
7. Remove/archive `src/api/predict.py` or prove it has a distinct purpose.
8. Replace placeholder robustness endpoints with real persistence or remove
   them from the production API.
9. Create CI with lint, type-check, unit tests, and artifact-size guard.
10. Write a model registry file identifying the one approved model/checkpoint,
    feature schema, threshold, and checksum.

### First Week Plan

Day 1:

- Fix ignore rules.
- Delete/move generated artifacts from repo workspace.
- Decide production MVP scope.
- Open tracking tickets for each P0.

Day 2:

- Restore `host_window_dataset.py` or remove host-aware from current CI.
- Fix frontend type-check.
- Make frontend lint/type-check green.

Day 3:

- Make Python tests collect.
- Run core unit tests.
- Fix any immediate import/schema regressions.

Day 4:

- Audit backend endpoint behavior.
- Remove placeholder endpoints from user-visible docs/API or implement real
  responses.
- Add backend smoke tests for health, upload validation, predict health.

Day 5:

- Create CI workflow.
- Produce cleaned docs set draft.
- Publish production MVP decision and target architecture ADR.

### First Month Plan

Week 1:

- Critical cleanup and green CI baseline.

Week 2:

- Storage/auth design.
- Artifact service refactor.
- Model registry and schema validation.
- Backend integration tests.

Week 3:

- Job queue for long-running evaluations.
- Frontend tests and API contract hardening.
- Containerization and deployment scripts.

Week 4:

- Observability, runbooks, security pass, load tests, release candidate
  checklist.

## Final Scores

Production Readiness Score: **31/100**  
Cleanup Score: **82/100** where 100 means very cluttered and cleanup-heavy  
Technical Debt Score: **76/100** where 100 means severe debt  

Single most important next step:

**Restore the source/test baseline: fix `.gitignore`, remove generated
artifacts, restore `host_window_dataset.py`, and get Python tests plus frontend
type-check green in CI.**

## Complete Immediate Action Checklist

### P0 - Do Now

- [ ] Update `.gitignore`:
  - [ ] Add `backend/artifacts/`
  - [ ] Add `.tmp_pytest*/`
  - [ ] Add `.pytest_tmp/`
  - [ ] Add `frontend/coverage/`
  - [ ] Add `frontend/tsconfig.tsbuildinfo`
  - [ ] Replace `models/` with root-scoped or artifact-specific ignore rule.
- [ ] Move/delete `backend/artifacts/` from repository workspace.
- [ ] Move/delete `results.json`.
- [ ] Move/delete frontend generated dirs: `.next`, `.swc`, `coverage`.
- [ ] Restore or reimplement `src/data_loader/host_window_dataset.py`.
- [ ] Fix `frontend/tsconfig.json` `ignoreDeprecations`.
- [ ] Run `npm run type-check`.
- [ ] Run `npm run lint`.
- [ ] Run `.venv\Scripts\python.exe -m pytest`.
- [ ] Decide whether `src/api/predict.py` is removed or archived.
- [ ] Remove placeholder robustness endpoints from production docs.

### P1 - Next

- [ ] Create CI workflow.
- [ ] Add backend API tests.
- [ ] Add real frontend tests.
- [ ] Add artifact upload size limit and streaming.
- [ ] Add NPZ schema validator.
- [ ] Add model registry and checksum validation.
- [ ] Split dependencies into runtime/dev/research groups.
- [ ] Add security scanning commands to CI.
- [ ] Add auth plan and implement API key or equivalent for MVP.
- [ ] Consolidate docs into minimum production set.

### P2 - Production Candidate

- [ ] Replace local artifact storage with object storage or managed volume plus
  DB metadata.
- [ ] Add queue/worker for long-running dataset evaluations and mutations.
- [ ] Add structured logging, metrics, tracing, request ids.
- [ ] Add Dockerfiles and deployment manifests.
- [ ] Add health and readiness checks that validate model availability.
- [ ] Add runbooks and incident response docs.
- [ ] Add load tests for upload, inference, mutation, and dataset evaluation.
- [ ] Add E2E browser tests.
- [ ] Create formal model card and data card.

