# CyberShield v2 Project Status

Date: 2026-05-08

## Summary

CyberShield has transitioned from v1 (flow-based Transformer classifier with external generalization issues) to **v2 (host-centric behavioral C2 detection system)**. The Phase 0 infrastructure has been scaffolded with all core new modules implemented.

## Core Architecture Shift

**v1 Problem:** The Transformer classifier achieved strong in-domain performance but failed on external/unseen traffic due to dataset-specific shortcuts and lack of behavioral grounding.

**v2 Solution:** Shift from isolated session classification to **host-centric behavioral analysis**. The system now asks "Is this behavior abnormal *for this specific host*?" rather than just "Does this session look like C2?"

## What Has Been Built (Phase 0 Scaffold)

### 1. Extended Feature Configuration
- **File:** `src/features/feature_config_v2.py`
- Backward-compatible with v1 `feature_config.py`
- Added v2 feature groups: behavioral (7), entropy (4), persistence (5), TLS (6)
- Added modern C2 family labels (Sliver, Havoc, Cobalt Strike, Mythic, etc.)
- Source metadata tracking (`SessionMetadata`, `DatasetSource`)
- Relaxed FPR budget: 0.015–0.03 (vs v1's 0.005)

### 2. Host Timeline Infrastructure
- **File:** `src/features/host_timeline.py`
- `HostTimeline`: Per-host behavioral history with rolling statistics
- `SessionSummary`: Lightweight session representation for timeline tracking
- `PartnerHistory`: Communication relationship tracking per destination
- `RollingStatistics`: Exponentially-decayed online statistics for anomaly detection
- `HostTimelineBuilder`: Builds timelines from flow DataFrames
- Computes Z-score deviations for new sessions against host baselines
- Activity rhythm tracking (hourly histograms)
- Destination evolution monitoring

### 3. UWF-ZeekData24 Processor
- **File:** `src/data_loader/uwf_processor.py`
- Auto-discovers conn.log files in UWF directory structure
- Classifies files as C2/benign based on MITRE ATT&CK naming
- Infers C2 family from directory paths
- Outputs session NPZ compatible with existing training pipeline
- Discovers supplementary logs (dns.log, ssl.log) for future TLS features

### 4. Source-Aware Evaluation Splits
- **File:** `src/data_loader/split_utils.py` (v2 rewrite)
- **Five strategies:** random stratified, family-separated, source-separated, time-separated, zero-shot
- Full leakage validation (index overlap checks between splits)
- Detailed split reporting with family/source provenance
- Backward-compatible v1 functions preserved (`create_session_splits`, `load_filtered_sessions`)

### 5. Enhanced Dataset Builder
- **File:** `src/features/dataset_builder_v2.py`
- Multi-source NPZ ingestion with source metadata
- Source specification format: `<npz_path>:<source_name>:<label_type>`
- Integrates with v2 split strategies
- Optional C2/benign ratio rebalancing
- Exports split metadata JSON for reproducibility

### 6. C2 Traffic Generation Infrastructure
- **File:** `src/red_agent/c2_generator.py`
- Pre-built capture profiles: 5 Sliver + 3 Havoc configurations
- Profiles cover easy (fast beacon) to hard (low-and-slow, event-driven)
- Documented capture procedures for lab environments
- JSON configuration export for reproducible captures

### 7. Phase 0 Test Suite
- **File:** `tests/test_phase0.py`
- Tests for all v2 modules: feature config, host timeline, split strategies
- Integration tests: flow → timeline → feature vector pipeline
- 25+ test cases covering backward compatibility, anomaly detection, leakage prevention

## v1 Code Preserved
All v1 modules remain functional and untouched:
- `src/features/feature_config.py` (original, still used by v1 imports)
- `src/features/zeek_parser.py`
- `src/features/session_builder.py`
- `src/models/transformer.py`
- `src/training/train_transformer.py`
- All evaluation and analysis modules

## Phase Status

| Phase | Status | Key Deliverables |
|---|---|---|
| Phase 0 — Foundation | 🟡 In Progress | Infrastructure scaffolded, needs data ingestion |
| Phase 1 — Behavioral Intelligence | 📋 Planned | Behavioral/entropy/persistence/TLS features |
| Phase 2 — Host Baselines | 📋 Planned | Host profiler + anomaly detector |
| Phase 3 — Multi-Signal Fusion | 📋 Planned | Detector fusion + risk aggregator |
| Phase 4 — Hardening | 📋 Planned | Ongoing iteration |

## Immediate Next Steps

1. **Acquire UWF-ZeekData24 dataset** and run `uwf_processor.py`
2. **Generate Sliver/Havoc captures** using the c2_generator profiles
3. **Build host timelines** from existing CTU-13 data to validate infrastructure
4. **Run Phase 0 test suite** to validate all modules
5. **Create first family-separated split** and compare against v1 baseline

## Key Files Map

### v2 New Modules
- [src/features/feature_config_v2.py](../src/features/feature_config_v2.py)
- [src/features/host_timeline.py](../src/features/host_timeline.py)
- [src/features/dataset_builder_v2.py](../src/features/dataset_builder_v2.py)
- [src/data_loader/uwf_processor.py](../src/data_loader/uwf_processor.py)
- [src/data_loader/split_utils.py](../src/data_loader/split_utils.py) (v2 rewrite + v1 compat)
- [src/red_agent/c2_generator.py](../src/red_agent/c2_generator.py)
- [tests/test_phase0.py](../tests/test_phase0.py)

### v1 Preserved Modules
- [src/features/feature_config.py](../src/features/feature_config.py)
- [src/features/zeek_parser.py](../src/features/zeek_parser.py)
- [src/features/session_builder.py](../src/features/session_builder.py)
- [src/models/transformer.py](../src/models/transformer.py)
- [src/training/train_transformer.py](../src/training/train_transformer.py)
