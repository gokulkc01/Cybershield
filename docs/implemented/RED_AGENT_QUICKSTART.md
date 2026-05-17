# Red-Agent v1: Mutation-Based Adversarial Robustness Evaluation

**Status**: Production-ready single-command pipeline for CI/reproducible job runs.

## Overview

Red-Agent v1 is a **controlled behavioral mutation framework** for stress-testing the CyberShield C2 detector. It systematically applies seeded, explainable mutations to session sequences and evaluates robustness under adversarial mutation pressure.

**Core Philosophy**:
- **Not**: Autonomous PPO-based attacker, self-evolving ecosystem
- **Yes**: Deterministic mutation testing, behavioral invariance validation, failure analysis

## Quick Start

### 1. Prepare a Config JSON

Create a mutation plan config. Example:

```json
{
  "checkpoint_path": "experiments/multifamily_generalization/strict_smoke/best_transformer.pth",
  "baseline_npz": "data/processed/experiment_10f_splits_strict/test_sessions.npz",
  "mutation_plan": [
    {
      "mutation_type": "timing_jitter",
      "severity": 0.4,
      "params": {}
    },
    {
      "mutation_type": "long_sleep",
      "severity": 0.7,
      "params": {}
    }
  ],
  "base_seed": 42,
  "batch_size": 256,
  "device": "cpu"
}
```

### 2. Run Single Command

```bash
python -m src.pipelines.red_agent_robustness \
  --config experiments/red_agent_config_example.json \
  --output_dir experiments/red_agent_run_001
```

### 3. Inspect Results

```
experiments/red_agent_run_001/
├── mutated_sessions.npz          # Mutated session data
├── mutation_metadata.jsonl        # Per-sample lineage (one JSON per line)
├── robustness_report.json         # Full metrics, failure analysis, breakdown
└── summary.txt                    # Human-readable summary
```

## Available Mutations

### Timing Mutations
- `timing_jitter`: Multiplicative jitter to inter-arrival times
- `burst_callback`: Bursty callback patterns (compress some intervals, expand others)
- `delayed_reconnect`: Long reconnect gaps

### Persistence Mutations
- `low_frequency_callback`: Reduce callback frequency (increase iat multiplier)
- `intermittent_communication`: Drop activity from random windows + sparse spikes
- `long_sleep`: Long beacon sleep intervals with sparse reconnects

### TLS/HTTPS Mutations
- `tls_padding`: Increase bytes-per-packet to mimic padded TLS records
- `session_reuse_shape`: Flatten handshake-like variance
- `handshake_variation`: Perturb packet/byte ratios

### Flow-Level Mutations
- `packet_count_variation`: Vary packet counts with directional asymmetry
- `byte_distribution_shift`: Shift byte distribution and bytes-per-packet

List all available mutations:

```bash
python -c "from src.red_agent import list_mutations; print([m.name for m in list_mutations()])"
```

## Config Schema

```json
{
  "checkpoint_path": "path/to/best_transformer.pth",      // [REQUIRED] Trained model
  "baseline_npz": "path/to/sessions.npz",                // [REQUIRED] Baseline session data
  "mutation_plan": [                                      // [REQUIRED] List of mutations
    {
      "mutation_type": "timing_jitter",                   // Mutation name (must be registered)
      "severity": 0.4,                                    // 0.0–1.0, higher = stronger mutation
      "params": {}                                        // Additional parameters (if needed)
    }
  ],
  "base_seed": 42,                                        // [OPTIONAL] Deterministic seed (default: 42)
  "batch_size": 256,                                      // [OPTIONAL] Inference batch size (default: 256)
  "device": "cpu"                                         // [OPTIONAL] "cpu" or "cuda" (default: "cpu")
}
```

## Output Files

### `mutated_sessions.npz`
- Same format as baseline NPZ (X, y, masks, feature_names)
- Applied mutations in sequence (deterministic per sample + seed)

### `mutation_metadata.jsonl`
- One JSON object per line
- Per-sample lineage including:
  - `sample_id`, `original_sample_id`, `original_label`
  - `mutation_type`, `mutation_parameters`, `mutation_severity`, `seed`
  - `mutation_history` (ordered sequence of mutations)
  - `evaluation_outcome` (populated after inference):
    - `threshold`, `mutated_score`, `is_detected`, `latency_per_sample_ms`

### `robustness_report.json`
- **baseline_metrics**: recall, precision, fpr, f1, confusion matrix on baseline
- **mutated_metrics**: same for mutated samples
- **robustness_summary**:
  - `recall_degradation`: baseline recall − mutated recall
  - `fpr_shift`: mutated fpr − baseline fpr
  - `precision_shift`, `f1_shift`
  - `behavioral_invariance_stability`: mutated recall / baseline recall
- **mutation_breakdown**: Per-mutation-type detection rates and failure counts
- **failure_analysis**:
  - Top feature shifts (identifies which features the detector over-relies on)
  - Fragile mutations (which mutations bypass detection most often)
  - Interpretable diagnostics for research

### `summary.txt`
- Human-readable one-page summary

## Robustness Metrics Explained

| Metric | Interpretation |
|--------|-----------------|
| `recall_degradation` | How much detection recall drops under mutation (higher = worse robustness) |
| `fpr_shift` | False positive rate increase (positive = more false alarms, negative = fewer) |
| `behavioral_invariance_stability` | Mutated recall / baseline recall (closer to 1.0 = more robust) |
| `fragile_mutation_types` | Mutations that evade detection most often (identifies weak points) |
| `top_shifted_features` | Features that shift most under mutation (detects over-reliance) |

## CI/Reproducible Job Runs

### Single Job Example

```bash
#!/bin/bash
set -e

CONFIG="experiments/red_agent_config_example.json"
OUTPUT_DIR="experiments/red_agent_$(date +%Y%m%d_%H%M%S)"

python -m src.pipelines.red_agent_robustness \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --verbose

echo "✓ Job complete: $OUTPUT_DIR"
```

### Matrix Job (Multiple Mutation Configs)

```bash
#!/bin/bash

for SEVERITY in 0.3 0.5 0.7; do
  CONFIG="experiments/red_agent_severity_${SEVERITY}.json"
  OUTPUT_DIR="experiments/red_agent_severity_${SEVERITY}"
  
  python -m src.pipelines.red_agent_robustness \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"
  
  echo "✓ Completed severity=$SEVERITY"
done
```

### GitHub Actions / CI Integration

```yaml
name: Red-Agent Robustness Tests

on: [push]

jobs:
  robustness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run robustness pipeline
        run: |
          python -m src.pipelines.red_agent_robustness \
            --config experiments/red_agent_config_example.json \
            --output_dir artifacts/red_agent_results
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: red-agent-results
          path: artifacts/red_agent_results/
```

## Advanced Usage

### Custom Mutation Plans

```python
# mutation_plan.json
[
  {
    "mutation_type": "timing_jitter",
    "severity": 0.3,
    "params": {
      "max_scale": 2.0,
      "min_scale": 0.5
    }
  },
  {
    "mutation_type": "intermittent_communication",
    "severity": 0.6,
    "params": {
      "quiet_fraction": 0.4,
      "spike_boost": 2.5
    }
  }
]
```

### Programmatic Usage (Python)

```python
from src.pipelines.red_agent_robustness import run_robustness_pipeline
import json

config = {
    "checkpoint_path": "experiments/...",
    "baseline_npz": "data/...",
    "mutation_plan": [...],
    "base_seed": 42
}

result = run_robustness_pipeline(config, "experiments/red_agent_run_001")
print(result["robustness_summary"])
```

## Future Work (Phase 6+)

Once mutation space is fully understood:
- PPO-based adversarial optimization
- Self-play adversarial systems
- Ensemble defenses

## References

- **Mutation Registry**: `src/red_agent/mutation_registry.py`
- **Available Operators**: `src/red_agent/timing|persistence|tls|flow_mutations.py`
- **Evaluation Pipeline**: `src/red_agent/evaluation_runner.py`
- **Failure Analysis**: `src/red_agent/behavioral_failure_analyzer.py`
- **Config Schema**: See config example above

---

**Status**: Production-ready for behavioral robustness research.  
**Maintained**: Core detector architecture unchanged; evaluation augmentation only.
