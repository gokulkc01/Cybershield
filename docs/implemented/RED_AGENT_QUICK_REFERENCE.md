# Red-Agent v1 Quick Reference

## One-Liners

### List All Mutations
```bash
python -c "from src.red_agent import list_mutations; [print(f'{m.name:30s} {m.category}') for m in list_mutations()]"
```

### Run Example Config
```bash
python -m src.pipelines.red_agent_robustness \
  --config experiments/red_agent_config_example.json \
  --output_dir experiments/my_run
```

### Run with Verbose Output
```bash
python -m src.pipelines.red_agent_robustness \
  --config config.json \
  --output_dir output/ \
  --verbose
```

## Common Configs

### Timing Robustness
```json
{
  "checkpoint_path": "experiments/multifamily_generalization/strict_smoke/best_transformer.pth",
  "baseline_npz": "data/processed/experiment_10f_splits_strict/test_sessions.npz",
  "mutation_plan": [
    {"mutation_type": "timing_jitter", "severity": 0.3, "params": {}},
    {"mutation_type": "timing_jitter", "severity": 0.6, "params": {}},
    {"mutation_type": "burst_callback", "severity": 0.5, "params": {}}
  ],
  "base_seed": 42,
  "batch_size": 256,
  "device": "cpu"
}
```

### Persistence Robustness (Low-and-Slow)
```json
{
  "checkpoint_path": "experiments/multifamily_generalization/strict_smoke/best_transformer.pth",
  "baseline_npz": "data/processed/experiment_10f_splits_strict/test_sessions.npz",
  "mutation_plan": [
    {"mutation_type": "low_frequency_callback", "severity": 0.7, "params": {}},
    {"mutation_type": "intermittent_communication", "severity": 0.6, "params": {}},
    {"mutation_type": "long_sleep", "severity": 0.8, "params": {}}
  ],
  "base_seed": 42,
  "batch_size": 256,
  "device": "cpu"
}
```

### TLS/Encryption Robustness
```json
{
  "checkpoint_path": "experiments/multifamily_generalization/strict_smoke/best_transformer.pth",
  "baseline_npz": "data/processed/experiment_10f_splits_strict/test_sessions.npz",
  "mutation_plan": [
    {"mutation_type": "tls_padding", "severity": 0.4, "params": {}},
    {"mutation_type": "session_reuse_shape", "severity": 0.5, "params": {}},
    {"mutation_type": "handshake_variation", "severity": 0.6, "params": {}}
  ],
  "base_seed": 42,
  "batch_size": 256,
  "device": "cpu"
}
```

### Flow-Level Robustness
```json
{
  "checkpoint_path": "experiments/multifamily_generalization/strict_smoke/best_transformer.pth",
  "baseline_npz": "data/processed/experiment_10f_splits_strict/test_sessions.npz",
  "mutation_plan": [
    {"mutation_type": "packet_count_variation", "severity": 0.4, "params": {}},
    {"mutation_type": "byte_distribution_shift", "severity": 0.5, "params": {}},
    {"mutation_type": "packet_count_variation", "severity": 0.7, "params": {}}
  ],
  "base_seed": 42,
  "batch_size": 256,
  "device": "cpu"
}
```

### Maximum Stress Test (All Mutations)
```json
{
  "checkpoint_path": "experiments/multifamily_generalization/strict_smoke/best_transformer.pth",
  "baseline_npz": "data/processed/experiment_10f_splits_strict/test_sessions.npz",
  "mutation_plan": [
    {"mutation_type": "timing_jitter", "severity": 0.5, "params": {}},
    {"mutation_type": "burst_callback", "severity": 0.5, "params": {}},
    {"mutation_type": "delayed_reconnect", "severity": 0.5, "params": {}},
    {"mutation_type": "low_frequency_callback", "severity": 0.5, "params": {}},
    {"mutation_type": "intermittent_communication", "severity": 0.5, "params": {}},
    {"mutation_type": "long_sleep", "severity": 0.5, "params": {}},
    {"mutation_type": "tls_padding", "severity": 0.5, "params": {}},
    {"mutation_type": "session_reuse_shape", "severity": 0.5, "params": {}},
    {"mutation_type": "handshake_variation", "severity": 0.5, "params": {}},
    {"mutation_type": "packet_count_variation", "severity": 0.5, "params": {}},
    {"mutation_type": "byte_distribution_shift", "severity": 0.5, "params": {}}
  ],
  "base_seed": 42,
  "batch_size": 256,
  "device": "cpu"
}
```

## Output Inspection

### View Summary
```bash
cat experiments/my_run/summary.txt
```

### Check Failure Analysis (Pretty Print)
```bash
python -c "import json; print(json.dumps(json.load(open('experiments/my_run/robustness_report.json')), indent=2)['failure_analysis'])"
```

### Count Detection Failures
```bash
python -c "
import json
meta = [json.loads(line) for line in open('experiments/my_run/mutation_metadata.jsonl')]
failures = sum(1 for m in meta if not m['evaluation_outcome'].get('is_detected', False))
print(f'Detection failures: {failures}/{len(meta)}')"
```

### Top Fragile Mutations
```bash
python -c "
import json
report = json.load(open('experiments/my_run/robustness_report.json'))
for mut, count in sorted(report['failure_analysis']['fragile_mutation_types'].items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f'{mut:30s}: {count:6d} undetected')"
```

## Programmatic API

### Direct Usage
```python
from src.red_agent import build_adversarial_dataset, run_mutation_evaluation
import json

config = {
    "checkpoint_path": "experiments/.../best_transformer.pth",
    "baseline_npz": "data/test.npz",
    "mutation_plan": [
        {"mutation_type": "timing_jitter", "severity": 0.4, "params": {}},
    ],
    "base_seed": 42,
    "batch_size": 256,
    "device": "cpu"
}

# Build dataset
build_result = build_adversarial_dataset(
    input_npz=config["baseline_npz"],
    output_npz="data/mutated.npz",
    output_metadata_jsonl="data/meta.jsonl",
    mutation_plan=config["mutation_plan"],
    base_seed=config["base_seed"]
)

# Evaluate
result = run_mutation_evaluation(
    checkpoint_path=config["checkpoint_path"],
    baseline_npz=config["baseline_npz"],
    mutated_npz="data/mutated.npz",
    metadata_jsonl="data/meta.jsonl",
    output_json="report.json",
    batch_size=config["batch_size"],
    device=config["device"]
)

print(f"Recall degradation: {result.robustness_summary['recall_degradation']:.4f}")
print(f"Invariance stability: {result.robustness_summary['behavioral_invariance_stability']:.4f}")
```

## Severity Guide

| Severity | Mutation Intensity | Use Case |
|----------|-------------------|----------|
| 0.1–0.2  | Light | Baseline robustness check |
| 0.3–0.5  | Moderate | Standard evaluation (recommended) |
| 0.6–0.8  | Heavy | Stress testing, edge cases |
| 0.9–1.0  | Extreme | Breaking-point identification |

## Device Selection

```bash
# CPU (default, slower but stable)
--device cpu

# GPU (if available, much faster)
--device cuda

# Auto-detect
# (omit --device flag; will use CUDA if available)
```

## Batch Size Tuning

```bash
# Small (slower, less memory)
--batch_size 64

# Standard (balanced)
--batch_size 256

# Large (faster, more memory)
--batch_size 512
```

## CI Integration Example

```yaml
name: Red-Agent Robustness

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run robustness pipeline
        run: |
          python -m src.pipelines.red_agent_robustness \
            --config experiments/red_agent_config_example.json \
            --output_dir artifacts/red_agent_results
      
      - name: Check results
        run: |
          DEGRADATION=$(python -c "import json; r=json.load(open('artifacts/red_agent_results/robustness_report.json')); print(r['robustness_summary']['recall_degradation'])")
          if (( $(echo "$DEGRADATION > 0.2" | bc -l) )); then
            echo "WARNING: Significant recall degradation detected: $DEGRADATION"
            exit 1
          fi
      
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: red-agent-results
          path: artifacts/red_agent_results/
```

## Troubleshooting

### Command Not Found
```bash
# Make sure .venv is activated
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### OOM (Out of Memory)
```bash
# Reduce batch size
--batch_size 128  # Or even 64
```

### Checkpoint Not Found
```bash
# Verify path exists
ls experiments/multifamily_generalization/strict_smoke/best_transformer.pth

# Use absolute path if relative fails
--checkpoint /full/path/to/checkpoint.pth
```

### Module Import Error
```bash
# Ensure you're in project root
cd /path/to/CyberShield

# Reinstall package
pip install -e .
```

---

**Status**: All mutations registered and available.  
**Questions**: See RED_AGENT_QUICKSTART.md for detailed documentation.
