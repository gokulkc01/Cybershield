#!/usr/bin/env python
"""Monitor build progress and auto-continue pipeline when ready."""

import json
import subprocess
import time
from pathlib import Path

def get_completed_scenarios():
    """Check how many scenarios are in the generated manifest."""
    manifest = Path("experiments/multifamily_generalization/ctu13_sources_generated.json")
    if not manifest.exists():
        return 0
    
    with open(manifest) as f:
        data = json.load(f)
    
    # Count non-benign scenarios
    scenarios = [s for s in data.get("sources", []) if s.get("label_type") == "c2"]
    return len(scenarios)

def wait_for_build_completion(timeout_seconds=3600):
    """Wait for build to complete (expect 3 scenarios)."""
    print("[MONITOR] Waiting for build_ctu13_scenarios to complete...")
    start = time.time()
    last_count = 0
    
    while time.time() - start < timeout_seconds:
        completed = get_completed_scenarios()
        if completed != last_count:
            print(f"[MONITOR] {completed}/3 scenarios completed at {time.strftime('%H:%M:%S')}")
            last_count = completed
        
        if completed >= 3:
            print("[MONITOR] ✓ All scenarios completed! Manifest ready for split preparation.")
            return True
        
        time.sleep(10)
    
    print(f"[MONITOR] Timeout after {timeout_seconds}s, {completed}/3 scenarios completed")
    return False

if __name__ == "__main__":
    if wait_for_build_completion():
        print("\n[NEXT] Build complete. Ready to run prepare_multifamily_splits")
        print("Run: python -m src.pipelines.prepare_multifamily_splits \\")
        print("  --sources_manifest experiments/multifamily_generalization/ctu13_sources_generated.json \\")
        print("  --out_dir data/processed/experiment_10f_splits_strict \\")
        print("  --train_families neris,kraken \\")
        print("  --test_families conficker")
    else:
        print("[FAILED] Build did not complete in time")
