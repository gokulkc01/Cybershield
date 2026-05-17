import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

LOG_FILES = [
    Path("experiments/red_agent_phase1_smoke/phase1_training_log.jsonl"),
    Path("experiments/red_agent_phase1_smoke_batch4/phase1_training_log.jsonl"),
    Path("experiments/red_agent_phase1_quick_batch4/phase1_training_log.jsonl"),
]
OUT_DIR = Path("experiments/phase1_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    ("reward", "Reward"),
    ("realism_score", "Realism Score"),
    ("detector_confidence_original", "Conf Orig"),
    ("detector_confidence_mutated", "Conf Mut"),
    ("detector_confidence_original_mutated_delta", "Conf Delta"),
    ("constraint_violations", "Constraint Violations"),
]


def read_log(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                records.append(j)
            except Exception:
                continue
    return records


def extract_field(records, name):
    vals = []
    for r in records:
        if name == "detector_confidence_original_mutated_delta":
            o = r.get("detector_confidence_original")
            m = r.get("detector_confidence_mutated")
            if o is not None and m is not None:
                try:
                    vals.append(float(o) - float(m))
                except Exception:
                    pass
        else:
            v = r.get(name)
            if v is None:
                # if constraint list -> use length
                if name == "constraint_violations":
                    v = r.get("constraint_violations")
                    if v is None:
                        vals.append(0)
                        continue
                    try:
                        vals.append(len(v))
                        continue
                    except Exception:
                        pass
                continue
            try:
                vals.append(float(v))
            except Exception:
                # fallback: if list, length
                try:
                    vals.append(len(v))
                except Exception:
                    continue
    return np.array(vals)


def plot_histograms(all_data, labels, field_key, title, out_path, bins=80, range=None):
    plt.figure(figsize=(8, 5))
    for data, label in zip(all_data, labels):
        if data.size == 0:
            continue
        plt.hist(data, bins=bins, alpha=0.5, label=label, density=True, range=range)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    inputs = []
    labels = []
    for p in LOG_FILES:
        recs = read_log(p)
        inputs.append(recs)
        labels.append(p.name if p.exists() else f"MISSING:{p.name}")

    for field, pretty in FIELDS:
        all_vals = [extract_field(recs, field) for recs in inputs]
        # compute a reasonable range for some fields
        rng = None
        if field == "reward":
            rng = (-3, 1)
        if field == "detector_confidence_original_mutated_delta":
            rng = (-0.5, 0.5)
        out_file = OUT_DIR / f"{field}.png"
        plot_histograms(all_vals, labels, field, pretty, out_file, bins=100, range=rng)
        print(f"Wrote {out_file}")

    # also write a small CSV summary of means
    summary = {}
    for p, recs in zip(LOG_FILES, inputs):
        k = p.name
        summary[k] = {}
        for field, pretty in FIELDS:
            vals = extract_field(recs, field)
            if vals.size == 0:
                summary[k][field] = None
            else:
                summary[k][field] = float(np.nanmean(vals))
    out_summary = OUT_DIR / "phase1_logs_summary.json"
    with out_summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {out_summary}")


if __name__ == '__main__':
    main()
