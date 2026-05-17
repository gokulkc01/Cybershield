import json
import statistics
import math


def stats(vals):
    n = len(vals)
    if n == 0:
        return {"count": 0, "mean": None, "median": None, "stdev": None, "min": None, "max": None}
    mean = statistics.mean(vals)
    median = statistics.median(vals)
    stdev = statistics.stdev(vals) if n > 1 else 0.0
    return {"count": n, "mean": mean, "median": median, "stdev": stdev, "min": min(vals), "max": max(vals)}


def analyze(path):
    rewards = []
    realism = []
    violations = []
    conf_delta = []
    for line in open(path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        j = json.loads(line)
        # reward
        r = j.get("reward")
        if r is not None and not (isinstance(r, float) and math.isnan(r)):
            rewards.append(float(r))
        # realism
        rs = j.get("realism_score")
        if rs is not None and not (isinstance(rs, float) and math.isnan(rs)):
            realism.append(float(rs))
        # violations
        viol = j.get("constraint_violations")
        if viol is None:
            violations.append(0)
        else:
            try:
                violations.append(len(viol))
            except Exception:
                violations.append(int(viol) if isinstance(viol, (int, float)) else 0)
        # confidence delta (original - mutated)
        c1 = j.get("detector_confidence_original")
        c2 = j.get("detector_confidence_mutated")
        if c1 is not None and c2 is not None:
            try:
                conf_delta.append(float(c1) - float(c2))
            except Exception:
                pass

    return {
        "file": path,
        "reward": stats(rewards),
        "realism": stats(realism),
        "violations": stats(violations),
        "conf_delta": stats(conf_delta),
    }


if __name__ == '__main__':
    files = [
        r"experiments/red_agent_phase1_smoke/phase1_training_log.jsonl",
        r"experiments/red_agent_phase1_smoke_batch4/phase1_training_log.jsonl",
    ]

    for f in files:
        try:
            res = analyze(f)
        except FileNotFoundError:
            print(f"MISSING: {f}")
            continue
        print(f"FILE: {f}")
        for key in ["reward", "realism", "conf_delta", "violations"]:
            s = res[key]
            print(f"  {key}: count={s['count']}, mean={s['mean']}, median={s['median']}, stdev={s['stdev']}, min={s['min']}, max={s['max']}")
        print("-")
