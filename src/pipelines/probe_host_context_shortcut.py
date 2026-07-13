"""Diagnose the host-context shortcut behind the host-aware model's val saturation.

Phase 0 watch item: the host-aware model saturates the same-family validation
axis (val AUC 0.995-1.0). This probe quantifies how much of that is available
without any deep model, from the 15 host summary features alone, and exposes
the structural reason: every host in every split is label-pure (a single
infected source host per capture), so identifying *which host* a session
belongs to is equivalent to predicting its label.

Outputs a JSON report with:
- logistic regression on host_features alone (train -> val/test AUC),
- per-feature single-feature AUCs (direction-agnostic),
- per-host session counts and label purity per split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data_loader.host_window_dataset import load_host_windows_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe host-feature label shortcut")
    parser.add_argument("--host_split_dir", default="data/processed/host_aware_ctu13_same_provenance")
    parser.add_argument(
        "--out",
        default="experiments/phase1_baselines_ctu13_same_provenance/host_context_probe.json",
    )
    args = parser.parse_args()

    split_dir = Path(args.host_split_dir)
    arrays = {name: load_host_windows_npz(split_dir / f"{name}_host_windows.npz") for name in ("train", "val", "test")}
    feature_names = [str(name) for name in arrays["train"].host_feature_names]

    scaler = StandardScaler().fit(arrays["train"].host_features)
    probe = LogisticRegression(max_iter=2000).fit(
        scaler.transform(arrays["train"].host_features), arrays["train"].labels
    )
    logreg_auc = {
        name: float(
            roc_auc_score(
                arrays[name].labels,
                probe.predict_proba(scaler.transform(arrays[name].host_features))[:, 1],
            )
        )
        for name in ("val", "test")
    }

    single_feature_auc = {}
    for idx, feature in enumerate(feature_names):
        single_feature_auc[feature] = {
            name: direction_agnostic_auc(arrays[name].labels, arrays[name].host_features[:, idx])
            for name in ("val", "test")
        }

    host_purity = {name: per_host_purity(arrays[name]) for name in ("train", "val", "test")}
    pure_hosts = all(
        stats["frac_c2"] in (0.0, 1.0) for split in host_purity.values() for stats in split.values()
    )

    report = {
        "host_split_dir": str(split_dir),
        "logreg_on_host_features_alone_auc": logreg_auc,
        "single_host_feature_auc_direction_agnostic": single_feature_auc,
        "per_host_label_purity": host_purity,
        "all_hosts_label_pure": bool(pure_hosts),
        "reading": (
            "Host summary features alone nearly saturate the validation axis; every host is "
            "label-pure (one infected source host per capture), so host identification is "
            "label prediction on this data. Host-aware advantages must be discounted accordingly."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] LogReg on host_features alone: val AUC={logreg_auc['val']:.4f}, test AUC={logreg_auc['test']:.4f}")
    print(f"[INFO] All hosts label-pure: {pure_hosts}")
    print(f"[SUCCESS] Wrote {out_path}")


def direction_agnostic_auc(labels: np.ndarray, values: np.ndarray) -> float:
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc)


def per_host_purity(arrays) -> dict[str, dict]:
    out: dict[str, dict] = {}
    hosts = arrays.host_ids
    for host in sorted(set(str(h) for h in hosts.tolist())):
        mask = hosts == host
        out[host] = {"sessions": int(mask.sum()), "frac_c2": float(arrays.labels[mask].mean())}
    return out


if __name__ == "__main__":
    main()
