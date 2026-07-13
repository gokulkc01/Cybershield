"""Render the Phase 1 test-axis precision-recall figure.

Two shared-axis panels keep each to <= 4 series: learned models on the left,
floor/classic detectors on the right. Phase 1 method scores come from the
saved per-seed score NPZs; the host-aware model's scores from its Phase 0
training results; the c2_transformer is re-scored from its Phase 0 seed-42
checkpoint with the identical preprocessing used at training time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.models.transformer import C2Transformer
from src.pipelines.benchmark_host_aware_models import (
    load_preprocessed_session_splits,
    score_session_model,
)

# Validated reference palette (dataviz skill), fixed slot order.
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 1 PR curves (test axis)")
    parser.add_argument("--phase1_dir", default="experiments/phase1_baselines_ctu13_same_provenance")
    parser.add_argument("--phase0_dir", default="experiments/host_aware_benchmark_ctu13_same_provenance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="docs/validated/figures/phase1_pr_curves.png")
    args = parser.parse_args()

    phase1_scores = Path(args.phase1_dir) / "scores"
    phase0_dir = Path(args.phase0_dir)

    def phase1_curve(method: str):
        data = np.load(phase1_scores / f"{method}_seed{args.seed}.npz")
        return data["test_y"], data["test_scores"]

    host_aware = json.loads(
        (phase0_dir / f"seed_{args.seed}" / "host_aware_domain_adaptive_transformer" / "training_results.json")
        .read_text(encoding="utf-8")
    )
    transformer_y, transformer_scores = rescore_c2_transformer(phase0_dir, seed=args.seed)

    left_panel = [
        ("Host-aware transformer", np.asarray(host_aware["test_labels"]), np.asarray(host_aware["test_scores"])),
        ("Random forest", *phase1_curve("random_forest")),
        ("C2 transformer", transformer_y, transformer_scores),
    ]
    right_panel = [
        ("resp_bytes (1 feature)", *phase1_curve("resp_bytes_single_feature")),
        ("Logistic probe", *phase1_curve("logistic_probe")),
        ("Beaconing (oriented)", *phase1_curve("beaconing_detector_oriented")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200, sharey=True, facecolor=SURFACE)
    prevalence = float(np.mean(left_panel[0][1]))
    slot = 0
    for ax, series, title in zip(axes, (left_panel, right_panel), ("Learned models", "Floor + classic detectors")):
        ax.set_facecolor(SURFACE)
        for name, y_true, scores in series:
            precision, recall, _ = precision_recall_curve(y_true, scores)
            pr_auc = average_precision_score(y_true, scores)
            ax.plot(recall, precision, lw=2, color=SERIES_COLORS[slot], label=f"{name}  (PR-AUC {pr_auc:.3f})")
            slot += 1
        ax.axhline(prevalence, lw=1, ls=(0, (4, 3)), color=BASELINE, zorder=1)
        ax.text(0.02, prevalence + 0.015, f"chance = test prevalence ({prevalence:.2f})", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlabel("Recall", fontsize=9, color=MUTED)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.legend(loc="lower left", fontsize=7.5, frameon=False, labelcolor=INK)
    axes[0].set_ylabel("Precision", fontsize=9, color=MUTED)
    fig.suptitle(
        "Test axis (unseen family Rbot) - precision-recall, seed 42",
        fontsize=11,
        color=INK,
        x=0.055,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    print(f"[SUCCESS] Wrote {out_path}")


def rescore_c2_transformer(phase0_dir: Path, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    session_npz_dir = phase0_dir / "baseline_session_npz"
    session_npzs = {name: str(session_npz_dir / f"{name}_sessions.npz") for name in ("train", "val", "test")}
    session_data = load_preprocessed_session_splits(session_npzs, normalize_features=True)

    checkpoint = torch.load(
        phase0_dir / f"seed_{seed}" / "c2_transformer" / "best_model.pth",
        map_location="cpu",
        weights_only=False,
    )
    model = C2Transformer(feature_dim=session_data.feature_dim, seq_len=session_data.session_len)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(session_data.test_x, dtype=torch.float32),
        torch.tensor(~session_data.test_masks.astype(bool), dtype=torch.bool),
        torch.tensor(session_data.test_y, dtype=torch.long),
        torch.zeros(len(session_data.test_y), dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    scores, targets = score_session_model(
        model,
        loader,
        torch.device("cpu"),
        lambda mdl, x, mask, domain: mdl(x, mask),
    )
    return targets, scores


if __name__ == "__main__":
    main()
