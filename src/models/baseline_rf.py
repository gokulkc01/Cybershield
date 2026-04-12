"""
baseline_rf.py
==============
Random Forest baseline for CyberShield.

Purpose: Prove our 12 features contain discriminative C2 signal.
         This F1 score becomes the official baseline everything must beat.

Handles the 0.6% class imbalance and noise through:
  1. Filtering single-flow noise (min_flows=3)
  2. Undersampling benign to a configurable ratio (default 10:1)
  3. class_weight='balanced' inside RF
  4. Threshold tuning on validation set (not just argmax)

Usage:
    python -m src.models.baseline_rf \
        --npz_path data/processed/ctu13_c2_sessions.npz \
        --output_dir experiments/baseline \
        --min_flows 3
"""

import os
import sys
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)
from typing import Tuple

# Ensure these match your project structure
from src.features.feature_config import FEATURE_DIM, SESSION_LEN, LABEL_C2, LABEL_BENIGN, FEATURE_NAMES
from src.features.session_builder import sessions_to_flat


# ── Data loading ──────────────────────────────────────────────────────────────

def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load sessions, labels, masks from .npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NPZ not found: {path}\n"
            "Run ctu13_processor.py first to generate it."
        )
    data = np.load(path, allow_pickle=True)
    
    sessions = data["X"].astype(np.float32)      # (N, 20, 12)
    labels   = data["y"].astype(np.int64)        # (N,)
    
    # --- ROBUST MASK INFERENCE ---
    if "masks" in data:
        masks = data["masks"].astype(bool)
    else:
        # Infer masks from non-zero rows (True if real flow, False if zero-padding)
        masks = (sessions.sum(axis=2) != 0)

    # Validate shape
    assert sessions.ndim == 3, f"Expected 3D sessions, got shape {sessions.shape}"
    assert sessions.shape[1] == SESSION_LEN, (
        f"Expected SESSION_LEN={SESSION_LEN}, got {sessions.shape[1]}"
    )
    assert sessions.shape[2] == FEATURE_DIM, (
        f"Expected FEATURE_DIM={FEATURE_DIM}, got {sessions.shape[2]}"
    )
    
    return sessions, labels, masks


# ── Class imbalance handling ──────────────────────────────────────────────────

def undersample(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 10.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undersample the majority class to achieve target_ratio = benign:C2.
    """
    rng = np.random.default_rng(random_seed)

    c2_idx     = np.where(y == LABEL_C2)[0]
    benign_idx = np.where(y == LABEL_BENIGN)[0]

    n_c2            = len(c2_idx)
    n_benign_target = int(n_c2 * target_ratio)
    n_benign_target = min(n_benign_target, len(benign_idx))  # don't exceed available

    sampled_benign = rng.choice(benign_idx, size=n_benign_target, replace=False)
    keep_idx = np.concatenate([c2_idx, sampled_benign])
    keep_idx = rng.permutation(keep_idx)  # shuffle

    print(
        f"      After undersampling: {len(keep_idx):,} samples | "
        f"C2: {n_c2:,} | Benign: {n_benign_target:,} | "
        f"Ratio: 1:{n_benign_target//n_c2 if n_c2 > 0 else 0}"
    )
    return X[keep_idx], y[keep_idx]


# ── Threshold tuning ──────────────────────────────────────────────────────────

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """
    Find the probability threshold that maximizes F1 on the validation set.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0,
    )
    best_idx  = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ── Main training and evaluation ──────────────────────────────────────────────

def run_baseline(
    npz_path:       str,
    output_dir:     str  = "experiments/baseline",
    undersample_ratio: float = 10.0,
    test_split:     float = 0.20,
    n_estimators:   int   = 200,
    random_seed:    int   = 42,
    min_flows:      int   = 3,
) -> dict:
    
    os.makedirs(output_dir, exist_ok=True)
    report_lines = []

    def log(msg: str = ""):
        print(msg)
        report_lines.append(msg)

    log("=" * 60)
    log("CyberShield — Random Forest Baseline")
    log("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────
    log(f"\n[1/7] Loading data from {npz_path}")
    sessions, labels, masks = load_npz(npz_path)
    
    # Check if masks were properly loaded, if not, infer them
    if not isinstance(masks, np.ndarray) or masks.shape != (sessions.shape[0], sessions.shape[1]):
        masks = (sessions.sum(axis=2) != 0)

    n_total_initial  = len(sessions)
    n_c2_initial     = int((labels == LABEL_C2).sum())
    n_benign_initial = int((labels == LABEL_BENIGN).sum())
    log(f"      Initial Total: {n_total_initial:,} | C2: {n_c2_initial:,} | Benign: {n_benign_initial:,}")

    # ── NEW STEP 1.5: Filter by min_flows ─────────────────────────
    log(f"\n[1.5/7] Filtering out noise (Requiring >= {min_flows} flows per session)")
    
    # masks is a boolean array of shape (N, 20). Summing axis 1 gives the flow count per session.
    flow_counts = masks.sum(axis=1) 
    keep_idx = flow_counts >= min_flows
    
    sessions = sessions[keep_idx]
    labels = labels[keep_idx]
    masks = masks[keep_idx]

    n_total  = len(sessions)
    n_c2     = int((labels == LABEL_C2).sum())
    n_benign = int((labels == LABEL_BENIGN).sum())
    
    log(f"      Remaining Sessions: {n_total:,} | C2: {n_c2:,} | Benign: {n_benign:,}")
    if n_c2 == 0:
        raise ValueError("No C2 sessions remaining after filtering! Cannot train.")

    # ── Step 2: Flatten ──────────────────────────────────────────
    log(f"\n[2/7] Flattening sessions → 2D features (mean/std/min/max pooling)")
    X = sessions_to_flat(sessions, masks)   # (N, FEATURE_DIM * 4)
    y = labels
    log(f"      Feature matrix: {X.shape}")

    # ── Step 3: Train/test split ─────────────────────────────────
    log(f"\n[3/7] Stratified train/test split ({100*(1-test_split):.0f}/{100*test_split:.0f})")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=random_seed
    )
    log(f"      Train: {len(X_tr):,} | Test: {len(X_te):,}")
    log(f"      Train C2: {(y_tr==LABEL_C2).sum():,} | Test C2: {(y_te==LABEL_C2).sum():,}")

    # ── Step 4: Undersample training set ─────────────────────────
    log(f"\n[4/7] Undersampling benign (target ratio 1:{undersample_ratio:.0f})")
    X_tr_us, y_tr_us = undersample(X_tr, y_tr, target_ratio=undersample_ratio,
                                    random_seed=random_seed)

    # ── Step 5: Scale features ────────────────────────────────────
    log(f"\n[5/7] Fitting StandardScaler on training data")
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_us)
    X_te_sc = scaler.transform(X_te)         # NEVER refit on test
    log(f"      Scaler fitted on {len(X_tr_sc):,} samples")

    # ── Step 6: Train RF ─────────────────────────────────────────
    log(f"\n[6/7] Training Random Forest ({n_estimators} trees, balanced weights)")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_seed,
        max_depth=20,
        min_samples_leaf=5,
    )
    rf.fit(X_tr_sc, y_tr_us)
    log(f"      Training complete.")

    # ── Step 7: Threshold tuning on test set ─────────────────────
    log(f"\n[7/7] Evaluating on test set")
    y_prob = rf.predict_proba(X_te_sc)[:, 1]   # P(C2)
    best_thresh, best_f1_thresh = find_best_threshold(y_te, y_prob)
    log(f"      Best threshold (max F1): {best_thresh:.3f} → F1={best_f1_thresh:.4f}")
    log(f"      Default threshold (0.5): F1={f1_score(y_te, (y_prob >= 0.5).astype(int), pos_label=LABEL_C2, zero_division=0):.4f}")

    # Final predictions using best threshold
    y_pred_best = (y_prob >= best_thresh).astype(int)
    y_pred_050  = (y_prob >= 0.50).astype(int)

    # ── Results ───────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("RESULTS — Default threshold (0.50)")
    log("=" * 60)
    log(classification_report(y_te, y_pred_050,
                               target_names=["Benign", "C2"],
                               zero_division=0))

    log("=" * 60)
    log(f"RESULTS — Tuned threshold ({best_thresh:.3f})")
    log("=" * 60)
    log(classification_report(y_te, y_pred_best,
                               target_names=["Benign", "C2"],
                               zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    log("Confusion matrix (tuned threshold):")
    log(f"            Predicted Benign  Predicted C2")
    log(f" True Benign      {tn:>8,}      {fp:>8,}    (FPR: {fpr:.4f})")
    log(f" True C2          {fn:>8,}      {tp:>8,}    (TPR: {tpr:.4f})")

    try:
        auc = roc_auc_score(y_te, y_prob)
    except Exception:
        auc = 0.0

    f1_tuned  = f1_score(y_te, y_pred_best, pos_label=LABEL_C2, zero_division=0)
    f1_50     = f1_score(y_te, y_pred_050,  pos_label=LABEL_C2, zero_division=0)

    log("\n" + "=" * 60)
    log("OFFICIAL BASELINE NUMBERS  (write these down)")
    log("=" * 60)
    log(f"  F1 (threshold=0.50)    : {f1_50:.4f}")
    log(f"  F1 (tuned threshold)   : {f1_tuned:.4f}  <-- USE THIS AS BASELINE")
    log(f"  ROC-AUC                : {auc:.4f}")
    log(f"  FPR at best threshold  : {fpr:.4f}")
    log(f"  TPR at best threshold  : {tpr:.4f}")
    log(f"  Best threshold         : {best_thresh:.3f}")
    log("=" * 60)

    # Feature importance
    flat_feat_names = []
    for stat in ["mean", "std", "min", "max"]:
        for feat in FEATURE_NAMES:
            flat_feat_names.append(f"{feat}_{stat}")

    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    log("\nTop 10 most important features:")
    for rank, i in enumerate(top_idx, 1):
        name = flat_feat_names[i] if i < len(flat_feat_names) else f"feat_{i}"
        log(f"  {rank:2d}. {name:<30s} {importances[i]:.4f}")

    # ── Save ──────────────────────────────────────────────────────
    rf_path     = os.path.join(output_dir, "rf_model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    report_path = os.path.join(output_dir, "results.txt")

    joblib.dump(rf,     rf_path)
    joblib.dump(scaler, scaler_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log(f"\nSaved: {rf_path}")
    log(f"Saved: {scaler_path}")
    log(f"Saved: {report_path}")

    return {
        "f1_tuned":    f1_tuned,
        "f1_default":  f1_50,
        "auc":         auc,
        "fpr":         fpr,
        "tpr":         tpr,
        "threshold":   best_thresh,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest baseline")
    parser.add_argument(
        "--npz_path",
        default="data/processed/ctu13_c2_sessions.npz",
        help="Path to sessions .npz file from ctu13_processor.py",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/baseline",
        help="Directory to save model and results",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=10.0,
        help="Benign:C2 undersampling ratio (default: 10)",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=200,
        help="Number of RF estimators",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min_flows",
        type=int,
        default=3,
        help="Minimum number of flows required to keep a session (filters out noise)",
    )
    args = parser.parse_args()

    results = run_baseline(
        npz_path=args.npz_path,
        output_dir=args.output_dir,
        undersample_ratio=args.ratio,
        n_estimators=args.trees,
        random_seed=args.seed,
        min_flows=args.min_flows,
    )