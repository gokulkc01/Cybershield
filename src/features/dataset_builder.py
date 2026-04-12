"""
dataset_builder.py
==================
Mixes C2 sessions and benign sessions into a unified training dataset.

This is the file that enforces the labeling strategy:
  label=1 : C2 traffic  (from CTU-13 botnet scenarios)
  label=0 : non-C2      (benign from CICIoT2023 + any non-C2 attacks)

Pipeline position:
    zeek_parser + session_builder → THIS FILE → train_detector.py

Responsibilities:
  1. Load C2 sessions (from CTU-13 labeled flows)
  2. Load benign sessions (from CICIoT2023 benign-only flows)
  3. Apply strict binary labeling (relabel any non-C2 attack as 0)
  4. Mix with configurable C2/benign ratio
  5. Train/val split with no leakage
  6. Save final .npz files

Usage:
    python -m src.features.dataset_builder \
        --c2_npz    data/processed/ctu13_c2_sessions.npz \
        --benign_npz data/processed/ciciot_benign_sessions.npz \
        --out_dir   data/processed/final \
        --c2_ratio  0.15
"""

import argparse
import os
import numpy as np
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split

from src.features.feature_config import (
    FEATURE_DIM, SESSION_LEN, LABEL_C2, LABEL_BENIGN, VAL_SPLIT
)
from src.features.session_builder import save_sessions, load_sessions


def mix_sessions(
    c2_sessions:     np.ndarray,
    c2_labels:       np.ndarray,
    c2_masks:        np.ndarray,
    benign_sessions: np.ndarray,
    benign_labels:   np.ndarray,
    benign_masks:    np.ndarray,
    c2_ratio:        float = 0.15,
    random_seed:     int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine C2 and benign sessions into one dataset with controlled C2 ratio.

    C2 ratio controls class balance in training.
    0.15 = 15% C2, 85% benign — realistic but enough C2 for learning.
    If you have fewer C2 sessions than the ratio demands, all C2 are kept
    and benign is downsampled. Never upsample C2 here (session_builder
    handles oversampling in the DataLoader).

    Parameters
    ----------
    c2_ratio : fraction of final dataset that should be C2 (0.0–1.0)
    """
    rng = np.random.default_rng(random_seed)

    # Verify all labels are correct
    assert (c2_labels == LABEL_C2).all(), "C2 sessions must all have label=1"
    assert (benign_labels == LABEL_BENIGN).all(), "Benign sessions must all have label=0"

    n_c2     = len(c2_sessions)
    n_target = int(n_c2 / c2_ratio)          # total dataset size to achieve ratio
    n_benign = min(n_target - n_c2, len(benign_sessions))

    # Sample benign sessions
    idx_benign = rng.choice(len(benign_sessions), size=n_benign, replace=False)
    sampled_benign   = benign_sessions[idx_benign]
    sampled_b_labels = benign_labels[idx_benign]
    sampled_b_masks  = benign_masks[idx_benign]

    # Concatenate
    sessions = np.concatenate([c2_sessions,  sampled_benign],  axis=0)
    labels   = np.concatenate([c2_labels,    sampled_b_labels], axis=0)
    masks    = np.concatenate([c2_masks,     sampled_b_masks],  axis=0)

    # Shuffle
    shuffle_idx = rng.permutation(len(sessions))
    sessions = sessions[shuffle_idx]
    labels   = labels[shuffle_idx]
    masks    = masks[shuffle_idx]

    n_total  = len(sessions)
    n_c2_out = int((labels == LABEL_C2).sum())
    print(
        f"Mixed dataset: {n_total:,} sessions | "
        f"C2: {n_c2_out:,} ({100*n_c2_out/n_total:.1f}%) | "
        f"Benign: {n_total-n_c2_out:,} ({100*(n_total-n_c2_out)/n_total:.1f}%)"
    )
    return sessions, labels, masks


def train_val_split(
    sessions:   np.ndarray,
    labels:     np.ndarray,
    masks:      np.ndarray,
    val_split:  float = VAL_SPLIT,
    random_seed: int  = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Stratified train/val split.
    Returns ((X_train, y_train, m_train), (X_val, y_val, m_val))
    """
    idx = np.arange(len(sessions))
    idx_train, idx_val = train_test_split(
        idx,
        test_size=val_split,
        stratify=labels,
        random_state=random_seed,
    )
    train = (sessions[idx_train], labels[idx_train], masks[idx_train])
    val   = (sessions[idx_val],   labels[idx_val],   masks[idx_val])

    print(
        f"Train: {len(idx_train):,} | "
        f"Val: {len(idx_val):,} | "
        f"Val C2 ratio: {labels[idx_val].mean():.3f}"
    )
    return train, val


def build_dataset(
    c2_npz:       str,
    benign_npz:   str,
    out_dir:      str,
    c2_ratio:     float = 0.15,
    val_split:    float = VAL_SPLIT,
    random_seed:  int   = 42,
) -> None:
    """
    Full dataset build pipeline.

    Loads C2 and benign session .npz files, mixes them, splits into
    train/val, and saves final .npz files to out_dir.

    Output files:
        out_dir/train_sessions.npz
        out_dir/val_sessions.npz
    """
    print("Loading C2 sessions...")
    c2_sess, c2_lbl, c2_msk = load_sessions(c2_npz)
    print(f"  C2 sessions: {len(c2_sess):,}")

    print("Loading benign sessions...")
    b_sess, b_lbl, b_msk = load_sessions(benign_npz)
    print(f"  Benign sessions: {len(b_sess):,}")

    # Enforce labels — never trust loaded labels blindly
    c2_lbl = np.ones(len(c2_sess),  dtype=np.int64)
    b_lbl  = np.zeros(len(b_sess),  dtype=np.int64)

    # Validate shapes
    assert c2_sess.shape[1:] == (SESSION_LEN, FEATURE_DIM), (
        f"C2 session shape mismatch: {c2_sess.shape}"
    )
    assert b_sess.shape[1:] == (SESSION_LEN, FEATURE_DIM), (
        f"Benign session shape mismatch: {b_sess.shape}"
    )

    # Mix
    sessions, labels, masks = mix_sessions(
        c2_sess, c2_lbl, c2_msk,
        b_sess,  b_lbl,  b_msk,
        c2_ratio=c2_ratio, random_seed=random_seed,
    )

    # Split
    (X_tr, y_tr, m_tr), (X_v, y_v, m_v) = train_val_split(
        sessions, labels, masks, val_split=val_split, random_seed=random_seed
    )

    # Save
    os.makedirs(out_dir, exist_ok=True)
    save_sessions(X_tr, y_tr, m_tr, os.path.join(out_dir, "train_sessions.npz"))
    save_sessions(X_v,  y_v,  m_v,  os.path.join(out_dir, "val_sessions.npz"))
    print(f"\nDataset saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2_npz",     required=True, help="Path to C2 sessions .npz")
    parser.add_argument("--benign_npz", required=True, help="Path to benign sessions .npz")
    parser.add_argument("--out_dir",    required=True, help="Output directory")
    parser.add_argument("--c2_ratio",   type=float, default=0.15)
    parser.add_argument("--val_split",  type=float, default=VAL_SPLIT)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    build_dataset(
        c2_npz=args.c2_npz,
        benign_npz=args.benign_npz,
        out_dir=args.out_dir,
        c2_ratio=args.c2_ratio,
        val_split=args.val_split,
        random_seed=args.seed,
    )