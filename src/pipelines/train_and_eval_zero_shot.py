"""Train on UWF and evaluate on held-out CTU-13 / MCFP: the decisive zero-shot test.

This script orchestrates the core validation experiment:
1. Train the Transformer on UWF sessions only (training split).
2. Validate on UWF sessions (validation split).
3. Evaluate blind on CTU-13 and MCFP held-out datasets.

This proves whether learned behavioral patterns transfer across C2 families
and network environments, or whether the model is fitting dataset shortcuts.

Usage:
    python -m src.pipelines.train_and_eval_zero_shot \\
        --uwf_train data/processed/v2_uwf/train_sessions.npz \\
        --uwf_val data/processed/v2_uwf/val_sessions.npz \\
        --mixed_test data/processed/v2_mixed_session_only/test_sessions.npz \\
        --mixed_metadata data/processed/v2_mixed_session_only/split_metadata.json \\
        --model_dir experiments/transformer_uwf_only \\
        --epochs 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow direct execution with proper imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.train_transformer import train_model
from src.evaluation.evaluate_zero_shot import evaluate_zero_shot, print_results


def train_and_eval_zero_shot(
    uwf_train: str,
    uwf_val: str,
    mixed_test: str,
    mixed_metadata: str,
    model_dir: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    patience: int = 10,
    focal_alpha: float = 0.5,
    focal_gamma: float = 1.0,
    max_fpr_budget: float = 0.015,
) -> dict:
    """Train on UWF, evaluate on mixed held-out data."""
    
    os.makedirs(model_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("ZERO-SHOT GENERALIZATION TEST")
    print("=" * 70)
    print(f"\n[STEP 1/2] TRAINING on UWF-only dataset...")
    print(f"  Train: {uwf_train}")
    print(f"  Val:   {uwf_val}")
    print(f"  Model: {model_dir}")
    
    # Train on UWF splits
    train_model(
        npz_path=uwf_train,
        val_npz_path=uwf_val,
        model_save_dir=model_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        max_fpr_budget=max_fpr_budget,
        run_final_test_eval=False,  # Skip UWF internal test; we're going external
    )
    
    best_model_path = Path(model_dir) / "best_transformer.pth"
    if not best_model_path.exists():
        raise RuntimeError(f"Training did not produce {best_model_path}")
    
    print(f"\n[STEP 2/2] EVALUATING on held-out CTU-13 / MCFP...")
    print(f"  Test: {mixed_test}")
    print(f"  Metadata: {mixed_metadata}")
    
    # Evaluate blind on held-out sources
    results = evaluate_zero_shot(
        checkpoint_path=str(best_model_path),
        test_npz=mixed_test,
        split_metadata=mixed_metadata,
        batch_size=batch_size,
    )
    
    print_results(results)
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on UWF, evaluate zero-shot on held-out C2 sources"
    )
    parser.add_argument("--uwf_train", required=True, help="UWF training NPZ")
    parser.add_argument("--uwf_val", required=True, help="UWF validation NPZ")
    parser.add_argument("--mixed_test", required=True, help="Mixed test set (CTU-13 + MCFP)")
    parser.add_argument("--mixed_metadata", required=True, help="Split metadata JSON")
    parser.add_argument("--model_dir", default="experiments/transformer_uwf_only")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--focal_alpha", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--max_fpr_budget", type=float, default=0.015)
    args = parser.parse_args()
    
    train_and_eval_zero_shot(
        uwf_train=args.uwf_train,
        uwf_val=args.uwf_val,
        mixed_test=args.mixed_test,
        mixed_metadata=args.mixed_metadata,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        max_fpr_budget=args.max_fpr_budget,
    )


if __name__ == "__main__":
    main()
