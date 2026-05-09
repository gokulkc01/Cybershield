"""Train and evaluate the controlled CTU-13 multifamily experiment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.evaluate_zero_shot import evaluate_zero_shot, print_results
from src.features.feature_config_experiment import FEATURE_DIM, FEATURE_NAMES, SESSION_LEN
from src.training.train_transformer import train_model


def train_and_eval_multifamily(
    train_npz: str,
    val_npz: str,
    test_npz: str,
    split_metadata: str,
    model_dir: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    patience: int = 10,
    focal_alpha: float = 0.5,
    focal_gamma: float = 1.0,
    max_fpr_budget: float = 0.015,
) -> dict:
    """Train on multifamily CTU-13 data and evaluate on the held-out family."""

    os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("CTU-13 MULTIFAMILY GENERALIZATION TEST")
    print("=" * 70)
    print("\n[STEP 1/2] TRAINING on multifamily CTU-13 data...")
    print(f"  Train: {train_npz}")
    print(f"  Val:   {val_npz}")
    print(f"  Model: {model_dir}")

    train_model(
        npz_path=train_npz,
        val_npz_path=val_npz,
        model_save_dir=model_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        max_fpr_budget=max_fpr_budget,
        run_final_test_eval=False,
        expected_feature_names=tuple(FEATURE_NAMES),
        expected_feature_dim=FEATURE_DIM,
        expected_session_len=SESSION_LEN,
    )

    best_model_path = Path(model_dir) / "best_transformer.pth"
    if not best_model_path.exists():
        raise RuntimeError(f"Training did not produce {best_model_path}")

    print("\n[STEP 2/2] EVALUATING on held-out multifamily test set...")
    print(f"  Test: {test_npz}")
    print(f"  Metadata: {split_metadata}")

    results = evaluate_zero_shot(
        checkpoint_path=str(best_model_path),
        test_npz=test_npz,
        split_metadata=split_metadata,
        batch_size=batch_size,
        expected_feature_names=tuple(FEATURE_NAMES),
        expected_feature_dim=FEATURE_DIM,
        expected_session_len=SESSION_LEN,
    )

    print_results(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on CTU-13 multifamily data and evaluate zero-shot")
    parser.add_argument("--train_npz", required=True, help="Training NPZ")
    parser.add_argument("--val_npz", required=True, help="Validation NPZ")
    parser.add_argument("--test_npz", required=True, help="Held-out test NPZ")
    parser.add_argument("--split_metadata", required=True, help="Split metadata JSON")
    parser.add_argument("--model_dir", default="experiments/multifamily_generalization")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--focal_alpha", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--max_fpr_budget", type=float, default=0.015)
    args = parser.parse_args()

    train_and_eval_multifamily(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        test_npz=args.test_npz,
        split_metadata=args.split_metadata,
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
