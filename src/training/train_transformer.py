"""
train_transformer.py
====================
Training loop for the CyberShield Transformer.

Handles PyTorch device allocation, training/validation loops, early stopping,
dynamic threshold tuning (to prevent data leaks), and final blind test evaluation.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# Import our custom architecture and dataloader
from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.npz_utils import load_session_npz
from src.data_loader.normalization import fit_feature_normalizer
from src.data_loader.torch_dataset import create_dataloaders, C2SessionDataset
from src.models.transformer import C2Transformer
from src.evaluation.operating_point import find_threshold_under_fpr_budget, is_better_operating_point
from src.losses.focal_loss import FocalLoss
from src.features.feature_config import FEATURE_DIM, FOCAL_ALPHA, FOCAL_GAMMA, MAX_FPR_BUDGET, SESSION_LEN


def _parse_feature_list(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return ()
    return tuple(feature.strip() for feature in raw_value.split(",") if feature.strip())

def train_model(
    npz_path: str = "data/processed/ctu13_c2_sessions.npz",
    model_save_dir: str = "experiments/transformer",
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-4,
    patience: int = 10,
    min_flows: int = 5,
    focal_alpha: float = FOCAL_ALPHA,
    focal_gamma: float = FOCAL_GAMMA,
    max_fpr_budget: float = MAX_FPR_BUDGET,
    fpr_guard_band: float = 1.0,
    use_derivative_features: bool = False,
    normalize_features: bool = True,
    log_scale_features: tuple[str, ...] | None = None,
    ablate_features: tuple[str, ...] = (),
    run_final_test_eval: bool = True,
    val_npz_path: str | None = None,
):
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_transformer.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Initializing Training on Device: {device}")

    transform_config = FeatureTransformConfig(
        log_scale_features=tuple(log_scale_features) if log_scale_features is not None else FeatureTransformConfig().log_scale_features,
        ablate_features=tuple(ablate_features),
    )

    # Data loaders: use separate val NPZ if provided (for v2 splits), else internal split
    if val_npz_path:
        print(f"[INFO] Loading train/val from separate NPZs (v2 mode)...")
        train_seq, train_labels, train_masks = load_session_npz(npz_path)
        val_seq, val_labels, val_masks = load_session_npz(val_npz_path)
        
        train_seq = apply_feature_transforms(train_seq, train_masks, transform_config)
        val_seq = apply_feature_transforms(val_seq, val_masks, transform_config)
        
        if normalize_features:
            normalizer = fit_feature_normalizer(train_seq, train_masks)
            train_seq = normalizer.transform(train_seq, train_masks)
            val_seq = normalizer.transform(val_seq, val_masks)
        else:
            normalizer = None
        
        train_dataset = C2SessionDataset(train_seq, train_masks, train_labels)
        val_dataset = C2SessionDataset(val_seq, val_masks, val_labels)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = None
    else:
        # Data loaders are intentionally built with natural class priors (original behavior).
        train_loader, val_loader, test_loader, normalizer, transform_config = create_dataloaders(
            npz_path,
            batch_size,
            min_flows,
            normalize=normalize_features,
            transform_config=transform_config,
        )

    model = C2Transformer(
        feature_dim=FEATURE_DIM,
        seq_len=SESSION_LEN,
        use_derivative_features=use_derivative_features,
    ).to(device)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    effective_fpr_budget = max_fpr_budget * fpr_guard_band

    print(f"[INFO] Loss: FocalLoss(alpha={focal_alpha}, gamma={focal_gamma})")
    print(f"[INFO] Derivative feature augmentation: {use_derivative_features}")
    print(f"[INFO] Feature normalization: {normalize_features}")
    print(f"[INFO] Feature ablations: {list(transform_config.ablate_features)}")
    print(
        f"[INFO] Validation threshold policy: max Recall under FPR <= {effective_fpr_budget:.4f} "
        f"(external budget {max_fpr_budget:.4f}, guard band {fpr_guard_band:.3f})"
    )

    best_val_recall_budget = 0.0
    best_val_f1 = 0.0
    best_val_fpr_budget = float("inf")
    best_val_threshold = 0.5
    epochs_no_improve = 0

    print("\n" + "="*60)
    print("BEGINNING TRAINING LOOP")
    print("="*60)

    # --- MAIN EPOCH LOOP ---
    for epoch in range(epochs):
        # -- Training Phase --
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_mask, batch_y in train_loader:
            batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        val_probs, val_targets = [], []
        
        with torch.no_grad():
            for batch_x, batch_mask, batch_y in val_loader:
                batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                
                logits = model(batch_x, batch_mask)
                loss = criterion(logits, batch_y.float())
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_probs = np.array(val_probs)
        val_targets = np.array(val_targets)
        
        # Calculate AUC
        try:
            val_auc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            val_auc = 0.0
            
        # Validation operating point: maximize recall under an explicit FPR budget.
        epoch_thresh, val_recall_budget, val_fpr_budget, has_feasible = find_threshold_under_fpr_budget(
            val_targets, val_probs, max_fpr=effective_fpr_budget
        )
        val_preds = (val_probs >= epoch_thresh).astype(int)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        scheduler.step(val_recall_budget)

        feasibility = "yes" if has_feasible else "no"
        print(
            f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | "
            f"Val Recall@FPR<=B: {val_recall_budget:.4f} | Val FPR: {val_fpr_budget:.4f} | "
            f"Val F1: {val_f1:.4f} (Thresh: {epoch_thresh:.4f}, Feasible: {feasibility})"
        )

        # -- Early Stopping & Model Checkpointing --
        if is_better_operating_point(
            candidate_recall=val_recall_budget,
            candidate_f1=val_f1,
            candidate_fpr=val_fpr_budget,
            best_recall=best_val_recall_budget,
            best_f1=best_val_f1,
            best_fpr=best_val_fpr_budget,
        ):
            best_val_recall_budget = val_recall_budget
            best_val_f1 = val_f1
            best_val_fpr_budget = val_fpr_budget
            best_val_threshold = epoch_thresh
            epochs_no_improve = 0
            
            # Save Model AND Threshold
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimal_threshold': best_val_threshold,
                'max_fpr_budget': max_fpr_budget,
                'validation_fpr_budget': effective_fpr_budget,
                'fpr_guard_band': fpr_guard_band,
                'val_recall_budget': val_recall_budget,
                'val_fpr_budget': val_fpr_budget,
                'val_f1_at_budget': val_f1,
                'best_val_recall_budget': best_val_recall_budget,
                'best_val_f1': best_val_f1,
                'best_val_fpr_budget': best_val_fpr_budget,
                'use_derivative_features': use_derivative_features,
                'normalize_features': normalize_features,
                'feature_normalizer': normalizer.to_checkpoint_dict() if normalizer is not None else None,
                'feature_transform_config': transform_config.to_checkpoint_dict(),
                'feature_dim': FEATURE_DIM,
                'session_len': SESSION_LEN,
            }, best_model_path)
            print(f"  --> Model improved! Saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[INFO] Early stopping triggered after {epoch+1} epochs.")
                break

    if run_final_test_eval:
        # --- FINAL EVALUATION ON TEST SET ---
        print("\n" + "="*60)
        print("LOADING BEST MODEL FOR FINAL BLIND TEST EVALUATION")
        print("="*60)
        
        # Load state dict and optimal threshold
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        blind_threshold = checkpoint['optimal_threshold']
        saved_fpr_budget = checkpoint.get('max_fpr_budget', max_fpr_budget)
        
        model.eval()
        test_probs, test_targets = [], []
        
        with torch.no_grad():
            for batch_x, batch_mask, batch_y in test_loader:
                batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                logits = model(batch_x, batch_mask)
                probs = torch.sigmoid(logits)
                
                test_probs.extend(probs.cpu().numpy())
                test_targets.extend(batch_y.cpu().numpy())
                
        test_probs = np.array(test_probs)
        test_targets = np.array(test_targets)
        
        # BLIND PREDICTION (Using the Validation Threshold)
        final_preds = (test_probs >= blind_threshold).astype(int)
        
        try:
            test_auc = roc_auc_score(test_targets, test_probs)
        except ValueError:
            test_auc = 0.0
            
        final_f1 = f1_score(test_targets, final_preds, zero_division=0)
        
        print(f"Blind Validation Threshold : {blind_threshold:.4f}")
        print(f"FPR Budget                : {saved_fpr_budget:.4f}")
        print(f"Final Test AUC             : {test_auc:.4f}")
        print(f"Final Test F1              : {final_f1:.4f}\n")
        
        print("Classification Report:")
        print(classification_report(test_targets, final_preds, target_names=["Benign", "C2"], zero_division=0))
        
        cm = confusion_matrix(test_targets, final_preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        budget_ok = fpr <= (saved_fpr_budget + 1e-12)
        
        print("\nConfusion Matrix:")
        print(f"            Predicted Benign  Predicted C2")
        print(f" True Benign      {tn:>8,}      {fp:>8,}    (FPR: {fpr:.4f})")
        print(f" True C2          {fn:>8,}      {tp:>8,}    (TPR: {tpr:.4f})")
        print(f"FPR budget met           : {budget_ok}")

    return best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CyberShield Transformer")
    parser.add_argument("--npz_path", default="data/processed/ctu13_c2_sessions.npz")
    parser.add_argument("--model_save_dir", default="experiments/transformer")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_flows", type=int, default=5)
    parser.add_argument("--focal_alpha", type=float, default=FOCAL_ALPHA)
    parser.add_argument("--focal_gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--max_fpr_budget", type=float, default=MAX_FPR_BUDGET)
    parser.add_argument("--guard_band", type=float, default=1.0)
    parser.add_argument("--use_derivative_features", action="store_true")
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument("--normalize_features", dest="normalize_features", action="store_true")
    normalize_group.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    parser.set_defaults(normalize_features=True)
    parser.add_argument("--log_scale_features", default=None, help="Comma-separated feature names to log-scale")
    parser.add_argument("--ablate_features", default="", help="Comma-separated feature names to zero out")
    args = parser.parse_args()

    train_model(
        npz_path=args.npz_path,
        model_save_dir=args.model_save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        min_flows=args.min_flows,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        max_fpr_budget=args.max_fpr_budget,
        fpr_guard_band=args.guard_band,
        use_derivative_features=args.use_derivative_features,
        normalize_features=args.normalize_features,
        log_scale_features=_parse_feature_list(args.log_scale_features),
        ablate_features=_parse_feature_list(args.ablate_features),
    )
