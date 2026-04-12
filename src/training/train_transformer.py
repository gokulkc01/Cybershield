"""
train_transformer.py
====================
Training loop for the CyberShield Transformer.

Handles PyTorch device allocation, training/validation loops, early stopping,
dynamic threshold tuning (to prevent data leaks), and final blind test evaluation.
"""

import os
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, roc_auc_score, roc_curve

# Import our custom architecture and dataloader
from src.models.transformer import C2Transformer
from src.data_loader.torch_dataset import create_dataloaders
from src.losses.focal_loss import FocalLoss
from src.features.feature_config import FOCAL_ALPHA, FOCAL_GAMMA, MAX_FPR_BUDGET

def find_best_threshold(y_true, y_prob):
    """Finds the optimal probability threshold to maximize F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]
    
    # Avoid division by zero
    f1_scores = np.where((precisions + recalls) > 0, 
                         2 * (precisions * recalls) / (precisions + recalls), 
                         0.0)
    
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def find_threshold_under_fpr_budget(y_true, y_prob, max_fpr: float = 0.005):
    """Select threshold maximizing recall while keeping FPR <= max_fpr.

    Tie-break rule: choose the highest threshold among candidates with equal recall
    to reduce avoidable false positives at deployment.
    """
    if not (0.0 <= max_fpr <= 1.0):
        raise ValueError(f"max_fpr must be in [0, 1], got {max_fpr}")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.size == 0:
        return 1.0, 0.0, 0.0, False

    has_positive = np.any(y_true == 1)
    has_negative = np.any(y_true == 0)

    if not has_positive:
        return 1.0, 0.0, 0.0, False
    if not has_negative:
        return 0.0, 1.0, 0.0, True

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_idx = np.where(fpr <= max_fpr)[0]

    if len(valid_idx) == 0:
        # No feasible operating point under the budget; pick strictest threshold.
        return 1.0, 0.0, 0.0, False

    valid_fpr = fpr[valid_idx]
    valid_tpr = tpr[valid_idx]
    valid_thr = thresholds[valid_idx]

    best_tpr = float(valid_tpr.max())
    best_candidates = np.where(valid_tpr == best_tpr)[0]
    chosen = best_candidates[np.argmax(valid_thr[best_candidates])]

    return float(valid_thr[chosen]), float(valid_tpr[chosen]), float(valid_fpr[chosen]), True


def is_better_operating_point(
    candidate_recall: float,
    candidate_f1: float,
    candidate_fpr: float,
    best_recall: float,
    best_f1: float,
    best_fpr: float,
) -> bool:
    """Return True when the candidate operating point is preferred.

    Primary objective: maximize Recall under the FPR budget.
    Tie-breaks: higher F1, then lower FPR.
    """
    if candidate_recall > best_recall + 1e-12:
        return True
    if abs(candidate_recall - best_recall) <= 1e-12:
        if candidate_f1 > best_f1 + 1e-12:
            return True
        if abs(candidate_f1 - best_f1) <= 1e-12 and candidate_fpr < best_fpr - 1e-12:
            return True
    return False

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
):
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_transformer.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Initializing Training on Device: {device}")

    # Data loaders are intentionally built with natural class priors.
    train_loader, val_loader, test_loader = create_dataloaders(npz_path, batch_size, min_flows)

    model = C2Transformer().to(device)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    print(f"[INFO] Loss: FocalLoss(alpha={focal_alpha}, gamma={focal_gamma})")
    print(f"[INFO] Validation threshold policy: max Recall under FPR <= {max_fpr_budget:.4f}")

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
            val_targets, val_probs, max_fpr=max_fpr_budget
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
                'val_recall_budget': val_recall_budget,
                'val_fpr_budget': val_fpr_budget,
                'val_f1_at_budget': val_f1,
                'best_val_recall_budget': best_val_recall_budget,
                'best_val_f1': best_val_f1,
                'best_val_fpr_budget': best_val_fpr_budget,
            }, best_model_path)
            print(f"  --> Model improved! Saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[INFO] Early stopping triggered after {epoch+1} epochs.")
                break

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

if __name__ == "__main__":
    train_model(
        npz_path="data/processed/ctu13_c2_sessions.npz",
        model_save_dir="experiments/transformer",
        batch_size=64,
        epochs=50,
        lr=1e-4,     
        patience=8,  
        min_flows=5,
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA,
        max_fpr_budget=MAX_FPR_BUDGET,
    )