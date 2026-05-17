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
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# Import our custom architecture and dataloader
from src.data_loader.extended_feature_transforms import (
    ExtendedFeatureTransformConfig,
    apply_extended_feature_transforms,
)
from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.npz_utils import detect_npz_schema, load_session_npz
from src.data_loader.normalization import fit_feature_normalizer
from src.data_loader.torch_dataset import create_dataloaders, C2SessionDataset
from src.models.transformer import C2Transformer
from src.evaluation.operating_point import find_threshold_under_fpr_budget, is_better_operating_point
from src.losses.focal_loss import FocalLoss
from src.features.feature_config import FEATURE_DIM, FOCAL_ALPHA, FOCAL_GAMMA, MAX_FPR_BUDGET, SESSION_LEN
from src.features.feature_config_extended import FEATURE_NAMES_EXTENDED, FEATURE_SCHEMA_VERSION as EXTENDED_SCHEMA_VERSION


def _parse_feature_list(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return ()
    return tuple(feature.strip() for feature in raw_value.split(",") if feature.strip())


def _coerce_transform_config(schema_version: str, transform_config: FeatureTransformConfig | ExtendedFeatureTransformConfig | None):
    if schema_version == EXTENDED_SCHEMA_VERSION:
        if isinstance(transform_config, ExtendedFeatureTransformConfig):
            return transform_config
        if transform_config is None:
            return ExtendedFeatureTransformConfig()
        return ExtendedFeatureTransformConfig(
            log_scale_features=tuple(transform_config.log_scale_features),
            ablate_features=tuple(transform_config.ablate_features),
            ablate_groups=(),
        )

    if isinstance(transform_config, FeatureTransformConfig):
        return transform_config
    if transform_config is None:
        return FeatureTransformConfig()
    return FeatureTransformConfig(
        log_scale_features=tuple(transform_config.log_scale_features),
        ablate_features=tuple(transform_config.ablate_features),
    )


def _apply_transforms_for_schema(
    sequences: np.ndarray,
    masks: np.ndarray,
    schema_version: str,
    transform_config: FeatureTransformConfig | ExtendedFeatureTransformConfig,
    feature_names: tuple[str, ...] | None,
) -> np.ndarray:
    if schema_version == EXTENDED_SCHEMA_VERSION:
        return apply_extended_feature_transforms(sequences, masks, transform_config)  # type: ignore[arg-type]
    return apply_feature_transforms(sequences, masks, transform_config, feature_names=feature_names)


def _load_npz_dataset(
    npz_path: str,
    resolved_feature_names: tuple[str, ...] | None,
    resolved_session_len: int,
    schema_version: str,
    transform_config: FeatureTransformConfig | ExtendedFeatureTransformConfig,
    normalize: bool,
    normalizer,
):
    sequences, labels, masks = load_session_npz(
        npz_path,
        expected_feature_names=resolved_feature_names,
        expected_session_len=resolved_session_len,
    )
    sequences = _apply_transforms_for_schema(sequences, masks, schema_version, transform_config, resolved_feature_names)
    if normalize:
        sequences = normalizer.transform(sequences, masks)
    return C2SessionDataset(sequences, masks, labels)

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
    balanced_sampling: bool = False,
    seed: int = 42,
    init_checkpoint_path: str | None = None,
    val_npz_path: str | None = None,
    test_npz_path: str | None = None,
    replay_npz_path: str | None = None,
    teacher_checkpoint_path: str | None = None,
    replay_loss_weight: float = 1.0,
    distill_weight: float = 0.5,
    expected_feature_names: tuple[str, ...] | None = None,
    expected_feature_dim: int = FEATURE_DIM,
    expected_session_len: int = SESSION_LEN,
):
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_transformer.pth")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Initializing Training on Device: {device}")

    train_schema = detect_npz_schema(npz_path)
    resolved_feature_names = tuple(expected_feature_names) if expected_feature_names is not None else tuple(train_schema["feature_names"]) if train_schema["feature_names"] else None
    resolved_feature_dim = expected_feature_dim
    if resolved_feature_names is not None:
        resolved_feature_dim = len(resolved_feature_names)
    elif expected_feature_dim == FEATURE_DIM and train_schema["feature_dim"]:
        resolved_feature_dim = int(train_schema["feature_dim"])
    resolved_session_len = expected_session_len or int(train_schema["session_len"]) or SESSION_LEN
    schema_version = str(train_schema["schema_version"])

    base_transform_config = FeatureTransformConfig(
        log_scale_features=tuple(log_scale_features) if log_scale_features is not None else FeatureTransformConfig().log_scale_features,
        ablate_features=tuple(ablate_features),
    )
    transform_config = _coerce_transform_config(schema_version, base_transform_config)

    # Data loaders: use separate val NPZ if provided (for v2 splits), else internal split
    if val_npz_path:
        print(f"[INFO] Loading train/val from separate NPZs (v2 mode)...")
        val_schema = detect_npz_schema(val_npz_path)
        if val_schema["feature_names"] and resolved_feature_names and tuple(val_schema["feature_names"]) != resolved_feature_names:
            raise ValueError("Train/validation NPZ feature schemas do not match.")
        train_seq, train_labels, train_masks = load_session_npz(
            npz_path,
            expected_feature_names=resolved_feature_names,
            expected_session_len=resolved_session_len,
        )
        val_seq, val_labels, val_masks = load_session_npz(
            val_npz_path,
            expected_feature_names=resolved_feature_names,
            expected_session_len=resolved_session_len,
        )
        
        train_seq = _apply_transforms_for_schema(train_seq, train_masks, schema_version, transform_config, resolved_feature_names)
        val_seq = _apply_transforms_for_schema(val_seq, val_masks, schema_version, transform_config, resolved_feature_names)
        
        if normalize_features:
            normalizer = fit_feature_normalizer(train_seq, train_masks)
            train_seq = normalizer.transform(train_seq, train_masks)
            val_seq = normalizer.transform(val_seq, val_masks)
        else:
            normalizer = None
        
        train_dataset = C2SessionDataset(train_seq, train_masks, train_labels)
        val_dataset = C2SessionDataset(val_seq, val_masks, val_labels)

        if balanced_sampling:
            label_tensor = torch.tensor(train_labels, dtype=torch.long)
            class_counts = torch.bincount(label_tensor)
            class_weights = 1.0 / (class_counts.float() + 1e-6)
            sample_weights = class_weights[label_tensor]
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if test_npz_path:
            test_schema = detect_npz_schema(test_npz_path)
            if test_schema["feature_names"] and resolved_feature_names and tuple(test_schema["feature_names"]) != resolved_feature_names:
                raise ValueError("Train/test NPZ feature schemas do not match.")
            test_seq, test_labels, test_masks = load_session_npz(
                test_npz_path,
                expected_feature_names=resolved_feature_names,
                expected_session_len=resolved_session_len,
            )
            test_seq = _apply_transforms_for_schema(test_seq, test_masks, schema_version, transform_config, resolved_feature_names)
            if normalize_features:
                test_seq = normalizer.transform(test_seq, test_masks)
            test_dataset = C2SessionDataset(test_seq, test_masks, test_labels)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            test_loader = None
    else:
        # Data loaders are intentionally built with natural class priors (original behavior).
        train_loader, val_loader, test_loader, normalizer, transform_config = create_dataloaders(
            npz_path,
            batch_size,
            min_flows,
            normalize=normalize_features,
            transform_config=transform_config,
            expected_feature_names=resolved_feature_names,
            expected_session_len=resolved_session_len,
        )

        # Optionally replace train loader with a balanced sampler
        if balanced_sampling:
            label_tensor = train_loader.dataset.labels
            if not isinstance(label_tensor, torch.Tensor):
                label_tensor = torch.tensor(label_tensor, dtype=torch.long)
            class_counts = torch.bincount(label_tensor)
            class_weights = 1.0 / (class_counts.float() + 1e-6)
            sample_weights = class_weights[label_tensor]
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
            train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    replay_loader = None
    teacher_model = None
    if replay_npz_path:
        print(f"[INFO] Loading replay data from {replay_npz_path}...")
        replay_dataset = _load_npz_dataset(
            replay_npz_path,
            resolved_feature_names,
            resolved_session_len,
            schema_version,
            transform_config,
            normalize_features,
            normalizer,
        )
        replay_loader = torch.utils.data.DataLoader(replay_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        teacher_path = teacher_checkpoint_path or init_checkpoint_path
        if teacher_path:
            print(f"[INFO] Loading teacher checkpoint from {teacher_path}...")
            teacher_checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)
            teacher_model = C2Transformer(
                feature_dim=resolved_feature_dim,
                seq_len=resolved_session_len,
                use_derivative_features=use_derivative_features,
            ).to(device)
            teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"])
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False

    model = C2Transformer(
        feature_dim=resolved_feature_dim,
        seq_len=resolved_session_len,
        use_derivative_features=use_derivative_features,
    ).to(device)

    if init_checkpoint_path:
        print(f"[INFO] Loading initial weights from {init_checkpoint_path}...")
        init_checkpoint = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" not in init_checkpoint:
            raise ValueError("init checkpoint is missing model_state_dict")
        model.load_state_dict(init_checkpoint["model_state_dict"])

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
    print(f"[INFO] Schema: {schema_version} | feature_dim={resolved_feature_dim} | seq_len={resolved_session_len}")

    best_val_recall_budget = 0.0
    best_val_f1 = 0.0
    best_val_fpr_budget = float("inf")
    best_val_threshold = 0.5
    epochs_no_improve = 0

    print("\n" + "="*60)
    print("BEGINNING TRAINING LOOP")
    print("="*60)

    replay_ce = torch.nn.BCEWithLogitsLoss()

    # --- MAIN EPOCH LOOP ---
    for epoch in range(epochs):
        # -- Training Phase --
        model.train()
        train_loss = 0.0

        replay_iter = iter(replay_loader) if replay_loader is not None else None
        for batch_x, batch_mask, batch_y in train_loader:
            batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y.float())

            if replay_iter is not None and teacher_model is not None:
                try:
                    replay_batch_x, replay_batch_mask, replay_batch_y = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_batch_x, replay_batch_mask, replay_batch_y = next(replay_iter)

                replay_batch_x = replay_batch_x.to(device)
                replay_batch_mask = replay_batch_mask.to(device)
                replay_batch_y = replay_batch_y.to(device)

                replay_logits = model(replay_batch_x, replay_batch_mask)
                replay_loss = criterion(replay_logits, replay_batch_y.float())

                with torch.no_grad():
                    teacher_logits = teacher_model(replay_batch_x, replay_batch_mask)
                    teacher_probs = torch.sigmoid(teacher_logits)

                distill_loss = replay_ce(replay_logits, teacher_probs)
                loss = loss + replay_loss_weight * replay_loss + distill_weight * distill_loss

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
                'schema_version': schema_version,
                'feature_names': list(resolved_feature_names) if resolved_feature_names is not None else None,
                'feature_dim': resolved_feature_dim,
                'session_len': resolved_session_len,
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
        print(
            classification_report(
                test_targets,
                final_preds,
                labels=[0, 1],
                target_names=["Benign", "C2"],
                zero_division=0,
            )
        )

        cm = confusion_matrix(test_targets, final_preds, labels=[0, 1])
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
    parser.add_argument("--balanced", action="store_true", help="Use balanced sampling for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--init_checkpoint", default=None, help="Optional checkpoint to initialize weights from")
    parser.add_argument("--val_npz_path", default=None, help="Optional separate validation NPZ")
    parser.add_argument("--test_npz_path", default=None, help="Optional separate test NPZ for final evaluation")
    parser.add_argument("--replay_npz_path", default=None, help="Optional replay NPZ for continual learning")
    parser.add_argument("--teacher_checkpoint_path", default=None, help="Optional frozen teacher checkpoint for replay distillation")
    parser.add_argument("--replay_loss_weight", type=float, default=1.0, help="Weight for replay supervised loss")
    parser.add_argument("--distill_weight", type=float, default=0.5, help="Weight for teacher distillation loss")
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
        balanced_sampling=getattr(args, "balanced", False),
        seed=args.seed,
        init_checkpoint_path=args.init_checkpoint,
        val_npz_path=args.val_npz_path,
        test_npz_path=args.test_npz_path,
        replay_npz_path=args.replay_npz_path,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        replay_loss_weight=args.replay_loss_weight,
        distill_weight=args.distill_weight,
    )
