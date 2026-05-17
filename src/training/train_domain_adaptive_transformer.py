"""Train a shared-backbone, domain-specific-head transformer across multiple domains.

This script is intended for continual or multi-task learning where the goal is
to keep performance on a mixed-split domain while also adapting to UWF.
It trains one shared backbone with per-domain heads and calibration layers.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score

from src.data_loader.extended_feature_transforms import (
    ExtendedFeatureTransformConfig,
    apply_extended_feature_transforms,
)
from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.npz_utils import detect_npz_schema, load_session_npz
from src.data_loader.normalization import fit_feature_normalizer
from src.features.feature_config import FOCAL_ALPHA, FOCAL_GAMMA, MAX_FPR_BUDGET, SESSION_LEN, FEATURE_DIM
from src.features.feature_config_extended import FEATURE_NAMES_EXTENDED, FEATURE_SCHEMA_VERSION as EXTENDED_SCHEMA_VERSION
from src.evaluation.operating_point import find_threshold_under_fpr_budget
from src.models.domain_adaptive_transformer import DomainAdaptiveC2Transformer


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


class DomainSessionDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: np.ndarray, masks: np.ndarray, labels: np.ndarray, domain_id: int):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.padding_masks = torch.tensor(~masks, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.domain_ids = torch.full((len(labels),), int(domain_id), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.padding_masks[idx], self.labels[idx], self.domain_ids[idx]


def _load_domain_split(
    npz_path: str,
    domain_id: int,
    resolved_feature_names: tuple[str, ...] | None,
    resolved_session_len: int,
    schema_version: str,
    transform_config: FeatureTransformConfig | ExtendedFeatureTransformConfig,
    normalize: bool,
    normalizer,
) -> DomainSessionDataset:
    sequences, labels, masks = load_session_npz(
        npz_path,
        expected_feature_names=resolved_feature_names,
        expected_session_len=resolved_session_len,
    )
    sequences = _apply_transforms_for_schema(sequences, masks, schema_version, transform_config, resolved_feature_names)
    if normalize:
        sequences = normalizer.transform(sequences, masks)
    return DomainSessionDataset(sequences, masks, labels, domain_id=domain_id)


def _evaluate_domain(
    model: DomainAdaptiveC2Transformer,
    device: torch.device,
    dataset: DomainSessionDataset,
    batch_size: int,
    max_fpr_budget: float,
) -> dict:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_mask, batch_y, batch_domain in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_domain = batch_domain.to(device)
            logits = model(batch_x, batch_mask, batch_domain)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(batch_y.numpy())

    probs = np.asarray(probs)
    targets = np.asarray(targets)
    threshold, recall, fpr, _ = find_threshold_under_fpr_budget(targets, probs, max_fpr=max_fpr_budget)
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(targets, preds, zero_division=0)
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.0
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "fpr": float(fpr),
        "auc": float(auc),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_samples": int(len(targets)),
    }


def train_domain_adaptive_model(
    mixed_train_npz: str,
    mixed_val_npz: str,
    mixed_test_npz: str,
    uwf_train_npz: str,
    uwf_val_npz: str,
    uwf_test_npz: str,
    model_save_dir: str,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-4,
    patience: int = 5,
    min_flows: int = 1,
    max_fpr_budget: float = MAX_FPR_BUDGET,
    normalize_features: bool = True,
    log_scale_features: tuple[str, ...] | None = None,
    ablate_features: tuple[str, ...] = (),
    use_derivative_features: bool = False,
    seed: int = 42,
    init_checkpoint_path: str | None = None,
    replay_weight: float = 1.0,
):
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_transformer.pth")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Initializing training on device: {device}")

    mixed_schema = detect_npz_schema(mixed_train_npz)
    resolved_feature_names = tuple(mixed_schema["feature_names"]) if mixed_schema["feature_names"] else None
    resolved_feature_dim = len(resolved_feature_names) if resolved_feature_names is not None else int(mixed_schema["feature_dim"] or FEATURE_DIM)
    resolved_session_len = int(mixed_schema["session_len"] or SESSION_LEN)
    schema_version = str(mixed_schema["schema_version"])

    transform_config = _coerce_transform_config(
        schema_version,
        FeatureTransformConfig(
            log_scale_features=tuple(log_scale_features) if log_scale_features is not None else FeatureTransformConfig().log_scale_features,
            ablate_features=tuple(ablate_features),
        ),
    )

    mixed_train_seq, mixed_train_labels, mixed_train_masks = load_session_npz(
        mixed_train_npz,
        expected_feature_names=resolved_feature_names,
        expected_session_len=resolved_session_len,
    )
    mixed_val_seq, mixed_val_labels, mixed_val_masks = load_session_npz(
        mixed_val_npz,
        expected_feature_names=resolved_feature_names,
        expected_session_len=resolved_session_len,
    )
    mixed_test_seq, mixed_test_labels, mixed_test_masks = load_session_npz(
        mixed_test_npz,
        expected_feature_names=resolved_feature_names,
        expected_session_len=resolved_session_len,
    )
    uwf_train_seq, uwf_train_labels, uwf_train_masks = load_session_npz(
        uwf_train_npz,
        expected_feature_names=resolved_feature_names,
        expected_session_len=resolved_session_len,
    )
    uwf_val_seq, uwf_val_labels, uwf_val_masks = load_session_npz(
        uwf_val_npz,
        expected_feature_names=resolved_feature_names,
        expected_session_len=resolved_session_len,
    )
    uwf_test_seq, uwf_test_labels, uwf_test_masks = load_session_npz(
        uwf_test_npz,
        expected_feature_names=resolved_feature_names,
        expected_session_len=resolved_session_len,
    )

    mixed_train_seq = _apply_transforms_for_schema(mixed_train_seq, mixed_train_masks, schema_version, transform_config, resolved_feature_names)
    mixed_val_seq = _apply_transforms_for_schema(mixed_val_seq, mixed_val_masks, schema_version, transform_config, resolved_feature_names)
    mixed_test_seq = _apply_transforms_for_schema(mixed_test_seq, mixed_test_masks, schema_version, transform_config, resolved_feature_names)
    uwf_train_seq = _apply_transforms_for_schema(uwf_train_seq, uwf_train_masks, schema_version, transform_config, resolved_feature_names)
    uwf_val_seq = _apply_transforms_for_schema(uwf_val_seq, uwf_val_masks, schema_version, transform_config, resolved_feature_names)
    uwf_test_seq = _apply_transforms_for_schema(uwf_test_seq, uwf_test_masks, schema_version, transform_config, resolved_feature_names)

    if normalize_features:
        normalizer = fit_feature_normalizer(np.concatenate([mixed_train_seq, uwf_train_seq], axis=0), np.concatenate([mixed_train_masks, uwf_train_masks], axis=0))
        mixed_train_seq = normalizer.transform(mixed_train_seq, mixed_train_masks)
        mixed_val_seq = normalizer.transform(mixed_val_seq, mixed_val_masks)
        mixed_test_seq = normalizer.transform(mixed_test_seq, mixed_test_masks)
        uwf_train_seq = normalizer.transform(uwf_train_seq, uwf_train_masks)
        uwf_val_seq = normalizer.transform(uwf_val_seq, uwf_val_masks)
        uwf_test_seq = normalizer.transform(uwf_test_seq, uwf_test_masks)
    else:
        normalizer = None

    mixed_train_ds = DomainSessionDataset(mixed_train_seq, mixed_train_masks, mixed_train_labels, domain_id=0)
    mixed_val_ds = DomainSessionDataset(mixed_val_seq, mixed_val_masks, mixed_val_labels, domain_id=0)
    mixed_test_ds = DomainSessionDataset(mixed_test_seq, mixed_test_masks, mixed_test_labels, domain_id=0)
    uwf_train_ds = DomainSessionDataset(uwf_train_seq, uwf_train_masks, uwf_train_labels, domain_id=1)
    uwf_val_ds = DomainSessionDataset(uwf_val_seq, uwf_val_masks, uwf_val_labels, domain_id=1)
    uwf_test_ds = DomainSessionDataset(uwf_test_seq, uwf_test_masks, uwf_test_labels, domain_id=1)

    combined_train = torch.utils.data.ConcatDataset([mixed_train_ds, uwf_train_ds])
    combined_labels = torch.cat([mixed_train_ds.labels, uwf_train_ds.labels])
    combined_domains = torch.cat([mixed_train_ds.domain_ids, uwf_train_ds.domain_ids])
    sample_keys = combined_domains * 2 + combined_labels
    key_counts = torch.bincount(sample_keys, minlength=4).float()
    sample_weights = (1.0 / (key_counts[sample_keys] + 1e-6)).tolist()
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = torch.utils.data.DataLoader(combined_train, batch_size=batch_size, sampler=sampler, drop_last=True)

    model = DomainAdaptiveC2Transformer(
        domains=["mixed", "uwf"],
        feature_dim=resolved_feature_dim,
        seq_len=resolved_session_len,
        use_derivative_features=use_derivative_features,
    ).to(device)

    if init_checkpoint_path:
        print(f"[INFO] Loading initialization checkpoint from {init_checkpoint_path}...")
        checkpoint = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        backbone_state = {key.replace("backbone.", ""): value for key, value in checkpoint["model_state_dict"].items() if key.startswith(("input_projection", "input_norm", "input_dropout", "cls_token", "pos_encoder", "transformer_encoder", "final_norm"))}
        model.backbone.load_state_dict(backbone_state, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_objective = -float("inf")
    epochs_no_improve = 0

    print("\n" + "=" * 60)
    print("BEGINNING DOMAIN-ADAPTIVE TRAINING LOOP")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_mask, batch_y, batch_domain in train_loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            batch_domain = batch_domain.to(device)

            optimizer.zero_grad()
            logits = model(batch_x, batch_mask, batch_domain)
            loss = criterion(logits, batch_y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        mixed_val_metrics = _evaluate_domain(model, device, mixed_val_ds, batch_size, max_fpr_budget)
        uwf_val_metrics = _evaluate_domain(model, device, uwf_val_ds, batch_size, max_fpr_budget)
        objective = (mixed_val_metrics["f1"] + uwf_val_metrics["f1"]) / 2.0

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {total_loss / len(train_loader):.4f} | "
            f"Mixed Val F1: {mixed_val_metrics['f1']:.4f} | UWF Val F1: {uwf_val_metrics['f1']:.4f} | "
            f"Objective: {objective:.4f}"
        )

        if objective > best_objective:
            best_objective = objective
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "domains": model.domains,
                    "feature_names": list(resolved_feature_names) if resolved_feature_names is not None else None,
                    "feature_dim": resolved_feature_dim,
                    "session_len": resolved_session_len,
                    "schema_version": schema_version,
                    "normalize_features": normalize_features,
                    "feature_normalizer": normalizer.to_checkpoint_dict() if normalizer is not None else None,
                    "feature_transform_config": transform_config.to_checkpoint_dict(),
                    "mixed_val_threshold": mixed_val_metrics["threshold"],
                    "uwf_val_threshold": uwf_val_metrics["threshold"],
                },
                best_model_path,
            )
            print(f"  --> Model improved! Saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[INFO] Early stopping triggered after {epoch+1} epochs.")
                break

    print("\n" + "=" * 60)
    print("FINAL DOMAIN EVALUATION")
    print("=" * 60)
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    mixed_test_metrics = _evaluate_domain(model, device, mixed_test_ds, batch_size, max_fpr_budget)
    uwf_test_metrics = _evaluate_domain(model, device, uwf_test_ds, batch_size, max_fpr_budget)

    print("Mixed test:")
    print(f"  AUC: {mixed_test_metrics['auc']:.4f} | F1: {mixed_test_metrics['f1']:.4f} | FPR: {mixed_test_metrics['fpr']:.4f} | Recall: {mixed_test_metrics['recall']:.4f}")
    print("UWF test:")
    print(f"  AUC: {uwf_test_metrics['auc']:.4f} | F1: {uwf_test_metrics['f1']:.4f} | FPR: {uwf_test_metrics['fpr']:.4f} | Recall: {uwf_test_metrics['recall']:.4f}")

    return {
        "best_model_path": best_model_path,
        "mixed_test": mixed_test_metrics,
        "uwf_test": uwf_test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a domain-adaptive transformer with shared backbone and domain heads")
    parser.add_argument("--mixed_train_npz", default="data/processed/extended_mixed_real_corrected/train_sessions.npz")
    parser.add_argument("--mixed_val_npz", default="data/processed/extended_mixed_real_corrected/val_sessions.npz")
    parser.add_argument("--mixed_test_npz", default="data/processed/extended_mixed_real_corrected/test_sessions.npz")
    parser.add_argument("--uwf_train_npz", default="data/processed/uwf_adaptation_split/train_sessions.npz")
    parser.add_argument("--uwf_val_npz", default="data/processed/uwf_adaptation_split/val_sessions.npz")
    parser.add_argument("--uwf_test_npz", default="data/processed/uwf_adaptation_split/test_sessions.npz")
    parser.add_argument("--model_save_dir", default="experiments/domain_adaptive_transformer")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_flows", type=int, default=1)
    parser.add_argument("--max_fpr_budget", type=float, default=MAX_FPR_BUDGET)
    parser.add_argument("--use_derivative_features", action="store_true")
    parser.add_argument("--normalize_features", action="store_true")
    parser.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    parser.set_defaults(normalize_features=True)
    parser.add_argument("--log_scale_features", default=None)
    parser.add_argument("--ablate_features", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_checkpoint", default=None)
    args = parser.parse_args()

    train_domain_adaptive_model(
        mixed_train_npz=args.mixed_train_npz,
        mixed_val_npz=args.mixed_val_npz,
        mixed_test_npz=args.mixed_test_npz,
        uwf_train_npz=args.uwf_train_npz,
        uwf_val_npz=args.uwf_val_npz,
        uwf_test_npz=args.uwf_test_npz,
        model_save_dir=args.model_save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        min_flows=args.min_flows,
        max_fpr_budget=args.max_fpr_budget,
        normalize_features=args.normalize_features,
        log_scale_features=_parse_feature_list(args.log_scale_features),
        ablate_features=_parse_feature_list(args.ablate_features),
        use_derivative_features=args.use_derivative_features,
        seed=args.seed,
        init_checkpoint_path=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
