"""PyTorch Dataset and loaders for CyberShield sequence classification."""

import torch
from torch.utils.data import Dataset, DataLoader

from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.normalization import fit_feature_normalizer
from src.data_loader.split_utils import create_session_splits

class C2SessionDataset(Dataset):
    def __init__(self, sequences, masks, labels):
        """
        sequences: (N, 20, 12) float32 array
        masks:     (N, 20) boolean array (True if real flow, False if padding)
        labels:    (N,) int64 array
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        # PyTorch attention expects True for tokens that should be ignored.
        self.padding_masks = torch.tensor(~masks, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.padding_masks[idx], self.labels[idx]

def create_dataloaders(
    npz_path: str,
    batch_size: int = 64,
    min_flows: int = 5,
    normalize: bool = True,
    transform_config: FeatureTransformConfig | None = None,
    expected_feature_names: tuple[str, ...] | None = None,
    expected_session_len: int | None = None,
):
    """
    Loads the NPZ, applies the min_flows filter, splits the data, 
    and returns PyTorch DataLoaders.
    """
    print(f"[INFO] Loading PyTorch Dataset from {npz_path}...")
    splits = create_session_splits(
        npz_path,
        min_flows=min_flows,
        expected_feature_names=expected_feature_names,
        expected_session_len=expected_session_len,
    )
    print(f"[INFO] Total valid sessions (>= {min_flows} flows): {len(splits.y_train) + len(splits.y_val) + len(splits.y_test)}")

    if transform_config is None:
        transform_config = FeatureTransformConfig()
    normalizer = None
    x_train, x_val, x_test = splits.x_train, splits.x_val, splits.x_test
    x_train = apply_feature_transforms(x_train, splits.m_train, transform_config)
    x_val = apply_feature_transforms(x_val, splits.m_val, transform_config)
    x_test = apply_feature_transforms(x_test, splits.m_test, transform_config)
    print(
        f"[INFO] Applied feature transforms: "
        f"log_scale={list(transform_config.log_scale_features)} | "
        f"ablate={list(transform_config.ablate_features)}"
    )
    if normalize:
        normalizer = fit_feature_normalizer(x_train, splits.m_train)
        x_train = normalizer.transform(x_train, splits.m_train)
        x_val = normalizer.transform(x_val, splits.m_val)
        x_test = normalizer.transform(x_test, splits.m_test)
        print("[INFO] Applied train-only feature normalization.")

    # Create Datasets
    train_dataset = C2SessionDataset(x_train, splits.m_train, splits.y_train)
    val_dataset = C2SessionDataset(x_val, splits.m_val, splits.y_val)
    test_dataset = C2SessionDataset(x_test, splits.m_test, splits.y_test)

    # Train on natural priors (no oversampling) so the model learns true base rates.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_c2_ratio = float(splits.y_train.mean()) if len(splits.y_train) > 0 else 0.0
    print(f"[INFO] Train class prior: C2={train_c2_ratio:.4%} | Benign={1.0-train_c2_ratio:.4%}")
    print(f"[INFO] Dataloaders Ready: Train={len(train_loader)} batches | Val={len(val_loader)} batches | Test={len(test_loader)} batches")
    return train_loader, val_loader, test_loader, normalizer, transform_config
