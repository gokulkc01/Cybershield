"""
torch_dataset.py
================
PyTorch Dataset class for the CyberShield Transformer.
Loads the unflattened (N, 20, 12) sequences directly from the .npz file.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class C2SessionDataset(Dataset):
    def __init__(self, sequences, masks, labels):
        """
        sequences: (N, 20, 12) float32 array
        masks:     (N, 20) boolean array (True if real flow, False if padding)
        labels:    (N,) int64 array
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        # PyTorch attention mechanisms usually expect the mask to be boolean where 
        # True means "DO NOT attend to this" (padding), so we invert our mask.
        self.padding_masks = torch.tensor(~masks, dtype=torch.bool) 
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.padding_masks[idx], self.labels[idx]

def create_dataloaders(npz_path: str, batch_size: int = 64, min_flows: int = 5):
    """
    Loads the NPZ, applies the min_flows filter, splits the data, 
    and returns PyTorch DataLoaders.
    """
    print(f"[INFO] Loading PyTorch Dataset from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    
    if "masks" in data:
        masks = data["masks"].astype(bool)
    else:
        masks = (X.sum(axis=2) != 0)

    # Apply min_flows filter
    flow_counts = masks.sum(axis=1)
    keep_idx = flow_counts >= min_flows
    X, y, masks = X[keep_idx], y[keep_idx], masks[keep_idx]
    
    print(f"[INFO] Total valid sessions (>= {min_flows} flows): {len(y)}")

    # Split: 70% Train, 15% Val, 15% Test
    X_temp, X_test, y_temp, y_test, m_temp, m_test = train_test_split(
        X, y, masks, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
        X_temp, y_temp, m_temp, test_size=0.1765, stratify=y_temp, random_state=42 # 0.15 / 0.85 approx
    )

    # Create Datasets
    train_dataset = C2SessionDataset(X_train, m_train, y_train)
    val_dataset = C2SessionDataset(X_val, m_val, y_val)
    test_dataset = C2SessionDataset(X_test, m_test, y_test)

    # Train on natural priors (no oversampling) so the model learns true base rates.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_c2_ratio = float(y_train.mean()) if len(y_train) > 0 else 0.0
    print(f"[INFO] Train class prior: C2={train_c2_ratio:.4%} | Benign={1.0-train_c2_ratio:.4%}")
    print(f"[INFO] Dataloaders Ready: Train={len(train_loader)} batches | Val={len(val_loader)} batches | Test={len(test_loader)} batches")
    return train_loader, val_loader, test_loader