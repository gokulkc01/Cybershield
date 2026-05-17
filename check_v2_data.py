#!/usr/bin/env python
"""Check v2 mixed session data structure."""
import numpy as np
from src.data_loader.npz_utils import detect_npz_schema

train_file = 'data/processed/v2_mixed_session_only/train_sessions.npz'
train_data = np.load(train_file, allow_pickle=True)

print(f'Train NPZ keys: {list(train_data.keys())}')
print(f'X shape: {train_data["X"].shape if "X" in train_data else "N/A"}')
print(f'y shape: {train_data["y"].shape if "y" in train_data else "N/A"}')

# Try to detect schema
schema = detect_npz_schema(train_file)
print(f'\nSchema:')
for k, v in schema.items():
    if k != 'feature_names':
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: {v}')
