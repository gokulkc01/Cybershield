#!/usr/bin/env python
"""Check existing CTU-13 NPZ schemas."""
from src.data_loader.npz_utils import detect_npz_schema
import os

files = [
    'data/processed/ctu13_c2_sessions.npz',
    'data/processed/ctu13_scenario1_neris.npz',
    'data/processed/ctu13_scenario2_kraken.npz',
]

for f in files:
    if os.path.exists(f):
        try:
            schema = detect_npz_schema(f)
            print(f'\n{f}:')
            print(f'  Schema Version: {schema.get("schema_version", "N/A")}')
            print(f'  Feature Dim: {schema.get("feature_dim", "N/A")}')
            print(f'  Session Len: {schema.get("session_len", "N/A")}')
            print(f'  Feature Names Count: {len(schema.get("feature_names", []))}')
            print(f'  Feature Names: {schema.get("feature_names", [])}')
        except Exception as e:
            print(f'{f}: ERROR - {e}')
    else:
        print(f'{f}: NOT FOUND')
