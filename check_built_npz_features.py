import numpy as np

files = [
    'data/processed/ctu13_scenario1_neris.npz',
    'data/processed/ctu13_scenario2_kraken.npz',
    'data/processed/ctu13_scenario9_conficker.npz'
]

for f in files:
    npz = np.load(f, allow_pickle=True)
    features = npz.get('feature_names')
    if features is not None:
        print(f"{f.split('/')[-1]}: {len(features)} features - {list(features)}")
    else:
        print(f"{f.split('/')[-1]}: No feature_names, X shape: {npz['X'].shape}")
