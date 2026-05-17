import numpy as np

benign_file = 'data/processed/benign_mix/ctu13_benign_sample_sessions.npz'
npz = np.load(benign_file, allow_pickle=True)
features = npz.get('feature_names')
if features is not None:
    print(f"Benign NPZ: {len(features)} features")
    print(f"Features: {list(features)}")
else:
    print(f"No feature_names, X shape: {npz['X'].shape}")
