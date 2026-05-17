import numpy as np
npz = np.load(r"data/processed/ctu13_scenario1_neris.npz", allow_pickle=True)
print("Keys in NPZ:", list(npz.keys()))
feature_names = npz.get("feature_names")
if feature_names is not None:
    print(f"Feature names ({len(feature_names)}): {list(feature_names)}")
else:
    print("No feature_names found in NPZ")
    print(f"X shape: {npz['X'].shape}")
