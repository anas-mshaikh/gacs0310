"""
좌표계 v2.0 → k=5로 업데이트
KMeans만 교체, PCA/scaler 유지
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = "/home/jongcircle/dev/gacs0202"
PKL_PATH = os.path.join(ROOT, "research/2026-03-16_coordinate_v2/gacs_coordinate_system_v2.pkl")
COORDS_CSV = os.path.join(ROOT, "research/2026-03-16_coordinate_v2/gacs_coordinates_v2.csv")
DATASET_CSV = os.path.join(ROOT, "gacs_0202/data/embeddings/gacs_dataset.csv")
K5_PROFILES = os.path.join(ROOT, "research/2026-03-16_coordinate_v2/cluster_profiles_k5.json")

RANDOM_SEED = 42
N_STABLE = 7

# ── Load ──
with open(PKL_PATH, "rb") as f:
    cs = pickle.load(f)

coords = pd.read_csv(COORDS_CSV)
dataset = pd.read_csv(DATASET_CSV)
pc_cols = [f"PC{i+1}" for i in range(14)]
X_pca = coords[pc_cols].values

with open(K5_PROFILES) as f:
    k5_data = json.load(f)

print(f"Loaded coordinate system v{cs['metadata']['version']}")
print(f"Scenes: {len(coords)}")

# ── Refit KMeans k=5 ──
km5 = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init=10)
labels = km5.fit_predict(X_pca)
print("\nKMeans k=5 fitted")
print(f"Cluster sizes: {np.bincount(labels).tolist()}")

# ── Update pkl ──
cs["kmeans"] = km5
cs["metadata"]["best_k"] = 5
cs["metadata"]["best_silhouette"] = k5_data["silhouette_avg"]
cs["metadata"]["date"] = "2026-03-18"
cs["cluster_profiles"] = k5_data["profiles"]

with open(PKL_PATH, "wb") as f:
    pickle.dump(cs, f)
print(f"\nUpdated: {PKL_PATH}")
print(f"  k={cs['metadata']['best_k']}, sil={cs['metadata']['best_silhouette']}")

# ── Update main coordinates CSV ──
coords["cluster_label"] = labels
coords.to_csv(COORDS_CSV, index=False)
print(f"Updated: {COORDS_CSV} (cluster_label → k=5)")

# ── Update gacs_transform.py ──
transform_path = os.path.join(ROOT, "research/2026-03-12_coordinate_stabilization/gacs_transform.py")
# Also copy updated pkl path info
print("\nCoordinate system v2.0 updated to k=5")
print(f"  Axes: {cs['axis_names'][:N_STABLE]}")
print(f"  Clusters: {list(k5_data['profiles'].keys())}")
for ck, prof in k5_data["profiles"].items():
    print(f"    {ck}: {prof['name']} ({prof['n_scenes']} scenes, {prof['pct']}%)")

print("\n=== DONE ===")
