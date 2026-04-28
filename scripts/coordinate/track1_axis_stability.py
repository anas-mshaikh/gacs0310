"""
Track 1: Axis Stability Test & Versioned Coordinate System
Tests whether PCA axes are stable across data subsets and builds
a serializable coordinate system for new data.
"""

import json
import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = "/home/jongcircle/dev/gacs0202"
EMB_PATH = os.path.join(ROOT, "data/step3/emb_emotion_distil_norm.npy")
LABELS_PATH = os.path.join(ROOT, "gacs0202(kushi)/gacs_labeled_scenes.csv")
RESEARCH = os.path.join(ROOT, "research")
AXIS_INTERP = os.path.join(RESEARCH, "axis_interpretation.json")

N_COMPONENTS = 14
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Load Data ────────────────────────────────────────────────────────────
print("=" * 70)
print("TRACK 1: AXIS STABILITY & VERSIONED COORDINATE SYSTEM")
print("=" * 70)

embeddings = np.load(EMB_PATH)
df = pd.read_csv(LABELS_PATH)
print(f"\nLoaded embeddings: {embeddings.shape}")
print(f"Loaded labels: {df.shape}")

# Derive category from video_name
def get_category(vname):
    vname = vname.lower()
    if "advertisements" in vname:
        return "advertisements"
    elif "animations" in vname:
        return "animations"
    elif "emotional_shorts" in vname:
        return "emotional_shorts"
    elif "movie_trailers" in vname or "trailer" in vname:
        return "movie_trailers"
    else:
        return "unknown"

df["category"] = df["video_name"].apply(get_category)
categories = sorted(df["category"].unique())
print(f"Categories: {categories}")
for cat in categories:
    print(f"  {cat}: {(df['category'] == cat).sum()} scenes")

# ── Load axis names from previous research ───────────────────────────────
with open(AXIS_INTERP) as f:
    axis_data = json.load(f)
axis_names = []
axis_variance = []
for i in range(1, N_COMPONENTS + 1):
    key = f"PC{i}"
    if key in axis_data:
        axis_names.append(axis_data[key]["name"])
        axis_variance.append(round(axis_data[key]["variance_explained_pct"] / 100, 4))
    else:
        axis_names.append(f"PC{i}")
        axis_variance.append(0.0)

print(f"\nAxis names: {axis_names}")

# ── Full-data reference PCA ──────────────────────────────────────────────
print("\n--- Fitting full-data PCA (14 components) ---")
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(embeddings)
pca_full = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
X_pca_full = pca_full.fit_transform(X_scaled)
total_var = sum(pca_full.explained_variance_ratio_)
print(f"Total variance explained by {N_COMPONENTS} PCs: {total_var:.4f}")

# ── Helper: cosine similarity between corresponding PCs ──────────────────
def pc_cosine_similarities(pca_ref, pca_test, n_components=N_COMPONENTS):
    """Absolute cosine similarity between corresponding PC axes."""
    sims = []
    for i in range(n_components):
        v1 = pca_ref.components_[i]
        v2 = pca_test.components_[i]
        cos = np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sims.append(round(float(cos), 4))
    return sims

# ══════════════════════════════════════════════════════════════════════════
# PART A: AXIS STABILITY TESTS
# ══════════════════════════════════════════════════════════════════════════

# ── A1: Bootstrap Stability (10 iterations, 80% sample) ──────────────────
print("\n" + "=" * 70)
print("A1: BOOTSTRAP STABILITY TEST (10 iterations, 80% sample)")
print("=" * 70)

n_bootstrap = 10
n_sample = int(0.8 * len(embeddings))  # 1020
bootstrap_sims = []  # list of arrays, each (14,)

for i in range(n_bootstrap):
    idx = np.random.choice(len(embeddings), size=n_sample, replace=False)
    X_sub = embeddings[idx]
    sc = StandardScaler()
    X_sub_sc = sc.fit_transform(X_sub)
    pca_sub = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    pca_sub.fit(X_sub_sc)
    sims = pc_cosine_similarities(pca_full, pca_sub)
    bootstrap_sims.append(sims)
    print(f"  Iter {i+1:2d}: PC1={sims[0]:.4f}  PC7={sims[6]:.4f}  PC14={sims[13]:.4f}")

bootstrap_sims = np.array(bootstrap_sims)  # (10, 14)
bootstrap_mean = np.mean(bootstrap_sims, axis=0)
bootstrap_std = np.std(bootstrap_sims, axis=0)

print("\n  Bootstrap summary (mean cos sim):")
for j in range(N_COMPONENTS):
    print(f"    PC{j+1:2d}: {bootstrap_mean[j]:.4f} +/- {bootstrap_std[j]:.4f}")

# ── A2: Split-Half Test (5 random splits) ────────────────────────────────
print("\n" + "=" * 70)
print("A2: SPLIT-HALF TEST (5 random 50/50 splits)")
print("=" * 70)

n_splits = 5
split_sims = []

for i in range(n_splits):
    perm = np.random.permutation(len(embeddings))
    half = len(embeddings) // 2
    idx_a, idx_b = perm[:half], perm[half:half*2]

    sc_a = StandardScaler()
    X_a = sc_a.fit_transform(embeddings[idx_a])
    pca_a = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    pca_a.fit(X_a)

    sc_b = StandardScaler()
    X_b = sc_b.fit_transform(embeddings[idx_b])
    pca_b = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    pca_b.fit(X_b)

    # Compare halves to each other
    sims = pc_cosine_similarities(pca_a, pca_b)
    split_sims.append(sims)
    print(f"  Split {i+1}: PC1={sims[0]:.4f}  PC7={sims[6]:.4f}  PC14={sims[13]:.4f}")

split_sims = np.array(split_sims)  # (5, 14)
split_mean = np.mean(split_sims, axis=0)
split_std = np.std(split_sims, axis=0)

print("\n  Split-half summary (mean cos sim):")
for j in range(N_COMPONENTS):
    print(f"    PC{j+1:2d}: {split_mean[j]:.4f} +/- {split_std[j]:.4f}")

# ── A3: Category-Dropout Test ────────────────────────────────────────────
print("\n" + "=" * 70)
print("A3: CATEGORY-DROPOUT TEST")
print("=" * 70)

dropout_categories = ["movie_trailers", "advertisements", "emotional_shorts", "animations"]
category_dropout_results = {}

for drop_cat in dropout_categories:
    mask = df["category"] != drop_cat
    X_sub = embeddings[mask.values]
    n_dropped = (~mask).sum()
    sc = StandardScaler()
    X_sub_sc = sc.fit_transform(X_sub)
    pca_sub = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    pca_sub.fit(X_sub_sc)
    sims = pc_cosine_similarities(pca_full, pca_sub)
    category_dropout_results[drop_cat] = sims
    print(f"  Drop '{drop_cat}' ({n_dropped} scenes removed, {X_sub.shape[0]} remain):")
    print(f"    PC1={sims[0]:.4f}  PC7={sims[6]:.4f}  PC14={sims[13]:.4f}")

# Category dropout mean per PC
cat_dropout_arr = np.array(list(category_dropout_results.values()))  # (4, 14)
cat_dropout_mean = np.mean(cat_dropout_arr, axis=0)
cat_dropout_std = np.std(cat_dropout_arr, axis=0)

print("\n  Category-dropout summary (mean cos sim across all dropouts):")
for j in range(N_COMPONENTS):
    print(f"    PC{j+1:2d}: {cat_dropout_mean[j]:.4f} +/- {cat_dropout_std[j]:.4f}")

# ── A4: Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STABILITY SUMMARY")
print("=" * 70)

stability_results = {
    "bootstrap": {},
    "split_half": {},
    "category_dropout": {},
    "summary": []
}

for j in range(N_COMPONENTS):
    pc_key = f"PC{j+1}"

    # Bootstrap
    stability_results["bootstrap"][pc_key] = {
        "mean_cos": round(float(bootstrap_mean[j]), 4),
        "std": round(float(bootstrap_std[j]), 4),
        "stable": bool(bootstrap_mean[j] > 0.9)
    }

    # Split-half
    stability_results["split_half"][pc_key] = {
        "mean_cos": round(float(split_mean[j]), 4),
        "std": round(float(split_std[j]), 4),
        "stable": bool(split_mean[j] > 0.9)
    }

    # Category dropout (per dropped category)
    stability_results["category_dropout"][pc_key] = {}
    for drop_cat in dropout_categories:
        stability_results["category_dropout"][pc_key][drop_cat] = round(
            float(category_dropout_results[drop_cat][j]), 4
        )

    # Overall: average across all three test types
    overall_mean = round(float(np.mean([bootstrap_mean[j], split_mean[j], cat_dropout_mean[j]])), 4)
    if overall_mean > 0.9:
        status = "STABLE"
    elif overall_mean < 0.7:
        status = "UNSTABLE"
    else:
        status = "MODERATE"

    stability_results["summary"].append({
        "pc": pc_key,
        "name": axis_names[j],
        "bootstrap_mean": round(float(bootstrap_mean[j]), 4),
        "split_half_mean": round(float(split_mean[j]), 4),
        "cat_dropout_mean": round(float(cat_dropout_mean[j]), 4),
        "overall_mean": overall_mean,
        "status": status
    })

    print(f"  {pc_key:4s} ({axis_names[j]:25s}): "
          f"boot={bootstrap_mean[j]:.4f}  split={split_mean[j]:.4f}  "
          f"cat={cat_dropout_mean[j]:.4f}  => {status}")

# Count stable/unstable
n_stable = sum(1 for s in stability_results["summary"] if s["status"] == "STABLE")
n_moderate = sum(1 for s in stability_results["summary"] if s["status"] == "MODERATE")
n_unstable = sum(1 for s in stability_results["summary"] if s["status"] == "UNSTABLE")
print(f"\n  STABLE (>0.9): {n_stable} PCs")
print(f"  MODERATE (0.7-0.9): {n_moderate} PCs")
print(f"  UNSTABLE (<0.7): {n_unstable} PCs")

# Save JSON
json_path = os.path.join(RESEARCH, "axis_stability.json")
with open(json_path, "w") as f:
    json.dump(stability_results, f, indent=2)
print(f"\nSaved: {json_path}")

# ── A5: Heatmap Visualization ────────────────────────────────────────────
print("\n--- Generating axis_stability.png ---")

fig, ax = plt.subplots(figsize=(10, 8))

# Build heatmap data: rows=PCs, cols=test types
# Columns: bootstrap_mean, split_half_mean, cat_dropout per category (4), cat_dropout_mean
col_labels = ["Bootstrap", "Split-Half"] + [f"Drop {c}" for c in dropout_categories]
n_cols = len(col_labels)
heatmap_data = np.zeros((N_COMPONENTS, n_cols))

for j in range(N_COMPONENTS):
    heatmap_data[j, 0] = bootstrap_mean[j]
    heatmap_data[j, 1] = split_mean[j]
    for k, cat in enumerate(dropout_categories):
        heatmap_data[j, 2 + k] = category_dropout_results[cat][j]

row_labels = [f"PC{j+1}: {axis_names[j]}" for j in range(N_COMPONENTS)]

cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect="auto")

# Annotate cells
for i in range(N_COMPONENTS):
    for j_col in range(n_cols):
        val = heatmap_data[i, j_col]
        color = "white" if val < 0.5 else "black"
        ax.text(j_col, i, f"{val:.3f}", ha="center", va="center",
                fontsize=7, color=color, fontweight="bold")

ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(N_COMPONENTS))
ax.set_yticklabels(row_labels, fontsize=9)
ax.set_title("PCA Axis Stability: Cosine Similarity Across Tests", fontsize=13, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Absolute Cosine Similarity", fontsize=10)

plt.tight_layout()
png_path = os.path.join(RESEARCH, "axis_stability.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {png_path}")

# ══════════════════════════════════════════════════════════════════════════
# PART B: VERSIONED COORDINATE SYSTEM
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART B: VERSIONED COORDINATE SYSTEM")
print("=" * 70)

# ── B1: Fit KMeans (k=4 matching previous research) ─────────────────────
print("\n--- Fitting KMeans (k=4) ---")
kmeans = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca_full)
print(f"  Cluster sizes: {np.bincount(cluster_labels).tolist()}")

# ── B2: Save coordinate system pkl ──────────────────────────────────────
coordinate_system = {
    "scaler": scaler_full,
    "pca": pca_full,
    "axis_names": axis_names,
    "axis_variance": axis_variance,
    "kmeans": kmeans,
    "metadata": {
        "version": "1.0",
        "n_scenes_trained": 1275,
        "embedding_model": "j-hartmann/emotion-english-distilroberta-base",
        "date": "2026-03-12"
    }
}

pkl_path = os.path.join(RESEARCH, "gacs_coordinate_system_v1.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(coordinate_system, f)
print(f"  Saved: {pkl_path}")
print(f"  Size: {os.path.getsize(pkl_path) / 1024:.1f} KB")

# ── B3: Write standalone transform module ────────────────────────────────
transform_code = '''"""
GACS Coordinate System Transform (v1.0)
Standalone module for transforming new scene embeddings into GACS coordinates.

Usage:
    from gacs_transform import load_coordinate_system, transform_new_scenes

    cs = load_coordinate_system("research/gacs_coordinate_system_v1.pkl")
    results = transform_new_scenes(cs, new_embeddings_768d)
    # results["coordinates"]   -> (N, 14) array
    # results["cluster_labels"] -> (N,) array
    # results["axis_names"]    -> list of 14 axis names
"""

import pickle
import numpy as np


def load_coordinate_system(pkl_path: str) -> dict:
    """Load a saved GACS coordinate system from a pickle file."""
    with open(pkl_path, "rb") as f:
        cs = pickle.load(f)
    print(f"Loaded GACS coordinate system v{cs['metadata']['version']}")
    print(f"  Trained on {cs['metadata']['n_scenes_trained']} scenes")
    print(f"  Embedding model: {cs['metadata']['embedding_model']}")
    print(f"  Date: {cs['metadata']['date']}")
    return cs


def transform_new_scenes(cs: dict, embeddings_768d: np.ndarray) -> dict:
    """
    Transform new scene embeddings into GACS coordinates.

    Parameters
    ----------
    cs : dict
        Loaded coordinate system (from load_coordinate_system).
    embeddings_768d : np.ndarray
        Shape (N, 768) array of emotion-distilroberta embeddings.

    Returns
    -------
    dict with keys:
        coordinates : np.ndarray, shape (N, 14)
        cluster_labels : np.ndarray, shape (N,)
        axis_names : list of str
        axis_variance : list of float
    """
    if embeddings_768d.ndim == 1:
        embeddings_768d = embeddings_768d.reshape(1, -1)
    assert embeddings_768d.shape[1] == 768, (
        f"Expected 768-dim embeddings, got {embeddings_768d.shape[1]}"
    )

    # Apply saved scaler -> PCA
    X_scaled = cs["scaler"].transform(embeddings_768d)
    coordinates = cs["pca"].transform(X_scaled)

    # Predict cluster labels
    cluster_labels = cs["kmeans"].predict(coordinates)

    return {
        "coordinates": coordinates,
        "cluster_labels": cluster_labels,
        "axis_names": cs["axis_names"],
        "axis_variance": cs["axis_variance"],
    }


if __name__ == "__main__":
    import sys
    # Quick test
    cs = load_coordinate_system(
        sys.argv[1] if len(sys.argv) > 1
        else "research/gacs_coordinate_system_v1.pkl"
    )
    # Generate a random test embedding
    rng = np.random.RandomState(0)
    test_emb = rng.randn(3, 768).astype(np.float32)
    result = transform_new_scenes(cs, test_emb)
    print(f"\\nTest: {test_emb.shape[0]} scenes -> "
          f"coordinates {result['coordinates'].shape}, "
          f"clusters {result['cluster_labels']}")
'''

transform_path = os.path.join(RESEARCH, "gacs_transform.py")
with open(transform_path, "w") as f:
    f.write(transform_code)
print(f"  Saved: {transform_path}")

# ── B4: Verification Test ────────────────────────────────────────────────
print("\n--- Verification: load pkl and transform 10 hold-out scenes ---")

with open(pkl_path, "rb") as f:
    cs_loaded = pickle.load(f)

# Pick 10 random scenes
test_idx = np.random.choice(len(embeddings), size=10, replace=False)
test_emb = embeddings[test_idx]

# Transform using loaded system
X_test_sc = cs_loaded["scaler"].transform(test_emb)
coords_loaded = cs_loaded["pca"].transform(X_test_sc)
clusters_loaded = cs_loaded["kmeans"].predict(coords_loaded)

# Compare to full-data computation
coords_full = X_pca_full[test_idx]
clusters_full = cluster_labels[test_idx]

# Re-transform the same scenes through the full pipeline for apples-to-apples comparison
coords_verify = cs_loaded["pca"].transform(cs_loaded["scaler"].transform(embeddings[test_idx]))
max_coord_diff = np.max(np.abs(coords_loaded - coords_verify))
clusters_match = np.all(clusters_loaded == clusters_full)

# Also compare to the batch-computed coordinates (may differ slightly due to float32->64)
max_diff_vs_batch = np.max(np.abs(coords_loaded - coords_full))

print(f"  Test scenes indices: {test_idx.tolist()}")
print(f"  Max coordinate diff (reload vs re-transform): {max_coord_diff:.2e}")
print(f"  Max coordinate diff (reload vs batch fit_transform): {max_diff_vs_batch:.2e}")
print(f"  Coordinates match (within tol 1e-10): {max_coord_diff < 1e-10}")
print(f"  Cluster labels match: {clusters_match}")

assert max_coord_diff < 1e-10, f"Coordinate mismatch! max diff = {max_coord_diff}"
assert clusters_match, "Cluster label mismatch!"
print("  VERIFICATION PASSED")

# ── Final Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\nOutputs saved to {RESEARCH}/:")
print("  - axis_stability.json       (stability test results)")
print("  - axis_stability.png        (heatmap visualization)")
print("  - gacs_coordinate_system_v1.pkl  (versioned coordinate system)")
print("  - gacs_transform.py         (standalone transform module)")
print("\nAxis Stability:")
for s in stability_results["summary"]:
    print(f"  {s['pc']:4s} {s['name']:25s}  overall={s['overall_mean']:.4f}  [{s['status']}]")
print(f"\n  {n_stable} STABLE / {n_moderate} MODERATE / {n_unstable} UNSTABLE out of {N_COMPONENTS} PCs")
print("\nCoordinate System v1.0:")
print("  Scaler: StandardScaler (768 features)")
print(f"  PCA: {N_COMPONENTS} components, {total_var:.4f} variance explained")
print("  KMeans: k=4")
print("  Verification: PASSED")
print("\nDone.")
