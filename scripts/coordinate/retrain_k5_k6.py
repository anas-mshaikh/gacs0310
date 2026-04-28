"""
K=5, K=6 클러스터 학습 + 프로파일링 비교
기존 v2 임베딩/PCA 재사용, KMeans만 재학습
"""
import json
import os
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

warnings.filterwarnings("ignore")

ROOT = "/home/jongcircle/dev/gacs0202"
COORDS_CSV = os.path.join(ROOT, "research/2026-03-16_coordinate_v2/gacs_coordinates_v2.csv")
DATASET_CSV = os.path.join(ROOT, "gacs_0202/data/embeddings/gacs_dataset.csv")
PKL_PATH = os.path.join(ROOT, "research/2026-03-16_coordinate_v2/gacs_coordinate_system_v2.pkl")
OUT = os.path.join(ROOT, "research/2026-03-16_coordinate_v2")

RANDOM_SEED = 42
N_STABLE = 7

# ── Load ──
coords = pd.read_csv(COORDS_CSV)
dataset = pd.read_csv(DATASET_CSV)
with open(PKL_PATH, "rb") as f:
    cs = pickle.load(f)

pc_cols = [f"PC{i+1}" for i in range(14)]
X_pca = coords[pc_cols].values
print(f"Loaded: {len(coords)} scenes, {X_pca.shape[1]} PCs")

# ── Mood labels ──
mood_cols = [f"mood_{i}" for i in range(1, 6)]
scene_moods = []
all_mood_counts = {}
for _, row in dataset.iterrows():
    moods = set()
    for col in mood_cols:
        if pd.notna(row.get(col)) and str(row[col]).strip():
            m = str(row[col]).strip().lower()
            moods.add(m)
            all_mood_counts[m] = all_mood_counts.get(m, 0) + 1
    scene_moods.append(moods)

valid_moods = sorted([m for m, c in all_mood_counts.items() if c >= 20])

# ── Train k=5 and k=6 ──
results = {}
for k in [3, 4, 5, 6]:
    print(f"\n{'='*60}")
    print(f"K = {k}")
    print(f"{'='*60}")

    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_pca)
    sil_avg = silhouette_score(X_pca, labels)
    sil_per_sample = silhouette_samples(X_pca, labels)

    print(f"  Silhouette (avg): {sil_avg:.4f}")
    print(f"  Cluster sizes: {np.bincount(labels).tolist()}")

    # Per-cluster silhouette
    cluster_sils = {}
    for c in range(k):
        mask = labels == c
        cluster_sils[c] = float(np.mean(sil_per_sample[mask]))

    # Cluster profiles
    profiles = {}
    for c in range(k):
        mask = labels == c
        n = int(mask.sum())
        cl_scenes = [scene_moods[i] for i in range(len(mask)) if mask[i]]

        # Top moods
        mood_freq = {}
        for s in cl_scenes:
            for m in s:
                if m in valid_moods:
                    mood_freq[m] = mood_freq.get(m, 0) + 1
        top_moods = sorted(mood_freq.items(), key=lambda x: x[1], reverse=True)[:8]

        # Mean PCs (stable only)
        mean_pcs = {}
        for i in range(N_STABLE):
            mean_pcs[f"PC{i+1}"] = round(float(X_pca[mask, i].mean()), 2)

        # Category distribution
        cat_dist = {}
        for i in range(len(mask)):
            if mask[i]:
                cat = coords.iloc[i]["category"]
                cat_dist[cat] = cat_dist.get(cat, 0) + 1

        # Name from top 3 moods
        top3 = [m for m, _ in top_moods[:3]]
        name = "/".join(m.capitalize() for m in top3)

        profiles[f"C{c}"] = {
            "name": name,
            "n_scenes": n,
            "pct": round(n / len(coords) * 100, 1),
            "silhouette": round(cluster_sils[c], 4),
            "top_moods": [[m, c_] for m, c_ in top_moods],
            "mean_pcs": mean_pcs,
            "categories": cat_dist,
        }

        print(f"\n  C{c} ({name}): {n} scenes ({n/len(coords)*100:.1f}%), sil={cluster_sils[c]:.3f}")
        print(f"    Moods: {', '.join(f'{m}({cnt})' for m, cnt in top_moods[:5])}")
        print(f"    Categories: {cat_dist}")

    # Max cluster size ratio (balance indicator)
    sizes = np.bincount(labels)
    balance = round(float(sizes.max() / sizes.min()), 2)

    results[k] = {
        "silhouette_avg": round(float(sil_avg), 4),
        "cluster_sizes": sizes.tolist(),
        "balance_ratio": balance,
        "profiles": profiles,
        "kmeans": km,
        "labels": labels,
    }

# ── Comparison table ──
print(f"\n\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"{'k':>3} | {'Sil':>6} | {'Balance':>7} | {'Sizes':<30} | {'Largest %':>9}")
for k in [3, 4, 5, 6]:
    r = results[k]
    sizes = r["cluster_sizes"]
    largest_pct = max(sizes) / sum(sizes) * 100
    print(f"  {k} | {r['silhouette_avg']:.4f} | {r['balance_ratio']:>7.2f} | {str(sizes):<30} | {largest_pct:>8.1f}%")

# ── Save k=5 and k=6 results ──
for k in [5, 6]:
    r = results[k]
    save_data = {
        "k": k,
        "silhouette_avg": r["silhouette_avg"],
        "cluster_sizes": r["cluster_sizes"],
        "balance_ratio": r["balance_ratio"],
        "profiles": r["profiles"],
    }
    # Remove non-serializable
    for ck in save_data["profiles"]:
        save_data["profiles"][ck]["top_moods"] = save_data["profiles"][ck]["top_moods"]

    with open(os.path.join(OUT, f"cluster_profiles_k{k}.json"), "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: cluster_profiles_k{k}.json")

    # Save coordinates CSV with this k's labels
    coords_k = coords.copy()
    coords_k["cluster_label"] = r["labels"]
    coords_k.to_csv(os.path.join(OUT, f"gacs_coordinates_k{k}.csv"), index=False)
    print(f"Saved: gacs_coordinates_k{k}.csv")

# ── Visualization ──
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for idx, k in enumerate([3, 4, 5, 6]):
    ax = axes[idx // 2][idx % 2]
    r = results[k]
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=r["labels"], cmap="tab10", alpha=0.4, s=6
    )
    ax.set_xlabel("PC1: Tense ↔ Joyful")
    ax.set_ylabel("PC2: Somber ↔ Intense")
    ax.set_title(f"k={k} (sil={r['silhouette_avg']:.3f}, balance={r['balance_ratio']:.1f}x)")

    # Annotate cluster centers
    km = r["kmeans"]
    for c in range(k):
        center = km.cluster_centers_[c]
        name = r["profiles"][f"C{c}"]["name"].split("/")[0]
        ax.annotate(f"C{c}: {name}", (center[0], center[1]),
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.suptitle("Cluster Comparison: k=3,4,5,6", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "cluster_comparison_k3456.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: cluster_comparison_k3456.png")

print("\n=== DONE ===")
