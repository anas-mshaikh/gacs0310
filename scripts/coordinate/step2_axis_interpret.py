"""
V2 좌표계 축 재해석: 3,355씬 mood 라벨 ↔ PC좌표 상관분석
v1과 동일한 방법론 (Pearson r, extreme scene mood 분포)
"""

import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = "/home/jongcircle/dev/gacs0202"
COORDS_CSV = os.path.join(ROOT, "research/2026-03-16_coordinate_v2/gacs_coordinates_v2.csv")
DATASET_CSV = os.path.join(ROOT, "gacs_0202/data/embeddings/gacs_dataset.csv")
OUT = os.path.join(ROOT, "research/2026-03-16_coordinate_v2")

N_COMPONENTS = 14
N_STABLE = 7  # PC1-7 are STABLE

# ── Load data ──
coords = pd.read_csv(COORDS_CSV)
dataset = pd.read_csv(DATASET_CSV)
print(f"Coordinates: {len(coords)} rows")
print(f"Dataset: {len(dataset)} rows")

# ── Extract all mood labels per scene ──
mood_cols = [f"mood_{i}" for i in range(1, 6)]
all_moods = set()
scene_moods = []  # list of sets
for _, row in dataset.iterrows():
    moods = set()
    for col in mood_cols:
        if pd.notna(row.get(col)) and str(row[col]).strip():
            moods.add(str(row[col]).strip().lower())
    scene_moods.append(moods)
    all_moods.update(moods)

# Filter moods with >= 20 occurrences
mood_counts = {}
for moods in scene_moods:
    for m in moods:
        mood_counts[m] = mood_counts.get(m, 0) + 1

valid_moods = sorted([m for m, c in mood_counts.items() if c >= 20])
print(f"\nTotal unique moods: {len(all_moods)}")
print(f"Valid moods (>=20 occurrences): {len(valid_moods)}")

# ── Binary mood indicators ──
mood_binary = {}
for mood in valid_moods:
    mood_binary[mood] = np.array([1 if mood in s else 0 for s in scene_moods])

# ── Correlation analysis: each PC vs each mood ──
print("\n" + "=" * 70)
print("AXIS INTERPRETATION (PC-Mood Correlations)")
print("=" * 70)

axis_interpretation = {}

for pc_idx in range(N_COMPONENTS):
    pc_key = f"PC{pc_idx + 1}"
    pc_values = coords[pc_key].values
    var_pct = float(coords[pc_key].var())  # we'll use actual from PCA

    # Correlate with each mood
    correlations = []
    for mood in valid_moods:
        r, p = stats.pointbiserialr(mood_binary[mood], pc_values)
        correlations.append({"mood": mood, "r": round(float(r), 4), "p": round(float(p), 6)})

    correlations.sort(key=lambda x: x["r"], reverse=True)

    top5_pos = correlations[:5]
    top5_neg = correlations[-5:][::-1]  # most negative first

    # Extreme scenes (top/bottom 10% by PC value)
    n_extreme = max(int(len(pc_values) * 0.1), 50)
    top_idx = np.argsort(pc_values)[-n_extreme:]
    bot_idx = np.argsort(pc_values)[:n_extreme]

    def count_moods_in_idx(idx):
        counts = {}
        for i in idx:
            for m in scene_moods[i]:
                if m in valid_moods:
                    counts[m] = counts.get(m, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]

    extreme_pos = count_moods_in_idx(top_idx)
    extreme_neg = count_moods_in_idx(bot_idx)

    # Name the axis
    pos_name = top5_pos[0]["mood"].capitalize()
    neg_name = top5_neg[0]["mood"].capitalize()
    axis_name = f"{pos_name}-{neg_name}"

    axis_interpretation[pc_key] = {
        "name": axis_name,
        "label": f"{pc_key}: {axis_name} ({top5_pos[0]['mood']} r={top5_pos[0]['r']:+.4f} vs {top5_neg[0]['mood']} r={top5_neg[0]['r']:+.4f})",
        "top5_positive": top5_pos,
        "top5_negative": top5_neg,
        "extreme_positive_moods": extreme_pos,
        "extreme_negative_moods": extreme_neg,
    }

    stable = "STABLE" if pc_idx < N_STABLE else "moderate"
    print(f"\n  {pc_key} [{stable}]: {axis_name}")
    print(f"    (+) {', '.join(f'{c['mood']}({c['r']:+.3f})' for c in top5_pos)}")
    print(f"    (-) {', '.join(f'{c['mood']}({c['r']:+.3f})' for c in top5_neg)}")

# ── Save interpretation ──
json_path = os.path.join(OUT, "axis_interpretation_v2.json")
with open(json_path, "w") as f:
    json.dump(axis_interpretation, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {json_path}")

# ── Cluster profiling ──
print("\n" + "=" * 70)
print("CLUSTER PROFILES (k=3)")
print("=" * 70)

cluster_profiles = {}
for cl in sorted(coords["cluster_label"].unique()):
    mask = coords["cluster_label"] == cl
    cl_scenes = [scene_moods[i] for i in range(len(mask)) if mask.iloc[i]]
    n = int(mask.sum())

    # Top moods in this cluster
    mood_freq = {}
    for s in cl_scenes:
        for m in s:
            if m in valid_moods:
                mood_freq[m] = mood_freq.get(m, 0) + 1
    top_moods = sorted(mood_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    # Mean PC values
    mean_pcs = {f"PC{i+1}": round(float(coords.loc[mask, f"PC{i+1}"].mean()), 3) for i in range(N_STABLE)}

    # Category distribution
    cat_dist = coords.loc[mask, "category"].value_counts().to_dict()

    cluster_profiles[f"C{cl}"] = {
        "n_scenes": n,
        "pct": round(n / len(coords) * 100, 1),
        "top_moods": top_moods,
        "mean_pcs": mean_pcs,
        "categories": {k: int(v) for k, v in cat_dist.items()},
    }

    print(f"\n  C{cl}: {n} scenes ({n/len(coords)*100:.1f}%)")
    print(f"    Top moods: {', '.join(f'{m}({c})' for m,c in top_moods[:5])}")
    print(f"    Mean PCs: {mean_pcs}")
    print(f"    Categories: {cat_dist}")

# Name clusters based on dominant moods
for ck, prof in cluster_profiles.items():
    top3 = [m for m, _ in prof["top_moods"][:3]]
    prof["suggested_name"] = "/".join(m.capitalize() for m in top3)
    print(f"\n  {ck} suggested name: {prof['suggested_name']}")

json_path = os.path.join(OUT, "cluster_profiles_v2.json")
with open(json_path, "w") as f:
    json.dump(cluster_profiles, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {json_path}")

# ── Visualization: PC1 vs PC2 colored by cluster ──
fig, axes_plt = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: PC1 vs PC2, colored by cluster
scatter = axes_plt[0].scatter(
    coords["PC1"], coords["PC2"],
    c=coords["cluster_label"], cmap="tab10", alpha=0.4, s=8
)
axes_plt[0].set_xlabel(f"PC1: {axis_interpretation['PC1']['name']}")
axes_plt[0].set_ylabel(f"PC2: {axis_interpretation['PC2']['name']}")
axes_plt[0].set_title(f"PC1 vs PC2 (clusters, n={len(coords)})")
plt.colorbar(scatter, ax=axes_plt[0], label="Cluster")

# Plot 2: PC1 vs PC3
scatter2 = axes_plt[1].scatter(
    coords["PC1"], coords["PC3"],
    c=coords["cluster_label"], cmap="tab10", alpha=0.4, s=8
)
axes_plt[1].set_xlabel(f"PC1: {axis_interpretation['PC1']['name']}")
axes_plt[1].set_ylabel(f"PC3: {axis_interpretation['PC3']['name']}")
axes_plt[1].set_title("PC1 vs PC3")
plt.colorbar(scatter2, ax=axes_plt[1], label="Cluster")

# Plot 3: Axis correlation heatmap (top 10 moods × 7 stable PCs)
top_moods_list = sorted(valid_moods, key=lambda m: mood_counts[m], reverse=True)[:15]
heatmap_data = np.zeros((len(top_moods_list), N_STABLE))
for i, mood in enumerate(top_moods_list):
    for j in range(N_STABLE):
        pc_key = f"PC{j+1}"
        for c in axis_interpretation[pc_key]["top5_positive"] + axis_interpretation[pc_key]["top5_negative"]:
            if c["mood"] == mood:
                heatmap_data[i, j] = c["r"]
                break
        else:
            # compute if not in top5
            r, _ = stats.pointbiserialr(mood_binary[mood], coords[pc_key].values)
            heatmap_data[i, j] = r

im = axes_plt[2].imshow(heatmap_data, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
axes_plt[2].set_xticks(range(N_STABLE))
axes_plt[2].set_xticklabels([f"PC{i+1}" for i in range(N_STABLE)], fontsize=8)
axes_plt[2].set_yticks(range(len(top_moods_list)))
axes_plt[2].set_yticklabels(top_moods_list, fontsize=8)
axes_plt[2].set_title("Mood × Axis Correlation")
for i in range(len(top_moods_list)):
    for j in range(N_STABLE):
        axes_plt[2].text(j, i, f"{heatmap_data[i,j]:.2f}", ha="center", va="center", fontsize=6)
plt.colorbar(im, ax=axes_plt[2], label="Pearson r")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "axis_interpretation_v2.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT}/axis_interpretation_v2.png")

# ── v1 vs v2 axis name comparison ──
print("\n" + "=" * 70)
print("v1 vs v2 AXIS NAME COMPARISON")
print("=" * 70)

v1_path = os.path.join(ROOT, "research/2026-03-10_coordinate_discovery/axis_interpretation.json")
if os.path.exists(v1_path):
    with open(v1_path) as f:
        v1 = json.load(f)
    for i in range(N_STABLE):
        pc = f"PC{i+1}"
        v1_name = v1.get(pc, {}).get("name", "?")
        v2_name = axis_interpretation[pc]["name"]
        print(f"  {pc}: v1={v1_name:25s}  →  v2={v2_name}")

# ── Update coordinate system pkl with axis names ──
import pickle

pkl_path = os.path.join(OUT, "gacs_coordinate_system_v2.pkl")
with open(pkl_path, "rb") as f:
    cs = pickle.load(f)

cs["axis_names"] = [axis_interpretation[f"PC{i+1}"]["name"] for i in range(N_COMPONENTS)]
cs["axis_interpretation"] = axis_interpretation
cs["cluster_profiles"] = cluster_profiles

with open(pkl_path, "wb") as f:
    pickle.dump(cs, f)
print(f"\nUpdated {pkl_path} with axis names and cluster profiles")

print("\n=== DONE ===")
