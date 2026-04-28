"""
Step 5: Axis Discovery
Discover interpretable coordinate axes from the emotion embedding space.
- PCA to find principal axes and their mood correlates
- UMAP 2D/3D projections colored by mood category
- Axis-mood correlation heatmap
- Proposed axis names saved to JSON
"""

import ast
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────
EMB_PATH = "data/step3/emb_emotion_distil_norm.npy"
CSV_PATH = "gacs0202(kushi)/gacs_labeled_scenes.csv"
OUT_DIR  = "data/step5"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Mood normalization map ───────────────────────────────────────────────
MOOD_GROUPS = {
    "joy":        ["joyful", "happy", "cheerful", "delighted", "ecstatic", "lighthearted", "upbeat", "bright"],
    "excitement": ["excited", "energetic", "vibrant", "lively", "dynamic", "thrilling", "exhilarating"],
    "calm":       ["calm", "relaxed", "tranquil", "serene", "peaceful", "gentle", "soothing", "mellow"],
    "warmth":     ["warm", "tender", "loving", "caring", "heartwarming", "sweet", "compassionate"],
    "sadness":    ["sad", "somber", "melancholic", "gloomy", "sorrowful", "tragic", "bleak"],
    "tension":    ["tense", "intense", "suspenseful", "anxious", "uneasy", "stressed"],
    "fear":       ["scared", "terrifying", "frightening", "eerie", "creepy", "ominous", "menacing"],
    "anger":      ["angry", "furious", "aggressive", "fierce", "violent", "brutal"],
    "mystery":    ["mysterious", "enigmatic", "surreal", "ethereal", "dreamlike", "mystical"],
    "neutral":    ["neutral", "ordinary", "mundane", "casual", "simple", "plain"],
}

# Invert: raw_word -> group
WORD_TO_GROUP = {}
for group, words in MOOD_GROUPS.items():
    for w in words:
        WORD_TO_GROUP[w] = group


def parse_mood_list(mood_json_str):
    """Parse the mood_json column (stringified Python list)."""
    try:
        return [w.strip().lower() for w in ast.literal_eval(mood_json_str)]
    except Exception:
        return []


def assign_dominant_group(mood_words):
    """Given a list of raw mood words, return the dominant mood group."""
    groups = [WORD_TO_GROUP[w] for w in mood_words if w in WORD_TO_GROUP]
    if not groups:
        return "other"
    return Counter(groups).most_common(1)[0][0]


# =====================================================================
# 1. Load data
# =====================================================================
print("Loading embeddings and labels...")
embeddings = np.load(EMB_PATH)
df = pd.read_csv(CSV_PATH)
assert len(df) == embeddings.shape[0], "Row count mismatch"
print(f"  Embeddings: {embeddings.shape}")

# Parse moods
df["mood_list"] = df["mood_json"].apply(parse_mood_list)
df["dominant_mood"] = df["mood_list"].apply(assign_dominant_group)

# Collect all unique raw mood words
all_mood_words = sorted({w for words in df["mood_list"] for w in words})
print(f"  Unique raw mood words: {len(all_mood_words)}")
print(f"  Dominant mood distribution:\n{df['dominant_mood'].value_counts().to_string()}\n")

# Binary mood indicators for each raw word
mood_matrix = np.zeros((len(df), len(all_mood_words)), dtype=np.float32)
for i, words in enumerate(df["mood_list"]):
    for w in words:
        if w in all_mood_words:
            mood_matrix[i, all_mood_words.index(w)] = 1.0

# Binary mood-group indicators
mood_groups_list = sorted(MOOD_GROUPS.keys())
group_matrix = np.zeros((len(df), len(mood_groups_list)), dtype=np.float32)
for i, words in enumerate(df["mood_list"]):
    for w in words:
        g = WORD_TO_GROUP.get(w)
        if g and g in mood_groups_list:
            group_matrix[i, mood_groups_list.index(g)] = 1.0

# =====================================================================
# 2. PCA on StandardScaled embeddings
# =====================================================================
print("Running PCA...")
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)

pca = PCA(n_components=10, random_state=42)
pca_coords = pca.fit_transform(emb_scaled)

print(f"  Explained variance (top 10 PCs): {pca.explained_variance_ratio_}")
print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_)}\n")

# For each PC, find top-20 mood words by Spearman correlation
pc_mood_correlations = {}  # pc_idx -> list of (word, rho)
for pc_idx in range(10):
    pc_vals = pca_coords[:, pc_idx]
    corrs = []
    for j, word in enumerate(all_mood_words):
        col = mood_matrix[:, j]
        if col.sum() < 3:  # skip very rare words
            continue
        rho, _ = spearmanr(pc_vals, col)
        corrs.append((word, rho))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    pc_mood_correlations[pc_idx] = corrs[:20]

# Print PC interpretations
for pc_idx in range(10):
    ev = pca.explained_variance_ratio_[pc_idx]
    top = pc_mood_correlations[pc_idx]
    pos_words = [(w, r) for w, r in top if r > 0][:5]
    neg_words = [(w, r) for w, r in top if r < 0][:5]
    print(f"PC{pc_idx+1} ({ev:.1%} var):")
    print(f"  + : {', '.join(f'{w}({r:+.3f})' for w,r in pos_words)}")
    print(f"  - : {', '.join(f'{w}({r:+.3f})' for w,r in neg_words)}")
    print()

# =====================================================================
# 3. UMAP 2D and 3D projections
# =====================================================================
# Color palette for mood groups
GROUP_COLORS = {
    "joy":        "#FFD700",
    "excitement": "#FF6347",
    "calm":       "#87CEEB",
    "warmth":     "#FF8C69",
    "sadness":    "#4682B4",
    "tension":    "#DC143C",
    "fear":       "#2F4F4F",
    "anger":      "#8B0000",
    "mystery":    "#9370DB",
    "neutral":    "#A9A9A9",
    "other":      "#D3D3D3",
}

print("Running UMAP 2D...")
reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
umap_2d = reducer_2d.fit_transform(emb_scaled)

fig, ax = plt.subplots(figsize=(12, 9))
groups_present = sorted(df["dominant_mood"].unique())
for g in groups_present:
    mask = df["dominant_mood"] == g
    ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
               c=GROUP_COLORS.get(g, "#999999"), label=g,
               s=18, alpha=0.65, edgecolors="none")
ax.set_xlabel("UMAP-1", fontsize=13)
ax.set_ylabel("UMAP-2", fontsize=13)
ax.set_title("UMAP 2D Projection — Emotion Embeddings\nColored by Dominant Mood Category", fontsize=15)
ax.legend(title="Mood Group", fontsize=10, title_fontsize=11,
          markerscale=2, loc="best", framealpha=0.9)
ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "umap_2d_mood.png"), dpi=200)
plt.close(fig)
print(f"  Saved {OUT_DIR}/umap_2d_mood.png")

print("Running UMAP 3D...")
reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=30, min_dist=0.3)
umap_3d = reducer_3d.fit_transform(emb_scaled)

fig = plt.figure(figsize=(20, 6))
angles = [(30, 45), (30, 135), (60, 225)]
for vi, (elev, azim) in enumerate(angles):
    ax = fig.add_subplot(1, 3, vi + 1, projection="3d")
    for g in groups_present:
        mask = df["dominant_mood"] == g
        ax.scatter(umap_3d[mask, 0], umap_3d[mask, 1], umap_3d[mask, 2],
                   c=GROUP_COLORS.get(g, "#999999"), label=g if vi == 0 else None,
                   s=12, alpha=0.55, edgecolors="none")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("U1", fontsize=10)
    ax.set_ylabel("U2", fontsize=10)
    ax.set_zlabel("U3", fontsize=10)
    ax.set_title(f"View (elev={elev}, azim={azim})", fontsize=11)
    ax.tick_params(labelsize=8)

fig.suptitle("UMAP 3D Projection — Emotion Embeddings\nColored by Dominant Mood Category",
             fontsize=14, y=1.02)
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=min(len(groups_present), 6),
           fontsize=9, markerscale=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "umap_3d_mood.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_DIR}/umap_3d_mood.png")

# =====================================================================
# 4. Axis-mood-group correlation heatmap
# =====================================================================
print("\nComputing axis-mood-group correlation matrix...")
n_pcs = 5
corr_matrix = np.zeros((n_pcs, len(mood_groups_list)))
pval_matrix = np.zeros_like(corr_matrix)

for pc_idx in range(n_pcs):
    for gi, g in enumerate(mood_groups_list):
        rho, pval = spearmanr(pca_coords[:, pc_idx], group_matrix[:, gi])
        corr_matrix[pc_idx, gi] = rho
        pval_matrix[pc_idx, gi] = pval

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            xticklabels=mood_groups_list,
            yticklabels=[f"PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})" for i in range(n_pcs)],
            ax=ax, linewidths=0.5, vmin=-0.4, vmax=0.4)

# Mark significant correlations
for i in range(n_pcs):
    for j in range(len(mood_groups_list)):
        if pval_matrix[i, j] < 0.001:
            ax.text(j + 0.5, i + 0.82, "***", ha="center", va="center", fontsize=7, color="black")
        elif pval_matrix[i, j] < 0.01:
            ax.text(j + 0.5, i + 0.82, "**", ha="center", va="center", fontsize=7, color="black")
        elif pval_matrix[i, j] < 0.05:
            ax.text(j + 0.5, i + 0.82, "*", ha="center", va="center", fontsize=7, color="black")

ax.set_title("Spearman Correlation: Principal Components vs. Mood Groups\n(* p<0.05, ** p<0.01, *** p<0.001)",
             fontsize=13)
ax.set_xlabel("Mood Group", fontsize=12)
ax.set_ylabel("Principal Component", fontsize=12)
ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "axis_mood_heatmap.png"), dpi=200)
plt.close(fig)
print(f"  Saved {OUT_DIR}/axis_mood_heatmap.png")

# =====================================================================
# 5. Propose axis names and save definitions
# =====================================================================
print("\nProposing axis names...")

axis_definitions = {}
for pc_idx in range(5):
    ev = float(pca.explained_variance_ratio_[pc_idx])
    top = pc_mood_correlations[pc_idx]
    pos_words = [(w, float(r)) for w, r in top if r > 0][:10]
    neg_words = [(w, float(r)) for w, r in top if r < 0][:10]

    # Top mood groups
    group_corrs = {mood_groups_list[j]: float(corr_matrix[pc_idx, j])
                   for j in range(len(mood_groups_list))}
    top_pos_groups = sorted(group_corrs.items(), key=lambda x: x[1], reverse=True)[:3]
    top_neg_groups = sorted(group_corrs.items(), key=lambda x: x[1])[:3]

    # Heuristic axis naming
    pos_group_names = [g for g, _ in top_pos_groups if _ > 0.05]
    neg_group_names = [g for g, _ in top_neg_groups if _ < -0.05]

    if pos_group_names and neg_group_names:
        axis_name = f"{'/'.join(pos_group_names[:2]).title()} vs {'/'.join(neg_group_names[:2]).title()}"
    elif pos_group_names:
        axis_name = f"{'/'.join(pos_group_names[:2]).title()} Axis"
    elif neg_group_names:
        axis_name = f"Anti-{'/'.join(neg_group_names[:2]).title()} Axis"
    else:
        axis_name = f"PC{pc_idx+1} (undetermined)"

    axis_definitions[f"PC{pc_idx+1}"] = {
        "proposed_name": axis_name,
        "explained_variance": round(ev, 4),
        "positive_end_moods": pos_words,
        "negative_end_moods": neg_words,
        "positive_groups": top_pos_groups,
        "negative_groups": top_neg_groups,
        "group_correlations": group_corrs,
    }

    print(f"  PC{pc_idx+1}: {axis_name}  (var={ev:.1%})")
    print(f"    + groups: {top_pos_groups}")
    print(f"    - groups: {top_neg_groups}")

# Save
with open(os.path.join(OUT_DIR, "axis_definitions.json"), "w") as f:
    json.dump(axis_definitions, f, indent=2, ensure_ascii=False)
print(f"\n  Saved {OUT_DIR}/axis_definitions.json")

# Also save PCA coords and UMAP coords for downstream use
np.save(os.path.join(OUT_DIR, "pca_10d.npy"), pca_coords)
np.save(os.path.join(OUT_DIR, "umap_2d.npy"), umap_2d)
np.save(os.path.join(OUT_DIR, "umap_3d.npy"), umap_3d)
print(f"  Saved pca_10d.npy, umap_2d.npy, umap_3d.npy to {OUT_DIR}/")

print("\n✓ Step 5 complete.")
