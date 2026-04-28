"""
Step 2 Retrain: 3,355씬 전체 emotion_distil_norm 임베딩 + 좌표계 재학습
기존 1,275씬 → 3,355씬 확장 데이터로 재생성.

Phase 1:  gacs_dataset.csv → 텍스트 추출 → emotion_distil_norm 임베딩 (768d)
Phase 2:  PCA(14) + KMeans sweep → 좌표계 v2.0 학습 + 안정성 검증
Output:   data/step2_full/, research/2026-03-16_coordinate_v2/
"""

import json
import os
import pickle
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = "/home/jongcircle/dev/gacs0202"
DATASET_CSV = os.path.join(ROOT, "gacs_0202/data/embeddings/gacs_dataset.csv")
OUT_EMB = os.path.join(ROOT, "data/step2_full")
OUT_RESEARCH = os.path.join(ROOT, "research/2026-03-16_coordinate_v2")
os.makedirs(OUT_EMB, exist_ok=True)
os.makedirs(OUT_RESEARCH, exist_ok=True)

N_COMPONENTS = 14
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: LOAD DATA + EMBED
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: FULL DATASET EMBEDDING (3,355 scenes)")
print("=" * 70)

df = pd.read_csv(DATASET_CSV)
print(f"Loaded: {len(df)} rows from gacs_dataset.csv")
print(f"Columns: {list(df.columns)}")

# ── 텍스트 생성: mood + style + object → 하나의 문장 ──
def build_text(row):
    """mood/style/object 컬럼에서 텍스트 생성 (기존 texts.csv 'original' 포맷과 유사)"""
    parts = []
    # mood_1 ~ mood_5
    for i in range(1, 6):
        col = f"mood_{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]).strip())
    # style_1 ~ style_3
    for i in range(1, 4):
        col = f"style_{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]).strip())
    # object_1 ~ object_5
    for i in range(1, 6):
        col = f"object_{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]).strip())
    return " ".join(parts) if parts else ""

def build_text_normalized(row):
    """구조화된 텍스트 (normalized 포맷)"""
    moods = []
    for i in range(1, 6):
        col = f"mood_{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            moods.append(str(row[col]).strip())
    styles = []
    for i in range(1, 4):
        col = f"style_{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            styles.append(str(row[col]).strip())
    objects = []
    for i in range(1, 6):
        col = f"object_{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            objects.append(str(row[col]).strip())

    parts = []
    if moods:
        parts.append(f"mood: {', '.join(moods)}")
    if styles:
        parts.append(f"style: {', '.join(styles)}")
    if objects:
        parts.append(f"objects: {', '.join(objects)}")
    return ". ".join(parts) if parts else ""

print("\nBuilding text descriptions...")
df["text_original"] = df.apply(build_text, axis=1)
df["text_normalized"] = df.apply(build_text_normalized, axis=1)

# 빈 텍스트 체크
empty_mask = df["text_original"].str.strip() == ""
n_empty = empty_mask.sum()
print(f"  Empty texts: {n_empty} / {len(df)}")
if n_empty > 0:
    print(f"  Dropping {n_empty} empty rows")
    df = df[~empty_mask].reset_index(drop=True)
    print(f"  Remaining: {len(df)} rows")

# 텍스트 저장
texts_df = df[["text_original", "text_normalized"]].copy()
texts_df.columns = ["original", "normalized"]
texts_df.to_csv(os.path.join(OUT_EMB, "texts.csv"), index=False)

# scene_index 저장
scene_index = df[["scene_id", "video_id"]].copy()
scene_index.to_csv(os.path.join(OUT_EMB, "scene_index.csv"), index=False)
print(f"  Saved texts.csv ({len(texts_df)} rows) and scene_index.csv")

# ── Derive category from video_id ──
def get_category(vid):
    vid = str(vid).lower()
    if "advertisement" in vid:
        return "advertisements"
    elif "animation" in vid:
        return "animations"
    elif "emotional" in vid:
        return "emotional_shorts"
    elif "trailer" in vid or "movie" in vid:
        return "movie_trailers"
    else:
        return "other"

df["category"] = df["video_id"].apply(get_category)
print("\nCategories:")
for cat, cnt in df["category"].value_counts().items():
    print(f"  {cat}: {cnt}")

# ── Emotion Embedding ──
print("\n--- Loading j-hartmann/emotion-english-distilroberta-base ---")
model = SentenceTransformer(
    "j-hartmann/emotion-english-distilroberta-base", device="cuda"
)
print("  Model loaded, device=cuda")

print("\nEncoding original texts (normalized embeddings)...")
t0 = time.time()
emb_norm = model.encode(
    texts_df["original"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"  Shape: {emb_norm.shape}, time: {time.time()-t0:.1f}s")
np.save(os.path.join(OUT_EMB, "emb_emotion_distil_norm.npy"), emb_norm)

print("Encoding normalized texts...")
t0 = time.time()
emb_norm_v2 = model.encode(
    texts_df["normalized"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"  Shape: {emb_norm_v2.shape}, time: {time.time()-t0:.1f}s")
np.save(os.path.join(OUT_EMB, "emb_emotion_distil_norm_v2.npy"), emb_norm_v2)

print(f"\nSaved to {OUT_EMB}/")

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: COORDINATE SYSTEM TRAINING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: COORDINATE SYSTEM v2.0 TRAINING")
print("=" * 70)

embeddings = emb_norm  # use original text, normalized embeddings (winner from step3)
n_scenes = len(embeddings)
print(f"\nTraining on {n_scenes} scenes, {embeddings.shape[1]}-dim embeddings")

# ── PCA ──
print("\n--- StandardScaler + PCA ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(embeddings)
pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"  PCA {N_COMPONENTS} components: {cumvar[-1]:.4f} total variance")
for i in range(N_COMPONENTS):
    print(f"    PC{i+1:2d}: {pca.explained_variance_ratio_[i]:.4f} (cum: {cumvar[i]:.4f})")

# ── KMeans Sweep ──
print("\n--- KMeans Sweep k=3..15 ---")
sil_scores = {}
for k in range(3, 16):
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=min(2000, n_scenes))
    sil_scores[k] = sil
    print(f"  k={k:2d}: silhouette={sil:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
best_sil = sil_scores[best_k]
print(f"\n  Best: k={best_k}, silhouette={best_sil:.4f}")

# ── Final KMeans ──
print(f"\n--- Fitting final KMeans (k={best_k}) ---")
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)
print(f"  Cluster sizes: {np.bincount(cluster_labels).tolist()}")

# ══════════════════════════════════════════════════════════════════════════
# PHASE 3: AXIS STABILITY TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: AXIS STABILITY TESTS")
print("=" * 70)

def pc_cosine_similarities(pca_ref, pca_test, n=N_COMPONENTS):
    sims = []
    for i in range(n):
        v1 = pca_ref.components_[i]
        v2 = pca_test.components_[i]
        cos = np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sims.append(round(float(cos), 4))
    return sims

# ── Bootstrap ──
print("\nA1: Bootstrap (10 iter, 80% sample)")
n_bootstrap = 10
n_sample = int(0.8 * n_scenes)
bootstrap_sims = []
for i in range(n_bootstrap):
    idx = np.random.choice(n_scenes, size=n_sample, replace=False)
    sc = StandardScaler()
    X_sub = sc.fit_transform(embeddings[idx])
    pca_sub = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    pca_sub.fit(X_sub)
    sims = pc_cosine_similarities(pca, pca_sub)
    bootstrap_sims.append(sims)
    if i % 3 == 0:
        print(f"  Iter {i+1:2d}: PC1={sims[0]:.4f}  PC7={sims[6]:.4f}  PC14={sims[13]:.4f}")

bootstrap_sims = np.array(bootstrap_sims)
bootstrap_mean = np.mean(bootstrap_sims, axis=0)

# ── Split-Half ──
print("\nA2: Split-Half (5 splits)")
split_sims = []
for i in range(5):
    perm = np.random.permutation(n_scenes)
    half = n_scenes // 2
    sc_a = StandardScaler()
    X_a = sc_a.fit_transform(embeddings[perm[:half]])
    pca_a = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    pca_a.fit(X_a)
    sc_b = StandardScaler()
    X_b = sc_b.fit_transform(embeddings[perm[half:half*2]])
    pca_b = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    pca_b.fit(X_b)
    sims = pc_cosine_similarities(pca_a, pca_b)
    split_sims.append(sims)
    print(f"  Split {i+1}: PC1={sims[0]:.4f}  PC7={sims[6]:.4f}  PC14={sims[13]:.4f}")

split_sims = np.array(split_sims)
split_mean = np.mean(split_sims, axis=0)

# ── Category Dropout ──
categories_list = sorted(df["category"].unique())
print(f"\nA3: Category Dropout ({categories_list})")
cat_dropout = {}
for drop_cat in categories_list:
    mask = df["category"] != drop_cat
    X_sub = embeddings[mask.values]
    sc = StandardScaler()
    X_sub_sc = sc.fit_transform(X_sub)
    pca_sub = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    pca_sub.fit(X_sub_sc)
    sims = pc_cosine_similarities(pca, pca_sub)
    cat_dropout[drop_cat] = sims
    n_dropped = (~mask).sum()
    print(f"  Drop '{drop_cat}' (-{n_dropped}): PC1={sims[0]:.4f}  PC7={sims[6]:.4f}")

cat_dropout_arr = np.array(list(cat_dropout.values()))
cat_dropout_mean = np.mean(cat_dropout_arr, axis=0)

# ── Stability Summary ──
print("\n" + "=" * 70)
print("STABILITY SUMMARY")
print("=" * 70)

stability = {"bootstrap": {}, "split_half": {}, "category_dropout": {}, "summary": []}
for j in range(N_COMPONENTS):
    pc_key = f"PC{j+1}"
    overall = round(float(np.mean([bootstrap_mean[j], split_mean[j], cat_dropout_mean[j]])), 4)
    status = "STABLE" if overall > 0.9 else ("MODERATE" if overall >= 0.7 else "UNSTABLE")
    stability["bootstrap"][pc_key] = round(float(bootstrap_mean[j]), 4)
    stability["split_half"][pc_key] = round(float(split_mean[j]), 4)
    stability["category_dropout"][pc_key] = {c: round(float(cat_dropout[c][j]), 4) for c in categories_list}
    stability["summary"].append({
        "pc": pc_key,
        "bootstrap": round(float(bootstrap_mean[j]), 4),
        "split_half": round(float(split_mean[j]), 4),
        "cat_dropout": round(float(cat_dropout_mean[j]), 4),
        "overall": overall,
        "status": status,
        "variance_pct": round(float(pca.explained_variance_ratio_[j] * 100), 2),
    })
    print(f"  {pc_key:4s}: boot={bootstrap_mean[j]:.4f}  split={split_mean[j]:.4f}  "
          f"cat={cat_dropout_mean[j]:.4f}  => {status}  (var={pca.explained_variance_ratio_[j]*100:.2f}%)")

n_stable = sum(1 for s in stability["summary"] if s["status"] == "STABLE")
n_moderate = sum(1 for s in stability["summary"] if s["status"] == "MODERATE")
n_unstable = sum(1 for s in stability["summary"] if s["status"] == "UNSTABLE")
print(f"\n  STABLE: {n_stable} / MODERATE: {n_moderate} / UNSTABLE: {n_unstable}")

with open(os.path.join(OUT_RESEARCH, "axis_stability_v2.json"), "w") as f:
    json.dump(stability, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
# PHASE 4: SAVE COORDINATE SYSTEM v2.0
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: SAVING COORDINATE SYSTEM v2.0")
print("=" * 70)

# Axis names (will be interpreted from top mood correlations)
# For now, use PC indices; can be renamed after analysis
axis_names = [f"PC{i+1}" for i in range(N_COMPONENTS)]

coordinate_system = {
    "scaler": scaler,
    "pca": pca,
    "axis_names": axis_names,
    "axis_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
    "kmeans": kmeans,
    "metadata": {
        "version": "2.0",
        "n_scenes_trained": n_scenes,
        "best_k": best_k,
        "best_silhouette": round(float(best_sil), 4),
        "embedding_model": "j-hartmann/emotion-english-distilroberta-base",
        "embedding_type": "original_text_normalized_emb",
        "date": "2026-03-16",
        "stability": {
            "n_stable": n_stable,
            "n_moderate": n_moderate,
            "n_unstable": n_unstable,
        },
    },
}

pkl_path = os.path.join(OUT_RESEARCH, "gacs_coordinate_system_v2.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(coordinate_system, f)
print(f"  Saved: {pkl_path} ({os.path.getsize(pkl_path) / 1024:.1f} KB)")

# ── Save coordinates CSV ──
coords_df = df[["scene_id", "video_id", "category"]].copy()

# dominant mood
coords_df["dominant_mood"] = df["mood_1"]

# PC columns
for i in range(N_COMPONENTS):
    coords_df[f"PC{i+1}"] = X_pca[:, i]

coords_df["cluster_label"] = cluster_labels

csv_path = os.path.join(OUT_RESEARCH, "gacs_coordinates_v2.csv")
coords_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path} ({len(coords_df)} rows)")

# ── KMeans sweep plot ──
fig, axes_plt = plt.subplots(1, 2, figsize=(14, 5))

# Silhouette sweep
ks = sorted(sil_scores.keys())
axes_plt[0].bar(ks, [sil_scores[k] for k in ks], color="steelblue", alpha=0.8)
axes_plt[0].axvline(x=best_k, color="red", linestyle="--", label=f"best k={best_k}")
axes_plt[0].set_xlabel("k")
axes_plt[0].set_ylabel("Silhouette Score")
axes_plt[0].set_title(f"KMeans Sweep (n={n_scenes})")
axes_plt[0].legend()

# PCA scree
axes_plt[1].bar(range(1, N_COMPONENTS+1), pca.explained_variance_ratio_, color="steelblue", alpha=0.7)
axes_plt[1].plot(range(1, N_COMPONENTS+1), cumvar, "ro-", markersize=4)
axes_plt[1].axhline(y=0.8, color="red", linestyle="--", alpha=0.3, label="80%")
axes_plt[1].set_xlabel("Component")
axes_plt[1].set_ylabel("Variance Explained")
axes_plt[1].set_title(f"PCA Scree (cum={cumvar[-1]:.1%})")
axes_plt[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_RESEARCH, "training_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Stability heatmap ──
fig, ax = plt.subplots(figsize=(10, 8))
col_labels = ["Bootstrap", "Split-Half"] + [f"Drop {c}" for c in categories_list]
n_cols = len(col_labels)
heatmap_data = np.zeros((N_COMPONENTS, n_cols))
for j in range(N_COMPONENTS):
    heatmap_data[j, 0] = bootstrap_mean[j]
    heatmap_data[j, 1] = split_mean[j]
    for k, cat in enumerate(categories_list):
        heatmap_data[j, 2 + k] = cat_dropout[cat][j]

row_labels = [f"PC{j+1}" for j in range(N_COMPONENTS)]
cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect="auto")
for i in range(N_COMPONENTS):
    for j_col in range(n_cols):
        val = heatmap_data[i, j_col]
        color = "white" if val < 0.5 else "black"
        ax.text(j_col, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color, fontweight="bold")
ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(N_COMPONENTS))
ax.set_yticklabels(row_labels, fontsize=9)
ax.set_title(f"Axis Stability v2.0 ({n_scenes} scenes)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8).set_label("Cosine Similarity")
plt.tight_layout()
plt.savefig(os.path.join(OUT_RESEARCH, "axis_stability_v2.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PHASE 5: COMPARE v1 vs v2
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: v1 (1,275) vs v2 (3,355) COMPARISON")
print("=" * 70)

v1_pkl = os.path.join(ROOT, "research/2026-03-12_coordinate_stabilization/gacs_coordinate_system_v1.pkl")
if os.path.exists(v1_pkl):
    with open(v1_pkl, "rb") as f:
        cs_v1 = pickle.load(f)
    # Compare PC loadings
    print("\nPC Loading Cosine Similarity (v1 vs v2):")
    for i in range(N_COMPONENTS):
        v1_comp = cs_v1["pca"].components_[i]
        v2_comp = pca.components_[i]
        cos = np.abs(np.dot(v1_comp, v2_comp) / (np.linalg.norm(v1_comp) * np.linalg.norm(v2_comp)))
        print(f"  PC{i+1:2d}: {cos:.4f}")
else:
    print(f"  v1 pkl not found at {v1_pkl}, skipping comparison")

# ── Final Summary ──
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n  Dataset:        {n_scenes} scenes")
print("  Embedding:      emotion_distil_norm (768d, L2 normalized)")
print(f"  PCA:            {N_COMPONENTS} components, {cumvar[-1]:.4f} variance")
print(f"  Best k:         {best_k} (silhouette={best_sil:.4f})")
print(f"  Cluster sizes:  {np.bincount(cluster_labels).tolist()}")
print(f"  Stability:      {n_stable} STABLE / {n_moderate} MODERATE / {n_unstable} UNSTABLE")
print("\nOutputs:")
print(f"  {OUT_EMB}/emb_emotion_distil_norm.npy ({n_scenes}x768)")
print(f"  {OUT_RESEARCH}/gacs_coordinate_system_v2.pkl")
print(f"  {OUT_RESEARCH}/gacs_coordinates_v2.csv ({n_scenes} rows)")
print(f"  {OUT_RESEARCH}/axis_stability_v2.json")
print(f"  {OUT_RESEARCH}/training_summary.png")
print(f"  {OUT_RESEARCH}/axis_stability_v2.png")
print("\nDone.")
