"""
research_3_coordinate_validation.py
Build GACS coordinate mapping and validate it.
"""

import ast
import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = "/home/jongcircle/dev/gacs0202"
OUT = os.path.join(ROOT, "research")
os.makedirs(OUT, exist_ok=True)

# ── MOOD_MAP ──
MOOD_MAP = {
    "joy": ["joyful","joyous","happy","cheerful","jubilant","overjoyed","delighted","ecstatic","lighthearted","upbeat","bright","radiant","glowing"],
    "excitement": ["excited","exciting","exhilarating","thrilling","energetic","vibrant","lively","dynamic","active","vigorous","spirited","boisterous","fast-paced","fast","bustling"],
    "celebration": ["celebratory","festive","triumphant","proud","accomplished"],
    "amusement": ["playful","whimsical","humorous","amusing","amused","comical","silly","mischievous","quirky","cute","adorable"],
    "wonder": ["magical","wondrous","enchanting","enchanted","majestic","awe-inspiring","breathtaking","captivating","grand","epic","impressive"],
    "adventure": ["adventurous","daring","bold","wild","free","exploratory","ambitious"],
    "calm": ["calm","relaxed","tranquil","serene","peaceful","quiet","gentle","mild","mellow","soothing","soft","subtle","understated","composed"],
    "warmth": ["warm","tender","affectionate","loving","caring","nurturing","supportive","kind","compassionate","empathetic","heartwarming","heartfelt","sweet","endearing","fond"],
    "contentment": ["content","pleasant","comfortable","cozy","homey","safe","welcoming","inviting","friendly","amicable","convivial","carefree"],
    "hope": ["hopeful","optimistic","encouraging","uplifting","inspiring","inspired","positive","resilient","determined"],
    "nostalgia": ["nostalgic","sentimental","wistful","bittersweet"],
    "romance": ["romantic","intimate","flirtatious","passionate"],
    "gratitude": ["grateful","appreciative","relieved","moved","touched","sincere"],
    "neutral": ["neutral","ordinary","mundane","everyday","routine","casual","simple","plain","blank","unremarkable","commonplace"],
    "focus": ["focused","attentive","observant","watchful","engaged","precise","careful","concentrated","absorbed","diligent","dedicated","purposeful","resolute"],
    "contemplation": ["pensive","thoughtful","contemplative","reflective","introspective","questioning","curious","intrigued","intriguing"],
    "seriousness": ["serious","solemn","formal","dignified","reserved","stoic","stately","commanding","authoritative"],
    "informative": ["informative","informational","professional","technical","clinical","factual","objective","direct","functional","practical"],
    "sadness": ["sad","somber","melancholic","melancholy","sorrowful","tearful","heartbreaking","tragic","dejected","unhappy","disappointed","gloomy","bleak"],
    "loneliness": ["lonely","solitary","isolated","desolate","abandoned","stranded","empty","detached","distant"],
    "weariness": ["tired","weary","exhausted","fatigued","sleepy","bored","apathetic","disengaged","subdued","muted"],
    "vulnerability": ["vulnerable","helpless","defeated","overwhelmed","struggling","desperate","pleading","pained"],
    "poignance": ["poignant","touching","emotional","moving","profound","raw","expressive"],
    "tension": ["tense","intense","suspenseful","anticipatory","expectant","anxious","uneasy","apprehensive","stressed","uncomfortable","uncertain","hesitant","cautious","wary"],
    "fear": ["scared","fearful","terrifying","terrified","frightening","alarming","horrific","eerie","creepy","unsettling","ominous","foreboding","menacing","threatening","perilous","dangerous"],
    "anger": ["angry","furious","enraged","aggressive","confrontational","defiant","rebellious","fierce","violent","brutal"],
    "chaos": ["chaotic","frantic","frenzied","rushed","overwhelming","explosive","destructive","catastrophic","apocalyptic"],
    "darkness": ["dark","grim","gritty","cold","stark","oppressive","haunting","gothic","murky"],
    "mystery": ["mysterious","enigmatic","surreal","ethereal","dreamlike","dreamy","otherworldly","mystical","fantastical","cosmic","abstract"],
    "elegance": ["elegant","luxurious","sophisticated","refined","glamorous","opulent","extravagant","sleek","prestigious","regal","graceful"],
    "domesticity": ["domestic","familial","communal","social","collaborative","inclusive","united","connected"],
    "nature": ["natural","organic","rustic","earthy","verdant","tropical","wintry","aquatic","idyllic"],
    "urban": ["urban","modern","futuristic","digital","industrial","corporate","busy","crowded"],
    "tradition": ["traditional","historical","historic","cultural","ceremonial","classical","classic","vintage","ancient"],
}

# Build reverse map: raw_mood -> category
REVERSE_MOOD = {}
for cat, words in MOOD_MAP.items():
    for w in words:
        REVERSE_MOOD[w.lower()] = cat


def extract_category(video_name):
    """Extract video category from video_name."""
    if video_name.startswith("gacs_"):
        parts = video_name.split("_")
        for j, p in enumerate(parts):
            if len(p) >= 2 and p[:2].isdigit():
                return "_".join(parts[1:j])
        return "_".join(parts[1:-1])
    return "other"


def parse_mood_list(mood_json_str):
    """Parse mood_json string into a list of strings."""
    try:
        return ast.literal_eval(mood_json_str)
    except Exception:
        return []


def get_dominant_mood(mood_list):
    """Normalize mood words and return most frequent category."""
    cats = []
    for w in mood_list:
        w_low = w.strip().lower()
        if w_low in REVERSE_MOOD:
            cats.append(REVERSE_MOOD[w_low])
    if not cats:
        return "unknown"
    # Most frequent
    from collections import Counter
    return Counter(cats).most_common(1)[0][0]


# ═══════════════════════════════════════════════════════════════
# PART A: Coordinate Mapping
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PART A: Coordinate Mapping")
print("=" * 60)

# 1. Load data
emb = np.load(os.path.join(ROOT, "data/step3/emb_emotion_distil_norm.npy"))
scene_idx = pd.read_csv(os.path.join(ROOT, "data/step2/scene_index.csv"))
labeled = pd.read_csv(os.path.join(ROOT, "gacs0202(kushi)/gacs_labeled_scenes.csv"))

print(f"Embeddings shape: {emb.shape}")
print(f"Scene index rows: {len(scene_idx)}")
print(f"Labeled scenes rows: {len(labeled)}")

# StandardScaler + PCA
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(emb)

# Determine N by cumvar >= 80%
pca_full = PCA(n_components=min(emb.shape))
pca_full.fit(emb_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
N_axes = int(np.argmax(cumvar >= 0.80) + 1)
print(f"PCA axes for 80% variance: N = {N_axes}")
print(f"Cumulative variance at N={N_axes}: {cumvar[N_axes-1]:.4f}")

# Final PCA
pca = PCA(n_components=N_axes)
coords = pca.fit_transform(emb_scaled)
var_explained = pca.explained_variance_ratio_
print(f"Variance explained per axis (first 5): {[f'{v:.4f}' for v in var_explained[:5]]}")

# 2. Build coordinate DataFrame
pc_cols = [f"PC{i+1}" for i in range(N_axes)]
df_coords = pd.DataFrame(coords, columns=pc_cols)
df_coords["scene_id"] = scene_idx["scene_id"].values
df_coords["video_name"] = scene_idx["video_name"].values

# Category
df_coords["category"] = df_coords["video_name"].apply(extract_category)

# Dominant mood
mood_lists = labeled["mood_json"].apply(parse_mood_list)
df_coords["dominant_mood"] = mood_lists.apply(get_dominant_mood)

# 3. KMeans silhouette sweep
print("\nKMeans silhouette sweep k=3..15:")
best_k, best_sil = 3, -1
sil_scores = {}
for k in range(3, 16):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(coords)
    s = silhouette_score(coords, labels)
    sil_scores[k] = round(s, 4)
    if s > best_sil:
        best_sil = s
        best_k = k
    print(f"  k={k}: silhouette={s:.4f}")

print(f"Best k={best_k}, silhouette={best_sil:.4f}")

km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df_coords["cluster_label"] = km_final.fit_predict(coords)

# Reorder columns
col_order = ["scene_id", "video_name", "category", "dominant_mood"] + pc_cols + ["cluster_label"]
df_coords = df_coords[col_order]

# Save
df_coords.to_csv(os.path.join(OUT, "gacs_coordinates.csv"), index=False)
print(f"\nSaved gacs_coordinates.csv ({len(df_coords)} rows, {len(col_order)} cols)")

# ═══════════════════════════════════════════════════════════════
# PART B: Validation
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: Validation")
print("=" * 60)

validation = {}

# 4. Intra-class vs inter-class distance
print("\n--- Intra-class vs Inter-class Distance (35 mood categories) ---")
mood_groups = df_coords.groupby("dominant_mood")
mood_cats_present = [m for m in mood_groups.groups if m != "unknown"]

intra_inter = {}
for mood in sorted(mood_cats_present):
    idx_in = df_coords[df_coords["dominant_mood"] == mood].index.values
    idx_out = df_coords[df_coords["dominant_mood"] != mood].index.values

    if len(idx_in) < 2:
        continue

    coords_in = coords[idx_in]
    coords_out = coords[idx_out]

    # Intra-class: mean pairwise distance among in-class
    intra_dists = pdist(coords_in, metric="euclidean")
    intra_mean = float(np.mean(intra_dists))

    # Inter-class: mean distance between in-class and out-class
    inter_dists = cdist(coords_in, coords_out, metric="euclidean")
    inter_mean = float(np.mean(inter_dists))

    ratio = inter_mean / intra_mean if intra_mean > 0 else 0.0
    intra_inter[mood] = {
        "n_scenes": int(len(idx_in)),
        "intra_mean": round(intra_mean, 4),
        "inter_mean": round(inter_mean, 4),
        "ratio": round(ratio, 4),
    }
    print(f"  {mood:20s} n={len(idx_in):4d}  intra={intra_mean:.4f}  inter={inter_mean:.4f}  ratio={ratio:.4f}")

overall_ratios = [v["ratio"] for v in intra_inter.values()]
overall_avg_ratio = round(float(np.mean(overall_ratios)), 4)
print(f"\n  Overall average inter/intra ratio: {overall_avg_ratio}")

validation["intra_inter_distances"] = intra_inter
validation["overall_avg_ratio"] = overall_avg_ratio

# 5. Category (video type) analysis
print("\n--- Category Analysis (ANOVA / Kruskal-Wallis) ---")
categories = sorted(df_coords["category"].unique())
print(f"Categories: {categories}")

cat_stats = {}
for cat in categories:
    sub = df_coords[df_coords["category"] == cat][pc_cols].values
    means = [round(float(m), 4) for m in np.mean(sub, axis=0)]
    stds = [round(float(s), 4) for s in np.std(sub, axis=0)]
    cat_stats[cat] = {"n": int(len(sub)), "mean": means[:5], "std": stds[:5]}  # store first 5 for brevity
    print(f"  {cat:25s} n={len(sub):4d}  PC1_mean={means[0]:.4f}  PC2_mean={means[1]:.4f}")

# Kruskal-Wallis per axis
anova_results = {}
for i, pc in enumerate(pc_cols):
    groups = [df_coords[df_coords["category"] == cat][pc].values for cat in categories]
    # Filter out groups with < 2 samples
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        stat, pval = kruskal(*groups)
        anova_results[pc] = {"H_statistic": round(float(stat), 4), "p_value": round(float(pval), 4)}
        if i < 10:
            print(f"  {pc}: H={stat:.4f}, p={pval:.6f}")

validation["category_stats"] = cat_stats
validation["kruskal_wallis"] = anova_results

# 6. Cross-validation: mood prediction
print("\n--- Mood Prediction (5-fold CV, RandomForest) ---")
mood_counts = df_coords["dominant_mood"].value_counts()
print(f"Mood distribution (top 10):\n{mood_counts.head(10)}")

# Scenes with moods >= 20 samples; others → "other"
valid_moods = set(mood_counts[mood_counts >= 20].index)
y_raw = df_coords["dominant_mood"].apply(lambda m: m if m in valid_moods else "other").values
X = coords.copy()

print(f"Valid mood classes (>=20 samples): {sorted(valid_moods)}")
print(f"Total 'other' scenes: {sum(y_raw == 'other')}")

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X, y_raw, cv=skf)

acc = round(float(accuracy_score(y_raw, y_pred)), 4)
macro_f1 = round(float(f1_score(y_raw, y_pred, average="macro")), 4)
print(f"\n  Overall accuracy: {acc}")
print(f"  Macro-F1: {macro_f1}")

cls_report = classification_report(y_raw, y_pred, output_dict=True, zero_division=0)
per_class_f1 = {}
for cls_name in sorted(set(y_raw)):
    if cls_name in cls_report:
        per_class_f1[cls_name] = round(cls_report[cls_name]["f1-score"], 4)
        print(f"  {cls_name:20s} F1={per_class_f1[cls_name]:.4f}  support={cls_report[cls_name]['support']}")

validation["classification"] = {
    "accuracy": acc,
    "macro_f1": macro_f1,
    "per_class_f1": per_class_f1,
}

# Confusion matrix
labels_sorted = sorted(set(y_raw))
cm = confusion_matrix(y_raw, y_pred, labels=labels_sorted)

# ═══════════════════════════════════════════════════════════════
# PART C: Expectancy Violation Distance
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART C: Expectancy Violation Distance")
print("=" * 60)

# 7. EV distance = weighted Euclidean (weight = variance explained by each PC)
weights = var_explained  # shape (N_axes,)
sample_coords = coords[:100]

# 8. Weighted Euclidean distance matrix
def weighted_euclidean(u, v, w):
    return np.sqrt(np.sum(w * (u - v) ** 2))

n_sample = len(sample_coords)
ev_matrix = np.zeros((n_sample, n_sample))
for i in range(n_sample):
    for j in range(i + 1, n_sample):
        d = weighted_euclidean(sample_coords[i], sample_coords[j], weights)
        ev_matrix[i, j] = d
        ev_matrix[j, i] = d

# 9. Example pairs
triu_idx = np.triu_indices(n_sample, k=1)
triu_dists = ev_matrix[triu_idx]
sorted_pair_idx = np.argsort(triu_dists)

def pair_info(rank_idx):
    idx = sorted_pair_idx[rank_idx]
    i, j = triu_idx[0][idx], triu_idx[1][idx]
    d = triu_dists[idx]
    return {
        "scene_i": int(df_coords.iloc[i]["scene_id"]),
        "scene_j": int(df_coords.iloc[j]["scene_id"]),
        "video_i": df_coords.iloc[i]["video_name"],
        "video_j": df_coords.iloc[j]["video_name"],
        "mood_i": df_coords.iloc[i]["dominant_mood"],
        "mood_j": df_coords.iloc[j]["dominant_mood"],
        "ev_distance": round(float(d), 4),
    }

n_pairs = len(sorted_pair_idx)
mid = n_pairs // 2
example_pairs = {
    "most_similar": pair_info(0),
    "mid_range_1": pair_info(mid - mid // 3),
    "mid_range_2": pair_info(mid),
    "mid_range_3": pair_info(mid + mid // 3),
    "most_different": pair_info(n_pairs - 1),
}

print("\n5 Example Pairs:")
for label, info in example_pairs.items():
    print(f"  {label}: scenes ({info['scene_i']}, {info['scene_j']}) "
          f"moods=({info['mood_i']}, {info['mood_j']}) "
          f"EV_dist={info['ev_distance']:.4f}")

validation["ev_distance"] = {
    "sample_size": n_sample,
    "mean_ev_dist": round(float(np.mean(triu_dists)), 4),
    "std_ev_dist": round(float(np.std(triu_dists)), 4),
    "min_ev_dist": round(float(np.min(triu_dists)), 4),
    "max_ev_dist": round(float(np.max(triu_dists)), 4),
    "example_pairs": example_pairs,
}

# ═══════════════════════════════════════════════════════════════
# Save validation JSON
# ═══════════════════════════════════════════════════════════════
with open(os.path.join(OUT, "coordinate_validation.json"), "w") as f:
    json.dump(validation, f, indent=2)
print("\nSaved coordinate_validation.json")

# ═══════════════════════════════════════════════════════════════
# Plots: 2x2
# ═══════════════════════════════════════════════════════════════
print("\nGenerating validation_plots.png ...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# (1) Confusion matrix
ax = axes[0, 0]
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Confusion Matrix (5-fold CV)", fontsize=12)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
tick_marks = np.arange(len(labels_sorted))
ax.set_xticks(tick_marks)
ax.set_xticklabels(labels_sorted, rotation=90, fontsize=7)
ax.set_yticks(tick_marks)
ax.set_yticklabels(labels_sorted, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (2) Intra vs inter distance bar chart
ax = axes[0, 1]
moods_plot = sorted(intra_inter.keys())
intra_vals = [intra_inter[m]["intra_mean"] for m in moods_plot]
inter_vals = [intra_inter[m]["inter_mean"] for m in moods_plot]
x_pos = np.arange(len(moods_plot))
width = 0.35
ax.bar(x_pos - width / 2, intra_vals, width, label="Intra-class", color="steelblue")
ax.bar(x_pos + width / 2, inter_vals, width, label="Inter-class", color="coral")
ax.set_xticks(x_pos)
ax.set_xticklabels(moods_plot, rotation=90, fontsize=7)
ax.set_ylabel("Mean Euclidean Distance")
ax.set_title("Intra vs Inter-class Distance by Mood", fontsize=12)
ax.legend(fontsize=9)

# (3) Category means per PC1/PC2
ax = axes[1, 0]
colors_cat = plt.cm.Set2(np.linspace(0, 1, len(categories)))
for ci, cat in enumerate(categories):
    sub = df_coords[df_coords["category"] == cat]
    ax.scatter(sub["PC1"], sub["PC2"], alpha=0.3, s=10, color=colors_cat[ci], label=cat)
    mean1 = sub["PC1"].mean()
    mean2 = sub["PC2"].mean()
    ax.scatter(mean1, mean2, s=120, color=colors_cat[ci], edgecolors="black", linewidths=1.5, marker="D", zorder=5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Category Means in PC1/PC2 Space", fontsize=12)
ax.legend(fontsize=8, loc="best")

# (4) EV distance distribution histogram
ax = axes[1, 1]
ax.hist(triu_dists, bins=60, color="mediumpurple", edgecolor="black", alpha=0.8)
ax.axvline(np.mean(triu_dists), color="red", linestyle="--", label=f"Mean={np.mean(triu_dists):.4f}")
ax.set_xlabel("EV Distance (weighted Euclidean)")
ax.set_ylabel("Frequency")
ax.set_title("EV Distance Distribution (first 100 scenes)", fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "validation_plots.png"), dpi=150)
plt.close()
print("Saved validation_plots.png")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  PCA dimensions (80% var): {N_axes}")
print(f"  Best KMeans k: {best_k} (silhouette={best_sil:.4f})")
print(f"  Overall inter/intra ratio: {overall_avg_ratio}")
print(f"  Classification accuracy: {acc}")
print(f"  Macro-F1: {macro_f1}")
print(f"  EV distance mean: {validation['ev_distance']['mean_ev_dist']}")
print(f"  EV distance range: [{validation['ev_distance']['min_ev_dist']}, {validation['ev_distance']['max_ev_dist']}]")
print(f"\nOutputs in {OUT}/:")
print("  - gacs_coordinates.csv")
print("  - coordinate_validation.json")
print("  - validation_plots.png")
print("Done.")
