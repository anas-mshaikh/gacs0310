"""
track1_cluster_subdivision.py
Subdivide large Cluster 3 (~64% of scenes) into meaningful sub-clusters
and create a hierarchical cluster scheme.
"""

import ast
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

# ── Paths ──────────────────────────────────────────────────────────────
EMB_PATH   = "data/step3/emb_emotion_distil_norm.npy"
COORD_PATH = "research/gacs_coordinates.csv"
LABEL_PATH = "gacs0202(kushi)/gacs_labeled_scenes.csv"
OUT_JSON   = "research/cluster_subdivision.json"
OUT_PNG    = "research/cluster_subdivision.png"
OUT_CSV    = "research/gacs_coordinates_v2.csv"

# ── MOOD_MAP ───────────────────────────────────────────────────────────
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

# Build reverse map: word -> canonical mood
WORD_TO_MOOD = {}
for mood, words in MOOD_MAP.items():
    for w in words:
        WORD_TO_MOOD[w.lower()] = mood
    WORD_TO_MOOD[mood.lower()] = mood  # map canonical to itself


def parse_json_col(val):
    """Parse a JSON-like column value into a list of strings."""
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed]
    except Exception:
        pass
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed]
    except Exception:
        pass
    return [x.strip().lower() for x in str(val).split(",") if x.strip()]


def normalize_moods(mood_list):
    """Map raw mood words to canonical mood categories."""
    return [WORD_TO_MOOD.get(w, w) for w in mood_list]


def top_n_from_counter(counter, n=10):
    """Return top-n items from a Counter as list of (word, count)."""
    return [(w, c) for w, c in counter.most_common(n)]


# ── 1. Load data ──────────────────────────────────────────────────────
print("=" * 70)
print("TRACK 1: CLUSTER 3 SUBDIVISION")
print("=" * 70)

emb = np.load(EMB_PATH)
coords = pd.read_csv(COORD_PATH)
labels_df = pd.read_csv(LABEL_PATH)

print(f"\nEmbeddings shape: {emb.shape}")
print(f"Coordinates shape: {coords.shape}")
print(f"Labels shape: {labels_df.shape}")

# ── 2. Identify Cluster 3 ────────────────────────────────────────────
c3_mask = coords["cluster_label"] == 3
c3_idx = coords.index[c3_mask].values
n_c3 = len(c3_idx)
print(f"\nCluster 3: {n_c3} scenes ({n_c3/len(coords)*100:.1f}% of total)")

emb_c3 = emb[c3_idx]
print(f"Cluster 3 embeddings shape: {emb_c3.shape}")

# ── 3. KMeans sweep on Cluster 3 ─────────────────────────────────────
print("\n--- Sub-cluster sweep (k=2..8) ---")
k_range = range(2, 9)
sil_scores = []
ch_scores = []
db_scores = []
km_models = {}

for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    sub_labels = km.fit_predict(emb_c3)
    sil = silhouette_score(emb_c3, sub_labels)
    ch = calinski_harabasz_score(emb_c3, sub_labels)
    db = davies_bouldin_score(emb_c3, sub_labels)
    sil_scores.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)
    km_models[k] = (km, sub_labels)
    print(f"  k={k}: Silhouette={sil:.4f}  CH={ch:.4f}  DB={db:.4f}")

# Pick optimal: best silhouette
best_k = list(k_range)[np.argmax(sil_scores)]
best_sil = max(sil_scores)
print(f"\nOptimal sub-k = {best_k} (silhouette = {best_sil:.4f})")

best_km, best_sub_labels = km_models[best_k]

# ── 4. Profile each sub-cluster ──────────────────────────────────────
print(f"\n--- Sub-cluster profiles (k={best_k}) ---")

# Gather label data for cluster 3 scenes
c3_labels_df = labels_df.iloc[c3_idx].copy()
c3_labels_df["sub_cluster"] = best_sub_labels
c3_coords = coords.iloc[c3_idx].copy()
c3_coords["sub_cluster"] = best_sub_labels

profiles = {}
for sc in range(best_k):
    sc_mask = c3_labels_df["sub_cluster"] == sc
    sc_df = c3_labels_df[sc_mask]
    sc_coords = c3_coords[sc_mask]
    n_sc = len(sc_df)

    # Moods
    mood_counter = Counter()
    for val in sc_df["mood_json"]:
        raw = parse_json_col(val)
        normalized = normalize_moods(raw)
        mood_counter.update(normalized)
    top_moods = top_n_from_counter(mood_counter, 10)

    # Styles
    style_counter = Counter()
    for val in sc_df["style_json"]:
        style_counter.update(parse_json_col(val))
    top_styles = top_n_from_counter(style_counter, 5)

    # Objects
    obj_counter = Counter()
    for val in sc_df["objects_json"]:
        obj_counter.update(parse_json_col(val))
    top_objects = top_n_from_counter(obj_counter, 5)

    # Category distribution
    cat_dist = sc_coords["category"].value_counts().to_dict()

    # Mean PC1 (Valence) and PC2 (Arousal)
    mean_pc1 = float(sc_coords["PC1"].mean())
    mean_pc2 = float(sc_coords["PC2"].mean())

    profiles[f"C3{chr(ord('a') + sc)}"] = {
        "sub_cluster_id": sc,
        "n_scenes": int(n_sc),
        "pct_of_c3": round(n_sc / n_c3 * 100, 4),
        "mean_PC1_valence": round(mean_pc1, 4),
        "mean_PC2_arousal": round(mean_pc2, 4),
        "top_moods": [(m, int(c)) for m, c in top_moods],
        "top_styles": [(s, int(c)) for s, c in top_styles],
        "top_objects": [(o, int(c)) for o, c in top_objects],
        "category_distribution": {k: int(v) for k, v in cat_dist.items()},
    }

    print(f"\n  Sub-cluster C3{chr(ord('a') + sc)}: {n_sc} scenes ({n_sc/n_c3*100:.1f}%)")
    print(f"    Mean PC1 (Valence): {mean_pc1:.4f},  Mean PC2 (Arousal): {mean_pc2:.4f}")
    print(f"    Top moods: {', '.join(m for m, _ in top_moods[:5])}")
    print(f"    Top styles: {', '.join(s for s, _ in top_styles[:3])}")
    print(f"    Top objects: {', '.join(o for o, _ in top_objects[:3])}")
    cats_str = ", ".join(f"{k}:{v}" for k, v in sorted(cat_dist.items(), key=lambda x: -x[1])[:4])
    print(f"    Categories: {cats_str}")

# ── 5. Hierarchical cluster labels ───────────────────────────────────
print("\n--- Building hierarchical cluster scheme ---")

# Create new label column
coords["cluster_v2"] = ""
for idx in coords.index:
    orig = coords.loc[idx, "cluster_label"]
    if orig != 3:
        coords.loc[idx, "cluster_v2"] = f"C{orig}"
    else:
        # Find position in c3_idx
        pos = np.where(c3_idx == idx)[0][0]
        sc = best_sub_labels[pos]
        coords.loc[idx, "cluster_v2"] = f"C3{chr(ord('a') + sc)}"

label_dist = coords["cluster_v2"].value_counts().sort_index()
print("\nHierarchical label distribution:")
for lbl, cnt in label_dist.items():
    print(f"  {lbl}: {cnt} scenes ({cnt/len(coords)*100:.1f}%)")

total_clusters = len(label_dist)
print(f"\nTotal clusters: {total_clusters} (3 original + {best_k} sub-clusters)")

# ── 6. Full-dataset silhouette with hierarchical labels ──────────────
print("\n--- Full-dataset silhouette with hierarchical labels ---")

# Encode cluster_v2 as integers for silhouette
label_map = {lbl: i for i, lbl in enumerate(sorted(coords["cluster_v2"].unique()))}
hier_int_labels = coords["cluster_v2"].map(label_map).values

sil_hier = silhouette_score(emb, hier_int_labels)
ch_hier = calinski_harabasz_score(emb, hier_int_labels)
db_hier = davies_bouldin_score(emb, hier_int_labels)

# Compare with original 4-cluster
sil_orig = silhouette_score(emb, coords["cluster_label"].values)
ch_orig = calinski_harabasz_score(emb, coords["cluster_label"].values)
db_orig = davies_bouldin_score(emb, coords["cluster_label"].values)

print(f"  Original (k=4):     Silhouette={sil_orig:.4f}  CH={ch_orig:.4f}  DB={db_orig:.4f}")
print(f"  Hierarchical:       Silhouette={sil_hier:.4f}  CH={ch_hier:.4f}  DB={db_hier:.4f}")

# ── 7. Save JSON ─────────────────────────────────────────────────────
output = {
    "subdivision_target": "Cluster 3",
    "cluster3_n_scenes": int(n_c3),
    "cluster3_pct": round(n_c3 / len(coords) * 100, 4),
    "sweep_results": {
        str(k): {
            "silhouette": round(sil_scores[i], 4),
            "calinski_harabasz": round(ch_scores[i], 4),
            "davies_bouldin": round(db_scores[i], 4),
        }
        for i, k in enumerate(k_range)
    },
    "optimal_sub_k": best_k,
    "optimal_silhouette": round(best_sil, 4),
    "sub_cluster_profiles": profiles,
    "hierarchical_scheme": {
        "level_1": "Original KMeans k=4 (C0, C1, C2, C3)",
        "level_2": f"Cluster 3 subdivided into {best_k} sub-clusters",
        "total_clusters": total_clusters,
        "label_distribution": {k: int(v) for k, v in label_dist.items()},
    },
    "full_dataset_metrics": {
        "original_k4": {
            "silhouette": round(sil_orig, 4),
            "calinski_harabasz": round(ch_orig, 4),
            "davies_bouldin": round(db_orig, 4),
        },
        "hierarchical": {
            "silhouette": round(sil_hier, 4),
            "calinski_harabasz": round(ch_hier, 4),
            "davies_bouldin": round(db_hier, 4),
        },
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT_JSON}")

# ── 8. Save updated coordinates ──────────────────────────────────────
coords.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")

# ── 9. Visualization ─────────────────────────────────────────────────
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("Cluster 3 Subdivision Analysis", fontsize=16, fontweight="bold", y=0.98)

# Colors for hierarchical clusters
hier_labels_sorted = sorted(coords["cluster_v2"].unique())
cmap = plt.cm.tab10
color_map = {lbl: cmap(i) for i, lbl in enumerate(hier_labels_sorted)}

# (1) Sub-cluster silhouette sweep
ax1 = axes[0, 0]
ax1.plot(list(k_range), sil_scores, "o-", color="steelblue", linewidth=2, markersize=8)
ax1.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Optimal k={best_k}")
ax1.set_xlabel("Number of sub-clusters (k)")
ax1.set_ylabel("Silhouette Score")
ax1.set_title("Cluster 3 Sub-cluster Silhouette Sweep")
ax1.legend()
ax1.grid(True, alpha=0.3)
for i, k in enumerate(k_range):
    ax1.annotate(f"{sil_scores[i]:.3f}", (k, sil_scores[i]),
                 textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

# (2) 2D scatter PC1/PC2 with hierarchical labels
ax2 = axes[0, 1]
for lbl in hier_labels_sorted:
    mask = coords["cluster_v2"] == lbl
    ax2.scatter(coords.loc[mask, "PC1"], coords.loc[mask, "PC2"],
                c=[color_map[lbl]], label=lbl, alpha=0.5, s=15, edgecolors="none")
ax2.set_xlabel("PC1 (Valence)")
ax2.set_ylabel("PC2 (Arousal)")
ax2.set_title("Hierarchical Clusters — PC1 vs PC2")
ax2.legend(fontsize=8, markerscale=2, loc="best")
ax2.grid(True, alpha=0.3)

# (3) Sub-cluster mood distribution stacked bar
ax3 = axes[1, 0]
# Collect top moods across all sub-clusters
all_top_moods = set()
for p in profiles.values():
    for m, _ in p["top_moods"][:7]:
        all_top_moods.add(m)
all_top_moods = sorted(all_top_moods)

sub_labels_list = sorted(profiles.keys())
mood_matrix = []
for mood in all_top_moods:
    row = []
    for lbl in sub_labels_list:
        total = sum(c for _, c in profiles[lbl]["top_moods"])
        count = 0
        for m, c in profiles[lbl]["top_moods"]:
            if m == mood:
                count = c
                break
        row.append(count / total * 100 if total > 0 else 0)
    mood_matrix.append(row)

x = np.arange(len(sub_labels_list))
bottom = np.zeros(len(sub_labels_list))
mood_cmap = plt.cm.Set3
for i, mood in enumerate(all_top_moods):
    vals = mood_matrix[i]
    ax3.bar(x, vals, bottom=bottom, label=mood, color=mood_cmap(i / max(len(all_top_moods) - 1, 1)))
    bottom += np.array(vals)

ax3.set_xticks(x)
ax3.set_xticklabels(sub_labels_list)
ax3.set_ylabel("Mood share (%)")
ax3.set_title("Sub-cluster Mood Distribution")
ax3.legend(fontsize=6, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.0))

# (4) Full dataset with hierarchical labels — PC1 vs PC2
ax4 = axes[1, 1]
for lbl in hier_labels_sorted:
    mask = coords["cluster_v2"] == lbl
    n_lbl = mask.sum()
    ax4.scatter(coords.loc[mask, "PC1"], coords.loc[mask, "PC2"],
                c=[color_map[lbl]], label=f"{lbl} (n={n_lbl})", alpha=0.5, s=15, edgecolors="none")
ax4.set_xlabel("PC1 (Valence)")
ax4.set_ylabel("PC2 (Arousal)")
ax4.set_title(f"Full Dataset — Hierarchical Labels (Sil={sil_hier:.4f})")
ax4.legend(fontsize=7, markerscale=2, loc="best")
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PNG}")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Cluster 3: {n_c3} scenes ({n_c3/len(coords)*100:.1f}% of dataset)")
print(f"Optimal sub-clusters: {best_k} (silhouette={best_sil:.4f})")
print(f"\nHierarchical scheme ({total_clusters} total clusters):")
for lbl in hier_labels_sorted:
    n = (coords["cluster_v2"] == lbl).sum()
    pct = n / len(coords) * 100
    print(f"  {lbl}: {n:>4d} scenes ({pct:>5.1f}%)")
print("\nFull-dataset quality comparison:")
print(f"  Original k=4:  Sil={sil_orig:.4f}  CH={ch_orig:.4f}  DB={db_orig:.4f}")
print(f"  Hierarchical:  Sil={sil_hier:.4f}  CH={ch_hier:.4f}  DB={db_hier:.4f}")
print("\nOutputs:")
print(f"  {OUT_JSON}")
print(f"  {OUT_PNG}")
print(f"  {OUT_CSV}")
print("=" * 70)
