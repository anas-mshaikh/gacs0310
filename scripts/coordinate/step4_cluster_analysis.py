"""
step4_cluster_analysis.py
Deep cluster analysis on emotion_distil_norm embeddings.
"""

import ast
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
EMB_PATH = ROOT / "data/step3/emb_emotion_distil_norm.npy"
LABELS_PATH = ROOT / "gacs0202(kushi)/gacs_labeled_scenes.csv"
OUT_DIR = ROOT / "data/step4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ────────────────────────────────────────────────────────
print("Loading embeddings and labels ...")
emb = np.load(EMB_PATH)
df = pd.read_csv(LABELS_PATH)
assert len(df) == emb.shape[0], f"Row mismatch: {len(df)} vs {emb.shape[0]}"
print(f"  embeddings: {emb.shape}  |  labels: {len(df)} rows")


# ── helper: parse stringified list column ────────────────────────────────
def parse_list_col(series):
    """Parse a column of stringified Python lists into actual lists."""
    out = []
    for val in series:
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                out.append([str(w).strip().lower() for w in parsed])
            else:
                out.append([str(parsed).strip().lower()])
        except Exception:
            out.append([])
    return out


mood_lists = parse_list_col(df["mood_json"])
style_lists = parse_list_col(df["style_json"])
obj_lists = parse_list_col(df["objects_json"])

# video category = video_name without extension
df["video_cat"] = df["video_name"].str.replace(r"\.\w+$", "", regex=True)

# ── 2. KMeans sweep k=3..20 ─────────────────────────────────────────────
K_RANGE = range(3, 21)
results = {
    "k": [],
    "inertia": [],
    "silhouette": [],
    "calinski_harabasz": [],
    "davies_bouldin": [],
}

print("Running KMeans sweep k=3..20 ...")
km_models = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(emb)
    km_models[k] = (km, labels)

    sil = silhouette_score(emb, labels)
    ch = calinski_harabasz_score(emb, labels)
    db = davies_bouldin_score(emb, labels)

    results["k"].append(k)
    results["inertia"].append(float(km.inertia_))
    results["silhouette"].append(float(sil))
    results["calinski_harabasz"].append(float(ch))
    results["davies_bouldin"].append(float(db))

    print(f"  k={k:2d}  sil={sil:.4f}  CH={ch:.1f}  DB={db:.4f}  inertia={km.inertia_:.1f}")

metrics_df = pd.DataFrame(results)

# optimal k = highest silhouette
best_idx = int(np.argmax(metrics_df["silhouette"]))
optimal_k = int(metrics_df["k"].iloc[best_idx])
print(f"\nOptimal k (max silhouette): {optimal_k}")

# ── 3. Elbow + silhouette plots ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(metrics_df["k"], metrics_df["inertia"], "o-", color="steelblue")
ax.axvline(optimal_k, ls="--", color="red", alpha=0.7, label=f"optimal k={optimal_k}")
ax.set_xlabel("k")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Plot (Inertia)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(metrics_df["k"], metrics_df["silhouette"], "o-", color="darkorange")
ax.axvline(optimal_k, ls="--", color="red", alpha=0.7, label=f"optimal k={optimal_k}")
ax.set_xlabel("k")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "cluster_metrics.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved metrics plot -> {plot_path}")


# ── 4. Cluster profile analysis ─────────────────────────────────────────
def build_profiles(k, km_labels):
    """Build cluster profiles for a given k."""
    profiles = []
    for c in range(k):
        mask = km_labels == c
        idx = np.where(mask)[0]
        size = int(mask.sum())

        # mood words
        mood_counter = Counter()
        for i in idx:
            mood_counter.update(mood_lists[i])
        top_moods = mood_counter.most_common(10)

        # style words
        style_counter = Counter()
        for i in idx:
            style_counter.update(style_lists[i])
        top_styles = style_counter.most_common(5)

        # object words
        obj_counter = Counter()
        for i in idx:
            obj_counter.update(obj_lists[i])
        top_objects = obj_counter.most_common(5)

        # video category distribution
        cat_counter = Counter(df["video_cat"].iloc[idx].tolist())
        top_cats = cat_counter.most_common(5)

        profile = {
            "cluster": c,
            "size": size,
            "pct": round(100 * size / len(df), 1),
            "top_moods": top_moods,
            "top_styles": top_styles,
            "top_objects": top_objects,
            "top_videos": top_cats,
        }
        profiles.append(profile)
    return profiles


def print_profiles(k, profiles):
    """Pretty-print cluster profiles."""
    print(f"\n{'='*80}")
    print(f"  CLUSTER PROFILES  (k={k})")
    print(f"{'='*80}")
    for p in profiles:
        print(f"\n--- Cluster {p['cluster']}  |  size={p['size']} ({p['pct']}%) ---")
        moods = ", ".join(f"{w}({n})" for w, n in p["top_moods"])
        styles = ", ".join(f"{w}({n})" for w, n in p["top_styles"])
        objs = ", ".join(f"{w}({n})" for w, n in p["top_objects"])
        vids = ", ".join(f"{v}({n})" for v, n in p["top_videos"])
        print(f"  Moods:   {moods}")
        print(f"  Styles:  {styles}")
        print(f"  Objects: {objs}")
        print(f"  Videos:  {vids}")


# Analyze optimal k
_, labels_opt = km_models[optimal_k]
profiles_opt = build_profiles(optimal_k, labels_opt)
print_profiles(optimal_k, profiles_opt)

# Analyze k=5 for comparison (unless optimal_k == 5)
compare_k = 5
if compare_k != optimal_k and compare_k in km_models:
    _, labels_5 = km_models[compare_k]
    profiles_5 = build_profiles(compare_k, labels_5)
    print_profiles(compare_k, profiles_5)
else:
    profiles_5 = None


# ── 5. Print summary table ──────────────────────────────────────────────
def profiles_to_table(profiles):
    rows = []
    for p in profiles:
        rows.append({
            "cluster": p["cluster"],
            "size": p["size"],
            "pct": p["pct"],
            "top_3_moods": ", ".join(w for w, _ in p["top_moods"][:3]),
            "top_3_styles": ", ".join(w for w, _ in p["top_styles"][:3]),
            "top_3_objects": ", ".join(w for w, _ in p["top_objects"][:3]),
            "top_video": p["top_videos"][0][0] if p["top_videos"] else "",
        })
    return pd.DataFrame(rows)


print(f"\n\n{'='*80}")
print("  SUMMARY TABLE  (optimal k={})".format(optimal_k))
print(f"{'='*80}")
tbl = profiles_to_table(profiles_opt)
print(tbl.to_string(index=False))

if profiles_5 is not None:
    print(f"\n{'='*80}")
    print("  SUMMARY TABLE  (k=5)")
    print(f"{'='*80}")
    tbl5 = profiles_to_table(profiles_5)
    print(tbl5.to_string(index=False))


# ── 6. Save outputs ─────────────────────────────────────────────────────
# JSON: full results
output_json = {
    "metrics": results,
    "optimal_k": optimal_k,
    "profiles_optimal": [
        {**p, "top_moods": [list(x) for x in p["top_moods"]],
              "top_styles": [list(x) for x in p["top_styles"]],
              "top_objects": [list(x) for x in p["top_objects"]],
              "top_videos": [list(x) for x in p["top_videos"]]}
        for p in profiles_opt
    ],
}
if profiles_5 is not None:
    output_json["profiles_k5"] = [
        {**p, "top_moods": [list(x) for x in p["top_moods"]],
              "top_styles": [list(x) for x in p["top_styles"]],
              "top_objects": [list(x) for x in p["top_objects"]],
              "top_videos": [list(x) for x in p["top_videos"]]}
        for p in profiles_5
    ]

json_path = OUT_DIR / "cluster_analysis.json"
with open(json_path, "w") as f:
    json.dump(output_json, f, indent=2)
print(f"\nSaved JSON -> {json_path}")

# CSV: cluster profiles for optimal k
csv_path = OUT_DIR / "cluster_profiles.csv"
tbl.to_csv(csv_path, index=False)
print(f"Saved CSV  -> {csv_path}")

print("\nDone.")
