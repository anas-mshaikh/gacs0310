"""
research_1_clustering.py
Compare clustering algorithms on emotion embeddings.
"""
import ast
import json
import os
import warnings
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("WARNING: hdbscan not installed, skipping HDBSCAN.")

warnings.filterwarnings('ignore')

BASE = '/home/jongcircle/dev/gacs0202'
OUT = os.path.join(BASE, 'research')
os.makedirs(OUT, exist_ok=True)

# ── Load data ──
emb = np.load(os.path.join(BASE, 'data/step3/emb_emotion_distil_norm.npy'))
df = pd.read_csv(os.path.join(BASE, 'gacs0202(kushi)/gacs_labeled_scenes.csv'))
print(f"Embeddings: {emb.shape}, Labels: {df.shape}")

scaler = StandardScaler()
X = scaler.fit_transform(emb)

# ── Helper: compute metrics ──
def compute_metrics(X, labels):
    """Compute silhouette, calinski-harabasz, davies-bouldin. Returns None if < 2 clusters."""
    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return None
    mask = np.array(labels) != -1
    if mask.sum() < len(unique) + 1:
        return None
    try:
        sil = round(silhouette_score(X[mask], np.array(labels)[mask]), 4)
        ch = round(calinski_harabasz_score(X[mask], np.array(labels)[mask]), 4)
        db = round(davies_bouldin_score(X[mask], np.array(labels)[mask]), 4)
        return {'silhouette': sil, 'calinski_harabasz': ch, 'davies_bouldin': db}
    except Exception:
        return None

# ── Run algorithms ──
all_results = []

# KMeans
print("\n--- KMeans ---")
kmeans_sils = []
kmeans_inertias = []
kmeans_ks = list(range(3, 26))
for k in kmeans_ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    m = compute_metrics(X, labels)
    kmeans_inertias.append(float(km.inertia_))
    if m:
        kmeans_sils.append(m['silhouette'])
        all_results.append({
            'algorithm': 'KMeans', 'params': {'k': k},
            **m, 'n_clusters': k, 'noise_points': 0
        })
        print(f"  k={k:2d}  sil={m['silhouette']:.4f}  CH={m['calinski_harabasz']:.4f}  DB={m['davies_bouldin']:.4f}")
    else:
        kmeans_sils.append(None)

# DBSCAN
print("\n--- DBSCAN ---")
eps_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
# Also try larger eps for high-dimensional data
eps_values_extra = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0]
for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int(np.sum(labels == -1))
    m = compute_metrics(X, labels)
    if m:
        all_results.append({
            'algorithm': 'DBSCAN', 'params': {'eps': eps, 'min_samples': 5},
            **m, 'n_clusters': n_clusters, 'noise_points': noise
        })
        print(f"  eps={eps:.1f}  clusters={n_clusters}  noise={noise}  sil={m['silhouette']:.4f}  CH={m['calinski_harabasz']:.4f}  DB={m['davies_bouldin']:.4f}")
    else:
        print(f"  eps={eps:.1f}  clusters={n_clusters}  noise={noise}  (not enough clusters)")

for eps in eps_values_extra:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int(np.sum(labels == -1))
    m = compute_metrics(X, labels)
    if m:
        all_results.append({
            'algorithm': 'DBSCAN', 'params': {'eps': eps, 'min_samples': 5},
            **m, 'n_clusters': n_clusters, 'noise_points': noise
        })
        print(f"  eps={eps:.1f}  clusters={n_clusters}  noise={noise}  sil={m['silhouette']:.4f}  CH={m['calinski_harabasz']:.4f}  DB={m['davies_bouldin']:.4f}")
    else:
        print(f"  eps={eps:.1f}  clusters={n_clusters}  noise={noise}  (not enough clusters)")

# HDBSCAN
if HAS_HDBSCAN:
    print("\n--- HDBSCAN ---")
    for mcs in [5, 10, 15, 20, 30, 50]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int(np.sum(labels == -1))
        m = compute_metrics(X, labels)
        if m:
            all_results.append({
                'algorithm': 'HDBSCAN', 'params': {'min_cluster_size': mcs},
                **m, 'n_clusters': n_clusters, 'noise_points': noise
            })
            print(f"  min_cluster_size={mcs:2d}  clusters={n_clusters}  noise={noise}  sil={m['silhouette']:.4f}  CH={m['calinski_harabasz']:.4f}  DB={m['davies_bouldin']:.4f}")
        else:
            print(f"  min_cluster_size={mcs:2d}  clusters={n_clusters}  noise={noise}  (not enough clusters)")
else:
    print("\n--- HDBSCAN skipped (not installed) ---")

# GaussianMixture
print("\n--- GaussianMixture ---")
gmm_sils = []
gmm_ks = list(range(3, 16))
for k in gmm_ks:
    gm = GaussianMixture(n_components=k, random_state=42, n_init=3,
                         reg_covar=1e-3, covariance_type='diag')
    try:
        labels = gm.fit_predict(X)
    except Exception as e:
        print(f"  k={k:2d}  FAILED: {e}")
        gmm_sils.append(None)
        continue
    m = compute_metrics(X, labels)
    if m:
        gmm_sils.append(m['silhouette'])
        all_results.append({
            'algorithm': 'GaussianMixture', 'params': {'n_components': k},
            **m, 'n_clusters': k, 'noise_points': 0
        })
        print(f"  k={k:2d}  sil={m['silhouette']:.4f}  CH={m['calinski_harabasz']:.4f}  DB={m['davies_bouldin']:.4f}")
    else:
        gmm_sils.append(None)

# ── Sort by silhouette ──
all_results.sort(key=lambda x: x['silhouette'], reverse=True)

# Save comparison JSON
with open(os.path.join(OUT, 'clustering_comparison.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved {len(all_results)} results to research/clustering_comparison.json")

# ── Plot 2x2 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Clustering Algorithm Comparison', fontsize=14, fontweight='bold')

# (1) KMeans silhouette vs k
ax = axes[0, 0]
ax.plot(kmeans_ks, kmeans_sils, 'bo-', markersize=4)
ax.set_xlabel('k')
ax.set_ylabel('Silhouette Score')
ax.set_title('KMeans: Silhouette vs k')
ax.grid(True, alpha=0.3)

# (2) KMeans elbow
ax = axes[0, 1]
ax.plot(kmeans_ks, kmeans_inertias, 'ro-', markersize=4)
ax.set_xlabel('k')
ax.set_ylabel('Inertia')
ax.set_title('KMeans: Elbow Plot')
ax.grid(True, alpha=0.3)

# (3) GMM silhouette vs k
ax = axes[1, 0]
ax.plot(gmm_ks, gmm_sils, 'go-', markersize=4)
ax.set_xlabel('n_components')
ax.set_ylabel('Silhouette Score')
ax.set_title('GaussianMixture: Silhouette vs n_components')
ax.grid(True, alpha=0.3)

# (4) Best configs bar chart - top 10
ax = axes[1, 1]
top_n = min(10, len(all_results))
top = all_results[:top_n]
labels_bar = [f"{r['algorithm']}\n{r['params']}" for r in top]
# Shorten labels
short_labels = []
for r in top:
    algo = r['algorithm'][:6]
    p = str(r['params']).replace("'", "").replace("{", "").replace("}", "")
    if len(p) > 20:
        p = p[:20]
    short_labels.append(f"{algo}\n{p}")
sils_bar = [r['silhouette'] for r in top]
colors = {'KMeans': '#4C72B0', 'DBSCAN': '#DD8452', 'HDBSCAN': '#55A868', 'GaussianMixture': '#C44E52'}
bar_colors = [colors.get(r['algorithm'], '#999999') for r in top]
ax.barh(range(top_n), sils_bar, color=bar_colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(short_labels, fontsize=7)
ax.set_xlabel('Silhouette Score')
ax.set_title('Top Configurations by Silhouette')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'clustering_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved research/clustering_comparison.png")

# ── Best config & cluster profiling ──
best = all_results[0]
print("\n=== BEST CONFIG ===")
print(f"Algorithm: {best['algorithm']}")
print(f"Params: {best['params']}")
print(f"Silhouette: {best['silhouette']}")
print(f"Calinski-Harabasz: {best['calinski_harabasz']}")
print(f"Davies-Bouldin: {best['davies_bouldin']}")

# Re-fit best model
algo = best['algorithm']
params = best['params']
if algo == 'KMeans':
    model = KMeans(n_clusters=params['k'], random_state=42, n_init=10)
    best_labels = model.fit_predict(X)
elif algo == 'DBSCAN':
    model = DBSCAN(eps=params['eps'], min_samples=params.get('min_samples', 5))
    best_labels = model.fit_predict(X)
elif algo == 'HDBSCAN':
    model = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'])
    best_labels = model.fit_predict(X)
elif algo == 'GaussianMixture':
    model = GaussianMixture(n_components=params['n_components'], random_state=42, n_init=3)
    best_labels = model.fit_predict(X)

df['cluster'] = best_labels

# Extract category from video_name
def extract_category(vname):
    vname = vname.replace('.mp4', '')
    if vname.startswith('gacs_'):
        parts = vname.split('_')
        # gacs_movie_trailers_00_xxx -> find the numeric part
        for i, p in enumerate(parts):
            if p.isdigit() and len(p) == 2:
                return '_'.join(parts[1:i])
        return '_'.join(parts[1:-1])
    else:
        return vname.rsplit('_', 1)[0] if '_' in vname else vname

df['category'] = df['video_name'].apply(extract_category)

def safe_parse(val):
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed]
        return [str(parsed).strip().lower()]
    except Exception:
        return []

# Profile clusters
cluster_profiles = []
unique_clusters = sorted([c for c in set(best_labels) if c != -1])
total = len(df)

print(f"\n{'='*80}")
print(f"{'Cluster':>8} {'Size':>6} {'Pct':>6}  {'Top Moods':<50}")
print(f"{'='*80}")

for cid in unique_clusters:
    mask = df['cluster'] == cid
    subset = df[mask]
    size = int(mask.sum())
    pct = round(size / total * 100, 4)

    # Top moods
    all_moods = []
    for val in subset['mood_json']:
        all_moods.extend(safe_parse(val))
    top_moods = Counter(all_moods).most_common(10)

    # Top styles
    all_styles = []
    for val in subset['style_json']:
        all_styles.extend(safe_parse(val))
    top_styles = Counter(all_styles).most_common(5)

    # Top objects
    all_objects = []
    for val in subset['objects_json']:
        all_objects.extend(safe_parse(val))
    top_objects = Counter(all_objects).most_common(5)

    # Category distribution
    cat_dist = subset['category'].value_counts().to_dict()
    cat_dist = {k: int(v) for k, v in cat_dist.items()}

    profile = {
        'cluster_id': int(cid),
        'size': size,
        'percentage': pct,
        'top_10_moods': [{'word': w, 'count': int(c)} for w, c in top_moods],
        'top_5_styles': [{'word': w, 'count': int(c)} for w, c in top_styles],
        'top_5_objects': [{'word': w, 'count': int(c)} for w, c in top_objects],
        'category_distribution': cat_dist
    }
    cluster_profiles.append(profile)

    mood_str = ', '.join([f"{w}({c})" for w, c in top_moods[:5]])
    print(f"{cid:>8} {size:>6} {pct:>5.1f}%  {mood_str}")

# Noise cluster if present
noise_count = int(np.sum(best_labels == -1))
if noise_count > 0:
    print(f"{'noise':>8} {noise_count:>6} {noise_count/total*100:>5.1f}%")

# Save optimal clusters JSON
optimal_output = {
    'best_algorithm': best['algorithm'],
    'best_params': best['params'],
    'metrics': {
        'silhouette': best['silhouette'],
        'calinski_harabasz': best['calinski_harabasz'],
        'davies_bouldin': best['davies_bouldin']
    },
    'n_clusters': int(best['n_clusters']),
    'noise_points': int(best.get('noise_points', 0)),
    'hdbscan_available': HAS_HDBSCAN,
    'cluster_profiles': cluster_profiles
}

with open(os.path.join(OUT, 'optimal_clusters.json'), 'w') as f:
    json.dump(optimal_output, f, indent=2)
print("\nSaved research/optimal_clusters.json")

# ── Final summary table ──
print(f"\n{'='*100}")
print(f"{'FINAL SUMMARY TABLE':^100}")
print(f"{'='*100}")
print(f"{'Algorithm':<18} {'Params':<35} {'Silhouette':>10} {'CH Index':>12} {'DB Index':>10} {'Clusters':>8}")
print(f"{'-'*100}")
for r in all_results[:20]:
    p = str(r['params']).replace("'", "").replace("{", "").replace("}", "")
    if len(p) > 33:
        p = p[:33]
    print(f"{r['algorithm']:<18} {p:<35} {r['silhouette']:>10.4f} {r['calinski_harabasz']:>12.4f} {r['davies_bouldin']:>10.4f} {r['n_clusters']:>8}")

print(f"\n*** BEST: {best['algorithm']} with {best['params']} ***")
print(f"    Silhouette={best['silhouette']:.4f}  CH={best['calinski_harabasz']:.4f}  DB={best['davies_bouldin']:.4f}")
print("Done.")
