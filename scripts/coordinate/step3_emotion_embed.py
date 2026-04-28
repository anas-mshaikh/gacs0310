"""
Step 3: Emotion-Specific Embedding + 3-Model Comparison
Models: (1) paraphrase-MiniLM-L3-v2 (Kushi), (2) all-MiniLM-L6-v2, (3) roberta-base-go_emotions
"""
import json
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

KUSHI = Path("gacs0202(kushi)")
STEP2 = Path("data/step2")
OUT = Path("data/step3")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(KUSHI / "gacs_labeled_scenes.csv")
texts = pd.read_csv(STEP2 / "texts.csv")
print(f"Loaded: {len(df)} rows")

# ─── Load Step 2 embeddings ───
emb_L6_orig = np.load(STEP2 / "emb_original_L6v2.npy")
emb_L6_norm = np.load(STEP2 / "emb_normalized_L6v2.npy")
print(f"L6 original: {emb_L6_orig.shape}")
print(f"L6 normalized: {emb_L6_norm.shape}")

# ─── Parse Kushi's original L3 embeddings ───
def parse_embedding(val):
    if pd.isna(val): return None
    s = str(val).strip().replace('[','').replace(']','').replace('\n',' ')
    try:
        nums = [float(x) for x in s.split() if x]
        if len(nums) == 384: return np.array(nums, dtype=np.float32)
    except: pass
    return None

df['emb_parsed'] = df['embedding'].apply(parse_embedding)
emb_L3 = np.stack(df['emb_parsed'].values)
print(f"L3 original: {emb_L3.shape}")

# ─── Compute go_emotions embedding ───
print("\nLoading SamLowe/roberta-base-go_emotions...")
# go_emotions은 28개 감정 분류 모델. 직접 사용은 classification이지만,
# sentence-transformers로 feature extraction하면 768-dim 감성 임베딩으로 사용 가능.
# 대안: j-hartmann/emotion-english-distilroberta-base (7 emotions)

# 여러 emotion 모델 시도
emotion_models = [
    ("j-hartmann/emotion-english-distilroberta-base", "emotion_distil"),
]

emotion_embeddings = {}
for model_name, short_name in emotion_models:
    try:
        print(f"\nLoading {model_name}...")
        model = SentenceTransformer(model_name, device='cuda')
        t0 = time.time()
        # 원본 텍스트로 임베딩
        emb = model.encode(texts['original'].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        print(f"  {short_name}: {emb.shape} in {time.time()-t0:.1f}s")
        emotion_embeddings[short_name] = emb
        np.save(OUT / f"emb_{short_name}.npy", emb)
        # normalized 텍스트로도
        emb_n = model.encode(texts['normalized'].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        emotion_embeddings[f"{short_name}_norm"] = emb_n
        np.save(OUT / f"emb_{short_name}_norm.npy", emb_n)
    except Exception as e:
        print(f"  FAILED: {e}")
        # Fallback: 다른 감성 모델
        try:
            alt = "sentence-transformers/all-mpnet-base-v2"
            print(f"  Trying fallback: {alt}")
            model = SentenceTransformer(alt, device='cuda')
            emb = model.encode(texts['original'].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            emotion_embeddings["mpnet_v2"] = emb
            np.save(OUT / "emb_mpnet_v2.npy", emb)
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")

# ─── Compare All Models ───
print("\n\n=== MODEL COMPARISON ===")

all_models = {
    "L3_original (Kushi)": emb_L3,
    "L6_original": emb_L6_orig,
    "L6_normalized": emb_L6_norm,
}
all_models.update({k: v for k, v in emotion_embeddings.items()})

comparison = []
for name, emb in all_models.items():
    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb)

    # PCA variance
    pca = PCA(n_components=min(20, emb.shape[1]))
    pca.fit(emb_s)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # KMeans sweep
    best_sil = -1
    best_k = 3
    for k in range(3, 16):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(emb_s)
        sil = silhouette_score(emb_s, labels, sample_size=min(1000, len(emb_s)))
        if sil > best_sil:
            best_sil = sil
            best_k = k

    row = {
        'model': name,
        'dim': emb.shape[1],
        'best_k': best_k,
        'best_silhouette': best_sil,
        'var_2d': cumvar[1] if len(cumvar) > 1 else 0,
        'var_5d': cumvar[4] if len(cumvar) > 4 else 0,
        'var_10d': cumvar[9] if len(cumvar) > 9 else 0,
    }
    comparison.append(row)
    print(f"\n  {name} ({emb.shape[1]}d):")
    print(f"    Best k={best_k}, silhouette={best_sil:.4f}")
    print(f"    PCA var: 2D={cumvar[1]:.1%}, 5D={cumvar[4] if len(cumvar)>4 else 0:.1%}, 10D={cumvar[9] if len(cumvar)>9 else 0:.1%}")

# ─── Visualization: Side-by-side UMAP ───
n_models = len(all_models)
fig, axes = plt.subplots(2, n_models, figsize=(7*n_models, 12))
fig.suptitle('GACS Step 3: Embedding Model Comparison', fontsize=16, fontweight='bold', y=0.98)

for i, (name, emb) in enumerate(all_models.items()):
    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb)

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    u2d = reducer.fit_transform(emb_s)

    # KMeans with best k for this model
    best_k = comparison[i]['best_k']
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_s)

    # Top row: clusters
    scatter = axes[0, i].scatter(u2d[:, 0], u2d[:, 1], c=labels, cmap='tab10', alpha=0.5, s=12)
    axes[0, i].set_title(f'{name}\nk={best_k}, sil={comparison[i]["best_silhouette"]:.3f}', fontsize=10)

    # Bottom row: PCA scree
    pca = PCA(n_components=min(20, emb.shape[1]))
    pca.fit(emb_s)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    axes[1, i].bar(range(1, len(cumvar)+1), pca.explained_variance_ratio_, color='steelblue', alpha=0.7)
    axes[1, i].plot(range(1, len(cumvar)+1), cumvar, 'ro-', markersize=3)
    axes[1, i].axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
    axes[1, i].set_title(f'PCA Variance ({name})', fontsize=9)
    axes[1, i].set_xlabel('Component')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "step3_comparison.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT}/step3_comparison.png")

# Save comparison
comp_df = pd.DataFrame(comparison)
comp_df.to_csv(OUT / "model_comparison.csv", index=False)
with open(OUT / "step3_results.json", 'w') as f:
    json.dump(comparison, f, indent=2, default=str)
print(f"Saved: {OUT}/step3_results.json")

print("\n=== STEP 3 COMPLETE ===")
print("\nSummary:")
for c in sorted(comparison, key=lambda x: x['best_silhouette'], reverse=True):
    print(f"  {c['model']:30s}: sil={c['best_silhouette']:.4f} (k={c['best_k']})")
