"""
research_2_axis_discovery.py
Discover and name interpretable coordinate axes from emotion embeddings via PCA,
correlating PC scores with mood categories, valence, and arousal.
"""

import ast
import json

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────────

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

VALENCE = {"joy":1.0,"excitement":0.9,"celebration":0.9,"amusement":0.8,"wonder":0.7,"adventure":0.7,"calm":0.6,"warmth":0.7,"contentment":0.6,"hope":0.7,"nostalgia":0.3,"romance":0.6,"gratitude":0.7,"neutral":0,"focus":0.1,"contemplation":0,"seriousness":-0.1,"informative":0,"sadness":-0.7,"loneliness":-0.6,"weariness":-0.4,"vulnerability":-0.6,"poignance":-0.3,"tension":-0.3,"fear":-0.8,"anger":-0.7,"chaos":-0.6,"darkness":-0.5,"mystery":-0.1,"elegance":0.3,"domesticity":0.3,"nature":0.4,"urban":0,"tradition":0.1}
AROUSAL = {"joy":0.6,"excitement":0.9,"celebration":0.8,"amusement":0.5,"wonder":0.4,"adventure":0.8,"calm":-0.7,"warmth":-0.3,"contentment":-0.5,"hope":0.2,"nostalgia":-0.3,"romance":0.1,"gratitude":-0.2,"neutral":0,"focus":0.2,"contemplation":-0.3,"seriousness":0.1,"informative":-0.1,"sadness":-0.4,"loneliness":-0.5,"weariness":-0.8,"vulnerability":-0.2,"poignance":-0.1,"tension":0.7,"fear":0.8,"anger":0.9,"chaos":1.0,"darkness":0.3,"mystery":0.3,"elegance":-0.2,"domesticity":-0.3,"nature":-0.4,"urban":0.3,"tradition":-0.2}

CATEGORIES = sorted(MOOD_MAP.keys())  # 35 categories

# Build reverse map: raw mood word -> category
RAW_TO_CAT = {}
for cat, words in MOOD_MAP.items():
    for w in words:
        RAW_TO_CAT[w.lower()] = cat

# ── Paths ──────────────────────────────────────────────────────────────────────

EMB_PATH = "data/step3/emb_emotion_distil_norm.npy"
CSV_PATH = "gacs0202(kushi)/gacs_labeled_scenes.csv"
OUT_DIR  = "research"

# ── 1. Load & normalize ───────────────────────────────────────────────────────

print("=" * 70)
print("STEP 1: Load embeddings and normalize")
print("=" * 70)

emb = np.load(EMB_PATH).astype(np.float64)
print(f"  Loaded embeddings: {emb.shape}")

scaler = StandardScaler()
emb_scaled = scaler.fit_transform(emb)
print(f"  StandardScaler applied: mean~{emb_scaled.mean():.6f}, std~{emb_scaled.std():.6f}")

# ── 2. PCA until >=80% variance ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 2: PCA — find N components for >= 80% cumulative variance")
print("=" * 70)

pca_full = PCA()
pc_scores_full = pca_full.fit_transform(emb_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
N = int(np.searchsorted(cum_var, 0.80) + 1)

print(f"  Components needed for >= 80% variance: N = {N}")
print(f"  Cumulative variance at N={N}: {cum_var[N-1]*100:.2f}%")
print(f"  Top 10 individual variances: {[f'{v*100:.2f}%' for v in pca_full.explained_variance_ratio_[:10]]}")

pc_scores = pc_scores_full[:, :N]  # (1275, N)

# ── 3. Parse moods → binary indicator matrix & valence/arousal scores ─────────

print("\n" + "=" * 70)
print("STEP 3: Build mood indicator matrix & compute correlations")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
n_scenes = len(df)
assert n_scenes == emb.shape[0], f"Mismatch: {n_scenes} vs {emb.shape[0]}"

# Parse mood_json and map to categories
mood_indicator = np.zeros((n_scenes, len(CATEGORIES)), dtype=int)
valence_scores = np.zeros(n_scenes)
arousal_scores = np.zeros(n_scenes)

unmapped = set()
for i, row in df.iterrows():
    raw_moods = ast.literal_eval(row["mood_json"])
    cats_present = set()
    for m in raw_moods:
        m_low = m.lower().strip()
        if m_low in RAW_TO_CAT:
            cats_present.add(RAW_TO_CAT[m_low])
        else:
            unmapped.add(m_low)
    # Fill indicator
    for cat in cats_present:
        idx = CATEGORIES.index(cat)
        mood_indicator[i, idx] = 1
    # Valence / arousal: average over present categories
    if cats_present:
        valence_scores[i] = np.mean([VALENCE[c] for c in cats_present])
        arousal_scores[i] = np.mean([AROUSAL[c] for c in cats_present])

print(f"  Scenes: {n_scenes}")
print(f"  Mood categories: {len(CATEGORIES)}")
print(f"  Unmapped raw moods ({len(unmapped)}): {sorted(unmapped)[:20]}{'...' if len(unmapped)>20 else ''}")
print(f"  Indicator density: {mood_indicator.sum()}/{mood_indicator.size} = {mood_indicator.sum()/mood_indicator.size:.4f}")

# ── Correlations ───────────────────────────────────────────────────────────────

# correlation_matrix[pc_idx][cat] = {"r": ..., "p": ...}
correlation_matrix = {}
for pc_i in range(N):
    pc_key = f"PC{pc_i+1}"
    correlation_matrix[pc_key] = {}
    for j, cat in enumerate(CATEGORIES):
        col = mood_indicator[:, j]
        if col.sum() == 0 or col.sum() == n_scenes:
            r, p = 0.0, 1.0
        else:
            r, p = pointbiserialr(col, pc_scores[:, pc_i])
        correlation_matrix[pc_key][cat] = {"r": round(float(r), 4), "p": round(float(p), 4)}

# Valence / arousal correlation with each PC
va_corr = {}
for pc_i in range(N):
    pc_key = f"PC{pc_i+1}"
    r_val, p_val = pearsonr(pc_scores[:, pc_i], valence_scores)
    r_aro, p_aro = pearsonr(pc_scores[:, pc_i], arousal_scores)
    va_corr[pc_key] = {
        "valence_r": round(float(r_val), 4), "valence_p": round(float(p_val), 4),
        "arousal_r": round(float(r_aro), 4), "arousal_p": round(float(p_aro), 4),
    }

# ── 4 & 5. Extreme analysis + axis naming ─────────────────────────────────────

print("\n" + "=" * 70)
print("STEPS 4-5: Extreme analysis & axis naming")
print("=" * 70)

axis_interpretation = {}
for pc_i in range(N):
    pc_key = f"PC{pc_i+1}"
    scores = pc_scores[:, pc_i]
    var_exp = round(float(pca_full.explained_variance_ratio_[pc_i]) * 100, 4)

    # Sort moods by correlation
    corrs = correlation_matrix[pc_key]
    sorted_pos = sorted(corrs.items(), key=lambda x: x[1]["r"], reverse=True)[:5]
    sorted_neg = sorted(corrs.items(), key=lambda x: x[1]["r"])[:5]

    # Extreme scenes (top/bottom 5%)
    k = max(1, int(0.05 * n_scenes))
    top_idx = np.argsort(scores)[-k:]
    bot_idx = np.argsort(scores)[:k]

    def top_moods_at(indices):
        counts = mood_indicator[indices].sum(axis=0)
        ranked = sorted(zip(CATEGORIES, counts), key=lambda x: -x[1])
        return [(c, int(n)) for c, n in ranked[:5]]

    top_moods_pos = top_moods_at(top_idx)
    top_moods_neg = top_moods_at(bot_idx)

    # Name the axis
    pos_name = sorted_pos[0][0]
    neg_name = sorted_neg[0][0]
    pos_r = sorted_pos[0][1]["r"]
    neg_r = sorted_neg[0][1]["r"]
    axis_name = f"{pos_name.title()}-{neg_name.title()}"

    axis_interpretation[pc_key] = {
        "name": axis_name,
        "label": f"{pc_key}: {axis_name} ({pos_name} r={pos_r:+.4f} vs {neg_name} r={neg_r:+.4f})",
        "variance_explained_pct": var_exp,
        "top5_positive": [{"mood": m, "r": corrs[m]["r"], "p": corrs[m]["p"]} for m, _ in sorted_pos],
        "top5_negative": [{"mood": m, "r": corrs[m]["r"], "p": corrs[m]["p"]} for m, _ in sorted_neg],
        "extreme_positive_moods": top_moods_pos,
        "extreme_negative_moods": top_moods_neg,
        "valence_r": va_corr[pc_key]["valence_r"],
        "valence_p": va_corr[pc_key]["valence_p"],
        "arousal_r": va_corr[pc_key]["arousal_r"],
        "arousal_p": va_corr[pc_key]["arousal_p"],
    }

    print(f"\n  {axis_interpretation[pc_key]['label']}")
    print(f"    Variance explained: {var_exp:.2f}%")
    print(f"    Top 5 + moods: {[(m, corrs[m]['r']) for m,_ in sorted_pos]}")
    print(f"    Top 5 - moods: {[(m, corrs[m]['r']) for m,_ in sorted_neg]}")
    print(f"    Extreme + scenes moods: {top_moods_pos}")
    print(f"    Extreme - scenes moods: {top_moods_neg}")
    print(f"    Valence r={va_corr[pc_key]['valence_r']:.4f} (p={va_corr[pc_key]['valence_p']:.4f})")
    print(f"    Arousal r={va_corr[pc_key]['arousal_r']:.4f} (p={va_corr[pc_key]['arousal_p']:.4f})")

# ── 6. Validation ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 6: Validation — PC1 vs Valence, PC2 vs Arousal")
print("=" * 70)

r1v, p1v = pearsonr(pc_scores[:, 0], valence_scores)
r2a, p2a = pearsonr(pc_scores[:, 1], arousal_scores)
print(f"  PC1 ~ Valence:  r = {r1v:.4f}, p = {p1v:.6f}")
print(f"  PC2 ~ Arousal:  r = {r2a:.4f}, p = {p2a:.6f}")

# Also check which PC best correlates with valence/arousal
best_val_pc = max(range(N), key=lambda i: abs(va_corr[f"PC{i+1}"]["valence_r"]))
best_aro_pc = max(range(N), key=lambda i: abs(va_corr[f"PC{i+1}"]["arousal_r"]))
print(f"  Best PC for Valence: PC{best_val_pc+1} (r={va_corr[f'PC{best_val_pc+1}']['valence_r']:.4f})")
print(f"  Best PC for Arousal: PC{best_aro_pc+1} (r={va_corr[f'PC{best_aro_pc+1}']['arousal_r']:.4f})")

# ── Save outputs ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Saving outputs to research/")
print("=" * 70)

with open(f"{OUT_DIR}/axis_correlation_matrix.json", "w") as f:
    json.dump(correlation_matrix, f, indent=2)
print(f"  Saved axis_correlation_matrix.json ({N} PCs x {len(CATEGORIES)} moods)")

with open(f"{OUT_DIR}/axis_interpretation.json", "w") as f:
    json.dump(axis_interpretation, f, indent=2)
print(f"  Saved axis_interpretation.json ({N} PCs)")

# ── Visualization: heatmap of top 10 PCs x 35 moods ──────────────────────────

n_show = min(10, N)
mat = np.zeros((n_show, len(CATEGORIES)))
pvals = np.zeros((n_show, len(CATEGORIES)))
for i in range(n_show):
    pc_key = f"PC{i+1}"
    for j, cat in enumerate(CATEGORIES):
        mat[i, j] = correlation_matrix[pc_key][cat]["r"]
        pvals[i, j] = correlation_matrix[pc_key][cat]["p"]

fig, ax = plt.subplots(figsize=(18, 8))
vmax = max(abs(mat.min()), abs(mat.max()))
im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

# Significance stars
for i in range(n_show):
    for j in range(len(CATEGORIES)):
        p = pvals[i, j]
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = ""
        if stars:
            color = "white" if abs(mat[i, j]) > vmax * 0.6 else "black"
            ax.text(j, i, stars, ha="center", va="center", fontsize=7, color=color, fontweight="bold")

# Labels
pc_labels = [f"PC{i+1} ({pca_full.explained_variance_ratio_[i]*100:.1f}%)" for i in range(n_show)]
ax.set_yticks(range(n_show))
ax.set_yticklabels(pc_labels, fontsize=10)
ax.set_xticks(range(len(CATEGORIES)))
ax.set_xticklabels(CATEGORIES, rotation=55, ha="right", fontsize=9)

ax.set_title("Point-Biserial Correlation: PC Scores vs Mood Categories\n(*p<0.05  **p<0.01  ***p<0.001)", fontsize=13, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Correlation (r)", fontsize=10)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/axis_visualization.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved axis_visualization.png (dpi=150)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
