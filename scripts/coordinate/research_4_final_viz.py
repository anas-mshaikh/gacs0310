"""
GACS Final Visualizations
Generates publication-quality scatter plots for the research report.
"""
import matplotlib

matplotlib.use('Agg')
import json

import matplotlib.pyplot as plt
import pandas as pd

# ─── Load data ───
df = pd.read_csv('research/gacs_coordinates.csv')
with open('research/axis_interpretation.json') as f:
    axes = json.load(f)

# ─── Color palettes (colorblind-friendly) ───
# Wong palette extended
cluster_colors = {0: '#E69F00', 1: '#56B4E9', 2: '#009E73', 3: '#CC79A7'}
cluster_names = {
    0: 'C0: Tension/Fear',
    1: 'C1: Sadness/Darkness',
    2: 'C2: Wonder/Tension',
    3: 'C3: Calm/Joy/Warmth'
}

# ─── Axis labels ───
xlabel = f"PC1: {axes['PC1']['name']} ({axes['PC1']['variance_explained_pct']:.1f}% var)"
ylabel = f"PC2: {axes['PC2']['name']} ({axes['PC2']['variance_explained_pct']:.1f}% var)"

# ════════════════════════════════════════════════════════════════
# 1. 2D Scatter — Color by Cluster
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 10))

for cid in sorted(df['cluster_label'].unique()):
    mask = df['cluster_label'] == cid
    ax.scatter(
        df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
        c=cluster_colors[cid], label=cluster_names[cid],
        alpha=0.5, s=20, edgecolors='none'
    )

# Centroids
for cid in sorted(df['cluster_label'].unique()):
    mask = df['cluster_label'] == cid
    cx = df.loc[mask, 'PC1'].mean()
    cy = df.loc[mask, 'PC2'].mean()
    ax.scatter(cx, cy, c=cluster_colors[cid], s=300, marker='*',
               edgecolors='black', linewidths=1.0, zorder=10)
    ax.annotate(f'C{cid}', (cx, cy), fontsize=11, fontweight='bold',
                ha='center', va='bottom', xytext=(0, 12),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

ax.set_xlabel(xlabel, fontsize=13)
ax.set_ylabel(ylabel, fontsize=13)
ax.set_title('GACS 2D Emotional Coordinate Space (KMeans k=4)', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig('research/final_scatter_2d.png', dpi=150)
plt.close()
print("[OK] final_scatter_2d.png")

# ════════════════════════════════════════════════════════════════
# 2. 2D Scatter — Color by Dominant Mood (top 8 + other)
# ════════════════════════════════════════════════════════════════
top_moods = df['dominant_mood'].value_counts().head(8).index.tolist()
df['mood_group'] = df['dominant_mood'].apply(lambda x: x if x in top_moods else 'other')

mood_colors = {
    'tension': '#D55E00',
    'calm': '#0072B2',
    'joy': '#F0E442',
    'excitement': '#E69F00',
    'warmth': '#CC79A7',
    'contemplation': '#56B4E9',
    'seriousness': '#999999',
    'mystery': '#009E73',
    'other': '#BBBBBB'
}

fig, ax = plt.subplots(figsize=(12, 10))

# Plot 'other' first (background)
mask = df['mood_group'] == 'other'
ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
           c=mood_colors['other'], label='other', alpha=0.3, s=15, edgecolors='none')

for mood in top_moods:
    mask = df['mood_group'] == mood
    ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
               c=mood_colors[mood], label=mood, alpha=0.6, s=25, edgecolors='none')

ax.set_xlabel(xlabel, fontsize=13)
ax.set_ylabel(ylabel, fontsize=13)
ax.set_title('GACS 2D Space by Dominant Mood', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig('research/final_scatter_2d_mood.png', dpi=150)
plt.close()
print("[OK] final_scatter_2d_mood.png")

# ════════════════════════════════════════════════════════════════
# 3. 2D Scatter — Color by Video Category
# ════════════════════════════════════════════════════════════════
cat_colors = {
    'movie_trailers': '#D55E00',
    'advertisements': '#0072B2',
    'emotional_shorts': '#009E73',
    'animations': '#CC79A7',
    'other': '#999999'
}

fig, ax = plt.subplots(figsize=(12, 10))

for cat in ['other', 'advertisements', 'animations', 'emotional_shorts', 'movie_trailers']:
    mask = df['category'] == cat
    if mask.sum() == 0:
        continue
    ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
               c=cat_colors[cat], label=cat.replace('_', ' ').title(),
               alpha=0.5, s=20, edgecolors='none')

ax.set_xlabel(xlabel, fontsize=13)
ax.set_ylabel(ylabel, fontsize=13)
ax.set_title('GACS 2D Space by Video Category', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig('research/final_scatter_2d_category.png', dpi=150)
plt.close()
print("[OK] final_scatter_2d_category.png")

# ════════════════════════════════════════════════════════════════
# 4. 3D Interactive — Plotly (PC1 vs PC2 vs PC3)
# ════════════════════════════════════════════════════════════════
try:
    import plotly.express as px

    df['cluster_name'] = df['cluster_label'].map(cluster_names)
    df['hover_text'] = (
        'Scene: ' + df['scene_id'].astype(str) +
        '<br>Video: ' + df['video_name'] +
        '<br>Category: ' + df['category'] +
        '<br>Mood: ' + df['dominant_mood']
    )

    fig3d = px.scatter_3d(
        df, x='PC1', y='PC2', z='PC3',
        color='cluster_name',
        color_discrete_map={v: cluster_colors[k] for k, v in cluster_names.items()},
        hover_data={'hover_text': True, 'PC1': ':.2f', 'PC2': ':.2f', 'PC3': ':.2f',
                    'cluster_name': False},
        custom_data=['hover_text'],
        opacity=0.6,
        title='GACS 3D Emotional Coordinate Space'
    )
    fig3d.update_traces(
        marker=dict(size=3),
        hovertemplate='%{customdata[0]}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>'
    )
    fig3d.update_layout(
        scene=dict(
            xaxis_title=f"PC1: {axes['PC1']['name']}",
            yaxis_title=f"PC2: {axes['PC2']['name']}",
            zaxis_title=f"PC3: {axes['PC3']['name']}"
        ),
        width=1000, height=800
    )
    fig3d.write_html('research/final_scatter_3d.html')
    print("[OK] final_scatter_3d.html")
except ImportError:
    print("[SKIP] plotly not installed, skipping 3D visualization")
except Exception as e:
    print(f"[SKIP] 3D plot failed: {e}")

print("\nAll visualizations complete.")
