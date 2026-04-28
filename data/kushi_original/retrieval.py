import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

labeled_df = pd.read_csv("data/gacs_labeled_scenes.csv")
scenes_df = pd.read_csv("data/scene_metadata.csv")


def parse_embedding(emb_str):
    """Safe embedding parsing from CSV"""
    clean = str(emb_str).strip('[]').replace('\n', ' ').replace(',', ' ')
    nums = [float(x) for x in clean.split() if x.strip()]
    return np.array(nums)


embeddings = np.array([parse_embedding(e) for e in labeled_df['embedding']])
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

print(f"{len(labeled_df)} labeled / {len(scenes_df)} total scenes")
if 'label_ok' in labeled_df.columns:
    print(f"Label quality: {labeled_df['label_ok'].mean():.1%} OK")
else:
    print("No label_ok column - assuming all good")
print(f"Embeddings shape: {embeddings.shape}")


print("\nYOUR LABELED SCENES:")
for i, row in labeled_df[['video_name', 'scene_id', 'mood_str', 'objects_str']].head(10).iterrows():
    print(f"  {row['video_name']} s{int(row['scene_id']):02d}: {row['mood_str']} | {row['objects_str']}")


def search_labeled(query, top_k=5):
    """Search ONLY labeled scenes"""
    query_emb = embedder.encode([query])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]

    print(f"\n'{query}' → Top {top_k} LABELED:")
    print("=" * 80)
    for i, idx in enumerate(top_idx):
        row = labeled_df.iloc[idx]
        score = sims[idx]
        print(f"{i + 1:2d}. [{score:.3f}] {row['video_name']} s{int(row['scene_id']):02d}")
        print(f"   {row.get('mood_str', 'N/A')} | {row.get('objects_str', 'N/A')} | {row.get('style_str', 'N/A')}")


def search_all_scenes(query, top_k=10):
    """Search ALL 641 scenes (labeled + unlabeled) using hybrid similarity"""
    # 1. Query embedding
    query_emb = embedder.encode([query])

    # 2. Search LABELED scenes (exact semantic match)
    labeled_sims = cosine_similarity(query_emb, embeddings)[0]
    labeled_top_idx = np.argsort(labeled_sims)[-3:][::-1]

    # 3. Video-level similarity for UNLABELED scenes
    try:
        labeled_index = labeled_df.set_index(['video_name', 'scene_id']).index
        scenes_index = scenes_df.set_index(['video_name', 'scene_id']).index
        unlabeled_scenes = scenes_df[~scenes_index.isin(labeled_index)].reset_index(drop=True)
    except:
        unlabeled_scenes = scenes_df.head(10)

    # Extract video names from query for video similarity
    video_keywords = extract_video_keywords(query)
    video_scores = {}

    for video_name in unlabeled_scenes['video_name'].unique():
        video_score = sum(1 for kw in video_keywords if kw in video_name.lower())
        video_scores[video_name] = video_score

    # 4. Get top unlabeled scenes by video similarity
    top_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    unlabeled_top_scenes = []

    for video_name, score in top_videos:
        video_scenes = unlabeled_scenes[unlabeled_scenes['video_name'] == video_name]
        if len(video_scenes) > 0:
            top_scene = video_scenes.head(1).iloc[0]
            unlabeled_top_scenes.append(top_scene)

    # 5. COMBINE results
    print(f"\n'{query}' → ALL SCENES (641 total):")
    print("=" * 100)
    print("LABELED (Exact semantic match):")
    for i, idx in enumerate(labeled_top_idx):
        row = labeled_df.iloc[idx]
        score = labeled_sims[idx]
        print(f"  {i + 1}. [{score:.3f}] {row['video_name']} s{int(row['scene_id']):02d}")
        print(f"     {row.get('mood_str', 'N/A')} | {row.get('objects_str', 'N/A')}")

    print("\nUNLABELED (Video similarity):")
    for i, scene in enumerate(unlabeled_top_scenes):
        print(f"  {i + 1}.  {scene['video_name']} s{int(scene['scene_id']):02d}")
        print(f"     {scene['rep_frame_path']}")

    return labeled_top_idx, unlabeled_top_scenes


def extract_video_keywords(query):
    """Extract video-related keywords from query"""
    keywords = ['action', 'emotional', 'trailer', 'shorts', 'baby', 'ready', 'toy', 'story']
    return [kw for kw in keywords if kw in query.lower()]



print("\n" + "=" * 80)
print("SELF-TEST (Scene vs itself = 1.0)")
print("=" * 80)


test_scene_0 = labeled_df.iloc[0]
self_query_0 = f"{test_scene_0['mood_str']} {test_scene_0['objects_str']}"
print(f"TEST 1: '{self_query_0[:60]}...'")
search_labeled(self_query_0, top_k=1)

print("\n" + "=" * 80)
print("REAL QUERIES FROM YOUR DATASET")
print("=" * 80)


real_queries = [
    # From ready_or_not_trailer
    "weary tense somber gritty man fluorescent light hallway",
    "terrifying frantic violent blood jacket blinds",

    # From toy_story_5_trailer
    "playful creative bright optimistic lamp letters text",
    "surprised alarmed concerned fearful woody buzz lightyear",

    # Generic moods
    "dark tense horror scary anxious distressed",
    "happy playful colorful animated creative"
]

for q in real_queries:
    search_labeled(q, top_k=3)

print("\n" + "=" * 80)
print("FULL SEARCH TEST")
print("=" * 80)
queries_full = [
    "dark horror ready or not trailer blood",
    "toy story woody buzz lightyear animated",
    "baby funny man vs baby trailer"
]

for q in queries_full:
    search_all_scenes(q)
