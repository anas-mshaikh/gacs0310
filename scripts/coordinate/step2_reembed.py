"""
Step 2: SBERT Re-embedding with all-MiniLM-L6-v2
원본 combined_text + 정규화된 mood text 두 가지 버전 임베딩 생성
"""
import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

KUSHI = Path("gacs0202(kushi)")
OUT = Path("data/step2")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(KUSHI / "gacs_labeled_scenes.csv")
print(f"Loaded: {len(df)} rows")

# ─── Mood Normalization Map (Step 1에서 가져옴) ───
MOOD_MAP = {
    "joy": ["joyful", "joyous", "happy", "cheerful", "jubilant", "overjoyed", "delighted", "ecstatic", "lighthearted", "upbeat", "bright", "radiant", "glowing"],
    "excitement": ["excited", "exciting", "exhilarating", "thrilling", "energetic", "vibrant", "lively", "dynamic", "active", "vigorous", "spirited", "boisterous", "fast-paced", "fast", "bustling"],
    "celebration": ["celebratory", "festive", "triumphant", "proud", "accomplished"],
    "amusement": ["playful", "whimsical", "humorous", "amusing", "amused", "comical", "silly", "mischievous", "quirky", "cute", "adorable"],
    "wonder": ["magical", "wondrous", "enchanting", "enchanted", "majestic", "awe-inspiring", "breathtaking", "captivating", "grand", "epic", "impressive"],
    "adventure": ["adventurous", "daring", "bold", "wild", "free", "exploratory", "ambitious"],
    "calm": ["calm", "relaxed", "tranquil", "serene", "peaceful", "quiet", "gentle", "mild", "mellow", "soothing", "soft", "subtle", "understated", "composed"],
    "warmth": ["warm", "tender", "affectionate", "loving", "caring", "nurturing", "supportive", "kind", "compassionate", "empathetic", "heartwarming", "heartfelt", "sweet", "endearing", "fond"],
    "contentment": ["content", "pleasant", "comfortable", "cozy", "homey", "safe", "welcoming", "inviting", "friendly", "amicable", "convivial", "carefree"],
    "hope": ["hopeful", "optimistic", "encouraging", "uplifting", "inspiring", "inspired", "positive", "resilient", "determined"],
    "nostalgia": ["nostalgic", "sentimental", "wistful", "bittersweet"],
    "romance": ["romantic", "intimate", "flirtatious", "passionate"],
    "gratitude": ["grateful", "appreciative", "relieved", "moved", "touched", "sincere"],
    "neutral": ["neutral", "ordinary", "mundane", "everyday", "routine", "casual", "simple", "plain", "blank", "unremarkable", "commonplace"],
    "focus": ["focused", "attentive", "observant", "watchful", "engaged", "precise", "careful", "concentrated", "absorbed", "diligent", "dedicated", "purposeful", "resolute"],
    "contemplation": ["pensive", "thoughtful", "contemplative", "reflective", "introspective", "questioning", "curious", "intrigued", "intriguing"],
    "seriousness": ["serious", "solemn", "formal", "dignified", "reserved", "stoic", "stately", "commanding", "authoritative"],
    "informative": ["informative", "informational", "professional", "technical", "clinical", "factual", "objective", "direct", "functional", "practical"],
    "sadness": ["sad", "somber", "melancholic", "melancholy", "sorrowful", "tearful", "heartbreaking", "tragic", "dejected", "unhappy", "disappointed", "gloomy", "bleak"],
    "loneliness": ["lonely", "solitary", "isolated", "desolate", "abandoned", "stranded", "empty", "detached", "distant"],
    "weariness": ["tired", "weary", "exhausted", "fatigued", "sleepy", "bored", "apathetic", "disengaged", "subdued", "muted"],
    "vulnerability": ["vulnerable", "helpless", "defeated", "overwhelmed", "struggling", "desperate", "pleading", "pained"],
    "poignance": ["poignant", "touching", "emotional", "moving", "profound", "raw", "expressive"],
    "tension": ["tense", "intense", "suspenseful", "anticipatory", "expectant", "anxious", "uneasy", "apprehensive", "stressed", "uncomfortable", "uncertain", "hesitant", "cautious", "wary"],
    "fear": ["scared", "fearful", "terrifying", "terrified", "frightening", "alarming", "horrific", "eerie", "creepy", "unsettling", "ominous", "foreboding", "menacing", "threatening", "perilous", "dangerous"],
    "anger": ["angry", "furious", "enraged", "aggressive", "confrontational", "defiant", "rebellious", "fierce", "violent", "brutal"],
    "chaos": ["chaotic", "frantic", "frenzied", "rushed", "overwhelming", "explosive", "destructive", "catastrophic", "apocalyptic"],
    "darkness": ["dark", "grim", "gritty", "cold", "stark", "oppressive", "haunting", "gothic", "murky"],
    "mystery": ["mysterious", "enigmatic", "surreal", "ethereal", "dreamlike", "dreamy", "otherworldly", "mystical", "fantastical", "cosmic", "abstract"],
    "elegance": ["elegant", "luxurious", "sophisticated", "refined", "glamorous", "opulent", "extravagant", "sleek", "prestigious", "regal", "graceful"],
    "domesticity": ["domestic", "familial", "communal", "social", "collaborative", "inclusive", "united", "connected"],
    "nature": ["natural", "organic", "rustic", "earthy", "verdant", "tropical", "wintry", "aquatic", "idyllic"],
    "urban": ["urban", "modern", "futuristic", "digital", "industrial", "corporate", "busy", "crowded"],
    "tradition": ["traditional", "historical", "historic", "cultural", "ceremonial", "classical", "classic", "vintage", "ancient"],
}
REVERSE = {}
for cat, words in MOOD_MAP.items():
    for w in words:
        REVERSE[w] = cat

def normalize_mood(m):
    m = m.strip().lower()
    return REVERSE.get(m, m)

def parse_and_normalize(val):
    if pd.isna(val): return []
    try:
        moods = ast.literal_eval(str(val))
        if isinstance(moods, list):
            return list(dict.fromkeys(normalize_mood(str(m)) for m in moods if str(m).strip()))
    except: pass
    return []

# ─── Prepare texts ───
# Version A: Original combined_text (as-is from Kushi)
texts_original = df['combined_text'].fillna('').tolist()

# Version B: Normalized mood + style + objects
df['norm_moods'] = df['mood_json'].apply(parse_and_normalize)

def build_normalized_text(row):
    moods = row['norm_moods']
    styles = []
    objects = []
    try:
        s = ast.literal_eval(str(row['style_json']))
        if isinstance(s, list): styles = [str(x).strip().lower() for x in s]
    except: pass
    try:
        o = ast.literal_eval(str(row['objects_json']))
        if isinstance(o, list): objects = [str(x).strip().lower() for x in o]
    except: pass
    parts = []
    if moods: parts.append("mood: " + ", ".join(moods))
    if styles: parts.append("style: " + ", ".join(styles))
    if objects: parts.append("objects: " + ", ".join(objects))
    return ". ".join(parts)

texts_normalized = df.apply(build_normalized_text, axis=1).tolist()

# ─── Embed ───
print("Loading all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
print(f"Model loaded on {model.device}")

print("\nEncoding original texts...")
t0 = time.time()
emb_original = model.encode(texts_original, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
print(f"  Done in {time.time()-t0:.1f}s, shape: {emb_original.shape}")

print("\nEncoding normalized texts...")
t0 = time.time()
emb_normalized = model.encode(texts_normalized, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
print(f"  Done in {time.time()-t0:.1f}s, shape: {emb_normalized.shape}")

# ─── Save ───
np.save(OUT / "emb_original_L6v2.npy", emb_original)
np.save(OUT / "emb_normalized_L6v2.npy", emb_normalized)

# Save texts for reference
df[['scene_id', 'video_name']].to_csv(OUT / "scene_index.csv", index=False)
pd.DataFrame({'original': texts_original, 'normalized': texts_normalized}).to_csv(OUT / "texts.csv", index=False)

print(f"\nSaved to {OUT}/")
print("  emb_original_L6v2.npy")
print("  emb_normalized_L6v2.npy")
print("  scene_index.csv")
print("  texts.csv")
print("\n=== STEP 2 COMPLETE ===")
