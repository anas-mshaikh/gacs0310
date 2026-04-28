"""
track1_fix_mood_map.py
Analyze unmapped mood words and extend MOOD_MAP using semantic similarity.
"""
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────
CSV_PATH = "gacs0202(kushi)/gacs_labeled_scenes.csv"
RESEARCH_DIR = Path("research")
RESEARCH_DIR.mkdir(exist_ok=True)

# ── Current MOOD_MAP ───────────────────────────────────────────────────
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

# ── Step 1-2: Parse mood_json, flatten, count ─────────────────────────
print("=" * 70)
print("STEP 1-2: Parsing mood_json and counting word frequencies")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
all_words = []
parse_errors = 0

for idx, row in df.iterrows():
    raw = row["mood_json"]
    if pd.isna(raw):
        continue
    try:
        words = ast.literal_eval(raw)
        # Normalize: lowercase, strip
        words = [w.strip().lower() for w in words if isinstance(w, str)]
        all_words.extend(words)
    except Exception:
        parse_errors += 1

freq = Counter(all_words)
total_unique = len(freq)
total_tokens = len(all_words)
print(f"Total tokens:       {total_tokens}")
print(f"Unique words:       {total_unique}")
print(f"Parse errors:       {parse_errors}")

# ── Step 3: Build reverse map ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Building reverse map from MOOD_MAP")
print("=" * 70)

reverse_map = {}
for category, words in MOOD_MAP.items():
    for w in words:
        reverse_map[w.lower()] = category

mapped_words_set = set(reverse_map.keys())
print(f"MOOD_MAP categories: {len(MOOD_MAP)}")
print(f"MOOD_MAP words:      {len(mapped_words_set)}")

# ── Step 4: Find unmapped words ────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Finding unmapped words")
print("=" * 70)

unmapped = {w for w in freq if w not in mapped_words_set}
mapped_before = total_unique - len(unmapped)
print(f"Mapped (before):    {mapped_before}")
print(f"Unmapped (before):  {len(unmapped)}")

# ── Step 5: Print unmapped sorted by frequency ────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Unmapped words by frequency (top 50 shown)")
print("=" * 70)

unmapped_freq = [(w, freq[w]) for w in unmapped]
unmapped_freq.sort(key=lambda x: -x[1])

for i, (w, c) in enumerate(unmapped_freq[:50]):
    print(f"  {i+1:3d}. {w:30s}  freq={c}")
if len(unmapped_freq) > 50:
    print(f"  ... and {len(unmapped_freq) - 50} more")

# ── Step 6: Semantic similarity assignment ─────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Assigning unmapped words via semantic similarity")
print("=" * 70)

model = SentenceTransformer("all-MiniLM-L6-v2")

category_names = list(MOOD_MAP.keys())
cat_embeddings = model.encode(category_names, show_progress_bar=False)

unmapped_list = [w for w, _ in unmapped_freq]  # sorted by freq desc
if unmapped_list:
    word_embeddings = model.encode(unmapped_list, show_progress_bar=False)
    sims = cosine_similarity(word_embeddings, cat_embeddings)  # (n_unmapped, n_cats)

assignments = []       # word -> (category, similarity)
needs_new_cat = []     # word -> (best_match, similarity)

THRESHOLD = 0.3

for i, word in enumerate(unmapped_list):
    best_idx = int(np.argmax(sims[i]))
    best_sim = float(sims[i, best_idx])
    best_cat = category_names[best_idx]

    if best_sim < THRESHOLD:
        needs_new_cat.append({
            "word": word,
            "best_match": best_cat,
            "similarity": round(best_sim, 4),
            "frequency": freq[word],
        })
    else:
        assignments.append({
            "word": word,
            "category": best_cat,
            "similarity": round(best_sim, 4),
            "frequency": freq[word],
        })

print(f"Assigned to existing category: {len(assignments)}")
print(f"Needs new category (sim < {THRESHOLD}): {len(needs_new_cat)}")

# ── Step 7: Show flagged words ─────────────────────────────────────────
if needs_new_cat:
    print("\n  Flagged words (needs_new_category):")
    for item in needs_new_cat:
        print(f"    {item['word']:30s} -> {item['best_match']:15s} sim={item['similarity']:.4f}  freq={item['frequency']}")

# ── Step 8: Build extended MOOD_MAP ────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: Building extended MOOD_MAP")
print("=" * 70)

import copy

extended = copy.deepcopy(MOOD_MAP)

for a in assignments:
    extended[a["category"]].append(a["word"])

# Sort words within each category
for cat in extended:
    extended[cat] = sorted(set(extended[cat]))

# Print new additions per category
additions_by_cat = {}
for a in assignments:
    additions_by_cat.setdefault(a["category"], []).append(a["word"])

for cat in sorted(additions_by_cat):
    words = sorted(additions_by_cat[cat])
    print(f"  {cat:20s} += {words}")

# ── Step 9: Coverage stats ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9: Coverage summary")
print("=" * 70)

# After extension
extended_reverse = {}
for category, words in extended.items():
    for w in words:
        extended_reverse[w.lower()] = category

unmapped_after = {w for w in freq if w not in extended_reverse}
mapped_after = total_unique - len(unmapped_after)

print(f"  Total unique mood words:  {total_unique}")
print(f"  Mapped BEFORE:            {mapped_before}  ({mapped_before/total_unique*100:.4f}%)")
print(f"  Unmapped BEFORE:          {len(unmapped)}")
print(f"  Mapped AFTER:             {mapped_after}  ({mapped_after/total_unique*100:.4f}%)")
print(f"  Still unmapped AFTER:     {len(unmapped_after)}")
if unmapped_after:
    print(f"  Still unmapped words:     {sorted(unmapped_after)}")

# ── Output: mood_map_analysis.json ─────────────────────────────────────
analysis = {
    "total_unique_words": total_unique,
    "total_tokens": total_tokens,
    "mapped_before": mapped_before,
    "unmapped_before": {
        "count": len(unmapped),
        "words": [{"word": w, "frequency": freq[w]} for w, _ in unmapped_freq],
    },
    "mapped_after": mapped_after,
    "still_unmapped": {
        "count": len(unmapped_after),
        "words": sorted(unmapped_after),
    },
    "new_assignments": assignments,
    "needs_new_category": needs_new_cat,
}

json_path = RESEARCH_DIR / "mood_map_analysis.json"
with open(json_path, "w") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {json_path}")

# ── Output: mood_map_extended.py ───────────────────────────────────────
py_path = RESEARCH_DIR / "mood_map_extended.py"
with open(py_path, "w") as f:
    f.write('"""Extended MOOD_MAP — auto-generated by track1_fix_mood_map.py"""\n\n')
    f.write("MOOD_MAP = {\n")
    for cat in extended:
        words_repr = ", ".join(f'"{w}"' for w in extended[cat])
        f.write(f'    "{cat}": [{words_repr}],\n')
    f.write("}\n")
print(f"Saved: {py_path}")

print("\nDone.")
