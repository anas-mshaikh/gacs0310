import json
import os
import re
import time
from pathlib import Path

import google.generativeai as genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

DATA_DIR = Path("data")
SCENES_DIR = Path("data/scenes")
scene_meta_path = DATA_DIR / "scene_metadata.csv"
labeled_path = DATA_DIR / "gacs_labeled_scenes.csv"

if not scene_meta_path.exists():
    print("scene_metadata.csv not found. Run frame_extraction.py first.")
    raise SystemExit(1)

scenes_df = pd.read_csv(scene_meta_path)

if labeled_path.exists():
    labeled_df = pd.read_csv(labeled_path)
    labeled_set = set(zip(labeled_df["video_name"], labeled_df["scene_id"]))
    new_scenes = scenes_df[~scenes_df.apply(lambda row: (row["video_name"], row["scene_id"]) in labeled_set, axis=1)]
    print(f"Found {len(scenes_df)} total scenes, {len(new_scenes)} new to label")
else:
    new_scenes = scenes_df
    print(f"Labeling all {len(new_scenes)} new scenes")

# LOAD EXISTING LABELS (for instant merge)
existing_labels = pd.read_csv(labeled_path) if labeled_path.exists() else pd.DataFrame()


def extract_json_from_response(raw_response: str) -> dict:
    """Robust JSON extraction from Gemini response."""
    if not raw_response:
        return {"mood": [], "objects": [], "style": [], "label_ok": False}

    if isinstance(raw_response, list):
        raw_response = raw_response[0] if raw_response else ""
    raw_response = str(raw_response).strip()

    # Method 1: Extract from `````` blocks
    json_match = re.search(r'``````', raw_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Method 2: Largest JSON object
        json_match = re.search(r'\{[^{}]*"[^"]*"\s*:\s*\[[^\]]*\][^{}]*\}', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Method 3: First complete JSON
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw_response

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"JSON parse failed: {json_str[:200]}...")
        return {"mood": [], "objects": [], "style": [], "label_ok": False}


def label_scene_with_gemini(image_path: Path):
    """Extract mood/objects/style per GACS spec."""
    if not image_path.exists():
        print(f"Missing: {image_path}")
        return {"mood": [], "objects": [], "style": [], "label_ok": False}

    try:
        img = genai.upload_file(str(image_path))
    except Exception as e:
        print(f"Upload failed {image_path.name}: {e}")
        return {"mood": [], "objects": [], "style": [], "label_ok": False}

    prompt = """
You are labeling a single video frame for an affective video dataset.

Return ONLY valid JSON in this exact structure:

{
  "mood": ["dark", "tense", "mysterious"],
  "objects": ["sword", "castle", "knight"],
  "style": ["cinematic", "medieval", "dramatic"]
}

Rules:
- mood: 3–5 adjectives describing mood/feeling
- objects: 3–5 nouns naming main visible objects  
- style: 3 adjectives describing visual style
- lowercase English words only
- NO extra text/keys/markdown
"""

    try:
        response = model.generate_content([prompt, img])
        parsed_data = extract_json_from_response(response.text)
        time.sleep(1)  # Rate limiting safety
        return {
            "mood": parsed_data.get("mood", []),
            "objects": parsed_data.get("objects", []),
            "style": parsed_data.get("style", []),
            "label_ok": bool(parsed_data.get("mood") and parsed_data.get("objects") and parsed_data.get("style")),
        }
    except Exception as e:
        print(f"Gemini API error {image_path.name}: {e}")
        return {"mood": [], "objects": [], "style": [], "label_ok": False}


# INSTANT SAVE AFTER EVERY SCENE
print("Starting incremental labeling with instant saves...")
total_labeled = len(existing_labels)
save_every = 10

for idx, row in new_scenes.iterrows():
    print(f"[{idx + 1}/{len(new_scenes)}] {row['video_name']} scene {row['scene_id']}")

    img_path = Path(row["rep_frame_path"])
    labels = label_scene_with_gemini(img_path)

    # INSTANT ADD TO EXISTING
    new_row = {
        **row.to_dict(),
        "mood_json": labels["mood"],
        "objects_json": labels["objects"],
        "style_json": labels["style"],
        "mood_str": " ".join(labels["mood"]),
        "objects_str": " ".join(labels["objects"]),
        "style_str": " ".join(labels["style"]),
        "label_ok": labels["label_ok"],
    }

    # INSTANT MERGE & SAVE
    temp_df = pd.concat([existing_labels, pd.DataFrame([new_row])], ignore_index=True)
    temp_df = temp_df.drop_duplicates(subset=["video_name", "scene_id", "rep_frame_path"], keep="last")
    temp_df.to_csv(labeled_path, index=False)
    existing_labels = temp_df  # Update in-memory

    total_labeled += 1
    if (idx + 1) % save_every == 0:
        print(f"Saved {total_labeled} total scenes | {len(new_scenes) - idx - 1} left")

print(f"\nLabeling COMPLETE! {total_labeled} total scenes labeled")
print(f"CSV: {labeled_path}")

# COMPUTE EMBEDDINGS ONLY AT END
print("\nComputing embeddings for ALL labeled scenes...")
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

final_df = pd.read_csv(labeled_path)
sentences = [f"{row['mood_str']} {row['objects_str']} {row['style_str']}".strip() or "neutral scene"
             for _, row in final_df.iterrows()]
all_embeddings = embedder.encode(sentences)

final_df["combined_text"] = sentences

# FINAL SAVES
embeddings_path = DATA_DIR / "gacs_embeddings.npy"
final_df.to_csv(labeled_path, index=False)
np.save(embeddings_path, all_embeddings)

print(f"{len(final_df)} total labeled scenes")
print(f"Embeddings: {embeddings_path}")
