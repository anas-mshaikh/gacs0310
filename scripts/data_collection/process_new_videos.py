"""
Stage 1 Processing for New Videos
==================================
Processes unprocessed videos through the full Stage 1 pipeline:
  1. Clean partial download artifacts
  2. Scene detection (PySceneDetect)
  3. Keyframe extraction (OpenCV)
  4. Multi-model labeling (Claude, GPT-4o, Gemini)
  5. SBERT embedding generation
  6. Append to dataset CSV

Usage:
    python process_new_videos.py                    # Full pipeline
    python process_new_videos.py --phase scenes     # Scene detection only
    python process_new_videos.py --phase label      # Labeling only (after scenes)
    python process_new_videos.py --phase embed      # Embedding only (after labels)
    python process_new_videos.py --limit 5          # Process only 5 videos
"""
import argparse
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path("gacs_0202/.env"))

# Paths
RAW_VIDEOS_DIR = Path("gacs_0202/data/raw_videos")
SCENES_DIR = Path("gacs_0202/data/scenes")
ANNOTATIONS_DIR = Path("gacs_0202/data/annotations")
EMBEDDINGS_DIR = Path("gacs_0202/data/embeddings")
DATASET_CSV = EMBEDDINGS_DIR / "gacs_dataset.csv"
LOG_DIR = Path("research/reports")

ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Prompts
MOOD_PROMPT = """Analyze the mood/emotion of this video frame.
Return EXACTLY 5 mood descriptor words, comma-separated.
Example: tense, mysterious, dark, suspenseful, foreboding
Output ONLY the 5 words, nothing else."""

STYLE_PROMPT = """Analyze the visual style of this video frame.
Return EXACTLY 3 style descriptor words, comma-separated.
Example: cinematic, warm-toned, shallow-focus
Output ONLY the 3 words, nothing else."""

OBJECT_PROMPT = """List the main objects/subjects visible in this video frame.
Return 2-5 object names, comma-separated.
Example: car, road, sunset, mountains
Output ONLY the object names, nothing else."""


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================================
# Phase 0: Clean artifacts
# ============================================================================

def clean_artifacts():
    """Remove partial download files (.fXXX.mp4)."""
    partials = [f for f in RAW_VIDEOS_DIR.glob("*.mp4") if ".f" in f.stem]
    if partials:
        log(f"Cleaning {len(partials)} partial artifacts...")
        for p in partials:
            p.unlink()
        log(f"Cleaned {len(partials)} files")
    else:
        log("No partial artifacts to clean")


# ============================================================================
# Phase 1: Scene Detection + Keyframe Extraction
# ============================================================================

def get_unprocessed_videos() -> list:
    """Find videos not yet in gacs_dataset.csv."""
    if DATASET_CSV.exists():
        df = pd.read_csv(DATASET_CSV)
        processed_ids = set(df["video_id"].unique())
    else:
        processed_ids = set()

    all_videos = sorted(RAW_VIDEOS_DIR.glob("*.mp4"))
    valid = [v for v in all_videos if v.stat().st_size > 10000 and ".f" not in v.stem]
    unprocessed = [v for v in valid if v.stem not in processed_ids]
    return unprocessed


def detect_scenes_and_extract(video_path: Path) -> list:
    """Detect scenes and extract keyframes for a single video."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video_id = video_path.stem
    scenes_info = []

    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            log(f"  No scenes detected in {video_id}")
            return []

        # Save scene metadata
        scene_dir = SCENES_DIR / video_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        cap = cv2.VideoCapture(str(video_path))

        for idx, (start, end) in enumerate(scene_list):
            scene_id = f"{video_id}_s{idx:04d}"
            start_time = start.get_seconds()
            end_time = end.get_seconds()
            mid_time = (start_time + end_time) / 2

            # Extract keyframe at midpoint
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
            ret, frame = cap.read()
            if ret:
                keyframe_name = f"{video_id}_scene_{idx:02d}.jpg"
                keyframe_path = SCENES_DIR / keyframe_name
                cv2.imwrite(str(keyframe_path), frame)

                scene_info = {
                    "scene_id": scene_id,
                    "video_id": video_id,
                    "scene_index": idx,
                    "start_frame": start.get_frames(),
                    "end_frame": end.get_frames(),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "keyframe_path": str(keyframe_path),
                }
                metadata.append(scene_info)
                scenes_info.append(scene_info)

        cap.release()

        # Save scene_metadata.json
        with open(scene_dir / "scene_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return scenes_info

    except Exception as e:
        log(f"  ERROR in scene detection for {video_id}: {e}")
        return []


def phase_scenes(videos: list) -> list:
    """Phase 1: Scene detection + keyframe extraction."""
    log(f"PHASE 1: Scene Detection ({len(videos)} videos)")
    all_scenes = []

    for i, vpath in enumerate(videos):
        log(f"  [{i+1}/{len(videos)}] {vpath.stem}")
        scenes = detect_scenes_and_extract(vpath)
        all_scenes.extend(scenes)
        log(f"    {len(scenes)} scenes detected")

    log(f"Phase 1 complete: {len(all_scenes)} total scenes from {len(videos)} videos")
    return all_scenes


# ============================================================================
# Phase 2: Multi-model Labeling
# ============================================================================

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def parse_words(text: str) -> list:
    # Clean up common issues
    text = text.strip().strip('"').strip("'")
    words = [w.strip().lower().rstrip(".").strip('"').strip("'")
             for w in text.split(",") if w.strip()]
    # Filter out non-word responses
    words = [w for w in words if len(w) < 30 and not w.startswith("i ")]
    return words


def call_claude(image_path: Path, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    img_data = encode_image(image_path)
    media_type = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return msg.content[0].text.strip()


def call_openai(image_path: Path, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    img_data = encode_image(image_path)
    media_type = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=128,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_data}"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return resp.choices[0].message.content.strip()


def call_gemini(image_path: Path, prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    from PIL import Image
    img = Image.open(image_path)
    resp = model.generate_content([prompt, img])
    return resp.text.strip()


MODEL_CALLERS = {
    "claude": call_claude,
    "gpt4o": call_openai,
    "gemini": call_gemini,
}


def label_single_image(image_path: Path, model_name: str) -> dict:
    """Label a single image with one model. Returns dict with mood, style, objects."""
    caller = MODEL_CALLERS[model_name]
    result = {"model": model_name}

    try:
        mood_text = caller(image_path, MOOD_PROMPT)
        result["mood"] = parse_words(mood_text)
        time.sleep(0.3)

        style_text = caller(image_path, STYLE_PROMPT)
        result["style"] = parse_words(style_text)
        time.sleep(0.3)

        obj_text = caller(image_path, OBJECT_PROMPT)
        result["objects"] = parse_words(obj_text)
        time.sleep(0.3)

    except Exception as e:
        log(f"    {model_name} ERROR: {e}")
        result.setdefault("mood", [])
        result.setdefault("style", [])
        result.setdefault("objects", [])

    return result


def phase_label(scenes: list, models: list = None) -> dict:
    """Phase 2: Label all keyframes with specified models."""
    if models is None:
        models = ["claude", "gpt4o", "gemini"]

    log(f"PHASE 2: Labeling ({len(scenes)} scenes × {len(models)} models)")
    all_labels = {}  # image_name -> {model: {mood, style, objects}}

    for i, scene in enumerate(scenes):
        kf_path = Path(scene["keyframe_path"])
        if not kf_path.exists():
            log(f"  [{i+1}/{len(scenes)}] SKIP (no keyframe): {scene['scene_id']}")
            continue

        image_name = kf_path.name
        cache_path = ANNOTATIONS_DIR / f"{kf_path.stem}_multi.json"

        # Check cache
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            all_labels[image_name] = cached
            if (i + 1) % 50 == 0:
                log(f"  [{i+1}/{len(scenes)}] cached: {scene['scene_id']}")
            continue

        log(f"  [{i+1}/{len(scenes)}] {scene['scene_id']}")
        labels = {}
        for model_name in models:
            result = label_single_image(kf_path, model_name)
            labels[model_name] = result
            mood_str = ", ".join(result["mood"][:3])
            log(f"    {model_name:8s}: {mood_str}...")
            time.sleep(0.5)  # Rate limit buffer between models

        # Cache labels
        with open(cache_path, "w") as f:
            json.dump(labels, f, indent=2)

        all_labels[image_name] = labels

    log(f"Phase 2 complete: {len(all_labels)} images labeled")
    return all_labels


# ============================================================================
# Phase 3: Embedding + Dataset Update
# ============================================================================

def phase_embed(scenes: list, labels: dict):
    """Phase 3: Generate SBERT embeddings and update dataset CSV."""
    from sentence_transformers import SentenceTransformer

    log("PHASE 3: Embedding generation")

    # Load SBERT - use same model as existing dataset for compatibility
    sbert_384 = SentenceTransformer("all-MiniLM-L6-v2")  # 384d, matches existing
    log("  Loaded all-MiniLM-L6-v2 (384d)")

    # Also load emotion model for multi-model coordinates
    sbert_768 = SentenceTransformer("j-hartmann/emotion-english-distilroberta-base")
    log("  Loaded emotion-distilroberta (768d)")

    rows = []
    for scene in scenes:
        kf_path = Path(scene["keyframe_path"])
        image_name = kf_path.name

        if image_name not in labels:
            continue

        scene_labels = labels[image_name]

        # Use Claude labels for main dataset (backward compatible with existing data)
        claude_labels = scene_labels.get("claude", {})
        mood_words = claude_labels.get("mood", [])
        style_words = claude_labels.get("style", [])
        objects = claude_labels.get("objects", [])

        if not mood_words:
            # Fallback to GPT-4o if Claude failed
            gpt_labels = scene_labels.get("gpt4o", {})
            mood_words = gpt_labels.get("mood", [])
            style_words = gpt_labels.get("style", [])
            objects = gpt_labels.get("objects", [])

        if not mood_words:
            continue

        # Generate embedding (384d for main dataset compatibility)
        mood_sentence = " ".join(mood_words)
        embedding_384 = sbert_384.encode(mood_sentence).tolist()

        # Build row matching existing CSV schema
        row = {
            "image_id": kf_path.stem,
            "video_id": scene["video_id"],
            "scene_id": scene["scene_id"],
            "rep_frame_path": f"data/scenes/{image_name}",
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "duration": scene["duration"],
        }

        # Mood columns (pad to 5)
        for j in range(5):
            row[f"mood_{j+1}"] = mood_words[j] if j < len(mood_words) else ""

        # Style columns (pad to 3)
        for j in range(3):
            row[f"style_{j+1}"] = style_words[j] if j < len(style_words) else ""

        # Object columns (pad to 5)
        for j in range(5):
            row[f"object_{j+1}"] = objects[j] if j < len(objects) else ""

        row["mood_embedding"] = json.dumps(embedding_384)
        rows.append(row)

        # Also save multi-model embeddings separately
        multi_emb = {}
        for model_name in ["claude", "gpt4o", "gemini"]:
            model_labels = scene_labels.get(model_name, {})
            model_mood = model_labels.get("mood", [])
            if model_mood:
                sentence = " ".join(model_mood)
                emb = sbert_768.encode(sentence).tolist()
                multi_emb[model_name] = {
                    "mood": model_mood,
                    "embedding_768": emb,
                }

        emb_cache_path = ANNOTATIONS_DIR / f"{kf_path.stem}_embeddings.json"
        with open(emb_cache_path, "w") as f:
            json.dump(multi_emb, f)

    if not rows:
        log("No rows to add")
        return

    new_df = pd.DataFrame(rows)
    log(f"  New rows: {len(new_df)}")

    # Append to existing dataset
    if DATASET_CSV.exists():
        existing_df = pd.read_csv(DATASET_CSV)
        # Remove any duplicates
        existing_ids = set(existing_df["scene_id"])
        new_df = new_df[~new_df["scene_id"].isin(existing_ids)]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Backup existing
    if DATASET_CSV.exists():
        backup = DATASET_CSV.with_suffix(".csv.bak")
        DATASET_CSV.rename(backup)
        log(f"  Backed up existing dataset to {backup.name}")

    combined_df.to_csv(DATASET_CSV, index=False)
    log(f"  Dataset updated: {len(combined_df)} total rows ({len(new_df)} new)")

    return new_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process new videos through Stage 1")
    parser.add_argument("--phase", choices=["clean", "scenes", "label", "embed", "all"],
                        default="all", help="Which phase to run")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos")
    parser.add_argument("--models", nargs="+", default=["claude", "gpt4o", "gemini"],
                        help="Models for labeling")
    args = parser.parse_args()

    log("=" * 70)
    log("STAGE 1: PROCESS NEW VIDEOS")
    log("=" * 70)
    log(f"Phase: {args.phase}")
    log(f"Models: {args.models}")
    start_time = datetime.now()

    # Clean artifacts
    if args.phase in ("clean", "all"):
        clean_artifacts()

    # Get unprocessed videos
    videos = get_unprocessed_videos()
    if args.limit > 0:
        videos = videos[:args.limit]
    log(f"Videos to process: {len(videos)}")

    if not videos:
        log("No new videos to process!")
        return

    # Phase 1: Scene detection
    scenes = []
    if args.phase in ("scenes", "all"):
        scenes = phase_scenes(videos)

        # Save scenes list for later phases
        scenes_cache = Path("research/new_scenes_cache.json")
        with open(scenes_cache, "w") as f:
            json.dump(scenes, f, indent=2)
        log(f"Scenes cached to {scenes_cache}")

    # Load cached scenes if running label/embed only
    if args.phase in ("label", "embed") and not scenes:
        scenes_cache = Path("research/new_scenes_cache.json")
        if scenes_cache.exists():
            with open(scenes_cache) as f:
                scenes = json.load(f)
            log(f"Loaded {len(scenes)} scenes from cache")
        else:
            log("ERROR: No scenes cache found. Run --phase scenes first.")
            return

    # Phase 2: Labeling
    labels = {}
    if args.phase in ("label", "all"):
        labels = phase_label(scenes, args.models)

    # Load cached labels if running embed only
    if args.phase == "embed" and not labels:
        log("Loading cached labels from annotations...")
        for scene in scenes:
            kf_path = Path(scene["keyframe_path"])
            cache_path = ANNOTATIONS_DIR / f"{kf_path.stem}_multi.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    labels[kf_path.name] = json.load(f)
        log(f"Loaded {len(labels)} cached labels")

    # Phase 3: Embedding
    if args.phase in ("embed", "all"):
        phase_embed(scenes, labels)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    log("")
    log("=" * 70)
    log("PROCESSING COMPLETE")
    log(f"  Videos processed: {len(videos)}")
    log(f"  Scenes detected: {len(scenes)}")
    log(f"  Labels generated: {len(labels)}")
    log(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log("=" * 70)


if __name__ == "__main__":
    main()
