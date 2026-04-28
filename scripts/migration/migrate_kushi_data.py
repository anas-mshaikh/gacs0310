#!/usr/bin/env python3
"""
Migrate Kushi's GACS labeled data to the current pipeline schema (v1.0.0).

Transforms:
  - mood_json / objects_json / style_json (string lists) → mood_1..5, object_1..5, style_1..3
  - scene_id (int) → {video_id}_s{NNNN}
  - rep_frame_path: Windows backslashes → POSIX forward slashes
  - Re-computes SBERT embeddings with all-MiniLM-L6-v2 (replacing paraphrase-MiniLM-L3-v2)

Usage:
  python migrate_kushi_data.py --dry-run   # preview without writing
  python migrate_kushi_data.py --full       # full migration with embedding recomputation
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
KUSHI_DIR = PROJECT_ROOT.parent / "gacs0202(kushi)"
KUSHI_CSV = KUSHI_DIR / "gacs_labeled_scenes.csv"
KUSHI_MANIFEST = KUSHI_DIR / "video_manifest.csv"
KUSHI_SCENES_DIR = KUSHI_DIR / "scenes"

OUTPUT_CSV = PROJECT_ROOT / "data" / "embeddings" / "gacs_dataset.csv"
OUTPUT_SCENES_DIR = PROJECT_ROOT / "data" / "scenes"
OUTPUT_MANIFEST = PROJECT_ROOT / "video_manifest.csv"


def parse_json_list(val, max_items):
    """Parse a stringified Python list like \"['a', 'b']\" into a list of strings."""
    if pd.isna(val) or not val:
        return [""] * max_items
    try:
        items = ast.literal_eval(val)
        if isinstance(items, list):
            items = [str(x).strip() for x in items]
        else:
            items = [str(items).strip()]
    except (ValueError, SyntaxError):
        items = [str(val).strip()]

    # Pad or truncate to max_items
    items = items[:max_items]
    items.extend([""] * (max_items - len(items)))
    return items


def build_video_id_from_name(video_name: str) -> str:
    """Extract video_id from filename like 'gacs_movie_trailers_00_UDfjsSqC.mp4' → 'gacs_movie_trailers_00_UDfjsSqC'."""
    return Path(video_name).stem


def build_scene_id(video_id: str, scene_num: int) -> str:
    """Format scene_id as '{video_id}_s{NNNN}'."""
    return f"{video_id}_s{scene_num:04d}"


def build_image_id(video_id: str, scene_num: int) -> str:
    """Build image_id consistent with dataset_builder convention."""
    return f"{video_id}_scene_{scene_num:02d}"


def fix_frame_path(rep_frame_path: str) -> str:
    """Convert Windows backslash paths to POSIX and make relative to data/scenes/."""
    p = str(rep_frame_path).replace("\\", "/")
    # Extract just the filename
    filename = Path(p).name
    return f"data/scenes/{filename}"


def migrate(dry_run: bool = False, recompute_embeddings: bool = True, batch_size: int = 128):
    print(f"{'[DRY-RUN] ' if dry_run else ''}Starting Kushi data migration...")
    print(f"  Source: {KUSHI_CSV}")
    print(f"  Target: {OUTPUT_CSV}")
    print()

    # Load Kushi data
    df = pd.read_csv(KUSHI_CSV)
    print(f"  Loaded {len(df)} rows, {df['video_name'].nunique()} videos")

    # Filter to label_ok == True only
    before = len(df)
    df = df[df["label_ok"] == True].copy()  # noqa: E712
    print(f"  Filtered label_ok=True: {before} → {len(df)} rows")

    # Build video_id and scene_id
    df["video_id"] = df["video_name"].apply(build_video_id_from_name)
    df["scene_num"] = df["scene_id"].astype(int)
    df["scene_id"] = df.apply(lambda r: build_scene_id(r["video_id"], r["scene_num"]), axis=1)
    df["image_id"] = df.apply(lambda r: build_image_id(r["video_id"], r["scene_num"]), axis=1)

    # Parse JSON lists into separate columns
    mood_cols = [f"mood_{i}" for i in range(1, 6)]
    style_cols = [f"style_{i}" for i in range(1, 4)]
    object_cols = [f"object_{i}" for i in range(1, 6)]

    mood_parsed = df["mood_json"].apply(lambda x: parse_json_list(x, 5))
    style_parsed = df["style_json"].apply(lambda x: parse_json_list(x, 3))
    objects_parsed = df["objects_json"].apply(lambda x: parse_json_list(x, 5))

    for i, col in enumerate(mood_cols):
        df[col] = mood_parsed.apply(lambda x: x[i])
    for i, col in enumerate(style_cols):
        df[col] = style_parsed.apply(lambda x: x[i])
    for i, col in enumerate(object_cols):
        df[col] = objects_parsed.apply(lambda x: x[i])

    # Fix rep_frame_path
    df["rep_frame_path"] = df["rep_frame_path"].apply(fix_frame_path)

    # Recompute embeddings with all-MiniLM-L6-v2
    if recompute_embeddings:
        print()
        print("  Recomputing SBERT embeddings (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Build mood text for embedding (same as mood_str)
        mood_texts = df["mood_str"].fillna("").tolist()
        print(f"  Encoding {len(mood_texts)} mood strings (batch_size={batch_size})...")

        embeddings = model.encode(mood_texts, batch_size=batch_size, show_progress_bar=True)
        df["mood_embedding"] = [json.dumps(emb.tolist()) for emb in embeddings]
        print(f"  Embedding dim: {embeddings.shape[1]}")
    else:
        # Keep original embeddings (different model, different dim)
        print("  Skipping embedding recomputation (using original).")
        df["mood_embedding"] = df["embedding"]

    # Select output columns in schema order
    output_cols = (
        ["image_id", "video_id", "scene_id", "rep_frame_path",
         "start_time", "end_time", "duration"]
        + mood_cols + style_cols + object_cols
        + ["mood_embedding"]
    )
    out = df[output_cols].copy()

    print()
    print(f"  Output: {len(out)} rows, {out['video_id'].nunique()} videos")
    print(f"  Columns: {list(out.columns)}")
    print(f"  mood_1 non-empty: {(out['mood_1'] != '').sum()}")
    print(f"  Embedding sample: {str(out['mood_embedding'].iloc[0])[:60]}...")

    if dry_run:
        print()
        print("[DRY-RUN] No files written. Run with --full to execute.")
        return out

    # Write output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Written: {OUTPUT_CSV} ({len(out)} rows)")

    # Copy video_manifest.csv
    if KUSHI_MANIFEST.exists():
        manifest = pd.read_csv(KUSHI_MANIFEST)
        # Fix Windows paths
        if "local_path" in manifest.columns:
            manifest["local_path"] = manifest["local_path"].apply(
                lambda x: str(x).replace("\\", "/") if pd.notna(x) else x
            )
        manifest.to_csv(OUTPUT_MANIFEST, index=False)
        print(f"  Written: {OUTPUT_MANIFEST} ({len(manifest)} rows)")

    return out


def main():
    parser = argparse.ArgumentParser(description="Migrate Kushi GACS data to pipeline schema v1.0.0")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Preview migration without writing files")
    group.add_argument("--full", action="store_true", help="Full migration with embedding recomputation")
    parser.add_argument("--batch-size", type=int, default=128, help="SBERT encoding batch size")

    args = parser.parse_args()

    if not KUSHI_CSV.exists():
        print(f"ERROR: Kushi data not found at {KUSHI_CSV}")
        sys.exit(1)

    migrate(dry_run=args.dry_run, recompute_embeddings=not args.dry_run, batch_size=args.batch_size)
    print("\nDone.")


if __name__ == "__main__":
    main()
