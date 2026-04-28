import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yt_dlp

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_videos"
MANIFEST_PATH = DATA_DIR / "video_manifest.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def discover_gacs_videos(limit_per_category=5):
    """GACS categories: trailers, ads, emotional shorts, animations"""

    CATEGORIES = {
        "movie_trailers": [
            "Wake Up Dead Man John Wick 5",
            "Mufasa Lion King prequel trailer",
            "Captain America Brave New World",
            "Thunderbolts Marvel trailer",
            "Shankara cinematic universe trailer",
        ],
        "advertisements": [
            "Tiffany Blue Book 2026 campaign",
            "Hermes Birkin bag commercial",
            "Lamborghini Revuelto supercar ad",
            "Apple Vision Pro 2 ad",
            "Moncler genius fashion show",
        ],
        "emotional_shorts": [
            "Blind painter sees colors emotional",
            "WW2 veteran last parade story",
            "Orphaned foal adopted by horse",
            "Refugee family first home moment",
            "Elderly painter final masterpiece",
        ],
        "animations": [
            "Mufasa Lion King animated scenes",
            "How to Train Dragon live trailer",
            "Puss in Boots 3 concept",
            "Tales of Arcadians animated short",
            "Over the Moon sequel trailer",
        ],
    }

    print("GACS Versatile Video Discovery")
    videos = []
    scrape_date = datetime.utcnow().strftime("%Y-%m-%d")

    for category, queries in CATEGORIES.items():
        print(f"\n {category.upper()} ({limit_per_category} videos)")

        for i, query in enumerate(queries[:limit_per_category]):
            print(f"   [{i + 1}] '{query}'")

            search_url = f"ytsearch1:{query}"
            ydl_opts = {"quiet": True, "no_warnings": True}

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(search_url, download=False)
                    if info.get("entries") and info["entries"][0]:
                        entry = info["entries"][0]
                        vid = entry["id"]
                        video_id = f"gacs_{category}_{i:02d}_{vid[:8]}"

                        videos.append({
                            "video_id": video_id,
                            "source": f"youtube_{category}",
                            "source_url": f"https://youtube.com/watch?v={vid}",
                            "title": entry.get("title", "Unknown")[:100],
                            "category": category,
                            "duration": entry.get("duration", 0),
                            "view_count": entry.get("view_count", 0),
                            "uploader": entry.get("uploader", ""),
                            "upload_date": entry.get("upload_date", ""),
                            "scrape_date": scrape_date,
                        })
                        print(f"      {entry.get('title', 'Unknown')[:60]}...")
            except Exception as e:
                print(f"      {e}")

            time.sleep(random.uniform(1, 2))

    return videos


def download_youtube_video(video_meta: dict, manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Download with GACS + metadata, update manifest incrementally."""
    video_id = video_meta["video_id"]
    youtube_url = video_meta["source_url"]
    title = video_meta["title"]
    category = video_meta["category"]
    duration = video_meta["duration"]

    existing = manifest_df[manifest_df["video_id"] == video_id]
    if not existing.empty and bool(existing.iloc[0].get("download_ok", False)):
        print(f"{video_id}: already downloaded")
        return manifest_df

    local_path_template = RAW_DIR / f"{video_id}.%(ext)s"
    print(f"[{category}] {title[:50]}...")

    ydl_opts = {
        "format": "best[height<=720]",
        "outtmpl": str(local_path_template),
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        actual_file = RAW_DIR / f"{video_id}.mp4"
        if actual_file.exists() and actual_file.stat().st_size > 1024 * 500:
            print(f"{video_id} ({actual_file.stat().st_size / 1024 / 1024:.1f}MB)")
            row = {
                **video_meta,
                "local_path": str(actual_file),
                "download_ok": True,
                "processed": False,
            }
        else:
            print(f"{video_id}: invalid file")
            row = {
                **video_meta,
                "local_path": "",
                "download_ok": False,
                "processed": False,
            }
    except Exception as e:
        print(f"{video_id}: {e}")
        row = {
            **video_meta,
            "local_path": "",
            "download_ok": False,
            "processed": False,
        }

    return upsert_manifest_row(manifest_df, row)


def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH)
    return pd.DataFrame(columns=[
        "video_id", "source", "source_url", "local_path", "title",
        "category", "duration", "view_count", "uploader",
        "upload_date", "scrape_date", "download_ok", "processed",
    ])


def upsert_manifest_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    df = df.copy()
    idx = df.index[df["video_id"] == row["video_id"]].tolist()
    if idx:
        df.loc[idx[0]] = row
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def save_manifest(df: pd.DataFrame):
    df.to_csv(MANIFEST_PATH, index=False)
    #df.to_excel(DATA_DIR / "video_manifest.xlsx", index=False)


if __name__ == "__main__":
    # 1) Discover diverse GACS videos with metadata
    videos = discover_gacs_videos(limit_per_category=5)

    # 2) Load existing manifest (incremental)
    manifest = load_manifest()
    print(f"\nLoaded manifest: {len(manifest)} entries")

    # 3) Download new videos + update manifest
    for video in videos:
        print(f"\n--- {video['category'].upper()}: {video['title'][:50]} ---")
        manifest = download_youtube_video(video_meta=video, manifest_df=manifest)
        time.sleep(random.uniform(3, 6))

    # 4) Save manifest
    save_manifest(manifest)

    successful = manifest[manifest["download_ok"] == True]
    print("\nGACS DOWNLOAD COMPLETE!")
    print(f"{len(successful)}/{len(manifest)} videos ready")
    print("By category:")
    for cat in successful["category"].dropna().unique():
        count = len(successful[successful["category"] == cat])
        print(f"   {cat}: {count}")
