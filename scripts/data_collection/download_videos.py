"""
Download all videos from video_manifest.csv using yt-dlp.
Saves to gacs_0202/data/raw_videos/
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

MANIFEST = Path("gacs_0202/video_manifest.csv")
OUTPUT_DIR = Path("gacs_0202/data/raw_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = Path("download_log.txt")

df = pd.read_csv(MANIFEST)
print(f"Total videos to download: {len(df)}")
print(f"Output directory: {OUTPUT_DIR}")
print()

success = 0
skipped = 0
failed = []

with open(LOG_FILE, "w") as log:
    for i, row in df.iterrows():
        video_id = row["video_id"]
        url = row["source_url"]
        output_path = OUTPUT_DIR / f"{video_id}.mp4"

        # Skip if already downloaded
        if output_path.exists() and output_path.stat().st_size > 10000:
            skipped += 1
            msg = f"[{i+1}/{len(df)}] SKIP (exists): {video_id}"
            print(msg)
            log.write(msg + "\n")
            continue

        print(f"[{i+1}/{len(df)}] Downloading: {video_id}")
        print(f"  URL: {url}")

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "yt_dlp",
                    "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best",
                    "--merge-output-format", "mp4",
                    "-o", str(output_path),
                    "--no-playlist",
                    "--socket-timeout", "30",
                    "--retries", "3",
                    "--quiet",
                    "--no-warnings",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if output_path.exists() and output_path.stat().st_size > 10000:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                success += 1
                msg = f"[{i+1}/{len(df)}] OK: {video_id} ({size_mb:.1f} MB)"
                print(f"  {msg}")
                log.write(msg + "\n")
            else:
                failed.append(video_id)
                err = result.stderr[:200] if result.stderr else "unknown"
                msg = f"[{i+1}/{len(df)}] FAIL: {video_id} — {err}"
                print(f"  {msg}")
                log.write(msg + "\n")

        except subprocess.TimeoutExpired:
            failed.append(video_id)
            msg = f"[{i+1}/{len(df)}] TIMEOUT: {video_id}"
            print(f"  {msg}")
            log.write(msg + "\n")
        except Exception as e:
            failed.append(video_id)
            msg = f"[{i+1}/{len(df)}] ERROR: {video_id} — {e}"
            print(f"  {msg}")
            log.write(msg + "\n")

        # Small delay to avoid rate limiting
        time.sleep(1)

print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print(f"  Success: {success}")
print(f"  Skipped (already existed): {skipped}")
print(f"  Failed: {len(failed)}")
if failed:
    print(f"  Failed IDs: {failed[:10]}{'...' if len(failed) > 10 else ''}")
print(f"  Log: {LOG_FILE}")

# Count total downloaded
existing = list(OUTPUT_DIR.glob("*.mp4"))
total_size = sum(f.stat().st_size for f in existing) / (1024 * 1024 * 1024)
print(f"  Total files in raw_videos/: {len(existing)}")
print(f"  Total size: {total_size:.2f} GB")
