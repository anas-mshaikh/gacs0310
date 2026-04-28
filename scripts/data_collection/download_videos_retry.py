"""
Retry failed video downloads with explicit ffmpeg path.
Also cleans up partial .f*.mp4 files.
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

MANIFEST = Path("gacs_0202/video_manifest.csv")
OUTPUT_DIR = Path("gacs_0202/data/raw_videos")

# Get ffmpeg path from imageio_ffmpeg
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
print(f"ffmpeg: {FFMPEG}")

# Clean up partial files
partials = list(OUTPUT_DIR.glob("*.f*.mp4")) + list(OUTPUT_DIR.glob("*.f*.webm")) + list(OUTPUT_DIR.glob("*.part"))
if partials:
    print(f"Cleaning {len(partials)} partial files...")
    for p in partials:
        p.unlink()

df = pd.read_csv(MANIFEST)

# Find missing videos
existing = {f.stem for f in OUTPUT_DIR.glob("*.mp4") if f.stat().st_size > 10000}
to_download = df[~df["video_id"].isin(existing)]
print(f"Already downloaded: {len(existing)}")
print(f"To download: {len(to_download)}")
print()

success = 0
failed = []

for i, row in to_download.iterrows():
    video_id = row["video_id"]
    url = row["source_url"]
    output_path = OUTPUT_DIR / f"{video_id}.mp4"

    print(f"[{success + len(failed) + 1}/{len(to_download)}] {video_id}")

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "yt_dlp",
                "--ffmpeg-location", str(Path(FFMPEG).parent),
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
            print(f"  OK ({size_mb:.1f} MB)")
        else:
            failed.append(video_id)
            err = result.stderr[:200] if result.stderr else "unknown"
            print(f"  FAIL: {err}")

    except subprocess.TimeoutExpired:
        failed.append(video_id)
        print("  TIMEOUT")
    except Exception as e:
        failed.append(video_id)
        print(f"  ERROR: {e}")

    time.sleep(1)

# Final count
all_files = list(OUTPUT_DIR.glob("*.mp4"))
valid = [f for f in all_files if f.stat().st_size > 10000]
total_gb = sum(f.stat().st_size for f in valid) / (1024**3)

print()
print("=" * 60)
print("RETRY COMPLETE")
print(f"  New downloads: {success}")
print(f"  Failed: {len(failed)}")
print(f"  Total valid mp4: {len(valid)} / {len(df)}")
print(f"  Total size: {total_gb:.2f} GB")
if failed:
    print(f"  Still missing ({len(failed)}):")
    for fid in failed[:20]:
        print(f"    {fid}")
