"""
Final retry: download missing videos without --quiet, with proper file matching.
"""
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import pandas as pd

MANIFEST = Path("gacs_0202/video_manifest.csv")
OUTPUT_DIR = Path("gacs_0202/data/raw_videos")
FFMPEG_DIR = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)

df = pd.read_csv(MANIFEST)

# Find truly missing
existing = {f.stem for f in OUTPUT_DIR.glob("*.mp4") if f.stat().st_size > 10000}
to_download = df[~df["video_id"].isin(existing)]
print(f"Already have: {len(existing)}")
print(f"Need to download: {len(to_download)}")
print(f"ffmpeg dir: {FFMPEG_DIR}")
print()

success = 0
failed = []

for idx, (_, row) in enumerate(to_download.iterrows()):
    video_id = row["video_id"]
    url = row["source_url"]
    output_path = OUTPUT_DIR / f"{video_id}.mp4"

    print(f"[{idx+1}/{len(to_download)}] {video_id} ...", end=" ", flush=True)

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "yt_dlp",
                "--ffmpeg-location", FFMPEG_DIR,
                "-f", "best[height<=720][ext=mp4]/best[height<=720]/best",
                "-o", str(output_path),
                "--no-playlist",
                "--socket-timeout", "30",
                "--retries", "3",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Check for any mp4 that starts with this video_id
        matches = list(OUTPUT_DIR.glob(f"{video_id}*"))
        found = None
        for m in matches:
            if m.suffix == ".mp4" and m.stat().st_size > 10000:
                if m.name != f"{video_id}.mp4":
                    m.rename(output_path)
                found = output_path
                break

        if found or (output_path.exists() and output_path.stat().st_size > 10000):
            size_mb = output_path.stat().st_size / (1024 * 1024)
            success += 1
            print(f"OK ({size_mb:.1f} MB)")
        else:
            failed.append(video_id)
            # Extract meaningful error
            stderr = result.stderr or ""
            for line in stderr.split("\n"):
                if "ERROR" in line:
                    print(f"FAIL: {line.strip()[:100]}")
                    break
            else:
                print(f"FAIL: rc={result.returncode}")

    except subprocess.TimeoutExpired:
        failed.append(video_id)
        print("TIMEOUT")
    except Exception as e:
        failed.append(video_id)
        print(f"ERROR: {e}")

    time.sleep(0.5)

# Final tally
all_valid = [f for f in OUTPUT_DIR.glob("*.mp4") if f.stat().st_size > 10000]
total_gb = sum(f.stat().st_size for f in all_valid) / (1024**3)
matched = {f.stem for f in all_valid} & set(df["video_id"])

print()
print("=" * 60)
print("FINAL RESULT")
print(f"  New downloads: {success}")
print(f"  Failed: {len(failed)}")
print(f"  Total valid mp4: {len(all_valid)}")
print(f"  Matched to manifest: {len(matched)} / {len(df)}")
print(f"  Total size: {total_gb:.2f} GB")
print(f"  Coverage: {len(matched)/len(df)*100:.1f}%")
if failed:
    print(f"\n  Unavailable ({len(failed)}):")
    for fid in failed:
        print(f"    {fid}")
