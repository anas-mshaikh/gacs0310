from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_DIR = Path("data/raw_videos")
FRAMES_DIR = Path("data/frames")
SCENES_DIR = Path("data/scenes")
DATA_DIR = Path("data")
MANIFEST_PATH = DATA_DIR / "video_manifest.csv"

DATA_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
SCENES_DIR.mkdir(exist_ok=True)


def load_video_manifest():
    """Load manifest to check which videos are unprocessed."""
    if MANIFEST_PATH.exists():
        manifest = pd.read_csv(MANIFEST_PATH)
        return manifest
    return pd.DataFrame()


def mark_video_processed(manifest_df: pd.DataFrame, video_file: Path) -> pd.DataFrame:
    """Mark video as processed in manifest."""
    manifest_df = manifest_df.copy()
    local_path = str(video_file)

    # Find matching row by local_path
    mask = manifest_df["local_path"] == local_path
    if mask.any():
        manifest_df.loc[mask, "processed"] = True
        manifest_df.to_csv(MANIFEST_PATH, index=False)

    return manifest_df


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        print(f"Invalid metadata: {video_path} (fps={fps}, frames={frame_count})")
        return None, None, None, None

    duration = frame_count / fps
    return fps, duration, width, height


def extract_frames(video_path, fps=2, max_frames=3000):
    """Extract frames with safety cap."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot extract frames: {video_path}")
        return []

    frames = []
    frame_idx = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps <= 0:
        print(f"Invalid FPS={video_fps}: {video_path}")
        cap.release()
        return []

    sample_rate = max(1, int(video_fps / fps))

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            frames.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / video_fps,
                "frame": frame.copy(),
            })
        frame_idx += 1

    cap.release()
    if frame_idx >= max_frames:
        print(f"Hit max_frames={max_frames} for {video_path.name}")
    return frames


def detect_scenes_fixed_interval(frames, video_path):
    """8-second scenes with black-frame filtering (OpenCV)."""
    scenes = []
    if not frames:
        return scenes

    scene_interval = 8.0
    total_duration = frames[-1]["timestamp"]
    start_times = np.arange(0, total_duration, scene_interval)

    for start_time in start_times:
        end_time = start_time + scene_interval
        scene_frames = [f for f in frames if start_time <= f["timestamp"] < end_time]

        if scene_frames and len(scene_frames) > 2:
            mid_frame = scene_frames[len(scene_frames) // 2]
            gray = cv2.cvtColor(mid_frame["frame"], cv2.COLOR_BGR2GRAY)

            if np.mean(gray) <= 30:
                continue

            scene_path = SCENES_DIR / f"{video_path.stem}_scene_{len(scenes):02d}.jpg"
            cv2.imwrite(str(scene_path), mid_frame["frame"])

            scenes.append({
                "scene_id": len(scenes),
                "video_name": video_path.name,
                "start_time": float(start_time),
                "end_time": float(min(end_time, total_duration)),
                "duration": float(min(end_time, total_duration) - start_time),
                "frame_count": len(scene_frames),
                "rep_frame_path": str(scene_path),
                "method": "fixed_8s_opencv",
            })

    return scenes


def save_incremental_csvs(video_records, scene_records):
    """Merge new data into existing CSVs."""
    video_df = pd.DataFrame(video_records)
    scene_df = pd.DataFrame(scene_records)

    video_meta_path = DATA_DIR / "video_metadata.csv"
    scene_meta_path = DATA_DIR / "scene_metadata.csv"

    if video_meta_path.exists():
        old_video_df = pd.read_csv(video_meta_path)
        video_df = pd.concat([old_video_df, video_df], ignore_index=True)
        video_df = video_df.drop_duplicates(subset=["filename"], keep="last")
    video_df.to_csv(video_meta_path, index=False)

    if scene_meta_path.exists():
        old_scene_df = pd.read_csv(scene_meta_path)
        scene_df = pd.concat([old_scene_df, scene_df], ignore_index=True)
        scene_df = scene_df.drop_duplicates(
            subset=["video_name", "scene_id", "rep_frame_path"], keep="last"
        )
    scene_df.to_csv(scene_meta_path, index=False)

    return len(video_df), len(scene_df)


if __name__ == "__main__":
    manifest = load_video_manifest()
    if manifest.empty:
        print("No video_manifest.csv found. Processing ALL videos.")

    video_records = []
    scene_records = []
    processed_count = 0
    new_processed = 0

    videos_found = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos_found:
        print("No .mp4 files in data/raw_videos/")
        raise SystemExit(1)

    print(f"Found {len(videos_found)} videos, checking manifest...")

    for video_file in videos_found:
        local_path = str(video_file)
        video_row = manifest[manifest["local_path"] == local_path]

        if not video_row.empty and bool(video_row.iloc[0].get("processed", False)):
            print(f"Skipping processed: {video_file.name}")
            processed_count += 1
            continue

        print(f"\nNEW: Processing {video_file.name}")

        fps, duration, width, height = get_video_metadata(video_file)
        if fps is None:
            print(f"Invalid metadata: {video_file.name}")
            continue

        print(f"   Meta: fps={fps:.1f}, duration={duration:.1f}s, size={width}x{height}")

        video_records.append({
            "filename": video_file.name,
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "path": str(video_file),
        })

        frames = extract_frames(video_file, fps=2)
        if not frames:
            print(f"No frames: {video_file.name}")
            continue

        frame_dir = FRAMES_DIR / video_file.stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(frame_dir / f"frame_{i:06d}.jpg"), f["frame"])

        print(f"   {len(frames)} frames extracted")

        scenes = detect_scenes_fixed_interval(frames, video_file)
        scene_records.extend(scenes)
        print(f"   {len(scenes)} scenes detected")

        manifest = mark_video_processed(manifest, video_file)
        new_processed += 1

    total_videos, total_scenes = save_incremental_csvs(video_records, scene_records)

    print(f"Processed {new_processed} NEW videos → {len(scene_records)} new scenes")
    print(f"TOTAL: {total_videos} videos, {total_scenes} scenes")
