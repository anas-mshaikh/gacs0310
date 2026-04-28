"""
GACS Phase 2: Coordinate-Based Video Generation

좌표 공간에서 타겟 포인트를 지정하면, 가장 가까운 씬들을 선택해서 영상을 합성한다.

Usage:
    # 클러스터 중심 기반 (5개 클러스터 각각 대표 영상)
    python scripts/coordinate_video_generator.py --mode cluster

    # 축 트래버설 (PC1을 -2σ → +2σ 로 5단계)
    python scripts/coordinate_video_generator.py --mode traverse --axis PC1 --steps 5

    # 특정 좌표 지정
    python scripts/coordinate_video_generator.py --mode target --pc1 -10 --pc2 5
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths (gacs_0202/ 기준) ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # gacs_0202/
REPO_ROOT = PROJECT_ROOT.parent   # gacs0202/

COORDS_CSV = REPO_ROOT / "research/2026-03-16_coordinate_v2/gacs_coordinates_v2.csv"
DATASET_CSV = PROJECT_ROOT / "data/embeddings/gacs_dataset.csv"
MANIFEST_CSV = PROJECT_ROOT / "video_manifest.csv"
SCENES_DIR = PROJECT_ROOT / "data/scenes"
RAW_VIDEOS_DIR = PROJECT_ROOT / "data/raw_videos"
OUTPUT_DIR = PROJECT_ROOT / "data/generated/coordinate"

# ── Config ──
TARGET_DURATION = 30  # seconds
MIN_SCENES = 5
MAX_SCENES = 8
MAX_PER_VIDEO = 2  # diversity: max scenes from same source video
TRANSITION_DURATION = 0.5
OUTPUT_FPS = 24
N_STABLE_PCS = 7


def load_data():
    """Load coordinates and dataset."""
    coords = pd.read_csv(COORDS_CSV)
    dataset = pd.read_csv(DATASET_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)

    # Normalize manifest paths
    manifest["local_path"] = (
        manifest["local_path"]
        .str.replace("\\\\", "/", regex=False)
        .str.replace("\\", "/", regex=False)
    )

    print(f"Loaded: {len(coords)} scenes with coordinates")
    print(f"Dataset: {len(dataset)} scenes")
    print(f"Manifest: {len(manifest)} videos")
    return coords, dataset, manifest


def get_video_path(scene_id, manifest):
    """scene_id → source MP4 path."""
    parts = scene_id.rsplit("_s", 1)
    if len(parts) != 2:
        return None
    video_id = parts[0]
    row = manifest[manifest["video_id"] == video_id]
    if row.empty:
        return None
    local_path = row.iloc[0]["local_path"]
    if pd.isna(local_path):
        return None
    path = PROJECT_ROOT / str(local_path)
    return path if path.exists() else None


def get_scene_time_range(scene_id):
    """scene_id → (start_time, end_time)."""
    parts = scene_id.rsplit("_s", 1)
    if len(parts) != 2:
        return 0, 5
    video_id = parts[0]
    meta_path = SCENES_DIR / video_id / "scene_metadata.json"
    if not meta_path.exists():
        return 0, 8
    with open(meta_path) as f:
        scenes_meta = json.load(f)
    for scene in scenes_meta:
        if scene["scene_id"] == scene_id:
            return scene["start_time"], scene["end_time"]
    return 0, 8


def select_nearest_scenes(coords, target_pcs, n_scenes=6, weight_by_variance=True):
    """
    좌표 공간에서 타겟에 가장 가까운 씬 선택.
    diversity 제약: 같은 소스 비디오에서 최대 MAX_PER_VIDEO개.
    """
    pc_cols = [f"PC{i+1}" for i in range(N_STABLE_PCS)]
    X = coords[pc_cols].values

    # Target vector (stable PCs only)
    target = np.zeros(N_STABLE_PCS)
    for i in range(N_STABLE_PCS):
        key = f"PC{i+1}"
        if key in target_pcs:
            target[i] = target_pcs[key]

    # Weighted Euclidean distance (variance as weights)
    if weight_by_variance:
        # Higher variance PCs get more weight
        variances = np.array([coords[f"PC{i+1}"].var() for i in range(N_STABLE_PCS)])
        weights = variances / variances.sum()
    else:
        weights = np.ones(N_STABLE_PCS) / N_STABLE_PCS

    distances = np.sqrt(np.sum(weights * (X - target) ** 2, axis=1))
    sorted_idx = np.argsort(distances)

    # Select with diversity constraint
    selected = []
    video_counts = {}
    for idx in sorted_idx:
        if len(selected) >= n_scenes:
            break
        scene_id = coords.iloc[idx]["scene_id"]
        video_id = scene_id.rsplit("_s", 1)[0] if "_s" in scene_id else scene_id
        if video_counts.get(video_id, 0) >= MAX_PER_VIDEO:
            continue
        video_counts[video_id] = video_counts.get(video_id, 0) + 1
        selected.append({
            "idx": int(idx),
            "scene_id": scene_id,
            "video_id": video_id,
            "distance": float(distances[idx]),
            "cluster": int(coords.iloc[idx]["cluster_label"]),
            "dominant_mood": coords.iloc[idx].get("dominant_mood", ""),
        })

    return selected


def render_video(scene_ids, output_path, manifest):
    """MoviePy로 씬 합성 → MP4 렌더링."""
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
        from moviepy.video.fx import FadeIn, FadeOut
    except ImportError:
        print("ERROR: moviepy not installed. pip install moviepy")
        return None, scene_ids

    clips = []
    skipped = []
    remaining = TARGET_DURATION
    scenes_left = len(scene_ids)

    for scene_id in scene_ids:
        video_path = get_video_path(scene_id, manifest)
        if video_path is None:
            print(f"  SKIP: {scene_id} (no video file)")
            skipped.append(scene_id)
            scenes_left -= 1
            continue

        start_time, end_time = get_scene_time_range(scene_id)
        clip_dur = min(end_time - start_time, remaining / max(1, scenes_left))
        adj_end = start_time + clip_dur

        try:
            clip = VideoFileClip(str(video_path)).subclipped(start_time, adj_end)
            if len(clips) > 0:
                clip = clip.with_effects([FadeIn(TRANSITION_DURATION)])
            clip = clip.with_effects([FadeOut(TRANSITION_DURATION)])
            clips.append(clip)
            remaining -= clip.duration
            scenes_left -= 1
            print(f"  OK: {scene_id} ({clip.duration:.1f}s)")
        except Exception as e:
            print(f"  SKIP: {scene_id} ({e})")
            skipped.append(scene_id)
            scenes_left -= 1

    if not clips:
        print("ERROR: No clips loaded")
        return None, skipped

    final = concatenate_videoclips(clips, method="compose")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Rendering: {output_path} ({final.duration:.1f}s)")
    final.write_videofile(
        str(output_path),
        fps=OUTPUT_FPS,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    duration = final.duration
    for clip in clips:
        clip.close()
    final.close()

    return duration, skipped


def mode_cluster(coords, dataset, manifest):
    """각 클러스터 중심에서 가장 가까운 씬으로 대표 영상 생성."""
    pc_cols = [f"PC{i+1}" for i in range(N_STABLE_PCS)]
    cluster_labels = sorted(coords["cluster_label"].unique())

    print(f"\n{'='*60}")
    print(f"MODE: CLUSTER (k={len(cluster_labels)})")
    print(f"{'='*60}")

    for cl in cluster_labels:
        mask = coords["cluster_label"] == cl
        center = {f"PC{i+1}": float(coords.loc[mask, f"PC{i+1}"].mean()) for i in range(N_STABLE_PCS)}

        # Get cluster name from mood
        cl_moods = []
        for idx in coords[mask].index[:200]:
            m = coords.iloc[idx].get("dominant_mood", "")
            if m:
                cl_moods.append(m)
        from collections import Counter
        top_mood = Counter(cl_moods).most_common(1)[0][0] if cl_moods else f"cluster{cl}"

        print(f"\n--- C{cl}: {top_mood} ---")
        print(f"  Center: {center}")

        selected = select_nearest_scenes(coords, center, n_scenes=6)
        print(f"  Selected {len(selected)} scenes:")
        for s in selected:
            print(f"    {s['scene_id']} (d={s['distance']:.2f}, mood={s['dominant_mood']})")

        scene_ids = [s["scene_id"] for s in selected]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / f"cluster{cl}_{ts}"
        out_path = out_dir / "video.mp4"

        duration, skipped = render_video(scene_ids, out_path, manifest)

        # Save metadata
        meta = {
            "mode": "cluster",
            "cluster": int(cl),
            "cluster_mood": top_mood,
            "center_pcs": center,
            "selected_scenes": selected,
            "skipped": skipped,
            "duration": duration,
            "created_at": datetime.now().isoformat(),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

        if duration:
            print(f"  Output: {out_path} ({duration:.1f}s)")
        else:
            print("  FAILED: no clips rendered")


def mode_traverse(coords, dataset, manifest, axis="PC1", steps=5):
    """축을 따라 이동하며 영상 생성 (예: Tense → Joyful)."""
    print(f"\n{'='*60}")
    print(f"MODE: TRAVERSE axis={axis}, steps={steps}")
    print(f"{'='*60}")

    values = coords[axis].values
    mean = float(np.mean(values))
    std = float(np.std(values))

    # -2σ to +2σ
    positions = np.linspace(mean - 2 * std, mean + 2 * std, steps)
    print(f"  {axis} range: mean={mean:.2f}, std={std:.2f}")
    print(f"  Traversal points: {[f'{p:.2f}' for p in positions]}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    traverse_dir = OUTPUT_DIR / f"traverse_{axis}_{ts}"
    traverse_meta = {
        "mode": "traverse",
        "axis": axis,
        "steps": steps,
        "mean": mean,
        "std": std,
        "positions": positions.tolist(),
        "videos": [],
    }

    for i, pos in enumerate(positions):
        target = {axis: pos}  # other PCs at 0 (mean-centered)
        sigma_pos = (pos - mean) / std

        print(f"\n--- Step {i+1}/{steps}: {axis}={pos:.2f} ({sigma_pos:+.1f}σ) ---")

        selected = select_nearest_scenes(coords, target, n_scenes=6)
        print(f"  Selected {len(selected)} scenes:")
        for s in selected:
            print(f"    {s['scene_id']} (d={s['distance']:.2f}, mood={s['dominant_mood']})")

        scene_ids = [s["scene_id"] for s in selected]
        step_dir = traverse_dir / f"step{i}_{sigma_pos:+.1f}sigma"
        out_path = step_dir / "video.mp4"

        duration, skipped = render_video(scene_ids, out_path, manifest)

        video_meta = {
            "step": i,
            "sigma": round(sigma_pos, 2),
            "target_value": round(pos, 2),
            "selected_scenes": selected,
            "skipped": skipped,
            "duration": duration,
        }
        traverse_meta["videos"].append(video_meta)

        # Save per-step metadata
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(video_meta, f, indent=2, ensure_ascii=False, default=str)

        if duration:
            print(f"  Output: {out_path} ({duration:.1f}s)")

    # Save traverse-level metadata
    with open(traverse_dir / "metadata.json", "w") as f:
        json.dump(traverse_meta, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  All outputs in: {traverse_dir}")


def mode_target(coords, dataset, manifest, target_pcs):
    """특정 좌표 지정 → 영상 생성."""
    print(f"\n{'='*60}")
    print("MODE: TARGET")
    print(f"  Target: {target_pcs}")
    print(f"{'='*60}")

    selected = select_nearest_scenes(coords, target_pcs, n_scenes=6)
    print(f"\n  Selected {len(selected)} scenes:")
    for s in selected:
        print(f"    {s['scene_id']} (d={s['distance']:.2f}, mood={s['dominant_mood']})")

    scene_ids = [s["scene_id"] for s in selected]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"target_{ts}"
    out_path = out_dir / "video.mp4"

    duration, skipped = render_video(scene_ids, out_path, manifest)

    meta = {
        "mode": "target",
        "target_pcs": target_pcs,
        "selected_scenes": selected,
        "skipped": skipped,
        "duration": duration,
        "created_at": datetime.now().isoformat(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    if duration:
        print(f"\n  Output: {out_path} ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="GACS Coordinate-Based Video Generator")
    parser.add_argument("--mode", choices=["cluster", "traverse", "target"], required=True)
    parser.add_argument("--axis", default="PC1", help="Axis for traverse mode")
    parser.add_argument("--steps", type=int, default=5, help="Steps for traverse mode")
    parser.add_argument("--pc1", type=float, default=None)
    parser.add_argument("--pc2", type=float, default=None)
    parser.add_argument("--pc3", type=float, default=None)
    parser.add_argument("--pc4", type=float, default=None)
    parser.add_argument("--pc5", type=float, default=None)
    parser.add_argument("--pc6", type=float, default=None)
    parser.add_argument("--pc7", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Scene selection only, no rendering")
    args = parser.parse_args()

    coords, dataset, manifest = load_data()

    if args.mode == "cluster":
        mode_cluster(coords, dataset, manifest)
    elif args.mode == "traverse":
        mode_traverse(coords, dataset, manifest, axis=args.axis, steps=args.steps)
    elif args.mode == "target":
        target = {}
        for i in range(1, 8):
            val = getattr(args, f"pc{i}", None)
            if val is not None:
                target[f"PC{i}"] = val
        if not target:
            print("ERROR: target mode requires at least one --pcN argument")
            sys.exit(1)
        mode_target(coords, dataset, manifest, target)


if __name__ == "__main__":
    main()
