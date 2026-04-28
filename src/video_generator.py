"""
GACS Video Generator Module
===========================

Generate new short videos using GACS mood coordinates via scene splicing.

Pipeline Overview:
1. Load GACS dataset from `data/embeddings/gacs_dataset.csv`
2. Cluster mood vectors into K clusters
3. For each cluster centroid, generate GACS videos
4. Generate baseline (random) videos for comparison
5. Render videos and save metadata

Usage:
    python -m src.video_generator --clusters 5 --output-dir data/generated
    python -m src.video_generator --quick-test

Author: Claude Code
Version: 1.0
"""

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import numpy as np
import pandas as pd
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.fx import FadeIn, FadeOut
from tqdm import tqdm

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gacs_config import (
    CLAUDE_MODEL,
    DEFAULT_K_CLUSTERS,
    EMBEDDINGS_DIR,
    GENERATED_DIR,
    MAX_SCENES,
    MIN_SCENES,
    OUTPUT_FPS,
    # Directories
    PROJECT_ROOT,
    QUICK_TEST_MODE,
    SCENES_DIR,
    TARGET_DURATION,
    TRANSITION_DURATION,
    USE_NVENC,
    USE_RAPIDS,
    # Functions
    setup_logging,
    validate_config,
)

logger = setup_logging(__name__)

# Handle RAPIDS imports
if USE_RAPIDS:
    try:
        from cuml.cluster import KMeans
        from cuml.manifold import TSNE
        logger.info("Using RAPIDS cuML for KMeans and t-SNE")
    except ImportError:
        logger.warning("RAPIDS requested but not available, falling back to scikit-learn")
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
else:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SceneData:
    """Scene data from GACS dataset."""
    image_id: str
    video_id: str
    scene_id: str
    keyframe_path: str
    mood_vector: np.ndarray
    mood_words: List[str]
    style_words: List[str]


@dataclass
class GeneratedVideoMetadata:
    """Metadata for generated video."""
    video_id: str
    group: str  # "gacs" or "baseline"
    mood_vector: List[float]
    mood_words: List[str]
    prompt_used: str
    source_scenes: List[str]
    created_at: str
    duration: float


# ============================================================================
# Dataset Loading
# ============================================================================

def load_gacs_dataset(dataset_path: Optional[Path] = None) -> Tuple[List[SceneData], Dict[str, SceneData]]:
    """
    Load GACS dataset from CSV file.

    Args:
        dataset_path: Path to gacs_dataset.csv (defaults to EMBEDDINGS_DIR)

    Returns:
        Tuple of (scenes_list, scene_lookup_dict)

    Raises:
        FileNotFoundError: If dataset file does not exist
    """
    if dataset_path is None:
        dataset_path = EMBEDDINGS_DIR / "gacs_dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Please run gacs_dataset_builder.ipynb first."
        )

    logger.info(f"Loading GACS dataset from {dataset_path}")
    gacs_df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(gacs_df)} entries from GACS dataset")

    # Parse mood embeddings from JSON arrays back to numpy arrays
    def parse_mood_vector(v):
        import json as _json
        try:
            parsed = _json.loads(str(v))
            return np.array(parsed, dtype=float)
        except (ValueError, _json.JSONDecodeError):
            return np.array([float(x) for x in str(v).split(',')])

    # Support both column naming conventions
    emb_col = 'mood_embedding' if 'mood_embedding' in gacs_df.columns else 'mood_vector'
    gacs_df['mood_vector_array'] = gacs_df[emb_col].apply(parse_mood_vector)

    frame_col = 'rep_frame_path' if 'rep_frame_path' in gacs_df.columns else 'keyframe_path'

    # Build mood_words and style_words from separate columns if needed
    mood_word_cols = [c for c in gacs_df.columns if c.startswith('mood_') and c not in ('mood_embedding', 'mood_vector', 'mood_vector_array')]
    style_word_cols = [c for c in gacs_df.columns if c.startswith('style_')]

    # Convert to SceneData objects
    scenes_data = []
    for _, row in gacs_df.iterrows():
        # Gather mood words from mood_1..5 columns or mood_words column
        if 'mood_words' in gacs_df.columns:
            mood_words = [w.strip() for w in str(row['mood_words']).split(',')]
        else:
            mood_words = [str(row[c]).strip() for c in mood_word_cols if str(row.get(c, '')).strip() and str(row[c]) != 'nan']

        if 'style_words' in gacs_df.columns:
            style_words = [w.strip() for w in str(row['style_words']).split(',')]
        else:
            style_words = [str(row[c]).strip() for c in style_word_cols if str(row.get(c, '')).strip() and str(row[c]) != 'nan']

        scene = SceneData(
            image_id=str(row['image_id']),
            video_id=str(row['video_id']),
            scene_id=str(row['scene_id']),
            keyframe_path=str(row[frame_col]),
            mood_vector=row['mood_vector_array'],
            mood_words=mood_words,
            style_words=style_words,
        )
        scenes_data.append(scene)

    # Create lookup by scene_id
    scene_lookup = {s.scene_id: s for s in scenes_data}

    logger.info(f"Converted to {len(scenes_data)} SceneData objects")
    if scenes_data:
        sample = scenes_data[0]
        logger.debug(f"Sample scene: {sample.scene_id}")
        logger.debug(f"  Mood words: {sample.mood_words}")
        logger.debug(f"  Style words: {sample.style_words}")

    return scenes_data, scene_lookup


# ============================================================================
# Mood Vector Clustering
# ============================================================================

def cluster_mood_vectors(
    scenes: List[SceneData],
    k: int = DEFAULT_K_CLUSTERS
) -> Tuple[KMeans, Dict[int, Dict]]:
    """
    Cluster mood vectors into K clusters.

    Args:
        scenes: List of SceneData objects
        k: Number of clusters

    Returns:
        Tuple of (KMeans model, cluster info dict)

    Raises:
        ValueError: If scenes list is empty
    """
    if not scenes:
        raise ValueError("Cannot cluster empty scenes list")

    # Stack mood vectors
    vectors = np.stack([s.mood_vector for s in scenes])
    logger.info(f"Clustering {len(vectors)} mood vectors into {k} clusters...")

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    # Build cluster info
    cluster_info = {}
    for i in range(k):
        cluster_scenes = [s for s, l in zip(scenes, labels) if l == i]

        # Aggregate mood words for this cluster
        all_mood_words = []
        for s in cluster_scenes:
            all_mood_words.extend(s.mood_words)

        # Get top mood words
        word_counts = pd.Series(all_mood_words).value_counts()
        top_words = word_counts.head(5).index.tolist()

        cluster_info[i] = {
            'centroid': kmeans.cluster_centers_[i],
            'num_scenes': len(cluster_scenes),
            'scene_ids': [s.scene_id for s in cluster_scenes],
            'top_mood_words': top_words
        }

    logger.info("Clustering complete. Cluster summary:")
    for i, info in cluster_info.items():
        logger.info(f"  Cluster {i}: {info['num_scenes']} scenes, "
                   f"mood: {info['top_mood_words']}")

    return kmeans, cluster_info


def visualize_clusters(
    scenes: List[SceneData],
    kmeans: KMeans,
    output_path: Optional[Path] = None
) -> None:
    """
    Visualize mood clusters using t-SNE.

    Args:
        scenes: List of SceneData objects
        kmeans: Fitted KMeans model
        output_path: Path to save the visualization (defaults to EMBEDDINGS_DIR)
    """
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = EMBEDDINGS_DIR / 'mood_clusters.png'

    logger.info("Visualizing clusters with t-SNE...")
    vectors = np.stack([s.mood_vector for s in scenes])
    labels = kmeans.labels_

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
    vectors_2d = tsne.fit_transform(vectors)

    # Also transform centroids
    centroids_2d = tsne.fit_transform(kmeans.cluster_centers_)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, kmeans.n_clusters))

    for i in range(kmeans.n_clusters):
        mask = labels == i
        ax.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1],
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)

    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
               marker='X', s=300, c='red', edgecolors='black', linewidths=2,
               label='Centroids')

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('GACS Mood Clusters Visualization')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Cluster visualization saved: {output_path}")
    plt.close()


# ============================================================================
# GACS Generation Prompts & Scene Selection
# ============================================================================

GACS_GENERATION_PROMPT = """You are a GACS video generator.
Given:
- target mood words: {mood_words}
- scene pool with mood embeddings

Task:
Select 5 to 8 scenes whose mood vectors are closest to target.
Arrange into a 30-second emotional narrative.

Rules:
- Maintain emotional continuity.
- Avoid abrupt visual jumps.
- No duplicate scenes.
Output:
Ordered list of scene IDs.

Available scenes (scene_id: mood_words):
{scene_list}

Respond with ONLY a comma-separated list of scene IDs in the order they should appear.
Example: scene_001, scene_005, scene_003, scene_012, scene_008"""

BASELINE_PROMPT = """Generate a random but visually coherent sequence of scenes.
Ignore mood vectors.

Select 5 to 8 scenes from the pool below.
Prioritize visual variety and coherent transitions.

Available scenes (scene_id: style_words):
{scene_list}

Respond with ONLY a comma-separated list of scene IDs.
Example: scene_001, scene_005, scene_003, scene_012, scene_008"""


def enforce_diversity(scene_ids: List[str], max_per_video: int = 2) -> List[str]:
    """
    Enforce maximum number of scenes from the same source video.

    Args:
        scene_ids: List of scene identifiers
        max_per_video: Maximum scenes allowed per source video

    Returns:
        Filtered list of scene IDs
    """
    video_counts = {}
    diverse_ids = []
    for sid in scene_ids:
        video_id = sid.rsplit('_s', 1)[0] if '_s' in sid else sid
        video_counts[video_id] = video_counts.get(video_id, 0) + 1
        if video_counts[video_id] <= max_per_video:
            diverse_ids.append(sid)
    return diverse_ids


def select_scenes_gacs(
    target_mood_words: List[str],
    scenes: List[SceneData],
    num_candidates: int = 30
) -> Tuple[List[str], str]:
    """
    Select scenes using GACS method (mood-based).

    Args:
        target_mood_words: Target mood words for the video
        scenes: Pool of available scenes
        num_candidates: Number of candidates to show Claude

    Returns:
        Tuple of (ordered scene IDs, prompt used)
    """
    logger.info(f"Selecting scenes for GACS video with mood: {target_mood_words}")

    # Pre-filter: find scenes closest to target mood
    target_set = set(w.lower() for w in target_mood_words)

    scored_scenes = []
    for scene in scenes:
        scene_words = set(w.lower() for w in scene.mood_words)
        overlap = len(target_set & scene_words)
        scored_scenes.append((scene, overlap))

    # Sort by overlap and take top candidates
    scored_scenes.sort(key=lambda x: x[1], reverse=True)
    candidates = [s[0] for s in scored_scenes[:num_candidates]]

    # Build scene list for prompt
    scene_list = "\n".join([
        f"{s.scene_id}: {', '.join(s.mood_words)}"
        for s in candidates
    ])

    # Format prompt
    prompt = GACS_GENERATION_PROMPT.format(
        mood_words=', '.join(target_mood_words),
        scene_list=scene_list
    )

    # Call Claude
    client = anthropic.Anthropic()
    try:
        logger.debug("Calling Claude API for GACS scene selection...")
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text.strip()

        # Parse scene IDs
        scene_ids = [s.strip() for s in response.split(',')]
        logger.info(f"Claude selected {len(scene_ids)} scenes: {scene_ids}")

        # Validate scene IDs
        valid_ids = [s.scene_id for s in candidates]
        scene_ids = [sid for sid in scene_ids if sid in valid_ids]

        # Enforce diversity: max 2 scenes from the same source video
        scene_ids = enforce_diversity(scene_ids)

        # Ensure we have enough scenes
        if len(scene_ids) < MIN_SCENES:
            logger.warning(f"Only {len(scene_ids)} scenes selected, filling to minimum {MIN_SCENES}")
            for s in candidates:
                if s.scene_id not in scene_ids:
                    scene_ids.append(s.scene_id)
                    scene_ids = enforce_diversity(scene_ids)
                if len(scene_ids) >= MIN_SCENES:
                    break

        return scene_ids[:MAX_SCENES], prompt

    except Exception as e:
        logger.error(f"Error calling Claude: {e}")
        # Fallback: return top candidates by mood overlap
        logger.info("Using fallback: selecting top candidates by mood overlap")
        fallback = [s.scene_id for s in candidates[:MAX_SCENES]]
        return enforce_diversity(fallback), prompt


def select_scenes_baseline(
    scenes: List[SceneData],
    num_candidates: int = 30
) -> Tuple[List[str], str]:
    """
    Select scenes using baseline method (random but coherent).

    Args:
        scenes: Pool of available scenes
        num_candidates: Number of candidates to show Claude

    Returns:
        Tuple of (ordered scene IDs, prompt used)
    """
    logger.info("Selecting scenes for baseline video (random but coherent)")

    # Random selection of candidates
    candidates = random.sample(scenes, min(num_candidates, len(scenes)))

    # Build scene list for prompt
    scene_list = "\n".join([
        f"{s.scene_id}: {', '.join(s.style_words)}"
        for s in candidates
    ])

    # Format prompt
    prompt = BASELINE_PROMPT.format(scene_list=scene_list)

    # Call Claude
    client = anthropic.Anthropic()
    try:
        logger.debug("Calling Claude API for baseline scene selection...")
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text.strip()

        # Parse scene IDs
        scene_ids = [s.strip() for s in response.split(',')]
        logger.info(f"Claude selected {len(scene_ids)} scenes: {scene_ids}")

        # Validate scene IDs
        valid_ids = [s.scene_id for s in candidates]
        scene_ids = [sid for sid in scene_ids if sid in valid_ids]

        # Enforce diversity: max 2 scenes from the same source video
        scene_ids = enforce_diversity(scene_ids)

        # Ensure we have enough scenes
        if len(scene_ids) < MIN_SCENES:
            logger.warning(f"Only {len(scene_ids)} scenes selected, filling to minimum {MIN_SCENES}")
            for s in candidates:
                if s.scene_id not in scene_ids:
                    scene_ids.append(s.scene_id)
                    scene_ids = enforce_diversity(scene_ids)
                if len(scene_ids) >= MIN_SCENES:
                    break

        return scene_ids[:MAX_SCENES], prompt

    except Exception as e:
        logger.error(f"Error calling Claude: {e}")
        # Fallback: random selection
        logger.info("Using fallback: random selection from candidates")
        fallback = [s.scene_id for s in candidates[:MAX_SCENES]]
        return enforce_diversity(fallback), prompt


# ============================================================================
# Video Rendering
# ============================================================================

def get_scene_time_range(scene_id: str) -> Tuple[float, float]:
    """
    Get the time range for a scene from scene metadata.

    Args:
        scene_id: Scene identifier (format: {video_id}_s{index})

    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    # Parse video_id from scene_id
    parts = scene_id.rsplit('_s', 1)
    if len(parts) != 2:
        logger.debug(f"Could not parse scene_id {scene_id}, using default 5s")
        return 0, 5  # Default 5 second clip

    video_id = parts[0]

    # Load scene metadata
    metadata_path = SCENES_DIR / video_id / "scene_metadata.json"
    if not metadata_path.exists():
        logger.debug(f"No metadata for {video_id}, using default 3s")
        return 0, 3

    try:
        with open(metadata_path, 'r') as f:
            scenes_meta = json.load(f)

        for scene in scenes_meta:
            if scene['scene_id'] == scene_id:
                return scene['start_time'], scene['end_time']
    except Exception as e:
        logger.debug(f"Error loading metadata for {video_id}: {e}")

    return 0, 3  # Default


def get_video_path(scene_id: str) -> Optional[Path]:
    """
    Get the source video path for a scene.

    Args:
        scene_id: Scene identifier

    Returns:
        Path to video file or None if not found
    """
    # Parse video_id from scene_id
    parts = scene_id.rsplit('_s', 1)
    if len(parts) != 2:
        return None

    video_id = parts[0]

    # Load manifest
    manifest_path = PROJECT_ROOT / "video_manifest.csv"
    if not manifest_path.exists():
        logger.warning(f"Video manifest not found at {manifest_path}")
        return None

    try:
        manifest_df = pd.read_csv(manifest_path)
        manifest_df['local_path'] = (
            manifest_df['local_path']
            .str.replace('\\\\', '/', regex=False)
            .str.replace('\\', '/', regex=False)
        )

        # Find video
        video_row = manifest_df[manifest_df['video_id'] == video_id]
        if video_row.empty:
            logger.debug(f"Video {video_id} not found in manifest")
            return None

        video_path = PROJECT_ROOT / video_row.iloc[0]['local_path']
        if video_path.exists():
            return video_path
        else:
            logger.debug(f"Video file does not exist: {video_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading video manifest: {e}")
        return None


def render_video(
    scene_ids: List[str],
    output_path: Path,
    target_duration: float = TARGET_DURATION
) -> Tuple[Optional[float], List[str]]:
    """
    Render a video by concatenating scenes with duration fitting.

    Args:
        scene_ids: Ordered list of scene IDs
        output_path: Path to save the output video
        target_duration: Target video duration in seconds

    Returns:
        Tuple of (actual duration or None if failed, list of skipped scene IDs)
    """
    clips = []
    skipped_scenes = []

    # Duration fitting: track remaining budget
    remaining = target_duration
    scenes_left = len(scene_ids)

    logger.info(f"Loading {len(scene_ids)} clips for rendering...")

    for scene_id in tqdm(scene_ids, desc="Loading clips", disable=QUICK_TEST_MODE):
        video_path = get_video_path(scene_id)
        if video_path is None:
            logger.warning(f"Skipping {scene_id}: video file not found")
            skipped_scenes.append(scene_id)
            scenes_left -= 1
            continue

        start_time, end_time = get_scene_time_range(scene_id)

        # Calculate clip duration to fit remaining budget
        clip_dur = min(end_time - start_time, remaining / max(1, scenes_left))
        end_time = start_time + clip_dur

        try:
            clip = VideoFileClip(str(video_path)).subclipped(start_time, end_time)

            # Add fade transitions
            if len(clips) > 0:
                clip = clip.with_effects([FadeIn(TRANSITION_DURATION)])
            clip = clip.with_effects([FadeOut(TRANSITION_DURATION)])

            clips.append(clip)
            remaining -= clip.duration
            scenes_left -= 1
        except Exception as e:
            logger.warning(f"Skipping {scene_id}: {e}")
            skipped_scenes.append(scene_id)
            scenes_left -= 1
            continue

    if not clips:
        logger.error("No clips to render.")
        return None, skipped_scenes

    # Concatenate
    logger.info(f"Concatenating {len(clips)} clips...")
    final = concatenate_videoclips(clips, method="compose")

    # Duration deviation check
    if abs(final.duration - target_duration) > 3:
        logger.warning(
            f"Duration {final.duration:.1f}s deviates from target {target_duration}s"
        )

    # Export with optional NVENC encoding
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting video to {output_path}...")

    ffmpeg_params = []
    if USE_NVENC:
        logger.info("Using NVENC hardware encoding (h264_nvenc)")
        ffmpeg_params = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-tune', 'hq']
    else:
        logger.info("Using CPU encoding (libx264)")
        ffmpeg_params = ['-c:v', 'libx264', '-preset', 'medium']

    final.write_videofile(
        str(output_path),
        fps=OUTPUT_FPS,
        codec='libx264',
        audio_codec='aac',
    )

    duration = final.duration

    # Cleanup
    for clip in clips:
        clip.close()
    final.close()

    logger.info(f"Video saved: {output_path} ({duration:.1f}s)")
    if skipped_scenes:
        logger.info(f"Skipped {len(skipped_scenes)} scenes")

    return duration, skipped_scenes


# ============================================================================
# Video Generation Pipeline
# ============================================================================

def generate_gacs_video(
    cluster_id: int,
    cluster_info: Dict,
    scenes: List[SceneData],
    output_dir: Optional[Path] = None
) -> Optional[GeneratedVideoMetadata]:
    """
    Generate a GACS video for a mood cluster.

    Args:
        cluster_id: Cluster index
        cluster_info: Info dict for this cluster
        scenes: All available scenes
        output_dir: Output directory (defaults to GENERATED_DIR/gacs)

    Returns:
        GeneratedVideoMetadata or None if failed
    """
    if output_dir is None:
        output_dir = GENERATED_DIR / "gacs"

    logger.info(f"Generating GACS video for Cluster {cluster_id}")
    logger.info(f"Target mood: {cluster_info['top_mood_words']}")

    # Select scenes using GACS method
    scene_ids, prompt_used = select_scenes_gacs(
        cluster_info['top_mood_words'],
        scenes
    )

    logger.info(f"Selected {len(scene_ids)} scenes: {scene_ids}")

    # Generate video ID
    video_id = f"gacs_cluster{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{video_id}.mp4"

    # Render video
    duration, skipped_scenes = render_video(scene_ids, output_path)

    if duration is None:
        logger.error(f"Failed to render video {video_id}")
        return None

    # Create metadata
    metadata = GeneratedVideoMetadata(
        video_id=video_id,
        group="gacs",
        mood_vector=cluster_info['centroid'].tolist(),
        mood_words=cluster_info['top_mood_words'],
        prompt_used=prompt_used,
        source_scenes=scene_ids,
        created_at=datetime.now().isoformat(),
        duration=duration
    )

    # Enrich with render stats
    metadata_dict = asdict(metadata)
    metadata_dict['render_stats'] = {
        'scenes_requested': len(scene_ids),
        'scenes_used': len(scene_ids) - len(skipped_scenes),
        'scenes_skipped': len(skipped_scenes),
        'actual_duration': duration,
        'target_duration': TARGET_DURATION,
        'duration_error': abs(duration - TARGET_DURATION)
    }

    # Save metadata
    metadata_path = output_dir / f"{video_id}" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2, default=str)

    # Also copy video to metadata folder
    shutil.copy(output_path, metadata_path.parent / "video.mp4")

    logger.info(f"GACS video metadata saved: {metadata_path}")

    return metadata


def generate_baseline_video(
    cluster_id: int,
    scenes: List[SceneData],
    output_dir: Optional[Path] = None
) -> Optional[GeneratedVideoMetadata]:
    """
    Generate a baseline (random) video.

    Args:
        cluster_id: Cluster index (for naming only)
        scenes: All available scenes
        output_dir: Output directory (defaults to GENERATED_DIR/baseline)

    Returns:
        GeneratedVideoMetadata or None if failed
    """
    if output_dir is None:
        output_dir = GENERATED_DIR / "baseline"

    logger.info(f"Generating Baseline video {cluster_id}")

    # Select scenes using baseline method
    scene_ids, prompt_used = select_scenes_baseline(scenes)

    logger.info(f"Selected {len(scene_ids)} scenes: {scene_ids}")

    # Generate video ID
    video_id = f"baseline_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = output_dir / f"{video_id}.mp4"

    # Render video
    duration, skipped_scenes = render_video(scene_ids, output_path)

    if duration is None:
        logger.error(f"Failed to render video {video_id}")
        return None

    # Create metadata
    metadata = GeneratedVideoMetadata(
        video_id=video_id,
        group="baseline",
        mood_vector=[],  # No target mood for baseline
        mood_words=[],
        prompt_used=prompt_used,
        source_scenes=scene_ids,
        created_at=datetime.now().isoformat(),
        duration=duration
    )

    # Enrich with render stats
    metadata_dict = asdict(metadata)
    metadata_dict['render_stats'] = {
        'scenes_requested': len(scene_ids),
        'scenes_used': len(scene_ids) - len(skipped_scenes),
        'scenes_skipped': len(skipped_scenes),
        'actual_duration': duration,
        'target_duration': TARGET_DURATION,
        'duration_error': abs(duration - TARGET_DURATION)
    }

    # Save metadata
    metadata_path = output_dir / f"{video_id}" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2, default=str)

    # Also copy video
    shutil.copy(output_path, metadata_path.parent / "video.mp4")

    logger.info(f"Baseline video metadata saved: {metadata_path}")

    return metadata


# ============================================================================
# Orchestration
# ============================================================================

def run_pipeline(
    num_clusters: int = DEFAULT_K_CLUSTERS,
    output_dir: Optional[Path] = None,
    generate_baseline: bool = True,
    visualize: bool = True
) -> Dict[str, List[GeneratedVideoMetadata]]:
    """
    Run the complete GACS video generation pipeline.

    Args:
        num_clusters: Number of mood clusters to create
        output_dir: Output directory for generated videos
        generate_baseline: Whether to generate baseline videos
        visualize: Whether to create cluster visualization

    Returns:
        Dictionary with 'gacs' and 'baseline' lists of metadata

    Raises:
        RuntimeError: If dataset loading or clustering fails
    """
    logger.info("=" * 70)
    logger.info("GACS Video Generation Pipeline")
    logger.info("=" * 70)

    # Validate configuration
    validate_config()

    # Load dataset
    try:
        scenes_data, scene_lookup = load_gacs_dataset()
    except FileNotFoundError as e:
        logger.error(str(e))
        raise RuntimeError(f"Failed to load dataset: {e}")

    if not scenes_data:
        raise RuntimeError("Dataset is empty")

    # Cluster mood vectors
    try:
        kmeans_model, cluster_info = cluster_mood_vectors(scenes_data, k=num_clusters)
    except ValueError as e:
        logger.error(str(e))
        raise RuntimeError(f"Failed to cluster moods: {e}")

    # Visualize clusters
    if visualize:
        try:
            visualize_clusters(scenes_data, kmeans_model)
        except Exception as e:
            logger.warning(f"Could not visualize clusters: {e}")

    # Generate videos for each cluster
    all_gacs_videos = []
    all_baseline_videos = []

    gacs_output_dir = output_dir / "gacs" if output_dir else GENERATED_DIR / "gacs"
    baseline_output_dir = output_dir / "baseline" if output_dir else GENERATED_DIR / "baseline"

    clusters_to_process = range(len(cluster_info))
    if QUICK_TEST_MODE:
        clusters_to_process = range(min(1, len(cluster_info)))
        logger.info("QUICK_TEST_MODE: Processing only 1 cluster")

    for cluster_id in clusters_to_process:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing Cluster {cluster_id} / {len(cluster_info) - 1}")
        logger.info(f"{'=' * 70}")

        # Generate GACS video
        gacs_meta = generate_gacs_video(
            cluster_id,
            cluster_info[cluster_id],
            scenes_data,
            gacs_output_dir
        )
        if gacs_meta:
            all_gacs_videos.append(gacs_meta)

        # Generate corresponding baseline video
        if generate_baseline:
            baseline_meta = generate_baseline_video(
                cluster_id,
                scenes_data,
                baseline_output_dir
            )
            if baseline_meta:
                all_baseline_videos.append(baseline_meta)

    logger.info(f"\n{'=' * 70}")
    logger.info("Pipeline Complete!")
    logger.info(f"{'=' * 70}")
    logger.info(f"Generated {len(all_gacs_videos)} GACS videos")
    logger.info(f"Generated {len(all_baseline_videos)} Baseline videos")

    return {
        'gacs': all_gacs_videos,
        'baseline': all_baseline_videos
    }


def list_generated_videos(output_dir: Optional[Path] = None) -> List[Dict]:
    """
    List all generated videos with metadata.

    Args:
        output_dir: Output directory (defaults to GENERATED_DIR)

    Returns:
        List of video info dictionaries
    """
    if output_dir is None:
        output_dir = GENERATED_DIR

    videos = []

    # Check GACS videos
    gacs_dir = output_dir / "gacs"
    if gacs_dir.exists():
        for video_dir in gacs_dir.iterdir():
            if video_dir.is_dir():
                metadata_path = video_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                    videos.append({
                        'video_id': meta['video_id'],
                        'group': meta['group'],
                        'mood_words': ', '.join(meta.get('mood_words', [])[:3]),
                        'duration': f"{meta['duration']:.1f}s",
                        'num_scenes': len(meta['source_scenes']),
                        'path': str(video_dir / 'video.mp4')
                    })

    # Check baseline videos
    baseline_dir = output_dir / "baseline"
    if baseline_dir.exists():
        for video_dir in baseline_dir.iterdir():
            if video_dir.is_dir():
                metadata_path = video_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                    videos.append({
                        'video_id': meta['video_id'],
                        'group': meta['group'],
                        'mood_words': '(random)',
                        'duration': f"{meta['duration']:.1f}s",
                        'num_scenes': len(meta['source_scenes']),
                        'path': str(video_dir / 'video.mp4')
                    })

    if videos:
        logger.info(f"\nGenerated Videos ({len(videos)} total):")
        df = pd.DataFrame(videos)
        logger.info(f"\n{df.to_string()}\n")

    return videos


# ============================================================================
# Main / CLI
# ============================================================================

def main():
    """Command-line interface for video generator."""
    parser = argparse.ArgumentParser(
        description="GACS Video Generator - Generate mood-based videos"
    )

    parser.add_argument(
        '--clusters',
        type=int,
        default=DEFAULT_K_CLUSTERS,
        help=f'Number of mood clusters (default: {DEFAULT_K_CLUSTERS})'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=GENERATED_DIR,
        help=f'Output directory for generated videos (default: {GENERATED_DIR})'
    )

    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip generation of baseline videos'
    )

    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip cluster visualization'
    )

    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list existing generated videos'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run in quick test mode (1 cluster, 3 scenes max)'
    )

    args = parser.parse_args()

    try:
        if args.quick_test:
            import os
            os.environ['GACS_QUICK_TEST'] = '1'

        if args.list_only:
            list_generated_videos(args.output_dir)
        else:
            results = run_pipeline(
                num_clusters=args.clusters,
                output_dir=args.output_dir,
                generate_baseline=not args.no_baseline,
                visualize=not args.no_visualize
            )

            logger.info("\n" + "=" * 70)
            logger.info("Video Generation Results")
            logger.info("=" * 70)

            if results['gacs']:
                logger.info(f"\nGACS Videos ({len(results['gacs'])} total):")
                for i, meta in enumerate(results['gacs'], 1):
                    logger.info(f"  {i}. {meta.video_id} ({meta.duration:.1f}s)")
                    logger.info(f"     Moods: {', '.join(meta.mood_words)}")

            if results['baseline']:
                logger.info(f"\nBaseline Videos ({len(results['baseline'])} total):")
                for i, meta in enumerate(results['baseline'], 1):
                    logger.info(f"  {i}. {meta.video_id} ({meta.duration:.1f}s)")

            list_generated_videos(args.output_dir)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
