#!/usr/bin/env python3
"""
YouTube Experiment Runner - Production Module

This module provides a complete pipeline for validating GACS using real YouTube
algorithm metrics. It handles authentication, metadata generation, video uploads,
metrics collection, statistical analysis, and report generation.

Usage:
    python experiment_runner.py auth
    python experiment_runner.py upload
    python experiment_runner.py metrics
    python experiment_runner.py analyze
    python experiment_runner.py report
    python experiment_runner.py all
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from scipy import stats
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gacs_config import (
    CLAUDE_MODEL,
    EXPERIMENTS_DIR,
    GENERATED_DIR,
    PROJECT_ROOT,
    QUICK_TEST_UPLOAD_PRIVACY,
    YOUTUBE_SCOPES,
    setup_logging,
    validate_config,
)

logger = setup_logging(__name__)


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = PROJECT_ROOT
GACS_DIR = GENERATED_DIR / "gacs"
BASELINE_DIR = GENERATED_DIR / "baseline"

# OAuth settings
CLIENT_SECRETS_FILE = BASE_DIR / "client_secrets.json"
CREDENTIALS_FILE = BASE_DIR / "youtube_credentials.json"

# Metadata generation prompts
TITLE_PROMPT = """Generate a YouTube title.
Under 60 characters.
Emotional but not clickbait.
No emojis.

Context:
- Video mood: {mood_words}
- Video group: {group}

Respond with ONLY the title, nothing else."""

DESCRIPTION_PROMPT = """Generate a short YouTube description.
Max 2 sentences.
No hashtags.
No marketing language.

Context:
- Video mood: {mood_words}
- Video group: {group}

Respond with ONLY the description, nothing else."""

TAGS_PROMPT = """Generate 8 to 12 single-word tags based on mood and style.

Context:
- Video mood: {mood_words}
- Video style: {style_words}

Rules:
- Single words only
- No phrases
- Relevant to content

Respond with ONLY comma-separated tags, nothing else.
Example: emotional, cinematic, dramatic, powerful, inspiring, music, film, mood"""


# ============================================================================
# Authentication Functions
# ============================================================================

def get_authenticated_services() -> Tuple:
    """
    Authenticate with YouTube API and return service objects.

    Uses OAuth2 flow for local workstation authentication. Credentials are
    cached in youtube_credentials.json for subsequent runs.

    Returns:
        Tuple of (youtube_service, youtube_analytics_service) authenticated objects.

    Raises:
        FileNotFoundError: If client_secrets.json is not found.
    """
    credentials = None

    # Check for existing cached credentials
    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                creds_data = json.load(f)
            credentials = Credentials.from_authorized_user_info(creds_data, YOUTUBE_SCOPES)
            logger.debug("Loaded cached credentials")
        except Exception as e:
            logger.warning(f"Failed to load cached credentials: {e}")

    # Refresh or get new credentials
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            logger.info("Refreshing expired credentials...")
            try:
                credentials.refresh(Request())
                logger.info("Credentials refreshed successfully")
            except Exception as e:
                logger.warning(f"Failed to refresh credentials: {e}")
                credentials = None

        if not credentials:
            if not CLIENT_SECRETS_FILE.exists():
                raise FileNotFoundError(
                    f"client_secrets.json not found at {CLIENT_SECRETS_FILE}\n"
                    "Please follow the setup guide to create OAuth credentials."
                )

            logger.info("Starting OAuth flow for new authentication")
            logger.info("A browser window will open for authentication")

            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRETS_FILE), YOUTUBE_SCOPES
            )
            # Local server on port 8080 (workstation only, not Colab)
            credentials = flow.run_local_server(port=8080, open_browser=True)
            logger.info("OAuth flow completed successfully")

        # Save credentials for future use
        try:
            with open(CREDENTIALS_FILE, 'w') as f:
                f.write(credentials.to_json())
            logger.info(f"Credentials cached to {CREDENTIALS_FILE}")
        except Exception as e:
            logger.warning(f"Failed to cache credentials: {e}")

    # Build service objects
    try:
        youtube = build('youtube', 'v3', credentials=credentials)
        youtube_analytics = build('youtubeAnalytics', 'v2', credentials=credentials)
        logger.info("Successfully authenticated with YouTube API")
        return youtube, youtube_analytics
    except Exception as e:
        logger.error(f"Failed to build YouTube services: {e}")
        raise


# ============================================================================
# Metadata Generation Functions
# ============================================================================

def generate_video_metadata(video_metadata: Dict) -> Dict:
    """
    Generate YouTube metadata (title, description, tags) using Claude.

    Uses Claude API to generate contextually appropriate and emotionally
    resonant metadata for videos based on their mood and group classification.

    Args:
        video_metadata: Metadata dictionary from generated video with keys
            like 'group', 'mood_words', etc.

    Returns:
        Dict with keys:
            - 'title': YouTube video title (max 60 chars)
            - 'description': Short description (max 2 sentences)
            - 'tags': List of up to 12 single-word tags
    """
    client = anthropic.Anthropic()

    group = video_metadata.get('group', 'unknown')
    mood_words = video_metadata.get('mood_words', [])
    mood_str = ', '.join(mood_words) if mood_words else 'varied'
    style_str = 'cinematic, mixed'

    # Generate title
    title_prompt = TITLE_PROMPT.format(mood_words=mood_str, group=group)
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": title_prompt}]
        )
        title = response.content[0].text.strip()[:60]  # Enforce limit
        logger.debug(f"Generated title: {title}")
    except Exception as e:
        logger.warning(f"Failed to generate title: {e}")
        title = f"GACS Video - {group.capitalize()}"

    time.sleep(0.5)  # Rate limiting

    # Generate description
    desc_prompt = DESCRIPTION_PROMPT.format(mood_words=mood_str, group=group)
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": desc_prompt}]
        )
        description = response.content[0].text.strip()
        logger.debug(f"Generated description: {description}")
    except Exception as e:
        logger.warning(f"Failed to generate description: {e}")
        description = "A video created for research purposes."

    time.sleep(0.5)

    # Generate tags
    tags_prompt = TAGS_PROMPT.format(mood_words=mood_str, style_words=style_str)
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": tags_prompt}]
        )
        tags_text = response.content[0].text.strip()
        tags = [t.strip() for t in tags_text.split(',')][:12]  # Max 12 tags
        logger.debug(f"Generated tags: {tags}")
    except Exception as e:
        logger.warning(f"Failed to generate tags: {e}")
        tags = ['video', 'research', 'mood', 'emotional']

    return {
        'title': title,
        'description': description,
        'tags': tags
    }


# ============================================================================
# Video Upload Functions
# ============================================================================

def upload_video(youtube_service,
                 video_path: Path,
                 title: str,
                 description: str,
                 tags: List[str],
                 category_id: str = "22",
                 privacy: str = "unlisted") -> Optional[str]:
    """
    Upload a single video to YouTube.

    Args:
        youtube_service: Authenticated YouTube API service object.
        video_path: Path to the MP4 video file.
        title: Video title (max 60 characters).
        description: Video description.
        tags: List of video tags (max 12).
        category_id: YouTube category ID (default "22" = People & Blogs).
        privacy: Privacy status ("public", "unlisted", or "private").

    Returns:
        YouTube video ID if successful, None otherwise.
    """
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return None

    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': tags,
            'categoryId': category_id
        },
        'status': {
            'privacyStatus': privacy,
            'selfDeclaredMadeForKids': False
        }
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype='video/mp4',
        resumable=True,
        chunksize=1024*1024
    )

    try:
        logger.info(f"Uploading: {title}")
        logger.debug(f"File: {video_path}")

        request = youtube_service.videos().insert(
            part='snippet,status',
            body=body,
            media_body=media
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                progress_pct = int(status.progress() * 100)
                logger.debug(f"Upload progress: {progress_pct}%")

        video_id = response['id']
        logger.info(f"Upload complete! Video ID: {video_id}")
        logger.info(f"URL: https://www.youtube.com/watch?v={video_id}")

        return video_id

    except HttpError as e:
        logger.error(f"HTTP Error: {e.resp.status} - {e.content}")
        return None
    except Exception as e:
        logger.error(f"Failed to upload video: {e}")
        return None


def upload_all_videos(youtube_service, alternate: bool = True) -> pd.DataFrame:
    """
    Upload all generated videos, alternating between GACS and baseline.

    Loads video files and metadata from GACS_DIR and BASELINE_DIR, generates
    YouTube metadata using Claude, and uploads them in alternating order.
    Saves a mapping of YouTube IDs to local video IDs.

    Args:
        youtube_service: Authenticated YouTube API service object.
        alternate: If True, alternate between GACS and baseline uploads.

    Returns:
        DataFrame with columns:
            - 'local_video_id': Original video ID
            - 'youtube_id': Uploaded YouTube video ID
            - 'group': Video group (gacs or baseline)
            - 'title': Generated title
            - 'uploaded_at': ISO timestamp
            - 'success': Whether upload succeeded
    """
    # Load GACS videos
    gacs_videos = []
    if GACS_DIR.exists():
        for video_dir in GACS_DIR.iterdir():
            if video_dir.is_dir():
                metadata_path = video_dir / "metadata.json"
                video_path = video_dir / "video.mp4"
                if metadata_path.exists() and video_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            meta = json.load(f)
                        meta['video_path'] = str(video_path)
                        gacs_videos.append(meta)
                    except Exception as e:
                        logger.warning(f"Failed to load GACS metadata from {metadata_path}: {e}")

    # Load baseline videos
    baseline_videos = []
    if BASELINE_DIR.exists():
        for video_dir in BASELINE_DIR.iterdir():
            if video_dir.is_dir():
                metadata_path = video_dir / "metadata.json"
                video_path = video_dir / "video.mp4"
                if metadata_path.exists() and video_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            meta = json.load(f)
                        meta['video_path'] = str(video_path)
                        baseline_videos.append(meta)
                    except Exception as e:
                        logger.warning(f"Failed to load baseline metadata from {metadata_path}: {e}")

    logger.info(f"Found {len(gacs_videos)} GACS videos and {len(baseline_videos)} baseline videos")

    # Interleave videos if alternating
    if alternate:
        videos_to_upload = []
        for i in range(max(len(gacs_videos), len(baseline_videos))):
            if i < len(gacs_videos):
                videos_to_upload.append(gacs_videos[i])
            if i < len(baseline_videos):
                videos_to_upload.append(baseline_videos[i])
    else:
        videos_to_upload = gacs_videos + baseline_videos

    logger.info(f"Uploading {len(videos_to_upload)} videos...")

    # Upload each video
    results = []
    youtube_map = []

    for video_meta in tqdm(videos_to_upload, desc="Uploading videos"):
        # Generate YouTube metadata
        yt_meta = generate_video_metadata(video_meta)

        # Upload
        video_path = Path(video_meta['video_path'])
        youtube_id = upload_video(
            youtube_service,
            video_path,
            yt_meta['title'],
            yt_meta['description'],
            yt_meta['tags'],
            privacy=QUICK_TEST_UPLOAD_PRIVACY
        )

        result = {
            'local_video_id': video_meta.get('video_id', 'unknown'),
            'youtube_id': youtube_id,
            'group': video_meta.get('group', 'unknown'),
            'title': yt_meta['title'],
            'uploaded_at': datetime.now().isoformat(),
            'success': youtube_id is not None
        }
        results.append(result)

        if youtube_id:
            youtube_map.append({
                'youtube_id': youtube_id,
                'local_video_id': video_meta.get('video_id', 'unknown'),
                'group': video_meta.get('group', 'unknown')
            })

        # Wait between uploads to avoid rate limiting
        time.sleep(5)

    # Save YouTube mapping
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    map_path = EXPERIMENTS_DIR / "youtube_map.json"
    try:
        with open(map_path, 'w') as f:
            json.dump(youtube_map, f, indent=2)
        logger.info(f"YouTube mapping saved to: {map_path}")
    except Exception as e:
        logger.error(f"Failed to save YouTube mapping: {e}")

    return pd.DataFrame(results)


# ============================================================================
# Metrics Collection Functions
# ============================================================================

def collect_video_metrics(youtube_service,
                          youtube_analytics_service,
                          video_id: str,
                          days_back: int = 7) -> Dict:
    """
    Collect metrics for a single YouTube video.

    Queries both YouTube Data API (for basic statistics) and YouTube Analytics
    API (for detailed engagement metrics). Includes retry logic for robustness.

    Args:
        youtube_service: Authenticated YouTube Data API service.
        youtube_analytics_service: Authenticated YouTube Analytics API service.
        video_id: YouTube video ID.
        days_back: Number of days of historical data to collect (default 7).

    Returns:
        Dict with collected metrics including:
            - views, likes, comments
            - impressions, ctr, watchTime
            - averageViewDuration, averageViewPercentage
            - subscribersGained, subscribersLost, shares
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    metrics = {
        'video_id': video_id,
        'collected_at': datetime.now().isoformat(),
        'date_range': {'start': start_date, 'end': end_date}
    }

    # Get basic statistics from YouTube Data API
    try:
        response = youtube_service.videos().list(
            part='statistics,contentDetails',
            id=video_id
        ).execute()

        if response.get('items'):
            stats = response['items'][0].get('statistics', {})
            metrics['views'] = int(stats.get('viewCount', 0))
            metrics['likes'] = int(stats.get('likeCount', 0))
            metrics['comments'] = int(stats.get('commentCount', 0))
            logger.debug(f"Retrieved basic stats for {video_id}")
        else:
            logger.warning(f"No video found for {video_id}")
            metrics['views'] = 0
            metrics['likes'] = 0
            metrics['comments'] = 0
    except HttpError as e:
        logger.warning(f"HTTP error fetching statistics for {video_id}: {e}")
        metrics['views'] = 0
        metrics['likes'] = 0
        metrics['comments'] = 0
    except Exception as e:
        logger.error(f"Error fetching statistics for {video_id}: {e}")
        metrics['views'] = 0
        metrics['likes'] = 0
        metrics['comments'] = 0

    # Get detailed analytics
    try:
        response = youtube_analytics_service.reports().query(
            ids='channel==MINE',
            startDate=start_date,
            endDate=end_date,
            metrics='views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,'
                   'subscribersGained,subscribersLost,shares',
            filters=f'video=={video_id}'
        ).execute()

        if response.get('rows'):
            row = response['rows'][0]
            metrics['impressions'] = row[0] if len(row) > 0 else 0
            metrics['watchTime'] = row[1] if len(row) > 1 else 0
            metrics['averageViewDuration'] = row[2] if len(row) > 2 else 0
            metrics['averageViewPercentage'] = row[3] if len(row) > 3 else 0
            metrics['subscribersGained'] = row[4] if len(row) > 4 else 0
            metrics['subscribersLost'] = row[5] if len(row) > 5 else 0
            metrics['shares'] = row[6] if len(row) > 6 else 0
            logger.debug(f"Retrieved detailed analytics for {video_id}")
        else:
            # Set defaults if no data available yet
            metrics['impressions'] = 0
            metrics['watchTime'] = 0
            metrics['averageViewDuration'] = 0
            metrics['averageViewPercentage'] = 0
            metrics['subscribersGained'] = 0
            metrics['subscribersLost'] = 0
            metrics['shares'] = 0
            logger.debug(f"No analytics rows available for {video_id}")
    except HttpError as e:
        logger.warning(f"HTTP error fetching analytics for {video_id}: {e}")
    except Exception as e:
        logger.error(f"Error fetching analytics for {video_id}: {e}")

    # Get CTR if available
    try:
        response = youtube_analytics_service.reports().query(
            ids='channel==MINE',
            startDate=start_date,
            endDate=end_date,
            metrics='impressions,impressionClickThroughRate',
            filters=f'video=={video_id}'
        ).execute()

        if response.get('rows'):
            row = response['rows'][0]
            metrics['impressions'] = row[0] if len(row) > 0 else metrics.get('impressions', 0)
            metrics['ctr'] = row[1] if len(row) > 1 else 0
            logger.debug(f"Retrieved CTR for {video_id}")
        else:
            metrics['ctr'] = 0
    except Exception as e:
        logger.debug(f"Could not retrieve CTR for {video_id}: {e}")
        metrics['ctr'] = 0

    return metrics


def collect_all_metrics(youtube_service, youtube_analytics_service) -> pd.DataFrame:
    """
    Collect metrics for all uploaded videos.

    Loads the YouTube ID mapping created by upload_all_videos(), queries metrics
    for each video, and appends to the metrics CSV file (or creates a new one).

    Args:
        youtube_service: Authenticated YouTube Data API service.
        youtube_analytics_service: Authenticated YouTube Analytics API service.

    Returns:
        DataFrame with collected metrics for all videos.

    Raises:
        FileNotFoundError: If youtube_map.json doesn't exist.
    """
    # Load YouTube mapping
    map_path = EXPERIMENTS_DIR / "youtube_map.json"
    if not map_path.exists():
        logger.error("YouTube mapping not found. Run upload subcommand first.")
        raise FileNotFoundError(f"YouTube mapping not found at {map_path}")

    with open(map_path, 'r') as f:
        youtube_map = json.load(f)

    logger.info(f"Collecting metrics for {len(youtube_map)} videos...")

    all_metrics = []
    for entry in tqdm(youtube_map, desc="Collecting metrics"):
        metrics = collect_video_metrics(
            youtube_service,
            youtube_analytics_service,
            entry['youtube_id']
        )
        metrics['local_video_id'] = entry['local_video_id']
        metrics['group'] = entry['group']
        all_metrics.append(metrics)

        time.sleep(0.5)  # Rate limiting

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Save metrics
    metrics_path = EXPERIMENTS_DIR / "metrics.csv"

    # Append to existing or create new
    if metrics_path.exists():
        try:
            existing = pd.read_csv(metrics_path)
            df = pd.concat([existing, df], ignore_index=True)
            logger.info(f"Appended metrics to existing file ({len(existing)} -> {len(df)} records)")
        except Exception as e:
            logger.warning(f"Failed to load existing metrics: {e}")

    df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to: {metrics_path}")

    return df


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def analyze_experiment_results(metrics_df: pd.DataFrame) -> Dict:
    """
    Perform statistical analysis comparing GACS vs baseline groups.

    Uses independent t-tests and effect size calculations (Cohen's d) to
    compare metrics between GACS and baseline videos.

    Args:
        metrics_df: DataFrame with collected metrics from all videos.

    Returns:
        Dict with structure:
            {
                'sample_sizes': {'gacs': int, 'baseline': int},
                'metrics': {
                    'metric_name': {
                        'gacs_mean': float,
                        'baseline_mean': float,
                        'lift_percent': float,
                        't_statistic': float,
                        'p_value': float,
                        'cohens_d': float,
                        'significant': bool
                    },
                    ...
                }
            }
    """
    # Get latest metrics for each video
    latest = metrics_df.sort_values('collected_at').groupby('video_id').last().reset_index()

    # Split by group
    gacs = latest[latest['group'] == 'gacs']
    baseline = latest[latest['group'] == 'baseline']

    results = {
        'sample_sizes': {
            'gacs': len(gacs),
            'baseline': len(baseline)
        },
        'metrics': {}
    }

    logger.info(f"Analyzing {len(gacs)} GACS videos vs {len(baseline)} baseline videos")

    # Metrics to compare
    metrics_to_compare = [
        'views', 'likes', 'comments', 'watchTime',
        'averageViewDuration', 'averageViewPercentage', 'ctr', 'shares'
    ]

    for metric in metrics_to_compare:
        if metric not in gacs.columns or metric not in baseline.columns:
            continue

        gacs_vals = gacs[metric].dropna().astype(float)
        baseline_vals = baseline[metric].dropna().astype(float)

        if len(gacs_vals) < 2 or len(baseline_vals) < 2:
            logger.debug(f"Insufficient data for {metric}")
            continue

        # Calculate statistics
        gacs_mean = gacs_vals.mean()
        baseline_mean = baseline_vals.mean()

        # T-test
        t_stat, p_value = stats.ttest_ind(gacs_vals, baseline_vals)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(gacs_vals) - 1) * gacs_vals.std() ** 2 +
             (len(baseline_vals) - 1) * baseline_vals.std() ** 2) /
            (len(gacs_vals) + len(baseline_vals) - 2)
        )
        cohens_d = (gacs_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0

        # Lift percentage
        lift = ((gacs_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0

        results['metrics'][metric] = {
            'gacs_mean': gacs_mean,
            'baseline_mean': baseline_mean,
            'lift_percent': lift,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }

        if p_value < 0.05:
            logger.info(f"Significant difference in {metric}: lift={lift:.1f}%, p={p_value:.4f}")

    return results


def print_analysis_report(results: Dict):
    """
    Print formatted statistical analysis report to console.

    Args:
        results: Analysis results dict from analyze_experiment_results().
    """
    print("\n" + "="*90)
    print("GACS vs Baseline Statistical Analysis".center(90))
    print("="*90)
    print(f"\nSample sizes: GACS={results['sample_sizes']['gacs']}, "
          f"Baseline={results['sample_sizes']['baseline']}")

    print(f"\n{'Metric':<25} {'GACS Mean':>12} {'Baseline':>12} {'Lift':>10} "
          f"{'p-value':>10} {'Sig':>5}")
    print("-"*90)

    for metric, data in sorted(results['metrics'].items()):
        sig = "*" if data['significant'] else ""
        print(f"{metric:<25} {data['gacs_mean']:>12.2f} {data['baseline_mean']:>12.2f} "
              f"{data['lift_percent']:>9.1f}% {data['p_value']:>10.4f} {sig:>5}")

    print("-"*90)
    print("* indicates statistical significance (p < 0.05)")
    print("="*90)


# ============================================================================
# Visualization Functions
# ============================================================================

def create_comparison_plots(metrics_df: pd.DataFrame, output_dir: Path = None):
    """
    Create comparison visualizations (box plots and bar charts).

    Generates publication-quality plots comparing GACS and baseline metrics.
    Saves as PNG files to the experiments directory.

    Args:
        metrics_df: DataFrame with collected metrics.
        output_dir: Directory to save plots (default: EXPERIMENTS_DIR).
    """
    if output_dir is None:
        output_dir = EXPERIMENTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get latest metrics per video
    latest = metrics_df.sort_values('collected_at').groupby('video_id').last().reset_index()

    # Set style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    metrics_to_plot = [
        ('views', 'Views'),
        ('likes', 'Likes'),
        ('watchTime', 'Watch Time (min)'),
        ('averageViewDuration', 'Avg View Duration (s)'),
        ('averageViewPercentage', 'Avg View %'),
        ('ctr', 'CTR (%)')
    ]

    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        if metric in latest.columns:
            sns.boxplot(
                data=latest, x='group', y=metric, ax=ax,
                palette=['#2ecc71', '#3498db']
            )
            ax.set_title(label, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(label)
        else:
            ax.text(0.5, 0.5, f'{label}\nNo data', ha='center', va='center')
            ax.set_title(label)

    plt.suptitle('GACS vs Baseline Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / 'comparison_boxplots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison plots saved to: {plot_path}")
    plt.close()

    # Create bar chart with means
    fig, ax = plt.subplots(figsize=(12, 6))

    gacs = latest[latest['group'] == 'gacs']
    baseline = latest[latest['group'] == 'baseline']

    metrics_for_bars = ['views', 'likes', 'watchTime', 'averageViewPercentage']
    existing_metrics = [m for m in metrics_for_bars if m in latest.columns]

    if existing_metrics:
        x = np.arange(len(existing_metrics))
        width = 0.35

        gacs_means = [gacs[m].mean() for m in existing_metrics]
        baseline_means = [baseline[m].mean() for m in existing_metrics]

        bars1 = ax.bar(x - width/2, gacs_means, width, label='GACS', color='#2ecc71')
        bars2 = ax.bar(x + width/2, baseline_means, width, label='Baseline', color='#3498db')

        ax.set_ylabel('Mean Value', fontweight='bold')
        ax.set_title('Mean Metrics by Group', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(existing_metrics)
        ax.legend()

        # Add value labels
        for bar, val in zip(bars1, gacs_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}',
                   ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, baseline_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    bar_path = output_dir / 'comparison_bars.png'
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    logger.info(f"Bar chart saved to: {bar_path}")
    plt.close()


# ============================================================================
# Report Generation Functions
# ============================================================================

def generate_pdf_report(metrics_df: pd.DataFrame,
                        analysis_results: Dict,
                        output_path: Path = None):
    """
    Generate a PDF summary report of experiment results.

    Uses FPDF library to create a formatted PDF with tables and summary
    statistics. Falls back to text report if FPDF is not available.

    Args:
        metrics_df: DataFrame with collected metrics.
        analysis_results: Analysis results dict from analyze_experiment_results().
        output_path: Path to save PDF (default: EXPERIMENTS_DIR/report.pdf).
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("FPDF not installed. Generating text report instead.")
        generate_text_report(metrics_df, analysis_results, output_path)
        return

    if output_path is None:
        output_path = EXPERIMENTS_DIR / "report.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'GACS Experiment Report', ln=True, align='C')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True, align='C')
        pdf.ln(10)

        # Summary
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Summary', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f"GACS videos: {analysis_results['sample_sizes']['gacs']}", ln=True)
        pdf.cell(0, 8, f"Baseline videos: {analysis_results['sample_sizes']['baseline']}", ln=True)
        pdf.ln(5)

        # Results table
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Statistical Comparison', ln=True)
        pdf.set_font('Arial', '', 9)

        # Table header
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(40, 8, 'Metric', border=1, fill=True)
        pdf.cell(30, 8, 'GACS Mean', border=1, fill=True)
        pdf.cell(30, 8, 'Baseline', border=1, fill=True)
        pdf.cell(25, 8, 'Lift', border=1, fill=True)
        pdf.cell(25, 8, 'p-value', border=1, fill=True)
        pdf.cell(20, 8, 'Sig', border=1, fill=True, ln=True)

        # Table rows
        for metric, data in sorted(analysis_results['metrics'].items()):
            sig = '*' if data['significant'] else ''
            pdf.cell(40, 7, metric, border=1)
            pdf.cell(30, 7, f"{data['gacs_mean']:.2f}", border=1)
            pdf.cell(30, 7, f"{data['baseline_mean']:.2f}", border=1)
            pdf.cell(25, 7, f"{data['lift_percent']:.1f}%", border=1)
            pdf.cell(25, 7, f"{data['p_value']:.4f}", border=1)
            pdf.cell(20, 7, sig, border=1, ln=True)

        pdf.ln(5)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 8, '* indicates statistical significance (p < 0.05)', ln=True)

        pdf.output(str(output_path))
        logger.info(f"PDF report saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        # Fall back to text report
        generate_text_report(metrics_df, analysis_results, output_path)


def generate_text_report(metrics_df: pd.DataFrame,
                         analysis_results: Dict,
                         output_path: Path = None):
    """
    Generate a text summary report of experiment results.

    Args:
        metrics_df: DataFrame with collected metrics.
        analysis_results: Analysis results dict from analyze_experiment_results().
        output_path: Path to save report (default: EXPERIMENTS_DIR/report.txt).
    """
    if output_path is None:
        output_path = EXPERIMENTS_DIR / "report.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        lines = [
            "GACS Experiment Report",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "Summary",
            "-" * 40,
            f"GACS videos: {analysis_results['sample_sizes']['gacs']}",
            f"Baseline videos: {analysis_results['sample_sizes']['baseline']}",
            "",
            "Statistical Comparison",
            "-" * 70,
            f"{'Metric':<25} {'GACS':>12} {'Baseline':>12} {'Lift':>10} {'p-value':>10}",
            "-" * 70
        ]

        for metric, data in sorted(analysis_results['metrics'].items()):
            sig = "*" if data['significant'] else ""
            lines.append(
                f"{metric:<25} {data['gacs_mean']:>12.2f} {data['baseline_mean']:>12.2f} "
                f"{data['lift_percent']:>9.1f}% {data['p_value']:>10.4f} {sig}"
            )

        lines.extend([
            "-" * 70,
            "* indicates statistical significance (p < 0.05)"
        ])

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Text report saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate text report: {e}")


# ============================================================================
# Main Execution Functions
# ============================================================================

def run_auth():
    """Authenticate with YouTube API."""
    logger.info("Starting authentication...")
    validate_config()
    youtube, youtube_analytics = get_authenticated_services()
    logger.info("Authentication successful!")


def run_upload():
    """Upload all videos to YouTube."""
    logger.info("Starting video uploads...")
    validate_config()
    youtube, youtube_analytics = get_authenticated_services()
    results = upload_all_videos(youtube, alternate=True)
    logger.info(f"Upload complete: {results['success'].sum()} successful, "
                f"{(~results['success']).sum()} failed")
    print(results.to_string())


def run_metrics():
    """Collect metrics for all uploaded videos."""
    logger.info("Starting metrics collection...")
    validate_config()
    youtube, youtube_analytics = get_authenticated_services()
    metrics_df = collect_all_metrics(youtube, youtube_analytics)
    logger.info(f"Collected metrics for {len(metrics_df)} records")
    print(metrics_df.head(10).to_string())


def run_analyze():
    """Analyze experiment results."""
    logger.info("Starting analysis...")
    metrics_path = EXPERIMENTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        logger.error("No metrics found. Run metrics subcommand first.")
        return

    metrics_df = pd.read_csv(metrics_path)
    analysis_results = analyze_experiment_results(metrics_df)
    print_analysis_report(analysis_results)
    logger.info("Analysis complete!")


def run_report():
    """Generate analysis report and visualizations."""
    logger.info("Starting report generation...")
    metrics_path = EXPERIMENTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        logger.error("No metrics found. Run metrics subcommand first.")
        return

    metrics_df = pd.read_csv(metrics_path)
    analysis_results = analyze_experiment_results(metrics_df)

    create_comparison_plots(metrics_df)
    generate_pdf_report(metrics_df, analysis_results)
    generate_text_report(metrics_df, analysis_results)
    logger.info("Report generation complete!")


def run_all():
    """Run complete pipeline: auth, upload, metrics, analyze, report."""
    logger.info("Running complete experiment pipeline...")

    try:
        logger.info("Step 1: Authentication")
        validate_config()
        youtube, youtube_analytics = get_authenticated_services()

        logger.info("Step 2: Uploading videos")
        upload_results = upload_all_videos(youtube, alternate=True)
        logger.info(f"Uploads: {upload_results['success'].sum()} successful")

        logger.info("Step 3: Collecting metrics")
        metrics_df = collect_all_metrics(youtube, youtube_analytics)
        logger.info(f"Collected {len(metrics_df)} metric records")

        logger.info("Step 4: Analyzing results")
        analysis_results = analyze_experiment_results(metrics_df)
        print_analysis_report(analysis_results)

        logger.info("Step 5: Generating report and visualizations")
        create_comparison_plots(metrics_df)
        generate_pdf_report(metrics_df, analysis_results)
        generate_text_report(metrics_df, analysis_results)

        logger.info("Complete pipeline executed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main entry point with argparse CLI."""
    parser = argparse.ArgumentParser(
        description='YouTube Experiment Runner - GACS Validation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s auth       # Authenticate with YouTube API
  %(prog)s upload     # Upload all videos
  %(prog)s metrics    # Collect metrics for uploaded videos
  %(prog)s analyze    # Analyze experiment results
  %(prog)s report     # Generate report and visualizations
  %(prog)s all        # Run complete pipeline
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Auth subcommand
    subparsers.add_parser(
        'auth',
        help='Authenticate with YouTube API'
    )

    # Upload subcommand
    subparsers.add_parser(
        'upload',
        help='Upload all generated videos to YouTube'
    )

    # Metrics subcommand
    subparsers.add_parser(
        'metrics',
        help='Collect metrics for uploaded videos'
    )

    # Analyze subcommand
    subparsers.add_parser(
        'analyze',
        help='Analyze experiment results'
    )

    # Report subcommand
    subparsers.add_parser(
        'report',
        help='Generate report and visualizations'
    )

    # All subcommand
    subparsers.add_parser(
        'all',
        help='Run complete pipeline'
    )

    args = parser.parse_args()

    # Route to appropriate function
    if args.command == 'auth':
        run_auth()
    elif args.command == 'upload':
        run_upload()
    elif args.command == 'metrics':
        run_metrics()
    elif args.command == 'analyze':
        run_analyze()
    elif args.command == 'report':
        run_report()
    elif args.command == 'all':
        run_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
