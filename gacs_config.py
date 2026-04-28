"""
GACS Research Pipeline - Central Configuration
================================================
Single source of truth for all constants, paths, and settings.

Pipeline modules (src/):
  - src/dataset_builder.py
  - src/video_generator.py
  - src/experiment_runner.py
  - src/labeling_backend.py

Usage:
    from gacs_config import *
    logger = setup_logging(__name__)
"""

import logging
import os
import warnings
from pathlib import Path

# ============================================================================
# Environment Variable Loading (.env file support)
# ============================================================================

def _load_dotenv(env_path: Path) -> None:
    """Load environment variables from a .env file if it exists.

    Parses KEY=VALUE lines. Supports '#' comments and blank lines.
    Quoted values (single or double) are unquoted automatically.
    Does NOT override variables already set in the environment.
    """
    if not env_path.is_file():
        return
    with open(env_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            # Do not override existing env vars
            if key not in os.environ:
                os.environ[key] = value


# Load .env from project root (must run before reading env vars)
_PROJECT_ROOT = Path(__file__).resolve().parent
_load_dotenv(_PROJECT_ROOT / ".env")


# ============================================================================
# Schema & Prompt Versioning
# ============================================================================

SCHEMA_VERSION = "1.0.0"
PROMPT_VERSION = "v1.0"


# ============================================================================
# API & Model Configuration
# ============================================================================

# Anthropic / Claude
CLAUDE_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Labeling backend: "claude" (API) or "local" (LLaVA on GPU)
LABELING_BACKEND = os.environ.get("GACS_LABELING_BACKEND", "claude").lower()

# Local vision model (used when LABELING_BACKEND == "local")
LOCAL_VISION_MODEL = os.environ.get(
    "GACS_LOCAL_VISION_MODEL", "llava-hf/llava-1.5-7b-hf"
)

# Sentence-BERT model for mood embeddings
SBERT_MODEL = "all-MiniLM-L6-v2"

# Rate limiting (seconds between API calls — only used for Claude API)
API_RATE_LIMIT_DELAY = 1.0


# ============================================================================
# GPU / CUDA Configuration
# ============================================================================

def _detect_gpu():
    """Detect CUDA GPU availability. Returns (available, device_name, vram_gb)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, "cuda", name, round(vram, 1)
    except ImportError:
        pass
    return False, "cpu", "N/A", 0.0


GPU_AVAILABLE, GPU_DEVICE, GPU_NAME, GPU_VRAM_GB = _detect_gpu()

# NVENC hardware encoding (4090 supports this)
USE_NVENC = GPU_AVAILABLE and os.environ.get("GACS_USE_NVENC", "1") == "1"

# RAPIDS cuML for GPU-accelerated KMeans/t-SNE
USE_RAPIDS = os.environ.get("GACS_USE_RAPIDS", "auto").lower()
if USE_RAPIDS == "auto":
    try:
        import cuml  # noqa: F401
        USE_RAPIDS = True
    except ImportError:
        USE_RAPIDS = False
else:
    USE_RAPIDS = USE_RAPIDS in ("1", "true", "yes")


# ============================================================================
# Scene Detection Settings
# ============================================================================

SCENE_THRESHOLD = 27.0      # ContentDetector threshold
MIN_SCENE_LENGTH = 15       # Minimum frames per scene


# ============================================================================
# Video Generation Settings
# ============================================================================

DEFAULT_K_CLUSTERS = 5      # Number of mood clusters
TARGET_DURATION = 30        # Target video length in seconds
MIN_SCENES = 5              # Minimum scenes per generated video
MAX_SCENES = 8              # Maximum scenes per generated video
OUTPUT_FPS = 30             # Frames per second for rendered videos
TRANSITION_DURATION = 0.5   # Fade transition duration in seconds


# ============================================================================
# YouTube API Settings
# ============================================================================

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


# ============================================================================
# Path Configuration (auto-created on import)
# ============================================================================

PROJECT_ROOT = _PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"

RAW_VIDEOS_DIR  = DATA_DIR / "raw_videos"
SCENES_DIR      = DATA_DIR / "scenes"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
EMBEDDINGS_DIR  = DATA_DIR / "embeddings"
GENERATED_DIR   = DATA_DIR / "generated"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
METRICS_DIR     = DATA_DIR / "metrics"
LOGS_DIR        = DATA_DIR / "logs"

# Create all directories on import
for _dir in [
    DATA_DIR,
    RAW_VIDEOS_DIR,
    SCENES_DIR,
    ANNOTATIONS_DIR,
    EMBEDDINGS_DIR,
    GENERATED_DIR,
    EXPERIMENTS_DIR,
    METRICS_DIR,
    LOGS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Quick-Test Mode (set env var GACS_QUICK_TEST=1 to enable)
# ============================================================================

QUICK_TEST_MODE = os.environ.get("GACS_QUICK_TEST", "").lower() in ("1", "true", "yes")

QUICK_TEST_MAX_VIDEOS = 1
QUICK_TEST_MAX_SCENES = 3
QUICK_TEST_UPLOAD_PRIVACY = "unlisted"


# ============================================================================
# Structured Logging
# ============================================================================

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def setup_logging(name: str) -> logging.Logger:
    """Return a configured logger with console (INFO) and file (DEBUG) handlers.

    Log files are written to ``data/logs/<name>.log``.  Calling this function
    multiple times with the same *name* returns the same logger instance
    without adding duplicate handlers.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` ready to use.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(console_handler)

    # File handler - DEBUG level
    log_file = LOGS_DIR / f"{name}.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger


# ============================================================================
# Config Validation
# ============================================================================

def validate_config() -> bool:
    """Check that essential environment variables and paths are usable.

    Emits warnings for missing or problematic settings but never raises,
    so the pipeline can still run in degraded mode.

    Returns:
        ``True`` if all checks pass, ``False`` if any warnings were issued.
    """
    ok = True

    # Check ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        warnings.warn(
            "ANTHROPIC_API_KEY is not set. Claude API calls will fail. "
            "Set it in your environment or in a .env file.",
            stacklevel=2,
        )
        ok = False

    # Check that key directories are writable
    for label, dir_path in [
        ("DATA_DIR", DATA_DIR),
        ("LOGS_DIR", LOGS_DIR),
        ("GENERATED_DIR", GENERATED_DIR),
        ("EXPERIMENTS_DIR", EXPERIMENTS_DIR),
    ]:
        if not dir_path.exists():
            warnings.warn(f"{label} does not exist: {dir_path}", stacklevel=2)
            ok = False
        elif not os.access(dir_path, os.W_OK):
            warnings.warn(f"{label} is not writable: {dir_path}", stacklevel=2)
            ok = False

    # Warn if quick-test mode is active
    if QUICK_TEST_MODE:
        warnings.warn(
            "QUICK_TEST_MODE is enabled. Processing will be limited to "
            f"{QUICK_TEST_MAX_VIDEOS} video(s) and {QUICK_TEST_MAX_SCENES} scene(s).",
            stacklevel=2,
        )

    # GPU info
    if GPU_AVAILABLE:
        logging.info(f"GPU detected: {GPU_NAME} ({GPU_VRAM_GB}GB VRAM)")
        logging.info(f"NVENC encoding: {'enabled' if USE_NVENC else 'disabled'}")
        logging.info(f"RAPIDS cuML: {'enabled' if USE_RAPIDS else 'disabled'}")
        logging.info(f"Labeling backend: {LABELING_BACKEND}")
    else:
        warnings.warn("No CUDA GPU detected. Running on CPU.", stacklevel=2)

    # Validate labeling backend
    if LABELING_BACKEND == "claude" and not ANTHROPIC_API_KEY:
        warnings.warn(
            "LABELING_BACKEND='claude' but ANTHROPIC_API_KEY is not set.",
            stacklevel=2,
        )
        ok = False
    elif LABELING_BACKEND == "local" and not GPU_AVAILABLE:
        warnings.warn(
            "LABELING_BACKEND='local' but no GPU detected. Will be very slow.",
            stacklevel=2,
        )

    return ok
