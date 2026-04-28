"""
GACS Labeling Backend - Dual Vision Model Support
===================================================
Supports two backends for keyframe labeling:
  - "claude": Anthropic Claude Vision API (high quality, costs money, rate limited)
  - "local":  Local LLaVA model on GPU (free, fast, slightly lower quality)

Controlled by LABELING_BACKEND in gacs_config.py or env var GACS_LABELING_BACKEND.

Usage:
    from src.labeling_backend import label_keyframe
    annotation = label_keyframe(image_path, video_id, scene_id)
"""

import base64
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gacs_config import (
    ANNOTATIONS_DIR,
    ANTHROPIC_API_KEY,
    API_RATE_LIMIT_DELAY,
    CLAUDE_MODEL,
    GPU_AVAILABLE,
    GPU_DEVICE,
    LABELING_BACKEND,
    LOCAL_VISION_MODEL,
    PROMPT_VERSION,
    setup_logging,
)

logger = setup_logging(__name__)


# ============================================================================
# Shared Data Structures
# ============================================================================

@dataclass
class Annotation:
    """Affective label annotation for a single keyframe."""
    image_id: str
    video_id: str
    scene_id: str
    mood: List[str]
    style: List[str]
    objects: List[str]
    prompt_version: str = PROMPT_VERSION
    backend: str = LABELING_BACKEND

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Prompts (versioned, shared across backends)
# ============================================================================

MOOD_PROMPT = """Analyze the mood/emotion of this video frame.
Return EXACTLY 5 mood descriptor words, comma-separated.
Example: tense, mysterious, dark, suspenseful, foreboding
Output ONLY the 5 words, nothing else."""

STYLE_PROMPT = """Analyze the visual style of this video frame.
Return EXACTLY 3 style descriptor words, comma-separated.
Example: cinematic, warm-toned, shallow-focus
Output ONLY the 3 words, nothing else."""

OBJECT_PROMPT = """List the main objects/subjects visible in this video frame.
Return 2-5 object names, comma-separated.
Example: car, road, sunset, mountains
Output ONLY the object names, nothing else."""


# ============================================================================
# Utility Functions
# ============================================================================

def _encode_image_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_media_type(image_path: Path) -> str:
    """Determine MIME type from file extension."""
    ext = image_path.suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        ext, "image/jpeg"
    )


def _parse_comma_list(text: str, expected_count: Optional[int] = None) -> List[str]:
    """Parse a comma-separated response into a clean list of words."""
    items = [w.strip().lower() for w in text.split(",") if w.strip()]
    if expected_count and len(items) != expected_count:
        logger.debug(
            f"Expected {expected_count} items, got {len(items)}: {items}"
        )
    return items


def _load_cached_annotation(image_id: str) -> Optional[Annotation]:
    """Load a cached annotation from disk if it exists."""
    cache_path = ANNOTATIONS_DIR / f"{image_id}.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            return Annotation(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Corrupt cache for {image_id}: {e}")
    return None


def _save_annotation(annotation: Annotation) -> None:
    """Save annotation to disk cache."""
    cache_path = ANNOTATIONS_DIR / f"{annotation.image_id}.json"
    cache_path.write_text(json.dumps(annotation.to_dict(), indent=2))


# ============================================================================
# Claude Vision API Backend
# ============================================================================

def _call_claude_vision(image_path: Path, prompt: str, max_retries: int = 3) -> str:
    """Call Claude Vision API with exponential backoff retry.

    Args:
        image_path: Path to the image file.
        prompt: Text prompt for the vision model.
        max_retries: Maximum number of retry attempts.

    Returns:
        Model response text.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    image_data = _encode_image_base64(image_path)
    media_type = _get_media_type(image_path)

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            time.sleep(API_RATE_LIMIT_DELAY)
            return message.content[0].text.strip()
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Claude API attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Claude API failed after {max_retries} attempts: {e}")


def _label_with_claude(image_path: Path, video_id: str, scene_id: str) -> Annotation:
    """Label a keyframe using Claude Vision API.

    Makes 3 separate API calls for mood, style, and objects.
    """
    image_id = image_path.stem

    mood_text = _call_claude_vision(image_path, MOOD_PROMPT)
    mood = _parse_comma_list(mood_text, expected_count=5)

    style_text = _call_claude_vision(image_path, STYLE_PROMPT)
    style = _parse_comma_list(style_text, expected_count=3)

    obj_text = _call_claude_vision(image_path, OBJECT_PROMPT)
    objects = _parse_comma_list(obj_text)

    return Annotation(
        image_id=image_id,
        video_id=video_id,
        scene_id=scene_id,
        mood=mood,
        style=style,
        objects=objects,
        backend="claude",
    )


# ============================================================================
# Local LLaVA Backend (GPU)
# ============================================================================

_local_model = None
_local_processor = None


def _get_local_model():
    """Lazy-load the local LLaVA model onto GPU.

    Returns:
        Tuple of (model, processor).
    """
    global _local_model, _local_processor

    if _local_model is not None:
        return _local_model, _local_processor

    logger.info(f"Loading local vision model: {LOCAL_VISION_MODEL}")
    logger.info(f"Device: {GPU_DEVICE}")

    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    _local_processor = AutoProcessor.from_pretrained(LOCAL_VISION_MODEL)
    _local_model = LlavaForConditionalGeneration.from_pretrained(
        LOCAL_VISION_MODEL,
        torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
        device_map=GPU_DEVICE if GPU_AVAILABLE else "cpu",
        low_cpu_mem_usage=True,
    )

    logger.info("Local vision model loaded successfully")
    return _local_model, _local_processor


def _call_local_vision(image_path: Path, prompt: str) -> str:
    """Call the local LLaVA model for image analysis.

    Args:
        image_path: Path to the image file.
        prompt: Text prompt for the vision model.

    Returns:
        Model response text.
    """
    import torch
    from PIL import Image

    model, processor = _get_local_model()
    image = Image.open(image_path).convert("RGB")

    # Format prompt for LLaVA
    conversation = f"USER: <image>\n{prompt}\nASSISTANT:"

    inputs = processor(text=conversation, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
        )

    # Decode and extract only the assistant's response
    full_text = processor.decode(output[0], skip_special_tokens=True)
    response = full_text.split("ASSISTANT:")[-1].strip()
    return response


def _label_with_local(image_path: Path, video_id: str, scene_id: str) -> Annotation:
    """Label a keyframe using local LLaVA model on GPU.

    Makes 3 separate inference calls for mood, style, and objects.
    """
    image_id = image_path.stem

    mood_text = _call_local_vision(image_path, MOOD_PROMPT)
    mood = _parse_comma_list(mood_text, expected_count=5)

    style_text = _call_local_vision(image_path, STYLE_PROMPT)
    style = _parse_comma_list(style_text, expected_count=3)

    obj_text = _call_local_vision(image_path, OBJECT_PROMPT)
    objects = _parse_comma_list(obj_text)

    return Annotation(
        image_id=image_id,
        video_id=video_id,
        scene_id=scene_id,
        mood=mood,
        style=style,
        objects=objects,
        backend="local",
    )


# ============================================================================
# Public API
# ============================================================================

def label_keyframe(
    image_path: Path,
    video_id: str,
    scene_id: str,
    skip_cache: bool = False,
    max_label_retries: int = 2,
) -> Optional[Annotation]:
    """Label a keyframe image using the configured backend.

    Automatically uses cache if available. Backend is determined by
    LABELING_BACKEND config (or GACS_LABELING_BACKEND env var).

    Args:
        image_path: Path to the keyframe image.
        video_id: Source video identifier.
        scene_id: Scene identifier within the video.
        skip_cache: If True, ignore cached annotations.
        max_label_retries: Number of retries on labeling failure.

    Returns:
        Annotation object, or None if labeling fails.
    """
    image_id = image_path.stem

    # Check cache first
    if not skip_cache:
        cached = _load_cached_annotation(image_id)
        if cached is not None:
            logger.debug(f"Cache hit: {image_id}")
            return cached

    # Select backend
    backend_fn = _label_with_claude if LABELING_BACKEND == "claude" else _label_with_local
    backend_name = LABELING_BACKEND

    for attempt in range(max_label_retries):
        try:
            annotation = backend_fn(image_path, video_id, scene_id)

            # Validate: at least some labels were produced
            if not annotation.mood and not annotation.style:
                logger.warning(f"Empty labels for {image_id}, retry {attempt+1}")
                continue

            _save_annotation(annotation)
            logger.info(
                f"Labeled {image_id} ({backend_name}): "
                f"mood={annotation.mood}, style={annotation.style}"
            )
            return annotation

        except Exception as e:
            logger.error(f"Labeling failed for {image_id} (attempt {attempt+1}): {e}")
            if attempt < max_label_retries - 1:
                time.sleep(2 ** attempt)

    logger.error(f"All labeling attempts failed for {image_id}")
    return None


def get_backend_info() -> Dict[str, Any]:
    """Return information about the current labeling backend.

    Returns:
        Dict with backend name, model, device, and availability info.
    """
    info = {
        "backend": LABELING_BACKEND,
        "prompt_version": PROMPT_VERSION,
    }

    if LABELING_BACKEND == "claude":
        info["model"] = CLAUDE_MODEL
        info["api_key_set"] = bool(ANTHROPIC_API_KEY)
        info["rate_limit_delay"] = API_RATE_LIMIT_DELAY
    else:
        info["model"] = LOCAL_VISION_MODEL
        info["device"] = GPU_DEVICE
        info["gpu_available"] = GPU_AVAILABLE

    return info


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GACS Labeling Backend")
    parser.add_argument("image", type=Path, nargs="?", help="Image to label")
    parser.add_argument("--backend", choices=["claude", "local"], help="Override backend")
    parser.add_argument("--info", action="store_true", help="Show backend info")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache")
    args = parser.parse_args()

    if args.info:
        info = get_backend_info()
        for k, v in info.items():
            print(f"  {k}: {v}")
    elif args.image:
        if args.backend:
            import gacs_config
            gacs_config.LABELING_BACKEND = args.backend

        result = label_keyframe(
            args.image,
            video_id="test",
            scene_id="test_s0001",
            skip_cache=args.no_cache,
        )
        if result:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("Labeling failed.")
    else:
        parser.print_help()
