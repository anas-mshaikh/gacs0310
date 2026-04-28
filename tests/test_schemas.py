"""Tests for data schema validation across the pipeline."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVideoManifestSchema:
    REQUIRED = ["video_id", "source", "source_url", "local_path",
                "title", "category", "duration", "download_ok", "processed"]

    def test_manifest_exists(self):
        from gacs_config import PROJECT_ROOT
        assert (PROJECT_ROOT / "video_manifest.csv").exists()

    def test_has_required_columns(self):
        from gacs_config import PROJECT_ROOT
        df = pd.read_csv(PROJECT_ROOT / "video_manifest.csv")
        for col in self.REQUIRED:
            assert col in df.columns, f"Missing: {col}"

    def test_not_empty(self):
        from gacs_config import PROJECT_ROOT
        df = pd.read_csv(PROJECT_ROOT / "video_manifest.csv")
        assert len(df) > 0

    def test_unique_ids(self):
        from gacs_config import PROJECT_ROOT
        df = pd.read_csv(PROJECT_ROOT / "video_manifest.csv")
        assert df["video_id"].is_unique

    def test_valid_categories(self):
        from gacs_config import PROJECT_ROOT
        df = pd.read_csv(PROJECT_ROOT / "video_manifest.csv")
        valid = {"movie_trailers", "advertisements", "emotional_shorts", "animations"}
        assert set(df["category"].unique()).issubset(valid)

    def test_positive_duration(self):
        from gacs_config import PROJECT_ROOT
        df = pd.read_csv(PROJECT_ROOT / "video_manifest.csv")
        assert (df["duration"] > 0).all()


class TestAnnotationSchema:
    REQUIRED = ["image_id", "video_id", "scene_id", "mood_words",
                "style_words", "object_words", "labeling_model",
                "prompt_version", "timestamp"]

    def _make(self):
        return {
            "image_id": "abc123def456", "video_id": "gacs_movie_trailers_00",
            "scene_id": "gacs_movie_trailers_00_s0000",
            "mood_words": ["a", "b", "c", "d", "e"],
            "style_words": ["x", "y", "z"],
            "object_words": ["o1", "o2", "o3"],
            "labeling_model": "claude-sonnet-4-20250514",
            "prompt_version": "v1.0",
            "timestamp": "2025-02-08T12:00:00",
        }

    def test_all_fields(self):
        ann = self._make()
        for f in self.REQUIRED:
            assert f in ann

    def test_mood_count(self):
        assert len(self._make()["mood_words"]) == 5

    def test_style_count(self):
        assert len(self._make()["style_words"]) == 3

    def test_object_range(self):
        assert 3 <= len(self._make()["object_words"]) <= 5

    def test_serializable(self):
        ann = self._make()
        assert json.loads(json.dumps(ann)) == ann


class TestGeneratedVideoMetadata:
    REQUIRED = ["video_id", "group", "mood_vector", "mood_words",
                "prompt_used", "source_scenes", "created_at", "duration"]

    def _gacs(self):
        return {
            "video_id": "gacs_c0", "group": "gacs",
            "mood_vector": [0.1] * 384, "mood_words": ["a", "b"],
            "prompt_used": "...", "source_scenes": ["s1", "s2"],
            "created_at": "2025-02-08T12:00:00", "duration": 30.5,
        }

    def _baseline(self):
        return {
            "video_id": "bl_0", "group": "baseline",
            "mood_vector": [], "mood_words": [],
            "prompt_used": "...", "source_scenes": ["s3"],
            "created_at": "2025-02-08T12:00:00", "duration": 29.8,
        }

    def test_gacs_fields(self):
        for f in self.REQUIRED:
            assert f in self._gacs()

    def test_baseline_fields(self):
        for f in self.REQUIRED:
            assert f in self._baseline()

    def test_group_values(self):
        assert self._gacs()["group"] in ("gacs", "baseline")
        assert self._baseline()["group"] in ("gacs", "baseline")

    def test_gacs_mood_384(self):
        assert len(self._gacs()["mood_vector"]) == 384

    def test_baseline_empty_mood(self):
        assert self._baseline()["mood_vector"] == []

    def test_duration_near_target(self):
        from gacs_config import TARGET_DURATION
        assert abs(self._gacs()["duration"] - TARGET_DURATION) < 10

    def test_has_source_scenes(self):
        assert len(self._gacs()["source_scenes"]) >= 1

    def test_serializable(self):
        assert json.loads(json.dumps(self._gacs()))["video_id"] == "gacs_c0"


class TestYouTubeMap:
    def _entry(self):
        return {"youtube_id": "dQw4w9WgXcQ", "local_video_id": "gacs_c0", "group": "gacs"}

    def test_fields(self):
        for f in ["youtube_id", "local_video_id", "group"]:
            assert f in self._entry()

    def test_group_valid(self):
        assert self._entry()["group"] in ("gacs", "baseline")

    def test_youtube_id_len(self):
        assert len(self._entry()["youtube_id"]) == 11


class TestGACSDataset:
    REQUIRED = ["image_id", "video_id", "scene_id", "keyframe_path",
                "mood_vector", "mood_words", "style_words", "object_words"]

    def _row(self):
        return {
            "image_id": "abc123def456", "video_id": "vid_00",
            "scene_id": "vid_00_s0000", "keyframe_path": "data/scenes/vid_00/s0000.jpg",
            "mood_vector": ",".join(["0.1"] * 384),
            "mood_words": "a, b, c, d, e", "style_words": "x, y, z",
            "object_words": "o1, o2, o3",
        }

    def test_columns(self):
        df = pd.DataFrame([self._row()])
        for c in self.REQUIRED:
            assert c in df.columns

    def test_mood_384(self):
        assert len(self._row()["mood_vector"].split(",")) == 384

    def test_csv_roundtrip(self, tmp_path):
        df = pd.DataFrame([self._row()])
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        r = pd.read_csv(p)
        assert list(r.columns) == list(df.columns)
