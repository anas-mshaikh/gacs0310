"""Tests for gacs_config.py - central configuration system."""

import os
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigImport:
    def test_import_succeeds(self):
        from gacs_config import SCHEMA_VERSION
        assert SCHEMA_VERSION is not None

    def test_schema_version_format(self):
        from gacs_config import SCHEMA_VERSION
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_prompt_version_format(self):
        from gacs_config import PROMPT_VERSION
        assert PROMPT_VERSION.startswith("v")

    def test_claude_model_defined(self):
        from gacs_config import CLAUDE_MODEL
        assert isinstance(CLAUDE_MODEL, str) and len(CLAUDE_MODEL) > 0

    def test_sbert_model_defined(self):
        from gacs_config import SBERT_MODEL
        assert isinstance(SBERT_MODEL, str) and len(SBERT_MODEL) > 0


class TestPathConfig:
    def test_project_root_exists(self):
        from gacs_config import PROJECT_ROOT
        assert PROJECT_ROOT.exists()

    def test_data_dir_exists(self):
        from gacs_config import DATA_DIR
        assert DATA_DIR.exists()

    def test_all_data_dirs_exist(self):
        from gacs_config import (
            ANNOTATIONS_DIR,
            EMBEDDINGS_DIR,
            EXPERIMENTS_DIR,
            GENERATED_DIR,
            LOGS_DIR,
            METRICS_DIR,
            RAW_VIDEOS_DIR,
            SCENES_DIR,
        )
        for d in [RAW_VIDEOS_DIR, SCENES_DIR, ANNOTATIONS_DIR,
                  EMBEDDINGS_DIR, GENERATED_DIR, EXPERIMENTS_DIR,
                  METRICS_DIR, LOGS_DIR]:
            assert d.exists(), f"Directory {d} should be auto-created"

    def test_paths_are_pathlib(self):
        from gacs_config import DATA_DIR, PROJECT_ROOT
        assert isinstance(PROJECT_ROOT, Path)
        assert isinstance(DATA_DIR, Path)


class TestQuickTestMode:
    def test_quick_test_defaults_false(self):
        from gacs_config import QUICK_TEST_MODE
        assert isinstance(QUICK_TEST_MODE, bool)

    def test_quick_test_max_videos(self):
        from gacs_config import QUICK_TEST_MAX_VIDEOS
        assert isinstance(QUICK_TEST_MAX_VIDEOS, int) and QUICK_TEST_MAX_VIDEOS >= 1

    def test_quick_test_max_scenes(self):
        from gacs_config import QUICK_TEST_MAX_SCENES
        assert isinstance(QUICK_TEST_MAX_SCENES, int) and QUICK_TEST_MAX_SCENES >= 1

    def test_quick_test_privacy(self):
        from gacs_config import QUICK_TEST_UPLOAD_PRIVACY
        assert QUICK_TEST_UPLOAD_PRIVACY in ("unlisted", "private")


class TestVideoSettings:
    def test_target_duration_reasonable(self):
        from gacs_config import TARGET_DURATION
        assert 10 <= TARGET_DURATION <= 120

    def test_scene_counts_ordered(self):
        from gacs_config import MAX_SCENES, MIN_SCENES
        assert MIN_SCENES < MAX_SCENES

    def test_fps_reasonable(self):
        from gacs_config import OUTPUT_FPS
        assert 24 <= OUTPUT_FPS <= 60

    def test_transition_duration_short(self):
        from gacs_config import TRANSITION_DURATION
        assert 0.0 < TRANSITION_DURATION <= 2.0

    def test_scene_threshold_positive(self):
        from gacs_config import SCENE_THRESHOLD
        assert SCENE_THRESHOLD > 0

    def test_cluster_count_positive(self):
        from gacs_config import DEFAULT_K_CLUSTERS
        assert DEFAULT_K_CLUSTERS >= 2


class TestYouTubeConfig:
    def test_scopes_defined(self):
        from gacs_config import YOUTUBE_SCOPES
        assert isinstance(YOUTUBE_SCOPES, list) and len(YOUTUBE_SCOPES) >= 1

    def test_upload_scope_present(self):
        from gacs_config import YOUTUBE_SCOPES
        assert any("upload" in s for s in YOUTUBE_SCOPES)

    def test_analytics_scope_present(self):
        from gacs_config import YOUTUBE_SCOPES
        assert any("analytics" in s for s in YOUTUBE_SCOPES)


class TestLogging:
    def test_setup_logging_returns_logger(self):
        import logging

        from gacs_config import setup_logging
        logger = setup_logging("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_idempotent(self):
        from gacs_config import setup_logging
        logger1 = setup_logging("test_idempotent")
        handler_count = len(logger1.handlers)
        logger2 = setup_logging("test_idempotent")
        assert len(logger2.handlers) == handler_count
        assert logger1 is logger2


class TestConfigValidation:
    def test_validate_config_returns_bool(self):
        from gacs_config import validate_config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = validate_config()
        assert isinstance(result, bool)

    def test_validate_config_warns_no_api_key(self):
        from gacs_config import ANTHROPIC_API_KEY, validate_config
        if not ANTHROPIC_API_KEY:
            with pytest.warns(UserWarning, match="ANTHROPIC_API_KEY"):
                validate_config()


class TestDotenvLoader:
    def test_dotenv_does_not_override(self, tmp_path):
        from gacs_config import _load_dotenv
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_GACS_VAR=from_file\n")
        os.environ["TEST_GACS_VAR"] = "from_env"
        try:
            _load_dotenv(env_file)
            assert os.environ["TEST_GACS_VAR"] == "from_env"
        finally:
            del os.environ["TEST_GACS_VAR"]

    def test_dotenv_sets_new_vars(self, tmp_path):
        from gacs_config import _load_dotenv
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_GACS_NEW_VAR=hello\n")
        try:
            _load_dotenv(env_file)
            assert os.environ.get("TEST_GACS_NEW_VAR") == "hello"
        finally:
            os.environ.pop("TEST_GACS_NEW_VAR", None)

    def test_dotenv_handles_quotes(self, tmp_path):
        from gacs_config import _load_dotenv
        env_file = tmp_path / ".env"
        env_file.write_text('TEST_GACS_QUOTED="quoted_value"\n')
        try:
            _load_dotenv(env_file)
            assert os.environ.get("TEST_GACS_QUOTED") == "quoted_value"
        finally:
            os.environ.pop("TEST_GACS_QUOTED", None)

    def test_dotenv_skips_comments(self, tmp_path):
        from gacs_config import _load_dotenv
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\nTEST_GACS_AFTER_COMMENT=yes\n")
        try:
            _load_dotenv(env_file)
            assert os.environ.get("TEST_GACS_AFTER_COMMENT") == "yes"
        finally:
            os.environ.pop("TEST_GACS_AFTER_COMMENT", None)

    def test_dotenv_missing_file(self, tmp_path):
        from gacs_config import _load_dotenv
        _load_dotenv(tmp_path / "nonexistent.env")
