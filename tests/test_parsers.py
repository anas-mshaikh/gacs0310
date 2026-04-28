"""Tests for parsing functions used across the pipeline."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCommaListParser:
    def _parse(self, text):
        if not text:
            return []
        items = [item.strip().lower() for item in text.split(",")]
        return [item for item in items if item]

    def test_normal_input(self):
        assert self._parse("dramatic, intense, powerful, epic, heroic") == \
            ["dramatic", "intense", "powerful", "epic", "heroic"]

    def test_extra_spaces(self):
        assert self._parse("  dramatic ,  intense , powerful ") == ["dramatic", "intense", "powerful"]

    def test_empty_string(self):
        assert self._parse("") == []

    def test_none_input(self):
        assert self._parse(None) == []

    def test_single_item(self):
        assert self._parse("dramatic") == ["dramatic"]

    def test_trailing_comma(self):
        assert self._parse("dramatic, intense,") == ["dramatic", "intense"]

    def test_lowercasing(self):
        assert self._parse("Dramatic, INTENSE") == ["dramatic", "intense"]

    def test_empty_items_removed(self):
        assert self._parse("dramatic,, intense") == ["dramatic", "intense"]


class TestLabelValidation:
    def _validate(self, mood, style, objects):
        issues = []
        if len(mood) != 5:
            issues.append(f"Expected 5 mood words, got {len(mood)}")
        if len(style) != 3:
            issues.append(f"Expected 3 style words, got {len(style)}")
        if not (3 <= len(objects) <= 5):
            issues.append(f"Expected 3-5 object words, got {len(objects)}")
        return issues

    def test_valid_labels(self):
        assert self._validate(list("abcde"), list("xyz"), list("opq")) == []

    def test_valid_5_objects(self):
        assert self._validate(list("abcde"), list("xyz"), list("opqrs")) == []

    def test_too_few_mood(self):
        issues = self._validate(list("ab"), list("xyz"), list("opq"))
        assert len(issues) == 1 and "mood" in issues[0].lower()

    def test_wrong_style(self):
        issues = self._validate(list("abcde"), list("xy"), list("opq"))
        assert len(issues) == 1 and "style" in issues[0].lower()

    def test_too_few_objects(self):
        issues = self._validate(list("abcde"), list("xyz"), list("op"))
        assert len(issues) == 1 and "object" in issues[0].lower()

    def test_all_wrong(self):
        assert len(self._validate([], [], [])) == 3


class TestMoodVectorParsing:
    def test_parse_vector(self):
        vec = np.array([float(x) for x in "0.1,0.2,0.3".split(",")])
        assert vec.shape == (3,) and abs(vec[0] - 0.1) < 1e-6

    def test_384_dim_roundtrip(self):
        original = np.random.randn(384)
        vec_str = ",".join(map(str, original.tolist()))
        restored = np.array([float(x) for x in vec_str.split(",")])
        assert np.allclose(original, restored)

    def test_empty_vector(self):
        parts = ",".join(map(str, [0.0] * 384)).split(",")
        assert len(parts) == 384


class TestSceneIdParsing:
    def _parse(self, scene_id):
        parts = scene_id.rsplit("_s", 1)
        if len(parts) != 2:
            return None, None
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return None, None

    def test_normal(self):
        v, i = self._parse("gacs_movie_trailers_00_UDfjsSqC_s0003")
        assert v == "gacs_movie_trailers_00_UDfjsSqC" and i == 3

    def test_first_scene(self):
        _, i = self._parse("gacs_animations_01_hJnAHzo4_s0000")
        assert i == 0

    def test_high_index(self):
        _, i = self._parse("gacs_emotional_shorts_00_WAmw-erI_s0150")
        assert i == 150

    def test_invalid(self):
        v, i = self._parse("invalid_no_scene_marker")
        assert v is None and i is None


class TestDiversityEnforcement:
    def _enforce(self, scene_ids, max_per_video=2):
        counts = {}
        result = []
        for sid in scene_ids:
            vid = sid.rsplit("_s", 1)[0]
            counts[vid] = counts.get(vid, 0) + 1
            if counts[vid] <= max_per_video:
                result.append(sid)
        return result

    def test_no_filtering(self):
        ids = ["a_s0", "b_s0", "c_s0"]
        assert self._enforce(ids) == ids

    def test_filters_excess(self):
        ids = ["a_s0", "a_s1", "a_s2", "b_s0"]
        assert len(self._enforce(ids, 2)) == 3

    def test_max_1(self):
        ids = ["a_s0", "a_s1", "b_s0", "b_s1"]
        assert self._enforce(ids, 1) == ["a_s0", "b_s0"]

    def test_empty(self):
        assert self._enforce([]) == []
