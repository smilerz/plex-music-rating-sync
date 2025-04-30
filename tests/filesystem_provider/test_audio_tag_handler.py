# tests/unit/test_audio_tag_handler.py

import pytest

from filesystem_provider import AudioTagHandler, DefaultPlayerTags, ID3Handler
from ratings import Rating, RatingScale
from sync_items import AudioTag
from tests.helpers import make_raw_rating

# ---- Fake handler for base class testing ----


class FakeAudioTagHandler(AudioTagHandler):
    def can_handle(self, audio_file):
        return True

    def extract_metadata(self, audio_file):
        return {}, {}

    def _try_normalize(self, value, key):
        return None

    def apply_tags(self, audio_file, metadata, rating):
        return audio_file


# ---- Fixtures ----


@pytest.fixture
def handler():
    return ID3Handler(
        tagging_policy={
            "conflict_resolution_strategy": "highest",
            "tag_write_strategy": "WRITE_DEFAULT",
            "default_tag": "MEDIAMONKEY",
        }
    )


# ---- Metadata extraction tests ----


class TestMetadataExtraction:
    def test_extract_metadata_popm_and_txxx(self, mp3_file_factory, handler):
        id3 = mp3_file_factory(rating=0.9, rating_tags=[DefaultPlayerTags.TEXT, DefaultPlayerTags.MEDIAMONKEY, "POPM:test@test"])
        tag, raw = handler.extract_metadata(id3)
        assert isinstance(tag, AudioTag)
        assert DefaultPlayerTags.TEXT.name in raw
        assert DefaultPlayerTags.MEDIAMONKEY.name in raw
        assert any(key.startswith("UNKNOWN") for key in raw)

    def test_try_normalize_popm_and_text(self, handler):
        r1 = handler._try_normalize(str(make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.8)), DefaultPlayerTags.MEDIAMONKEY.name)
        r2 = handler._try_normalize(make_raw_rating(DefaultPlayerTags.TEXT, 0.9), DefaultPlayerTags.TEXT.name)
        assert isinstance(r1, Rating)
        assert isinstance(r2, Rating)
        assert r1.to_float(RatingScale.NORMALIZED) == 0.8
        assert r2.to_float(RatingScale.NORMALIZED) == 0.9


# ---- Conflict resolution tests ----


class TestConflictResolution:
    def test_resolve_rating_simple(self, handler):
        tag = AudioTag(ID="a", title="t", artist="a", album="b", track=1)
        raw = {"TEXT": make_raw_rating("TEXT", 0.8)}
        rating = handler.resolve_rating(raw, tag)
        assert isinstance(rating, Rating)
        assert not rating.is_unrated

    def test_conflict_resolution_highest(self, handler):
        tag = AudioTag(ID="b", title="t", artist="x", album="y", track=1)
        raw = {"TEXT": make_raw_rating("TEXT", 0.6), "MEDIAMONKEY": make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.8)}
        rating = handler.resolve_rating(raw, tag)
        assert rating.to_float(RatingScale.ZERO_TO_FIVE) == 4.0

    def test_conflict_resolution_lowest(self):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": "lowest",
                "tag_write_strategy": "WRITE_DEFAULT",
                "default_tag": "MEDIAMONKEY",
            }
        )
        tag = AudioTag(ID="c", title="t", artist="x", album="y", track=1)
        raw = {"TEXT": make_raw_rating("TEXT", 0.6), "MEDIAMONKEY": make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.2)}
        rating = handler.resolve_rating(raw, tag)
        assert rating.to_float(RatingScale.NORMALIZED) == 0.2

    def test_conflict_resolution_average(self):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": "average",
                "tag_write_strategy": "WRITE_DEFAULT",
                "default_tag": "MEDIAMONKEY",
            }
        )
        tag = AudioTag(ID="d", title="t", artist="x", album="y", track=1)
        raw = {"TEXT": make_raw_rating("TEXT", 0.6), "MEDIAMONKEY": make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.2)}
        rating = handler.resolve_rating(raw, tag)
        assert rating.to_float(RatingScale.ZERO_TO_FIVE) == 2.0

    def test_resolve_rating_with_unregistered_tag(self, handler):
        tag = AudioTag(ID="e", title="t", artist="x", album="y", track=1)
        raw = {"UNKNOWN_TAG": "3.5"}
        rating = handler.resolve_rating(raw, tag)
        assert isinstance(rating, Rating)
        assert rating.to_float(RatingScale.ZERO_TO_FIVE) == 3.5

    def test_resolve_rating_with_invalid_popm(self, handler):
        tag = AudioTag(ID="f", title="t", artist="x", album="y", track=1)
        raw = {"MEDIAMONKEY": "999"}
        rating = handler.resolve_rating(raw, tag)
        assert rating.is_unrated

    def test_resolve_rating_with_non_numeric_text(self, handler):
        tag = AudioTag(ID="g", title="t", artist="x", album="y", track=1)
        raw = {"TEXT": "five stars"}
        rating = handler.resolve_rating(raw, tag)
        assert rating.is_unrated


# ---- Resolve rating edge case tests ----


class TestResolveRatingEdgeCases:
    def test_resolve_rating_all_failures_returns_unrated(self, handler):
        tag = AudioTag(ID="fail_all", title="Bad Track", artist="Artist", album="Album", track=1)
        raw = {"TEXT": "banana", "MEDIAMONKEY": "not_a_rating", "UNKNOWN_TAG": "another_fail"}
        rating = handler.resolve_rating(raw, tag)
        assert isinstance(rating, Rating)
        assert rating.is_unrated

    def test_resolve_rating_partial_success_returns_none(self, handler):
        tag = AudioTag(ID="partial_success", title="Half Good", artist="Artist", album="Album", track=2)
        raw = {"TEXT": "4.0", "UNKNOWN_TAG": "bad!"}
        rating = handler.resolve_rating(raw, tag)
        assert rating is None

    def test_resolve_rating_conflict_triggers_conflict_resolution(self, handler):
        tag = AudioTag(ID="conflict_case", title="Conflict Track", artist="Artist", album="Album", track=3)
        raw = {"TEXT": "2.5", "MEDIAMONKEY": "4.5"}
        rating = handler.resolve_rating(raw, tag)
        assert isinstance(rating, Rating)
        assert not rating.is_unrated
