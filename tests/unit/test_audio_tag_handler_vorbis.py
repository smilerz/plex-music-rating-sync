"""
Unit tests for the VorbisHandler: tag parsing, scale inference, and strategy finalization.

# TODO: Add test for apply_tags writing to both FMPS_RATING and RATING
# TODO: Add test for fallback to unrated if normalization fails without aggressive inference
# TODO: Add test for finalize_rating_strategy tie-break behavior
# TODO: Add test for finalize_rating_strategy no-conflicts no-op
# TODO: Add test for is_strategy_supported() returning False for PRIORITIZED_ORDER
"""

import pytest

from filesystem_provider import VorbisField, VorbisHandler
from ratings import Rating, RatingScale
from sync_items import AudioTag
from tests.helpers import make_raw_rating


@pytest.fixture
def handler():
    return VorbisHandler(tagging_policy={"conflict_resolution_strategy": "highest"})


@pytest.fixture
def fake_vorbis_file():
    file = type("FakeVorbis", (), {})()
    file.filename = "song.flac"
    file.tags = {
        "ARTIST": ["Artist"],
        "ALBUM": ["Album"],
        "TITLE": ["Track"],
        "TRACKNUMBER": ["1"],
        "FMPS_RATING": ["0.8"],
        "RATING": ["4.5"],
    }
    file.info = type("FakeInfo", (), {"length": 123})()
    return file


def test_extract_metadata(monkeypatch, vorbis_file_factory, handler):
    fake_file = vorbis_file_factory()
    monkeypatch.setattr("mutagen.File", lambda *a, **kw: fake_file)

    tag, raw = handler.extract_metadata(fake_file)
    assert isinstance(tag, AudioTag)
    assert "FMPS_RATING" in raw
    assert "RATING" in raw


@pytest.mark.parametrize(
    "value,field,expected",
    [
        (make_raw_rating("FMPS_RATING", 0.8, rating_scale=RatingScale.NORMALIZED), VorbisField.FMPS_RATING, Rating),
        (make_raw_rating("RATING", 0.9, rating_scale=RatingScale.ZERO_TO_FIVE), VorbisField.RATING, Rating),
    ],
)
def test_try_normalize_inference(value, field, expected, handler):
    result = handler._try_normalize(value, field)
    assert isinstance(result, expected)


def test_conflict_resolution_highest(handler):
    tag = AudioTag(ID="z", title="T", artist="A", album="B", track=1)
    raw = {
        VorbisField.FMPS_RATING: make_raw_rating("FMPS_RATING", 0.7, rating_scale=RatingScale.NORMALIZED),
        VorbisField.RATING: make_raw_rating("RATING", 0.9, rating_scale=RatingScale.ZERO_TO_FIVE),
    }
    resolved = handler.resolve_rating(raw, tag)
    assert isinstance(resolved, Rating)
    assert round(resolved.to_float(RatingScale.ZERO_TO_FIVE), 1) == 4.5
