import pytest

from filesystem_provider import ID3Handler
from manager.config_manager import ConflictResolutionStrategy, TagWriteStrategy
from ratings import Rating, RatingScale
from sync_items import AudioTag
from tests.helpers import make_raw_rating


@pytest.fixture
def handler():
    return ID3Handler(
        tagging_policy={
            "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
            "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
            "default_tag": "MEDIAMONKEY",
        }
    )


def test_extract_metadata_popm_and_txxx(mp3_file_factory, handler):
    id3 = mp3_file_factory(txxx_rating=4.5, popm_rating=196)

    tag, raw = handler.extract_metadata(id3)
    assert isinstance(tag, AudioTag)
    assert "MEDIAMONKEY" in raw
    assert "TEXT" in raw


def test_try_normalize_popm_and_text(handler):
    r1 = handler._try_normalize(str(make_raw_rating("MEDIAMONKEY", 0.8, frame_type="POPM")), "MEDIAMONKEY")
    r2 = handler._try_normalize(make_raw_rating("TEXT", 0.9, frame_type="TEXT"), "TEXT")

    assert isinstance(r1, Rating)
    assert isinstance(r2, Rating)

    assert round(r1.to_float(RatingScale.NORMALIZED), 2) == 0.8
    assert round(r2.to_float(RatingScale.NORMALIZED), 2) == 0.9


def test_resolve_rating_simple(handler):
    tag = AudioTag(ID="a", title="t", artist="a", album="b", track=1)
    raw = {"TEXT": make_raw_rating("TEXT", 0.8, frame_type="TEXT")}
    rating = handler.resolve_rating(raw, tag)
    assert isinstance(rating, Rating)
    assert not rating.is_unrated


def test_conflict_resolution_highest(handler):
    tag = AudioTag(ID="b", title="t", artist="x", album="y", track=1)
    raw = {
        "TEXT": make_raw_rating("TEXT", 0.6, frame_type="TEXT"),
        "MEDIAMONKEY": str(make_raw_rating("MEDIAMONKEY", 0.8, frame_type="POPM")),
    }
    rating = handler.resolve_rating(raw, tag)
    assert rating.to_float(RatingScale.ZERO_TO_FIVE) == 4.0
