from unittest.mock import MagicMock

import pytest
from mutagen.id3 import POPM

from filesystem_provider import DefaultPlayerTags, ID3Field, ID3Handler
from manager.config_manager import ConflictResolutionStrategy, TagWriteStrategy
from ratings import Rating, RatingScale
from sync_items import AudioTag
from tests.helpers import get_popm_email, make_raw_rating


@pytest.fixture
def handler():
    return ID3Handler(
        tagging_policy={
            "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
            "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
            "default_tag": "MEDIAMONKEY",
        }
    )


class TestMetadataExtraction:
    def test_extract_metadata_popm_and_txxx(self, mp3_file_factory, handler):
        # Using explicit parameters for clarity
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


class TestConflictResolution:
    def test_resolve_rating_simple(self, handler):
        tag = AudioTag(ID="a", title="t", artist="a", album="b", track=1)
        raw = {"TEXT": make_raw_rating("TEXT", 0.8)}
        rating = handler.resolve_rating(raw, tag)
        assert isinstance(rating, Rating)
        assert not rating.is_unrated

    def test_conflict_resolution_highest(self, handler):
        tag = AudioTag(ID="b", title="t", artist="x", album="y", track=1)
        raw = {
            "TEXT": make_raw_rating("TEXT", 0.6),
            "MEDIAMONKEY": make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.8),
        }
        rating = handler.resolve_rating(raw, tag)
        assert rating.to_float(RatingScale.ZERO_TO_FIVE) == 4.0

    def test_conflict_resolution_lowest(self):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": ConflictResolutionStrategy.LOWEST,
                "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
                "default_tag": "MEDIAMONKEY",
            }
        )
        tag = AudioTag(ID="c", title="t", artist="x", album="y", track=1)
        raw = {
            "TEXT": make_raw_rating("TEXT", 0.6),
            "MEDIAMONKEY": make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.2),
        }
        rating = handler.resolve_rating(raw, tag)
        assert rating.to_float(RatingScale.NORMALIZED) == 0.2

    def test_conflict_resolution_average(self):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": ConflictResolutionStrategy.AVERAGE,
                "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
                "default_tag": "MEDIAMONKEY",
            },
        )
        tag = AudioTag(ID="d", title="t", artist="x", album="y", track=1)
        raw = {
            "TEXT": make_raw_rating("TEXT", 0.6),
            "MEDIAMONKEY": make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.2),
        }
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
        raw = {
            "MEDIAMONKEY": "999"  # invalid POPM value
        }
        rating = handler.resolve_rating(raw, tag)
        assert rating.is_unrated

    def test_resolve_rating_with_non_numeric_text(self, handler):
        tag = AudioTag(ID="g", title="t", artist="x", album="y", track=1)
        raw = {"TEXT": "five stars"}
        rating = handler.resolve_rating(raw, tag)
        assert rating.is_unrated


class TestApplyTagsBehavior:
    def test_apply_tags_write_default(self, monkeypatch, mp3_file_factory, handler):
        audio = mp3_file_factory()
        tag = AudioTag(ID="track1", title="Title", artist="Artist", album="Album", track=1, rating=Rating(4.5))

        handler.apply_tags(audio, tag, tag.rating)

        assert audio.tags[ID3Field.TITLE].text[0] == "Title"
        assert audio.tags[ID3Field.ARTIST].text[0] == "Artist"
        assert audio.tags[ID3Field.ALBUM].text[0] == "Album"
        assert audio.tags[ID3Field.TRACKNUMBER].text[0] == "1"
        assert any(k.startswith("POPM:") for k in audio.tags)

    def test_apply_tags_write_all(self, monkeypatch, mp3_file_factory):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
                "tag_write_strategy": TagWriteStrategy.WRITE_ALL,
                "default_tag": "MEDIAMONKEY",
            }
        )
        handler.discovered_rating_tags = {"MEDIAMONKEY", "TEXT"}

        audio = mp3_file_factory()
        monkeypatch.setattr(audio, "save", lambda: None)

        tag = AudioTag(ID="track2", title="Song", artist="Artist", album="Album", track=2, rating=Rating(4.0))

        handler.apply_tags(audio, tag, tag.rating)

        # Find all rating fields
        popm_frames = [f for k, f in audio.tags.items() if k.startswith("POPM:")]
        text_frame = audio.tags.get("TXXX:RATING", None)

        assert len(popm_frames) >= 1, "Expected at least one POPM frame"
        assert text_frame is not None, "Expected a TEXT (TXXX:RATING) frame"

        # Normalize ratings
        normalized_ratings = []

        for frame in popm_frames:
            normalized_ratings.append(Rating(frame.rating, RatingScale.POPM)._normalized)

        if text_frame:
            try:
                text_value = float(text_frame.text[0])
                normalized_ratings.append(Rating(text_value, RatingScale.ZERO_TO_FIVE)._normalized)
            except (ValueError, TypeError):
                pytest.fail("TXXX:RATING value is not a valid float")

        # All normalized ratings should be identical
        base = round(normalized_ratings[0], 2)
        for rating in normalized_ratings:
            assert round(rating, 2) == base, f"Normalized rating {rating} did not match expected {base}"

        # Optional: assert normalized value is approximately 0.8 (for 4.0 stars)
        assert base == 0.8

    def test_apply_tags_overwrite_default(self, monkeypatch, mp3_file_factory):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
                "tag_write_strategy": TagWriteStrategy.OVERWRITE_DEFAULT,
                "default_tag": "MEDIAMONKEY",
            }
        )
        # Create audio file with pre-existing ratings
        audio = mp3_file_factory(
            rating=0.6,
            rating_tags=["TEXT", "old1@email.com", "old2@email.com", "old3@email.com"],
            title="Old Title",
            artist="Old Artist",
        )

        tag = AudioTag(ID="track3", title="New Title", artist="New Artist", album="New Album", track=3, rating=Rating(5.0))

        handler.apply_tags(audio, tag, tag.rating)

        # Postcondition asserts
        assert audio.tags[ID3Field.TITLE].text[0] == "New Title"
        assert audio.tags[ID3Field.ARTIST].text[0] == "New Artist"
        assert audio.tags[ID3Field.ALBUM].text[0] == "New Album"
        assert audio.tags[ID3Field.TRACKNUMBER].text[0] == "3"

        post_popm_keys = [k for k in audio.tags if k.startswith("POPM:")]
        assert len(post_popm_keys) == 1, "Expected one new POPM frame after update"
        assert post_popm_keys[0] == DefaultPlayerTags.MEDIAMONKEY, "Expected new POPM tag for default player"
        assert audio.tags[post_popm_keys[0]].rating == 255, "Expected new POPM rating to be 255"

    def test_apply_tags_audiotag_only(self, monkeypatch, mp3_file_factory):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
                "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
                "default_tag": "MEDIAMONKEY",
            }
        )
        audio = mp3_file_factory()

        popm_email = get_popm_email(DefaultPlayerTags.MEDIAMONKEY)
        audio.tags[DefaultPlayerTags.MEDIAMONKEY] = POPM(email=popm_email, rating=196, count=0)

        original_rating = audio.tags[DefaultPlayerTags.MEDIAMONKEY].rating

        tag = AudioTag(ID="error2", title="Test Title", artist="Test Artist", album="Test Album", track=5)

        handler.apply_tags(audio, tag, None)

        # Metadata updated
        assert audio.tags[ID3Field.TITLE].text[0] == "Test Title"
        assert audio.tags[ID3Field.ARTIST].text[0] == "Test Artist"
        assert audio.tags[ID3Field.ALBUM].text[0] == "Test Album"
        assert audio.tags[ID3Field.TRACKNUMBER].text[0] == "5"

        # Rating still preserved
        assert DefaultPlayerTags.MEDIAMONKEY in audio.tags
        assert audio.tags[DefaultPlayerTags.MEDIAMONKEY].rating == original_rating

    def test_apply_tags_updates_rating_only(self, monkeypatch, mp3_file_factory):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
                "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
                "default_tag": "MEDIAMONKEY",
            }
        )
        audio = mp3_file_factory()

        # Capture existing metadata before update
        original_title = audio.tags.get(ID3Field.TITLE)
        original_artist = audio.tags.get(ID3Field.ARTIST)
        original_album = audio.tags.get(ID3Field.ALBUM)
        original_tracknumber = audio.tags.get(ID3Field.TRACKNUMBER).text[0]
        tag = AudioTag(ID="error3", title=None, artist=None, album=None, track=None)

        handler.apply_tags(audio, tag, Rating(4.5))

        # Verify rating field was updated
        assert any(k.startswith("POPM:") for k in audio.tags)
        assert audio.tags[DefaultPlayerTags.MEDIAMONKEY].rating == 242  # 4.5 stars mapped

        # Verify metadata fields were NOT changed
        assert audio.tags.get(ID3Field.TITLE).text[0] == original_title
        assert audio.tags.get(ID3Field.ARTIST).text[0] == original_artist
        assert audio.tags.get(ID3Field.ALBUM).text[0] == original_album
        assert audio.tags.get(ID3Field.TRACKNUMBER).text[0] == original_tracknumber


class TestRobustness:
    def test_apply_tags_on_bad_file(self, monkeypatch):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
                "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
                "default_tag": "MEDIAMONKEY",
            }
        )
        audio = MagicMock()
        audio.tags = None  # Simulate a corrupted or unloaded file
        audio.filename = "mock.mp3"
        audio.info = MagicMock(length=300)

        tag = AudioTag(ID="error3", title="Test", artist="Artist", album="Album", track=3, rating=Rating(4.0))

        handler.apply_tags(audio, tag, tag.rating)


class TestEdgeCases:
    def test_write_text_rating(self, monkeypatch, mp3_file_factory):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
                "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
                "default_tag": "TEXT",
            }
        )
        # Only use TEXT rating, no POPM
        audio = mp3_file_factory(rating=0.6, rating_tags=["TEXT"])

        tag = AudioTag(ID="text1", title="T", artist="A", album="B", track=1, rating=Rating(4.0))
        handler.apply_tags(audio, tag, tag.rating)

        assert DefaultPlayerTags.TEXT in audio.tags
        assert audio.tags[DefaultPlayerTags.TEXT].text[0] == "4"

    def test_write_to_invalid_default_tag(self, monkeypatch, mp3_file_factory):
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
                "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
                "default_tag": "FOOBAR",  # Not registered
            }
        )
        audio = mp3_file_factory()

        tag = AudioTag(ID="badtag", title="NoWrite", artist="N/A", album="N/A", track=9, rating=Rating(3.0))

        handler.apply_tags(audio, tag, tag.rating)

        # Should not raise, but also should not write rating
        assert all(
            not (isinstance(f, POPM) and f.rating == 176)  # 3.0 mapped POPM
            for f in audio.tags.values()
        )
        # Assert audio.tags exists after recovery
        assert audio.tags is not None, "audio.tags should not be None after applying tags"
        assert isinstance(audio.tags, dict) or hasattr(audio.tags, "keys"), "audio.tags should be dict-like after recovery"
