import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mutagen.id3 import COMM, POPM, TXXX
from mutagen.mp3 import MP3

from filesystem_provider import ID3, DefaultPlayerTags, ID3Field, ID3Handler
from manager.config_manager import ConflictResolutionStrategy, TagWriteStrategy
from ratings import Rating, RatingScale
from sync_items import AudioTag
from tests.helpers import add_or_update_id3frame, get_popm_email, make_raw_rating


@pytest.fixture
def mp3_file_factory():
    """Returns a function that creates a fresh copy of tests/test.mp3 for each test."""

    test_mp3_path = Path("tests/test.mp3")

    def _factory(rating: float = 1.0, rating_tags: list[str] | str | None = None, **kwargs):
        # Create a new temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        temp_path = Path(temp_path)

        # Copy template silent MP3
        shutil.copyfile(test_mp3_path, temp_path)

        # Load into Mutagen
        audio = MP3(temp_path)

        # Override save to prevent real writes during tests
        audio.save = MagicMock(side_effect=lambda *args, **kw: audio.save())

        if not rating_tags:
            rating_tags = [DefaultPlayerTags.TEXT, DefaultPlayerTags.MEDIAMONKEY]

        return add_or_update_id3frame(
            audio,
            title=kwargs.get("title", "Default Title"),
            artist=kwargs.get("artist", "Default Artist"),
            album=kwargs.get("album", "Default Album"),
            track=kwargs.get("track", "1/10"),
            rating=rating,
            rating_tags=rating_tags,
        )

    return _factory


@pytest.fixture
def handler():
    handler = ID3Handler(
        tagging_policy={
            "conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST,
            "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
            "default_tag": "MEDIAMONKEY",
        }
    )
    handler.logger = MagicMock()
    handler.stats_mgr = MagicMock()
    handler.stats_mgr.get.return_value = {"TEXT": 5, "MEDIAMONKEY": 3}
    return handler


@pytest.fixture
def id3_file_with_tags(mp3_file_factory):
    return mp3_file_factory()


@pytest.fixture
def id3_file_without_tags(mp3_file_factory):
    """Creates a real ID3FileType object with no tags."""
    audio = mp3_file_factory()
    audio.tags = None
    return audio


@pytest.fixture
def fake_file_with_tags():
    """Creates a non-ID3 file that has tags."""
    fake = MagicMock()
    fake.tags = ID3()
    fake.tags[ID3Field.TITLE] = TXXX(encoding=3, text=["Fake Title"])
    fake.tags[ID3Field.ARTIST] = TXXX(encoding=3, text=["Fake Artist"])
    fake.tags[ID3Field.ALBUM] = TXXX(encoding=3, text=["Fake Album"])
    fake.tags[ID3Field.TRACKNUMBER] = TXXX(encoding=3, text=["1/10"])
    fake.tags[DefaultPlayerTags.TEXT] = TXXX(encoding=3, text=["5"])
    return fake


@pytest.fixture
def random_object():
    """Completely random object."""
    return object()


@pytest.fixture
def conflict_ratings():
    return {
        DefaultPlayerTags.TEXT.name: Rating(2.5, RatingScale.ZERO_TO_FIVE),
        DefaultPlayerTags.MEDIAMONKEY.name: Rating(4.5, RatingScale.POPM),
    }


@pytest.fixture
def track(track_factory):
    return track_factory(ID="conflict", title="Test Track", artist="Artist", album="Album", track=1)


class TestCanHandle:
    @pytest.mark.parametrize(
        "file_fixture_name, expected",
        [
            ("id3_file_with_tags", True),
            ("id3_file_without_tags", True),
            ("fake_file_with_tags", False),
            ("random_object", False),
        ],
    )
    def test_can_handle_success_or_failure_based_on_file_type(self, file_fixture_name, expected, request, handler):
        file_obj = request.getfixturevalue(file_fixture_name)
        assert handler.can_handle(file_obj) == expected


class TestExtractMetadata:
    def test_extract_metadata_success_popm_and_txxx(self, mp3_file_factory, handler):
        id3 = mp3_file_factory(rating=0.9, rating_tags=[DefaultPlayerTags.TEXT, DefaultPlayerTags.MEDIAMONKEY, "POPM:test@test"])
        tag, raw = handler.extract_metadata(id3)
        assert isinstance(tag, AudioTag)
        assert DefaultPlayerTags.TEXT.name in raw
        assert DefaultPlayerTags.MEDIAMONKEY.name in raw
        assert any(key.startswith("UNKNOWN") for key in raw)

    def test_try_normalize_success_popm_and_text(self, handler):
        r1 = handler._try_normalize(str(make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.8)), DefaultPlayerTags.MEDIAMONKEY.name)
        r2 = handler._try_normalize(make_raw_rating(DefaultPlayerTags.TEXT, 0.9), DefaultPlayerTags.TEXT.name)
        assert isinstance(r1, Rating)
        assert isinstance(r2, Rating)
        assert r1.to_float(RatingScale.NORMALIZED) == 0.8
        assert r2.to_float(RatingScale.NORMALIZED) == 0.9

    def test_extract_metadata_ignores_non_popm_txxx_frames(self, handler, mp3_file_factory):
        """Should ignore frames that are not POPM or TXXX even if the key matches."""
        audio = mp3_file_factory()
        # Add a COMM frame with a rating key
        audio.tags["TXXX:RATING"] = COMM(encoding=3, lang="eng", desc="desc", text=["not a rating"])
        tag, raw = handler.extract_metadata(audio)
        # Should not include the COMM frame in raw
        assert "TEXT" not in raw and "TXXX:RATING" not in raw


class TestResolveRating:
    def test_resolve_rating_success_simple_rating(self, handler):
        tag = AudioTag(ID="a", title="t", artist="a", album="b", track=1)
        raw = {"TEXT": make_raw_rating("TEXT", 0.8)}
        rating = handler.resolve_rating(raw, tag)
        assert isinstance(rating, Rating)
        assert not rating.is_unrated

    @pytest.mark.parametrize(
        "conflict_strategy, expected, priority_order",
        [
            (ConflictResolutionStrategy.HIGHEST, 5.0, None),
            (ConflictResolutionStrategy.LOWEST, 1.0, None),
            (ConflictResolutionStrategy.AVERAGE, 3.0, None),
            (None, None, None),
            (ConflictResolutionStrategy.PRIORITIZED_ORDER, 5.0, ["MUSICBEE", "MEDIAMONKEY", "WINAMP"]),
            (ConflictResolutionStrategy.PRIORITIZED_ORDER, 3.0, ["WINAMP", "MEDIAMONKEY", "MUSICBEE"]),
            (ConflictResolutionStrategy.PRIORITIZED_ORDER, 0.0, ["WINAMP", "WINDOWSMEDIAPLAYER", "UNKNOWN"]),
        ],
    )
    def test_resolve_rating_conflict_resolution_strategies(self, conflict_strategy, expected, priority_order, handler):
        raw_input = {
            DefaultPlayerTags.TEXT.name: make_raw_rating(DefaultPlayerTags.TEXT, 0.2),
            DefaultPlayerTags.MEDIAMONKEY.name: make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.6),
            DefaultPlayerTags.MUSICBEE.name: make_raw_rating(DefaultPlayerTags.MUSICBEE, 1),
        }

        handler.conflict_resolution_strategy = conflict_strategy
        handler.tag_priority_order = priority_order
        tag = AudioTag(ID="conflict_case", title="Conflict", artist="Artist", album="Album", track=1)
        rating = handler.resolve_rating(raw_input, tag)

        if expected is None:
            assert rating is None
        else:
            assert rating.to_float(RatingScale.ZERO_TO_FIVE) == expected

    def test_resolve_rating_success_with_unregistered_tag(self, handler):
        tag = AudioTag(ID="e", title="t", artist="x", album="y", track=1)
        raw = {"POPM:test@test": "255"}
        rating = handler.resolve_rating(raw, tag)
        assert isinstance(rating, Rating)
        assert rating.to_float(RatingScale.ZERO_TO_FIVE) == 5.0

    def test_resolve_prioritized_order_failure_no_priority_order(self, handler):
        with pytest.raises(ValueError, match="No tag_priority_order for PRIORITIZED_ORDER"):
            handler._resolve_prioritized_order({"TEXT": Rating(4)}, AudioTag(ID="test", title="t", artist="a", album="b", track=1))

    @pytest.mark.parametrize("raw", [{"MEDIAMONKEY": "999"}, {"TEXT": "five stars"}])
    def test_resolve_rating_failure_invalid_inputs(self, handler, raw):
        tag = AudioTag(ID="fail", title="t", artist="x", album="y", track=1)
        rating = handler.resolve_rating(raw, tag)
        assert rating.is_unrated

    def test_resolve_rating_failure_all_failures_return_unrated(self, handler):
        tag = AudioTag(ID="fail_all", title="Bad Track", artist="Artist", album="Album", track=1)
        raw = {"TEXT": "banana", "MEDIAMONKEY": "not_a_rating"}

        rating = handler.resolve_rating(raw, tag)
        assert isinstance(rating, Rating)
        assert rating.is_unrated

    def test_resolve_rating_failure_partial_success(self, handler):
        tag = AudioTag(ID="partial_success", title="Half Good", artist="Artist", album="Album", track=2)
        raw = {"TEXT": "4.0", "MEDIAMONKEY": "bad!"}

        rating = handler.resolve_rating(raw, tag)
        assert rating == Rating(4.0, scale=RatingScale.ZERO_TO_FIVE)

    def test_resolve_rating_unknown_tag_failure(self, handler):
        tag = AudioTag(ID="partial_success", title="Half Good", artist="Artist", album="Album", track=2)
        raw = {"TEXT": "4.0", "UNKNOWN_TAG": "4.0"}

        with pytest.raises(ValueError, match="Unknown tag_key 'UNKNOWN_TAG'."):
            handler.resolve_rating(raw, tag)

    def test_resolve_rating_failure_unknown_strategy_fallback(self):
        # Force a bad strategy manually
        handler = ID3Handler(
            tagging_policy={
                "conflict_resolution_strategy": None,  # set initially None
                "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT,
                "default_tag": DefaultPlayerTags.MEDIAMONKEY.name,
            }
        )
        tag = AudioTag(ID="bad_conflict", title="Unknown Strategy", artist="Artist", album="Album", track=1)

        # Patch the strategy after init
        handler.conflict_resolution_strategy = "INVALID_STRATEGY"

        raw_input = {
            DefaultPlayerTags.TEXT.name: make_raw_rating(DefaultPlayerTags.TEXT, 0.2),
            DefaultPlayerTags.MEDIAMONKEY.name: make_raw_rating(DefaultPlayerTags.MEDIAMONKEY, 0.8),
        }

        rating = handler.resolve_rating(raw_input, tag)

        assert rating is None, "Expected None when unknown strategy is forced"

    def test_resolve_rating_all_fail_returns_unrated(self, handler):
        """If all tag normalizations fail, should return Rating.unrated()."""
        handler._try_normalize = MagicMock(return_value=None)
        tag = MagicMock(ID="dummy")

        result = handler.resolve_rating({"TEXT": "x", "MM": "y"}, tag)
        assert result == Rating.unrated()

    def test_resolve_rating_partial_fail_returns_none(self, handler):
        """If some tags normalize and some fail, should defer (return None)."""
        handler._try_normalize = lambda val, key: Rating(1.0) if key == "TEXT" else None
        tag = MagicMock(ID="dummy")

        result = handler.resolve_rating({"TEXT": "5", "MM": "???"}, tag)
        assert result is None

    def test_resolve_rating_all_match_returns_rating(self, handler):
        """If all normalized ratings are identical, that rating is returned."""
        r = Rating(3.5)
        handler._try_normalize = lambda val, key: r
        tag = MagicMock(ID="dummy")

        result = handler.resolve_rating({"TEXT": "3.5", "MM": "3.5"}, tag)
        assert result == r


class TestApplyTags:
    def test_audio_without_tags_success(self, handler, mp3_file_factory):
        audio = mp3_file_factory()
        audio.tags = {}
        expected_rating = Rating(4.5)
        handler.apply_tags(audio, None, expected_rating)
        assert audio.tags[DefaultPlayerTags.MEDIAMONKEY].rating == expected_rating.to_int(RatingScale.POPM)

    def test_apply_tags_success_metadata_only(self, handler, mp3_file_factory):
        audio = mp3_file_factory()
        popm_email = get_popm_email(DefaultPlayerTags.MEDIAMONKEY)
        audio.tags[DefaultPlayerTags.MEDIAMONKEY] = POPM(email=popm_email, rating=196, count=0)

        original_rating = audio.tags[DefaultPlayerTags.MEDIAMONKEY].rating

        tag = AudioTag(ID="error2", title="Test Title", artist="Test Artist", album="Test Album", track=5)

        handler.apply_tags(audio, tag, None)

        assert audio.tags[ID3Field.TITLE].text[0] == "Test Title"
        assert audio.tags[ID3Field.ARTIST].text[0] == "Test Artist"
        assert audio.tags[ID3Field.ALBUM].text[0] == "Test Album"
        assert audio.tags[ID3Field.TRACKNUMBER].text[0] == "5"
        assert audio.tags[DefaultPlayerTags.MEDIAMONKEY].rating == original_rating

    def test_apply_tags_success_rating_only(self, handler, mp3_file_factory):
        audio = mp3_file_factory(rating=0.6, rating_tags=[DefaultPlayerTags.TEXT, DefaultPlayerTags.MEDIAMONKEY])

        original_title = audio.tags.get(ID3Field.TITLE)
        original_artist = audio.tags.get(ID3Field.ARTIST)
        original_album = audio.tags.get(ID3Field.ALBUM)
        original_tracknumber = audio.tags.get(ID3Field.TRACKNUMBER).text[0]
        tag = AudioTag(ID="error3", title=None, artist=None, album=None, track=None)

        expected_rating = Rating(4.5)
        handler.apply_tags(audio, tag, expected_rating)

        assert any(k.startswith("POPM:") for k in audio.tags)
        assert audio.tags[DefaultPlayerTags.MEDIAMONKEY].rating == expected_rating.to_int(RatingScale.POPM)
        assert audio.tags.get(ID3Field.TITLE).text[0] == original_title
        assert audio.tags.get(ID3Field.ARTIST).text[0] == original_artist
        assert audio.tags.get(ID3Field.ALBUM).text[0] == original_album
        assert audio.tags.get(ID3Field.TRACKNUMBER).text[0] == original_tracknumber

    def test_apply_tags_write_all_no_tags(self, handler, mp3_file_factory):
        handler.tag_write_strategy = TagWriteStrategy.WRITE_ALL
        handler.discovered_rating_tags = {}

        rating = Rating(0.5, RatingScale.NORMALIZED)
        audio = mp3_file_factory(rating=rating)

        tag = AudioTag(ID="test", title="A")

        handler.apply_tags(audio, tag, Rating(4.5))

        assert audio.tags[DefaultPlayerTags.MEDIAMONKEY].rating == rating.to_int(RatingScale.POPM)
        assert audio.tags[DefaultPlayerTags.TEXT].text[0] == rating.to_str(RatingScale.ZERO_TO_FIVE)
        assert audio.tags.get(ID3Field.TITLE).text[0] == "A"

    @pytest.mark.parametrize(
        "write_strategy, expected_tags",
        [
            (TagWriteStrategy.OVERWRITE_DEFAULT, {DefaultPlayerTags.MEDIAMONKEY: 0.8}),
            (TagWriteStrategy.WRITE_ALL, {DefaultPlayerTags.MEDIAMONKEY: 0.8, DefaultPlayerTags.TEXT: 0.8, DefaultPlayerTags.WINAMP: 0.8}),
            (TagWriteStrategy.WRITE_DEFAULT, {DefaultPlayerTags.MEDIAMONKEY: 0.8, DefaultPlayerTags.TEXT: 0.6, DefaultPlayerTags.WINAMP: 0.6}),
            (None, {DefaultPlayerTags.TEXT: 0.6, DefaultPlayerTags.WINAMP: 0.6}),
        ],
    )
    def test_apply_tags_success_expected_tags(self, write_strategy, expected_tags, handler, mp3_file_factory):
        handler.tag_write_strategy = write_strategy

        all_tags = [DefaultPlayerTags.TEXT, DefaultPlayerTags.MEDIAMONKEY, DefaultPlayerTags.WINAMP]
        handler.discovered_rating_tags = {tag.name for tag in all_tags}
        audio = mp3_file_factory(rating=0.6, rating_tags=[DefaultPlayerTags.TEXT, DefaultPlayerTags.WINAMP])

        assert DefaultPlayerTags.MEDIAMONKEY not in audio.tags

        handler.apply_tags(audio, None, Rating(4))
        for tag in all_tags:
            frame = audio.tags.get(tag)

            if tag in expected_tags:
                expected = Rating(expected_tags[tag])
                assert frame is not None, f"{tag} should exist"
                if tag == "TXXX:RATING":
                    assert isinstance(frame, TXXX)
                    assert frame.text[0] == expected.to_str(RatingScale.ZERO_TO_FIVE)
                else:
                    assert isinstance(frame, POPM)
                    assert frame.rating == expected.to_float(RatingScale.POPM)
            else:
                assert frame is None, f"{tag} should NOT exist"

    def test_remove_existing_id3_tags_success(self, handler, mp3_file_factory):
        audio = mp3_file_factory()
        # Prepopulate with old rating frames
        assert DefaultPlayerTags.TEXT in audio.tags
        assert DefaultPlayerTags.MEDIAMONKEY in audio.tags

        handler._remove_existing_id3_tags(audio)

        assert DefaultPlayerTags.TEXT not in audio.tags
        assert DefaultPlayerTags.MEDIAMONKEY not in audio.tags


class TestReadTags:
    def test_read_tags_success_returns_raw_ratings(self, handler, mp3_file_factory):
        # Create mp3 file with conflicting ratings
        handler.conflict_resolution_strategy = None
        audio = mp3_file_factory(rating_tags=[DefaultPlayerTags.TEXT, DefaultPlayerTags.MEDIAMONKEY])
        audio = add_or_update_id3frame(audio, rating=0.2, rating_tags=DefaultPlayerTags.TEXT)
        audio = add_or_update_id3frame(audio, rating=0.8, rating_tags=DefaultPlayerTags.MEDIAMONKEY)

        track, raw = handler.read_tags(audio)

        assert isinstance(track, AudioTag)
        assert raw is not None, "Expected raw ratings returned when conflict resolution deferred"
        assert DefaultPlayerTags.TEXT.name in raw or DefaultPlayerTags.MEDIAMONKEY.name in raw

    def test_read_tags_success_no_conflicts(self, handler, mp3_file_factory):
        # Create mp3 file with no conflicting ratings
        audio = mp3_file_factory(rating=0.3, rating_tags=[DefaultPlayerTags.TEXT, DefaultPlayerTags.MEDIAMONKEY])

        track, raw = handler.read_tags(audio)

        assert isinstance(track, AudioTag)
        assert raw is None, "Expected None when no conflicts exist"


class TestResolveChoice:
    def test_resolve_choice_options_shape_correct(self, handler, conflict_ratings, track):
        # Capture the options list
        with patch.object(handler.prompt, "choice", return_value="Skip (no rating)") as mock_choice:
            handler._resolve_choice(conflict_ratings, track)
            args, kwargs = mock_choice.call_args
            _message, options = args[0], args[1]

            # Check options formatting
            assert options[-1] == "Skip (no rating)", "Last option should be Skip"
            for key, rating in conflict_ratings.items():
                player_name = handler.tag_registry.get_player_name_for_key(key)
                expected_option = f"{player_name:<30} : {rating.to_display()}"
                assert expected_option in options, f"Expected option '{expected_option}' missing"

    def test_resolve_choice_select_rating_success(self, handler, conflict_ratings, track):
        items = list(conflict_ratings.items())
        player_key, expected_rating = items[0]

        # Simulate user choosing the first player
        player_name = handler.tag_registry.get_player_name_for_key(player_key)
        expected_choice = f"{player_name:<30} : {expected_rating.to_display()}"

        with patch.object(handler.prompt, "choice", return_value=expected_choice):
            result = handler._resolve_choice(conflict_ratings, track)
            assert isinstance(result, Rating)
            assert result == expected_rating

    def test_resolve_choice_select_skip_success(self, handler, conflict_ratings, track):
        with patch.object(handler.prompt, "choice", return_value="Skip (no rating)"):
            result = handler._resolve_choice(conflict_ratings, track)
            assert result is None

    def test_resolve_choice_order_matches_items(self, handler, conflict_ratings, track):
        # Capture both items and options
        with patch.object(handler.prompt, "choice", side_effect=lambda message, options, **kwargs: options[0]):
            items = list(conflict_ratings.items())
            options = [f"{handler.tag_registry.get_player_name_for_key(key):<30} : {rating.to_display()}" for key, rating in items]
            options.append("Skip (no rating)")

            # Make sure options[i] matches items[i] (up to len(items))
            for idx in range(len(items)):
                player_key, rating = items[idx]
                player_name = handler.tag_registry.get_player_name_for_key(player_key)
                expected_option = f"{player_name:<30} : {rating.to_display()}"
                assert options[idx] == expected_option, f"Mismatch at index {idx}"

            # Confirm skip is last
            assert options[-1] == "Skip (no rating)"


@pytest.fixture
def mock_finalize_strategy_deps(request, handler, track_factory, monkeypatch):
    """
    Configures a handler with strategy settings, conflict state, and mocked dependencies
    for finalize_rating_strategy() tests.
    """
    config = request.param if hasattr(request, "param") else {}

    # Inject config attributes into handler
    for key in ("conflict_resolution_strategy", "tag_write_strategy", "default_tag", "tag_priority_order"):
        if key in config:
            setattr(handler, key, config[key])

    # Simulate handler-owned and external conflicts
    num_from_handler = config.get("conflicts_from_handler", 2)
    num_from_others = config.get("conflicts_from_others", 1)

    handler.conflicts = []
    for i in range(num_from_handler):
        handler.conflicts.append({"handler": handler, "track": track_factory(ID=f"own{i}")})
    for i in range(num_from_others):
        other = MagicMock()
        handler.conflicts.append({"handler": other, "track": track_factory(ID=f"other{i}")})

    handler.discovered_rating_tags = {"TEXT", "MEDIAMONKEY"}
    handler.stats_mgr.get.return_value = config.get("tag_counts", {"TEXT": 5, "MEDIAMONKEY": 3})

    # Mock UI and summary helpers
    handler._print_summary = MagicMock()
    handler._show_conflicts = MagicMock()
    handler.prompt = MagicMock()

    # Provide a mocked config manager
    handler.cfg = MagicMock()
    monkeypatch.setattr("filesystem_provider.get_manager", lambda: MagicMock(get_config_manager=lambda: handler.cfg))

    return handler


class TestFinalizeRatingStrategy:
    @pytest.mark.parametrize(
        "mock_finalize_strategy_deps, modify_handler_state",
        [
            # E1: No conflicts at all → exit unconditionally
            ({"conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST, "conflicts_from_handler": 0, "conflicts_from_others": 0, "tag_counts": {"TEXT": 5}}, False),
            # E2: Conflicts exist, but strategy is already set → exit
            # Fixture gives 2 handler-owned conflicts by default; delete one to suppress has_multiple
            ({"conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST, "tag_counts": {"TEXT": 5}}, True),
            # E3: PRIORITIZED_ORDER strategy, multiple tags, but tag_priority_order is set → exit
            ({"conflict_resolution_strategy": ConflictResolutionStrategy.PRIORITIZED_ORDER, "tag_priority_order": ["TEXT"], "tag_counts": {"TEXT": 5, "MM": 3}}, False),
            # E4a: Only one tag → avoids triggering has_multiple conditions → exit
            # Remove one conflict to suppress has_multiple
            ({"conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST, "tag_counts": {"TEXT": 5}}, True),
            # E4b: Multiple tags + tag write strategy is set → exit
            ({"conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST, "tag_write_strategy": TagWriteStrategy.WRITE_ALL, "tag_counts": {"TEXT": 5, "MM": 3}}, False),
            # E5a: Write strategy is WRITE_ALL (doesn't require default tag) → exit
            ({"tag_write_strategy": TagWriteStrategy.WRITE_ALL, "tag_counts": {"TEXT": 5, "MM": 3}}, False),
            # E5b: Write strategy is WRITE_DEFAULT, and default_tag is set → exit
            ({"tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT, "default_tag": "TEXT", "tag_counts": {"TEXT": 5, "MM": 3}}, False),
            ({"tag_write_strategy": TagWriteStrategy.OVERWRITE_DEFAULT, "default_tag": "TEXT", "tag_counts": {"TEXT": 5, "MM": 3}}, False),
        ],
        indirect=["mock_finalize_strategy_deps"],
    )
    def test_finalize_rating_strategy_early_exit_conditions(self, mock_finalize_strategy_deps, modify_handler_state):
        handler = mock_finalize_strategy_deps

        if modify_handler_state:
            # Drop one handler-owned conflict so only one remains
            handler.conflicts = [c for c in handler.conflicts if c["handler"] is not handler or c["track"].ID == "own0"]

            # Reduce tag_counts to 1 key to simulate has_multiple=False
            handler.discovered_rating_tags = {"TEXT"}

        handler.finalize_rating_strategy(handler.conflicts)
        handler._show_conflicts.assert_not_called()
        handler.prompt.assert_not_called()

    @pytest.mark.parametrize(
        "mock_finalize_strategy_deps",
        [{"conflict_resolution_strategy": ConflictResolutionStrategy.PRIORITIZED_ORDER, "tag_priority_order": ["TEXT"], "tag_write_strategy": None}],
        indirect=True,
    )
    def test_tag_priority_prompt_skipped_if_pre_set(self, mock_finalize_strategy_deps):
        handler = mock_finalize_strategy_deps

        handler.prompt.choice.side_effect = lambda msg, opts, **_: opts[0]
        handler.prompt.yes_no.return_value = True

        handler.finalize_rating_strategy(handler.conflicts)

        prompt_messages = [call.args[0] for call in handler.prompt.choice.call_args_list]
        assert not any("order of preference" in p.lower() for p in prompt_messages)

        # One prompt (write strategy), one confirm (save)
        assert handler.prompt.choice.call_count == 1
        assert handler.prompt.yes_no.call_count == 1
        assert handler._print_summary.called

    @pytest.mark.parametrize(
        "mock_finalize_strategy_deps",
        [{"conflict_resolution_strategy": ConflictResolutionStrategy.PRIORITIZED_ORDER, "tag_priority_order": None}],
        indirect=True,
    )
    def test_tag_priority_prompt_sets_value(self, mock_finalize_strategy_deps):
        handler = mock_finalize_strategy_deps

        # Return priority choice then write strategy
        handler.prompt.choice.side_effect = lambda msg, opts, **_: opts[:2]
        handler.prompt.yes_no.return_value = True

        handler.finalize_rating_strategy(handler.conflicts)

        expected_order = [handler.tag_registry.get_key_for_player_name(name) for name in handler.prompt.choice.call_args_list[0].args[1][:2]]

        assert handler.tag_priority_order == expected_order
        assert handler.prompt.choice.call_count == 1
        assert handler.prompt.yes_no.call_count == 1
        handler.cfg.save_config.assert_called_once()

    @pytest.mark.parametrize(
        "mock_finalize_strategy_deps",
        [{"conflict_resolution_strategy": None, "tag_write_strategy": TagWriteStrategy.WRITE_ALL}],
        indirect=True,
    )
    def test_write_strategy_prompt_skipped_if_pre_set(self, mock_finalize_strategy_deps):
        handler = mock_finalize_strategy_deps

        handler.prompt.choice.side_effect = [ConflictResolutionStrategy.HIGHEST.display]
        handler.prompt.yes_no.return_value = True

        handler.finalize_rating_strategy(handler.conflicts)

        prompts = [c.args[0] for c in handler.prompt.choice.call_args_list]
        assert not any("write strategy" in p.lower() for p in prompts)

        assert handler.prompt.choice.call_count == 1
        assert handler.prompt.yes_no.call_count == 1
        assert handler._print_summary.called

    @pytest.mark.parametrize(
        "mock_finalize_strategy_deps",
        [
            {"tag_write_strategy": TagWriteStrategy.WRITE_ALL},
            {"conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST, "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT, "default_tag": "TEXT"},
            {"conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST, "tag_write_strategy": TagWriteStrategy.OVERWRITE_DEFAULT, "default_tag": "TEXT"},
        ],
        indirect=True,
    )
    def test_default_tag_prompt_skipped_if_not_required(self, mock_finalize_strategy_deps):
        handler = mock_finalize_strategy_deps
        handler.prompt.choice.side_effect = lambda msg, opts, **_: opts[0]
        handler.prompt.yes_no.return_value = True

        handler.finalize_rating_strategy(handler.conflicts)

        prompt_messages = [call.args[0] for call in handler.prompt.choice.call_args_list]
        assert not any("media player do you use most often" in m.lower() for m in prompt_messages)

        # Now only assert prompt flow occurred *when* we expect it
        if handler.conflict_resolution_strategy is None:
            assert handler.prompt.choice.call_count >= 1
            assert handler.prompt.yes_no.call_count == 1
        else:
            assert handler.prompt.choice.call_count == 0
            assert handler.prompt.yes_no.call_count == 0

    @pytest.mark.parametrize(
        "mock_finalize_strategy_deps,user_choice_display,expected",
        [
            ({"conflict_resolution_strategy": None}, "Highest", ConflictResolutionStrategy.HIGHEST),
            ({"conflict_resolution_strategy": None}, "Lowest", ConflictResolutionStrategy.LOWEST),
            ({"conflict_resolution_strategy": None}, "Average", ConflictResolutionStrategy.AVERAGE),
        ],
        indirect=["mock_finalize_strategy_deps"],
    )
    def test_conflict_strategy_prompt_sets_value(self, mock_finalize_strategy_deps, user_choice_display, expected):
        handler = mock_finalize_strategy_deps
        handler.prompt.choice.side_effect = lambda msg, opts, **_: next(o for o in opts if user_choice_display in o)
        handler.prompt.yes_no.return_value = True
        handler.finalize_rating_strategy(handler.conflicts)
        assert handler.conflict_resolution_strategy == expected
        assert handler.prompt.choice.call_count == 1
        assert handler.prompt.yes_no.call_count == 1
        handler.cfg.save_config.assert_called_once()

    @pytest.mark.parametrize(
        "mock_finalize_strategy_deps,user_choice_display,expected",
        [
            ({"tag_write_strategy": None}, "Write All", TagWriteStrategy.WRITE_ALL),
            ({"tag_write_strategy": None}, "Write Default", TagWriteStrategy.WRITE_DEFAULT),
            ({"tag_write_strategy": None}, "Overwrite Default", TagWriteStrategy.OVERWRITE_DEFAULT),
        ],
        indirect=["mock_finalize_strategy_deps"],
    )
    def test_write_strategy_prompt_sets_value(self, mock_finalize_strategy_deps, user_choice_display, expected):
        handler = mock_finalize_strategy_deps
        handler.prompt.choice.side_effect = lambda msg, opts, **_: next(o for o in opts if user_choice_display in o)
        handler.prompt.yes_no.return_value = True
        handler.finalize_rating_strategy(handler.conflicts)
        assert handler.tag_write_strategy == expected
        assert handler.prompt.choice.call_count == 1
        assert handler.prompt.yes_no.call_count == 1
        handler.cfg.save_config.assert_called_once()

    @pytest.mark.parametrize(
        "mock_finalize_strategy_deps",
        [{"conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST, "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT, "default_tag": None}],
        indirect=True,
    )
    def test_default_tag_prompt_sets_value(self, mock_finalize_strategy_deps):
        handler = mock_finalize_strategy_deps
        handler.prompt.choice.side_effect = ["Text"]
        handler.prompt.yes_no.return_value = True
        handler.finalize_rating_strategy(handler.conflicts)
        assert handler.default_tag == "TEXT"
        assert handler.prompt.choice.call_count == 1
        assert handler.prompt.yes_no.call_count == 1
        handler.cfg.save_config.assert_called_once()

    @pytest.mark.parametrize("mock_finalize_strategy_deps", [{"conflict_resolution_strategy": None}], indirect=True)
    def test_user_declines_config_save(self, mock_finalize_strategy_deps):
        handler = mock_finalize_strategy_deps

        # Pick a safe strategy that avoids triggering PRIORITIZED_ORDER branch
        handler.prompt.choice.side_effect = [ConflictResolutionStrategy.HIGHEST.display]
        handler.prompt.yes_no.return_value = False

        handler.finalize_rating_strategy(handler.conflicts)

        assert handler.prompt.choice.call_count == 1
        assert handler.prompt.yes_no.call_count == 1
        handler.cfg.save_config.assert_not_called()

    @pytest.mark.parametrize("mock_finalize_strategy_deps", [{"conflict_resolution_strategy": None}], indirect=True)
    def test_show_conflicts_triggers_reprompt(self, mock_finalize_strategy_deps):
        handler = mock_finalize_strategy_deps
        handler.prompt.choice.side_effect = ["Show conflicts", ConflictResolutionStrategy.HIGHEST.display]
        handler.prompt.yes_no.return_value = True
        handler.finalize_rating_strategy(handler.conflicts)
        handler._show_conflicts.assert_called_once()
        assert handler.conflict_resolution_strategy == ConflictResolutionStrategy.HIGHEST
        assert handler.prompt.choice.call_count == 2
        assert handler.prompt.yes_no.call_count == 1
        handler.cfg.save_config.assert_called_once()

    @pytest.mark.parametrize(
        "mock_finalize_strategy_deps",
        [
            {"conflict_resolution_strategy": ConflictResolutionStrategy.HIGHEST, "tag_write_strategy": TagWriteStrategy.WRITE_DEFAULT, "default_tag": "TEXT"},
        ],
        indirect=True,
    )
    def test_finalize_rating_strategy_no_save_when_no_change(self, mock_finalize_strategy_deps):
        """Should not prompt to save or call save_config if no config values are changed (needs_save is False)."""
        handler = mock_finalize_strategy_deps
        # Patch prompt to always return the current config value, so nothing changes
        handler.prompt.choice.side_effect = lambda msg, opts, **_: opts[0] if opts else None
        handler.prompt.yes_no.return_value = True  # Should not be called

        handler.finalize_rating_strategy(handler.conflicts)

        handler.prompt.yes_no.assert_not_called()
        handler.cfg.save_config.assert_not_called()


class TestResolveConflictDispatch:
    def test_conflict_strategy_unsupported_falls_back_to_highest(self, handler):
        """If strategy is unsupported, fallback to HIGHEST should apply."""
        handler.conflict_resolution_strategy = MagicMock()
        handler.is_strategy_supported = lambda strat: False
        handler._resolve_highest = MagicMock(return_value=Rating(1))

        result = handler._resolve_conflict({"TEXT": Rating(1), "MM": Rating(2)}, MagicMock())
        assert result == Rating(1)
        handler._resolve_highest.assert_called_once()

    def test_conflict_strategy_unknown_uses_fallback(self, handler):
        """Unknown strategy dispatches to _resolve_unknown_strategy."""
        handler.conflict_resolution_strategy = "BOGUS"
        handler._resolve_unknown_strategy = MagicMock(return_value=None)

        result = handler._resolve_conflict({"TEXT": Rating(1), "MM": Rating(2)}, MagicMock())
        handler._resolve_unknown_strategy.assert_called_once()
        assert result is None


class TestApplyRating:
    def test_apply_rating_adds_txxx_rating(self, handler, mp3_file_factory):
        """Should add new TXXX:RATING tag when not present."""
        handler.tag_registry.get_id3_tag_for_key = lambda key: "TXXX:RATING"

        audio = mp3_file_factory()
        if "TXXX:RATING" in audio.tags:
            del audio.tags["TXXX:RATING"]
        assert "TXXX:RATING" not in audio.tags

        result = handler._apply_rating(audio, Rating(4.0), {"TEXT"})
        frame = audio.tags.get("TXXX:RATING")
        assert isinstance(frame, TXXX)
        assert frame.text[0] == "4"
        assert result is audio

    def test_apply_rating_updates_existing_txxx_rating(self, handler, mp3_file_factory):
        """Should update TXXX:RATING tag if it already exists."""
        handler.tag_registry.get_id3_tag_for_key = lambda key: "TXXX:RATING"

        audio = mp3_file_factory()
        add_or_update_id3frame(audio, rating=0.4, rating_tags=["TEXT"])

        result = handler._apply_rating(audio, Rating(3.0), {"TEXT"})
        frame = audio.tags.get("TXXX:RATING")
        assert isinstance(frame, TXXX)
        assert frame.text[0] == "3"
        assert result is audio

    def test_apply_rating_adds_popm(self, handler, mp3_file_factory):
        """Should add POPM tag if not already present."""
        handler.tag_registry.get_id3_tag_for_key = lambda key: "POPM:test@test"
        handler.tag_registry.get_popm_email_for_key = lambda key: "test@test"

        audio = mp3_file_factory()
        assert "POPM:test@test" not in audio.tags

        result = handler._apply_rating(audio, Rating(5.0), {"POPM"})
        frame = audio.tags.get("POPM:test@test")
        assert isinstance(frame, POPM)
        assert frame.rating == 255
        assert result is audio

    def test_apply_rating_updates_existing_popm(self, handler, mp3_file_factory):
        """Should update existing POPM tag if present."""
        handler.tag_registry.get_id3_tag_for_key = lambda key: "POPM:test@test"
        handler.tag_registry.get_popm_email_for_key = lambda key: "test@test"

        audio = mp3_file_factory()
        add_or_update_id3frame(audio, rating=0.5, rating_tags=["POPM:test@test"])

        result = handler._apply_rating(audio, Rating(1.0), {"POPM"})
        frame = audio.tags.get("POPM:test@test")
        assert isinstance(frame, POPM)
        assert frame.rating == 255
        assert result is audio

    def test_apply_rating_unknown_tag_warns_and_skips(self, handler, mp3_file_factory):
        """Should log warning and skip if tag is unknown."""
        handler.tag_registry.get_id3_tag_for_key = lambda key: None

        audio = mp3_file_factory()

        result = handler._apply_rating(audio, Rating(3.5), {"UNKNOWN"})
        assert result is audio
        assert "UNKNOWN" not in audio.tags
        handler.logger.warning.assert_called_once()

    def test_apply_rating_needs_save_false_non_popm(self, handler, mp3_file_factory):
        """Should not update TXXX:RATING if value is already correct (needs_save is False)."""
        handler.tag_registry.get_id3_tag_for_key = lambda key: "TXXX:RATING"
        audio = mp3_file_factory()
        # Set the tag to the correct value
        audio.tags["TXXX:RATING"].text = ["4"]
        result = handler._apply_rating(audio, Rating(4.0), {"TEXT"})
        # Should not change the value
        assert audio.tags["TXXX:RATING"].text[0] == "4"
        assert result is audio
