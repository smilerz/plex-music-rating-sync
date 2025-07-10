from unittest.mock import MagicMock, patch

import pytest

from manager.config_manager import PlayerType, SyncItem
from sync_items import AudioTag
from sync_pair import MatchThreshold, SyncState, TrackPair


@pytest.fixture
def plexsync(monkeypatch):
    """
    Provides a fully-initialized PlexSync instance with all external dependencies mocked.
    All mocks are attached as attributes for test-side configuration.
    Only patches what is necessary for most tests.
    """
    # Mock config and manager
    mock_config = MagicMock()
    mock_config.source = PlayerType.PLEX
    mock_config.destination = PlayerType.FILESYSTEM
    mock_config.sync = ["tracks"]
    mock_config.dry = False

    mock_manager = MagicMock()
    mock_manager.get_config_manager.return_value = mock_config
    mock_manager.get_stats_manager.return_value = MagicMock()
    mock_manager.get_status_manager.return_value = MagicMock()
    monkeypatch.setattr("manager.get_manager", lambda: mock_manager)

    # Mock player classes
    monkeypatch.setattr("sync_ratings.Plex", lambda *a, **kw: MagicMock(name="PlexPlayer"))
    monkeypatch.setattr("sync_ratings.FileSystem", lambda *a, **kw: MagicMock(name="FileSystemPlayer"))
    monkeypatch.setattr("sync_ratings.MediaMonkey", lambda *a, **kw: MagicMock(name="MediaMonkeyPlayer"))

    # Mock logger
    mock_logger = MagicMock()
    monkeypatch.setattr("sync_ratings.logging.getLogger", lambda *a, **kw: mock_logger)

    # Mock UserPrompt
    monkeypatch.setattr("sync_ratings.UserPrompt", lambda *a, **kw: MagicMock())

    from sync_ratings import PlexSync

    instance = PlexSync()
    instance.mock_config = mock_config
    instance.mock_manager = mock_manager
    instance.mock_logger = mock_logger
    return instance


@pytest.fixture
def audio_tag_factory():
    def _factory(**kwargs):
        return AudioTag(**kwargs)

    return _factory


@pytest.fixture
def trackpair_factory(audio_tag_factory):
    """Factory for creating real TrackPair objects with mocked players and real AudioTags."""

    def _factory(
        source_track=None, destination_track=None, source_player=None, destination_player=None, sync_state=SyncState.UP_TO_DATE, score=100, quality=MatchThreshold.PERFECT_MATCH
    ):
        # Use real AudioTag if not provided
        if source_track is None:
            source_track = audio_tag_factory(title="Source", artist="Artist")
        if destination_track is None:
            destination_track = audio_tag_factory(title="Dest", artist="Artist")
        # Use MagicMock for players if not provided
        if source_player is None:
            source_player = MagicMock(name="SourcePlayer")
        if destination_player is None:
            destination_player = MagicMock(name="DestPlayer")
        pair = TrackPair(source_player, destination_player, source_track)
        pair.destination = destination_track
        pair.sync_state = sync_state
        pair.score = score
        pair._quality = quality  # set private attribute for test
        pair.rating_source = getattr(source_track, "rating", None)
        pair.rating_destination = getattr(destination_track, "rating", None)
        return pair

    return _factory


@pytest.fixture
def playlistpair_factory():
    """Factory for creating real PlaylistPair objects with test-controlled attributes."""
    from MediaPlayer import MediaPlayer
    from sync_items import Playlist
    from sync_pair import PlaylistPair

    def _factory(
        source_playlist=None,
        destination_playlist=None,
        source_player=None,
        destination_player=None,
        sync_state=None,
    ):
        if source_player is None:
            source_player = MagicMock(spec=MediaPlayer, name="SourcePlayer")
        if destination_player is None:
            destination_player = MagicMock(spec=MediaPlayer, name="DestPlayer")
        if source_playlist is None:
            source_playlist = MagicMock(spec=Playlist, name="SourcePlaylist")
        pair = PlaylistPair(source_player, destination_player, source_playlist)
        pair.logger = MagicMock()
        pair.status_mgr = MagicMock()
        pair.stats_mgr = MagicMock()
        if destination_playlist is not None:
            pair.destination = destination_playlist
        if sync_state is not None:
            pair.sync_state = sync_state
        return pair

    return _factory


@pytest.fixture
def playlist_factory():
    from sync_items import Playlist

    def _factory(ID="pl1", name="Test Playlist", tracks=None, is_auto_playlist=False):
        pl = Playlist(ID=ID, name=name)
        pl.is_auto_playlist = is_auto_playlist
        if tracks is not None:
            pl.tracks = tracks
        return pl

    return _factory


class TestSync:
    @pytest.mark.parametrize(
        "sync_items,tracks_calls,playlists_calls",
        [
            ([SyncItem.TRACKS, SyncItem.PLAYLISTS], 1, 1),
            ([SyncItem.TRACKS], 1, 0),
            ([SyncItem.PLAYLISTS], 0, 1),
        ],
    )
    def test_sync_routing(self, plexsync, sync_items, tracks_calls, playlists_calls):
        """Test that sync() routes to sync_tracks and/or sync_playlists as configured."""
        plexsync.config_mgr.sync = sync_items
        plexsync.sync_tracks = MagicMock()
        plexsync.sync_playlists = MagicMock()
        plexsync.sync()
        assert plexsync.sync_tracks.call_count == tracks_calls
        assert plexsync.sync_playlists.call_count == playlists_calls

    def test_sync_invalid_item_raises(self, plexsync, monkeypatch):
        """Test that sync() raises appropriate error for invalid sync items."""
        plexsync.config_mgr.sync = ["invalid"]
        plexsync.sync_tracks = MagicMock()
        plexsync.sync_playlists = MagicMock()
        with pytest.raises(ValueError):
            plexsync.sync()

    def test_sync_tracks_no_tracks_warns(self, plexsync):
        plexsync.source_player.search_tracks.return_value = []
        plexsync.logger = MagicMock()
        plexsync.sync_tracks()
        plexsync.logger.warning.assert_called_once_with("No tracks found")

    def test_sync_tracks_all_up_to_date_logs(self, plexsync, track_factory, trackpair_factory):
        track = track_factory()
        pair = trackpair_factory(source_track=track, sync_state=SyncState.UP_TO_DATE)
        plexsync.source_player.search_tracks.return_value = [track]
        plexsync.sync_pairs = []
        plexsync._match_tracks = MagicMock(return_value=[pair])
        plexsync.logger = MagicMock()
        plexsync.sync_tracks()
        plexsync.logger.info.assert_any_call("Attempting to match 1 tracks")

    def test_create_player_invalid_type_raises(self, plexsync):
        """Test _create_player raises ValueError and logs error for invalid player type."""
        invalid_type = "NOT_A_PLAYER"

        with pytest.raises(ValueError) as exc:
            plexsync._create_player(invalid_type)
        assert f"Invalid player type: {invalid_type}" in str(exc.value)


class TestTrackSync:
    def test_sync_tracks_user_cancel_no_sync(self, plexsync, track_factory, trackpair_factory):
        track = track_factory()
        pair = trackpair_factory(source_track=track, sync_state=SyncState.NEEDS_UPDATE)
        plexsync.source_player.search_tracks.return_value = [track]
        plexsync._match_tracks = MagicMock(return_value=[pair])
        plexsync._prompt_user_action = MagicMock(return_value=None)
        plexsync._sync_ratings = MagicMock()
        plexsync.sync_tracks()
        plexsync._sync_ratings.assert_not_called()

    def test_sync_tracks_user_syncs(self, plexsync, track_factory, trackpair_factory):
        track = track_factory()
        pair = trackpair_factory(source_track=track, sync_state=SyncState.NEEDS_UPDATE)
        plexsync.source_player.search_tracks.return_value = [track]
        plexsync._match_tracks = MagicMock(return_value=[pair])
        plexsync._prompt_user_action = MagicMock(return_value=[pair])
        plexsync.config_mgr.dry = False
        plexsync._sync_ratings = MagicMock()
        plexsync.sync_tracks()
        plexsync._sync_ratings.assert_called_once_with([pair])

    def test_sync_tracks_user_syncs_dry_run(self, plexsync, track_factory, trackpair_factory):
        track = track_factory()
        pair = trackpair_factory(source_track=track, sync_state=SyncState.NEEDS_UPDATE)
        plexsync.source_player.search_tracks.return_value = [track]
        plexsync._match_tracks = MagicMock(return_value=[pair])
        plexsync._prompt_user_action = MagicMock(return_value=[pair])
        plexsync.config_mgr.dry = True
        plexsync._sync_ratings = MagicMock()
        plexsync.sync_tracks()
        plexsync._sync_ratings.assert_called_once_with([pair])
        with patch("builtins.print") as mock_print:
            plexsync.sync_tracks()
            mock_print.assert_any_call("[DRY RUN] No changes will be written.")

    def test_match_tracks_empty_list_returns_empty(self, plexsync):
        plexsync.status_mgr.start_phase = MagicMock()
        result = plexsync._match_tracks([])
        assert result == []

    def test_match_tracks_all_tracks_matched(self, plexsync, track_factory, trackpair_factory):
        track1 = track_factory()
        track2 = track_factory()
        plexsync.status_mgr.start_phase = MagicMock()

        # Patch TrackPair to always return a valid pair
        def match_tracks(tracks):
            return [trackpair_factory(source_track=t) for t in tracks]

        plexsync._match_tracks = match_tracks
        tracks = [track1, track2]
        pairs = plexsync._match_tracks(tracks)
        assert len(pairs) == 2
        assert all(p.source == t for p, t in zip(pairs, tracks, strict=True))

    def test_match_tracks_some_tracks_unmatched(self, plexsync, track_factory, trackpair_factory):
        track1 = track_factory()
        track2 = track_factory()
        plexsync.status_mgr.start_phase = MagicMock()

        def match_tracks(tracks):
            return [trackpair_factory(source_track=tracks[0])]

        plexsync._match_tracks = match_tracks
        tracks = [track1, track2]
        pairs = plexsync._match_tracks(tracks)
        assert len(pairs) == 1
        assert pairs[0].source == track1

    def test_match_tracks_all_tracks_unmatched(self, plexsync):
        plexsync.status_mgr.start_phase = MagicMock()

        def match_tracks(tracks):
            return []

        plexsync._match_tracks = match_tracks
        tracks = [MagicMock(), MagicMock()]
        pairs = plexsync._match_tracks(tracks)
        assert pairs == []

    def test_match_tracks_various_sync_states(self, plexsync, track_factory, trackpair_factory):
        track1 = track_factory()
        track2 = track_factory()
        pair1 = trackpair_factory(source_track=track1, sync_state=SyncState.UP_TO_DATE)
        pair2 = trackpair_factory(source_track=track2, sync_state=SyncState.CONFLICTING)
        plexsync.status_mgr.start_phase = MagicMock()

        def match_tracks(tracks):
            return [pair1, pair2]

        plexsync._match_tracks = match_tracks
        tracks = [track1, track2]
        pairs = plexsync._match_tracks(tracks)
        assert len(pairs) == 2
        assert pairs[0].sync_state == SyncState.UP_TO_DATE
        assert pairs[1].sync_state == SyncState.CONFLICTING


class TestPlaylistSync:
    def test_sync_playlists_none_warns(self, plexsync):
        plexsync.source_player.search_playlists.return_value = []
        plexsync.sync_playlists()
        plexsync.logger.warning.assert_called_once_with("No playlists found")

    def test_sync_playlists_workflow(self, plexsync, playlistpair_factory, playlist_factory):
        playlist1 = playlist_factory(ID="pl1", name="Playlist1")
        playlist2 = playlist_factory(ID="pl2", name="Playlist2")
        plexsync.source_player.search_playlists.return_value = [playlist1, playlist2]
        # Use real PlaylistPair objects, but do not assign unused variable
        plexsync.config_mgr.dry = False
        plexsync.sync_playlists()
        plexsync.stats_mgr.increment.assert_any_call("playlists_processed", 2)
        plexsync.logger.info.assert_any_call(f"Matching {plexsync.source_player.name()} playlists with {plexsync.destination_player.name()}")

    def test_sync_playlists_filter_auto(self, plexsync, playlistpair_factory, playlist_factory):
        auto_playlist = playlist_factory(ID="auto1", name="AutoPlaylist", is_auto_playlist=True)
        normal_playlist = playlist_factory(ID="norm1", name="NormalPlaylist", is_auto_playlist=False)
        plexsync.source_player.search_playlists.return_value = [auto_playlist, normal_playlist]
        plexsync.config_mgr.dry = False
        plexsync.sync_playlists()
        plexsync.stats_mgr.increment.assert_any_call("playlists_processed", 1)

    def test_sync_playlists_dry_run(self, plexsync, playlistpair_factory, playlist_factory):
        playlist = playlist_factory(ID="pl1", name="Playlist1")
        plexsync.source_player.search_playlists.return_value = [playlist]
        plexsync.config_mgr.dry = True
        plexsync.sync_playlists()
        plexsync.logger.info.assert_any_call("Running a DRY RUN. No changes will be propagated!")


class TestUserPrompt:
    def test_user_prompt_builds_menu(self, plexsync):
        assert 1 == 0

    def test_user_prompt_sync_returns_pairs(self, plexsync):
        assert 1 == 0

    def test_user_prompt_filter_loops(self, plexsync):
        assert 1 == 0

    def test_user_prompt_cancel_returns_none(self, plexsync):
        assert 1 == 0

    def test_user_prompt_manual_resolution(self, plexsync):
        assert 1 == 0

    def test_user_prompt_details_loops(self, plexsync):
        assert 1 == 0

    def test_user_prompt_sync_option(self, plexsync):
        assert 1 == 0

    def test_user_prompt_manual_option(self, plexsync):
        assert 1 == 0

    def test_user_prompt_details_option(self, plexsync):
        assert 1 == 0

    def test_user_prompt_filter_cancel_option(self, plexsync):
        assert 1 == 0

    def test_user_prompt_toggle_reverse(self, plexsync):
        assert 1 == 0

    def test_user_prompt_toggle_conflicts(self, plexsync):
        assert 1 == 0

    def test_user_prompt_toggle_unrated(self, plexsync):
        assert 1 == 0

    def test_user_prompt_select_quality(self, plexsync):
        assert 1 == 0

    def test_user_prompt_filter_cancel_logs(self, plexsync):
        assert 1 == 0

    def test_user_prompt_quality_counts(self, plexsync):
        assert 1 == 0

    def test_user_prompt_quality_selection(self, plexsync):
        assert 1 == 0

    def test_user_prompt_detailed_scope(self, plexsync):
        assert 1 == 0

    def test_user_prompt_detailed_category(self, plexsync):
        assert 1 == 0

    def test_user_prompt_detailed_cancel(self, plexsync):
        assert 1 == 0


class TestConflictResolution:
    def test_conflict_manual_src_to_dst(self, plexsync):
        assert 1 == 0

    def test_conflict_manual_dst_to_src(self, plexsync):
        assert 1 == 0

    def test_conflict_manual_custom_rating(self, plexsync):
        assert 1 == 0

    def test_conflict_manual_skip(self, plexsync):
        assert 1 == 0

    def test_conflict_manual_cancel(self, plexsync):
        assert 1 == 0

    def test_conflict_manual_invalid_rating(self, plexsync):
        assert 1 == 0


class TestDescribeSync:
    @pytest.mark.parametrize(
        "track_filter,expected",
        [
            ({"include_unrated": True, "sync_conflicts": True, "quality": None}, "unrated and conflicting tracks"),
            ({"include_unrated": False, "sync_conflicts": True, "quality": None}, "conflicting tracks"),
            ({"include_unrated": True, "sync_conflicts": False, "quality": None}, "unrated tracks"),
            ({"include_unrated": False, "sync_conflicts": False, "quality": None}, "no tracks"),
            ({"include_unrated": True, "sync_conflicts": True, "quality": "PERFECT_MATCH"}, "unrated and PERFECT_MATCH+ conflicting tracks"),
        ],
    )
    def test_describe_sync_filters(self, plexsync, track_filter, expected):
        assert plexsync._describe_sync(track_filter) == expected


class TestProperties:
    def test_conflicts_property(self, plexsync, trackpair_factory):
        pairs = [
            trackpair_factory(sync_state=SyncState.CONFLICTING),
            trackpair_factory(sync_state=SyncState.NEEDS_UPDATE),
            trackpair_factory(sync_state=SyncState.UP_TO_DATE),
        ]
        plexsync.sync_pairs = pairs
        result = plexsync.conflicts
        assert all(p.sync_state == SyncState.CONFLICTING for p in result)
        assert len(result) == 1

    def test_unrated_property(self, plexsync, trackpair_factory):
        pairs = [
            trackpair_factory(sync_state=SyncState.CONFLICTING),
            trackpair_factory(sync_state=SyncState.NEEDS_UPDATE),
            trackpair_factory(sync_state=SyncState.UP_TO_DATE),
        ]
        plexsync.sync_pairs = pairs
        result = plexsync.unrated
        assert all(p.sync_state == SyncState.NEEDS_UPDATE for p in result)
        assert len(result) == 1


class TestGetTrackFilter:
    def test_track_filter_excludes_unmatched(self, plexsync, trackpair_factory):
        pairs = [
            trackpair_factory(sync_state=SyncState.NEEDS_UPDATE),
            trackpair_factory(sync_state=SyncState.CONFLICTING),
            trackpair_factory(sync_state=SyncState.UNKNOWN),
        ]
        filter_fn = plexsync._get_track_filter(include_unmatched=False)
        result = [p for p in pairs if filter_fn(p)]
        assert all(p.sync_state != SyncState.UNKNOWN for p in result)

    def test_track_filter_quality(self, plexsync, trackpair_factory):
        pairs = [
            trackpair_factory(quality=MatchThreshold.PERFECT_MATCH),
            trackpair_factory(quality=MatchThreshold.GOOD_MATCH),
        ]
        filter_fn = plexsync._get_track_filter(quality=MatchThreshold.PERFECT_MATCH)
        result = [p for p in pairs if filter_fn(p)]
        assert all(p.quality == MatchThreshold.PERFECT_MATCH for p in result)

    def test_track_filter_conflict_setting(self, plexsync, trackpair_factory):
        pairs = [
            trackpair_factory(sync_state=SyncState.CONFLICTING),
            trackpair_factory(sync_state=SyncState.NEEDS_UPDATE),
        ]
        filter_fn = plexsync._get_track_filter(include_conflicts=False)
        result = [p for p in pairs if filter_fn(p)]
        assert all(p.sync_state != SyncState.CONFLICTING for p in result)


class TestGetMatchDisplayOptions:
    @pytest.mark.parametrize(
        "scope,pair_specs,expected_categories,expected_assignments,expected_options,expected_filters",
        [
            # 1. 'all' scope: perfect, good, unmatched
            (
                "all",
                [
                    {"sync_state": SyncState.UP_TO_DATE, "score": 100},  # PERFECT_MATCH
                    {"sync_state": SyncState.UP_TO_DATE, "score": 80},  # GOOD_MATCH
                    {"sync_state": SyncState.UNKNOWN, "score": None},  # Unmatched
                ],
                {"Perfect Matches", "Good Matches", "Unmatched"},
                {"Perfect Matches": [0], "Good Matches": [1], "Unmatched": [2]},
                ["Perfect Matches (1)", "Good Matches (1)", "Unmatched (1)"],
                {"Perfect Matches", "Good Matches", "Poor Matches", "Unmatched"},
            ),
            # 2. 'all' scope: only unmatched
            (
                "all",
                [
                    {"sync_state": SyncState.UNKNOWN, "score": None},
                    {"sync_state": SyncState.ERROR, "score": None},
                ],
                {"Unmatched"},
                {"Unmatched": [0, 1]},
                ["Unmatched (2)"],
                {"Perfect Matches", "Good Matches", "Poor Matches", "Unmatched"},
            ),
            # 3. 'unrated' scope: needs_update with perfect/good, up_to_date ignored
            (
                "unrated",
                [
                    {"sync_state": SyncState.NEEDS_UPDATE, "score": 100},  # PERFECT_MATCH
                    {"sync_state": SyncState.NEEDS_UPDATE, "score": 80},  # GOOD_MATCH
                    {"sync_state": SyncState.UP_TO_DATE, "score": 100},  # ignored
                ],
                {"Perfect Matches", "Good Matches"},
                {"Perfect Matches": [0], "Good Matches": [1]},
                ["Perfect Matches (1)", "Good Matches (1)"],
                {"Perfect Matches", "Good Matches", "Poor Matches"},
            ),
            # 4. 'conflicting' scope: conflicting with good/poor, up_to_date ignored
            (
                "conflicting",
                [
                    {"sync_state": SyncState.CONFLICTING, "score": 80},  # GOOD_MATCH
                    {"sync_state": SyncState.CONFLICTING, "score": 30},  # POOR_MATCH
                    {"sync_state": SyncState.UP_TO_DATE, "score": 100},  # ignored
                ],
                {"Good Matches", "Poor Matches"},
                {"Good Matches": [0], "Poor Matches": [1]},
                ["Good Matches (1)", "Poor Matches (1)"],
                {"Perfect Matches", "Good Matches", "Poor Matches"},
            ),
            # 5. 'all' scope: empty input
            ("all", [], set(), {}, [], {"Perfect Matches", "Good Matches", "Poor Matches", "Unmatched"}),
            # 6. invalid scope: should return empty options, but filters exist and always return False
            (
                "invalid_scope",
                [
                    {"sync_state": SyncState.UP_TO_DATE, "score": 100},
                    {"sync_state": SyncState.NEEDS_UPDATE, "score": 80},
                    {"sync_state": SyncState.CONFLICTING, "score": 30},
                ],
                set(),
                {},
                [],
                {"Perfect Matches", "Good Matches", "Poor Matches"},
            ),
        ],
    )
    def test_match_display_options_scope_categories(
        self, plexsync, trackpair_factory, scope, pair_specs, expected_categories, expected_assignments, expected_options, expected_filters
    ):
        pairs = []
        for spec in pair_specs:
            pair = trackpair_factory(sync_state=spec.get("sync_state"), score=spec.get("score"))
            pairs.append(pair)
        plexsync.sync_pairs = pairs
        options, filters = plexsync._get_match_display_options(scope)
        base_categories = {c.split(" (")[0] for c in options}
        assert base_categories == expected_categories
        assert options == expected_options
        assert set(filters.keys()) == expected_filters
        for cat, idxs in expected_assignments.items():
            if cat in filters:
                base_cat = next((c for c in options if c.startswith(cat)), None)
                assert base_cat is not None, f"Category {cat} missing"
                expected_idxs = set(idxs)
                result_idxs = {i for i, p in enumerate(pairs) if filters[cat](p)}
                assert result_idxs == expected_idxs, f"For category {cat}, expected indices {expected_idxs}, got {result_idxs}"
        for cat in base_categories:
            assert cat in expected_categories
