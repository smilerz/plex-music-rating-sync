from contextlib import nullcontext
from unittest.mock import MagicMock, call

import pytest
from plexapi.exceptions import BadRequest, NotFound

from ratings import Rating, RatingScale
from sync_items import Playlist


@pytest.fixture
def plex_api():
    """Single-responsibility Plex API mock (server side)."""
    api = MagicMock()
    api.libraries = None
    api.library_count = 1

    resource_mock = MagicMock(name="resource")
    connection_mock = MagicMock(name="connection")
    resource_mock.side_effect = lambda server: resource_mock
    api.resource = resource_mock

    def _connect_side_effect(*args, **kwargs):
        exc = getattr(api, "connect_raise", None)
        if exc is not None:
            raise exc
        return connection_mock

    def _make_default_libraries(count: int = 1):
        """Return a list of MagicMock sections shaped like Plex artist libraries, with fetchItem and createPlaylist side effects."""
        libraries = []
        for i in range(count):
            lib = MagicMock(
                key=i + 1,
                type="artist",
                title=f"Music Library {i + 1}",
            )
            lib.fetchItem.side_effect = _mock_fetch_item_side_effect
            lib.createPlaylist.side_effect = _make_side_effect("createPlaylist")
            libraries.append(lib)
        return libraries

    def _sections_side_effect():
        if api.libraries:
            return api.libraries
        return _make_default_libraries(api.library_count)

    def _find_track_by_id(test_tracks, value):
        """Return a track matching the given ID."""
        for track in test_tracks:
            track_id = getattr(track, "id", None)
            if track_id == value:
                return track
        raise RuntimeError("search error")

    def _find_playlist_by_id(test_playlists, value):
        """Return a playlist matching the given ID."""
        for playlist in test_playlists:
            playlist_id = getattr(playlist, "key", None)
            if playlist_id == str(value) or playlist_id == value:
                return playlist
        raise RuntimeError("search error")

    def _mock_search_tracks(api, *args, **kwargs):
        """Mock implementation of searchTracks for the plex_api fixture."""
        test_tracks = getattr(api, "test_tracks", [])
        if not args and not kwargs:
            return test_tracks
        if "title" in kwargs:
            title = kwargs["title"]
            return [t for t in test_tracks if getattr(t, "title", None) == title]
        if "track.userRating!" in kwargs:
            # Simulate rating search (very basic)
            return test_tracks
        if args:
            return _find_track_by_id(test_tracks, args[0])
        return test_tracks

    def _mock_fetch_item_side_effect(value):
        # ID Convention:
        # - Tracks: 1-999 (start at 1)
        # - Playlists: 1000+ (start at 1000)
        # - Nonexistent: 999999 (special case)

        value = int(value)
        if value == 999999:  # Special case for nonexistent items
            raise RuntimeError("search error")
        elif value < 1000:  # Track ID range
            test_tracks = getattr(api, "test_tracks", [])
            return _find_track_by_id(test_tracks, value)
        else:  # Playlist ID range (1000+)
            test_playlists = getattr(api, "test_playlists", [])
            return _find_playlist_by_id(test_playlists, value)

    def _make_side_effect(name):
        def _side_effect(*args, **kwargs):
            exc = getattr(api, f"{name}_raise", None)
            if exc is not None:
                raise exc
            if name == "playlists":
                return getattr(api, "test_playlists", [])
            if name == "searchTracks":
                return _mock_search_tracks(api, *args, **kwargs)
            if hasattr(api, f"{name}_return"):
                return getattr(api, f"{name}_return")
            return None

        return _side_effect

    resource_mock.connect = MagicMock(side_effect=_connect_side_effect)
    connection_mock.library = MagicMock(name="library")
    connection_mock.library.sections = MagicMock(side_effect=_sections_side_effect)
    connection_mock.playlists = MagicMock(side_effect=_make_side_effect("playlists"))

    api.playlists.side_effect = _make_side_effect("playlists")
    api.searchTracks.side_effect = _make_side_effect("searchTracks")

    # default: no error
    api.account_raise = api.connect_raise = api.fetchItem_raise = api.playlists_raise = api.createPlaylist_raise = api.searchTracks_raise = None
    return api


@pytest.fixture
def native_playlist_factory():
    """Factory for native playlist MagicMock objects with Plex shape."""

    def _factory(playlist):
        mock = MagicMock()
        mock.key = str(getattr(playlist, "ID", getattr(playlist, "key", "")))
        mock.title = getattr(playlist, "name", getattr(playlist, "title", ""))
        mock.smart = getattr(playlist, "is_auto_playlist", getattr(playlist, "smart", False))
        mock.playlistType = "audio"  # Required for _collect_playlists filter
        # items() returns tracks if set, else []
        tracks = getattr(playlist, "Tracks", getattr(playlist, "tracks", []))
        if not tracks and hasattr(playlist, "tracks"):
            tracks = playlist.tracks
        mock.items.return_value = tracks if tracks is not None else []
        return mock

    return _factory


@pytest.fixture
def native_track_factory():
    """Factory for native track MagicMock objects with Plex shape and correct key format."""

    def _factory(track):
        mock = MagicMock()
        mock.grandparentTitle = getattr(track, "artist", getattr(track, "grandparentTitle", ""))
        mock.parentTitle = getattr(track, "album", getattr(track, "parentTitle", ""))
        mock.title = getattr(track, "title", "")
        file_path = getattr(track, "file_path", None)
        mock.locations = [file_path] if file_path else getattr(track, "locations", ["/dev/null"])

        # Convert Rating object to float for Plex ZERO_TO_TEN scale
        rating = getattr(track, "rating", getattr(track, "userRating", None))
        if hasattr(rating, "to_float"):  # It's a Rating object
            mock.userRating = rating.to_float(RatingScale.ZERO_TO_TEN)
        else:
            mock.userRating = rating

        key_val = getattr(track, "ID", getattr(track, "key", getattr(track, "title", 1)))
        mock.id = key_val
        mock.key = f"/library/metadata/{key_val}"
        mock.index = getattr(track, "track", getattr(track, "index", 1))
        mock.duration = getattr(track, "duration", 1000)
        return mock

    return _factory


@pytest.fixture
def plex_player(monkeypatch, request, plex_api, native_playlist_factory, native_track_factory):
    """
    Refactored Plex player fixture using the new plex_api mock and native object factories.
    """
    # Build a MagicMock manager with all required attributes
    fake_manager = MagicMock()
    config_defaults = {
        "server": "mock_server",
        "username": "mock_user",
        "passwd": "mock_password",
        "token": None,
    }
    # Allow test-time overrides
    config_overrides = getattr(request, "param", {})
    config = MagicMock(**{**config_defaults, **config_overrides})
    fake_manager.get_config_manager.return_value = config

    def myplexaccount_patch(*args, **kwargs):
        # retry/exit tests use this hook
        if getattr(plex_api, "account_raise", None):
            raise plex_api.account_raise
        return plex_api

    monkeypatch.setattr("manager.get_manager", lambda: fake_manager)
    monkeypatch.setattr("MediaPlayer.MyPlexAccount", myplexaccount_patch)
    monkeypatch.setattr("MediaPlayer.getpass.getpass", lambda *args, **kwargs: "dummy_password")

    from MediaPlayer import Plex  # Import after patching

    plex = Plex()
    plex.config_mgr = config
    plex.logger = MagicMock()
    plex.status_mgr = MagicMock()  # Add missing status_mgr mock

    for k, v in config_overrides.items():
        setattr(plex.config_mgr, k, v)

    # Assign correct mocks for account, plex_api_connection, and music_library
    plex.account = plex_api
    plex.plex_api_connection = plex_api.resource(config["server"]).connect()
    plex.music_library = plex.plex_api_connection.library.sections()[0]

    return plex


class TestPlexConnect:
    @pytest.mark.parametrize(
        "password,token,account_return_error,expect_exit",
        [
            ("pass", None, None, False),
            (None, "token", None, False),
            ("pass", None, NotFound("not found"), True),
            ("pass", None, BadRequest("bad request"), True),
        ],
        ids=[
            "password_success",
            "token_success",
            "notfound_retries_and_exit",
            "badrequest_retries_and_exit",
        ],
    )
    def test_authenticate(self, plex_player, plex_api, password, token, account_return_error, expect_exit):
        plex_api.account_raise = account_return_error
        if expect_exit:
            with pytest.raises(SystemExit):
                plex_player._authenticate("server", "user", password, token)
        else:
            result = plex_player._authenticate("server", "user", password, token)
            assert isinstance(result, MagicMock)

    @pytest.mark.parametrize(
        "server,username,password,token,expect_exception,exception_type,exception_msg",
        [
            (None, "user", "pass", None, True, ValueError, "Plex server and username are required for Plex player"),
            ("server", None, "pass", None, True, ValueError, "Plex server and username are required for Plex player"),
            ("server", "user", None, None, True, ValueError, "Plex token or password is required for Plex player"),
            ("server", "user", "pass", "token", False, None, None),
        ],
        ids=[
            "missing_server",
            "missing_username",
            "missing_credentials",
            "all_present",
        ],
    )
    def test_connect_input_validation_errors(self, plex_player, server, username, password, token, expect_exception, exception_type, exception_msg, plex_api):
        plex_player.config_mgr.server = server
        plex_player.config_mgr.username = username
        plex_player.config_mgr.passwd = password
        plex_player.config_mgr.token = token
        plex_api.resource.connect.reset_mock()
        if expect_exception:
            with pytest.raises(exception_type) as e:
                plex_player.connect()
            assert exception_msg in str(e.value)
            assert plex_api.resource.connect.call_count == 0
        else:
            plex_player.connect()
            assert plex_api.resource.connect.call_count == 1

    @pytest.mark.parametrize(
        "library_count, simulate_input, expected_key, expect_exit",
        [
            (0, None, None, True),
            (1, None, 1, False),
            (2, "1", 1, False),
            (2, "bad\n2", 2, False),
        ],
        ids=[
            "no_libraries",
            "one_library",
            "multiple_valid",
            "multiple_retry_then_valid",
        ],
    )
    def test_connect_library_selection(self, plex_player, plex_api, monkeypatch, library_count, simulate_input, expected_key, expect_exit):
        plex_api.library_count = library_count

        if simulate_input is not None:
            inputs = simulate_input.splitlines()
            monkeypatch.setattr("builtins.input", lambda _: inputs.pop(0))

        context = pytest.raises(SystemExit) if expect_exit else nullcontext()

        with context:
            plex_player.connect()

        if not expect_exit:
            assert plex_player.music_library.key == expected_key


class TestPlexPlaylistSearch:
    @pytest.mark.parametrize(
        "key,value,return_native,setup_map,expect_error,expect_empty",
        [
            ("all", None, False, {}, False, False),
            ("title", "My Playlist", False, {"my playlist": 1000}, False, False),
            ("id", 1000, False, {}, False, False),
            ("badkey", None, False, {}, True, False),
            ("id", 999999, False, {}, False, True),
        ],
        ids=[
            "search_all_returns_list",
            "search_by_title_finds_playlist",
            "search_by_id_finds_playlist",
            "search_invalid_key_raises_error",
            "search_nonexistent_id_returns_empty",
        ],
    )
    def test_search_playlists_handles_various_parameters(self, plex_player, plex_api, key, value, return_native, setup_map, expect_error, expect_empty, native_playlist_factory):
        """Test playlist search handles different search parameters correctly."""
        plex_player._title_to_id_map = {k.lower(): v for k, v in setup_map.items()}

        if value == 1000:
            playlist = Playlist(ID=1000, name="My Playlist")
            native_playlist = native_playlist_factory(playlist)
            plex_api.test_playlists = [native_playlist]

        if expect_error:
            with pytest.raises(ValueError):
                plex_player.search_playlists(key, value, return_native)
        else:
            result = plex_player.search_playlists(key, value, return_native)
            if expect_empty:
                assert result == []
            else:
                assert isinstance(result, list)

    @pytest.mark.parametrize(
        "is_auto_playlist,num_items,expect_status_bar",
        [
            (True, 2, False),
            (False, 2, False),
            (False, 101, True),
        ],
        ids=[
            "auto_playlist_no_status_bar",
            "manual_playlist_few_tracks_no_status_bar",
            "manual_playlist_many_tracks_shows_status_bar",
        ],
    )
    def test_load_playlist_tracks_shows_progress_for_large_playlists(
        self, plex_player, plex_api, is_auto_playlist, num_items, expect_status_bar, track_factory, native_playlist_factory, native_track_factory
    ):
        """Test that progress bar appears only for large manual playlists."""
        playlist = Playlist(ID=1000, name="Test")
        playlist.is_auto_playlist = is_auto_playlist

        test_tracks = [track_factory(ID=i) for i in range(1, num_items + 1)]
        native_tracks = [native_track_factory(track) for track in test_tracks]

        native_playlist = native_playlist_factory(playlist)
        native_playlist.items.return_value = native_tracks

        plex_api.test_playlists = [native_playlist]
        plex_player.load_playlist_tracks(playlist)

        if expect_status_bar:
            plex_player.status_mgr.start_phase.assert_called_once()
        else:
            plex_player.status_mgr.start_phase.assert_not_called()

    def test_read_native_playlist_tracks_converts_to_audiotags(self, plex_player, track_factory, native_playlist_factory, native_track_factory):
        """Test reading tracks from native playlist converts all tracks to AudioTag objects."""
        # Create test tracks and convert to native format
        test_tracks = [track_factory(ID=1, title="Track 1"), track_factory(ID=2, title="Track 2")]
        native_tracks = [native_track_factory(track) for track in test_tracks]

        # Create native playlist
        native_playlist = native_playlist_factory(Playlist(ID=1001, name="Test"))
        native_playlist.items.return_value = native_tracks

        result = plex_player._read_native_playlist_tracks(native_playlist)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(hasattr(track, "title") and hasattr(track, "artist") for track in result)

    def test_get_native_playlist_track_count_returns_count(self, plex_player, track_factory, native_playlist_factory, native_track_factory):
        """Test counting tracks in native playlist returns accurate number."""
        # Create 3 test tracks
        test_tracks = [track_factory(ID=i, title=f"Track {i}") for i in range(1, 4)]
        native_tracks = [native_track_factory(track) for track in test_tracks]

        # Create native playlist
        native_playlist = native_playlist_factory(Playlist(ID=1002, name="Test"))
        native_playlist.items.return_value = native_tracks

        result = plex_player._get_native_playlist_track_count(native_playlist)

        assert result == 3

    def test_search_playlists_native_return_returns_native_objects(self, plex_player, plex_api, native_playlist_factory):
        """Test searching playlists with return_native=True returns Plex API objects."""
        playlist = Playlist(ID=1003, name="My Playlist")
        native_playlist = native_playlist_factory(playlist)
        plex_api.test_playlists = [native_playlist]

        result = plex_player.search_playlists("id", 1003, True)

        assert isinstance(result, list)
        assert len(result) == 1
        assert hasattr(result[0], "key")  # Native Plex objects have key attribute


class TestPlexPlaylistUpdate:
    def test_add_track_to_playlist_calls_addItems_on_native_playlist(self, plex_player, plex_api, track_factory, native_playlist_factory, native_track_factory):
        """Test adding track to playlist calls native addItems method correctly."""
        playlist = Playlist(ID=1002, name="Test")
        track = track_factory(ID=1)

        # Create native playlist and track
        native_playlist = native_playlist_factory(playlist)
        native_track = native_track_factory(track)

        # Setup mock to track addItems calls
        native_playlist.addItems = MagicMock()

        # Register with API mock
        plex_api.test_playlists = [native_playlist]
        plex_api.test_tracks = [native_track]

        plex_player._add_track_to_playlist(playlist, track)

        # Verify addItems was called with correct track
        native_playlist.addItems.assert_called_once_with(native_track)

    def test_remove_track_from_playlist_calls_removeItem_on_native_playlist(self, plex_player, plex_api, track_factory, native_playlist_factory, native_track_factory):
        """Test removing track from playlist calls native removeItem method correctly."""
        playlist = Playlist(ID=1003, name="Test")
        track = track_factory(ID=1)

        # Create native playlist and track
        native_playlist = native_playlist_factory(playlist)
        native_track = native_track_factory(track)

        # Setup mock to track removeItem calls
        native_playlist.removeItem = MagicMock()

        # Register with API mock
        plex_api.test_playlists = [native_playlist]
        plex_api.test_tracks = [native_track]

        plex_player._remove_track_from_playlist(playlist, track)

        # Verify removeItem was called with correct track
        native_playlist.removeItem.assert_called_once_with(native_track)

    def test_add_track_to_nonexistent_playlist_logs_warning_no_operation(self, plex_player, plex_api, track_factory):
        """Test adding track to nonexistent playlist logs warning and performs no operation."""
        playlist = Playlist(ID=999999, name="Missing")
        track = track_factory(ID=1)

        # Ensure no playlists exist in API mock
        plex_api.test_playlists = []

        # Mock logger to verify warning
        plex_player.logger.warning = MagicMock()

        plex_player._add_track_to_playlist(playlist, track)

        # Verify warning was logged (should be the final call)
        expected_calls = [call("Failed to retrieve playlist by ID '999999': search error"), call("Native playlist not found for Missing")]
        plex_player.logger.warning.assert_has_calls(expected_calls)

    def test_remove_track_from_nonexistent_playlist_logs_warning_no_operation(self, plex_player, plex_api, track_factory):
        """Test removing track from nonexistent playlist logs warning and performs no operation."""
        playlist = Playlist(ID=999999, name="Missing")
        track = track_factory(ID=1)

        # Ensure no playlists exist in API mock
        plex_api.test_playlists = []

        # Mock logger to verify warning
        plex_player.logger.warning = MagicMock()

        plex_player._remove_track_from_playlist(playlist, track)

        # Verify warning was logged (should be the final call)
        expected_calls = [call("Failed to retrieve playlist by ID '999999': search error"), call("Native playlist not found for Missing")]
        plex_player.logger.warning.assert_has_calls(expected_calls)

    def test_add_track_when_track_not_found_logs_warning_no_operation(self, plex_player, plex_api, track_factory, native_playlist_factory):
        """Test adding nonexistent track to playlist logs warning and performs no operation."""
        playlist = Playlist(ID=1004, name="Test")
        track = track_factory(ID=999999)  # Use non-existent integer ID

        # Create playlist but no tracks
        native_playlist = native_playlist_factory(playlist)
        native_playlist.addItems = MagicMock()
        plex_api.test_playlists = [native_playlist]
        plex_api.test_tracks = []

        # Mock logger to verify warning
        plex_player.logger.warning = MagicMock()

        plex_player._add_track_to_playlist(playlist, track)

        # Verify warning was logged and addItems not called
        plex_player.logger.warning.assert_called_once_with("No match found for track ID: 999999")
        native_playlist.addItems.assert_not_called()

    def test_remove_track_when_track_not_found_logs_warning_no_operation(self, plex_player, plex_api, track_factory, native_playlist_factory):
        """Test removing nonexistent track from playlist logs warning and performs no operation."""
        playlist = Playlist(ID=1005, name="Test")
        track = track_factory(ID=999999)

        # Create playlist but no tracks
        native_playlist = native_playlist_factory(playlist)
        native_playlist.removeItem = MagicMock()
        plex_api.test_playlists = [native_playlist]
        plex_api.test_tracks = []

        # Mock logger to verify warning
        plex_player.logger.warning = MagicMock()

        plex_player._remove_track_from_playlist(playlist, track)

        # Verify warning was logged and removeItem not called
        plex_player.logger.warning.assert_called_once_with("No match found for track ID: 999999")
        native_playlist.removeItem.assert_not_called()


class TestPlexPlaylistCreation:
    def test_create_playlist_all_tracks_success(self, plex_player, track_factory, native_playlist_factory, native_track_factory, plex_api):
        """Test creating a playlist when all tracks are found: API called with correct args, playlist state unchanged."""
        track1 = track_factory(ID=1)
        track2 = track_factory(ID=2)
        native_track1 = native_track_factory(track1)
        native_track2 = native_track_factory(track2)
        plex_api.test_tracks = [native_track1, native_track2]
        playlist_obj = native_playlist_factory(Playlist(ID=1000, name="foo"))
        plex_api.test_playlists.append(playlist_obj)
        plex_player.plex_api_connection.createPlaylist.reset_mock()

        plex_player._create_playlist("foo", [track1, track2])
        plex_player.plex_api_connection.createPlaylist.assert_called_once_with(title="foo", items=[native_track1, native_track2])

    def test_create_playlist_some_tracks_success(self, plex_player, track_factory, native_playlist_factory, native_track_factory, plex_api):
        """Test creating a playlist when some tracks are missing: API called with found tracks only."""
        track1 = track_factory(ID=1)
        missing_track = track_factory(ID=999999)
        native_track1 = native_track_factory(track1)
        plex_api.test_tracks = [native_track1]
        playlist_obj = native_playlist_factory(Playlist(ID=1001, name="foo"))
        plex_api.test_playlists.append(playlist_obj)
        plex_player.plex_api_connection.createPlaylist.reset_mock()

        plex_player._create_playlist("foo", [track1, missing_track])
        plex_player.plex_api_connection.createPlaylist.assert_called_once_with(title="foo", items=[native_track1])

    def test_create_playlist_no_tracks_noop(self, plex_player, plex_api):
        """Test creating a playlist with no tracks: API not called, playlist state unchanged."""
        plex_api.test_tracks = []
        plex_player.plex_api_connection.createPlaylist.reset_mock()

        result = plex_player._create_playlist("foo", [])
        plex_player.plex_api_connection.createPlaylist.assert_not_called()
        assert result is None

    def test_create_playlist_tracks_not_found_noop(self, plex_player, track_factory, plex_api):
        """Test creating a playlist when no tracks are found: API not called, playlist state unchanged."""
        track1 = track_factory(ID=1)
        plex_api.test_tracks = []
        plex_player.plex_api_connection.createPlaylist.reset_mock()

        result = plex_player._create_playlist("foo", [track1])
        plex_player.plex_api_connection.createPlaylist.assert_not_called()
        assert result is None

    def test_create_playlist_track_search_raises(self, plex_player, track_factory, plex_api):
        """Test exception during track search: error is logged, API not called, playlist state unchanged."""
        track = track_factory(ID=1)
        plex_api.test_tracks = [track]
        plex_player.plex_api_connection.createPlaylist.reset_mock()

        plex_player._create_playlist("foo", [track])
        plex_player.plex_api_connection.createPlaylist.assert_not_called()
        plex_player.logger.error.assert_any_call("Failed to search for track 1: search error")


class TestPlexTrackSearch:
    @pytest.mark.parametrize(
        "key,value,return_native,expect_error,expect_empty",
        [
            ("title", "Title", False, False, False),
            ("id", 1, False, False, False),
            ("rating", True, False, False, False),
            ("badkey", None, False, True, False),
            ("id", 999999, False, True, True),
        ],
        ids=[
            "search_by_title_with_existing_track_returns_track",
            "search_by_id_with_existing_track_returns_track",
            "search_by_rating_returns_tracks_with_ratings",
            "search_with_invalid_key_raises_keyerror",
            "search_by_nonexistent_id_raises_runtime_error",
        ],
    )
    def test_search_tracks_with_various_keys_and_values(self, plex_player, plex_api, track_factory, native_track_factory, key, value, return_native, expect_error, expect_empty):
        """Test track search handles different search parameters correctly."""
        if value == 1:
            test_track = track_factory(ID=1)
            native_track = native_track_factory(test_track)
            plex_api.test_tracks = [native_track]
        elif value == "Title":
            test_track = track_factory(title="Title")
            native_track = native_track_factory(test_track)
            plex_api.test_tracks = [native_track]
        elif value is True:  # rating search
            test_track = track_factory(ID=1, rating=0.5)
            native_track = native_track_factory(test_track)
            plex_api.test_tracks = [native_track]

        if expect_error:
            with pytest.raises((ValueError, KeyError, RuntimeError)):
                plex_player._search_tracks(key, value, return_native)
        else:
            result = plex_player._search_tracks(key, value, return_native)
            if expect_empty:
                assert result == []
            else:
                assert isinstance(result, list)

    def test_search_tracks_with_native_return_true_returns_native_objects(self, plex_player, plex_api, track_factory, native_track_factory):
        """Test searching tracks with return_native=True returns Plex API objects."""
        test_track = track_factory(ID=1, title="Title")
        native_track = native_track_factory(test_track)
        plex_api.test_tracks = [native_track]

        result = plex_player._search_tracks("id", 1, True)

        assert isinstance(result, list)
        assert len(result) == 1
        assert hasattr(result[0], "key")  # Native Plex objects have key attribute


class TestPlexTrackUpdate:
    def test_read_track_metadata_all_fields(self, plex_player, track_factory, native_track_factory):
        """Test reading all fields from a track's metadata."""
        track = track_factory(ID=1, artist="Artist")
        native_track = native_track_factory(track)
        result = plex_player._read_track_metadata(native_track)
        assert result.artist == "Artist"

    def test_update_rating_success(self, plex_player, plex_api, track_factory, native_track_factory):
        """Test updating the rating of a track succeeds and calls edit."""
        track = track_factory(ID=1)
        native_track = native_track_factory(track)
        plex_api.test_tracks = [native_track]
        called = {"edit": False}

        def edit_side_effect(*args, **kwargs):
            called["edit"] = True

        native_track.edit = edit_side_effect
        plex_player.music_library.fetchItem.side_effect = lambda v: native_track if (v == 1 or v == "1") else None
        plex_player._update_rating(native_track, Rating(5, scale=RatingScale.ZERO_TO_TEN))
        assert called["edit"]

    def test_update_rating_edit_raises(self, plex_player, plex_api, track_factory, native_track_factory):
        """Test updating the rating of a track raises and does not call edit."""
        track = track_factory(ID=1)
        native_track = native_track_factory(track)
        plex_api.test_tracks = [native_track]
        called = {"edit": False}

        def edit_side_effect(*args, **kwargs):
            called["edit"] = True
            raise RuntimeError("fail")

        native_track.edit = edit_side_effect
        plex_player.music_library.fetchItem.side_effect = lambda v: native_track if (v == 1 or v == "1") else None
        import pytest

        with pytest.raises(RuntimeError):
            plex_player._update_rating(native_track, Rating(5, scale=RatingScale.ZERO_TO_TEN))
        assert called["edit"]

    def test_update_rating_track_not_found(self, plex_player, plex_api, track_factory, native_track_factory):
        """Test updating the rating of a track that is not found does not call edit and raises."""
        track = track_factory(ID=1)
        native_track = native_track_factory(track)
        plex_api.test_tracks = []
        called = {"edit": False}

        def edit_side_effect(*args, **kwargs):
            called["edit"] = True

        native_track.edit = edit_side_effect
        plex_player.music_library.fetchItem.side_effect = lambda v: None
        import pytest

        with pytest.raises((RuntimeError, ValueError, KeyError)):
            plex_player._update_rating(native_track, Rating(5, scale=RatingScale.ZERO_TO_TEN))
        assert not called["edit"]
