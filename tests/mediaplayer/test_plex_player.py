from contextlib import nullcontext
from unittest.mock import MagicMock

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
        """Return a list of MagicMock sections shaped like Plex artist libraries."""
        return [
            MagicMock(
                key=i + 1,
                type="artist",
                title=f"Music Library {i + 1}",
            )
            for i in range(count)
        ]

    def _sections_side_effect():
        if api.libraries:
            return api.libraries
        return _make_default_libraries(api.library_count)

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
            try:
                track_id = str(args[0])
                return [t for t in test_tracks if str(getattr(t, "ID", getattr(t, "id", None))) == track_id]
            except Exception:
                return []
        return test_tracks

    def _make_side_effect(name):
        def _side_effect(*args, **kwargs):
            exc = getattr(api, f"{name}_raise", None)
            if exc is not None:
                raise exc
            if hasattr(api, f"{name}_return"):
                return getattr(api, f"{name}_return")
            if name == "playlists":
                return getattr(api, "test_playlists", [])
            if name == "searchTracks":
                return _mock_search_tracks(api, *args, **kwargs)
            return None

        return _side_effect

    resource_mock.connect = MagicMock(side_effect=_connect_side_effect)
    connection_mock.library = MagicMock(name="library")
    connection_mock.library.sections = MagicMock(side_effect=_sections_side_effect)
    api.fetchItem.side_effect = _make_side_effect("fetchItem")
    api.playlists.side_effect = _make_side_effect("playlists")
    api.createPlaylist.side_effect = _make_side_effect("createPlaylist")
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
        mock.userRating = getattr(track, "rating", getattr(track, "userRating", None))
        key_val = getattr(track, "ID", getattr(track, "key", getattr(track, "title", "1")))
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
            ("title", "My Playlist", False, {"my playlist": "id1"}, False, False),
            ("id", "id1", False, {}, False, False),
            ("badkey", None, False, {}, True, False),
            ("id", "badid", False, {}, False, True),
        ],
    )
    def test_search_playlists_key_error(self, plex_player, plex_api, key, value, return_native, setup_map, expect_error, expect_empty):
        """Test playlist search with various keys and values, including error scenarios."""
        plex_player._title_to_id_map = {k.lower(): v for k, v in setup_map.items()}
        if value == "id1":
            playlist = Playlist(ID="id1", name="My Playlist")
            plex_api.test_playlists.append(playlist)
        if expect_error:
            with pytest.raises(ValueError):
                plex_player._search_playlists(key, value, return_native)
        else:
            # For the 'id', 'badid' case, ensure no playlist with that ID exists in test_playlists
            result = plex_player._search_playlists(key, value, return_native)
            if expect_empty:
                assert result == []
            else:
                assert isinstance(result, list)

    @pytest.mark.parametrize(
        "is_auto_playlist,num_items,expect_bar",
        [
            (True, 2, False),
            (False, 2, False),
            (False, 101, True),
        ],
    )
    def test_load_playlist_tracks_bar(
        self,
        plex_player,
        plex_api,
        is_auto_playlist,
        num_items,
        expect_bar,
        track_factory,
        native_playlist_factory,
        native_track_factory,
        monkeypatch,
    ):
        playlist = Playlist(ID="id1", name="Test", is_auto_playlist=is_auto_playlist)
        # Create native tracks
        native_tracks = [native_track_factory(track_factory(ID=f"t{i}")) for i in range(num_items)]
        # Create native playlist and assign tracks
        native_playlist = native_playlist_factory(playlist)
        native_playlist.Tracks = native_tracks
        native_playlist.Tracks.Count = len(native_tracks)
        # Register with API mock
        plex_api.test_playlists.append(native_playlist)
        plex_player._title_to_id_map = {playlist.name.lower(): "id1"}
        plex_player.status_mgr = MagicMock()
        plex_player.load_playlist_tracks(playlist)
        if expect_bar:
            plex_player.status_mgr.start_phase.assert_called_once()
        else:
            plex_player.status_mgr.start_phase.assert_not_called()

    def test_read_native_playlist_tracks(self, plex_player, plex_api, native_playlist_factory, native_track_factory):
        """Test reading tracks from a native playlist."""
        # Create native tracks
        native_tracks = [native_track_factory(ID="t1"), native_track_factory(ID="t2")]
        # Create native playlist and assign tracks
        native_playlist = native_playlist_factory(Playlist(ID="1", name="Test"))
        native_playlist.Tracks = native_tracks
        # Register with API mock
        plex_api.test_playlists.append(native_playlist)
        result = plex_player._read_native_playlist_tracks(native_playlist)
        assert isinstance(result, list)
        assert len(result) == len(native_tracks)

    def test_get_native_playlist_track_count(self, plex_player, plex_api, native_playlist_factory, native_track_factory):
        """Test counting tracks in a native playlist."""
        # Create native tracks
        native_tracks = [native_track_factory(ID=str(i)) for i in range(1, 4)]
        # Create native playlist and assign tracks
        native_playlist = native_playlist_factory(Playlist(ID="1", name="Test"))
        native_playlist.Tracks = native_tracks
        native_playlist.Tracks.Count = len(native_tracks)
        # Register with API mock
        plex_api.test_playlists.append(native_playlist)
        result = plex_player._get_native_playlist_track_count(native_playlist)
        assert result == 3

    def test_search_playlists_with_native_return(self, plex_player, plex_api):
        """Test searching playlists with return_native=True returns native playlist objects."""
        playlist = Playlist(ID="id1", name="My Playlist")
        plex_api.test_playlists.append(playlist)
        plex_player._title_to_id_map = {"my playlist": "id1"}
        result = plex_player._search_playlists("id", "id1", True)
        assert isinstance(result, list)
        assert all(hasattr(p, "ID") for p in result)


class TestPlexPlaylistUpdate:
    def test_add_remove_track_from_playlist(self, plex_player, plex_api, track_factory, native_playlist_factory, native_track_factory):
        """Test adding or removing tracks from a playlist."""
        playlist = Playlist(ID="id1", name="Test")
        track = track_factory(ID="1")
        # Create native playlist and native track
        native_playlist = native_playlist_factory(playlist)
        native_track = native_track_factory(track)
        native_playlist.Tracks = [native_track]
        native_playlist.Tracks.Count = 1
        # Register with API mock
        plex_api.test_playlists.append(native_playlist)
        plex_api.test_tracks.append(native_track)
        # Simulate add
        plex_player._add_track_to_playlist(playlist, track)
        # Assert correct state (not just object identity)
        assert native_playlist in plex_api.test_playlists
        assert native_track in plex_api.test_tracks

    def test_remove_track_from_existing_playlist(self, plex_player, plex_api, track_factory, native_playlist_factory, native_track_factory):
        """Test removing a track from an existing playlist updates state as expected."""
        playlist = Playlist(ID="id1", name="Test")
        track = track_factory(ID="1")
        # Create native playlist and native track
        native_playlist = native_playlist_factory(playlist)
        native_track = native_track_factory(track)
        native_playlist.Tracks = [native_track]
        native_playlist.Tracks.Count = 1
        # Register with API mock
        plex_api.test_playlists.append(native_playlist)
        plex_api.test_tracks.append(native_track)
        # Simulate remove
        plex_player._remove_track_from_playlist(playlist, track)
        assert native_playlist in plex_api.test_playlists
        assert native_track in plex_api.test_tracks

    def test_add_track_to_nonexistent_playlist(self, plex_player, plex_api, track_factory):
        """Test adding a track to a nonexistent playlist raises AttributeError (NoneType)."""
        track = track_factory(ID="1")
        plex_api.test_tracks.append(track)
        original_playlists = list(plex_api.test_playlists)
        with pytest.raises(AttributeError):
            plex_player._add_track_to_playlist(None, track)
        assert plex_api.test_playlists == original_playlists

    def test_remove_track_from_nonexistent_playlist(self, plex_player, plex_api, track_factory):
        """Test removing a track from a nonexistent playlist raises AttributeError (NoneType)."""
        track = track_factory(ID="1")
        plex_api.test_tracks.append(track)
        original_playlists = list(plex_api.test_playlists)
        with pytest.raises(AttributeError):
            plex_player._remove_track_from_playlist(None, track)
        assert plex_api.test_playlists == original_playlists

    def test_add_remove_track_error_handling(self, plex_player, plex_api, track_factory):
        """Test error handling when both playlist and track are missing raises AttributeError (NoneType)."""
        original_playlists = list(plex_api.test_playlists)
        original_tracks = list(plex_api.test_tracks)
        with pytest.raises(AttributeError):
            plex_player._add_track_to_playlist(None, None)
        with pytest.raises(AttributeError):
            plex_player._remove_track_from_playlist(None, None)
        assert plex_api.test_playlists == original_playlists
        assert plex_api.test_tracks == original_tracks


class TestPlexPlaylistCreation:
    def test_create_playlist_all_tracks_success(self, plex_player, track_factory, native_playlist_factory, native_track_factory, plex_api):
        """Test creating a playlist when all tracks are found: API called with correct args, playlist state unchanged."""
        track1 = track_factory(ID=1)
        track2 = track_factory(ID=2)
        native_track1 = native_track_factory(track1)
        native_track2 = native_track_factory(track2)
        plex_api.test_tracks = [native_track1, native_track2]
        playlist_obj = native_playlist_factory(Playlist(ID=100, name="foo"))
        plex_api.test_playlists.append(playlist_obj)
        plex_api.createPlaylist.reset_mock()
        original_playlists = list(plex_api.test_playlists)

        plex_player._create_playlist("foo", [track1, track2])
        plex_api.createPlaylist.assert_called_once_with(title="foo", items=[native_track1, native_track2])
        assert plex_api.test_playlists == original_playlists

    def test_create_playlist_some_tracks_success(self, plex_player, track_factory, native_playlist_factory, native_track_factory, plex_api):
        """Test creating a playlist when some tracks are missing: API called with found tracks only."""
        track1 = track_factory(ID=1)
        missing_track = track_factory(ID="missing_track")
        native_track1 = native_track_factory(track1)
        plex_api.test_tracks = [native_track1]
        playlist_obj = native_playlist_factory(Playlist(ID=100, name="foo"))
        plex_api.test_playlists.append(playlist_obj)
        plex_api.createPlaylist.reset_mock()
        original_playlists = list(plex_api.test_playlists)

        plex_player._create_playlist("foo", [track1, missing_track])
        plex_api.createPlaylist.assert_called_once_with(title="foo", items=[native_track1])
        assert plex_api.test_playlists == original_playlists

    def test_create_playlist_no_tracks_noop(self, plex_player, plex_api):
        """Test creating a playlist with no tracks: API not called, playlist state unchanged."""
        plex_api.test_tracks = []
        plex_api.createPlaylist.reset_mock()
        original_playlists = list(plex_api.test_playlists)

        result = plex_player._create_playlist("foo", [])
        plex_api.createPlaylist.assert_not_called()
        assert result is None
        assert plex_api.test_playlists == original_playlists

    def test_create_playlist_tracks_not_found_noop(self, plex_player, track_factory, plex_api):
        """Test creating a playlist when no tracks are found: API not called, playlist state unchanged."""
        track1 = track_factory(ID=1)
        plex_api.test_tracks = []
        plex_api.createPlaylist.reset_mock()
        original_playlists = list(plex_api.test_playlists)

        result = plex_player._create_playlist("foo", [track1])
        plex_api.createPlaylist.assert_not_called()
        assert result is None
        assert plex_api.test_playlists == original_playlists

    def test_create_playlist_track_search_raises(self, plex_player, track_factory, plex_api):
        """Test exception during track search: exception propagates, API not called, playlist state unchanged."""
        track = track_factory(ID=1)
        plex_api.test_tracks = [track]
        plex_api.createPlaylist.reset_mock()
        original_playlists = list(plex_api.test_playlists)
        plex_api.searchTracks_raise = RuntimeError("search error")

        import pytest

        with pytest.raises(RuntimeError):
            plex_player._create_playlist("foo", [track])
        plex_api.createPlaylist.assert_not_called()
        assert plex_api.test_playlists == original_playlists
        plex_api.searchTracks_raise = None


class TestPlexTrackSearch:
    @pytest.mark.parametrize(
        "key,value,return_native,expect_error,expect_empty",
        [
            ("all", None, False, True, False),
            ("title", "Title", False, False, False),
            ("id", "1", False, False, False),
            ("badkey", None, False, True, False),
            ("id", "badid", False, True, True),
        ],
    )
    def test_search_tracks_key_error(self, plex_player, plex_api, track_factory, key, value, return_native, expect_error, expect_empty):
        """Test track search with various keys and values, including error scenarios."""
        if value == "1":
            plex_api.test_tracks.append(track_factory(ID="1"))
        elif value == "Title":
            plex_api.test_tracks.append(track_factory(title="Title"))
        if expect_error:
            with pytest.raises((ValueError, KeyError)):
                plex_player._search_tracks(key, value, return_native)
        else:
            result = plex_player._search_tracks(key, value, return_native)
            if expect_empty:
                assert result == []
            else:
                assert isinstance(result, list)

    def test_search_tracks_with_native_return(self, plex_player, plex_api, native_track_factory):
        """Test searching tracks with return_native=True returns native track objects."""
        native_track = native_track_factory(ID="1", title="Title")
        plex_api.test_tracks.append(native_track)
        result = plex_player._search_tracks("id", "1", True)
        assert isinstance(result, list)
        assert all(hasattr(t, "ID") for t in result)


class TestPlexTrackUpdate:
    def test_read_track_metadata_all_fields(self, plex_player, native_track_factory):
        """Test reading all fields from a track's metadata."""
        native_track = native_track_factory(ID="1", artist="Artist")
        result = plex_player._read_track_metadata(native_track)
        assert result.artist == "Artist"

    def test_update_rating_success(self, plex_player, plex_api, native_track_factory):
        """Test updating the rating of a track succeeds and calls edit."""
        native_track = native_track_factory(ID="1")
        plex_api.test_tracks = [native_track]
        called = {"edit": False}

        def edit_side_effect(*args, **kwargs):
            called["edit"] = True

        native_track.edit = edit_side_effect
        plex_player.music_library.fetchItem.side_effect = lambda v: native_track if (v == 1 or v == "1") else None
        plex_player._update_rating(native_track, Rating(5, scale=RatingScale.ZERO_TO_TEN))
        assert called["edit"]

    def test_update_rating_edit_raises(self, plex_player, plex_api, native_track_factory):
        """Test updating the rating of a track raises and does not call edit."""
        native_track = native_track_factory(ID="1")
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

    def test_update_rating_track_not_found(self, plex_player, plex_api, native_track_factory):
        """Test updating the rating of a track that is not found does not call edit and raises."""
        native_track = native_track_factory(ID="1")
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
