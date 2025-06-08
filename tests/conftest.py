import sys
from unittest.mock import MagicMock

import pytest

from ratings import Rating, RatingScale
from sync_items import AudioTag, Playlist


# TODO: remove extraneous track and playlist factories in teh rest of hte source code
@pytest.fixture
def track_factory():
    def _factory(ID="1", title="Title", artist="Artist", album="Album", track=1, rating=1.0):
        rating = Rating(rating, scale=RatingScale.NORMALIZED)
        return AudioTag(ID=ID, title=title, artist=artist, album=album, track=track, rating=rating)

    return _factory


@pytest.fixture
def playlist_factory(track_factory):
    def _factory(ID="pl1", name="My Playlist", tracks=None, is_auto_playlist=False):
        playlist = Playlist(ID=ID, name=name)
        playlist.tracks = tracks or [track_factory(ID="t1"), track_factory(ID="t2")]
        playlist.is_auto_playlist = is_auto_playlist
        return playlist

    return _factory


@pytest.fixture(scope="session")
def patch_paths(tmp_path_factory):
    logs_path = tmp_path_factory.mktemp("logs")
    cache_path = tmp_path_factory.mktemp("cache")
    return logs_path, cache_path


@pytest.fixture(scope="session")
def config_args():
    return ["test_runner.py", "--source", "plex", "--destination", "plex", "--sync", "tracks"]


@pytest.fixture(scope="function", autouse=True)
def initialize_manager(monkeypatch, patch_paths, config_args):
    sys.argv = config_args

    logs_path, cache_path = patch_paths

    monkeypatch.setattr("manager.log_manager.LogManager.LOG_DIR", str(logs_path))
    monkeypatch.setattr("manager.cache_manager.CacheManager.MATCH_CACHE_FILE", str(cache_path / "matches.pkl"))
    monkeypatch.setattr("manager.cache_manager.CacheManager.METADATA_CACHE_FILE", str(cache_path / "metadata.pkl"))
    monkeypatch.setattr("manager.config_manager.ConfigManager.CONFIG_FILE", "./does_not_exist.ini")
    from manager import get_manager

    # Avoid re-initialization
    mgr = get_manager()
    monkeypatch.setattr(mgr, "get_stats_manager", lambda: MagicMock())
    monkeypatch.setattr(mgr, "get_status_manager", lambda: MagicMock())
    if not getattr(mgr, "_initialized", False):
        mgr.initialize()


"""
Shared Plex API fixtures for testing.

This module provides comprehensive Plex API mocking capabilities for both unit tests
and end-to-end integration tests. The fixtures simulate realistic Plex server behavior
including search, fetchItem, playlist operations, and error conditions.
"""


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
        try:
            return [_find_track_by_id(test_tracks, args[0])]
        except RuntimeError:
            # searchTracks should return empty list when track not found
            return []
    return test_tracks


def _mock_fetch_item_side_effect(api, value):
    """Mock implementation of fetchItem for the plex_api fixture."""
    # Check for fetchItem_raise exception first (fixture API integration)
    exc = getattr(api, "fetchItem_raise", None)
    if exc is not None:
        raise exc

    # ID Convention:
    # - Tracks: 1-999 (start at 1)
    # - Playlists: 1000+ (start at 1000)

    value = int(value)
    if value < 1000:  # Track ID range
        test_tracks = getattr(api, "test_tracks", [])
        return _find_track_by_id(test_tracks, value)
    else:  # Playlist ID range (1000+)
        test_playlists = getattr(api, "test_playlists", [])
        return _find_playlist_by_id(test_playlists, value)


def _make_default_libraries(api, count: int = 1):
    """Return a list of MagicMock sections shaped like Plex artist libraries, with fetchItem and createPlaylist side effects."""
    libraries = []
    for i in range(count):
        lib = MagicMock(
            key=i + 1,
            type="artist",
            title=f"Music Library {i + 1}",
        )
        lib.fetchItem.side_effect = lambda v: _mock_fetch_item_side_effect(api, v)
        lib.createPlaylist.side_effect = _make_side_effect(api, "createPlaylist")
        lib.searchTracks = MagicMock(side_effect=lambda *args, **kwargs: _mock_search_tracks(api, *args, **kwargs))
        libraries.append(lib)
    return libraries


def _make_side_effect(api, name):
    """Create side effect function for various API methods."""

    def _side_effect(*args, **kwargs):
        exc = getattr(api, f"{name}_raise", None)
        if exc is not None:
            raise exc
        if name == "playlists":
            return getattr(api, "test_playlists", [])
        if hasattr(api, f"{name}_return"):
            return getattr(api, f"{name}_return")
        return None

    return _side_effect


@pytest.fixture
def plex_api():
    """
    Comprehensive Plex API mock for testing.

    Provides realistic simulation of Plex server behavior including:
    - Search operations (searchTracks with various query types)
    - Item fetching (fetchItem with ID routing)
    - Playlist operations (playlists, createPlaylist)
    - Library management (sections, library selection)
    - Error injection capabilities for testing failure scenarios

    ID Conventions:
    - Tracks: 1-999 (start at 1)
    - Playlists: 1000+ (start at 1000)

    Usage:
        api.test_tracks = [track1, track2, ...]     # Set tracks for search/fetch
        api.test_playlists = [playlist1, ...]       # Set playlists for fetch
        api.fetchItem_raise = SomeException()       # Inject fetch errors
        api.searchTracks_raise = SomeException()    # Inject search errors
        api.library_count = 2                       # Simulate multiple libraries
    """
    api = MagicMock()
    api.libraries = None
    api.library_count = 1

    resource_mock = MagicMock(name="resource")
    connection_mock = MagicMock(name="connection")

    # Create a server resource mock that has the connect method
    server_resource_mock = MagicMock(name="server_resource")

    def _connect_side_effect(*args, **kwargs):
        exc = getattr(api, "connect_raise", None)
        if exc is not None:
            raise exc
        return connection_mock

    server_resource_mock.connect = MagicMock(side_effect=_connect_side_effect)
    resource_mock.side_effect = lambda server: server_resource_mock
    api.resource = resource_mock

    def _sections_side_effect():
        if api.libraries:
            return api.libraries
        return _make_default_libraries(api, api.library_count)

    connection_mock.library = MagicMock(name="library")
    connection_mock.library.sections = MagicMock(side_effect=_sections_side_effect)
    connection_mock.playlists = MagicMock(side_effect=_make_side_effect(api, "playlists"))

    # default: no error injection
    api.account_raise = api.connect_raise = api.fetchItem_raise = api.playlists_raise = api.createPlaylist_raise = api.searchTracks_raise = None
    return api


@pytest.fixture
def plex_player(monkeypatch, request, plex_api):
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


@pytest.fixture
def plex_playlist_factory():
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
def plex_track_factory():
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
        if hasattr(rating, "to_float"):
            mock.userRating = rating.to_float()
        else:
            mock.userRating = float(rating) if rating is not None else None

        key_val = getattr(track, "ID", getattr(track, "key", getattr(track, "title", 1)))
        mock.id = key_val
        mock.key = str(key_val)

        return mock

    return _factory
