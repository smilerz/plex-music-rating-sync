import sys
from unittest.mock import MagicMock

import pytest

from manager import get_manager
from MediaPlayer import MediaPlayer
from ratings import Rating
from sync_items import AudioTag, Playlist
from sync_pair import PlaylistPair, TrackPair


@pytest.fixture
def track_factory():
    def _factory(ID="1", title="Title", artist="Artist", album="Album", track=1, rating=1.0):
        rating = Rating(rating)
        return AudioTag(ID=ID, title=title, artist=artist, album=album, track=track, rating=rating)

    return _factory


@pytest.fixture
def playlist_factory(track_factory):
    def _factory(ID="pl1", name="My Playlist", tracks=None):
        playlist = Playlist(ID=ID, name=name)
        playlist.tracks = tracks or [track_factory(ID="t1"), track_factory(ID="t2")]
        return playlist

    return _factory


@pytest.fixture
def plex_player(track_factory):
    player = MagicMock(spec=MediaPlayer)
    player.name.return_value = "Plex"
    player.album_empty.side_effect = lambda a: a == "[Unknown Album]"
    player.search_tracks.side_effect = lambda key, value, return_native=False: [track_factory(ID="plex1")] if key == "id" else []
    player.update_rating.side_effect = lambda track, rating: setattr(track, "rating", rating)
    player.search_playlists.side_effect = lambda key, value, return_native=False: []
    player.create_playlist.side_effect = lambda title: Playlist(ID="pl_new", name=title)
    player.update_playlist.side_effect = lambda pl, track, present=True: None
    return player


@pytest.fixture
def mediamonkey_player(track_factory):
    player = MagicMock(spec=MediaPlayer)
    player.name.return_value = "MediaMonkey"
    player.album_empty.side_effect = lambda a: a in ("", None)
    player.search_tracks.side_effect = lambda key, value, return_native=False: [track_factory(ID="mm1")] if key == "id" else []
    player.update_rating.side_effect = lambda track, rating: setattr(track, "rating", rating)
    return player


@pytest.fixture
def filesystem_player(track_factory):
    player = MagicMock(spec=MediaPlayer)
    player.name.return_value = "FileSystem"
    player.album_empty.side_effect = lambda a: a.strip() == ""
    player.search_tracks.side_effect = lambda key, value, return_native=False: [track_factory(ID="fs1")] if key == "id" else []
    player.update_rating.side_effect = lambda track, rating: setattr(track, "rating", rating)
    return player


@pytest.fixture
def track_pair_factory(filesystem_player, plex_player):
    def _factory(source_track):
        pair = TrackPair(filesystem_player, plex_player, source_track)
        return pair

    return _factory


@pytest.fixture
def playlist_pair_factory(filesystem_player, plex_player):
    def _factory(source_playlist):
        return PlaylistPair(filesystem_player, plex_player, source_playlist)

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

    # Avoid re-initialization
    mgr = get_manager()
    if not getattr(mgr, "_initialized", False):
        mgr.initialize()
