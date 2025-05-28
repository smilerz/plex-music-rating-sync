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
