import sys
from unittest.mock import MagicMock

import pytest

from manager import get_manager
from MediaPlayer import MediaPlayer
from ratings import Rating
from sync_items import AudioTag, Playlist


@pytest.fixture
def mock_player() -> MediaPlayer:
    return MagicMock(spec=MediaPlayer)


@pytest.fixture
def source_track() -> AudioTag:
    return AudioTag(ID="1", title="Song A", rating=Rating(1.0))


@pytest.fixture
def dest_track_same() -> AudioTag:
    return AudioTag(ID="2", title="Song A", rating=Rating(1.0))


@pytest.fixture
def dest_track_unrated() -> AudioTag:
    return AudioTag(ID="2", title="Song A", rating=Rating.unrated())


@pytest.fixture
def dest_track_conflict() -> AudioTag:
    return AudioTag(ID="2", title="Song A", rating=Rating(0.5))


@pytest.fixture
def sample_playlist() -> Playlist:
    return Playlist(ID="1", name="Sample Playlist")


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
