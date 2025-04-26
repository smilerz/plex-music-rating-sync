import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mutagen import FileType
from mutagen.mp3 import MP3

from filesystem_provider import DefaultPlayerTags, VorbisField
from manager import get_manager
from MediaPlayer import MediaPlayer
from ratings import Rating
from sync_items import AudioTag, Playlist
from sync_pair import PlaylistPair, TrackPair
from tests.helpers import add_or_update_id3frame


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
def vorbis_file_factory():
    def _factory(fmps_rating: str = "5", standard_rating: str = "5"):
        from unittest.mock import MagicMock

        vorbis = MagicMock(spec=FileType)
        vorbis.filename = "mock.flac"
        vorbis.info = MagicMock(length=222)
        vorbis.tags = {
            VorbisField.ARTIST: ["Default Artist"],
            VorbisField.ALBUM: ["Default Album"],
            VorbisField.TITLE: ["Default Title"],
            VorbisField.TRACKNUMBER: ["1/10"],
        }

        if fmps_rating is not None:
            vorbis.tags["FMPS_RATING"] = [fmps_rating]

        if standard_rating is not None:
            vorbis.tags["RATING"] = [standard_rating]

        # Patch .get() to act like a dict proxy
        vorbis.get = lambda key, default=None: vorbis.tags.get(key, default)

        return vorbis

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
