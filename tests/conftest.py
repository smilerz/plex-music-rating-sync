import sys
from unittest.mock import MagicMock

import pytest
from mutagen import FileType
from mutagen.id3 import POPM, TXXX, ID3FileType

from manager import get_manager
from MediaPlayer import MediaPlayer
from ratings import Rating
from sync_items import AudioTag, Playlist
from sync_pair import PlaylistPair, TrackPair


@pytest.fixture
def track_factory():
    def _factory(ID="1", title="Title", artist="Artist", album="Album", track=1, rating=Rating(1.0)):
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
    def _factory(txxx_rating: float | None = 5.0, popm_rating: int | None = 196, popm_email: str = "no@email"):
        audio = MagicMock(spec=ID3FileType)
        audio.filename = "mock.mp3"
        audio.info = MagicMock(length=245)
        audio.tags = {}

        if txxx_rating is not None:
            audio.tags["TXXX:RATING"] = TXXX(encoding=3, desc="RATING", text=[str(txxx_rating)])

        if popm_rating is not None:
            audio.tags[f"POPM:{popm_email}"] = POPM(email=popm_email, rating=popm_rating, count=0)

        return audio

    return _factory


@pytest.fixture
def vorbis_file_factory():
    def _factory(fmps_rating: str | None = "5", standard_rating: str | None = "5"):
        from unittest.mock import MagicMock

        vorbis = MagicMock(spec=FileType)
        vorbis.filename = "mock.flac"
        vorbis.info = MagicMock(length=222)
        vorbis.tags = {"ARTIST": ["Artist"], "ALBUM": ["Album"], "TITLE": ["Title"], "TRACKNUMBER": ["1"]}

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
