from pathlib import Path

from filesystem_provider import FileSystemProvider
from ratings import Rating
from sync_items import AudioTag

"""
# TODO: Add test for duplicate playlist title disambiguation
# TODO: Add test for playlist missing header (non-EXTM3U format)
# TODO: Add test for relative path calculation in add_track_to_playlistFunctional tests for playlist creation, metadata resolution, and track add/remove behavior."""
"""
# TODO: Add test for skipped write when rating remains unresolved
# TODO: Add test for tag write failure (simulate update_track_metadata error)Functional tests for FileSystemProvider.finalize_scan(): rating resolution and tag writing."""


class TestFileSystemProvider:
    def test_scan_media_files_discovers_audio_and_playlists(self):
        assert 0 == 1

    def test_read_track_metadata_handles_id3_and_vorbis(self):
        pass

    def test_update_track_metadata_writes_correctly(self):
        pass

    def test_finalize_scan_resolves_deferred_tracks(self):
        pass


def test_create_playlist_adds_tracks(monkeypatch, track_factory):
    fsp = FileSystemProvider()
    monkeypatch.setattr("filesystem_provider.Path.resolve", lambda self: self)
    monkeypatch.setattr(fsp, "path", Path("/music"))

    monkeypatch.setattr(fsp, "add_track_to_playlist", lambda pid, t, is_extm3u=False: True)

    playlist = fsp.create_playlist("MyPlaylist", is_extm3u=True)
    assert playlist is not None
    assert playlist.name == "MyPlaylist"


def test_read_playlist_metadata_extracts_title(tmp_path):
    playlist_file = tmp_path / "MyList.m3u"
    playlist_file.write_text("""#EXTM3U
#PLAYLIST:Chill Vibes
/music/song1.mp3
""")

    fsp = FileSystemProvider()
    fsp.playlist_path = tmp_path
    playlist = fsp.read_playlist_metadata(playlist_file)

    assert playlist.name == "Chill Vibes"
    assert playlist.ID.lower() == str(playlist_file).lower()


def test_get_tracks_from_playlist_filters_scope(monkeypatch, tmp_path):
    playlist_path = tmp_path / "pl.m3u"
    playlist_path.write_text("""out_of_scope.mp3
valid/track1.mp3
""")

    fsp = FileSystemProvider()
    fsp.path = tmp_path / "valid"
    fsp.playlist_path = tmp_path
    playlist_path = playlist_path.resolve()

    monkeypatch.setattr("filesystem_provider.Path.resolve", lambda self: self)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_relative_to", lambda self, other: str(self).startswith(str(other)))
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "suffix", property(lambda self: ".mp3"))

    tracks = fsp.get_tracks_from_playlist(playlist_path)
    assert all(fsp.path in t.parents or t == fsp.path for t in tracks)


def test_finalize_resolves_and_writes(monkeypatch):
    fsp = FileSystemProvider()

    dummy_track = AudioTag(ID="song.mp3", title="Title", artist="Artist", album="Album", track=1, file_path="song.mp3")
    handler = fsp.id3_handler

    monkeypatch.setattr("filesystem_provider.Path.resolve", lambda self: Path(self))

    monkeypatch.setattr(handler, "resolve_rating", lambda raw, track: Rating(4.5))
    monkeypatch.setattr(fsp, "update_track_metadata", lambda path, metadata=None, rating=None: rating)

    fsp.deferred_tracks = [
        {
            "track": dummy_track,
            "handler": handler,
            "raw": {"MEDIAMONKEY": "4.5"},
        }
    ]

    results = fsp.finalize_scan()

    assert len(results) == 1
    assert results[0].rating is not None
    assert results[0].rating.to_float() > 0


def test_finalize_infers_strategies(monkeypatch):
    fsp = FileSystemProvider()
    monkeypatch.setattr(fsp.id3_handler, "finalize_rating_strategy", lambda x: setattr(fsp.id3_handler, "called", True))
    monkeypatch.setattr(fsp.vorbis_handler, "finalize_rating_strategy", lambda x: setattr(fsp.vorbis_handler, "called", True))

    fsp.deferred_tracks = [
        {"track": AudioTag(ID="1", title="a", artist="b", album="c", track=1), "handler": fsp.id3_handler, "raw": {}},
        {"track": AudioTag(ID="2", title="x", artist="y", album="z", track=2), "handler": fsp.vorbis_handler, "raw": {}},
    ]

    _ = fsp.finalize_scan()

    assert getattr(fsp.id3_handler, "called", False)
    assert getattr(fsp.vorbis_handler, "called", False) is True


"""
# TODO: Add test for handler not found (unsupported file format)
# TODO: Add test for cache hit returning earlyFunctional tests for FileSystemProvider.read_track_metadata() and format dispatching."""


def test_reads_id3_metadata(monkeypatch, mp3_file_factory):
    path = Path("test.mp3")

    fsp = FileSystemProvider()

    monkeypatch.setattr("mutagen.File", lambda *a, **kw: mp3_file_factory())
    monkeypatch.setattr("filesystem_provider.Path.resolve", lambda self: self)

    tag = fsp.read_track_metadata(path)

    assert tag is not None
    assert tag.title == "Title"
    assert tag.rating is not None


def test_reads_vorbis_metadata(monkeypatch, vorbis_file_factory):
    path = Path("track.flac")

    fsp = FileSystemProvider()

    monkeypatch.setattr("mutagen.File", lambda *a, **kw: vorbis_file_factory())
    monkeypatch.setattr("filesystem_provider.Path.resolve", lambda self: self)

    tag = fsp.read_track_metadata(path)

    assert tag is not None
    assert tag.artist == "Artist"
    assert tag.rating is not None


def test_deferred_when_unresolved(monkeypatch, vorbis_file_factory):
    path = Path("conflicted.flac")
    f = vorbis_file_factory()
    f.tags["FMPS_RATING"] = ["2.5"]
    f.tags["RATING"] = ["4.5"]

    fsp = FileSystemProvider()

    monkeypatch.setattr("mutagen.File", lambda *a, **kw: f)
    monkeypatch.setattr("filesystem_provider.Path.resolve", lambda self: self)

    tag = fsp.read_track_metadata(path)

    assert tag is not None
    assert any(d["track"].ID == tag.ID for d in fsp.deferred_tracks)
