"""
# TODO: Add test for handler not found (unsupported file format)
# TODO: Add test for cache hit returning earlyFunctional tests for FileSystemProvider.read_track_metadata() and format dispatching."""

from pathlib import Path
from manager import get_manager
from filesystem_provider import FileSystemProvider


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
