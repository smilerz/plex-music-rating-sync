"""
# TODO: Add test for skipped write when rating remains unresolved
# TODO: Add test for tag write failure (simulate update_track_metadata error)Functional tests for FileSystemProvider.finalize_scan(): rating resolution and tag writing."""

from pathlib import Path
from ratings import Rating
from filesystem_provider import FileSystemProvider
from sync_items import AudioTag


def test_finalize_resolves_and_writes(monkeypatch):
    fsp = FileSystemProvider()

    dummy_track = AudioTag(ID="song.mp3", title="Title", artist="Artist", album="Album", track=1, file_path="song.mp3")
    handler = fsp.id3_handler

    monkeypatch.setattr("filesystem_provider.Path.resolve", lambda self: Path(self))

    monkeypatch.setattr(handler, "resolve_rating", lambda raw, track: Rating(4.5))
    monkeypatch.setattr(fsp, "update_track_metadata", lambda path, metadata=None, rating=None: rating)

    fsp.deferred_tracks = [{
        "track": dummy_track,
        "handler": handler,
        "raw": {"MEDIAMONKEY": "4.5"},
    }]

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
