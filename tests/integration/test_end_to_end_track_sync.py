"""
# TODO: Add test for match score boundary transitions (e.g. 79/80 edge)
# TODO: Add test for cache interaction (if applicable in lifecycle)
# TODO: Add test where update_rating raises an exceptionIntegration test for full track sync lifecycle: scan, match, resolve, sync, write."""

import pytest
from manager import get_manager
from filesystem_provider import FileSystemProvider
from MediaPlayer import FileSystem, Plex
from sync_pair import TrackPair, SyncState
from ratings import Rating
from sync_items import AudioTag


def test_track_sync_lifecycle(monkeypatch, track_factory):
    mgr = get_manager()
    mgr.config._initialized = True  # Avoid re-init from ConfigManager

    source = FileSystem()
    dest = Plex()

    # Prepare matching tracks with mismatched ratings
    track1 = track_factory(ID="track1", rating=Rating(4.5))
    track2 = track_factory(ID="track1", rating=Rating(3.0))

    # Patch player methods
    source.search_tracks = lambda k, v, **_: [track1] if v == "track1" else []
    dest.search_tracks = lambda k, v, **_: [track2] if v == "track1" else []

    source.update_rating = lambda t, r: setattr(t, "rating", r)

    # Pair and sync
    pair = TrackPair(source, dest, track1)
    pair.destination_track = track2
    pair.match()
    assert pair.sync_state == SyncState.CONFLICTING

    pair.sync()
    assert pair.destination_track.rating.to_float() == track1.rating.to_float()


def test_sync_skips_if_identical(monkeypatch, track_factory):
    fs = FileSystem()
    plex = Plex()

    track1 = track_factory(ID="t99", rating=Rating(4.0))
    track2 = track_factory(ID="t99", rating=Rating(4.0))

    fs.search_tracks = lambda k, v, **_: [track1]
    plex.search_tracks = lambda k, v, **_: [track2]
    plex.update_rating = lambda t, r: setattr(t, "updated", True)

    pair = TrackPair(fs, plex, track1)
    pair.destination_track = track2
    pair.match()
    pair.sync()

    assert not hasattr(track2, "updated")
