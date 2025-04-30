"""
# TODO: Add test for filtering when no tracks match any criteria
# TODO: Add test for toggling all filter flags dynamically
# TODO: Add test for unknown quality threshold valueIntegration test for CLI sync filters: quality threshold, unrated toggle, reverse mode."""

import pytest
from manager import get_manager
from sync_items import AudioTag
from ratings import Rating
from MediaPlayer import FileSystem, Plex
from sync_pair import TrackPair


@pytest.fixture
def filtered_track_pair_factory(track_factory):
    def _factory(source_rating, dest_rating):
        src = track_factory(ID="s", rating=Rating(source_rating) if source_rating is not None else None)
        dst = track_factory(ID="s", rating=Rating(dest_rating) if dest_rating is not None else None)
        return TrackPair(FileSystem(), Plex(), src), dst
    return _factory


def test_filter_excludes_below_quality(monkeypatch, filtered_track_pair_factory):
    pair, dst = filtered_track_pair_factory(4.5, 4.0)
    pair.destination_track = dst
    pair._score = 25  # poor

    monkeypatch.setattr("manager.config_manager.ConfigManager.quality", 80)

    filter_fn = lambda p: p._score >= 80
    assert not filter_fn(pair)


def test_filter_allows_unrated_toggle(monkeypatch, filtered_track_pair_factory):
    pair, dst = filtered_track_pair_factory(3.0, None)
    pair.destination_track = dst
    pair._score = 100

    monkeypatch.setattr("manager.config_manager.ConfigManager.include_unrated", True)

    include_unrated = lambda p: p.destination_track.rating is None
    assert include_unrated(pair)


def test_filter_respects_reverse(monkeypatch, filtered_track_pair_factory):
    pair, dst = filtered_track_pair_factory(None, 3.0)
    pair.destination_track = dst
    pair._score = 100

    pair = pair.reversed()

    is_reversed = lambda p: p.source_track.rating is not None and p.destination_track.rating is None
    assert is_reversed(pair)
