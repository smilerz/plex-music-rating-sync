"""
# TODO: Add test for sync when destination_track is missing
# TODO: Add test to confirm no update when source rating is unrated
# TODO: Add test to ensure sync sets updated rating on destinationFunctional tests for TrackPair.sync(): verifies rating propagation from source to destination."""

import pytest
from sync_pair import TrackPair, SyncState
from sync_items import AudioTag


@pytest.fixture
def dummy_players(filesystem_player, plex_player):
    return filesystem_player, plex_player


def test_sync_propagates_rating_when_needed(dummy_players):
    fs, plex = dummy_players
    src = AudioTag(ID="1", title="a", artist="x", album="z", track=1, rating=Rating(4.5))
    dst = AudioTag(ID="2", title="a", artist="x", album="z", track=1, rating=None)

    pair = TrackPair(fs, plex, src)
    pair.destination_track = dst
    pair.sync_state = SyncState.NEEDS_UPDATE

    pair.sync()

    plex.update_rating.assert_called_once_with(dst, src.rating)


def test_sync_skips_when_state_none(dummy_players):
    fs, plex = dummy_players
    src = AudioTag(ID="1", title="a", artist="x", album="z", track=1, rating=Rating(4.5))
    dst = AudioTag(ID="2", title="a", artist="x", album="z", track=1, rating=Rating(4.5))

    pair = TrackPair(fs, plex, src)
    pair.destination_track = dst
    pair.sync_state = None

    pair.sync()

    plex.update_rating.assert_not_called()


def test_sync_respects_dry_run(monkeypatch, dummy_players):
    fs, plex = dummy_players
    plex.dry_run = True
    src = AudioTag(ID="1", title="a", artist="x", album="z", track=1, rating=Rating(4.5))
    dst = AudioTag(ID="2", title="a", artist="x", album="z", track=1, rating=None)

    pair = TrackPair(fs, plex, src)
    pair.destination_track = dst
    pair.sync_state = SyncState.NEEDS_UPDATE

    pair.sync()

    plex.update_rating.assert_not_called()  # should be blocked by dry_run
