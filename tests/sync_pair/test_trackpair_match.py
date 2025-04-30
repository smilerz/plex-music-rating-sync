"""
# TODO: Add test for trackpair.match() when one track is missing
# TODO: Add test for score edge boundaries (e.g., 79 → POOR, 80 → GOOD)
# TODO: Add test for reversed match (destination as source)Functional tests for TrackPair.match(): verifies quality classification and sync state assignment."""

import pytest
from sync_pair import TrackPair, MatchThreshold, SyncState
from sync_items import AudioTag


@pytest.mark.parametrize("score,expected_quality", [
    (100, MatchThreshold.PERFECT_MATCH),
    (95, MatchThreshold.GOOD_MATCH),
    (50, MatchThreshold.POOR_MATCH),
    (20, None),  # below threshold
])
def test_match_quality_assignment(score, expected_quality):
    src = AudioTag(ID="1", title="a", artist="x", album="z", track=1)
    dst = AudioTag(ID="2", title="a", artist="x", album="z", track=1)

    pair = TrackPair(source_player="src", destination_player="dst", source_track=src)
    pair._score = score  # force inject similarity score
    pair.match()  # triggers internal state resolution

    if expected_quality:
        assert pair.quality == expected_quality
    else:
        assert pair.quality is None


def test_sync_state_needs_update_when_dst_unrated():
    src = AudioTag(ID="1", title="a", artist="x", album="z", track=1, rating=1.0)
    dst = AudioTag(ID="2", title="a", artist="x", album="z", track=1, rating=None)

    pair = TrackPair(source_player="src", destination_player="dst", source_track=src)
    pair._score = 100
    pair.match()

    assert pair.sync_state == SyncState.NEEDS_UPDATE


def test_sync_state_conflicting_when_mismatch():
    src = AudioTag(ID="1", title="a", artist="x", album="z", track=1, rating=4.0)
    dst = AudioTag(ID="2", title="a", artist="x", album="z", track=1, rating=2.0)

    pair = TrackPair(source_player="src", destination_player="dst", source_track=src)
    pair._score = 100
    pair.match()

    assert pair.sync_state == SyncState.CONFLICTING


def test_sync_state_none_when_identical():
    src = AudioTag(ID="1", title="a", artist="x", album="z", track=1, rating=3.5)
    dst = AudioTag(ID="2", title="a", artist="x", album="z", track=1, rating=3.5)

    pair = TrackPair(source_player="src", destination_player="dst", source_track=src)
    pair._score = 100
    pair.match()

    assert pair.sync_state is None
