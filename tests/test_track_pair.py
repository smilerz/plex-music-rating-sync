import pytest
from unittest.mock import MagicMock
from sync_pair import TrackPair, SyncState, MatchThreshold
from sync_items import AudioTag
from ratings import Rating
from MediaPlayer import MediaPlayer

@pytest.fixture
def mock_player():
    player = MagicMock(spec=MediaPlayer)
    player.name.return_value = "mock"
    player.album_empty.side_effect = lambda album: album in ("", None)
    player.search_tracks.return_value = []
    return player

@pytest.fixture
def rated_track():
    return AudioTag(ID="1", title="Title", artist="Artist", album="Album", track=1, rating=Rating(1.0))

@pytest.fixture
def same_rated_track():
    return AudioTag(ID="2", title="Title", artist="Artist", album="Album", track=1, rating=Rating(1.0))

@pytest.fixture
def unrated_track():
    return AudioTag(ID="3", title="Title", artist="Artist", album="Album", track=1, rating=Rating.unrated())

@pytest.fixture
def conflict_track():
    return AudioTag(ID="4", title="Title", artist="Artist", album="Album", track=1, rating=Rating(0.5))

def test_quality_levels(mock_player, rated_track):
    pair = TrackPair(mock_player, mock_player, rated_track)
    for score, expected in [(None, None), (20, None), (30, MatchThreshold.POOR_MATCH), (80, MatchThreshold.GOOD_MATCH), (100, MatchThreshold.PERFECT_MATCH)]:
        pair.score = score
        pair.sync_state = SyncState.UP_TO_DATE
        assert pair.quality == expected

def test_has_min_quality(mock_player, rated_track):
    pair = TrackPair(mock_player, mock_player, rated_track)
    pair.score = 80
    pair.sync_state = SyncState.UP_TO_DATE
    assert pair.has_min_quality(MatchThreshold.POOR_MATCH) is True
    assert pair.has_min_quality(MatchThreshold.PERFECT_MATCH) is False

def test_is_sync_candidate_and_unmatched(mock_player, rated_track):
    pair = TrackPair(mock_player, mock_player, rated_track)
    pair.sync_state = SyncState.NEEDS_UPDATE
    assert pair.is_sync_candidate()
    pair.sync_state = SyncState.CONFLICTING
    assert pair.is_sync_candidate()
    pair.sync_state = SyncState.UNKNOWN
    assert pair.is_unmatched()
    pair.sync_state = SyncState.ERROR
    assert pair.is_unmatched()

def test_reversed_pair(mock_player, rated_track, same_rated_track):
    pair = TrackPair(mock_player, mock_player, rated_track)
    pair.destination = same_rated_track
    pair.rating_source = Rating(1.0)
    pair.rating_destination = Rating(1.0)
    pair.score = 100
    rev = pair.reversed()
    assert rev.source == same_rated_track
    assert rev.destination == rated_track
    assert rev.source_player == mock_player
    assert rev.destination_player == mock_player
    assert rev.rating_source == pair.rating_destination
    assert rev.score == pair.score

def test_sync_state_transitions(mock_player, rated_track, same_rated_track, unrated_track, conflict_track):
    cases = [
        (same_rated_track, SyncState.UP_TO_DATE),
        (unrated_track, SyncState.NEEDS_UPDATE),
        (conflict_track, SyncState.CONFLICTING),
    ]
    for dest, expected_state in cases:
        pair = TrackPair(mock_player, mock_player, rated_track)
        pair.destination = dest
        pair.rating_source = rated_track.rating
        pair.rating_destination = dest.rating
        pair.match_score = 100
        pair.match()
        assert pair.sync_state == expected_state