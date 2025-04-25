import pytest
from unittest.mock import MagicMock
from sync_items import Playlist
from sync_pair import PlaylistPair, SyncState
from MediaPlayer import MediaPlayer

@pytest.fixture
def mock_player():
    p = MagicMock(spec=MediaPlayer)
    p.name.return_value = "mock"
    p.search_playlists.return_value = []
    return p

@pytest.fixture
def playlist():
    return Playlist(ID="1", name="Test Playlist")

def test_playlistpair_match_creates_if_none(mock_player, playlist):
    pair = PlaylistPair(mock_player, mock_player, playlist)
    pair.match()
    assert pair.sync_state == SyncState.NEEDS_UPDATE