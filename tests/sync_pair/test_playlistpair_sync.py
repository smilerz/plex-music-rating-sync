"""
# TODO: Add test for sync that removes a track (present = False)
# TODO: Add test for dry-run skipping all sync actionsFunctional tests for PlaylistPair.sync(): applies creation, update, and no-op behaviors."""

from sync_pair import PlaylistPair, SyncState
from sync_items import Playlist


def test_creates_missing_playlist(playlist_factory, filesystem_player, plex_player):
    src = playlist_factory(name="Sync Me")
    pair = PlaylistPair(filesystem_player, plex_player, src)
    pair.destination_playlist = None
    pair.sync_state = SyncState.MISSING

    plex_player.create_playlist.return_value = playlist_factory(name="Sync Me", ID="pl999")

    pair.sync()

    plex_player.create_playlist.assert_called_once()
    assert pair.destination_playlist is not None


def test_updates_existing_playlist(playlist_factory, track_factory, filesystem_player, plex_player):
    src = playlist_factory(name="To Update", tracks=[track_factory(ID="t1"), track_factory(ID="t2")])
    dst = playlist_factory(name="To Update", tracks=[track_factory(ID="t1")])

    pair = PlaylistPair(filesystem_player, plex_player, src)
    pair.destination_playlist = dst
    pair.sync_state = SyncState.CONFLICTING

    pair.sync()

    # Should try to add missing t2
    plex_player.update_playlist.assert_any_call(dst, src.tracks[1], True)


def test_sync_skips_when_state_none(playlist_factory, filesystem_player, plex_player):
    src = playlist_factory(name="Same")
    dst = playlist_factory(name="Same")

    pair = PlaylistPair(filesystem_player, plex_player, src)
    pair.destination_playlist = dst
    pair.sync_state = None

    pair.sync()

    plex_player.create_playlist.assert_not_called()
    plex_player.update_playlist.assert_not_called()
