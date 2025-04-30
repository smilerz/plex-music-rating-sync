"""
# TODO: Add test for match with mismatched titles but same track count
# TODO: Add test for match with identical playlists but different orderFunctional tests for PlaylistPair.match(): detects missing or mismatched playlists."""

from sync_pair import PlaylistPair, SyncState
from sync_items import Playlist


def test_detects_missing_destination_playlist(playlist_factory, filesystem_player, plex_player):
    src_playlist = playlist_factory(name="My Playlist")
    pair = PlaylistPair(filesystem_player, plex_player, src_playlist)

    pair.destination_playlist = None
    pair.match()

    assert pair.sync_state == SyncState.MISSING


def test_detects_matching_playlists(playlist_factory, filesystem_player, plex_player):
    src_playlist = playlist_factory(name="My Playlist")
    dst_playlist = playlist_factory(ID="pl2", name="My Playlist")

    pair = PlaylistPair(filesystem_player, plex_player, src_playlist)
    pair.destination_playlist = dst_playlist

    pair.match()

    assert pair.sync_state is None


def test_detects_mismatched_track_count(playlist_factory, filesystem_player, plex_player, track_factory):
    src_playlist = playlist_factory(name="My Playlist", tracks=[track_factory(), track_factory()])
    dst_playlist = playlist_factory(name="My Playlist", tracks=[track_factory()])

    pair = PlaylistPair(filesystem_player, plex_player, src_playlist)
    pair.destination_playlist = dst_playlist

    pair.match()

    assert pair.sync_state == SyncState.CONFLICTING
