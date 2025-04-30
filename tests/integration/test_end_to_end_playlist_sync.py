"""
# TODO: Add test for failed create_playlist scenario
# TODO: Add test for playlist update that removes a track
# TODO: Add test for playlist ID collision or duplicate titlesIntegration test for full playlist sync lifecycle: discover, match, create/update."""

from MediaPlayer import FileSystem, Plex
from sync_items import Playlist
from sync_pair import PlaylistPair, SyncState


def test_playlist_created_when_missing(playlist_factory):
    fs = FileSystem()
    plex = Plex()

    playlist = playlist_factory(name="New Sync List")

    fs.search_playlists = lambda key, val, **_: [playlist] if val == "New Sync List" else []
    plex.search_playlists = lambda key, val, **_: []

    fs.create_playlist = lambda title: Playlist(ID="pl-new", name=title)
    plex.create_playlist = lambda title: Playlist(ID="pl-created", name=title)

    pair = PlaylistPair(fs, plex, playlist)
    pair.destination_playlist = None
    pair.match()
    assert pair.sync_state == SyncState.MISSING

    pair.sync()
    assert pair.destination_playlist.name == "New Sync List"


def test_playlist_updated_when_conflicting(playlist_factory, track_factory):
    fs = FileSystem()
    plex = Plex()

    src = playlist_factory(name="Favorites", tracks=[track_factory(ID="a"), track_factory(ID="b")])
    dst = playlist_factory(name="Favorites", tracks=[track_factory(ID="a")])

    plex.update_playlist = lambda pl, track, present=True: pl.tracks.append(track)

    pair = PlaylistPair(fs, plex, src)
    pair.destination_playlist = dst
    pair.match()
    assert pair.sync_state == SyncState.CONFLICTING

    pair.sync()
    assert len(pair.destination_playlist.tracks) == 2
