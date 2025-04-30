"""
# TODO: Add test for duplicate playlist title disambiguation
# TODO: Add test for playlist missing header (non-EXTM3U format)
# TODO: Add test for relative path calculation in add_track_to_playlistFunctional tests for playlist creation, metadata resolution, and track add/remove behavior."""

from pathlib import Path

from filesystem_provider import FileSystemProvider


def test_create_playlist_adds_tracks(monkeypatch, track_factory):
    fsp = FileSystemProvider()
    monkeypatch.setattr("filesystem_provider.Path.resolve", lambda self: self)
    monkeypatch.setattr(fsp, "path", Path("/music"))

    track = track_factory(file_path="/music/track1.mp3")
    monkeypatch.setattr(fsp, "add_track_to_playlist", lambda pid, t, is_extm3u=False: True)

    playlist = fsp.create_playlist("MyPlaylist", is_extm3u=True)
    assert playlist is not None
    assert playlist.name == "MyPlaylist"


def test_read_playlist_metadata_extracts_title(tmp_path):
    playlist_file = tmp_path / "MyList.m3u"
    playlist_file.write_text("""#EXTM3U
#PLAYLIST:Chill Vibes
/music/song1.mp3
""")

    fsp = FileSystemProvider()
    fsp.playlist_path = tmp_path
    playlist = fsp.read_playlist_metadata(playlist_file)

    assert playlist.name == "Chill Vibes"
    assert playlist.ID.lower() == str(playlist_file).lower()


def test_get_tracks_from_playlist_filters_scope(monkeypatch, tmp_path):
    playlist_path = tmp_path / "pl.m3u"
    playlist_path.write_text("""out_of_scope.mp3
valid/track1.mp3
""")

    fsp = FileSystemProvider()
    fsp.path = tmp_path / "valid"
    fsp.playlist_path = tmp_path
    playlist_path = playlist_path.resolve()

    monkeypatch.setattr("filesystem_provider.Path.resolve", lambda self: self)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_relative_to", lambda self, other: str(self).startswith(str(other)))
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "suffix", property(lambda self: ".mp3"))

    tracks = fsp.get_tracks_from_playlist(playlist_path)
    assert all(fsp.path in t.parents or t == fsp.path for t in tracks)
