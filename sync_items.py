import logging
from typing import List, Optional, Union


# TODO: evaluate if storing native object on Playlist is necessary
class AudioTag(object):
    def __init__(self, artist: str = "", album: str = "", title: str = "", **kwargs):
        self.ID = kwargs.get("ID", None)
        self.album = album
        self.artist = artist
        self.title = title
        self.rating = kwargs.get("rating", None)
        self.genre = kwargs.get("genre", "")
        self.file_path = kwargs.get("file_path", None)
        self.track = kwargs.get("track", None)

    def __str__(self) -> str:
        return " - ".join([self.artist, self.album, self.title])

    def __repr__(self) -> str:
        return f"AudioTag({" - ".join([self.artist, self.album, self.title])})"

    @staticmethod
    def get_fields() -> List[str]:
        """Get list of fields that should be cached"""
        return ["ID", "album", "artist", "title", "rating", "genre", "file_path", "track"]

    def to_dict(self) -> dict:
        """Convert to dictionary for caching"""
        return {field: getattr(self, field) for field in self.get_fields()}

    @classmethod
    def from_dict(self, data: dict) -> "AudioTag":
        """Create AudioTag from cached dictionary"""
        return self(**data)

    @classmethod
    def from_id3(self, id3: object, file_path: str) -> "AudioTag":
        """Create AudioTag from ID3 object"""
        track = id3.get("TRCK", None).text[0]
        return self(
            artist=id3.get("TPE1", "").text[0],
            album=id3.get("TALB", "").text[0],
            title=id3.get("TIT2", "").text[0],
            file_path=str(file_path),
            rating=None,
            ID=str(file_path),
            track=int(track.split("/")[0] if "/" in track else track),
        )

    @classmethod
    def from_vorbis(self, vorbis: object, file_path: str) -> "AudioTag":
        """Create AudioTag from vorbis object"""
        track = vorbis.get("TRACKNUMBER", None)[0]
        return self(
            artist=vorbis.get("ARTIST", "")[0],
            album=vorbis.get("ALBUM", "")[0],
            title=vorbis.get("TITLE", "")[0],
            file_path=str(file_path),
            rating=None,
            ID=str(file_path),
            track=int(track.split("/")[0] if "/" in track else track),
        )


class Playlist(object):
    def __init__(self, name: str, parent_name: str = "", native_playlist: Optional[object] = None, player: Optional[object] = None):
        if parent_name != "":
            parent_name += "."
        self.name = parent_name + name
        self.tracks = []
        self.is_auto_playlist = False
        self._native_playlist = native_playlist
        self._player = player
        self._pending_changes = False
        self.logger = logging.getLogger("PlexSync.Playlist")

    def add_tracks(self, tracks: Union[List[AudioTag], AudioTag]) -> None:
        if not isinstance(tracks, (list, tuple)):
            tracks = [tracks]
        for track in tracks:
            self.add_track(track)

    def remove_tracks(self, tracks: Union[List[AudioTag], AudioTag]) -> None:
        if not isinstance(tracks, (list, tuple)):
            tracks = [tracks]
        for track in tracks:
            self.remove_track(track)

    def has_pending_changes(self) -> bool:
        return self._pending_changes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplementedError
        return self.name.lower() == other.name.lower()

    def _normalize_track(self, track: Union[AudioTag, object]) -> AudioTag:
        if isinstance(track, AudioTag):
            return track
        if hasattr(track, "title") and hasattr(track, "artist"):
            return AudioTag(artist=track.artist if isinstance(track.artist, str) else track.artist().title, title=track.title)
        raise TypeError("Track must be convertible to AudioTag with title and artist attributes")

    def has_track(self, track: AudioTag) -> bool:
        track = self._normalize_track(track)
        exists = any(t.title.lower() == track.title.lower() and t.artist.lower() == track.artist.lower() for t in self.tracks)
        if not exists:
            self.logger.warning(f"Track not found in playlist {self.name}: {track}")
        return exists

    def missing_tracks(self, other: "Playlist") -> List[AudioTag]:
        if not isinstance(other, type(self)):
            return []
        missing = [t for t in other.tracks if not self.has_track(t)]
        if missing:
            self.logger.info(f"Found {len(missing)} missing tracks in playlist {self.name}")
        return missing

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    def __str__(self) -> str:
        return f"{self.name}: {self.num_tracks} tracks"

    def add_track(self, track: AudioTag) -> None:
        self.tracks.append(self._normalize_track(track))
        self._pending_changes = True

    def remove_track(self, track: AudioTag) -> None:
        self.tracks.remove(self._normalize_track(track))
        self._pending_changes = True
