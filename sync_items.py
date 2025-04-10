import logging
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from MediaPlayer import MediaPlayer


# TODO: evaluate if storing native object on Playlist is necessary
class AudioTag(object):
    MAX_ARTIST_LENGTH = 25
    MAX_ALBUM_LENGTH = 30
    MAX_TITLE_LENGTH = 40
    MAX_FILE_PATH_LENGTH = 50
    DISPLAY_HEADER = (
        f"{'':<7} {'Track':<5} {'Artist':<{MAX_ARTIST_LENGTH}} " f"{'Album':<{MAX_ALBUM_LENGTH}} " f"{'Title':<{MAX_TITLE_LENGTH}} " f"{'File Path':<{MAX_FILE_PATH_LENGTH}}"
    )

    def __init__(self, artist: str = "", album: str = "", title: str = "", **kwargs):
        self.ID = kwargs.get("ID", None)
        self.album = album
        self.artist = artist
        self.title = title
        self.rating = kwargs.get("rating", None)
        self.genre = kwargs.get("genre", "")
        self.file_path = kwargs.get("file_path", None)
        self.track = kwargs.get("track", None)
        self.duration = kwargs.get("duration", -1)

    def __str__(self) -> str:
        return " - ".join([self.artist, self.album, self.title])

    def __repr__(self) -> str:
        return f"AudioTag({self!s})"

    @staticmethod
    def truncate(value: str, length: int, from_end: bool = True, default: str = "N/A") -> str:
        """Truncate a string to the specified length, adding '...' at the start or end."""
        value = str(value or default)
        if len(value) <= length:
            return value
        return f"{value[:length - 3]}..." if from_end else f"...{value[-(length - 3):]}"

    def details(self, player: "MediaPlayer") -> str:
        """Print formatted track details."""
        track_number = self.track or 0
        track_rating = player.get_5star_rating(self.rating) if self.rating else "N/A"
        artist = self.truncate(self.artist, self.MAX_ARTIST_LENGTH)
        album = self.truncate(self.album, self.MAX_ALBUM_LENGTH)
        title = self.truncate(self.title, self.MAX_TITLE_LENGTH)
        file_path = self.truncate(self.file_path, self.MAX_FILE_PATH_LENGTH, from_end=False)
        player_rating = f"{player.abbr}[{track_rating}]"
        return (
            f"{player_rating:<7} {track_number:{5}} {artist:<{self.MAX_ARTIST_LENGTH}} "
            f"{album:<{self.MAX_ALBUM_LENGTH}} {title:<{self.MAX_TITLE_LENGTH}} "
            f"{file_path:<{self.MAX_FILE_PATH_LENGTH}}"
        )

    @staticmethod
    def get_fields() -> List[str]:
        """Get list of fields that should be cached"""
        return ["ID", "album", "artist", "title", "rating", "genre", "file_path", "track", "duration"]

    def to_dict(self) -> dict:
        """Convert to dictionary for caching"""
        return {field: getattr(self, field) for field in self.get_fields()}

    @classmethod
    def from_dict(cls, data: dict) -> "AudioTag":
        """Create AudioTag from cached dictionary"""
        return cls(**data)

    @classmethod
    def from_id3(cls, id3: object, file_path: str, duration: int = -1) -> "AudioTag":
        from filesystem_provider import ID3Field

        """Create AudioTag from ID3 object."""
        track = id3.get(ID3Field.TRACKNUMBER, None).text[0]
        duration = int(duration or -1)
        return cls(
            artist=id3.get(ID3Field.ARTIST, "").text[0],
            album=id3.get(ID3Field.ALBUM, "").text[0],
            title=id3.get(ID3Field.TITLE, "").text[0],
            file_path=str(file_path),
            rating=None,
            ID=str(file_path),
            track=int(track.split("/")[0] if "/" in track else track),
            duration=int(duration),
        )

    @classmethod
    def from_vorbis(cls, vorbis: object, file_path: str) -> "AudioTag":
        from filesystem_provider import VorbisField

        """Create AudioTag from Vorbis object."""
        track = vorbis.get(VorbisField.TRACKNUMBER, None)[0]
        duration = vorbis.info.length if hasattr(vorbis, "info") and hasattr(vorbis.info, "length") else -1
        return cls(
            artist=vorbis.get(VorbisField.ARTIST, "")[0],
            album=vorbis.get(VorbisField.ALBUM, "")[0],
            title=vorbis.get(VorbisField.TITLE, "")[0],
            file_path=str(file_path),
            rating=None,
            ID=str(file_path),
            track=int(track.split("/")[0] if "/" in track else track),
            duration=int(duration),
        )


class Playlist(object):
    def __init__(self, name: str, player: Optional[object] = None):
        self.name = name
        self.tracks = []
        self.is_auto_playlist = False
        self.is_extm3u = True
        self._player = player
        self._pending_changes = False
        self.logger = logging.getLogger("PlexSync.Playlist")

    def __repr__(self) -> str:
        return f"{self._player} Playlist({self.name})"

    def __str__(self) -> str:
        return f"{self._player} Playlist({self.name})"

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
            return NotImplemented
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
        if not self.tracks:
            self._player.read_playlist_tracks(self)
        if not other.tracks:
            self._player.read_playlist_tracks(other)
        missing = [t for t in other.tracks if not self.has_track(t)]
        if missing:
            self.logger.info(f"Found {len(missing)} missing tracks in playlist {self.name}")
        return missing

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    def add_track(self, track: AudioTag) -> None:
        self.tracks.append(self._normalize_track(track))
        self._pending_changes = True

    def remove_track(self, track: AudioTag) -> None:
        self.tracks.remove(self._normalize_track(track))
        self._pending_changes = True
