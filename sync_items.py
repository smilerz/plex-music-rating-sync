import logging
from typing import List


class AudioTag(object):
    def __init__(self, artist="", album="", title="", **kwargs):
        self.ID = None
        self.album = album
        self.artist = artist
        self.title = title
        self.rating = kwargs.get("rating", 0)
        self.genre = kwargs.get("genre", "")
        self.file_path = kwargs.get("file_path", None)
        self.track = kwargs.get("rating", 0)

    def __str__(self):
        return " - ".join([self.artist, self.album, self.title])

    @staticmethod
    def get_fields():
        """Get list of fields that should be cached"""
        return ["ID", "album", "artist", "title", "rating", "genre", "file_path", "track"]

    def to_dict(self):
        """Convert to dictionary for caching"""
        return {field: getattr(self, field) for field in self.get_fields()}

    @classmethod
    def from_dict(cls, data: dict):
        """Create AudioTag from cached dictionary"""
        return cls(**data)


class Playlist(object):
    def __init__(self, name, parent_name="", native_playlist=None, player=None):
        """
        Initializes the playlist with a name
        :param name: Playlist name
        :param parent_name: Optional parent playlist name
        :param native_playlist: Native player playlist object
        :param player: Reference to the MediaPlayer instance
        """
        if parent_name != "":
            parent_name += "."
        self.name = parent_name + name
        self.tracks: List[AudioTag] = []
        self.is_auto_playlist = False
        self._native_playlist = native_playlist
        self._player = player
        self._pending_changes = False
        self.logger = logging.getLogger("PlexSync.Playlist")

    def add_tracks(self, tracks):
        """Add multiple tracks to the playlist"""
        if not isinstance(tracks, (list, tuple)):
            tracks = [tracks]
        for track in tracks:
            self.add_track(track)

    def remove_tracks(self, tracks):
        """Remove multiple tracks from the playlist"""
        if not isinstance(tracks, (list, tuple)):
            tracks = [tracks]
        for track in tracks:
            self.remove_track(track)

    def has_pending_changes(self):
        """Check if playlist has changes that need syncing"""
        return self._pending_changes

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplementedError
        return self.name.lower() == other.name.lower()

    def _normalize_track(self, track) -> AudioTag:
        """Convert track to AudioTag if possible, otherwise raise TypeError
        :param track: Track to normalize
        :return: AudioTag instance
        :raises TypeError: If track cannot be converted to AudioTag
        """
        if isinstance(track, AudioTag):
            return track
        if hasattr(track, "title") and hasattr(track, "artist"):
            return AudioTag(artist=track.artist if isinstance(track.artist, str) else track.artist().title, title=track.title)
        raise TypeError("Track must be convertible to AudioTag with title and artist attributes")

    def has_track(self, track: AudioTag) -> bool:
        """Check if a track exists in the playlist
        :param track: Track to check
        :return: True if track exists in playlist
        :raises TypeError: If track cannot be converted to AudioTag
        """
        track = self._normalize_track(track)
        exists = any(t.title.lower() == track.title.lower() and t.artist.lower() == track.artist.lower() for t in self.tracks)
        if not exists:
            self.logger.warning(f"Track not found in playlist {self.name}: {track}")
        return exists

    def missing_tracks(self, other) -> List[AudioTag]:
        """Get list of tracks that exist in other playlist but not in this one"""
        if not isinstance(other, type(self)):
            return []
        missing = [t for t in other.tracks if not self.has_track(t)]
        if missing:
            self.logger.info(f"Found {len(missing)} missing tracks in playlist {self.name}")
        return missing

    @property
    def num_tracks(self):
        return len(self.tracks)

    def __str__(self):
        return "{}: {} tracks".format(self.name, self.num_tracks)

    def add_track(self, track: AudioTag):
        """Add a track to the playlist
        :param track: AudioTag object
        """
        self.tracks.append(self._normalize_track(track))
        self._pending_changes = True

    def remove_track(self, track: AudioTag):
        self.tracks.remove(self._normalize_track(track))
        self._pending_changes = True
