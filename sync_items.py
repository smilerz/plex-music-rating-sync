from typing import List


class AudioTag(object):
    def __init__(self, artist="", album="", title="", file_path=None):
        self.album = album
        self.artist = artist
        self.title = title
        self.rating = 0
        self.genre = ""
        self.file_path = file_path

    def __str__(self):
        return " - ".join([self.artist, self.album, self.title])


class Playlist(object):
    def __init__(self, name, parent_name=""):
        """
        Initializes the playlist with a name
        :type name: str
        :type parent_name: str
        """
        if parent_name != "":
            parent_name += "."
        self.name = parent_name + name
        self.tracks: List[AudioTag] = []
        self.is_auto_playlist = False

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
        return any(t.title.lower() == track.title.lower() and t.artist.lower() == track.artist.lower() for t in self.tracks)

    def missing_tracks(self, other) -> List[AudioTag]:
        """Get list of tracks that exist in other playlist but not in this one"""
        if not isinstance(other, type(self)):
            return []
        return [t for t in other.tracks if not self.has_track(t)]

    @property
    def num_tracks(self):
        return len(self.tracks)

    def __str__(self):
        return "{}: {} tracks".format(self.name, self.num_tracks)

    def add_track(self, track: AudioTag):
        """Add a track to the playlist
        :param track: AudioTag object
        """
        if not isinstance(track, AudioTag):
            raise TypeError("Track must be an AudioTag object")
        self.tracks.append(track)

    def remove_track(self, track: AudioTag):
        self.tracks.remove(track)
