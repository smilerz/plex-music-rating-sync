import abc
import getpass
import logging
import time
from typing import Any, List, Optional, Union

import plexapi.audio
import plexapi.playlist
from plexapi.exceptions import BadRequest, NotFound
from plexapi.myplex import MyPlexAccount

from sync_items import AudioTag, Playlist


# TODO add better error handling
class MediaPlayer(abc.ABC):
    album_empty_alias = ""
    dry_run = False
    rating_maximum = 5

    @staticmethod
    @abc.abstractmethod
    def name():
        """
        The name of this media player
        :return: name of this media player
        :rtype: str
        """
        return ""

    @staticmethod
    @abc.abstractclassmethod
    def format(track):
        # TODO maybe makes more sense to create a track class and make utility functions for __str__, artist, album, title, etc
        # but having to know what player you are working with up front wasn't workable
        """
        Returns a formatted representation of a track in the format of
        artist name - album name - track title
        """
        return NotImplementedError

    def album_empty(self, album):
        if not isinstance(album, str):
            return False
        return album == self.album_empty_alias

    def connect(self, *args, **kwargs):
        return NotImplementedError

    @abc.abstractmethod
    def _create_native_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Any]:
        """Create a native playlist in the player
        :return: Native playlist object or None if dry run
        """
        pass

    @abc.abstractmethod
    def _get_native_playlists(self):
        """Get all native playlists from player"""
        pass

    @abc.abstractmethod
    def _find_native_playlist(self, title: str):
        """Find native playlist by title"""
        pass

    def sync_playlist(self, playlist: Playlist, updates: List[tuple]):
        """Sync playlist changes to native format
        :param playlist: Playlist to update
        :param updates: List of (track, present) tuples indicating changes
        """
        if not self.dry_run and playlist._native_playlist:
            for track, present in updates:
                self.update_playlist(playlist._native_playlist, track, present)

    def create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Playlist]:
        """Create a new playlist with the given tracks
        :return: Created Playlist or None if dry run
        """
        if self.dry_run:
            return None

        native_pl = self._create_native_playlist(title, tracks)
        if native_pl:
            return Playlist(title, native_playlist=native_pl, player=self)
        return None

    def read_playlists(self) -> List[Playlist]:
        """Read all playlists from the player
        :return: List of Playlist objects
        """
        playlists = []
        native_pls = self._get_native_playlists()

        for native_pl in native_pls:
            playlist = self._convert_native_playlist(native_pl)
            if playlist:
                playlists.append(playlist)

        return playlists

    def find_playlist(self, **kwargs) -> Optional[Playlist]:
        """Find playlist by title
        :return: Matching Playlist or None
        """
        title = kwargs.get("title")
        if not title:
            return None

        native_pl = self._find_native_playlist(title)
        if native_pl:
            return self._convert_native_playlist(native_pl)
        return None

    @abc.abstractmethod
    def _convert_native_playlist(self, native_playlist) -> Optional[Playlist]:
        """Convert native playlist to Playlist object"""
        pass

    @staticmethod
    def get_5star_rating(rating):
        return rating * 5 if rating else None

    def get_native_rating(self, normed_rating):
        return normed_rating * self.rating_maximum

    def get_normed_rating(self, rating: Optional[float]):
        return None if rating is None else max(0, rating) / self.rating_maximum

    @abc.abstractmethod
    def read_track_metadata(self, track) -> AudioTag:
        """

        :param track: The track for which to read the metadata.
        :return: The metadata stored in an audio tag instance.
        """

    @abc.abstractmethod
    def search_tracks(self, key: str, value: Union[bool, str]) -> List[AudioTag]:
        """Search the MediaMonkey music library for tracks matching the artist and track title.

        :param key: The search mode. Valid modes are:

                * *rating*  -- Search for tracks that have a rating.
                * *title*   -- Search by track title.
                * *query*   -- MediaMonkey query string, free form.

        :param value: The value to search for.

        :return: a list of matching tracks
        :rtype: list<sync_items.AudioTag>
        """
        pass

    @abc.abstractmethod
    def update_playlist(self, native_playlist, track, present: bool):
        """Updates the native playlist by adding or removing a track
        :param native_playlist: Native playlist object
        :param track: Track to add/remove
        :param present: True to add, False to remove
        """
        pass

    def update_rating(self, track, rating):
        """Updates the rating of the track, unless in dry run"""

    def __hash__(self):
        return hash(self.name().lower())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplementedError
        return other.name().lower() == self.name().lower()


class MediaMonkey(MediaPlayer):
    rating_maximum = 100

    def __init__(self):
        super(MediaMonkey, self).__init__()
        self.logger = logging.getLogger("PlexSync.MediaMonkey")
        self.sdb = None

    @staticmethod
    def name():
        return "MediaMonkey"

    @staticmethod
    def format(track):
        # TODO maybe makes more sense to create a track class and make utility functions for __str__, artist, album, title, etc
        return " - ".join([track.artist, track.album, track.title])

    def connect(self, *args):
        self.logger.info("Connecting to local player {}".format(self.name()))
        import win32com.client

        try:
            self.sdb = win32com.client.Dispatch("SongsDB.SDBApplication")
            self.sdb.ShutdownAfterDisconnect = False
        except Exception:
            self.logger.error("No scripting interface to MediaMonkey can be found. Exiting...")
            exit(1)

    def _create_native_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Any]:
        if not tracks:
            return None
        playlist = self.sdb.PlaylistByTitle("").CreateChildPlaylist(title)
        for track in tracks:
            song = self.sdb.Database.QuerySongs("ID=" + str(track.ID))
            playlist.AddTrack(song.Item)
        return playlist

    def _get_native_playlists(self):
        root = self.sdb.PlaylistByTitle("")
        playlists = []

        def get_playlists(parent):
            for i in range(len(parent.ChildPlaylists)):
                pl = parent.ChildPlaylists[i]
                playlists.append(pl)
                if len(pl.ChildPlaylists):
                    get_playlists(pl)

        get_playlists(root)
        return playlists

    def _find_native_playlist(self, title: str):
        playlists = []
        root = self.sdb.PlaylistByTitle("")

        def find_in_playlists(parent):
            for i in range(len(parent.ChildPlaylists)):
                pl = parent.ChildPlaylists[i]
                if pl.Title.lower() == title.lower():
                    playlists.append(pl)
                if len(pl.ChildPlaylists):
                    find_in_playlists(pl)

        find_in_playlists(root)
        return playlists[0] if playlists else None

    def _convert_native_playlist(self, native_playlist) -> Optional[Playlist]:
        playlist = Playlist(native_playlist.Title, parent_name=native_playlist.Title, native_playlist=native_playlist, player=self)
        playlist.is_auto_playlist = native_playlist.isAutoplaylist
        if not playlist.is_auto_playlist:
            for j in range(native_playlist.Tracks.Count):
                playlist.tracks.append(self.read_track_metadata(native_playlist.Tracks[j]))
        return playlist

    def read_track_metadata(self, track) -> AudioTag:
        tag = AudioTag(
            artist=track.Artist.Name,
            album=track.Album.Name,
            title=track.Title,
            file_path=track.Path,
        )
        tag.rating = self.get_normed_rating(track.Rating)
        tag.ID = track.ID
        tag.track = track.TrackOrder
        return tag

    def search_tracks(self, key: str, value: Union[bool, str]) -> List[AudioTag]:
        # TODO: implement caching of results to avoid repeated queries
        if not value:
            raise ValueError("value can not be empty.")
        if key == "title":
            title = value.replace('"', r'""')
            query = f'SongTitle = "{title}"'
        elif key == "rating":
            if value is True:
                value = "> 0"
            query = f"Rating {value}"
            self.logger.info("Reading tracks from the {} player".format(self.name()))
        elif key == "query":
            query = value
        else:
            raise KeyError(f"Invalid search mode {key}.")
        self.logger.debug(f"Executing query [{query}] against {self.name()}")

        it = self.sdb.Database.QuerySongs(query)
        tags = []
        counter = 0
        while not it.EOF:
            tags.append(self.read_track_metadata(it.Item))
            counter += 1
            it.Next()

        self.logger.info(f"Found {counter} tracks for query {query}.")
        return tags

    def update_playlist(self, native_playlist, track, present: bool):
        self.logger.debug("{} {} to playlist {}".format("Adding" if present else "Removing", self.format(track), native_playlist.Title))
        if not self.dry_run:
            song = self.sdb.Database.QuerySongs("ID=" + str(track.ID))
            if present:
                native_playlist.AddTrack(song.Item)
            else:
                native_playlist.RemoveTrack(song.Item)

    def update_rating(self, track, rating):
        self.logger.debug('Updating rating of track "{}" to {} stars'.format(self.format(track), self.get_5star_rating(rating)))
        if not self.dry_run:
            song = self.sdb.Database.QuerySongs("ID=" + str(track.ID))
            song.Item.Rating = self.get_native_rating(rating)
            song.Item.UpdateDB()


class PlexPlayer(MediaPlayer):
    # TODO logging needs to be updated to reflect whether Plex is source or destination
    maximum_connection_attempts = 3
    rating_maximum = 10
    album_empty_alias = "[Unknown Album]"

    def __init__(self):
        super(PlexPlayer, self).__init__()
        self.logger = logging.getLogger("PlexSync.PlexPlayer")
        self.account = None
        self.plex_api_connection = None
        self.music_library = None

    @staticmethod
    def name():
        return "PlexPlayer"

    @staticmethod
    def format(track):
        # TODO maybe makes more sense to create a track class and make utility functions for __str__, artist, album, title, etc
        try:
            return " - ".join([track.artist().title, track.album().title, track.title])
        except TypeError:
            return " - ".join([track.artist, track.album, track.title])

    def connect(self, server, username, password="", token=""):
        self.logger.info(f"Connecting to the Plex Server {server} with username {username}.")
        connection_attempts_left = self.maximum_connection_attempts
        while connection_attempts_left > 0:
            time.sleep(1)  # important. Otherwise, the above print statement can be flushed after
            if (not password) & (not token):
                password = getpass.getpass()
            try:
                if password:
                    self.account = MyPlexAccount(username=username, password=password)
                elif token:
                    self.account = MyPlexAccount(username=username, token=token)
                break
            except NotFound:
                print(f"Username {username}, password or token wrong for server {server}.")
                password = ""
                connection_attempts_left -= 1
            except BadRequest as error:
                self.logger.warning("Failed to connect: {}".format(error))
                connection_attempts_left -= 1
        if connection_attempts_left == 0 or self.account is None:
            self.logger.error("Exiting after {} failed attempts ...".format(self.maximum_connection_attempts))
            exit(1)

        self.logger.info("Connecting to remote player {} on the server {}".format(self.name(), server))
        try:
            self.plex_api_connection = self.account.resource(server).connect(timeout=5)
            self.logger.info("Successfully connected")
        except NotFound:
            # This also happens if the user is not the owner of the server
            self.logger.error("Error: Unable to connect")
            exit(1)

        self.logger.info("Looking for music libraries")
        music_libraries = {section.key: section for section in self.plex_api_connection.library.sections() if section.type == "artist"}

        if len(music_libraries) == 0:
            self.logger.error("No music library found")
            exit(1)
        elif len(music_libraries) == 1:
            self.music_library = list(music_libraries.values())[0]
            self.logger.debug("Found 1 music library")
        else:
            print("Found multiple music libraries:")
            for key, library in music_libraries.items():
                print("\t[{}]: {}".format(key, library.title))

            choice = input("Select the library to sync with: ")
            self.music_library = music_libraries[int(choice)]

    def read_track_metadata(self, track: plexapi.audio.Track) -> AudioTag:
        tag = AudioTag(
            artist=track.grandparentTitle,
            album=track.parentTitle,
            title=track.title,
            file_path=track.locations[0],
        )
        tag.rating = self.get_normed_rating(track.userRating)
        tag.track = track.index
        tag.ID = track.key
        return tag

    def _create_native_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Any]:
        if not tracks:
            return None
        plex_tracks = []
        for track in tracks:
            # If track is already a Plex track object, use it directly
            if hasattr(track, "type") and track.type == "track":
                plex_tracks.append(track)
            else:
                # Search using TrackPair's find_best_match
                from sync_pair import TrackPair

                pair = TrackPair(None, self, track)
                match, score = pair.find_best_match(candidates=self.music_library.searchTracks(title=track.title))
                if match:
                    plex_tracks.append(match)
                else:
                    self.logger.warning(f"Could not find match for track: {track}")

        if plex_tracks:
            return self.plex_api_connection.createPlaylist(title=title, items=plex_tracks)
        return None

    def _get_native_playlists(self):
        return [pl for pl in self.plex_api_connection.playlists() if pl.playlistType == "audio"]

    def _find_native_playlist(self, title: str):
        try:
            return self.plex_api_connection.playlist(title)
        except NotFound:
            return None

    def _convert_native_playlist(self, native_playlist) -> Optional[Playlist]:
        playlist = Playlist(native_playlist.title, native_playlist=native_playlist, player=self)
        playlist.is_auto_playlist = native_playlist.smart
        if not playlist.is_auto_playlist:
            for item in native_playlist.items():
                playlist.tracks.append(self.read_track_metadata(item))
        return playlist

    def read_playlists(self):
        """
        Read all playlists from the Plex server and convert them to internal Playlist objects.

        :return: List of Playlist objects
        :rtype: list<Playlist>
        """
        self.logger.info("Reading playlists from the {} player".format(self.name()))
        playlists = []

        try:
            plex_playlists = self.plex_api_connection.playlists()
            for plex_playlist in plex_playlists:
                if plex_playlist.playlistType != "audio":
                    continue

                playlist = Playlist(plex_playlist.title)
                playlist.is_auto_playlist = plex_playlist.smart

                if not playlist.is_auto_playlist:
                    for item in plex_playlist.items():
                        playlist.tracks.append(self.read_track_metadata(item))

                playlists.append(playlist)

            self.logger.info("Found {} playlists".format(len(playlists)))
            return playlists

        except Exception as e:
            self.logger.error(f"Failed to read playlists: {e!s}")
            return []

    def find_playlist(self, **kwargs) -> Optional[Playlist]:
        title = kwargs["title"]
        try:
            plex_pl = self.plex_api_connection.playlist(title)
            # Convert to standard Playlist
            playlist = Playlist(plex_pl.title, native_playlist=plex_pl, player=self)
            playlist.is_auto_playlist = plex_pl.smart
            if not playlist.is_auto_playlist:
                for item in plex_pl.items():
                    playlist.tracks.append(self.read_track_metadata(item))
            return playlist
        except NotFound:
            self.logger.debug(f"Playlist {title} not found")
            return None

    def search_tracks(self, key: str, value: Union[bool, str]) -> List[AudioTag]:
        if not value:
            raise ValueError("value can not be empty.")
        if key == "title":
            matches = self.music_library.searchTracks(title=value)
            n_matches = len(matches)
            s_matches = f"match{'es' if n_matches > 1 else ''}"
            self.logger.debug(f"Found {n_matches} {s_matches} for query title={value}")
        elif key == "rating":
            if value is True:
                value = "0"
            matches = self.music_library.searchTracks(**{"track.userRating!": value})
            tags = []
            counter = 0
            for x in matches:
                tags.append(self.read_track_metadata(x))
                counter += 1
            self.logger.info("Found {} tracks with a rating > 0 that need syncing".format(counter))
            matches = tags
        else:
            raise KeyError(f"Invalid search mode {key}.")
        return matches

    def update_playlist(self, native_playlist, track, present: bool):
        """Update Plex native playlist
        :type native_playlist: plexapi.playlist.Playlist
        :type track: Union[plexapi.audio.Track, AudioTag]
        :type present: bool
        """
        if not self.dry_run:
            if hasattr(track, "type") and track.type == "track":
                plex_track = track
            else:
                matches = self.music_library.searchTracks(title=track.title)
                if not matches:
                    self.logger.warning(f"Could not find match for track: {track}")
                    return
                plex_track = matches[0]

            self.logger.debug("{} {} to playlist {}".format("Adding" if present else "Removing", self.format(plex_track), native_playlist.title))

            if present:
                native_playlist.addItems(plex_track)
            else:
                native_playlist.removeItem(plex_track)

    def update_rating(self, track, rating):
        self.logger.debug('Updating rating of track "{}" to {} stars'.format(self.format(track), self.get_5star_rating(rating)))
        if not self.dry_run:
            try:
                track.edit(**{"userRating.value": self.get_native_rating(rating)})
            except AttributeError:
                song = [s for s in self.music_library.searchTracks(title=track.title) if s.key == track.ID][0]
                song.edit(**{"userRating.value": self.get_native_rating(rating)})
