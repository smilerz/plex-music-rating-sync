import abc
import getpass
import logging
import time
from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from cache_manager import CacheManager

import plexapi.audio
import plexapi.playlist
from plexapi.exceptions import BadRequest, NotFound
from plexapi.myplex import MyPlexAccount

from sync_items import AudioTag, Playlist


class MediaPlayer(abc.ABC):
    album_empty_alias = ""
    dry_run = False
    rating_maximum = 5

    def set_cache_manager(self, cache_manager: "CacheManager") -> None:
        self.cache_manager = cache_manager
        self.logger.debug(f"Cache manager for {self.name()} set to: {cache_manager.__class__.__name__}")

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """
        The name of this media player
        :return: name of this media player
        :rtype: str
        """
        return ""

    def album_empty(self, album: str) -> bool:
        if not isinstance(album, str):
            return False
        return album == self.album_empty_alias

    def connect(self, *args, **kwargs) -> None:
        return NotImplemented

    @abc.abstractmethod
    def _create_native_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Any]:
        """Create a native playlist in the player
        :return: Native playlist object or None if dry run
        """

    @abc.abstractmethod
    def _get_native_playlists(self):
        """Get all native playlists from player"""

    @abc.abstractmethod
    def _find_native_playlist(self, title: str):
        """Find native playlist by title"""

    @abc.abstractmethod
    def _search_native_tracks(self, key: str, value: Union[bool, str]) -> List[Any]:
        """Search for tracks in the native player format

        :param key: Search mode (rating, title, etc)
        :param value: Search value
        :return: List of native track objects
        """

    @abc.abstractmethod
    def _add_track_to_playlist(self, native_playlist: Any, native_track: Any) -> None:
        """Add a track to a native playlist"""

    @abc.abstractmethod
    def _remove_track_from_playlist(self, native_playlist: Any, native_track: Any) -> None:
        """Remove a track from a native playlist"""

    def search_tracks(self, key: str, value: Union[bool, str]) -> List[AudioTag]:
        """Search tracks and convert to AudioTag format

        :param key: Search mode (rating, title, etc)
        :param value: Search value
        :return: List of AudioTag objects
        """
        self.logger.debug(f"Searching tracks with {key}={value}")
        if not value:
            self.logger.error("Search value cannot be empty")
            raise ValueError("value can not be empty.")

        native_tracks = self._search_native_tracks(key, value)

        tags = []
        counter = 0
        for track in native_tracks:
            tag = self.read_track_metadata(track)
            tags.append(tag)
            counter += 1
            print(f"\rIdentifying metadata for {counter} tracks", end="", flush=True)

        self.logger.info(f"Found {counter} tracks for {key}={value}")
        return tags

    def sync_playlist(self, playlist: Playlist, updates: List[tuple]):
        """Sync playlist changes to native format
        :param playlist: Playlist to update
        :param updates: List of (track, present) tuples indicating changes
        """
        if not updates:
            self.logger.debug("No updates to sync")
            return

        if self.dry_run:
            self.logger.info(f"DRY RUN: Would sync {len(updates)} changes to playlist '{playlist.name}'")
            return

        if playlist._native_playlist:
            self.logger.info(f"Syncing {len(updates)} changes to playlist '{playlist.name}'")
            for track, present in updates:
                self.update_playlist(playlist._native_playlist, track, present)
            self.logger.debug("Playlist sync completed")

    def create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Playlist]:
        """Create a new playlist with the given tracks
        :return: Created Playlist or None if dry run
        """
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would create playlist '{title}' with {len(tracks)} tracks")
            return None

        self.logger.info(f"Creating playlist '{title}' with {len(tracks)} tracks")
        native_pl = self._create_native_playlist(title, tracks)
        if native_pl:
            self.logger.debug(f"Successfully created playlist '{title}'")
            return Playlist(title, native_playlist=native_pl, player=self)
        self.logger.error(f"Failed to create playlist '{title}'")
        return None

    def read_playlists(self) -> List[Playlist]:
        """Read all playlists from the player
        :return: List of Playlist objects
        """
        self.logger.info(f"Reading all playlists from {self.name()}")
        native_pls = self._get_native_playlists()
        self.logger.debug(f"Found {len(native_pls)} native playlists")

        playlists = []
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
            self.logger.warning("Find playlist called without title")
            return None

        self.logger.debug(f"Searching for playlist '{title}'")
        native_pl = self._find_native_playlist(title)
        if native_pl:
            self.logger.debug(f"Found playlist '{title}'")
            return self._convert_native_playlist(native_pl)

        self.logger.info(f"Playlist '{title}' not found")
        return None

    @abc.abstractmethod
    def _convert_native_playlist(self, native_playlist) -> Optional[Playlist]:
        """Convert native playlist to Playlist object"""

    @staticmethod
    def get_5star_rating(rating: float) -> float:
        return rating * 5

    def get_native_rating(self, normed_rating: float) -> float:
        return normed_rating * self.rating_maximum

    def get_normed_rating(self, rating: Optional[float]) -> float:
        return None if rating is None else max(0, rating) / self.rating_maximum

    @abc.abstractmethod
    def read_track_metadata(self, track) -> AudioTag:
        """
        Reads the metadata of a track.

        :param track: The track for which to read the metadata.
        :return: The metadata stored in an audio tag instance.
        """

    @abc.abstractmethod
    def update_rating(self, track: Any, rating: float) -> None:
        """Updates the rating of the track, unless in dry run"""

    def __hash__(self):
        return hash(self.name().lower())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplementedError
        return other.name().lower() == self.name().lower()

    def update_playlist(self, native_playlist, track, present: bool):
        """Updates the native playlist by adding or removing a track
        :param native_playlist: Native playlist object
        :param track: Track to add/remove
        :param present: True to add, False to remove
        """
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would {'add' if present else 'remove'} track from playlist")
            return

        matches = self._search_native_tracks("id", track.ID)
        if not matches:
            self.logger.warning(f"Could not find native track for: {track}")
            return
        native_track = matches[0]

        pl_name = native_playlist.title if hasattr(native_playlist, "title") else native_playlist.Title
        self.logger.info(f"{'Adding' if present else 'Removing'} track '{track}' {'to' if present else 'from'} playlist '{pl_name}'")

        try:
            if present:
                self._add_track_to_playlist(native_playlist, native_track)
            else:
                self._remove_track_from_playlist(native_playlist, native_track)
            self.logger.debug(f"Successfully {'added' if present else 'removed'} track")
        except Exception as e:
            self.logger.error(f"Failed to {'add' if present else 'remove'} track: {e!s}")
            raise


class MediaMonkey(MediaPlayer):
    rating_maximum = 100

    def __init__(self):
        super(MediaMonkey, self).__init__()
        self.logger = logging.getLogger("PlexSync.MediaMonkey")
        self.sdb = None

    @classmethod
    def name(self) -> str:
        return "MediaMonkey"

    def connect(self, *args) -> None:
        import win32com.client

        try:
            self.sdb = win32com.client.Dispatch("SongsDB.SDBApplication")
            self.sdb.ShutdownAfterDisconnect = False
            self.logger.info("Successfully connected to MediaMonkey")
        except Exception as e:
            self.logger.error(f"Failed to connect to MediaMonkey: {e!s}")
            raise

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

        # TODO: fix playlist naming
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
                playlist.tracks.append(track := self.read_track_metadata(native_playlist.Tracks[j]))
                self.logger.debug(f"Added track {track} to playlist")
        return playlist

    def read_track_metadata(self, track: Any) -> AudioTag:
        cached = self.cache_manager.get_metadata(self.name(), track.ID)
        if cached is not None:
            return cached

        tag = AudioTag(
            artist=track.Artist.Name,
            album=track.Album.Name,
            title=track.Title,
            file_path=track.Path,
            rating=self.get_normed_rating(track.Rating),
            ID=track.ID,
            track=track.TrackOrder,
        )
        self.cache_manager.set_metadata(self.name(), tag.ID, tag)

        return tag

    def _search_native_tracks(self, key: str, value: Union[bool, str]) -> List[Any]:
        if key == "title":
            title = value.replace('"', r'""')
            query = f'SongTitle = "{title}"'
        elif key == "rating":
            if value is True:
                value = "> 0"
            query = f"Rating {value}"
        elif key == "query":
            query = value
        elif key == "id":
            query = f"ID = {value}"
        else:
            raise KeyError(f"Invalid search mode {key}.")

        self.logger.debug(f"Executing query [{query}] against {self.name()}")
        it = self.sdb.Database.QuerySongs(query)

        results = []
        counter = 1
        while not it.EOF:
            print(f"\rReading track {counter} from {self.name()}", end="", flush=True)
            results.append(it.Item)
            it.Next()
            counter += 1
        return results

    def update_rating(self, track, rating: float) -> None:
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {rating}")
            return

        self.logger.debug(f"Updating rating for {track} to {rating}")
        try:
            song = self.sdb.Database.QuerySongs("ID=" + str(track.ID))
            song.Item.Rating = self.get_native_rating(rating)
            song.Item.UpdateDB()
            self.logger.info(f"Successfully updated rating for {track}")
        except Exception as e:
            self.logger.error(f"Failed to update rating: {e!s}")
            raise

    def _add_track_to_playlist(self, native_playlist: Any, native_track: Any) -> None:
        native_playlist.AddTrack(native_track)

    def _remove_track_from_playlist(self, native_playlist: Any, native_track: Any) -> None:
        native_playlist.RemoveTrack(native_track)


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
    def name() -> str:
        return "PlexPlayer"

    @staticmethod
    def format(track) -> str:
        # TODO maybe makes more sense to create a track class and make utility functions for __str__, artist, album, title, etc
        try:
            return " - ".join([track.artist().title, track.album().title, track.title])
        except TypeError:
            return " - ".join([track.artist, track.album, track.title])

    def connect(self, server: str, username: str, password: str = "", token: str = "") -> None:
        self.logger.info(f"Connecting to Plex server {server} as {username}")
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
                self.logger.warning(f"Failed to connect: {error}")
                connection_attempts_left -= 1
        if connection_attempts_left == 0 or self.account is None:
            self.logger.error(f"Exiting after {self.maximum_connection_attempts} failed attempts ...")
            exit(1)

        self.logger.info(f"Connecting to remote player {self.name()} on the server {server}")
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
            self.music_library = next(iter(music_libraries.values()))
            self.logger.debug("Found 1 music library")
        else:
            print("Found multiple music libraries:")
            for key, library in music_libraries.items():
                print(f"\t[{key}]: {library.title}")

            while True:
                try:
                    choice = input("Select the library to sync with: ")
                    self.music_library = music_libraries[int(choice)]
                    break
                except (ValueError, KeyError):
                    print("Invalid choice. Please enter a valid number corresponding to the library.")

    def read_track_metadata(self, track: plexapi.audio.Track) -> AudioTag:
        return AudioTag(
            artist=track.grandparentTitle,
            album=track.parentTitle,
            title=track.title,
            file_path=track.locations[0],
            rating=self.get_normed_rating(track.userRating),
            ID=track.key.split("/")[-1] if "/" in track.key else track.key,
            track=track.index,
        )

    def _create_native_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Any]:
        if not tracks:
            return None
        plex_tracks = []
        for track in tracks:
            self.logger.debug(f"Searching for track ID: {track.ID}")
            try:
                matches = self._search_native_tracks("id", track.ID)
                if matches:
                    plex_tracks.append(matches[0])
                    self.logger.debug(f"Found track: {matches[0].title}")
                else:
                    self.logger.warning(f"No match found for track ID: {track.ID}")
            except Exception as e:
                self.logger.error(f"Failed to search for track {track.ID}: {e!s}")

        if plex_tracks:
            return self.plex_api_connection.createPlaylist(title=title, items=plex_tracks)
        self.logger.warning("No tracks found to create playlist")
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

    def read_playlists(self) -> List[Playlist]:
        """
        Read all playlists from the Plex server and convert them to internal Playlist objects.

        :return: List of Playlist objects
        :rtype: list<Playlist>
        """
        self.logger.info(f"Reading playlists from the {self.name()} player")
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

            self.logger.info(f"Found {len(playlists)} playlists")
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

    def _search_native_tracks(self, key: str, value: Union[bool, str]) -> List[Any]:
        if key == "title":
            matches = self.music_library.searchTracks(title=value)
            n_matches = len(matches)
            s_matches = f"match{'es' if n_matches > 1 else ''}"
            self.logger.debug(f"Found {n_matches} {s_matches} for query title={value}")
            return matches

        elif key == "rating":
            if value is True:
                value = "0"
            matches = self.music_library.searchTracks(**{"track.userRating!": value})
            return matches

        elif key == "id":
            return [self.music_library.fetchItem(int(value))]

        raise KeyError(f"Invalid search mode {key}.")

    def update_rating(self, track: Any, rating: float) -> None:
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {rating}")
            return

        self.logger.debug(f"Updating rating for {track} to {rating}")
        try:
            track.edit(**{"userRating.value": self.get_native_rating(rating)})
            self.logger.info(f"Successfully updated rating for {track}")
        except Exception as e:
            self.logger.error(f"Failed to update rating: {e!s}")
            try:
                # Fallback to searching by title/ID
                song = next(s for s in self.music_library.searchTracks(title=track.title) if s.key == track.ID)
                song.edit(**{"userRating.value": self.get_native_rating(rating)})
                self.logger.info("Successfully updated rating using fallback method")
            except Exception as e2:
                self.logger.error(f"Failed to update rating using fallback: {e2!s}")
                raise

    def _add_track_to_playlist(self, native_playlist: Any, native_track: Any) -> None:
        native_playlist.addItems(native_track)

    def _remove_track_from_playlist(self, native_playlist: Any, native_track: Any) -> None:
        native_playlist.removeItem(native_track)
