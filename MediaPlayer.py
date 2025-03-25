import abc
import getpass
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from cache_manager import CacheManager
    from stats_manager import StatsManager

from pathlib import Path
from typing import Tuple

import mutagen
import plexapi.audio
import plexapi.playlist
from fuzzywuzzy import fuzz
from plexapi.exceptions import BadRequest, NotFound
from plexapi.myplex import MyPlexAccount

from sync_items import AudioTag, Playlist

NativePlaylist = Any
NativeTrack = Any
MediaMonkeyPlaylist = Any  # COM SDBPlaylist object
MediaMonkeyTrack = Any  # COM SDBSongData object
PlexPlaylist = Any  # plexapi.playlist.Playlist
PlexTrack = Any  # plexapi.audio.Track

# TODO: itunes
# TODO add mediamonkey 5
# TODO: add updating Album, Artist, Title, Rating, Track, Genre
# TODO: add setting source of record per attribute


class RatingTag(Enum):
    WINDOWSMEDIAPLAYER = ("POPM:Windows Media Player 9 Series", "Windows Media Player")
    MEDIAMONKEY = ("POPM:no@email", "MediaMonkey")
    MUSICBEE = ("POPM:MusicBee", "MusicBee")
    WINAMP = ("POPM:rating@winamp", "Winamp")
    TEXT = ("TXXX:RATING", "Text")

    def __init__(self, tag: str, player_name: str):
        self.tag = tag
        self.player_name = player_name

    @classmethod
    def from_tag(cls, tag: str) -> Optional["RatingTag"]:
        for item in cls:
            if item.tag == tag:
                return item
        return None

    @classmethod
    def from_value(cls, value: Optional[str]) -> Optional[Union["RatingTag", str]]:
        if value is None:
            return None
        if value.startswith("POPM:"):
            return value  # Return the string directly for "POPM:<some string>"
        for item in cls:
            if item.tag == value or item.player_name == value:
                return item
        raise ValueError(f"Invalid RatingTag: {value}")


class TagWriteStrategy(Enum):
    WRITE_ALL = "write_all"
    WRITE_EXISTING = "write_existing"
    WRITE_STANDARD = "write_standard"
    OVERWRITE_STANDARD = "overwrite_standard"

    @classmethod
    def from_value(cls, value: Optional[str]) -> Optional["TagWriteStrategy"]:
        if value is None:
            return None
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Invalid TagWriteStrategy: {value}")


class ConflictResolutionStrategy(Enum):
    PRIORITIZED_ORDER = "prioritized_order"
    HIGHEST = "highest"
    LOWEST = "lowest"
    AVERAGE = "average"

    @classmethod
    def from_value(cls, value: Optional[str]) -> Optional["ConflictResolutionStrategy"]:
        if value is None:
            return None
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Invalid ConflictResolutionStrategy: {value}")


class MediaPlayer(abc.ABC):
    album_empty_alias = ""
    dry_run = False
    rating_maximum = 5
    abbr = None

    def __init__(self, cache_manager: Optional["CacheManager"] = None, stats_manager: Optional["StatsManager"] = None):
        self.cache_manager = cache_manager
        self.stats_manager = stats_manager
        if cache_manager:
            self.logger.debug(f"Cache manager for {self.name()} set to: {cache_manager.__class__.__name__}")
        if stats_manager:
            self.logger.debug(f"Stats manager for {self.name()} set to: {stats_manager.__class__.__name__}")

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
    def _create_native_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[NativePlaylist]:
        """Create a native playlist in the player"""

    @abc.abstractmethod
    def _get_native_playlists(self) -> NativePlaylist:
        """Get all native playlists from player"""

    @abc.abstractmethod
    def _find_native_playlist(self, title: str) -> NativePlaylist:
        """Find native playlist by title"""

    @abc.abstractmethod
    def _search_native_tracks(self, key: str, value: Union[bool, str]) -> List[NativeTrack]:
        """Search for tracks in the native player format"""

    @abc.abstractmethod
    def _add_track_to_playlist(self, native_playlist: NativePlaylist, native_track: NativeTrack) -> None:
        """Add a track to a native playlist"""

    @abc.abstractmethod
    def _remove_track_from_playlist(self, native_playlist: NativePlaylist, native_track: NativeTrack) -> None:
        """Remove a track from a native playlist"""

    def search_tracks(self, key: str, value: Union[bool, str], track_status: bool = False) -> List[AudioTag]:
        """Search tracks and convert to AudioTag format"""
        self.logger.debug(f"Searching tracks with {key}={value}")
        if not value:
            self.logger.error("Search value cannot be empty")
            raise ValueError("value can not be empty.")

        native_tracks = self._search_native_tracks(key, value)

        tags = []
        status = None
        counter = 0

        try:
            total = len(native_tracks)
        except TypeError:
            total = None

        for track in native_tracks:
            if (total and total > 50) or counter >= 50:
                if not status:
                    status = self.stats_manager.get_status_handler()
                    bar = status.start_phase(f"Reading track metadata from {self.name()}", initial=counter, total=total)
                bar.update()
            tag = self.read_track_metadata(track)
            tags.append(tag)
            counter += 1

        bar.close() if status else None
        self.logger.debug(f"Found {len(tags)} tracks for {key}={value}")
        return tags

    def sync_playlist(self, playlist: Playlist, updates: List[Tuple[AudioTag, bool]]) -> None:
        """Sync playlist changes to native format"""
        if not updates:
            self.logger.debug("No updates to sync")
            return

        if self.dry_run:
            self.logger.info(f"DRY RUN: Would sync {len(updates)} changes to playlist '{playlist.name}'")
            return

        if playlist._native_playlist:
            if len(updates) > 0:
                self.logger.info(f"Syncing {len(updates)} changes to playlist '{playlist.name}'")
                status = self.stats_manager.get_status_handler()
                bar = status.start_phase("Syncing playlist updates", total=len(updates))
                for track, present in updates:
                    self.update_playlist(playlist._native_playlist, track, present)
                    bar.update()
                bar.close()
            self.logger.debug("Playlist sync completed")

    def create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Playlist]:
        """Create a new playlist with the given tracks"""
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

    def find_playlist(self, title: str) -> Optional[Playlist]:
        """Find playlist by title
        :return: Matching Playlist or None
        """
        if not title:
            self.logger.warning("Find playlist without title")
            return None

        self.logger.debug(f"Searching for playlist '{title}'")
        native_pl = self._find_native_playlist(title)
        if native_pl:
            self.logger.debug(f"Found playlist '{title}'")
            return self._convert_native_playlist(native_pl)

        self.logger.info(f"Playlist '{title}' not found")
        return None

    @abc.abstractmethod
    def _convert_native_playlist(self, native_playlist: NativePlaylist) -> Optional[Playlist]:
        """Convert native playlist to Playlist object"""

    @staticmethod
    def get_5star_rating(rating: float) -> float:
        return rating * 5

    def get_native_rating(self, normed_rating: float) -> float:
        return normed_rating * self.rating_maximum

    def get_normed_rating(self, rating: Optional[float]) -> float:
        return None if rating is None else max(0, rating) / self.rating_maximum

    @abc.abstractmethod
    def read_track_metadata(self, track: NativeTrack) -> AudioTag:
        """
        Reads the metadata of a track.

        :param track: The track for which to read the metadata.
        :return: The metadata stored in an audio tag instance.
        """

    @abc.abstractmethod
    def update_rating(self, track: AudioTag, rating: float) -> None:
        """Updates the rating of the track, unless in dry run"""

    def __hash__(self) -> int:
        return hash(self.name().lower())

    def __eq__(self, other: "MediaPlayer") -> bool:
        if not isinstance(other, type(self)):
            return NotImplementedError
        return other.name().lower() == self.name().lower()

    def update_playlist(self, native_playlist: NativePlaylist, track: AudioTag, present: bool) -> None:
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
            self.logger.warning(f"Could not find track for: {track} in {self.name()}")
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

    def __init__(self, cache_manager: Optional["CacheManager"] = None, stats_manager: Optional["StatsManager"] = None):
        self.logger = logging.getLogger("PlexSync.MediaMonkey")
        self.sdb = None
        self.abbr = "MM"
        super().__init__(cache_manager, stats_manager)

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

    def _create_native_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[MediaMonkeyPlaylist]:
        if not tracks:
            return None
        self.logger.info(f"Creating playlist {title} with {len(tracks)} tracks")
        playlist = self.sdb.PlaylistByTitle("").CreateChildPlaylist(title)
        if len(tracks) > 0:
            status = self.stats_manager.get_status_handler()
            bar = status.start_phase("Adding tracks to playlist", total=len(tracks))
            for track in tracks:
                song = self.sdb.Database.QuerySongs("ID=" + str(track.ID))
                playlist.AddTrack(song.Item)
                bar.update()
            bar.close()
        return playlist

    def _get_native_playlists(self) -> List[MediaMonkeyPlaylist]:
        root = self.sdb.PlaylistByTitle("")
        playlists = []

        # TODO: fix playlist naming
        def get_playlists(parent: MediaMonkeyPlaylist):
            for i in range(len(parent.ChildPlaylists)):
                pl = parent.ChildPlaylists[i]
                playlists.append(pl)
                if len(pl.ChildPlaylists):
                    get_playlists(pl)

        get_playlists(root)
        return playlists

    def _find_native_playlist(self, title: str) -> Optional[MediaMonkeyPlaylist]:
        playlists = []
        root = self.sdb.PlaylistByTitle("")

        def find_in_playlists(parent: MediaMonkeyPlaylist):
            for i in range(len(parent.ChildPlaylists)):
                pl = parent.ChildPlaylists[i]
                if pl.Title.lower() == title.lower():
                    playlists.append(pl)
                if len(pl.ChildPlaylists):
                    find_in_playlists(pl)

        find_in_playlists(root)
        return playlists[0] if playlists else None

    def _convert_native_playlist(self, native_playlist: MediaMonkeyPlaylist) -> Optional[Playlist]:
        playlist = Playlist(native_playlist.Title, parent_name=native_playlist.Title, native_playlist=native_playlist, player=self)
        playlist.is_auto_playlist = native_playlist.isAutoplaylist
        if not playlist.is_auto_playlist:
            for j in range(native_playlist.Tracks.Count):
                playlist.tracks.append(track := self.read_track_metadata(native_playlist.Tracks[j]))
                self.logger.debug(f"Added track {track} to playlist")
        return playlist

    def read_track_metadata(self, track: MediaMonkeyTrack) -> AudioTag:
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

    def _search_native_tracks(self, key: str, value: Union[bool, str]) -> List[MediaMonkeyTrack]:
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
        counter = 0
        status = None
        while not it.EOF:
            results.append(it.Item)
            it.Next()
            counter += 1
            if counter >= 50:
                if not status:
                    status = self.stats_manager.get_status_handler()
                    bar = status.start_phase(f"Collecting tracks from {self.name()}", initial=counter, total=None)
                bar.update()
        bar.close() if status else None
        return results

    def update_rating(self, track: AudioTag, rating: float) -> None:
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {self.get_5star_rating(rating)}")
            return

        self.logger.debug(f"Updating rating for {track} to {self.get_5star_rating(rating)}")
        try:
            song = self.sdb.Database.QuerySongs("ID=" + str(track.ID))
            song.Item.Rating = self.get_native_rating(rating)
            song.Item.UpdateDB()
        except Exception as e:
            self.logger.error(f"Failed to update rating: {e!s}")
            raise

    def _add_track_to_playlist(self, native_playlist: MediaMonkeyPlaylist, native_track: MediaMonkeyTrack) -> None:
        native_playlist.AddTrack(native_track)

    def _remove_track_from_playlist(self, native_playlist: MediaMonkeyPlaylist, native_track: MediaMonkeyTrack) -> None:
        native_playlist.RemoveTrack(native_track)


class PlexPlayer(MediaPlayer):
    maximum_connection_attempts = 3
    rating_maximum = 10
    album_empty_alias = "[Unknown Album]"

    def __init__(self, cache_manager: Optional["CacheManager"] = None, stats_manager: Optional["StatsManager"] = None):
        self.logger = logging.getLogger("PlexSync.PlexPlayer")
        self.abbr = "PP"
        self.account = None
        self.plex_api_connection = None
        self.music_library = None
        super().__init__(cache_manager, stats_manager)

    @staticmethod
    def name() -> str:
        return "PlexPlayer"

    @staticmethod
    def format(track: plexapi.audio.Track) -> str:
        try:
            return " - ".join([track.artist().title, track.album().title, track.title])
        except TypeError:
            return " - ".join([track.artist, track.album, track.title])

    def connect(self, server: str, username: str, password: str = "", token: str = "") -> None:
        self.logger.info(f"Connecting to Plex server {server} as {username}")
        self.account = self._authenticate(server, username, password, token)

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

    def _authenticate(self, server: str, username: str, password: str, token: str) -> MyPlexAccount:
        connection_attempts_left = self.maximum_connection_attempts
        while connection_attempts_left > 0:
            time.sleep(0.5)  # important. Otherwise, the above print statement can be flushed after
            if not password and not token:
                password = getpass.getpass()
            try:
                if password:
                    return MyPlexAccount(username=username, password=password)
                elif token:
                    return MyPlexAccount(username=username, token=token)
            except NotFound:
                print(f"Username {username}, password or token wrong for server {server}.")
                password = ""
                connection_attempts_left -= 1
            except BadRequest as error:
                self.logger.warning(f"Failed to connect: {error}")
                connection_attempts_left -= 1

        self.logger.error(f"Exiting after {self.maximum_connection_attempts} failed attempts ...")
        exit(1)

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

    def _create_native_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[PlexPlaylist]:
        if not tracks:
            return None
        plex_tracks = []
        self.logger.info("Creating playlist {title} with {len(tracks)} tracks")
        for track in tracks:
            try:
                matches = self._search_native_tracks("id", track.ID)
                if matches:
                    plex_tracks.append(matches[0])
                else:
                    self.logger.warning(f"No match found for track ID: {track.ID}")
            except Exception as e:
                self.logger.error(f"Failed to search for track {track.ID}: {e!s}")

        if plex_tracks:
            return self.plex_api_connection.createPlaylist(title=title, items=plex_tracks)
        self.logger.warning("No tracks found to create playlist")
        return None

    def _get_native_playlists(self) -> List[PlexPlaylist]:
        return [pl for pl in self.plex_api_connection.playlists() if pl.playlistType == "audio"]

    def _find_native_playlist(self, title: str) -> Optional[PlexPlaylist]:
        try:
            return self.plex_api_connection.playlist(title)
        except NotFound:
            return None

    def _convert_native_playlist(self, native_playlist: PlexPlaylist) -> Optional[Playlist]:
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

    def find_playlist(self, title: str) -> Optional[Playlist]:
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

    def _search_native_tracks(self, key: str, value: Union[bool, str]) -> List[PlexTrack]:
        if key == "title":
            matches = self.music_library.searchTracks(title=value)
            self.logger.debug(f"Found tracks for query title={value}")
            return matches
        elif key == "rating":
            if value is True:
                value = "0"
            print(f"Collecting tracks from {self.name()}.  This may take some time for large libraries.")
            matches = self.music_library.searchTracks(**{"track.userRating!": value})
            return matches
        elif key == "id":
            return [self.music_library.fetchItem(int(value))]

        raise KeyError(f"Invalid search mode {key}.")

    def update_rating(self, track: [PlexTrack, AudioTag], rating: float) -> None:
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {self.get_5star_rating(rating)}")
            return

        self.logger.debug(f"Updating rating for {track} to {self.get_5star_rating(rating)}")

        try:
            if isinstance(track, AudioTag):
                song = self._search_native_tracks("id", track.ID)[0]
                song.edit(**{"userRating.value": self.get_native_rating(rating)})
            else:
                track.edit(**{"userRating.value": self.get_native_rating(rating)})
        except Exception as e:
            self.logger.error(f"Failed to update rating using fallback: {e!s}")
            raise
        self.logger.info(f"Successfully updated rating for {track}")

    def _add_track_to_playlist(self, native_playlist: PlexPlaylist, native_track: PlexTrack) -> None:
        native_playlist.addItems(native_track)

    def _remove_track_from_playlist(self, native_playlist: PlexPlaylist, native_track: PlexTrack) -> None:
        native_playlist.removeItem(native_track)


class FileSystemPlayer(MediaPlayer):
    rating_maximum = 10
    album_empty_alias = "[Unknown Album]"
    SEARCH_THRESHOLD = 75
    DEFAULT_RATING_TAG = RatingTag.WINDOWSMEDIAPLAYER
    _audio_files = None

    def __init__(self, cache_manager: Optional["CacheManager"] = None, stats_manager: Optional["StatsManager"] = None):
        self.logger = logging.getLogger("PlexSync.FileSystem")
        self.abbr = "FS"
        super().__init__(cache_manager, stats_manager)

    @staticmethod
    def name() -> str:
        return "FileSystemPlayer"

    def connect(self, **kwargs) -> None:
        """Connect to filesystem music library with additional options."""
        self.path = Path(kwargs.get("path"))
        playlist_path = kwargs.get("playlist_path", None)
        if not self.path.exists():
            raise FileNotFoundError(f"Music directory not found: {self.path}")

        self.logger.info(f"Connected to filesystem music library at {self.path}")
        self.playlist_path = Path(playlist_path) if playlist_path else self.path
        self.playlist_path.mkdir(exist_ok=True)
        self.logger.info(f"Using playlists directory: {self.playlist_path}")

        self.tag_write_strategy = TagWriteStrategy.from_value(kwargs.get("tag_write_strategy"))
        self.standard_tag = RatingTag.from_value(kwargs.get("standard_tag"))
        self.conflict_resolution_strategy = ConflictResolutionStrategy.from_value(kwargs.get("conflict_resolution_strategy"))
        self.tag_priority_order = [RatingTag.from_value(tag) for tag in kwargs.get("tag_priority_order", [])]

        status = None
        if FileSystemPlayer._audio_files is None:
            self._scan_audio_files()
            for file_path in self._audio_files:
                if not status:
                    status = self.stats_manager.get_status_handler()
                    bar = status.start_phase(f"Reading track metadata from {self.name()}", total=len(FileSystemPlayer._audio_files))
                self.read_track_metadata(file_path)
        bar.close() if status else None

    @staticmethod
    def get_5star_rating(popm_rating: Optional[int]) -> Optional[float]:  # noqa
        if popm_rating is None or popm_rating == 0:
            return None
        elif popm_rating <= 22:
            return 0.5
        elif popm_rating <= 63:
            return 1.0
        elif popm_rating <= 95:
            return 1.5
        elif popm_rating <= 127:
            return 2.0
        elif popm_rating <= 159:
            return 2.5
        elif popm_rating <= 191:
            return 3.0
        elif popm_rating <= 223:
            return 3.5
        elif popm_rating <= 239:
            return 4.0
        elif popm_rating <= 252:
            return 4.5
        else:  # 253-255
            return 5.0

    def get_native_rating(self, normed_rating: Optional[float]) -> int:  # noqa
        if normed_rating is None or normed_rating <= 0:
            return 0
        elif normed_rating <= 0.1:
            return 16  # 0.5 stars
        elif normed_rating <= 0.2:
            return 48  # 1.0 star
        elif normed_rating <= 0.3:
            return 80  # 1.5 stars
        elif normed_rating <= 0.4:
            return 112  # 2.0 stars
        elif normed_rating <= 0.5:
            return 144  # 2.5 stars
        elif normed_rating <= 0.6:
            return 176  # 3.0 stars
        elif normed_rating <= 0.7:
            return 208  # 3.5 stars
        elif normed_rating <= 0.8:
            return 232  # 4.0 stars
        elif normed_rating <= 0.9:
            return 247  # 4.5 stars
        else:
            return 255  # 5.0 stars

    def get_normed_rating(self, popm_rating: Optional[int]) -> Optional[float]:  # noqa
        if popm_rating is None or popm_rating == 0:
            return None
        elif popm_rating <= 22:
            return 0.1  # 0.5 stars
        elif popm_rating <= 63:
            return 0.2  # 1.0 star
        elif popm_rating <= 95:
            return 0.3  # 1.5 stars
        elif popm_rating <= 127:
            return 0.4  # 2.0 stars
        elif popm_rating <= 159:
            return 0.5  # 2.5 stars
        elif popm_rating <= 191:
            return 0.6  # 3.0 stars
        elif popm_rating <= 223:
            return 0.7  # 3.5 stars
        elif popm_rating <= 239:
            return 0.8  # 4.0 stars
        elif popm_rating <= 252:
            return 0.9  # 4.5 stars
        else:
            return 1.0  # 5.0 stars

    def _scan_audio_files(self) -> None:
        """Scan directory structure and cache audio files"""
        FileSystemPlayer._audio_files = []
        self.logger.info(f"Scanning {self.path} for audio files...")
        audio_extensions = {".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aac"}

        status = self.stats_manager.get_status_handler() if self.stats_manager else None
        bar = status.start_phase(f"Collecting tracks from {self.name()}", total=None) if status else None

        for file_path in self.path.rglob("*"):
            if file_path.suffix.lower() in audio_extensions:
                FileSystemPlayer._audio_files.append(file_path)
                bar.update()
        bar.close()
        self.logger.info(f"Found {len(FileSystemPlayer._audio_files)} audio files")

    def read_track_metadata(self, file_path: Union[Path, str]) -> AudioTag:
        """Retrieve metadata from cache or read from file"""
        str_path = str(file_path)

        # Use CacheManager's metadata_cache with force_enable
        if self.cache_manager:
            cached = self.cache_manager.get_metadata(self.name(), str_path, force_enable=True)
            if cached:
                return cached

        # Read metadata from file
        try:
            audio_file = mutagen.File(file_path)
            if not audio_file:
                raise ValueError(f"Unsupported audio format: {file_path}")

            if hasattr(audio_file, "tags") and audio_file.tags:
                # For MP3 (ID3 tags)
                tags = audio_file.tags
                album = tags.get("TALB", [""])[0]
                artist = tags.get("TPE1", [""])[0]
                title = tags.get("TIT2", [""])[0]
                track_number = tags.get("TRCK", [""])[0]

                # Handle multiple POPM and TXXX:RATING tags
                rating_tags = {key: value for key, value in tags.items() if key.startswith("POPM") or key == "TXXX:Rating"}
                if rating_tags:
                    # magic happens here
                    pass
                else:
                    rating = 0
            elif hasattr(audio_file, "get"):
                # For FLAC, OGG, etc.
                album = audio_file.get("album", [""])[0]
                artist = audio_file.get("artist", [""])[0]
                title = audio_file.get("title", [""])[0]
                track_number = audio_file.get("tracknumber", [""])[0]
                rating = audio_file.get("rating", [""])[0]
            else:
                return None

            tag = AudioTag(
                artist=artist,
                album=album,
                title=title,
                file_path=str_path,
                rating=self.get_normed_rating(rating),
                ID=str_path,
                track=int(track_number.split("/")[0]) if track_number else 1,
            )

            if self.cache_manager:
                self.cache_manager.set_metadata(self.name(), tag.ID, tag, force_enable=True)

            return tag
        except Exception as e:
            self.logger.error(f"Error reading metadata from {file_path}: {e}")
            return None

    def _resolve_conflicting_ratings(self, ratings: dict) -> float:
        """Resolve conflicting ratings based on the configured strategy."""
        if self.conflict_resolution_strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER:
            for tag in self.tag_priority_order:
                if tag in ratings:
                    return ratings[tag]
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.HIGHEST:
            return max(ratings.values())
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.LOWEST:
            return min(ratings.values())
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.AVERAGE:
            return sum(ratings.values()) / len(ratings)
        return 0

    def _write_rating_tags(self, audio_file, rating: float) -> None:
        """Write rating tags to the audio file based on the configured strategy."""
        if self.tag_write_strategy == TagWriteStrategy.WRITE_ALL:
            for tag in self.tag_priority_order:
                audio_file[tag] = rating
        elif self.tag_write_strategy == TagWriteStrategy.WRITE_EXISTING:
            existing_tags = {tag: audio_file[tag] for tag in self.tag_priority_order if tag in audio_file}
            if existing_tags:
                for tag in existing_tags:
                    audio_file[tag] = rating
            else:
                audio_file[self.standard_tag] = rating
        elif self.tag_write_strategy == TagWriteStrategy.WRITE_STANDARD:
            audio_file[self.standard_tag] = rating
        elif self.tag_write_strategy == TagWriteStrategy.OVERWRITE_STANDARD:
            for tag in list(audio_file.keys()):
                if tag.startswith("POPM") or tag == "TXXX:Rating":
                    del audio_file[tag]
            audio_file[self.standard_tag] = rating
        audio_file.save()

    def _search_native_tracks(self) -> None:
        # you can't search for tracks in a filesystem, so they need converted to AudioTags first
        raise NotImplementedError("Search not implemented for FileSystemPlayer")

    def search_tracks(self, key: str, value: Union[bool, str]) -> List[AudioTag]:
        """Search for audio files matching criteria"""

        self.logger.debug(f"Searching tracks with {key}={value}")
        if not value:
            self.logger.error("Search value cannot be empty")
            raise ValueError("value can not be empty.")

        tracks = []

        if key == "id":
            tracks = [self.cache_manager.metadata_cache.get_metadata(self.name(), value, force_enable=True)]
        elif key == "title":
            mask = self.cache_manager.metadata_cache.cache[key].apply(lambda x: fuzz.ratio(str(x).lower(), str(value).lower()) >= self.SEARCH_THRESHOLD)
            tracks = self.cache_manager.get_tracks_by_filter(mask)
        elif key == "rating":
            if value is True:
                value = 0
            mask = (
                (self.cache_manager.metadata_cache.cache["player_name"] == self.name())
                & (self.cache_manager.metadata_cache.cache["rating"].notna())
                & (self.cache_manager.metadata_cache.cache["rating"] > float(value))
            )
            tracks = self.cache_manager.get_tracks_by_filter(mask)

        self.logger.debug(f"Found {len(tracks)} tracks for {key}={value}")

        return tracks

    def _match_rating(self, rating: float, value: Union[bool, str]) -> bool:
        """Check if a rating matches the given criteria"""
        if value is True:
            return rating > 0
        if isinstance(value, str):
            op, threshold = value[:2].strip(), float(value[2:].strip())
            return eval(f"{rating} {op} {threshold}")
        return False

    def update_rating(self, track: AudioTag, rating: float) -> None:
        """Update rating for a track"""
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {self.get_5star_rating(rating)}")
            return

        self.logger.debug(f"Updating rating for {track} to {self.get_5star_rating(rating)}")
        file_path = Path(track.file_path)
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return

        try:
            audio_file = mutagen.File(file_path, easy=False)
            if not audio_file:
                raise ValueError(f"Unsupported audio format: {file_path}")

            # Convert normalized rating to native scale for each format
            native_rating = self.get_native_rating(rating)

            if isinstance(audio_file, mutagen.mp3.MP3):
                # Handle MP3 files
                from mutagen.id3 import POPM

                if not audio_file.tags:
                    audio_file.add_tags()
                # Convert to 0-255 for POPM
                popm_rating = int(min(255, native_rating / self.rating_maximum * 255))
                audio_file.tags.add(POPM(email="plex-music-rating-sync", rating=popm_rating))

            elif isinstance(audio_file, mutagen.flac.FLAC) or isinstance(audio_file, mutagen.oggvorbis.OggVorbis):
                # Handle FLAC/OGG files
                audio_file["RATING"] = [str(int(native_rating))]

            elif isinstance(audio_file, mutagen.mp4.MP4):
                # Handle M4A/MP4 files
                # iTunes-style 0-100 rating
                itunes_rating = int(native_rating / self.rating_maximum * 100)
                audio_file["----:com.apple.iTunes:RATING"] = [str(itunes_rating).encode("utf-8")]

            # Save changes
            audio_file.save()
            self.logger.info(f"Successfully updated rating for {track}")

            # Update cache using CacheManager
            track.rating = rating
            if self.cache_manager:
                self.cache_manager.metadata_cache.set(self.name(), track.ID, track, force_enable=True)

        except Exception as e:
            self.logger.error(f"Failed to update rating: {e}")
            raise

    def _create_native_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Path]:
        """Create a new M3U playlist file"""
        if not tracks:
            return None

        if not self.playlist_path.exists():
            self.playlist_path.mkdir(parents=True, exist_ok=True)

        playlist_file = self.playlist_path / f"{title}.m3u"

        try:
            status = self.stats_manager.get_status_handler() if self.stats_manager else None
            bar = status.start_phase("Creating playlist", total=len(tracks)) if status else None

            with open(playlist_file, "w", encoding="utf-8") as f:
                f.write("#EXTM3U\n")
                for track in tracks:
                    f.write(f"#EXTINF:-1,{track.artist} - {track.title}\n")
                    f.write(f"{track.file_path}\n")
                    bar.update()
            bar.close()
            self.logger.info(f"Created playlist: {playlist_file}")
            return playlist_file
        except Exception as e:
            self.logger.error(f"Failed to create playlist: {e}")
            return None

    def _get_native_playlists(self) -> List[Path]:
        """Get all M3U playlists in the playlist directory"""
        if not self.playlist_path or not self.playlist_path.exists():
            return []

        return list(self.playlist_path.glob("*.m3u"))

    def _find_native_playlist(self, title: str) -> Optional[Path]:
        """Find a playlist by title"""
        if not self.playlist_path or not self.playlist_path.exists():
            return None

        playlist_file = self.playlist_path / f"{title}.m3u"
        return playlist_file if playlist_file.exists() else None

    def _convert_native_playlist(self, native_playlist: Path) -> Optional[Playlist]:
        """Convert a playlist file to a Playlist object"""
        if not native_playlist.exists():
            return None

        playlist = Playlist(native_playlist.stem, native_playlist=native_playlist, player=self)
        playlist.is_auto_playlist = False

        try:
            # Parse M3U file
            with open(native_playlist, "r", encoding="utf-8") as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Skip empty lines and comments (except EXTINF)
                if not line or (line.startswith("#") and not line.startswith("#EXTINF")):
                    i += 1
                    continue

                # Handle file paths
                if not line.startswith("#"):
                    track_path = Path(line)
                    if track_path.exists():
                        track = self.read_track_metadata(track_path)
                        playlist.tracks.append(track)
                i += 1

            return playlist

        except Exception as e:
            self.logger.error(f"Error parsing playlist {native_playlist}: {e}")
            return playlist

    def _add_track_to_playlist(self, native_playlist: Path, native_track: Path) -> None:
        """Add a track to a playlist

        :param native_playlist: Path to playlist file
        :param native_track: Path to track file
        """
        if not native_playlist.exists():
            self.logger.error(f"Playlist not found: {native_playlist}")
            return

        track = self.read_track_metadata(native_track)

        try:
            with open(native_playlist, "a", encoding="utf-8") as f:
                f.write(f"#EXTINF:-1,{track.artist} - {track.title}\n")
                f.write(f"{native_track}\n")
        except Exception as e:
            self.logger.error(f"Failed to add track to playlist: {e}")
            raise

    def _remove_track_from_playlist(self, native_playlist: Path, native_track: Path) -> None:
        """Remove a track from a playlist

        :param native_playlist: Path to playlist file
        :param native_track: Path to track file
        """
        if not native_playlist.exists():
            self.logger.error(f"Playlist not found: {native_playlist}")
            return

        try:
            # Read current playlist content
            with open(native_playlist, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Filter out the track and its EXTINF line
            with open(native_playlist, "w", encoding="utf-8") as f:
                i = 0
                while i < len(lines):
                    line = lines[i].strip()

                    # Check if this line is the track to remove
                    if line == str(native_track):
                        # Skip this line and the previous EXTINF line if it exists
                        if i > 0 and lines[i - 1].startswith("#EXTINF"):
                            i += 1
                            continue

                    # Write line to file
                    f.write(lines[i])
                    i += 1

        except Exception as e:
            self.logger.error(f"Failed to remove track from playlist: {e}")
            raise
