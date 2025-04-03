import abc
import getpass
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import plexapi.audio
import plexapi.playlist
from fuzzywuzzy import fuzz
from plexapi.exceptions import BadRequest, NotFound
from plexapi.myplex import MyPlexAccount

from filesystem_provider import FileSystemProvider, RatingTag
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


class MediaPlayer(abc.ABC):
    album_empty_alias = ""
    dry_run = False
    rating_maximum = 5
    abbr = None

    def __init__(self):
        from manager import manager

        self.mgr = manager

    def __hash__(self) -> int:
        return hash(self.name().lower())

    def __eq__(self, other: "MediaPlayer") -> bool:
        if not isinstance(other, type(self)):
            return NotImplementedError
        return other.name().lower() == self.name().lower()

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """The name of this media player"""
        return ""

    def album_empty(self, album: str) -> bool:
        if not isinstance(album, str):
            return False
        return album == self.album_empty_alias

    def connect(self, *args, **kwargs) -> None:
        return NotImplemented

    @abc.abstractmethod
    def _create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[NativePlaylist]:
        """Create a native playlist in the player"""

    @abc.abstractmethod
    def _get_playlists(self) -> NativePlaylist:
        """Get all native playlists from player"""

    @abc.abstractmethod
    def _find_playlist(self, title: str) -> NativePlaylist:
        """Find native playlist by title"""

    @abc.abstractmethod
    def _search_tracks(self, key: str, value: Union[bool, str]) -> List[NativeTrack]:
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

        tracks = self._search_tracks(key, value)

        self.logger.debug(f"Found {len(tracks)} tracks for {key}={value}")
        return tracks

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
                bar = self.mgr.status.start_phase("Syncing playlist updates", total=len(updates))
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
        native_pls = self._get_playlists()
        self.logger.debug(f"Found {len(native_pls)} native playlists")

        playlists = []
        for native_pl in native_pls:
            playlist = self._convert_playlist(native_pl)
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
        native_pl = self._find_playlist(title)
        if native_pl:
            self.logger.debug(f"Found playlist '{title}'")
            return self._convert_playlist(native_pl)

        self.logger.info(f"Playlist '{title}' not found")
        return None

    @staticmethod
    def get_5star_rating(rating: float) -> float:
        return rating * 5 if rating else 0.0

    def get_native_rating(self, normed_rating: float) -> float:
        return normed_rating * self.rating_maximum

    def get_normed_rating(self, rating: Optional[float]) -> float:
        return None if rating is None else max(0, rating) / self.rating_maximum

    @abc.abstractmethod
    def _read_track_metadata(self, track: NativeTrack) -> AudioTag:
        """Reads the metadata of a track."""

    @abc.abstractmethod
    def _convert_playlist(self, native_playlist: NativePlaylist) -> Optional[Playlist]:
        """Convert native playlist to Playlist object"""

    @abc.abstractmethod
    def update_rating(self, track: AudioTag, rating: float) -> None:
        """Updates the rating of the track, unless in dry run"""

    def update_playlist(self, native_playlist: NativePlaylist, track: AudioTag, present: bool) -> None:
        """Updates the native playlist by adding or removing a track
        :param native_playlist: Native playlist object
        :param track: Track to add/remove
        :param present: True to add, False to remove
        """
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would {'add' if present else 'remove'} track from playlist")
            return

        matches = self._search_tracks("id", track.ID)
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

    def __init__(self):
        self.logger = logging.getLogger("PlexSync.MediaMonkey")
        self.sdb = None
        self.abbr = "MM"
        super().__init__()

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

    def _create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[MediaMonkeyPlaylist]:
        if not tracks:
            return None
        self.logger.info(f"Creating playlist {title} with {len(tracks)} tracks")
        playlist = self.sdb.PlaylistByTitle("").CreateChildPlaylist(title)
        if len(tracks) > 0:
            bar = self.mgr.status.start_phase("Adding tracks to playlist", total=len(tracks))
            for track in tracks:
                song = self.sdb.Database.QuerySongs("ID=" + str(track.ID))
                playlist.AddTrack(song.Item)
                bar.update()
            bar.close()
        return playlist

    def _get_playlists(self) -> List[MediaMonkeyPlaylist]:
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

    def _find_playlist(self, title: str) -> Optional[MediaMonkeyPlaylist]:
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

    def _convert_playlist(self, native_playlist: MediaMonkeyPlaylist) -> Optional[Playlist]:
        playlist = Playlist(native_playlist.Title, parent_name=native_playlist.Title, native_playlist=native_playlist, player=self)
        playlist.is_auto_playlist = native_playlist.isAutoplaylist
        if not playlist.is_auto_playlist:
            for j in range(native_playlist.Tracks.Count):
                playlist.tracks.append(track := self._read_track_metadata(native_playlist.Tracks[j]))
                self.logger.debug(f"Added track {track} to playlist")
        return playlist

    def _read_track_metadata(self, track: MediaMonkeyTrack) -> AudioTag:
        cached = self.mgr.cache.get_metadata(self.name(), track.ID)
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
        self.mgr.cache.set_metadata(self.name(), tag.ID, tag)

        return tag

    def _search_tracks(self, key: str, value: Union[bool, str]) -> List[MediaMonkeyTrack]:
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
        bar = None
        while not it.EOF:
            results.append(self._read_track_metadata(it.Item))
            it.Next()
            counter += 1
            if counter >= 50:
                if not bar:
                    bar = self.mgr.status.start_phase(f"Collecting tracks from {self.name()}", initial=counter, total=None)
                bar.update()
        bar.close() if bar else None
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

    def __init__(self):
        self.logger = logging.getLogger("PlexSync.PlexPlayer")
        self.abbr = "PP"
        self.account = None
        self.plex_api_connection = None
        self.music_library = None
        super().__init__()

    @staticmethod
    def name() -> str:
        return "PlexPlayer"

    @staticmethod
    def format(track: plexapi.audio.Track) -> str:
        try:
            return " - ".join([track.artist().title, track.album().title, track.title])
        except TypeError:
            return " - ".join([track.artist, track.album, track.title])

    def connect(self) -> None:
        server = self.mgr.config.server
        username = self.mgr.config.username
        password = self.mgr.config.passwd
        token = self.mgr.config.token
        if not self.mgr.config.token and not self.mgr.config.passwd:
            self.logger.error("Plex token or password is required for Plex player")
            raise
        if not self.mgr.config.server or not self.mgr.config.username:
            self.logger.error("Plex server and username are required for Plex player")
            raise

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

    def _read_track_metadata(self, track: plexapi.audio.Track) -> AudioTag:
        return AudioTag(
            artist=track.grandparentTitle,
            album=track.parentTitle,
            title=track.title,
            file_path=track.locations[0],
            rating=self.get_normed_rating(track.userRating),
            ID=track.key.split("/")[-1] if "/" in track.key else track.key,
            track=track.index,
        )

    def _create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[PlexPlaylist]:
        if not tracks:
            return None
        plex_tracks = []
        self.logger.info("Creating playlist {title} with {len(tracks)} tracks")
        for track in tracks:
            try:
                matches = self._search_tracks("id", track.ID)
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

    def _get_playlists(self) -> List[PlexPlaylist]:
        return [pl for pl in self.plex_api_connection.playlists() if pl.playlistType == "audio"]

    def _find_playlist(self, title: str) -> Optional[PlexPlaylist]:
        try:
            return self.plex_api_connection.playlist(title)
        except NotFound:
            return None

    def _convert_playlist(self, native_playlist: PlexPlaylist) -> Optional[Playlist]:
        playlist = Playlist(native_playlist.title, native_playlist=native_playlist, player=self)
        playlist.is_auto_playlist = native_playlist.smart
        if not playlist.is_auto_playlist:
            for item in native_playlist.items():
                playlist.tracks.append(self._read_track_metadata(item))
        return playlist

    def read_playlists(self) -> List[Playlist]:
        """Read all playlists from the Plex server and convert them to internal Playlist objects."""
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
                        playlist.tracks.append(self._read_track_metadata(item))

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
                    playlist.tracks.append(self._read_track_metadata(item))
            return playlist
        except NotFound:
            self.logger.debug(f"Playlist {title} not found")
            return None

    def _search_tracks(self, key: str, value: Union[bool, str], return_native: bool = False) -> List[PlexTrack]:
        if key == "title":
            matches = self.music_library.searchTracks(title=value)
            self.logger.debug(f"Found tracks for query title={value}")
        elif key == "rating":
            if value is True:
                value = "0"
            print(f"Collecting tracks from {self.name()}.  This may take some time for large libraries.")
            matches = self.music_library.searchTracks(**{"track.userRating!": value})
        elif key == "id":
            matches = [self.music_library.fetchItem(int(value))]
        else:
            raise KeyError(f"Invalid search mode {key}.")
        if return_native:
            return matches

        tracks = []
        bar = None
        if len(matches) >= 500:
            bar = self.mgr.status.start_phase(f"Reading track metadata from {self.name()}", total=len(matches))

        for match in matches:
            bar.update() if bar else None
            track = self._read_track_metadata(match)
            tracks.append(track)

        bar.close() if bar else None
        return tracks

    def update_rating(self, track: [PlexTrack, AudioTag], rating: float) -> None:
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {self.get_5star_rating(rating)}")
            return

        self.logger.debug(f"Updating rating for {track} to {self.get_5star_rating(rating)}")

        try:
            if isinstance(track, AudioTag):
                song = self._search_tracks("id", track.ID, return_native=True)[0]
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
    _rating_map = [
        (0.0, 0),
        (0.1, 13),
        (0.2, 32),
        (0.3, 54),
        (0.4, 64),
        (0.5, 118),
        (0.6, 128),
        (0.7, 186),
        (0.8, 196),
        (0.9, 242),
        (1.0, 255),
    ]

    def __init__(self):
        self.fsp = None
        self.logger = logging.getLogger("PlexSync.FileSystem")
        self.abbr = "FS"
        super().__init__()

    @staticmethod
    def name() -> str:
        return "FileSystemPlayer"

    def connect(self) -> None:
        """Connect to filesystem music library with additional options."""
        self.fsp = FileSystemProvider()

        self.fsp.scan_audio_files()
        if "playlists" in self.mgr.config.sync:
            self.fsp.scan_playlist_files()

        bar = self.mgr.status.start_phase(f"Reading track metadata from {self.name()}", total=len(self.fsp._audio_files))
        for file_path in self.fsp._audio_files:
            self._read_track_metadata(file_path)
            bar.update()
        bar.close()

        finalizing_tracks = self.fsp.finalize_scan()
        bar = self.mgr.status.start_phase(f"Finalizing track indexing from {self.name()}", total=len(finalizing_tracks)) if finalizing_tracks else None
        for track in finalizing_tracks:
            self.mgr.cache.set_metadata(self.name(), track.ID, track)
            bar.update()
        bar.close() if bar else None
        # TODO: add playlist scanning

    @staticmethod
    def get_native_rating(normed_rating: Optional[float]) -> int:
        if normed_rating is None or normed_rating <= 0:
            return 0

        best_diff = float("inf")
        best_popm = 0

        for normed, popm in FileSystemPlayer._rating_map:
            diff = abs(normed_rating - normed)
            if diff < best_diff:
                best_diff = diff
                best_popm = popm

        return best_popm

    @staticmethod
    def get_normed_rating(popm_rating: Optional[int]) -> Optional[float]:
        if popm_rating is None or popm_rating == 0:
            return 0

        best_diff = float("inf")
        best_normed = 0.0

        for normed, popm in FileSystemPlayer._rating_map:
            diff = abs(popm_rating - popm)
            if diff < best_diff:
                best_diff = diff
                best_normed = normed

        return best_normed

    def _read_track_metadata(self, file_path: Union[Path, str]) -> AudioTag:
        """Retrieve metadata from cache or read from file."""
        str_path = str(file_path)

        cached = self.mgr.cache.get_metadata(self.name(), str_path, force_enable=True)
        if cached:
            self.logger.debug(f"Cache hit for {file_path}")
            return cached

        tag = self.fsp.read_metadata_from_file(file_path)
        self.mgr.cache.set_metadata(self.name(), tag.ID, tag, force_enable=True) if tag else None
        return tag

    def _search_tracks(self, key: str, value: Union[bool, str]) -> List[AudioTag]:
        """Search for audio files matching criteria"""

        tracks = []

        if key == "id":
            tracks = [self.mgr.cache.metadata_cache.get_metadata(self.name(), value, force_enable=True)]
        elif key == "title":
            mask = self.mgr.cache.metadata_cache.cache[key].apply(lambda x: fuzz.ratio(str(x).lower(), str(value).lower()) >= self.SEARCH_THRESHOLD)
            tracks = self.mgr.cache.get_tracks_by_filter(mask)
        elif key == "rating":
            if value is True:
                value = 0
            mask = (
                (self.mgr.cache.metadata_cache.cache["player_name"] == self.name())
                & (self.mgr.cache.metadata_cache.cache["rating"].notna())
                & (self.mgr.cache.metadata_cache.cache["rating"] > float(value))
            )
            tracks = self.mgr.cache.get_tracks_by_filter(mask)

        self.logger.debug(f"Found {len(tracks)} tracks for {key}={value}")

        return tracks

    def update_rating(self, track: AudioTag, rating: float) -> None:
        """Update rating for a track."""
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {self.get_5star_rating(rating)}")
            return

        self.logger.debug(f"Updating rating for {track} to {self.get_5star_rating(rating)}")
        try:
            self.fsp.update_metadata_in_file(file_path=track.file_path, rating=self.get_native_rating(rating))  # Delegates to update_metadata_in_file
            track.rating = rating
            self.mgr.cache.set_metadata(self.name(), track.ID, track, force_enable=True)
            self.logger.info(f"Successfully updated rating for {track}")
        except Exception as e:
            self.logger.error(f"Failed to update rating: {e}")
            raise

    def _create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Path]:
        """Create a new M3U playlist file"""
        if not tracks:
            self.logger.warning(f"No tracks provided to create playlist '{title}'")
            return None

        if not self.playlist_path.exists():
            self.playlist_path.mkdir(parents=True, exist_ok=True)

        playlist_file = self.playlist_path / f"{title}.m3u"

        try:
            bar = self.mgr.status.start_phase(f"Creating playlist '{title}'", total=len(tracks))

            with open(playlist_file, "w", encoding="utf-8") as f:
                f.write("#EXTM3U\n")
                for track in tracks:
                    f.write(f"#EXTINF:-1,{track.artist} - {track.title}\n")
                    f.write(f"{track.file_path}\n")
                    if bar:
                        bar.update()
            if bar:
                bar.close()
            self.logger.info(f"Created playlist: {playlist_file}")
            return playlist_file
        except Exception as e:
            self.logger.error(f"Failed to create playlist '{title}': {e}")
            return None

    def _get_playlists(self) -> List[Path]:
        """Get all M3U playlists in the playlist directory"""
        if not self.fsp.playlist_path or not self.fsp.playlist_path.exists():
            return []

        return list(self.fsp.playlist_path.glob("*.m3u"))

    def _find_playlist(self, title: str) -> Optional[Path]:
        """Find a playlist by title"""
        if not self.fsp.playlist_path or not self.fsp.playlist_path.exists():
            return None
        return None

    def _convert_playlist(self, native_playlist: Path) -> Optional[Playlist]:
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
                if not line or (line.startswith("#") and not line.startswith1("#EXTINF")):
                    i += 1
                    continue

                # Handle file paths
                if not line.startswith("#"):
                    track_path = Path(line)
                    if track_path.exists():
                        track = self._read_track_metadata(track_path)
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

        track = self._read_track_metadata(native_track)

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
