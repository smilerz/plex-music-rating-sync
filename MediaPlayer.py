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

from filesystem_provider import FileSystemProvider
from manager import get_manager
from ratings import Rating, RatingScale
from sync_items import AudioTag, Playlist

NativePlaylist = Any
NativeTrack = Any
MediaMonkeyPlaylist = Any  # COM SDBPlaylist object
MediaMonkeyTrack = Any  # COM SDBSongData object
PlexPlaylist = Any  # plexapi.playlist.Playlist
PlexTrack = Any  # plexapi.audio.Track
# TODO: creat PromptManager
# TODO: itunes
# TODO: add mediamonkey 5
# TODO: add updating Album, Artist, Title, Rating, Track, Genre
# TODO: add setting source of record per attribute


class MediaPlayer(abc.ABC):
    album_empty_alias = ""
    dry_run = False
    rating_scale = None

    def __init__(self):
        mgr = get_manager()
        self.config_mgr = mgr.get_config_manager()
        self.cache_mgr = mgr.get_cache_manager()
        self.status_mgr = mgr.get_status_manager()

    def __str__(self) -> int:
        return self.name()

    def __hash__(self) -> int:
        return hash(self.name().lower())

    def __eq__(self, other: "MediaPlayer") -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return other.name().lower() == self.name().lower()

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """The name of this media player"""
        return NotImplemented

    def album_empty(self, album: str) -> bool:
        if not isinstance(album, str):
            return False
        return album == self.album_empty_alias

    @abc.abstractmethod
    def connect(self, *args, **kwargs) -> None:
        """Connect to the media player."""

    @abc.abstractmethod
    def _create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Playlist]:
        """Create a native playlist in the player"""

    # TODO: replace with search_playlists
    def _get_playlists(self) -> List[NativePlaylist]:
        """Get all native playlists from player"""
        return None

    @abc.abstractmethod
    def _search_playlists(self, key: str, value: Union[str, bool], return_native: bool = False) -> List[Playlist]:
        """Search for playlists by key: 'all', 'title', or 'id'"""

    def search_playlists(self, key: str = "all", value: Union[str, bool] = True, return_native: bool = False) -> List[Playlist]:
        if key not in {"all", "title", "id"}:
            raise ValueError(f"Invalid search key: {key}")
        return self._search_playlists(key, value, return_native)

    @abc.abstractmethod
    def read_playlist_tracks(self, playlist: Playlist) -> None:
        """Read tracks from a native playlist"""

    # TODO: remove
    @abc.abstractmethod
    def _find_playlist(self, title: str, return_native: bool = False) -> Optional[Union[NativePlaylist, Playlist]]:
        """Find native playlist by title"""

    @abc.abstractmethod
    def _search_tracks(self, key: str, value: Union[bool, str], return_native: bool = False) -> List[Union[AudioTag, NativeTrack]]:
        """Search for tracks in the native player format"""

    @abc.abstractmethod
    def _add_track_to_playlist(self, playlist: Union[NativePlaylist, Playlist], track: AudioTag) -> None:
        """Add a track to a native playlist"""

    @abc.abstractmethod
    def _remove_track_from_playlist(self, playlist: Union[NativePlaylist, Playlist], track: AudioTag) -> None:
        """Remove a track from a native playlist"""

    def search_tracks(self, key: str, value: Union[bool, str], track_status: bool = False) -> List[AudioTag]:
        """Search tracks and convert to AudioTag format"""
        self.logger.debug(f"Searching tracks with {key}={value}")
        if not value:
            self.logger.error("Search value cannot be empty")
            raise ValueError("value can not be empty.")

        tracks = self._search_tracks(key, value)
        return tracks

    def sync_playlist(self, playlist: Playlist, updates: List[Tuple[AudioTag, bool]]) -> None:
        """Sync playlist changes to native format"""
        if not updates:
            self.logger.debug("No updates to sync")
            return

        if self.dry_run:
            self.logger.info(f"DRY RUN: Would sync {len(updates)} changes to playlist '{playlist.name}'")
            return

        if len(updates) > 0:
            self.logger.info(f"Syncing {len(updates)} changes to playlist '{playlist.name}'")
            bar = self.status_mgr.start_phase("Syncing playlist updates", total=len(updates))
            for track, present in updates:
                self.update_playlist(playlist, track, present)
                bar.update()
            bar.close()
        self.logger.debug("Playlist sync completed")

    def create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Playlist]:
        """Create a new playlist with the given tracks"""
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would create playlist '{title}' with {len(tracks)} tracks")
            return None

        if not tracks:
            self.logger.warning("No tracks provided for playlist creation")
            return None
        self.logger.info(f"Creating playlist '{title}' with {len(tracks)} tracks")
        playlist = self._create_playlist(title, tracks)
        if playlist:
            self.logger.debug(f"Successfully created playlist '{title}'")
            return playlist
        self.logger.error(f"Failed to create playlist '{title}'")
        return None

    def read_playlists(self) -> List[Playlist]:
        """Read all playlists from the player
        :return: List of Playlist objects
        """
        self.logger.info(f"Reading all playlists from {self.name()}")
        playlists = self._get_playlists()
        self.logger.debug(f"Found {len(playlists)} native playlists")
        return playlists

    def find_playlist(self, title: str) -> Optional[Union[NativePlaylist, Playlist]]:
        """Find playlist by title
        :return: Matching Playlist or None
        """
        if not title:
            self.logger.warning("Find playlist without title")
            return None

        self.logger.debug(f"Searching for playlist '{title}'")
        playlist = self._find_playlist(title)
        if playlist:
            self.logger.debug(f"Found playlist '{title}'")
            return playlist

        self.logger.info(f"Playlist '{title}' not found")
        return None

    @abc.abstractmethod
    def _read_track_metadata(self, track: NativeTrack) -> AudioTag:
        """Reads the metadata of a track."""

    @abc.abstractmethod
    def update_rating(self, track: AudioTag, rating: Rating) -> None:
        """Updates the rating of the track, unless in dry run"""

    def update_playlist(self, playlist: Playlist, track: AudioTag, present: bool) -> None:
        """Updates the playlist by adding or removing a track
        :param playlist: Playlist object
        :param track: Track to add/remove
        :param present: True to add, False to remove
        """
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would {'add' if present else 'remove'} track from playlist")
            return

        self.logger.info(f"{'Adding' if present else 'Removing'} track '{track}' {'to' if present else 'from'} playlist '{playlist.name}'")

        try:
            if present:
                self._add_track_to_playlist(playlist, track)
            else:
                self._remove_track_from_playlist(playlist, track)
            self.logger.debug(f"Successfully {'added' if present else 'removed'} track")
        except Exception as e:
            self.logger.error(f"Failed to {'add' if present else 'remove'} track: {e!s}")
            raise


class MediaMonkey(MediaPlayer):
    rating_scale = RatingScale.ZERO_TO_HUNDRED

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("PlexSync.MediaMonkey")
        self.sdb = None
        self.abbr = "MM"
        self._title_to_id_map = {}

    @staticmethod
    def name() -> str:
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
        if not title or not tracks:
            self.logger.warning("Title or tracks are empty for playlist creation")
            raise ValueError("Title and tracks cannot be empty.")
        nested_playlists = title.split(".")

        current_path = []
        current_playlist = self.sdb.PlaylistByTitle("")  # root

        for part in nested_playlists:
            current_path.append(part)
            path_title = ".".join(current_path)
            existing = self._search_playlists("title", path_title, return_native=True)

            if existing:
                current_playlist = existing
            else:
                current_playlist = current_playlist.CreateChildPlaylist(part)
        # Add tracks to the final playlist
        if tracks:
            bar = self.status_mgr.start_phase(f"Adding tracks to playlist {title}", total=len(tracks))
            for track in tracks:
                try:
                    song = self.sdb.Database.QuerySongs(f"ID={track.ID}").Item
                    if song:
                        current_playlist.AddTrack(song)
                    else:
                        self.logger.warning(f"Track with ID {track.ID} not found in MediaMonkey database")
                except Exception as e:
                    self.logger.error(f"Failed to add track ID {track.ID} to playlist: {e}")
                bar.update()
            bar.close()

        return current_playlist

    def _get_playlists(self) -> List[MediaMonkeyPlaylist]:
        return self._search_playlists("all")

    def read_playlist_tracks(self, playlist: Playlist) -> None:
        """Read tracks from a native playlist"""
        native_playlist = self._search_playlists("title", playlist.name, return_native=True)
        if not playlist.is_auto_playlist:
            bar = None
            for j in range(native_playlist.Tracks.Count):
                if not bar and native_playlist.Tracks.Count > 100:
                    bar = self.status_mgr.start_phase(f"Reading tracks from playlist {playlist.name}", total=native_playlist.Tracks.Count)
                playlist.tracks.append(track := self._read_track_metadata(native_playlist.Tracks[j]))
                if bar:
                    bar.update()
                    self.logger.debug(f"Reading track {track} from playlist {playlist.name}")
            bar.close() if bar else None

    def _find_playlist(self, title: str, return_native: bool = False) -> Optional[MediaMonkeyPlaylist]:
        results = self.search_playlists("title", title)
        return results[0] if results else None

    def _convert_playlist(self, native_playlist: MediaMonkeyPlaylist, title: str) -> Optional[Playlist]:
        """Convert a MediaMonkey native playlist to a standard Playlist object. Also register the title → ID mapping."""
        if not native_playlist or not title:
            self.logger.warning("Invalid playlist conversion request")
            return None

        playlist = Playlist(ID=native_playlist.ID, name=title, player=self)
        playlist.is_auto_playlist = native_playlist.isAutoplaylist
        # TODO: _native will be removed later once all paths use _search_playlists
        playlist._native = native_playlist

        return playlist

    def _register_playlist_title(self, title: str, ID: int) -> None:
        key = title.lower()
        if key in self._title_to_id_map:
            existing = self._title_to_id_map[key]
            if existing != ID:
                raise ValueError(f"Ambiguous playlist title '{title}' maps to multiple IDs: {existing}, {ID}. Please rename one of the playlists.")
        self._title_to_id_map[key] = ID

    def _collect_playlists(self, parent: MediaMonkeyPlaylist, prefix: str = "") -> List[Tuple[MediaMonkeyPlaylist, str]]:
        """Recursively collect playlists and their titles."""
        results = []
        for i in range(len(parent.ChildPlaylists)):
            pl = parent.ChildPlaylists[i]
            title = f"{prefix}.{pl.Title}" if prefix else pl.Title
            self._register_playlist_title(title, pl.ID)
            results.append((pl, title))
            if len(pl.ChildPlaylists):
                results.extend(self._collect_playlists(pl, title))
        return results

    def _search_playlists(self, key: str, value: Union[str, bool], return_native: bool = False) -> List[Union[Playlist, MediaMonkeyPlaylist]]:
        """Search for playlists by 'all', 'title', or 'id' using COM interface."""
        if key not in {"all", "title", "id"}:
            raise ValueError(f"Invalid search key: {key}")

        native_results = []

        if key == "all":
            native_results = self._collect_playlists(self.sdb.PlaylistByTitle(""))

        elif key == "title":
            playlist_id = self._title_to_id_map.get(value.lower())
            if playlist_id:
                return self._search_playlists("id", playlist_id, return_native)

        elif key == "id":
            try:
                pl = self.sdb.PlaylistByID(int(value))
                native_results.append((pl, pl.Title))
            except Exception:
                self.logger.warning(f"Failed to retrieve playlist by ID: {value}")
                return []

        if return_native:
            return [pl for pl, _ in native_results]
        else:
            return [self._convert_playlist(pl, title) for pl, title in native_results]

    def _read_track_metadata(self, track: MediaMonkeyTrack) -> AudioTag:
        cached = self.cache_mgr.get_metadata(self.name(), track.ID)
        if cached is not None:
            return cached

        tag = AudioTag(
            artist=track.Artist.Name,
            album=track.Album.Name,
            title=track.Title,
            file_path=track.Path,
            rating=Rating.try_create(track.Rating, scale=self.rating_scale) or Rating.unrated(),
            ID=track.ID,
            track=track.TrackOrder,
            duration=int(track.SongLength / 1000) if track.SongLength else -1,
        )
        self.cache_mgr.set_metadata(self.name(), tag.ID, tag)

        return tag

    def _search_tracks(self, key: str, value: Union[bool, str], return_native: bool = False) -> List[MediaMonkeyTrack]:
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
            if not return_native and (cached := self.cache_mgr.get_metadata(self.name(), str(value))):
                return [cached]
            query = f"ID = {value}"
        else:
            raise KeyError(f"Invalid search mode {key}.")

        self.logger.debug(f"Executing query [{query}] against {self.name()}")
        it = self.sdb.Database.QuerySongs(query)

        results = []
        counter = 0
        bar = None
        while not it.EOF:
            if return_native:
                results.append(it.Item)
            else:
                results.append(self._read_track_metadata(it.Item))
            it.Next()
            counter += 1
            if counter >= 50 and not bar:
                bar = self.status_mgr.start_phase(f"Collecting tracks from {self.name()}", initial=counter, total=None)
            bar.update() if bar else None
        bar.close() if bar else None
        return results

    def update_rating(self, track: AudioTag, rating: Rating) -> None:
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {rating.to_display()}")
            return

        self.logger.debug(f"Updating rating for {track} to {rating.to_display()}")
        try:
            song = self._search_tracks("id", track.ID, return_native=True)[0]
            song.Rating = rating.to_float(self.rating_scale)
            song.UpdateDB()
        except Exception as e:
            self.logger.error(f"Failed to update rating: {e!s}")
            raise

    def _add_track_to_playlist(self, playlist: Union[MediaMonkeyPlaylist, Playlist], track: AudioTag) -> None:
        """Add a track to a playlist using the Playlist object"""
        native_playlist = self._search_playlists("id", playlist.ID, return_native=True)[0]

        if not native_playlist:
            self.logger.warning(f"Native playlist not found for {playlist.name}")
            return

        matches = self._search_tracks("id", track.ID, return_native=True)
        if not matches:
            self.logger.warning(f"Could not find track for: {track} in {self.name()}")
            return
        native_playlist.AddTrack(matches[0])

    def _remove_track_from_playlist(self, playlist: Union[MediaMonkeyPlaylist, Playlist], track: AudioTag) -> None:
        """Remove a track from a playlist using the Playlist object"""
        native_playlist = self._search_playlists("id", playlist.ID, return_native=True)[0]
        if not native_playlist:
            self.logger.warning(f"Native playlist not found for {playlist.name}")
            return

        matches = self._search_tracks("id", track.ID)
        if not matches:
            self.logger.warning(f"Could not find track for: {track} in {self.name()}")
            return
        native_playlist.RemoveTrack(matches[0])


class PlexPlayer(MediaPlayer):
    maximum_connection_attempts = 3
    rating_scale = RatingScale.ZERO_TO_TEN
    album_empty_alias = "[Unknown Album]"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("PlexSync.PlexPlayer")
        self.abbr = "PP"
        self.account = None
        self.plex_api_connection = None
        self.music_library = None

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
        server = self.config_mgr.server
        username = self.config_mgr.username
        password = self.config_mgr.passwd
        token = self.config_mgr.token
        if not self.config_mgr.token and not self.config_mgr.passwd:
            self.logger.error("Plex token or password is required for Plex player")
            raise
        if not self.config_mgr.server or not self.config_mgr.username:
            self.logger.error("Plex server and username are required for Plex player")
            raise

        self.logger.info(f"Connecting to Plex server {server} as {username}")
        self.account = self._authenticate(server, username, password, token)

        self.logger.info(f"Connecting to remote player {self.name()} on the server {server}")
        try:
            self.plex_api_connection = self.account.resource(server).connect(timeout=5)
            self.logger.info(f"Successfully connected to {server}")
        except NotFound:
            # This also happens if the user is not the owner of the server
            self.logger.error(f"Failed to connect to {self.name()}.")
            exit(1)

        self.logger.info("Looking for music libraries")
        music_libraries = {section.key: section for section in self.plex_api_connection.library.sections() if section.type == "artist"}

        # TODO: offer to save library to config
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
            rating=Rating.try_create(track.userRating, scale=self.rating_scale) or Rating.unrated(),
            ID=track.key.split("/")[-1] if "/" in track.key else track.key,
            track=track.index,
            duration=int(track.duration / 1000) if track.duration else -1,  # Convert ms to seconds
        )

    def _create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[PlexPlaylist]:
        if not tracks:
            return None
        plex_tracks = []
        for track in tracks:
            try:
                matches = self._search_tracks("id", track.ID, return_native=True)
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

    def read_playlist_tracks(self, playlist: Playlist) -> None:
        """Read tracks from a native playlist"""
        native_playlist = self._find_playlist(playlist.name, return_native=True)
        if not native_playlist:
            self.logger.warning(f"Native playlist not found for {playlist.name}")
            return

        bar = None
        if not playlist.is_auto_playlist:
            for item in native_playlist.items():
                if not bar and len(native_playlist.items()) > 100:
                    bar = self.status_mgr.start_phase(f"Reading tracks from playlist {playlist.name}", total=len(native_playlist.items()))
                playlist.tracks.append(self._read_track_metadata(item))
                bar.update() if bar else None
                self.logger.debug(f"Reading track {item} from playlist {playlist.name}") if bar else None
            bar.close() if bar else None

    def _find_playlist(self, title: str, return_native: bool = False) -> Optional[Union[PlexPlaylist, Playlist]]:
        try:
            playlist = self.plex_api_connection.playlist(title)
            return playlist if return_native else self._convert_playlist(playlist)
        except NotFound:
            return None

    def _search_playlists(self, key: str, value: Union[str, bool], return_native: bool = False) -> List[Playlist]:
        raise NotImplementedError("PlexPlayer does not support searching for playlists by key.")

    def _convert_playlist(self, native_playlist: PlexPlaylist) -> Optional[Playlist]:
        ID = native_playlist.key.split("/")[-1] if "/" in native_playlist.key else native_playlist.key
        playlist = Playlist(ID=ID, name=native_playlist.title, player=self)  # TODO add id
        playlist.is_auto_playlist = native_playlist.smart
        playlist._native = native_playlist
        if not playlist.is_auto_playlist:
            for item in native_playlist.items():
                playlist.tracks.append(self._read_track_metadata(item))
        return playlist

    def _get_playlists(self) -> List[PlexPlaylist]:
        """Read all playlists from the Plex server and convert them to internal Playlist objects."""
        self.logger.info(f"Reading playlists from the {self.name()} player")
        playlists = []

        try:
            plex_playlists = self.plex_api_connection.playlists()
            for plex_playlist in plex_playlists:
                if plex_playlist.playlistType != "audio":
                    continue

                playlist = self._convert_playlist(plex_playlist)
                playlists.append(playlist)

            return playlists

        except Exception as e:
            self.logger.error(f"Failed to read playlists: {e!s}")
            return []

    def _search_tracks(self, key: str, value: Union[bool, str], return_native: bool = False) -> List[PlexTrack]:
        if key == "title":
            matches = self.music_library.searchTracks(title=value)
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
            bar = self.status_mgr.start_phase(f"Reading track metadata from {self.name()}", total=len(matches))

        for match in matches:
            bar.update() if bar else None
            track = self._read_track_metadata(match)
            tracks.append(track)

        bar.close() if bar else None
        return tracks

    def update_rating(self, track: AudioTag, rating: Rating) -> None:
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {rating.to_display()}")
            return

        self.logger.debug(f"Updating rating for {track} to {rating.to_display()}")

        try:
            song = self._search_tracks("id", track.ID, return_native=True)[0]
            song.edit(**{"userRating.value": Rating.to_float(rating, scale=self.rating_scale)})
        except Exception as e:
            self.logger.error(f"Failed to update rating using fallback: {e!s}")
            raise
        self.logger.info(f"Successfully updated rating for {track}")

    def _add_track_to_playlist(self, playlist: Union[Playlist, PlexPlaylist], track: AudioTag) -> None:
        """Add a track to a playlist using the Playlist object"""
        if isinstance(playlist, Playlist) and not playlist._native:
            playlist._native = self._find_playlist(playlist.name, return_native=True)
        elif not isinstance(playlist, Playlist):
            playlist._native = self._find_playlist(playlist.name, return_native=True)

        if not playlist._native:
            self.logger.warning(f"Native playlist not found for {playlist.name}")
            return

        try:
            matches = self._search_tracks("id", track.ID, return_native=True)
            if not matches:
                self.logger.warning(f"No match found for track ID: {track.ID}")
                return
            playlist._native.addItems(matches[0])
        except Exception as e:
            self.logger.error(f"Failed to search for track {track.ID}: {e!s}")
            raise

    def _remove_track_from_playlist(self, playlist: Union[PlexPlaylist, Playlist], track: AudioTag) -> None:
        """Remove a track from a playlist using the Playlist object"""
        if isinstance(playlist, Playlist) and not playlist._native:
            playlist._native = self._find_playlist(playlist.name, return_native=True)
        elif not isinstance(playlist, Playlist):
            playlist._native = self._find_playlist(playlist.name, return_native=True)

        if not playlist._native:
            self.logger.warning(f"Native playlist not found for {playlist.name}")
            return

        try:
            matches = self._search_tracks("id", track.ID, return_native=True)
            if not matches:
                self.logger.warning(f"No match found for track ID: {track.ID}")
                return
            playlist._native.removeItem(matches[0])
        except Exception as e:
            self.logger.error(f"Failed to search for track {track.ID}: {e!s}")
            raise


class FileSystemPlayer(MediaPlayer):
    RATING_SCALE = RatingScale.NORMALIZED
    album_empty_alias = "[Unknown Album]"
    SEARCH_THRESHOLD = 75  # Fuzzy search threshold for track title matching

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
        self.fsp.scan_media_files()

        tracks = self.fsp.get_tracks()
        bar = self.status_mgr.start_phase(f"Reading track metadata from {self.name()}", total=len(tracks))
        for file_path in tracks:
            self._read_track_metadata(file_path)
            bar.update()
        bar.close()

        finalizing_tracks = self.fsp.finalize_scan()
        bar = self.status_mgr.start_phase(f"Finalizing track indexing from {self.name()}", total=len(finalizing_tracks)) if finalizing_tracks else None
        for track in finalizing_tracks:
            self.cache_mgr.set_metadata(self.name(), track.ID, track)
            bar.update()
        bar.close() if bar else None

    def _read_track_metadata(self, file_path: Union[Path, str]) -> AudioTag:
        """Retrieve metadata from cache or read from file."""
        str_path = str(file_path)

        cached = self.cache_mgr.get_metadata(self.name(), str_path, force_enable=True)
        if cached:
            self.logger.debug(f"Cache hit for {file_path}")
            return cached

        tag = self.fsp.read_track_metadata(file_path)
        self.cache_mgr.set_metadata(self.name(), tag.ID, tag, force_enable=True) if tag else None
        return tag

    def _search_tracks(self, key: str, value: Union[bool, str]) -> List[AudioTag]:
        """Search for audio files matching criteria"""

        tracks = []
        player_mask = self.cache_mgr.metadata_cache.cache["player_name"] == self.name().lower()

        if key == "id":
            tracks = [self.cache_mgr.get_metadata(self.name(), value, force_enable=True)]
        elif key == "title":
            title_mask = self.cache_mgr.metadata_cache.cache[key].apply(lambda x: fuzz.ratio(str(x).lower(), str(value).lower()) >= self.SEARCH_THRESHOLD)
            tracks = self.cache_mgr.get_tracks_by_filter(player_mask & title_mask)
        elif key == "rating":
            if value is True:
                value = 0
            rating_mask = self.cache_mgr.metadata_cache.cache["rating"] > Rating(value, scale=RatingScale.ZERO_TO_FIVE).to_float(RatingScale.NORMALIZED)
            tracks = self.cache_mgr.get_tracks_by_filter(player_mask & rating_mask)
        return tracks

    def update_rating(self, track: AudioTag, rating: Rating) -> None:
        """Update rating for a track."""
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update rating for {track} to {rating.to_display()})")
            return

        self.fsp.update_track_metadata(file_path=track.ID, rating=rating)
        track.rating = rating
        self.cache_mgr.set_metadata(self.name(), track.ID, track, force_enable=True)
        self.logger.info(f"Successfully updated rating for {track}")

    def _create_playlist(self, title: str, tracks: List[AudioTag]) -> Optional[Playlist]:
        """Create a new M3U playlist file"""
        if not tracks:
            return None
        playlist = self.fsp.create_playlist(title, is_extm3u=True)
        bar = None
        if playlist:
            if len(tracks) > 100:
                bar = self.status_mgr.start_phase(f"Adding tracks to playlist {title}", total=len(tracks))
            for track in tracks:
                self._add_track_to_playlist(playlist, track)
                bar.update() if bar else None
            bar.close() if bar else None
        return playlist

    # TODO: delete once _search_playlists is implemented across all players
    def _find_playlist(self, title: str) -> Optional[Playlist]:
        """Find a playlist by title (legacy compatibility)."""
        results = self.search_playlists("title", title)
        return results[0] if results else None

    # TODO: delete once _search_playlists is implemented across all players
    def _get_playlists(self) -> List[MediaMonkeyPlaylist]:
        return self._search_playlists("all")

    # TODO: replace _find_playlists with search_playlists
    def _search_playlists(self, key: str, value: Optional[Union[str, int]] = None, return_native: bool = False) -> List[Playlist]:
        if key == "all":
            playlists = self.fsp.get_playlists()
        elif key == "title":
            playlists = self.fsp.get_playlists(title=value)
        elif key == "id":
            playlists = self.fsp.get_playlists(path=value)
        else:
            raise ValueError(f"Invalid search key {key}")

        # TODO: remove this once _player has been deprecated
        [setattr(p, "_player", self) for p in playlists]
        return playlists

    def read_playlist_tracks(self, playlist: Playlist) -> None:
        """Read tracks from a native playlist"""
        tracks = self.fsp.get_tracks_from_playlist(playlist.ID)
        bar = None
        if len(tracks) >= 100:
            bar = self.status_mgr.start_phase(f"Reading tracks from playlist {playlist.name}", total=len(tracks))
        for track in tracks:
            playlist.tracks.append(self._read_track_metadata(track))
            bar.update() if bar else None
            self.logger.debug(f"Reading track {track} from playlist {playlist.name}") if bar else None
        bar.close() if bar else None

    def _add_track_to_playlist(self, playlist: Playlist, track: AudioTag) -> None:
        """Add a track to a playlist"""
        if not playlist:
            self.logger.warning("Playlist not found or invalid")
            return
        self.logger.debug(f"Adding track {track.ID} to playlist {playlist.name}")
        self.fsp.add_track_to_playlist(playlist.ID, track, is_extm3u=playlist.is_extm3u)

    def _remove_track_from_playlist(self, playlist: Playlist, track: AudioTag) -> None:
        """Remove a track from a playlist"""
        raise NotImplementedError
        self.fsp.remove_track_from_playlist(playlist, track)
