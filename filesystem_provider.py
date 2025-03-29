import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from sync_items import AudioTag, Playlist

IGNORED_TAGS = set()  # Class-level set to track ignored tags


class RatingTag(Enum):
    WINDOWSMEDIAPLAYER = ("POPM:Windows Media Player 9 Series", "Windows Media Player")
    MEDIAMONKEY = ("POPM:no@email", "MediaMonkey")
    MUSICBEE = ("POPM:MusicBee", "MusicBee")
    WINAMP = ("POPM:rating@winamp.com", "Winamp")
    TEXT = ("TXXX:RATING", "Text")
    UNKNOWN1 = ("", "Unknown Player1")
    UNKNOWN2 = ("", "Unknown Player2")
    UNKNOWN3 = ("", "Unknown Player3")

    def __init__(self, tag: str, player_name: str):
        self.tag = tag
        self.player_name = player_name

    def __str__(self) -> str:
        if self.name.startswith("UNKNOWN"):
            return f"Unknown Player - {self.tag}"
        return self.player_name

    def __repr__(self) -> str:
        if self.name.startswith("UNKNOWN"):
            return f"Unknown Player - {self.tag}"
        return self.player_name

    @classmethod
    def from_value(self, value: Optional[str]) -> Optional["RatingTag"]:
        """Convert a tag identifier to a RatingTag."""
        if value is None or value in IGNORED_TAGS:
            return None

        for item in self:
            if item.tag == value or item.player_name == value:
                return item

        if value.startswith("POPM:"):
            return self._handle_unknown_tag(value)

        raise ValueError(f"Invalid RatingTag: {value}")

    @classmethod
    def _handle_unknown_tag(self, value: str) -> Optional["RatingTag"]:
        """Handle unknown but valid tags by assigning them to unused RatingTag entries."""
        unused_tags = [tag for tag in self if not tag.tag]
        if unused_tags:
            tag_to_use = unused_tags[0]
            tag_to_use.tag = value
            tag_to_use.player_name = f"{value}"
            return tag_to_use

        unknown_tags = [tag for tag in self if tag.name.startswith("UNKNOWN")]
        print(f"\nDiscovered unknown media player: {value}. Only {len(unknown_tags)} nonstandard players are supported.")
        while True:
            print("Options:")
            for idx, tag in enumerate(unknown_tags, start=1):
                print(f"{idx}. Delete {tag}")
            print(f"{len(unknown_tags) + 1}. Ignore this tag")
            choice = input(f"Enter your choice (1-{len(unknown_tags) + 1}): ").strip()
            if choice.isdigit():
                choice = int(choice)
                if 1 <= choice <= len(unknown_tags):
                    tag_to_overwrite = unknown_tags[choice - 1]
                    IGNORED_TAGS.add(tag_to_overwrite.tag)
                    print(f"Ignoring the unknown tag: {tag_to_overwrite.tag}")
                    tag_to_overwrite.tag = value
                    tag_to_overwrite.player_name = value
                    return tag_to_overwrite
                elif choice == len(unknown_tags) + 1:
                    print(f"Ignoring the unknown tag: {value}")
                    IGNORED_TAGS.add(value)
                    return None
            print("Invalid choice. Please enter a valid number.")

    @classmethod
    def resolve_tags(self, values: Optional[Union[str, List[str]]]) -> Union[Optional["RatingTag"], List["RatingTag"]]:
        """Resolve a single value, a list of values, or None to RatingTag(s)."""
        if values is None:
            return None
        if isinstance(values, str):
            return self.from_value(values)
        if isinstance(values, list):
            return [self.from_value(value) for value in values if value]
        raise ValueError("Invalid input type for resolve_tags. Expected str, list, or None.")


class TagWriteStrategy(Enum):
    WRITE_ALL = ("write_all", "Update ratings for all discovered media players.")
    WRITE_EXISTING = ("write_existing", "Only update ratings for media players in each file; fallback to default player if none exist.")
    WRITE_STANDARD = ("write_standard", "Update ratings only for the default player; do not remove other ratings.")
    OVERWRITE_STANDARD = ("overwrite_standard", "Update ratings only for the default player and delete all other ratings.")

    def __init__(self, value: str, description: str):
        self._value_ = value
        self.description = description

    @classmethod
    def from_value(self, value: Optional[str]) -> Optional["TagWriteStrategy"]:
        if value is None:
            return None
        for item in self:
            if item.value == value:
                return item
        raise ValueError(f"Invalid TagWriteStrategy: {value}")

    def requires_standard_tag(self) -> bool:
        """Check if the strategy requires a standard tag."""
        return self in {TagWriteStrategy.WRITE_STANDARD, TagWriteStrategy.OVERWRITE_STANDARD, TagWriteStrategy.WRITE_EXISTING}

    def __str__(self) -> str:
        """Return a friendly name followed by the description."""
        friendly_name = self.name.replace("_", " ").title()
        return f"{friendly_name:<20} : {self.description}"


class ConflictResolutionStrategy(Enum):
    PRIORITIZED_ORDER = ("prioritized_order", "Prioritized order of media players.")
    HIGHEST = ("highest", "Use the highest rating.")
    LOWEST = ("lowest", "Use the lowest rating among.")
    AVERAGE = ("average", "Use the average of all ratings.")

    def __init__(self, value: str, description: str):
        self._value_ = value
        self.description = description

    @classmethod
    def from_value(self, value: Optional[str]) -> Optional["ConflictResolutionStrategy"]:
        if value is None:
            return None
        for item in self:
            if item.value == value:
                return item
        raise ValueError(f"Invalid ConflictResolutionStrategy: {value}")

    def description(self) -> str:
        """Return the description of the conflict resolution strategy."""
        return self.description


class FileSystemAdapter:
    """Adapter class for handling filesystem operations for audio files and playlists."""

    def __init__(self, path: Path, playlist_path: Optional[Path] = None):
        from manager import manager

        self.mgr = manager
        self.path = path
        self.playlist_path = playlist_path or path
        self.logger = logging.getLogger("PlexSync.FileSystemAdapter")
        self._audio_files = []

    def get_audio_files(self) -> List[Path]:
        """Get the list of audio files, scanning if needed"""
        if not self._audio_files:
            return self.scan_audio_files()
        return self._audio_files

    # ======= CONNECTION / INITIALIZATION =======

    def connect(self) -> None:
        """Initialize any necessary resources or caches."""
        raise NotImplementedError

    def scan_audio_files(self) -> List[Path]:
        """Scan directory structure and cache audio files"""
        self._audio_files = []
        self.logger.info(f"Scanning {self.path} for audio files...")
        audio_extensions = {".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aac"}

        bar = self.mgr.status.start_phase("Collecting audio files", total=None)

        for file_path in self.path.rglob("*"):
            if file_path.suffix.lower() in audio_extensions:
                self._audio_files.append(file_path)
                if bar:
                    bar.update()
        if bar:
            bar.close()
        self.logger.info(f"Found {len(self._audio_files)} audio files")
        return self._audio_files

    def scan_playlists(self) -> List[Path]:
        """Scan and list all playlist files in the directory."""
        raise NotImplementedError

    # ======= TRACK OPERATIONS =======

    def read_track(self, file_path: Path) -> Optional[dict]:
        raise NotImplementedError

    def save_track(self, file_path: Path, metadata: dict) -> None:
        raise NotImplementedError

    # ======= PLAYLIST OPERATIONS =======

    def read_playlist(self, playlist_path: Path) -> Optional[List[Path]]:
        raise NotImplementedError

    def save_playlist(self, title: str, track_paths: List[Path]) -> Path:
        raise NotImplementedError

    def delete_playlist(self, playlist_path: Path) -> None:
        raise NotImplementedError


class FileSystemMetadataMapper:
    """Maps unstructured filesystem metadata into structured data and manages interactions with FileSystemPlayer."""

    def __init__(self):
        from manager import manager

        self.mgr = manager
        self.logger = logging.getLogger("PlexSync.FileSystemMetadataMapper")
        self.file_system_adapter = FileSystemAdapter(path=self.mgr.config.path, playlist_path=self.mgr.config.playlist_path)
        self.indexed_tracks: Dict[str, AudioTag] = {}  # Metadata cache
        self.indexed_playlists: Dict[str, Playlist] = {}  # Playlist cache
        self.conflicts = []  # Store unresolved rating conflicts

        # Configurable settings
        self.tag_write_strategy: Optional[TagWriteStrategy] = None
        self.standard_tag: Optional[RatingTag] = None
        self.conflict_resolution_strategy: Optional[ConflictResolutionStrategy] = None
        self.tag_priority_order: List[RatingTag] = []
        self.delete_ignored_tags = False

    ### **🔗 CONNECTION & INDEXING**

    def connect(self) -> None:
        """Initializes filesystem adapter and pre-loads metadata."""
        raise NotImplementedError

    def index_metadata(self) -> None:
        """Scans filesystem and builds structured metadata index for tracks & playlists."""
        raise NotImplementedError

    ### **🎵 TRACK MANAGEMENT**

    def read_track_metadata(self, file_path: Path) -> AudioTag:
        """Reads raw metadata from an audio file and converts it into an `AudioTag`."""
        raise NotImplementedError

    def update_rating(self, track: AudioTag, rating: float) -> None:
        """Updates the rating of a track, resolving conflicts before writing to the file."""
        raise NotImplementedError

    def search_tracks(self, key: str, value: Union[str, float]) -> List[AudioTag]:
        """Searches for tracks in the indexed metadata (e.g., title, rating)."""
        raise NotImplementedError

    ### **📋 PLAYLIST MANAGEMENT**

    def create_playlist(self, title: str, tracks: List[AudioTag]) -> Playlist:
        """Creates a new playlist from a list of tracks and saves it to the filesystem."""
        raise NotImplementedError

    def get_playlists(self) -> List[Playlist]:
        """Retrieves all playlists from the indexed metadata."""
        raise NotImplementedError

    def find_playlist(self, title: str) -> Optional[Playlist]:
        """Finds a specific playlist by title."""
        raise NotImplementedError

    def add_track_to_playlist(self, playlist: Playlist, track: AudioTag) -> None:
        """Adds a track to a playlist."""
        raise NotImplementedError

    def remove_track_from_playlist(self, playlist: Playlist, track: AudioTag) -> None:
        """Removes a track from a playlist."""
        raise NotImplementedError

    ### **⭐ RATING HANDLING**

    def resolve_rating_conflicts(self, ratings: Dict[str, int]) -> Optional[float]:
        """Applies the configured conflict resolution strategy to determine the best rating."""
        raise NotImplementedError

    def normalize_rating(self, raw_rating: int) -> float:
        """Converts a file-specific rating into a normalized scale (0.0 - 1.0)."""
        raise NotImplementedError

    def denormalize_rating(self, normed_rating: float) -> int:
        """Converts a normalized rating back to the file-specific format."""
        raise NotImplementedError

    ### **🛠 CONFIGURATION & SETTINGS**

    def configure_settings(self) -> None:
        """Prompts the user for metadata interpretation settings."""
        raise NotImplementedError

    def save_config(self) -> None:
        """Saves configuration settings to a persistent config file."""
        raise NotImplementedError
