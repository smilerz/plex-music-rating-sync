import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from sync_items import AudioTag

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
    WRITE_DEFAULT = ("write_default", "Update ratings only for the default player; do not remove other ratings.")
    OVERWRITE_DEFAULT = ("overwrite_default", "Update ratings only for the default player and delete all other ratings.")

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

    def requires_default_tag(self) -> bool:
        """Check if the strategy requires a default tag."""
        return self in {TagWriteStrategy.WRITE_DEFAULT, TagWriteStrategy.OVERWRITE_DEFAULT, TagWriteStrategy.WRITE_EXISTING}

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


class FileSystemProvider:
    """Adapter class for handling filesystem operations for audio files and playlists."""

    def __init__(self):
        from manager import manager

        self.mgr = manager
        self.logger = logging.getLogger("PlexSync.FileSystemProvider")
        self._audio_files = []
        self._playlist_files = []
        self.conflicts = []

        self.get_normed_rating = None
        self.get_native_rating = None

    # TODO: temporary solution until refactor is complete
    def initialize_settings(self) -> None:
        """Initialize settings for the filesystem provider."""
        from MediaPlayer import FileSystemPlayer

        self.get_normed_rating = FileSystemPlayer.get_normed_rating
        self.get_native_rating = FileSystemPlayer.get_native_rating

        self.conflict_resolution_strategy = ConflictResolutionStrategy.from_value(self.mgr.config.conflict_resolution_strategy)
        self.tag_write_strategy = TagWriteStrategy.from_value(self.mgr.config.tag_write_strategy)
        self.default_tag = RatingTag.from_value(self.mgr.config.default_tag)
        self.tag_priority_order = [RatingTag.from_value(tag) for tag in self.mgr.config.tag_priority_order] if self.mgr.config.tag_priority_order else None
        self.delete_ignored_tags = False

    def scan_audio_files(self) -> List[Path]:
        """Scan directory structure and cache audio files"""
        self.initialize_settings()
        path = self.mgr.config.path

        if not path:
            self.logger.error("Path is required for filesystem player")
            raise
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Music directory not found: {path}")

        self.logger.info(f"Scanning {path} for audio files...")
        audio_extensions = {".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aac"}

        bar = self.mgr.status.start_phase("Collecting audio files", total=None)

        for file_path in self.path.rglob("*"):
            if file_path.suffix.lower() in audio_extensions:
                self._audio_files.append(file_path)
                bar.update()

        bar.close()
        self.logger.info(f"Found {len(self._audio_files)} audio files")
        return self._audio_files

    def scan_playlist_files(self) -> List[Path]:
        """Scan and list all playlist files in the directory."""
        playlist_path = self.mgr.config.playlist_path
        self.playlist_path = Path(playlist_path) if playlist_path else self.path
        self.playlist_path.mkdir(exist_ok=True)
        playlist_extensions = {".m3u", ".m3u8", ".pls"}

        bar = self.mgr.status.start_phase("Collecting audio files", total=None)

        for file_path in self.playlist_path.rglob("*"):
            if file_path.suffix.lower() in playlist_extensions:
                self._audio_files.append(file_path)
                bar.update()
        bar.close()
        self.logger.info(f"Found {len(self._audio_files)} audio files")
        self.logger.info(f"Using playlists directory: {playlist_path}")

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

    def _handle_rating_tags(self, ratings: dict, track: AudioTag) -> Optional[float]:
        """Handle rating tags by validating configuration and routing to resolution or tracking conflicts."""
        if not ratings:
            return None

        # Normalize TXXX:Rating if present
        if "TXXX:Rating" in ratings.keys():
            try:
                ratings["TXXX:Rating"] = self.get_native_rating(float(ratings["TXXX:Rating"]) / 5)
            except ValueError:
                self.logger.warning(f"Invalid TXXX:Rating format: {ratings["TXXX:Rating"]}")
                ratings["TXXX:Rating"] = None

        # Track which tags are used
        for tag in ratings:
            if found_player := RatingTag.from_value(tag):
                found_player = found_player.player_name if isinstance(found_player, Enum) else found_player
                self.mgr.stats.increment(f"FileSystemPlayer::tags_used::{found_player}")

        # Check for conflicts
        unique_ratings = set(ratings.values())
        if len(unique_ratings) == 1:
            # No conflict, return the single score
            return self.get_normed_rating(next(iter(unique_ratings)))

        # Handle conflicts
        return self._resolve_conflicting_ratings(ratings, track)

    def _resolve_conflicting_ratings(self, ratings: dict, track: AudioTag) -> Optional[float]:
        """Resolve conflicting ratings based on the configured strategy."""
        if not self.conflict_resolution_strategy or (self.conflict_resolution_strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER and not self.tag_priority_order):
            # Store conflict if resolution is not possible
            self.conflicts.append({"track": track, "tags": ratings})
            self.mgr.stats.increment("FileSystemPlayer::tag_rating_conflict")
            return None

        # Resolve conflicts based on strategy
        if self.conflict_resolution_strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER:
            for tag in self.tag_priority_order:
                if tag in ratings:
                    return self.get_normed_rating(ratings[tag])
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.HIGHEST:
            return self.get_normed_rating(max(ratings.values()))
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.LOWEST:
            return self.get_normed_rating(min(ratings.values()))
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.AVERAGE:
            average_rating = sum(ratings.values()) / len(ratings)
            return self.get_normed_rating(round(average_rating * 2) / 2)  # Round to nearest 0.5-star equivalent
        return None
