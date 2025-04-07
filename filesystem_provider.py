import abc
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import mutagen
from mutagen.flac import FLAC
from mutagen.id3 import POPM, TXXX, ID3FileType
from mutagen.oggvorbis import OggVorbis

from sync_items import AudioTag, Playlist


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


class AudioFileManager(abc.ABC):
    @staticmethod
    def get_5star_rating(rating: Optional[float]) -> float:
        """Convert a normalized 0-1 scale rating to a 0-5 scale for display purposes."""
        return round(rating * 5, 1) if rating is not None else 0.0

    @abc.abstractmethod
    def read_tags(self, audio_file: mutagen.FileType) -> dict:
        """
        Read relevant tag data (album, artist, title, normalized rating, etc.)
        Return a dictionary or small data class with the results.
        """
        pass

    @abc.abstractmethod
    def write_tags(self, audio_file: mutagen.FileType, metadata: dict, rating: Optional[float] = None) -> mutagen.FileType:
        """
        Write or update tags (album, artist, rating frames) to the audio_file.
        Return the modified file object.
        """
        pass


class VorbisManager(AudioFileManager):
    def __init__(self):
        self.logger = logging.getLogger("PlexSync.VorbisManager")

    def read_tags(self, audio_file: mutagen.FileType) -> AudioTag:
        """Read metadata and ratings from Vorbis comments."""
        track = AudioTag.from_vorbis(audio_file, audio_file.filename)

        # Read ratings from both FMPS_RATING and RATING
        fmps_rating = self._parse_rating(audio_file.get("FMPS_RATING", [None])[0])
        rating = self._parse_rating(audio_file.get("RATING", [None])[0])

        # Handle conflicts between FMPS_RATING and RATING
        final_rating = self._resolve_rating_conflict(fmps_rating, rating)
        track.rating = final_rating

        return track

    def write_tags(self, audio_file: mutagen.FileType, metadata: dict, rating: Optional[float] = None) -> mutagen.FileType:
        """Write metadata and ratings to Vorbis comments."""
        # Update metadata
        if metadata:
            for key, value in metadata.items():
                if value:
                    audio_file[key] = value

        # Write ratings to both FMPS_RATING and RATING
        if rating is not None:
            audio_file["FMPS_RATING"] = str(rating)  # Normalize to 0-1 scale
            audio_file["RATING"] = str(self.get_5star_rating(rating))  # Store as 0-5 scale

        return audio_file

    def _parse_rating(self, raw_rating: Optional[str]) -> Optional[float]:
        """Parse a raw rating value and normalize it to a 0-1 scale."""
        if raw_rating is None:
            return None
        try:
            rating = float(raw_rating)
            if 0.0 <= rating <= 1.0:
                return round(rating, 3)
            elif rating <= 5.0:
                self.logger.debug(f"Scaling {rating} from 0–5 to 0–1")
                return round(rating / 5.0, 3)
            elif rating <= 100.0:
                self.logger.debug(f"Scaling {rating} from 0–100 to 0–1")
                return round(rating / 100.0, 3)
            else:
                self.logger.warning(f"Rating value out of expected bounds: {rating}")
                return None
        except ValueError:
            self.logger.warning(f"Invalid rating value: {raw_rating}")
            return None

    def _resolve_rating_conflict(self, fmps_rating: Optional[float], rating: Optional[float]) -> Optional[float]:
        """Resolve conflicts between FMPS_RATING and RATING."""
        if fmps_rating is None:
            return rating
        if rating is None:
            return fmps_rating
        if fmps_rating == rating:
            return rating

        # Prompt user to resolve conflict
        print(f"\nConflict detected between FMPS_RATING ({self.get_5star_rating(fmps_rating)}) and RATING ({self.get_5star_rating(rating)}).")
        while True:
            choice = input("Select the correct rating (0-5, half-star increments allowed): ").strip()
            try:
                selected_rating = float(choice)
                if 0 <= selected_rating <= 5 and (selected_rating * 2).is_integer():
                    return round(selected_rating / 5, 3)  # Normalize to 0-1 scale
            except ValueError:
                pass
            print("Invalid input. Please enter a number between 0 and 5 in half-star increments.")


class ID3Manager(AudioFileManager):
    RATING_MAP = [
        (0, 0),
        (0.1, 13),
        (0.2, 32),
        (0.3, 54),
        (0.4, 64),
        (0.5, 118),
        (0.6, 128),
        (0.7, 186),
        (0.8, 196),
        (0.9, 242),
        (1, 255),
    ]

    def __init__(self):
        from manager import manager

        self.mgr = manager
        self.discovered_rating_tags = set()
        self.conflicts = []
        self.rating_tags = {
            "WINDOWSMEDIAPLAYER": {"tag": "POPM:Windows Media Player 9 Series", "player": "Windows Media Player"},
            "MEDIAMONKEY": {"tag": "POPM:no@email", "player": "MediaMonkey"},
            "MUSICBEE": {"tag": "POPM:MusicBee", "player": "MusicBee"},
            "WINAMP": {"tag": "POPM:rating@winamp.com", "player": "Winamp"},
            "TEXT": {"tag": "TXXX:RATING", "player": "Text"},
        }
        self.ratingtag_to_key = {v["tag"]: k for k, v in self.rating_tags.items()}

        self.conflict_resolution_strategy = ConflictResolutionStrategy.from_value(self.mgr.config.conflict_resolution_strategy)
        self.tag_write_strategy = TagWriteStrategy.from_value(self.mgr.config.tag_write_strategy)
        self.default_tag = self.get_or_register_ratingtagkey(self.mgr.config.default_tag)
        self.tag_priority_order = [self.get_or_register_ratingtagkey(tag) for tag in self.mgr.config.tag_priority_order] if self.mgr.config.tag_priority_order else None

    # ------------------------------
    # Rating Normalization and Mapping
    # ------------------------------
    @classmethod
    def rating_to_popm(self, rating: float, five_star: bool = False) -> float:
        for val, byte in reversed(self.RATING_MAP):
            if rating >= val:
                return byte * 5 if five_star else byte
        return 0

    @classmethod
    def rating_from_popm(self, popm_value: int, five_star: bool = False) -> float:
        """Convert a POPM byte value (0-255) back to a rating (0-5)."""
        if popm_value == 0:
            return 0

        best_diff = float("inf")
        best_rating = 0.0

        for rating, byte in self.RATING_MAP:
            diff = abs(popm_value - byte)
            if diff < best_diff:
                best_diff = diff
                best_rating = rating * 5 if five_star else rating

        return best_rating

    def extract_rating_value(self, tag: str, frame: Union[POPM, TXXX]) -> Optional[int]:
        if isinstance(frame, POPM):
            return frame.rating
        elif tag == "TXXX:RATING" and isinstance(frame, TXXX):
            return self.rating_to_popm(float(frame.text[0]) / 5) if frame.text else None
        return None

    # ------------------------------
    # Rating Conflict Handling
    # ------------------------------
    def _handle_rating_tags(self, ratings: dict, track: AudioTag) -> Optional[float]:
        """Handle rating tags by validating configuration and routing to resolution or tracking conflicts."""
        if not ratings:
            return None

        # Normalize TXXX:Rating if present
        normalized_ratings = dict(ratings.items())
        if "TXXX:RATING" in normalized_ratings:
            normalized_ratings["TXXX:RATING"] = self.rating_to_popm(float(normalized_ratings["TXXX:RATING"]) / 5)

        # Track which tags are used
        for tag in normalized_ratings:
            if found_player := self.get_or_register_ratingtagkey(tag):
                self.mgr.stats.increment(f"FileSystemPlayer::tags_used::{self.rating_tags[found_player]['player']}")

        # Check for conflicts
        unique_ratings = set(normalized_ratings.values())
        if len(unique_ratings) == 1:
            return self.rating_from_popm(next(iter(unique_ratings)))

        self.mgr.stats.increment("FileSystemPlayer::tag_rating_conflict")
        self.conflicts.append({"track": track, "tags": normalized_ratings})

        # Handle conflicts
        return self._resolve_conflicting_ratings(normalized_ratings, track)

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
                raw_tag = self.rating_tags[tag]["tag"]
                if raw_tag in ratings and ratings[raw_tag] > 0:
                    return self.rating_from_popm(ratings[raw_tag])
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.HIGHEST:
            return self.rating_from_popm(max(ratings.values()))
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.LOWEST:
            return self.rating_from_popm(min(value for value in ratings.values() if value > 0))
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.AVERAGE:
            average_rating = sum(value for value in ratings.values() if value > 0) / len(ratings)
            return self.rating_from_popm(round(average_rating * 2) / 2)  # Round to nearest 0.5-star equivalent
        else:
            raise

    # ------------------------------
    # Tag Reading/Writing
    # ------------------------------
    def read_tags(self, audio_file: mutagen.FileType) -> AudioTag:
        id3_data = audio_file.tags
        rating_tags = {}
        for tag, frame in id3_data.items():
            rating = self.extract_rating_value(tag, frame)
            if rating is not None:
                rating_tags[tag] = rating
        track = AudioTag.from_id3(id3_data, audio_file.filename)
        track.rating = self._handle_rating_tags(rating_tags, track)
        return track

    def write_tags(self, audio_file: mutagen.FileType, metadata: dict, rating: Optional[float] = None) -> mutagen.FileType:
        """
        Write or update tags (album, artist, rating frames) to the audio_file.
        Return the modified file object.
        """
        # Update basic metadata
        if metadata:
            for key, value in metadata.items():
                if value:
                    audio_file[key] = value

        if rating is not None:
            # Determine which tags to write based on strategy
            if self.tag_write_strategy == TagWriteStrategy.OVERWRITE_DEFAULT:
                self._remove_existing_rating_tags(audio_file)
                tags_to_write = {self.default_tag}
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_ALL:
                tags_to_write = self.discovered_rating_tags
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_EXISTING:
                existing_tags = self._get_rating_tags(audio_file)
                tags_to_write = set(existing_tags) if existing_tags else {self.default_tag}
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_DEFAULT:
                tags_to_write = {self.default_tag}
            else:
                tags_to_write = set()  # Fallback: write nothing

            if tags_to_write:
                audio_file = self.apply_rating(audio_file, rating, tags_to_write)

        return audio_file

    def apply_rating(self, audio_file: mutagen.FileType, rating: float, valid_keys: set) -> mutagen.FileType:
        for key in valid_keys:
            tag = self.rating_tags[key]["tag"] if key in self.rating_tags else None
            if not tag:
                continue
            if tag == "TXXX:RATING":
                txt_rating = int(self.rating_from_popm(rating))
                if tag in audio_file.tags:
                    audio_file.tags[tag].text = [txt_rating]
                else:
                    new_txxx = TXXX(encoding=1, desc="RATING", text=[txt_rating])
                    audio_file.tags.add(new_txxx)
            else:
                popm_email = self.popm_email(tag)
                if tag in audio_file.tags:
                    audio_file.tags[tag].rating = int(rating)
                else:
                    new_popm = POPM(email=popm_email, rating=int(rating), count=0)
                    audio_file.tags.add(new_popm)
        return audio_file

    def _get_rating_tags(self, audio_file: mutagen.FileType) -> dict:
        """Retrieve rating tags from an audio file."""
        return {key: frame.rating for key, frame in audio_file.tags.items() if isinstance(frame, POPM)}

    def _remove_existing_rating_tags(self, audio_file: mutagen.FileType) -> None:
        for tag in list(audio_file.keys()):
            if tag == "TXXX:RATING" or tag.startswith("POPM:"):
                del audio_file[tag]

    # ------------------------------
    # Utility Methods
    # ------------------------------
    def get_tag_key(self, tag: str) -> Optional[str]:
        return self.ratingtag_to_key.get(tag)

    def register_unknown_tag(self, tag: str) -> str:
        safe_key = f"UNKNOWN{len(self.rating_tags)}"
        self.rating_tags[safe_key] = {"tag": tag, "player": tag}
        self.ratingtag_to_key[tag] = safe_key
        return safe_key

    def get_or_register_ratingtagkey(self, tag: str) -> str:
        if tag in self.ratingtag_to_key:
            return self.ratingtag_to_key[tag]
        safe_key = f"UNKNOWN{len(self.rating_tags)}"
        self.rating_tags[safe_key] = {"tag": tag, "player": tag}
        self.ratingtag_to_key[tag] = safe_key
        return safe_key

    @staticmethod
    def popm_email(tag: str) -> Optional[str]:
        return tag.split(":")[1] if ":" in tag else None

    def _configure_global_settings(self) -> None:
        """Prompt the user for global settings based on tag usage and conflicts."""
        tags_used = self.mgr.stats.get("FileSystemPlayer::tags_used")
        unique_tags = set(tags_used.keys())
        self.discovered_rating_tags = set(unique_tags)
        has_multiple_tags = len(unique_tags) > 1
        has_conflicts = len(self.conflicts) > 0

        # Skip prompts if unnecessary
        if not has_multiple_tags and not has_conflicts:
            return

        # Step 1: Prompt for conflict resolution strategy
        if has_conflicts:
            while True:
                print("-" * 50 + "\nRatings from multiple players found.  How should conflicts be resolved?")
                for idx, strategy in enumerate(ConflictResolutionStrategy, start=1):
                    print(f"  {idx}) {strategy.value.replace('_', ' ').capitalize()}")
                print(f"  {len(ConflictResolutionStrategy) + 1}) Show conflicts")
                print(f"  {len(ConflictResolutionStrategy) + 2}) Ignore files with conflicts")
                choice = input(f"Enter choice [1-{len(ConflictResolutionStrategy) + 2}]: ").strip()

                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(ConflictResolutionStrategy):
                        self.conflict_resolution_strategy = list(ConflictResolutionStrategy)[choice_num - 1]
                        break
                    elif choice_num == len(ConflictResolutionStrategy) + 1:
                        # Show conflicts
                        print("\nFiles with conflicting ratings:")
                        for conflict in self.conflicts:
                            track = conflict["track"]
                            print(f"\n{track.artist} | {track.album} | {track.title}")
                            for tag, rating in conflict["tags"].items():
                                tag_name = self.rating_tags[self.get_or_register_ratingtagkey(tag)]["player"] if self.get_or_register_ratingtagkey(tag) else tag
                                stars = self.rating_from_popm(rating)
                                print(f"\t{tag_name:<30} : {stars:<5}")
                        print("")
                        continue
                    elif choice_num == len(ConflictResolutionStrategy) + 2:
                        # Ignore files with conflicts
                        print("Conflicts will be ignored.")
                        break
                print("Invalid choice. Please try again.")

        # Step 2: Prompt for tag priority order (if required)
        available_tags = {self.get_or_register_ratingtagkey(tag): self.rating_tags[self.get_or_register_ratingtagkey(tag)]["player"] for tag in unique_tags}
        if self.conflict_resolution_strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER and has_conflicts and self.tag_priority_order is None:
            tag_list = list(available_tags.keys())

            while True:
                print("\nEnter media player priority order (highest priority first) by selecting numbers separated by commas.")
                print("Available media players:")
                for idx, tag in enumerate(tag_list, start=1):
                    print(f"  {idx}) {available_tags[tag]}")
                priority_order = input("Your input: ").strip()
                try:
                    selected_indices = [int(i) for i in priority_order.split(",")]
                    if all(1 <= idx <= len(tag_list) for idx in selected_indices):
                        selected_tags = [tag_list[idx - 1] for idx in selected_indices]
                        self.tag_priority_order = selected_tags
                        break
                except ValueError:
                    pass
                print("Invalid input. Please enter valid numbers separated by commas.")

        # Step 3: Prompt for tag write strategy
        if has_multiple_tags:
            while True:
                print("\nHow should ratings be written to files?")
                for idx, strategy in enumerate(TagWriteStrategy, start=1):
                    print(f"  {idx}) {strategy}")
                choice = input(f"Enter choice [1-{len(TagWriteStrategy)}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(TagWriteStrategy):
                    self.tag_write_strategy = list(TagWriteStrategy)[int(choice) - 1]
                    break
                print("Invalid choice. Please try again.")

        # Step 4: Prompt for default tag
        if self.tag_write_strategy and self.tag_write_strategy.requires_default_tag():
            valid_tags = list(available_tags.keys())

            while True:
                if len(valid_tags) == 1:
                    self.default_tag = valid_tags[0]
                    break
                print("\nWhich tag should be treated as the default for writing ratings?")
                for idx, tag in enumerate(valid_tags, start=1):
                    print(f"  {idx}) {available_tags[tag]}")
                choice = input(f"Enter choice [1-{len(valid_tags)}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(valid_tags):
                    self.default_tag = valid_tags[int(choice) - 1]
                    break
                print("Invalid choice. Please try again.")

        # Step 6: Prompt to save configuration
        save_config = False
        while True:
            print("\nWould you like to save these settings to config.ini for future runs?")
            choice = input("Your choice [yes/no]: ").strip().lower()
            if choice in {"y", "yes"}:
                save_config = True
                break
            elif choice in {"n", "no"}:
                break
            print("Invalid choice. Please enter 'y' or 'n'.")

        if save_config:
            self.save_config()

    # TODO: make sure this is in the right config.ini sections
    def save_config(self) -> None:
        """Save the configuration to config.ini."""
        config_path = Path("config.ini")

        # Function to get the appropriate identifier for a RatingTag
        def get_tag_identifier(safe_key: str) -> str:
            if safe_key.startswith("UNKNOWN"):
                return self.id3_mgr.rating_tags[safe_key]["tag"]
            return safe_key

        config_data = {
            "tag_write_strategy": self.tag_write_strategy.value if self.tag_write_strategy else None,
            "defaultd_tag": self.id3_mgr.rating_tags[self.default_tag]["tag"] if self.default_tag else None,
            "conflict_resolution_strategy": self.conflict_resolution_strategy.value if self.conflict_resolution_strategy else None,
            # Use enum name for defined tags, tag value for UNKNOWN tags
            "tag_priority_order": [get_tag_identifier(tag) for tag in self.tag_priority_order] if self.tag_priority_order else None,
        }

        with config_path.open("a", encoding="utf-8") as f:
            f.write("[filesystem]\n")
            for key, value in config_data.items():
                if value is not None:
                    f.write(f"{key} = {value}\n")
        print(f"Configuration saved to {config_path}")


class FileSystemProvider:
    """Adapter class for handling filesystem operations for audio files and playlists."""

    def __init__(self):
        from manager import manager

        self.mgr = manager
        self.id3_mgr = ID3Manager()
        self.vorbis_mgr = VorbisManager()
        self.logger = logging.getLogger("PlexSync.FileSystemProvider")
        self._audio_files = []
        self._playlist_files = []

    # ------------------------------
    # Core Manager Dispatch
    # ------------------------------
    def _get_manager(self, audio_file: mutagen.FileType) -> Optional[AudioFileManager]:
        if isinstance(audio_file, ID3FileType):
            return self.id3_mgr
        elif isinstance(audio_file, (FLAC, OggVorbis)):
            return self.vorbis_mgr
        return None

    # ------------------------------
    # File Handling (Low-Level Helpers)
    # ------------------------------
    def _open_track(self, file_path: Union[Path, str]) -> Optional[mutagen.FileType]:
        """Helper function to open an audio file using mutagen."""
        try:
            audio_file = mutagen.File(file_path, easy=False)
            if not audio_file:
                raise ValueError(f"Unsupported audio format: {file_path}")
            return audio_file
        except Exception as e:
            self.logger.error(f"Error opening file {file_path}: {e}")
            return None

    def _save_track(self, audio_file: mutagen.FileType) -> bool:
        """Helper function to save changes to an audio file."""
        try:
            if isinstance(audio_file, ID3FileType):
                audio_file.save(v2_version=3)
            else:
                audio_file.save()
            self.logger.info(f"Successfully saved changes to {audio_file.filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save file {audio_file.filename}: {e}")
            return False

    # ------------------------------
    # File Discovery (Scanning)
    # ------------------------------
    def scan_audio_files(self) -> List[Path]:
        """Scan directory structure for audio files"""
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

    # ------------------------------
    # Metadata Access and Update
    # ------------------------------
    def read_metadata_from_file(self, file_path: Union[Path, str]) -> Optional[AudioTag]:
        """Retrieve metadata from file."""
        audio_file = self._open_track(file_path)
        if not audio_file:
            return None

        manager = self._get_manager(audio_file)
        if not manager:
            self.logger.warning(f"No supported metadata found for {file_path}")
            return None

        tag = manager.read_tags(audio_file)
        self.logger.debug(f"Successfully read metadata for {file_path}")
        return tag

    def update_metadata_in_file(self, file_path: Union[Path, str], metadata: Optional[dict] = None, rating: Optional[float] = None) -> Optional[mutagen.File]:
        """Update metadata and/or rating in the audio file."""
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        audio_file = self._open_track(file_path)
        manager = self._get_manager(audio_file)
        if not manager:
            self.logger.warning(f"Cannot update metadata for unsupported format: {file_path}")
            return None

        updated_file = manager.write_tags(audio_file, metadata, rating)

        if metadata or rating is not None:
            if self._save_track(updated_file):
                return updated_file

        return None

    # ------------------------------
    # Playlist Operations
    # ------------------------------
    def create_playlist(self, title: str, is_extm3u: bool = False) -> Path:
        """Create a new M3U playlist file."""
        playlist_path = self.playlist_path / f"{title}.m3u"
        try:
            with playlist_path.open("w", encoding="utf-8") as file:
                if is_extm3u:
                    file.write("#EXTM3U\n")
                    file.write(f"#PLAYLIST:{title}\n")
            self.logger.info(f"Created playlist: {playlist_path}")
        except Exception as e:
            self.logger.error(f"Failed to create playlist {playlist_path}: {e}")
        return playlist_path

    def read_playlist(self, playlist_path: Path) -> Playlist:
        """Convert a text file into a Playlist object."""
        playlist = Playlist(name=playlist_path.stem.replace("_", " ").title())
        playlist.is_extm3u = False
        try:
            with playlist_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("#EXTM3U"):
                        playlist.is_extm3u = True
                    elif not playlist.is_extm3u:
                        # Return early if it's not an extended M3U playlist
                        return playlist
                    elif line.startswith("#PLAYLIST:"):
                        playlist.name = line.split(":", 1)[1].strip()
                    elif line.startswith("#EXTINF:") and not playlist.name:
                        return playlist
        except Exception as e:
            self.logger.error(f"Failed to read playlist {playlist_path}: {e}")
        return playlist

    def get_all_playlists(self) -> List[Playlist]:
        """Retrieve all M3U playlists in the playlist directory as Playlist objects."""
        playlists = []
        for playlist_path in self.playlist_path.glob("*.m3u"):
            playlists.append(self.read_playlist(playlist_path))
        return playlists

    # TODO:  think about how to match tracks to playlists and the cache - what to do if there is a root path conflict?
    def get_tracks_from_playlist(self, playlist_path: Path) -> List[Path]:
        """Retrieve all track paths from a playlist file."""
        tracks = []
        try:
            with playlist_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line and not line.startswith("#"):  # Ignore comments and directives
                        track_path = self.path / line
                        if track_path.exists():
                            tracks.append(track_path)
                        else:
                            self.logger.warning(f"Track not found: {track_path}")
        except Exception as e:
            self.logger.error(f"Failed to read playlist {playlist_path}: {e}")
        return tracks

    def add_track_to_playlist(self, playlist_path: Path, track: AudioTag, is_extm3u: bool = False) -> None:
        """Add a track to a playlist."""
        try:
            with playlist_path.open("a", encoding="utf-8") as file:
                if is_extm3u:
                    duration = track.duration if track.duration > 0 else -1
                    file.write(f"#EXTINF:{duration},{track.artist} - {track.title}\n")
                file.write(f"{track.file_path.relative_to(self.path)}\n")
            self.logger.info(f"Added track {track} to playlist {playlist_path}")
        except Exception as e:
            self.logger.error(f"Failed to add track to playlist {playlist_path}: {e}")

    def remove_track_from_playlist(self, playlist_path: Path, track: Path) -> None:
        """Remove a track from a playlist."""
        try:
            lines = []
            with playlist_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip() != str(track.file_path.relative_to(self.path)):
                        lines.append(line)
            with playlist_path.open("w", encoding="utf-8") as file:
                file.writelines(lines)
            self.logger.info(f"Removed track {track.file_path} from playlist {playlist_path}")
        except Exception as e:
            self.logger.error(f"Failed to remove track from playlist {playlist_path}: {e}")

    # ------------------------------
    # Post-Scan Logic / Summary
    # ------------------------------
    def finalize_scan(self) -> List[AudioTag]:
        """Finalize the scan by generating a summary and configuring global settings."""
        print(self._generate_summary())
        self.id3_mgr._configure_global_settings()
        resolved_conflicts = []
        bar = None
        for conflict in self.id3_mgr.conflicts:
            if len(self.id3_mgr.conflicts) > 100:
                bar = self.mgr.status.start_phase("Resolving rating conflicts", total=len(self.id3_mgr.conflicts))
            track = conflict["track"]
            self.logger.debug(f"Resolving conflicting ratings for {track.artist} | {track.album} | {track.title}")
            track.rating = self.id3_mgr._resolve_conflicting_ratings(conflict["tags"], track)
            resolved_conflicts.append(track)
            bar.update() if bar else None
        bar.close() if bar else None
        return resolved_conflicts

    def _generate_summary(self) -> str:
        """Generate a summary of rating tag usage, conflicts, and strategies."""
        total_files = len(self._audio_files)
        tag_usage = self.mgr.stats.get("FileSystemPlayer::tags_used")
        conflicts = len(self.id3_mgr.conflicts)

        # Format the summary
        summary = ["\n", "-" * 50, f"Scanned {total_files} files.\n"]
        if len(tag_usage) > 1:
            summary.append("Ratings Found:")
            for tag, count in tag_usage.items():
                summary.append(f"- {tag}: {count}")
        if conflicts > 0:
            summary.append(f"Files with conflicting ratings: {conflicts}")

        # Include strategies if set
        if self.id3_mgr.conflict_resolution_strategy:
            summary.append(f"\nConflict Resolution Strategy: {self.conflict_resolution_strategy.value}")
        if self.id3_mgr.tag_write_strategy:
            summary.append(f"Tag Write Strategy: {self.tag_write_strategy.value}")

        return "\n".join(summary)
