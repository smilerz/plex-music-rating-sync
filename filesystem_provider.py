import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import mutagen
from mutagen.id3 import POPM, TXXX

from sync_items import AudioTag

IGNORED_TAGS = set()
VALID_TAGS = set()


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

    @property
    def popm_email(self) -> str:
        return self.tag.split(":")[1] if ":" in self.tag else None

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

    def initialize_settings(self) -> None:
        """Initialize settings for the filesystem provider."""
        from MediaPlayer import FileSystemPlayer

        self.get_normed_rating = FileSystemPlayer.get_normed_rating
        self.get_native_rating = FileSystemPlayer.get_native_rating
        self.get_5star_rating = FileSystemPlayer.get_5star_rating

        self.conflict_resolution_strategy = ConflictResolutionStrategy.from_value(self.mgr.config.conflict_resolution_strategy)
        self.tag_write_strategy = TagWriteStrategy.from_value(self.mgr.config.tag_write_strategy)
        self.default_tag = RatingTag.from_value(self.mgr.config.default_tag)
        self.tag_priority_order = [RatingTag.from_value(tag) for tag in self.mgr.config.tag_priority_order] if self.mgr.config.tag_priority_order else None
        self.delete_ignored_tags = False

    def _save_track(self, audio_file: mutagen.FileType) -> bool:
        """Helper function to save changes to an audio file."""
        try:
            audio_file.save(v2_version=3)
            self.logger.info(f"Successfully saved changes to {audio_file.filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save file {audio_file.filename}: {e}")
            return False

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

    def scan_audio_files(self) -> List[Path]:
        """Scan directory structure for audio files"""
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

    def finalize_scan(self) -> List[AudioTag]:
        """Finalize the scan by generating a summary and configuring global settings."""
        print(self._generate_summary())
        self._configure_global_settings()
        resolved_conflicts = []
        bar = None
        for conflict in self.conflicts:
            if len(self.conflicts) > 100:
                bar = self.mgr.status.start_phase("Resolving rating conflicts", total=len(self.conflicts))
            track = conflict["track"]
            self.logger.debug(f"Resolving conflicting ratings for {track.artist} | {track.album} | {track.title}")
            track.rating = self._resolve_conflicting_ratings(conflict["tags"], track)
            resolved_conflicts.append(track)
            self.update_metadata_in_file(track.file_path, rating=self.get_native_rating(track.rating))
            bar.update() if bar else None
        bar.close() if bar else None
        return resolved_conflicts

    # ======= PLAYLIST OPERATIONS =======

    def read_playlist(self, playlist_path: Path) -> Optional[List[Path]]:
        raise NotImplementedError

    def save_playlist(self, title: str, track_paths: List[Path]) -> Path:
        raise NotImplementedError

    def delete_playlist(self, playlist_path: Path) -> None:
        raise NotImplementedError

    def _get_rating_tags(self, audio_file: mutagen.FileType) -> dict:
        return {
            **{key: frame.rating for key, frame in audio_file.tags.items() if isinstance(frame, POPM)},
            **{
                key: self.get_native_rating(float(frame.text[0])) if frame.text else None
                for key, frame in audio_file.tags.items()
                if key == "TXXX:RATING" and isinstance(frame, TXXX)
            },
        }

    def _handle_rating_tags(self, ratings: dict, track: AudioTag) -> Optional[float]:
        """Handle rating tags by validating configuration and routing to resolution or tracking conflicts."""
        if not ratings:
            return None

        # Normalize TXXX:Rating if present
        if "TXXX:Rating" in ratings.keys():
            ratings["TXXX:Rating"] = self.get_native_rating(float(ratings["TXXX:Rating"]) / 5)

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
                if tag.tag in ratings and ratings[tag.tag] > 0:
                    return self.get_normed_rating(ratings[tag.tag])
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.HIGHEST:
            return self.get_normed_rating(max(ratings.values()))
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.LOWEST:
            return self.get_normed_rating(min(value for value in ratings.values() if value > 0))
        elif self.conflict_resolution_strategy == ConflictResolutionStrategy.AVERAGE:
            average_rating = sum([value for value in ratings.values() if value > 0]) / len(ratings)
            return self.get_normed_rating(round(average_rating * 2) / 2)  # Round to nearest 0.5-star equivalent
        else:
            raise

    def read_metadata_from_file(self, file_path: Union[Path, str]) -> Optional[AudioTag]:
        """Retrieve metadata from file."""
        audio_file = self._open_track(file_path)
        if not audio_file:
            return None

        if hasattr(audio_file, "tags") and audio_file.tags:
            # For MP3 (ID3 tags)
            tags = audio_file.tags
            album = tags.get("TALB", [""])[0]
            artist = tags.get("TPE1", [""])[0]
            title = tags.get("TIT2", [""])[0]
            track_number = tags.get("TRCK", [""])[0]

            # Handle multiple POPM and TXXX:RATING tags
            rating_tags = {
                **{key: frame.rating for key, frame in tags.items() if isinstance(frame, POPM)},
                **{key: self.get_native_rating(float(frame.text[0])) if frame.text else None for key, frame in tags.items() if key == "TXXX:RATING" and isinstance(frame, TXXX)},
            }
            rating = self._handle_rating_tags(
                rating_tags,
                track=AudioTag(
                    artist=artist,
                    album=album,
                    title=title,
                    file_path=str(file_path),
                    rating=None,
                    ID=str(file_path),
                    track=int(track_number.split("/")[0]) if track_number else None,
                ),
            )
        elif hasattr(audio_file, "get"):
            # For FLAC, OGG, etc.
            album = audio_file.get("album", [""])[0]
            artist = audio_file.get("artist", [""])[0]
            title = audio_file.get("title", [""])[0]
            track_number = audio_file.get("tracknumber", [""])[0]
            rating = audio_file.get("rating", [0])[0]
        else:
            self.logger.warning(f"No metadata found for {file_path}")
            return None

        tag = AudioTag(
            artist=artist,
            album=album,
            title=title,
            file_path=str(file_path),
            rating=rating,
            ID=str(file_path),
            track=int(track_number.split("/")[0]) if track_number else 1,
        )

        self.logger.debug(f"Successfully read metadata for {file_path}")
        return tag

    def _apply_rating(self, audio_file: mutagen.FileType, rating: float, valid_tags: set) -> None:
        """
        Write ratings to the specified tags in the audio file.

        Args:
            audio_file: The audio file object to update.
            rating: The normalized rating to write.
            valid_tags: The set of valid tags to update.
        """
        for tag in valid_tags:
            txxx_key = "TXXX:RATING"
            if tag == txxx_key:
                # Handle TXXX:Rating tag
                txt_rating = int(self.get_5star_rating(self.get_normed_rating(rating)))

                if txxx_key in audio_file.tags:
                    existing_frame = audio_file.tags[txxx_key]
                    existing_frame.text = [txt_rating]
                else:
                    new_txxx = TXXX(encoding=1, desc="RATING", text=[txt_rating])
                    audio_file.tags.add(new_txxx)
            else:
                # Handle POPM tags
                tag = RatingTag.from_value(tag)
                popm_frame = audio_file.tags.get(tag.tag)
                if popm_frame:
                    popm_frame.rating = int(rating)
                else:
                    new_popm = POPM(email=tag.popm_email, rating=int(rating), count=0)
                    audio_file.tags.add(new_popm)
        return audio_file

    # TODO: need to handle non-MP3 files
    def update_metadata_in_file(self, file_path: Union[Path, str], metadata: Optional[dict] = None, rating: Optional[float] = None) -> Optional[mutagen.File]:
        """
        Update metadata and/or rating in the audio file.

        Args:
            file_path: Path to the audio file.
            metadata: Dictionary containing metadata (e.g., title, album, artist).
            rating: Normalized rating to write to the file.

        Returns:
            The updated audio file object, or None if the update failed.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        audio_file = self._open_track(file_path)
        if not audio_file:
            return None

        # Update metadata if provided
        if metadata:
            for key, value in metadata.items():
                if value:
                    audio_file[key] = value

        # Update rating if provided
        if rating is not None:
            global VALID_TAGS

            # Remove ignored tags if delete_ignored_tags is True
            if self.delete_ignored_tags:
                for tag in IGNORED_TAGS:
                    if tag in audio_file:
                        del audio_file[tag]

            # Handle tag write strategy
            if self.tag_write_strategy == TagWriteStrategy.WRITE_ALL:
                audio_file = self._apply_rating(audio_file, rating, VALID_TAGS)
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_EXISTING:
                existing_tags = self._get_rating_tags(audio_file)
                if existing_tags:
                    audio_file = self._apply_rating(audio_file, rating, set(existing_tags))
                else:
                    audio_file = self._apply_rating(audio_file, rating, [self.default_tag.tag])
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_DEFAULT:
                audio_file = self._apply_rating(audio_file, rating, [self.default_tag.tag])
            elif self.tag_write_strategy == TagWriteStrategy.OVERWRITE_DEFAULT:
                for tag in list(audio_file.keys()):
                    if tag == "TXXX:RATING" or "POPM:" in tag:
                        del audio_file[tag]
                audio_file = self._apply_rating(audio_file, rating, [self.default_tag.tag])
            if self._save_track(audio_file):
                return audio_file

    def _generate_summary(self) -> str:
        """Generate a summary of rating tag usage, conflicts, and strategies."""
        total_files = len(self._audio_files)
        tag_usage = self.mgr.stats.get("FileSystemPlayer::tags_used")
        conflicts = len(self.conflicts)

        # Format the summary
        summary = ["\n", "-" * 50, f"Scanned {total_files} files.\n"]
        if tag_usage:
            summary.append("Ratings Found:")
            for tag, count in tag_usage.items():
                summary.append(f"- {tag}: {count}")
        if conflicts > 0:
            summary.append(f"Files with conflicting ratings: {conflicts}")

        # Include strategies if set
        if self.conflict_resolution_strategy:
            summary.append(f"\nConflict Resolution Strategy: {self.conflict_resolution_strategy.value}")
        if self.tag_write_strategy:
            summary.append(f"Tag Write Strategy: {self.tag_write_strategy.value}")

        return "\n".join(summary)

    def _configure_global_settings(self) -> None:
        """Prompt the user for global settings based on tag usage and conflicts."""
        global IGNORED_TAGS
        global VALID_TAGS
        tags_used = self.mgr.stats.get("FileSystemPlayer::tags_used")
        unique_tags = set(tags_used.keys()) - set(IGNORED_TAGS)
        VALID_TAGS = set(unique_tags)
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
                                tag_name = RatingTag.from_value(tag).player_name if RatingTag.from_value(tag) else tag
                                stars = self.get_5star_rating(self.get_normed_rating(rating))
                                print(f"\t{tag_name:<30} : {stars:<5}")
                        print("")
                        continue
                    elif choice_num == len(ConflictResolutionStrategy) + 2:
                        # Ignore files with conflicts
                        print("Conflicts will be ignored.")
                        break
                print("Invalid choice. Please try again.")

        # Step 2: Prompt for tag priority order (if required)
        ignored_tags = set()
        available_tags = {tag: (RatingTag.from_value(tag).player_name if isinstance(RatingTag.from_value(tag), RatingTag) else tag) for tag in unique_tags}
        if self.conflict_resolution_strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER and has_conflicts and self.tag_priority_order is None:
            tag_list = list(available_tags.keys())

            while True:
                print("\nEnter media player priority order (highest priority first) by selecting numbers separated by commas.")
                print("Available media players:")
                for idx, tag in enumerate(tag_list, start=1):
                    print(f"  {idx}) {RatingTag.from_value(tag)}")
                priority_order = input("Your input: ").strip()
                try:
                    selected_indices = [int(i) for i in priority_order.split(",")]
                    if all(1 <= idx <= len(tag_list) for idx in selected_indices):
                        selected_tags = [tag_list[idx - 1] for idx in selected_indices]
                        self.tag_priority_order = RatingTag.resolve_tags(selected_tags)
                        ignored_tags = set(tag_list) - set(selected_tags)
                        break
                except ValueError:
                    pass
                print("Invalid input. Please enter valid numbers separated by commas.")

        # Step 3: Ask about deleting ignored ratings
        if ignored_tags or IGNORED_TAGS:
            IGNORED_TAGS = ignored_tags.union(IGNORED_TAGS)
            if IGNORED_TAGS:
                print("\nSome media players have been ignored:")
                for tag in IGNORED_TAGS:
                    tag_name = RatingTag.from_value(tag).player_name if RatingTag.from_value(tag) else tag
                    print(f"  - {tag_name}")

                while True:
                    choice = input("Delete these ratings? (yes/no): ").strip().lower()
                    if choice in ("yes", "y"):
                        self.delete_ignored_tags = True
                        break
                    elif choice in ("no", "n"):
                        self.delete_ignored_tags = False
                        break
                    print("Invalid choice. Please enter 'yes' or 'no'.")

        # Step 4: Prompt for tag write strategy
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

        # Step 5: Prompt for default tag
        if self.tag_write_strategy and self.tag_write_strategy.requires_default_tag():
            valid_tags = RatingTag.resolve_tags(list(set(available_tags.keys()) - IGNORED_TAGS))

            while True:
                if len(valid_tags) == 1:
                    self.default_tag = valid_tags[0]
                    break
                print("\nWhich tag should be treated as the default for writing ratings?")
                for idx, tag in enumerate(valid_tags, start=1):
                    print(f"  {idx}) {tag}")
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
        def get_tag_identifier(tag: RatingTag) -> str:
            if tag.name.startswith("UNKNOWN"):
                return tag.tag  # Use the actual tag for unknown/dynamic tags
            return tag.name  # Use enum name for defined tags

        config_data = {
            "tag_write_strategy": self.tag_write_strategy.value if self.tag_write_strategy else None,
            "defaultd_tag": self.default_tag.tag if self.default_tag else None,
            "conflict_resolution_strategy": self.conflict_resolution_strategy.value if self.conflict_resolution_strategy else None,
            # Use enum name for defined tags, tag value for UNKNOWN tags
            "tag_priority_order": [get_tag_identifier(tag) for tag in self.tag_priority_order] if self.tag_priority_order else None,
            "delete_ignored_tags": self.delete_ignored_tags,
        }

        with config_path.open("a", encoding="utf-8") as f:
            f.write("[filesystem]\n")
            for key, value in config_data.items():
                if value is not None:
                    f.write(f"{key} = {value}\n")
        print(f"Configuration saved to {config_path}")
