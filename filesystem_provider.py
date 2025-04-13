import abc
import logging
from enum import Enum, StrEnum, auto
from pathlib import Path
from typing import List, Optional, Union

import mutagen
from mutagen.id3 import POPM, TXXX, ID3FileType

from manager.config_manager import ConflictResolutionStrategy, TagWriteStrategy
from sync_items import AudioTag, Playlist


class RatingScale(Enum):
    NORMALIZED = auto()
    ZERO_TO_FIVE = auto()
    ZERO_TO_HUNDRED = auto()


class ID3Field(StrEnum):
    ARTIST = "TPE1"
    ALBUM = "TALB"
    TITLE = "TIT2"
    TRACKNUMBER = "TRCK"


class VorbisField(StrEnum):
    FMPS_RATING = "FMPS_RATING"
    RATING = "RATING"
    ARTIST = "ARTIST"
    ALBUM = "ALBUM"
    TITLE = "TITLE"
    TRACKNUMBER = "TRACKNUMBER"


class ID3TagRegistry:
    def __init__(self):
        self._tags_by_key = {}
        # TODO: deprecated
        self._tags_by_tag = {}
        # TODO: deprecated
        self._tags_by_player = {}

        initial = {
            "WINDOWSMEDIAPLAYER": {"tag": "POPM:Windows Media Player 9 Series", "player": "Windows Media Player"},
            "MEDIAMONKEY": {"tag": "POPM:no@email", "player": "MediaMonkey"},
            "MUSICBEE": {"tag": "POPM:MusicBee", "player": "MusicBee"},
            "WINAMP": {"tag": "POPM:rating@winamp.com", "player": "Winamp"},
            "TEXT": {"tag": "TXXX:RATING", "player": "Text"},
        }

        for key, entry in initial.items():
            self.register(entry["tag"], key=key, player=entry["player"])

    def register(self, tag: str, key: Optional[str] = None, player: Optional[str] = None) -> str:
        """Register a tag with a key and player name. If any are missing, generate fallbacks."""
        # Use existing if tag is known
        if id3tag := self.get_by_tag(tag):
            return id3tag

        # Generate a key if none provided
        key = key.upper() if key else f"UNKNOWN{len(self._tags_by_key)}"
        player = player or tag

        self._tags_by_key[key] = {"tag": tag, "player": player.strip()}
        self._tags_by_tag[tag] = key
        self._tags_by_player[player.lower()] = key

        return key

    def get_or_register_tag(self, tag: str) -> str:
        """Get the canonical key for a tag, registering it as UNKNOWN if not found."""
        return self._tags_by_tag.get(tag) or self.register(tag)

    def get_by_tag(self, tag: str) -> Optional[str]:
        return self._tags_by_tag.get(tag.upper())

    # TODO: deprecated
    def get_by_player(self, player: str) -> Optional[str]:
        return self._tags_by_player.get(player.lower())

    def get_tag(self, key: str) -> Optional[str]:
        return self._tags_by_key.get(key.upper(), {}).get("tag")

    def get_player(self, key: str) -> Optional[str]:
        return self._tags_by_key.get(key.upper(), {}).get("player")

    # TODO: deprecated
    def get_key(self, input_str: str) -> Optional[str]:
        # Try by tag, then by player
        return self.get_by_tag(input_str) or self.get_by_player(input_str)

    def known_keys(self) -> List[str]:
        return list(self._tags_by_key.keys())

    def get_config_value(self, key: str) -> str:
        """Return the appropriate value for saving into config:"""
        entry = self._tags_by_key.get(key)
        if not entry:
            return key

        if key.startswith("UNKNOWN"):
            return entry["tag"]
        return key

    # TODO: deprecated
    def resolve_to_tag(self, value: str) -> str:
        """
        Return the canonical tag (e.g. 'POPM:...') from any input:
        - If input is already a tag, key, or player name, resolve to full tag string.
        - Falls back to returning the original string if not found.
        """
        key = self.get_key(value)
        if key:
            tag = self.get_tag(key)
            if tag:
                return tag
        # Last-resort: value might already be a tag
        if value.upper().startswith("POPM:") or value.upper().startswith("TXXX:"):
            return value
        return value  # Unrecognized fallback

    def display_name_for_tag(self, tag: str) -> str:
        key = self.get_key(tag)
        return self.get_player(key) if key else tag


class AudioTagHandler(abc.ABC):
    @staticmethod
    def get_5star_rating(rating: Optional[float]) -> float:
        """Convert a normalized 0-1 scale rating to a 0-5 scale for display purposes."""
        return round(rating * 5, 1) if rating is not None else 0.0

    @staticmethod
    def has_rating_conflict(ratings: dict[str, Optional[float]]) -> bool:
        """Detect if multiple unique, non-null ratings exist among extracted values."""
        unique = {r for r in ratings.values() if r is not None}
        return len(unique) > 1

    @abc.abstractmethod
    def can_handle(self, file: mutagen.FileType) -> bool:
        """Return True if this manager can process the given file type."""
        pass

    def prepare(self, files: List[Path]) -> None:
        """Optionally analyze files before reading/writing. Default: no-op."""
        return

    @abc.abstractmethod
    def read_tags(self, audio_file: mutagen.FileType) -> (AudioTag, Optional[dict]):
        raise NotImplementedError("read_tags() must be implemented in subclasses.")

    @abc.abstractmethod
    def apply_tags(self, audio_file: mutagen.FileType, metadata: Optional[dict], rating: Optional[float] = None) -> mutagen.FileType:
        pass


class VorbisHandler(AudioTagHandler):
    def __init__(self):
        self.logger = logging.getLogger("PlexSync.VorbisManager")
        self.rating_scale = None
        self.fmps_rating_scale = None

    def prepare(self, files: List[mutagen.FileType]) -> None:
        if self.rating_scale and self.fmps_rating_scale:
            return

        self.logger.debug(f"Preparing VorbisManager by sampling from {len(files)} candidate files.")

        valid_files = []
        for audio_file in files:
            if audio_file and self.can_handle(audio_file):
                valid_files.append(audio_file)
                if len(valid_files) >= 10:
                    break

        if not valid_files:
            self.logger.warning("No valid Vorbis files found to sample for rating scale inference.")
            return

        self.logger.debug(f"Sampling {len(valid_files)} files for Vorbis rating scale detection.")
        self.sample_files(valid_files)

    def can_handle(self, file: mutagen.FileType) -> bool:
        return hasattr(file, "tags") and (VorbisField.RATING in file.tags or VorbisField.FMPS_RATING in file.tags)

    def read_tags(self, audio_file: mutagen.FileType) -> (AudioTag, Optional[dict]):
        """Read metadata and ratings from Vorbis comments."""
        track = AudioTag.from_vorbis(audio_file, audio_file.filename)

        fmps_rating = audio_file.get(VorbisField.FMPS_RATING)
        rating = audio_file.get(VorbisField.RATING)

        ratings = {
            VorbisField.FMPS_RATING: fmps_rating[0] if fmps_rating else None,
            VorbisField.RATING: rating[0] if rating else None,
        }

        if self.has_rating_conflict(ratings):
            resolved = self.resolve_rating_conflict(ratings)
            if resolved is not None:
                track.rating = resolved
                return track, None
            else:
                return track, ratings

        track.rating = fmps_rating or rating
        return track, None

    def apply_tags(self, audio_file: mutagen.FileType, metadata: Optional[dict] = None, rating: Optional[float] = None) -> mutagen.FileType:
        """Write metadata and ratings to Vorbis comments."""
        self.logger.debug(f"Starting to write tags for file: {audio_file.filename}")

        if metadata:
            self.logger.debug(f"Metadata to write: {metadata}")
            for key, value in metadata.items():
                if value:
                    self.logger.debug(f"Writing metadata key: {key}, value: {value}")
                    audio_file[key] = value

        if rating is not None:
            self.logger.debug(f"Rating to write: {rating}")
            # Write ratings to both FMPS_RATING and RATING
            fmps_rating_value = self._native_rating(rating, VorbisField.FMPS_RATING)
            rating_value = self._native_rating(rating, VorbisField.RATING)
            self.logger.debug(f"Normalized FMPS_RATING value: {fmps_rating_value}")
            self.logger.debug(f"Normalized RATING value: {rating_value}")

            audio_file[VorbisField.FMPS_RATING] = str(fmps_rating_value)
            audio_file[VorbisField.RATING] = str(rating_value)
            self.logger.info(f"Successfully wrote ratings: FMPS_RATING={fmps_rating_value}, RATING={rating_value}")

        self.logger.debug(f"Finished writing tags for file: {audio_file.filename}")
        return audio_file

    def sample_files(self, files: List[mutagen.FileType]) -> None:
        """Sample files to determine rating distribution characteristics."""
        rating_values = []
        fmps_rating_values = []

        for audio_file in files:
            rating_str = audio_file.get(VorbisField.RATING, [None])[0]
            fmps_str = audio_file.get(VorbisField.FMPS_RATING, [None])[0]

            try:
                if rating_str is not None:
                    rating = float(rating_str)
                    rating_values.append(rating)
            except ValueError:
                self.logger.debug(f"Invalid RATING value: {rating_str}")

            try:
                if fmps_str is not None:
                    fmps_rating = float(fmps_str)
                    fmps_rating_values.append(fmps_rating)
            except ValueError:
                self.logger.debug(f"Invalid FMPS_RATING value: {fmps_str}")

        self.rating_scale = self._infer_scale(rating_values)
        self.fmps_rating_scale = self._infer_scale(fmps_rating_values)

    def _normalize_rating(self, raw_rating: Union[str, float], field: str) -> Optional[float]:
        """Parse a raw rating value using the inferred scale."""
        try:
            rating = float(raw_rating or 0)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid rating value in {field}: {raw_rating}")
            return None

        scale = self.fmps_rating_scale if field.upper() == VorbisField.FMPS_RATING else self.rating_scale

        if scale == RatingScale.NORMALIZED:
            return round(rating, 3)
        elif scale == RatingScale.ZERO_TO_FIVE:
            return round(rating / 5.0, 3)
        elif scale == RatingScale.ZERO_TO_HUNDRED:
            return round(rating / 100.0, 3)
        else:
            self.logger.warning(f"Unknown rating scale for {field} — unable to parse value {rating}")
            return None

    def _native_rating(self, normalized: float, field: str) -> str:
        """Convert normalized 0-1 rating to appropriate scale for the target field."""
        scale = self.fmps_rating_scale if field.upper() == VorbisField.FMPS_RATING else self.rating_scale

        if scale == RatingScale.NORMALIZED:
            return str(round(normalized, 3))
        elif scale == RatingScale.ZERO_TO_FIVE:
            return str(round(normalized * 5.0, 1))
        elif scale == RatingScale.ZERO_TO_HUNDRED:
            return str(round(normalized * 100.0))
        else:
            self.logger.warning(f"Unknown scale for {field}, writing as normalized")
            return str(round(normalized, 3))

    def _infer_scale(self, values: List[float]) -> None:
        """Infer and store rating scales for RATING and FMPS_RATING based on sampled values."""

        if not values:
            return None

        min_val = min(values)
        max_val = max(values)

        if max_val > 10:
            return RatingScale.ZERO_TO_HUNDRED
        if 0.0 <= min_val < 1.0 and max_val <= 1.0:
            return RatingScale.NORMALIZED
        if 1.0 <= min_val and max_val <= 5.0:
            if min_val == max_val:
                self.logger.warning(f"Ambiguous values (all {min_val}) — defaulting to 0-5 scale")
                return RatingScale.ZERO_TO_FIVE
            return RatingScale.ZERO_TO_FIVE

        self.logger.warning(f"Ambiguous range: min={min_val}, max={max_val} — defaulting to 0-5 scale")
        return RatingScale.ZERO_TO_FIVE

    def resolve_rating_conflict(self, ratings: dict[str, float], tagging_strategy: Optional[dict] = None) -> Optional[float]:
        fmps = self._normalize_rating(ratings.get(VorbisField.FMPS_RATING), VorbisField.FMPS_RATING)
        other = self._normalize_rating(ratings.get(VorbisField.RATING), VorbisField.RATING)

        if fmps == other:
            return fmps

        print("\nConflicting ratings detected:")
        print(f"  FMPS_RATING: {self.get_5star_rating(fmps)}")
        print(f"  RATING     : {self.get_5star_rating(other)}")
        while True:
            choice = input("Select the correct rating (0-5, half-star increments allowed): ").strip()
            try:
                selected = float(choice)
                if 0 <= selected <= 5 and (selected * 2).is_integer():
                    return round(selected / 5, 3)  # normalize to 0-1 scale
            except ValueError:
                pass
            print("Invalid input. Please enter a number between 0 and 5 in half-star increments.")


class ID3Handler(AudioTagHandler):
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
        self.logger = logging.getLogger("PlexSync.ID3Manager")
        self.discovered_rating_tags = set()
        self.tag_registry = ID3TagRegistry()

        self.conflict_resolution_strategy = self.mgr.config.conflict_resolution_strategy
        self.tag_write_strategy = self.mgr.config.tag_write_strategy
        self.default_tag = self.mgr.config.default_tag
        self.tag_priority_order = self.mgr.config.tag_priority_order

    # ------------------------------
    # Rating Normalization and Mapping
    # ------------------------------
    @classmethod
    def rating_to_popm(cls, rating: float) -> float:
        for val, byte in reversed(cls.RATING_MAP):
            if rating >= val:
                return byte
        return 0

    @classmethod
    def rating_from_popm(cls, popm_value: int) -> float:
        """Convert a POPM byte value (0-255) back to a rating (0-5)."""
        if popm_value == 0:
            return 0

        best_diff = float("inf")
        best_rating = 0.0

        for rating, byte in cls.RATING_MAP:
            diff = abs(popm_value - byte)
            if diff < best_diff:
                best_diff = diff
                best_rating = rating

        return best_rating

    def _extract_rating_value(self, tag: str, frame: Union[POPM, TXXX]) -> Optional[float]:
        if isinstance(frame, POPM):
            return self.rating_from_popm(frame.rating) if frame.rating else None
        elif isinstance(frame, TXXX) and tag.upper() == "TXXX:RATING":
            try:
                return float(frame.text[0]) / 5
            except (ValueError, IndexError):
                return None
        return None

    # ------------------------------
    # Rating Conflict Handling
    # ------------------------------

    def resolve_rating_conflict(self, ratings: dict[str, float], tagging_strategy: dict) -> Optional[float]:
        """Resolve ID3 rating conflicts using the given strategy."""
        strategy = tagging_strategy.get("strategy", self.conflict_resolution_strategy)
        tag_priority_order = tagging_strategy.get("tag_priority_order", self.tag_priority_order)

        if not ratings:
            return None

        unique_ratings = set(ratings.values())
        if len(unique_ratings) == 1:
            return next(iter(unique_ratings))

        if strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER:
            if not tag_priority_order:
                self.logger.warning("No tag priority order defined for PRIORITIZED_ORDER strategy")
                return None
            for key in tag_priority_order:
                tag = self.tag_registry.resolve_to_tag(key)
                if tag in ratings and ratings[tag] > 0:
                    return ratings[tag]

        elif strategy == ConflictResolutionStrategy.HIGHEST:
            return max(ratings.values())

        elif strategy == ConflictResolutionStrategy.LOWEST:
            return min(value for value in ratings.values() if value > 0)

        elif strategy == ConflictResolutionStrategy.AVERAGE:
            filtered = [v for v in ratings.values() if v > 0]
            if not filtered:
                return None
            return round(sum(filtered) / len(filtered), 3)

        self.logger.warning(f"Unsupported conflict strategy: {strategy}") if strategy is not None else None
        return None

    # ------------------------------
    # Tag Reading/Writing
    # ------------------------------

    def read_tags(self, audio_file: mutagen.FileType) -> tuple[AudioTag, Optional[dict]]:
        rating_tags: dict[str, Optional[float]] = {}
        track = AudioTag.from_id3(audio_file.tags, audio_file.filename, duration=int(audio_file.info.length))

        for tag, frame in audio_file.tags.items():
            rating = self._extract_rating_value(tag, frame)
            if rating is not None:
                key = self.tag_registry.get_or_register_tag(tag)
                rating_tags[key] = rating
                self.discovered_rating_tags.add(tag)
                self.mgr.stats.increment(f"FileSystemPlayer::tags_used::{key}")

        if self.has_rating_conflict(rating_tags):
            resolved = self.resolve_rating_conflict(
                rating_tags,
                tagging_strategy={
                    "strategy": self.conflict_resolution_strategy,
                    "tag_priority_order": self.tag_priority_order,
                },
            )
            if resolved is not None:
                track.rating = resolved
                return track, None
            else:
                self.mgr.stats.increment("FileSystemPlayer::tag_rating_conflict")
                return track, rating_tags

        # No conflict — assign the single rating directly
        if rating_tags:
            track.rating = next(iter(rating_tags.values()))

        return track, None

    def apply_tags(self, audio_file: mutagen.FileType, metadata: dict, rating: Optional[float] = None) -> mutagen.FileType:
        """Write or update tags (album, artist, rating frames) to the audio_file."""
        self.logger.debug(f"Starting to write tags for file: {audio_file.filename}")

        if metadata:
            self.logger.debug(f"Metadata to write: {metadata}")
            for key, value in metadata.items():
                if value:
                    self.logger.debug(f"Writing metadata key: {key}, value: {value}")
                    audio_file[key] = value

        if rating is not None:
            self.logger.debug(f"Rating to write: {rating}")
            # Determine which tags to write based on strategy
            if self.tag_write_strategy == TagWriteStrategy.OVERWRITE_DEFAULT:
                self.logger.debug("Tag write strategy: OVERWRITE_DEFAULT")
                self._remove_existing_rating_tags(audio_file)
                tags_to_write = {self.default_tag}
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_ALL:
                self.logger.debug("Tag write strategy: WRITE_ALL")
                tags_to_write = self.discovered_rating_tags
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_EXISTING:
                self.logger.debug("Tag write strategy: WRITE_EXISTING")
                existing_tags = {key for key, frame in audio_file.tags.items() if isinstance(frame, POPM) or (isinstance(frame, TXXX) and key == "TXXX:RATING")}
                tags_to_write = set(existing_tags) if existing_tags else {self.default_tag}
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_DEFAULT:
                self.logger.debug("Tag write strategy: WRITE_DEFAULT")
                tags_to_write = {self.default_tag}
            else:
                self.logger.debug("Tag write strategy: Unknown or fallback")
                tags_to_write = set()  # Fallback: write nothing

            self.logger.debug(f"Tags determined for writing: {tags_to_write}")

            if tags_to_write:
                audio_file = self.apply_rating(audio_file, rating, tags_to_write)

        self.logger.debug(f"Finished writing tags for file: {audio_file.filename}")
        return audio_file

    def apply_rating(self, audio_file: mutagen.FileType, rating: float, valid_tags: set) -> mutagen.FileType:
        self.logger.debug(f"Applying rating: {rating} to tags: {valid_tags} for file: {audio_file.filename}")
        for raw_tag in valid_tags:
            if not raw_tag:
                self.logger.debug(f"Skipping empty tag in valid_tags: {valid_tags}")
                continue
            tag = self.tag_registry.resolve_to_tag(raw_tag)
            self.logger.debug(f"Resolved tag: {raw_tag} to canonical tag: {tag}")

            if tag.upper() == "TXXX:RATING":
                txt_rating = str(self.get_5star_rating(rating))
                self.logger.debug(f"Writing TXXX:RATING with value: {txt_rating} for tag: {tag}")
                if tag in audio_file.tags:
                    self.logger.debug(f"Updating existing TXXX:RATING tag with value: {audio_file.tags[tag].text}")
                    audio_file.tags[tag].text = [txt_rating]
                else:
                    self.logger.debug(f"Creating new TXXX:RATING tag with value: {txt_rating}")
                    new_txxx = TXXX(encoding=1, desc="RATING", text=[txt_rating])
                    audio_file.tags.add(new_txxx)
            else:
                popm_rating = int(self.rating_to_popm(rating))
                self.logger.debug(f"Writing POPM tag with rating value: {popm_rating} for tag: {tag}")
                if tag in audio_file.tags:
                    self.logger.debug(f"Updating existing POPM tag with current value: {audio_file.tags[tag].rating}")
                    audio_file.tags[tag].rating = popm_rating
                else:
                    popm_email = self.popm_email(tag)
                    self.logger.debug(f"Creating new POPM tag with email: {popm_email} and rating: {popm_rating}")
                    new_popm = POPM(email=popm_email, rating=popm_rating, count=0)
                    audio_file.tags.add(new_popm)
        self.logger.debug(f"Finished applying rating: {rating} for file: {audio_file.filename}")
        return audio_file

    def _remove_existing_rating_tags(self, audio_file: mutagen.FileType) -> None:
        self.logger.debug(f"Removing existing rating tags from file: {audio_file.filename}")
        for tag in list(audio_file.keys()):
            if tag == "TXXX:RATING" or tag.startswith("POPM:"):
                self.logger.debug(f"Removing tag: {tag} with current value: {audio_file[tag]}")
                del audio_file[tag]
        self.logger.debug(f"Finished removing existing rating tags from file: {audio_file.filename}")

    # ------------------------------
    # Utility Methods
    # ------------------------------

    def can_handle(self, file: mutagen.FileType) -> bool:
        return isinstance(file, ID3FileType) or (hasattr(file, "tags") and any(tag.startswith("TXXX:") or tag.startswith("POPM:") for tag in file.tags.keys()))

    @staticmethod
    def popm_email(tag: str) -> Optional[str]:
        return tag.split(":")[1] if ":" in tag else None

    def _configure_global_settings(self, conflicts: list[dict]) -> None:
        """Prompt the user for global settings based on tag usage and conflicts."""
        tags_used = self.mgr.stats.get("FileSystemPlayer::tags_used")
        if not tags_used:
            return

        unique_tags = [self.tag_registry._tags_by_key[key] for key in tags_used]
        has_multiple_tags = len(unique_tags) > 1
        has_conflicts = len(conflicts) > 0
        needs_saved = False

        if not has_multiple_tags and not has_conflicts:
            return

        # Step 1: Conflict resolution strategy
        if has_conflicts:
            while True:
                print("-" * 50 + "\nRatings from multiple players found.  How should conflicts be resolved?")
                for idx, strategy in enumerate(ConflictResolutionStrategy, start=1):
                    print(f"  {idx}) {strategy.display}")
                print(f"  {len(ConflictResolutionStrategy) + 1}) Show conflicts")
                print(f"  {len(ConflictResolutionStrategy) + 2}) Ignore files with conflicts")
                choice = input(f"Enter choice [1-{len(ConflictResolutionStrategy) + 2}]: ").strip()

                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(ConflictResolutionStrategy):
                        self.conflict_resolution_strategy = list(ConflictResolutionStrategy)[choice_num - 1]
                        needs_saved = True
                        break
                    elif choice_num == len(ConflictResolutionStrategy) + 1:
                        print("\nFiles with conflicting ratings:")
                        for conflict in self.conflicts:
                            track = conflict["track"]
                            print(f"\n{track.artist} | {track.album} | {track.title}")
                            for tag, rating in conflict["tags"].items():
                                print(f"\t{self.tag_registry.display_name_for_tag(tag):<30} : {self.get_5star_rating(rating):<5}")
                        print("")
                        continue
                    elif choice_num == len(ConflictResolutionStrategy) + 2:
                        print("Conflicts will be ignored.")
                        break
                print("Invalid choice. Please try again.")

        # Step 2: Prioritized tag order
        if self.conflict_resolution_strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER and has_conflicts and self.tag_priority_order is None:
            while True:
                print("\nEnter media player priority order (highest priority first) by selecting numbers separated by commas.")
                print("Available media players:")
                for idx, entry in enumerate(unique_tags, start=1):
                    print(f"  {idx}) {entry['player']}")
                priority_order = input("Your input: ").strip()
                try:
                    selected_indices = [int(i) for i in priority_order.split(",")]
                    if all(1 <= idx <= len(unique_tags) for idx in selected_indices):
                        self.tag_priority_order = [self.tag_registry.get_config_value(unique_tags[idx - 1]["tag"]) for idx in selected_indices]
                        needs_saved = True
                        break
                except ValueError:
                    pass
                print("Invalid input. Please enter valid numbers separated by commas.")

        # Step 3: Tag write strategy
        if has_multiple_tags and not self.tag_write_strategy:
            while True:
                print("\nHow should ratings be written to files?")
                for idx, strategy in enumerate(TagWriteStrategy, start=1):
                    print(f"  {idx}) {strategy.display}")
                choice = input(f"Enter choice [1-{len(TagWriteStrategy)}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(TagWriteStrategy):
                    self.tag_write_strategy = list(TagWriteStrategy)[int(choice) - 1]
                    needs_saved = True
                    break
                print("Invalid choice. Please try again.")

        # Step 4: Default tag
        if self.tag_write_strategy and self.tag_write_strategy.requires_default_tag() and not self.default_tag:
            while True:
                if len(unique_tags) == 1:
                    self.default_tag = unique_tags[0]["key"]
                    break
                print("\nWhich tag should be treated as the default for writing ratings?")
                for idx, entry in enumerate(unique_tags, start=1):
                    print(f"  {idx}) {entry['player']}")
                choice = input(f"Enter choice [1-{len(unique_tags)}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(unique_tags):
                    self.default_tag = self.tag_registry.get_config_value(unique_tags[int(choice) - 1]["tag"])
                    needs_saved = True
                    break
                print("Invalid choice. Please try again.")

        # Step 5: Save settings
        if needs_saved:
            while True:
                print("\nWould you like to save these settings to config.ini for future runs?")
                choice = input("Your choice [yes/no]: ").strip().lower()
                if choice in {"y", "yes"}:
                    self.mgr.config.conflict_resolution_strategy = self.conflict_resolution_strategy
                    self.mgr.config.tag_write_strategy = self.tag_write_strategy
                    self.mgr.config.default_tag = self.default_tag
                    self.mgr.config.tag_priority_order = self.tag_priority_order
                    self.mgr.config.save_config()
                    break
                elif choice in {"n", "no"}:
                    break
                print("Invalid choice. Please enter 'y' or 'n'.")


class FileSystemProvider:
    """Adapter class for handling filesystem operations for audio files and playlists."""

    TRACK_EXT = {".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aac"}
    PLAYLIST_EXT = {".m3u", ".m3u8", ".pls"}

    def __init__(self):
        from manager import manager

        self.mgr = manager
        self.id3_mgr = ID3Handler()
        self.vorbis_mgr = VorbisHandler()
        self._managers = [self.id3_mgr, self.vorbis_mgr]
        self.logger = logging.getLogger("PlexSync.FileSystemProvider")
        self._audio_files = []
        self._playlist_files = []
        self.conflicts = []
        self.tagging_policy = {
            "conflict_resolution_strategy": self.mgr.config.conflict_resolution_strategy,
            "tag_priority_order": self.mgr.config.tag_priority_order,
            "tag_write_strategy": self.mgr.config.tag_write_strategy,
            "default_tag": self.mgr.config.default_tag,
        }

    # ------------------------------
    # Core Manager Dispatch
    # ------------------------------
    def _get_manager(self, audio_file: mutagen.FileType) -> Optional[AudioTagHandler]:
        """Determine the appropriate manager for the given audio file."""
        return next((mgr for mgr in self._managers if mgr.can_handle(audio_file)), None)

    # ------------------------------
    # File Handling (Low-Level Helpers)
    # ------------------------------
    def _open_track(self, file_path: Union[Path, str]) -> Optional[mutagen.FileType]:
        """Helper function to open an audio file using mutagen."""
        self.logger.debug(f"Attempting to open file: {file_path}")
        try:
            audio_file = mutagen.File(file_path, easy=False)
            if not audio_file:
                self.logger.warning(f"Unsupported audio format for file: {file_path}")
                raise ValueError(f"Unsupported audio format: {file_path}")
            self.logger.info(f"Successfully opened file: {file_path}")
            return audio_file
        except Exception as e:
            self.logger.error(f"Error opening file {file_path}: {e}", exc_info=True)
            return None

    def _save_track(self, audio_file: mutagen.FileType) -> bool:
        """Helper function to save changes to an audio file."""
        self.logger.debug(f"Attempting to save file: {audio_file.filename}")
        if self.mgr.config.dry:
            self.logger.info(f"Dry run enabled. Changes to {audio_file.filename} will not be saved.")
            return True  # Simulate a successful save

        try:
            if isinstance(audio_file, ID3FileType):
                audio_file.save(v2_version=3)
            else:
                audio_file.save()
            self.logger.info(f"Successfully saved changes to {audio_file.filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save file {audio_file.filename}: {e}", exc_info=True)
            return False

    # ------------------------------
    # File Discovery (Scanning)
    # ------------------------------
    def scan_audio_files(self) -> List[Path]:
        """Scan directory structure for audio files."""
        path = self.mgr.config.path
        self.logger.debug(f"Starting scan for audio files in path: {path}")

        if not path:
            self.logger.error("Path is required for filesystem player")
            raise ValueError("Path is required for filesystem player")

        self.path = Path(path)
        if not self.path.exists():
            self.logger.error(f"Music directory not found: {path}")
            raise FileNotFoundError(f"Music directory not found: {path}")

        self.logger.info(f"Scanning {path} for audio files...")
        bar = self.mgr.status.start_phase("Collecting audio files", total=None)

        for file_path in self.path.rglob("*"):
            if file_path.suffix.lower() in self.TRACK_EXT:
                self.logger.debug(f"Discovered audio file: {file_path}")
                self._audio_files.append(file_path)
                bar.update()

        bar.close()
        self.logger.info(f"Found {len(self._audio_files)} audio files")
        for mgr in self._managers:
            mgr.prepare([mf for f in self._audio_files if (mf := mutagen.File(f, easy=False)) and mgr.can_handle(mf)])
        return self._audio_files

    def scan_playlist_files(self) -> List[Path]:
        """Scan and list all playlist files in the directory."""
        playlist_path = self.mgr.config.playlist_path
        self.playlist_path = Path(playlist_path) if playlist_path else self.path
        self.playlist_path.mkdir(exist_ok=True)
        # TODO: add support for m3u8 and pls

        bar = self.mgr.status.start_phase("Collecting audio files", total=None)

        for file_path in self.playlist_path.rglob("*"):
            if file_path.suffix.lower() in self.PLAYLIST_EXT:
                self._playlist_files.append(file_path)
                bar.update()
        bar.close()
        self.logger.info(f"Found {len(self._playlist_files)} playlist files")
        self.logger.info(f"Using playlists directory: {playlist_path}")

    def finalize_scan(self) -> List[AudioTag]:
        """Finalize the scan by generating a summary and configuring global settings."""
        print(self._generate_summary())
        self.id3_mgr._configure_global_settings(self.conflicts)
        resolved_conflicts = []
        bar = None
        for conflict in self.conflicts:
            if len(self.conflicts) > 100:
                bar = self.mgr.status.start_phase("Resolving rating conflicts", total=len(self.conflicts))
            track = conflict["track"]
            self.logger.debug(f"Resolving conflicting ratings for {track.artist} | {track.album} | {track.title}")
            track.rating = self.id3_mgr.resolve_rating_conflict(
                conflict["tags"],
                tagging_strategy={
                    "strategy": self.id3_mgr.conflict_resolution_strategy,
                    "tag_priority_order": self.id3_mgr.tag_priority_order,
                },
            )
            self.update_metadata_in_file(track.file_path, rating=track.rating)
            resolved_conflicts.append(track)
            bar.update() if bar else None
        bar.close() if bar else None
        return resolved_conflicts

    def get_tracks(self) -> List[Path]:
        """Return all discovered audio files."""
        return [t for t in self._audio_files if t.suffix.lower() in self.TRACK_EXT]

    def get_playlists(self) -> List[Path]:
        """Return all discovered playlist files."""
        return [p for p in self._playlist_files if p.suffix.lower() in self.PLAYLIST_EXT]

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

        tag, conflict = manager.read_tags(audio_file)
        if conflict:
            self.conflicts.append({"track": tag, "tags": conflict})
        self.logger.debug(f"Successfully read metadata for {file_path}")
        return tag

    def update_metadata_in_file(self, file_path: Union[Path, str], metadata: Optional[dict] = None, rating: Optional[float] = None) -> Optional[mutagen.File]:
        """Update metadata and/or rating in the audio file."""
        self.logger.debug(f"Updating metadata for file: {file_path} with metadata: {metadata} and rating: {rating}")
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        audio_file = self._open_track(file_path)
        if not audio_file:
            self.logger.warning(f"Failed to open file for metadata update: {file_path}")
            return None

        manager = self._get_manager(audio_file)
        if not manager:
            self.logger.warning(f"Cannot update metadata for unsupported format: {file_path}")
            return None

        updated_file = manager.apply_tags(audio_file, metadata, rating)
        if metadata or rating is not None:
            if self._save_track(updated_file):
                self.logger.info(f"Successfully updated metadata for file: {file_path}")
                return updated_file

        self.logger.warning(f"No changes were made to the file: {file_path}")
        return None

    # ------------------------------
    # Playlist Operations
    # ------------------------------
    def create_playlist(self, title: str, is_extm3u: bool = False) -> Playlist:
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
        return self.read_playlist(str(playlist_path))

    def read_playlist(self, playlist_path: str) -> Playlist:
        """Convert a text file into a Playlist object."""
        playlist = Playlist(name=Path(playlist_path).stem.replace("_", " ").title())
        playlist.file_path = str(playlist_path)
        playlist.is_extm3u = False
        playlist_path = Path(playlist_path)
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

    def get_tracks_from_playlist(self, playlist_path: str) -> List[Path]:
        """Retrieve all track paths from a playlist file."""
        playlist_path = Path(playlist_path)
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

    def add_track_to_playlist(self, playlist_path: str, track: AudioTag, is_extm3u: bool = False) -> None:
        """Add a track to a playlist."""
        playlist_path = Path(playlist_path)
        try:
            with playlist_path.open("a", encoding="utf-8") as file:
                if is_extm3u:
                    duration = int(track.duration) if track.duration > 0 else -1
                    file.write(f"#EXTINF:{duration},{track.artist} - {track.title}\n")
                file.write(f"{Path(track.file_path).relative_to(self.path)!s}\n")
            self.logger.info(f"Added track {track} to playlist {playlist_path}")
        except Exception as e:
            self.logger.error(f"Failed to add track to playlist {playlist_path}: {e}")

    def remove_track_from_playlist(self, playlist_path: str, track: Path) -> None:
        """Remove a track from a playlist."""
        playlist_path = Path(playlist_path)
        raise NotImplementedError
        try:
            lines = []
            with playlist_path.open("r", encoding="utf-8") as file:
                for line in file:
                    # TODO: Handle both relative and absolute paths
                    if line.strip() != str(Path(track.file_path).relative_to(self.path)):
                        lines.append(line)
            with playlist_path.open("w", encoding="utf-8") as file:
                file.writelines(lines)
            self.logger.info(f"Removed track {track.file_path} from playlist {playlist_path}")
        except Exception as e:
            self.logger.error(f"Failed to remove track from playlist {playlist_path}: {e}")

    # ------------------------------
    # Post-Scan Summary
    # ------------------------------

    def _generate_summary(self) -> str:
        """Generate a summary of rating tag usage, conflicts, and strategies."""
        total_files = len(self._audio_files)
        tag_usage = self.mgr.stats.get("FileSystemPlayer::tags_used")
        conflicts = len(self.conflicts)

        # Format the summary
        summary = ["\n", "-" * 50, f"Scanned {total_files} files.\n"]
        if len(tag_usage or []) > 1:
            summary.append("Ratings Found:")
            for key, count in tag_usage.items():
                player = self.id3_mgr.tag_registry.get_player(key)
                summary.append(f"- {player}: {count}")
        if conflicts > 0:
            summary.append(f"Files with conflicting ratings: {conflicts}")

        # Include strategies if set
        if self.id3_mgr.conflict_resolution_strategy:
            summary.append(f"\nConflict Resolution Strategy:\n\t{self.id3_mgr.conflict_resolution_strategy.display}")
            if self.id3_mgr.tag_priority_order:
                summary.append(f"\n\tTag Priority Order:\n\t{', '.join([self.id3_mgr.tag_registry.get_player(t) or t for t in self.id3_mgr.tag_priority_order])}")

        if self.id3_mgr.tag_write_strategy:
            summary.append(f"\nTag Write Strategy:\n\t{self.id3_mgr.tag_write_strategy.display}")

        if self.id3_mgr.default_tag:
            default_player = self.id3_mgr.tag_registry.get_player(self.id3_mgr.default_tag)
            summary.append(f"\nDefault Tag:\n\t{default_player or self.id3_mgr.default_tag}")

        return "\n".join(summary)
