import abc
import logging
from enum import StrEnum
from pathlib import Path
from typing import List, Optional, Union

import mutagen
from mutagen.id3 import POPM, TXXX, ID3FileType

from manager import get_manager
from manager.config_manager import ConflictResolutionStrategy, TagWriteStrategy
from ratings import Rating, RatingScale
from sync_items import AudioTag, Playlist


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
        self._entries_by_key = {}

        initial_entries = {
            "WINDOWSMEDIAPLAYER": {"id3_tag": "POPM:Windows Media Player 9 Series", "player_name": "Windows Media Player"},
            "MEDIAMONKEY": {"id3_tag": "POPM:no@email", "player_name": "MediaMonkey"},
            "MUSICBEE": {"id3_tag": "POPM:MusicBee", "player_name": "MusicBee"},
            "WINAMP": {"id3_tag": "POPM:rating@winamp.com", "player_name": "Winamp"},
            "TEXT": {"id3_tag": "TXXX:RATING", "player_name": "Text"},
        }

        for tag_key, entry in initial_entries.items():
            self.register(id3_tag=entry["id3_tag"], tag_key=tag_key, player_name=entry["player_name"])

    def register(self, id3_tag: str, tag_key: Optional[str] = None, player_name: Optional[str] = None) -> str:
        """Register a tag by ID3 tag string, returning its tag_key (creating one if necessary)."""
        tag_key = tag_key.upper() if tag_key else f"UNKNOWN{len(self._entries_by_key)}"

        # If already registered by ID3 tag, return existing key
        existing_key = self.get_key_for_id3_tag(id3_tag)
        if existing_key:
            return existing_key

        self._entries_by_key[tag_key] = {
            "id3_tag": id3_tag,
            "player_name": player_name or id3_tag,
        }

        return tag_key

    def get_id3_tag_for_key(self, tag_key: str) -> Optional[str]:
        return self._entries_by_key.get(tag_key.upper(), {}).get("id3_tag")

    def get_player_name_for_key(self, tag_key: str) -> Optional[str]:
        return self._entries_by_key.get(tag_key.upper(), {}).get("player_name")

    def get_key_for_id3_tag(self, id3_tag: str) -> Optional[str]:
        id3_tag_upper = id3_tag.upper()
        for key, entry in self._entries_by_key.items():
            if entry["id3_tag"].upper() == id3_tag_upper:
                return key
        return None

    def get_key_for_player_name_name(self, player_name: str) -> Optional[str]:
        player_lower = player_name.lower()
        for key, entry in self._entries_by_key.items():
            if entry["player_name"].lower() == player_lower:
                return key
        return None

    def resolve_key_from_input(self, input_str: str) -> Optional[str]:
        upper = input_str.upper()
        if upper in self._entries_by_key:
            return upper
        return self.get_key_for_id3_tag(input_str) or self.get_key_for_player_name_name(input_str)

    def known_keys(self) -> List[str]:
        return list(self._entries_by_key.keys())

    def get_config_value(self, tag_key: str) -> str:
        """Return config-friendly value for this tag_key. If UNKNOWN, return the raw ID3 tag."""
        entry = self._entries_by_key.get(tag_key)
        if not entry:
            return tag_key
        if tag_key.startswith("UNKNOWN"):
            return entry["id3_tag"]
        return tag_key

    def display_name(self, value: str) -> str:
        tag_key = self.resolve_key_from_input(value)
        return self.get_player_name_for_key(tag_key) or value

    def get_popm_email_for_key(self, key: str) -> Optional[str]:
        tag = self.get_id3_tag_for_key(key)
        if tag and tag.upper().startswith("POPM:"):
            return tag.split(":", 1)[1]
        return None


class AudioTagHandler(abc.ABC):
    def __init__(self, tagging_policy: Optional[dict] = None, **kwargs):
        self.logger = logging.getLogger(f"PlexSync.{self.__class__.__name__}")
        self.stats_mgr = get_manager().get_stats_manager()

        self.conflict_resolution_strategy = tagging_policy.get("conflict_resolution_strategy") if tagging_policy else None
        self.tag_write_strategy = tagging_policy.get("tag_write_strategy") if tagging_policy else None
        self.default_tag = tagging_policy.get("default_tag") if tagging_policy else None
        self.tag_priority_order = tagging_policy.get("tag_priority_order") if tagging_policy else None

    def is_strategy_supported(self, strategy: ConflictResolutionStrategy) -> bool:
        """Override to declare which conflict strategies this handler supports."""
        return True  # By default, assume all strategies are valid

    @abc.abstractmethod
    def finalize_rating_strategy(self, conflicts: list[dict]) -> None:
        """Decide on strategies or settings to resolve ratings during second pass."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _resolve_conflict(self, ratings_by_tag_key: dict[str, Rating], track: AudioTag) -> Optional[Rating]:
        """Resolve a normalized rating from tag_key → rating mappings using the configured strategy."""
        strategy = self.conflict_resolution_strategy
        if not strategy:
            return None

        if not self.is_strategy_supported(strategy):
            self.logger.warning(f"Strategy '{strategy}' is not supported by {self.__class__.__name__}; falling back to 'HIGHEST'")
            strategy = ConflictResolutionStrategy.HIGHEST

        if strategy == ConflictResolutionStrategy.CHOICE:
            return self._manual_conflict_resolution(ratings_by_tag_key, track)

        if strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER:
            if not self.tag_priority_order:
                self.logger.warning("No tag priority order defined for PRIORITIZED_ORDER strategy")
                return None
            for tag_key in self.tag_priority_order:
                if tag_key in ratings_by_tag_key and ratings_by_tag_key[tag_key] > 0:
                    return ratings_by_tag_key[tag_key]
            return None

        if strategy == ConflictResolutionStrategy.HIGHEST:
            return max(ratings_by_tag_key.values())

        if strategy == ConflictResolutionStrategy.LOWEST:
            return min((v for v in ratings_by_tag_key.values() if v > 0), default=None)

        if strategy == ConflictResolutionStrategy.AVERAGE:
            values = [v.to_float(RatingScale.NORMALIZED) for v in ratings_by_tag_key.values() if v and not v.is_unrated]
            return Rating(sum(values) / len(values), scale=RatingScale.NORMALIZED) if values else None

        self.logger.warning(f"Unsupported conflict strategy: {strategy}")
        return None

    @abc.abstractmethod
    def _manual_conflict_resolution(self, ratings_by_tag_key: dict[str, Rating], track: AudioTag) -> Optional[Rating]:
        """Interactive user choice for resolving conflicting ratings."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def can_handle(self, file: mutagen.FileType) -> bool:
        """Return True if this handler can process the given file."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def _extract_tags(self, audio_file: mutagen.FileType) -> tuple[AudioTag, dict]:
        """Extract metadata and ratings from the audio file."""
        raise NotImplementedError("Subclasses must implement this method.")

    def read_tags(self, audio_file: mutagen.FileType) -> tuple[AudioTag, Optional[dict]]:
        """
        Unified tag reading workflow:
        1. Extract raw and normalized data.
        2. Attempt to resolve the rating using get_normal_rating().
        3. If successful, assign the rating to the track.
        4. Otherwise, return the context for deferred resolution.
        """
        track, context = self._extract_tags(audio_file)

        rating = self.get_normal_rating(context)
        if rating is not None:
            track.rating = rating
            return track, None

        return track, context

    def get_normal_rating(self, context: dict) -> Optional[Rating]:
        normalized: dict[str, Optional[Rating]] = context.get("normalized", {})
        non_null_values = [r for r in normalized.values() if r is not None]

        if len(set(non_null_values)) == 1:
            return non_null_values[0]
        elif not non_null_values:
            return None

        return self._resolve_rating(context)

    @abc.abstractmethod
    def _resolve_rating(self, context: dict) -> Optional[Rating]:
        """Subclasses must implement custom resolution logic based on context."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def apply_tags(self, audio_file: mutagen.FileType, metadata: Optional[dict], rating: Optional[Rating] = None) -> mutagen.FileType:
        """Write metadata and rating to the audio file."""
        raise NotImplementedError("Subclasses must implement this method.")


class VorbisHandler(AudioTagHandler):
    def __init__(self, tagging_policy: Optional[dict] = None, **kwargs):
        super().__init__(tagging_policy=tagging_policy, **kwargs)
        self.rating_scale = None
        self.fmps_rating_scale = None
        self.aggressive_inference = False
        # TODO: add user prompt for strategy
        self.conflict_resolution_strategy = ConflictResolutionStrategy.HIGHEST

    def is_strategy_supported(self, strategy: ConflictResolutionStrategy) -> bool:
        return strategy != ConflictResolutionStrategy.PRIORITIZED_ORDER

    def can_handle(self, file: mutagen.FileType) -> bool:
        return hasattr(file, "tags") and (VorbisField.RATING in file.tags or VorbisField.FMPS_RATING in file.tags)

    def _manual_conflict_resolution(self, ratings: dict[str, Rating], track: AudioTag) -> Optional[Rating]:
        print("\nConflicting ratings detected:")
        print(track.details())
        for tag, rating in ratings.items():
            print(f"  {tag:<20} : {rating.to_display():<5}")

        while True:
            choice = input("Select the correct rating (0-5, half-star increments allowed): ").strip()
            validated = Rating.validate(choice, scale=RatingScale.ZERO_TO_FIVE)
            if validated is not None:
                return Rating(choice, scale=RatingScale.ZERO_TO_FIVE)
            print("Invalid input. Please enter a number between 0 and 5 in half-star increments.")

    def _get_scale_for(self, field: str) -> Optional[RatingScale]:
        if field == VorbisField.FMPS_RATING:
            return self.fmps_rating_scale
        if field == VorbisField.RATING:
            return self.rating_scale
        return None

    def _extract_tags(self, audio_file: mutagen.FileType) -> tuple[AudioTag, dict]:
        """Extract raw and normalized ratings for Vorbis fields."""
        track = self.get_audiotag(audio_file, audio_file.filename)

        raw: dict[str, Optional[str]] = {}
        normalized: dict[str, Optional[Rating]] = {}

        for field in [VorbisField.FMPS_RATING, VorbisField.RATING]:
            raw_value = audio_file.get(field, [None])[0]
            raw[field] = raw_value
            normalized[field] = Rating.try_create(raw_value, scale=self._get_scale_for(field), aggressive=self.aggressive_inference)

        return track, {
            "raw": raw,
            "normalized": normalized,
            "handler": self,
        }

    def _resolve_rating(self, context: dict) -> Optional[Rating]:
        ratings: dict[str, Rating] = {k: v for k, v in context.get("normalized", {}).items() if v is not None}

        if not ratings:
            return None

        return self._resolve_normalized_conflict(ratings, context.get("track"))

    def finalize_rating_strategy(self, conflicts: list[dict]) -> None:
        self._print_summary()

        def resolve_scale_for(field: str) -> Optional[RatingScale]:
            field_stats = self.stats_mgr.get(f"VorbisHandler::scale_inferred::{field}")
            if not field_stats:
                return None
            max_count = max(field_stats.values())
            tied = [RatingScale[key] for key, count in field_stats.items() if count == max_count]
            sorted_scales = sorted(tied, key=lambda s: s != RatingScale.NORMALIZED)
            return sorted_scales[0]

        self.fmps_rating_scale = resolve_scale_for(VorbisField.FMPS_RATING)
        self.rating_scale = resolve_scale_for(VorbisField.RATING)
        self.aggressive_inference = True

    def apply_tags(self, audio_file: mutagen.FileType, metadata: Optional[dict] = None, rating: Optional[Rating] = None) -> mutagen.FileType:
        """Write metadata and ratings to Vorbis comments."""
        if metadata:
            for key, value in metadata.items():
                if value:
                    audio_file[key] = value

        if rating is not None:
            audio_file[VorbisField.FMPS_RATING] = rating.to_str(self.fmps_rating_scale)
            audio_file[VorbisField.RATING] = rating.to_str(self.rating_scale)

        return audio_file

    def _print_summary(self) -> None:
        print("\nFLAC, OGG (and other Vorbis formats) Ratings:")

        # Inferred Scales
        rating_counts = self.stats_mgr.get("VorbisHandler::scale_inferred::RATING")
        fmps_counts = self.stats_mgr.get("VorbisHandler::scale_inferred::FMPS_RATING")

        if rating_counts or fmps_counts:
            print("  Inferred Rating Scales:")
            for field, counts in [("RATING", rating_counts), ("FMPS_RATING", fmps_counts)]:
                if not counts:
                    continue
                print(f"    {field}:")
                for scale, count in counts.items():
                    print(f"      - {scale.title()}: {count}")

        if self.conflict_resolution_strategy:
            print("\n  Conflict Resolution Strategy:")
            print(f"    {self.conflict_resolution_strategy.display}")

    def get_audiotag(self, vorbis: object, file_path: str) -> AudioTag:
        """Create an AudioTag from a Vorbis object."""
        track = vorbis.get(VorbisField.TRACKNUMBER, None)[0]
        duration = vorbis.info.length if hasattr(vorbis, "info") and hasattr(vorbis.info, "length") else -1
        return AudioTag(
            artist=vorbis.get(VorbisField.ARTIST, [""])[0],
            album=vorbis.get(VorbisField.ALBUM, [""])[0],
            title=vorbis.get(VorbisField.TITLE, [""])[0],
            file_path=str(file_path or ""),
            rating=None,
            ID=str(file_path),
            track=int(track.split("/")[0] if "/" in track else track),
            duration=int(duration or -1),
        )


class ID3Handler(AudioTagHandler):
    def __init__(self, tagging_policy: Optional[dict] = None, **kwargs):
        super().__init__(tagging_policy=tagging_policy, **kwargs)

        self.discovered_rating_tags = set()
        self.tag_registry = ID3TagRegistry()

    # ------------------------------
    # Rating Normalization and Mapping
    # ------------------------------

    def _manual_conflict_resolution(self, ratings: dict[str, Rating], track: AudioTag) -> Optional[Rating]:
        print("\nConflicting ratings detected:")
        print(track.details())
        for tag_key, rating in ratings.items():
            print(f"  {self.tag_registry.display_name(tag_key):<30} : {rating.to_display():<5}")

        while True:
            choice = input("Select the correct rating (0-5, half-star increments allowed): ").strip()
            validated = Rating.validate(choice, scale=RatingScale.ZERO_TO_FIVE)
            if validated is not None:
                return Rating(choice, scale=RatingScale.ZERO_TO_FIVE)
            print("Invalid input. Please enter a number between 0 and 5 in half-star increments.")

    # ------------------------------
    # Rating Conflict Handling
    # ------------------------------
    def _resolve_rating(self, context: dict) -> Optional[Rating]:
        """Resolve normalized rating from multiple ID3 tags using conflict strategy."""
        ratings: dict[str, Rating] = {}

        normalized = context.get("normalized", {})

        for key, value in normalized.items():
            if value is not None:
                ratings[key] = value

        if not ratings:
            return None

        unique_ratings = set(ratings.values())
        if len(unique_ratings) == 1:
            return next(iter(unique_ratings))

        return self._resolve_normalized_conflict(ratings)

    # ------------------------------
    # Tag Reading/Writing
    # ------------------------------

    def _extract_tags(self, audio_file: mutagen.FileType) -> tuple[AudioTag, dict]:
        """Extract raw and normalized ratings for ID3 frames."""
        track = self.get_audiotag(audio_file.tags, audio_file.filename, duration=int(audio_file.info.length))

        raw: dict[str, Optional[str]] = {}
        normalized: dict[str, Optional[Rating]] = {}

        for tag_key, frame in audio_file.tags.items():
            if not (tag_key.startswith("POPM:") or tag_key == "TXXX:RATING"):
                continue

            key = self.tag_registry.register(tag_key)
            self.discovered_rating_tags.add(key)
            self.stats_mgr.increment(f"FileSystemPlayer::tags_used::{key}")

            if isinstance(frame, POPM):
                raw_value = str(frame.rating)
            elif isinstance(frame, TXXX):
                raw_value = frame.text[0] if frame.text else None
            else:
                raw_value = None

            raw[key] = raw_value
            scale = RatingScale.ZERO_TO_FIVE if tag_key.upper() == "TXXX:RATING" else RatingScale.POPM
            normalized[key] = Rating(raw_value, scale=scale) if raw_value is not None else None

        return track, {
            "raw": raw,
            "normalized": normalized,
            "handler": self,
        }

    def apply_tags(self, audio_file: mutagen.FileType, metadata: dict, rating: Optional[Rating] = None) -> mutagen.FileType:
        """Write or update tags (album, artist, rating frames) to the audio_file."""
        if metadata:
            for key, value in metadata.items():
                if value:
                    audio_file[key] = value

        if rating is not None:
            # Determine which tag_keys to write based on configured strategy
            if self.tag_write_strategy == TagWriteStrategy.OVERWRITE_DEFAULT:
                self._remove_existing_id3_tags(audio_file)
                tag_keys_to_write = {self.default_tag}

            elif self.tag_write_strategy == TagWriteStrategy.WRITE_ALL:
                tag_keys_to_write = self.discovered_rating_tags

            elif self.tag_write_strategy == TagWriteStrategy.WRITE_EXISTING:
                # Normalize raw frame names into tag_keys using registry
                tag_keys_to_write = {
                    self.tag_registry.register(str(frame_key))
                    for frame_key, frame in audio_file.tags.items()
                    if isinstance(frame, POPM) or (isinstance(frame, TXXX) and frame_key == "TXXX:RATING")
                } or {self.default_tag}

            elif self.tag_write_strategy == TagWriteStrategy.WRITE_DEFAULT:
                tag_keys_to_write = {self.default_tag}

            else:
                self.logger.debug("Tag write strategy: Unknown or fallback")
                tag_keys_to_write = set()

            if tag_keys_to_write:
                audio_file = self.apply_rating(audio_file, rating, tag_keys_to_write)

        return audio_file

    def apply_rating(self, audio_file: mutagen.FileType, rating: Rating, tag_keys: set[str]) -> mutagen.FileType:
        self.logger.debug(f"Applying normalized rating {rating.to_display()} to file: {audio_file.filename}")

        for tag_key in tag_keys:
            if not tag_key:
                continue  # Invalid key; no log needed

            tag = self.tag_registry.get_id3_tag_for_key(tag_key)
            if not tag:
                self.logger.warning(f"Tag key '{tag_key}' has no registered tag string; skipping.")
                continue

            # Determine tag type and apply rating
            if tag.upper() == "TXXX:RATING":
                txt_rating = rating.to_str(RatingScale.ZERO_TO_FIVE)
                if tag in audio_file.tags:
                    self.logger.debug(f"Updating TXXX:RATING ({tag_key}) to value: {txt_rating}")
                    audio_file.tags[tag].text = [txt_rating]
                else:
                    self.logger.debug(f"Creating TXXX:RATING ({tag_key}) with value: {txt_rating}")
                    new_txxx = TXXX(encoding=1, desc="RATING", text=[txt_rating])
                    audio_file.tags.add(new_txxx)

            elif tag.upper().startswith("POPM:"):
                popm_rating = rating.to_int(RatingScale.POPM)
                popm_email = self.tag_registry.get_popm_email_for_key(tag_key)
                if tag in audio_file.tags:
                    self.logger.debug(f"Updating POPM ({tag_key}) rating to: {popm_rating}")
                    audio_file.tags[tag].rating = popm_rating
                else:
                    self.logger.debug(f"Creating POPM ({tag_key}) with email: {popm_email}, rating: {popm_rating}")
                    new_popm = POPM(email=popm_email, rating=popm_rating, count=0)
                    audio_file.tags.add(new_popm)

            else:
                self.logger.warning(f"Unrecognized tag format for tag_key '{tag_key}' resolved to: {tag}")

        self.logger.debug(f"Finished applying rating to: {audio_file.filename}")
        return audio_file

    def _remove_existing_id3_tags(self, audio_file: mutagen.FileType) -> None:
        self.logger.debug(f"Removing existing ID3 rating tags from file: {audio_file.filename}")
        for id3_tag in list(audio_file.keys()):
            if id3_tag == "TXXX:RATING" or id3_tag.startswith("POPM:"):
                self.logger.debug(f"Removing ID3 tag: {id3_tag} with current value: {audio_file[id3_tag]}")
                del audio_file[id3_tag]
        self.logger.debug(f"Finished removing rating ID3 tags from file: {audio_file.filename}")

    def get_audiotag(self, id3: object, file_path: str, duration: int = -1) -> AudioTag:
        """Create an AudioTag from an ID3 object."""

        def _safe_get(field: str) -> str:
            """Safely get ID3 field value."""
            return id3.get(field).text[0] if id3.get(field) else None

        track = _safe_get(ID3Field.TRACKNUMBER) or "0"
        return AudioTag(
            artist=_safe_get(ID3Field.ARTIST) or "",
            album=_safe_get(ID3Field.ALBUM) or "",
            title=_safe_get(ID3Field.TITLE) or "",
            file_path=str(file_path),
            rating=None,
            ID=str(file_path),
            track=int(track.split("/")[0] if "/" in track else track),
            duration=int(duration or -1),
        )

    # ------------------------------
    # Utility Methods
    # ------------------------------

    def can_handle(self, file: mutagen.FileType) -> bool:
        return isinstance(file, ID3FileType) or (hasattr(file, "tags") and any(tag.startswith("TXXX:") or tag.startswith("POPM:") for tag in file.tags.keys()))

    def _print_summary(self) -> None:
        print("\nMP3 Ratings:")

        tag_keys_used = self.stats_mgr.get("FileSystemPlayer::tags_used")
        if tag_keys_used:
            print("  Ratings Found:")
            for tag_key, count in tag_keys_used.items():
                player = self.tag_registry.get_player_name_for_key(tag_key)
                label = player or tag_key
                print(f"    - {label}: {count}")

        if self.conflict_resolution_strategy:
            print("\n  Conflict Resolution Strategy:")
            print(f"    {self.conflict_resolution_strategy.display}")
            if self.conflict_resolution_strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER and self.tag_priority_order:
                print("    Tag Priority Order:")
                for tag_key in self.tag_priority_order:
                    player = self.tag_registry.get_player_name_for_key(tag_key)
                    label = player or tag_key
                    print(f"      - {label}")

        if self.tag_write_strategy:
            print(f"\n  Tag Write Strategy:\n    {self.tag_write_strategy.display}")

        if self.default_tag:
            player = self.tag_registry.get_player_name_for_key(self.default_tag)
            label = player or self.default_tag
            print(f"\n  Default Tag:\n    {label}")

    def finalize_rating_strategy(self, conflicts: list[dict]) -> None:
        """Interactively resolve strategies and settings for rating conflicts."""
        tag_keys_used = self.stats_mgr.get("FileSystemPlayer::tags_used")
        if not tag_keys_used:
            return

        unique_tag_keys = [
            {
                "key": tag_key,
                "tag": self.tag_registry.get_id3_tag_for_key(tag_key),
                "player": self.tag_registry.get_player_name_for_key(tag_key),
            }
            for tag_key in tag_keys_used
        ]
        has_multiple_tags = len(unique_tag_keys) > 1
        has_conflicts = len(conflicts) > 0
        needs_saved = False

        if not has_multiple_tags and not has_conflicts:
            return

        self._print_summary()

        # Step 1: Conflict resolution strategy
        if has_conflicts and self.conflict_resolution_strategy is None:
            while True:
                print("-" * 50 + "\nRatings from multiple players found. How should conflicts be resolved?")
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
                        for conflict in conflicts:
                            if conflict.get("manager") is not self:
                                continue
                            track = conflict["track"]
                            print(f"\n{track.artist} | {track.album} | {track.title}")
                            for tag_key, rating in conflict["normalized"].items():
                                print(f"\t{self.tag_registry.display_name(tag_key):<30} : {rating.to_display():<5}")
                        print("")
                        continue
                    elif choice_num == len(ConflictResolutionStrategy) + 2:
                        print("Conflicts will be ignored.")
                        break
                print("Invalid choice. Please try again.")

        # Step 2: Tag priority order (if needed)
        if self.conflict_resolution_strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER and has_conflicts and self.tag_priority_order is None:
            while True:
                print("\nEnter media player priority order (highest priority first) by selecting numbers separated by commas.")
                print("Available media players:")
                for idx, entry in enumerate(unique_tag_keys, start=1):
                    print(f"  {idx}) {entry['player']}")
                priority_order = input("Your input: ").strip()
                selected_indices = [int(i) for i in priority_order.split(",")]
                if all(1 <= idx <= len(unique_tag_keys) for idx in selected_indices):
                    self.tag_priority_order = [self.tag_registry.get_config_value(unique_tag_keys[idx - 1]["key"]) for idx in selected_indices]
                    needs_saved = True
                    break
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
                if len(unique_tag_keys) == 1:
                    self.default_tag = unique_tag_keys[0]["key"]
                    break
                print("\nWhich tag should be treated as the default for writing ratings?")
                for idx, entry in enumerate(unique_tag_keys, start=1):
                    print(f"  {idx}) {entry['player']}")
                choice = input(f"Enter choice [1-{len(unique_tag_keys)}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(unique_tag_keys):
                    self.default_tag = self.tag_registry.get_config_value(unique_tag_keys[int(choice) - 1]["key"])
                    needs_saved = True
                    break
                print("Invalid choice. Please try again.")

        # Step 5: Save settings
        if needs_saved:
            while True:
                print("\nWould you like to save these settings to config.ini for future runs?")
                choice = input("Your choice [yes/no]: ").strip().lower()
                if choice in {"y", "yes"}:
                    config_mgr = get_manager().get_config_manager()
                    config_mgr.conflict_resolution_strategy = self.conflict_resolution_strategy
                    config_mgr.tag_write_strategy = self.tag_write_strategy
                    config_mgr.default_tag = self.default_tag
                    config_mgr.tag_priority_order = self.tag_priority_order
                    config_mgr.save_config()
                    break
                elif choice in {"n", "no"}:
                    break
                print("Invalid choice. Please enter 'y' or 'n'.")


class FileSystemProvider:
    """Adapter class for handling filesystem operations for audio files and playlists."""

    TRACK_EXT = {".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aac"}
    PLAYLIST_EXT = {".m3u", ".m3u8", ".pls"}

    def __init__(self):
        self.logger = logging.getLogger("PlexSync.FileSystemProvider")
        mgr = get_manager()
        self.config_mgr = mgr.get_config_manager()
        self.status_mgr = mgr.get_status_manager()
        self._audio_files = []
        self._playlist_files = []
        self.deferred_tracks = []

        self.id3_handler = ID3Handler(tagging_policy=self.config_mgr.config.to_dict())
        self.vorbis_handler = VorbisHandler()
        self._handlers = [self.id3_handler, self.vorbis_handler]

    # ------------------------------
    # Core Manager Dispatch
    # ------------------------------
    def _get_manager(self, audio_file: mutagen.FileType) -> Optional[AudioTagHandler]:
        """Determine the appropriate manager for the given audio file."""
        return next((handler for handler in self._handlers if handler.can_handle(audio_file)), None)

    # ------------------------------
    # File Handling (Low-Level Helpers)
    # ------------------------------
    def _open_track(self, file_path: Union[Path, str]) -> Optional[mutagen.FileType]:
        """Helper function to open an audio file using mutagen."""
        try:
            audio_file = mutagen.File(file_path, easy=False)
            if not audio_file:
                self.logger.warning(f"Unsupported audio format for file: {file_path}")
                raise ValueError(f"Unsupported audio format: {file_path}")
            return audio_file
        except Exception as e:
            self.logger.error(f"Error opening file {file_path}: {e}", exc_info=True)
            return None

    def _save_track(self, audio_file: mutagen.FileType) -> bool:
        """Helper function to save changes to an audio file."""
        if self.config_mgr.dry:
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
        path = self.config_mgr.path

        if not path:
            self.logger.error("Path is required for filesystem player")
            raise ValueError("Path is required for filesystem player")

        self.path = Path(path)
        if not self.path.exists():
            self.logger.error(f"Music directory not found: {path}")
            raise FileNotFoundError(f"Music directory not found: {path}")

        self.logger.info(f"Scanning {path} for audio files...")
        bar = self.status_mgr.start_phase("Collecting audio files", total=None)

        for file_path in self.path.rglob("*"):
            if file_path.suffix.lower() in self.TRACK_EXT:
                self._audio_files.append(file_path)
                bar.update()

        bar.close()
        self.logger.info(f"Found {len(self._audio_files)} audio files")
        return self._audio_files

    def scan_playlist_files(self) -> List[Path]:
        """Scan and list all playlist files in the directory."""
        playlist_path = self.config_mgr.playlist_path
        self.playlist_path = Path(playlist_path) if playlist_path else self.path
        self.playlist_path.mkdir(exist_ok=True)
        # TODO: add support for m3u8 and pls

        bar = self.status_mgr.start_phase("Collecting audio files", total=None)

        for file_path in self.playlist_path.rglob("*"):
            if file_path.suffix.lower() in self.PLAYLIST_EXT:
                self._playlist_files.append(file_path)
                bar.update()
        bar.close()
        self.logger.info(f"Found {len(self._playlist_files)} playlist files")
        self.logger.info(f"Using playlists directory: {playlist_path}")

    def finalize_scan(self) -> List[AudioTag]:
        """Finalize the scan by letting handlers resolve strategies and processing deferred tracks."""
        for handler in self._handlers:
            handler.finalize_rating_strategy(self.deferred_tracks)

        resolved_tracks = []
        bar = self.status_mgr.start_phase("Resolving rating conflicts", total=len(self.deferred_tracks)) if len(self.deferred_tracks) > 100 else None

        for deferred in self.deferred_tracks:
            track = deferred["track"]
            manager = deferred["manager"]

            context = {
                "raw": deferred.get("raw", {}),
                "normalized": deferred.get("normalized", {}),
                "handler": manager,
            }

            self.logger.debug(f"Resolving deferred rating for {track.artist} | {track.album} | {track.title}")

            track.rating = manager.get_normal_rating(context)
            self.update_metadata_in_file(track.file_path, rating=track.rating)
            resolved_tracks.append(track)

            bar.update() if bar else None

        bar.close() if bar else None
        return resolved_tracks

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
        audio_file = self._open_track(file_path)
        if not audio_file:
            return None

        manager = self._get_manager(audio_file)
        if not manager:
            self.logger.warning(f"No supported metadata found for {file_path}")
            return None

        tag, conflict_data = manager.read_tags(audio_file)
        if conflict_data:
            self.deferred_tracks.append(
                {
                    "track": tag,
                    "manager": manager,
                    "raw": conflict_data["raw"],
                    "normalized": conflict_data["normalized"],
                }
            )
        self.logger.debug(f"Successfully read metadata for {file_path}")
        return tag

    def update_metadata_in_file(self, file_path: Union[Path, str], metadata: Optional[dict] = None, rating: Optional[Rating] = None) -> Optional[mutagen.File]:
        """Update metadata and/or rating in the audio file."""
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
        try:
            lines = []
            track_relative = track.relative_to(self.path) if track.is_absolute() else track
            track_absolute = track.resolve()

            with playlist_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    # Skip lines that match either the relative or absolute path of the track
                    if line != str(track_relative) and line != str(track_absolute):
                        lines.append(line + "\n")

            with playlist_path.open("w", encoding="utf-8") as file:
                file.writelines(lines)

            self.logger.info(f"Removed track {track} from playlist {playlist_path}")
        except Exception as e:
            self.logger.error(f"Failed to remove track from playlist {playlist_path}: {e}")
