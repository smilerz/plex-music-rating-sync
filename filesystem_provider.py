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

    def display_name_for_id3_tag(self, id3_tag: str) -> str:
        tag_key = self.get_key_for_id3_tag(id3_tag)
        return self.get_player_name_for_key(tag_key) if tag_key else id3_tag

    def get_popm_email_for_key(self, key: str) -> Optional[str]:
        tag = self.get_id3_tag_for_key(key)
        if tag and tag.upper().startswith("POPM:"):
            return tag.split(":", 1)[1]
        return None


class AudioTagHandler(abc.ABC):
    def __init__(self, tagging_policy: Optional[dict] = None, **kwargs):
        from manager import manager

        self.mgr = manager
        self.logger = logging.getLogger(f"PlexSync.{self.__class__.__name__}")

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

    def _resolve_normalized_conflict(self, ratings_by_tag_key: dict[str, float]) -> Optional[float]:
        """Resolve a normalized rating from tag_key → rating mappings using the configured strategy."""
        strategy = self.conflict_resolution_strategy
        if not strategy:
            return None

        if not self.is_strategy_supported(strategy):
            self.logger.warning(f"Strategy '{strategy}' is not supported by {self.__class__.__name__}; falling back to 'HIGHEST'")
            strategy = ConflictResolutionStrategy.HIGHEST

        if strategy == ConflictResolutionStrategy.CHOICE:
            return self._manual_conflict_resolution(ratings_by_tag_key)

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
            return min(v for v in ratings_by_tag_key.values() if v > 0)

        if strategy == ConflictResolutionStrategy.AVERAGE:
            values = [v for v in ratings_by_tag_key.values() if v > 0]
            return round(sum(values) / len(values), 3) if values else None

        self.logger.warning(f"Unsupported conflict strategy: {strategy}")
        return None

    @abc.abstractmethod
    def _manual_conflict_resolution(self, ratings_by_tag_key: dict[str, float]) -> Optional[float]:
        """Interactive user choice for resolving conflicting ratings."""
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def get_5star_rating(rating: Optional[float]) -> float:
        """Convert a normalized 0–1 rating to a 0–5 scale."""
        return round(rating * 5, 1) if rating is not None else 0.0

    @staticmethod
    def has_rating_conflict(ratings_by_tag_key: dict[str, Optional[float]]) -> bool:
        """Detect if multiple unique, non-null ratings exist."""
        unique = {r for r in ratings_by_tag_key.values() if r is not None}
        return len(unique) > 1

    @abc.abstractmethod
    def can_handle(self, file: mutagen.FileType) -> bool:
        """Return True if this handler can process the given file."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def _extract_tags(self, audio_file: mutagen.FileType) -> tuple[AudioTag, dict]:
        """Extract metadata and normalized ratings from the audio file."""
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

    def get_normal_rating(self, context: dict) -> Optional[float]:
        """
        Resolve a final normalized rating from context:
        - If exactly one rating exists, return it.
        - If no valid ratings exist, return None.
        - Otherwise, defer to _resolve_rating.
        """
        normalized: dict[str, Optional[float]] = context.get("normalized", {})
        non_null_values = [v for v in normalized.values() if v is not None]
        unique_values = set(non_null_values)

        if len(unique_values) == 1:
            return non_null_values[0]
        elif not non_null_values:
            return None

        return self._resolve_rating(context)

    @abc.abstractmethod
    def _resolve_rating(self, context: dict) -> Optional[float]:
        """Subclasses must implement custom resolution logic based on context."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def apply_tags(self, audio_file: mutagen.FileType, metadata: Optional[dict], rating: Optional[float] = None) -> mutagen.FileType:
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

    def _manual_conflict_resolution(self, ratings: dict[str, float]) -> Optional[float]:
        print("\nConflicting ratings detected (Vorbis):")
        for tag, rating in ratings.items():
            print(f"  {tag:<20} : {self.get_5star_rating(rating):<5}")
        while True:
            choice = input("Select the correct rating (0-5, half-star increments allowed): ").strip()
            try:
                selected = float(choice)
                if 0 <= selected <= 5 and (selected * 2).is_integer():
                    return round(selected / 5, 3)
            except ValueError:
                pass
            print("Invalid input. Please enter a number between 0 and 5 in half-star increments.")

    def _extract_tags(self, audio_file: mutagen.FileType) -> tuple[AudioTag, dict]:
        """Extract raw and normalized ratings for Vorbis fields."""
        track = AudioTag.from_vorbis(audio_file, audio_file.filename)

        raw: dict[str, Optional[str]] = {}
        normalized: dict[str, Optional[float]] = {}

        for field in [VorbisField.FMPS_RATING, VorbisField.RATING]:
            raw_value = audio_file.get(field, [None])[0]
            raw[field] = raw_value
            normalized[field] = self._normalize_value(field, float(raw_value)) if raw_value is not None else None

        return track, {
            "raw": raw,
            "normalized": normalized,
            "handler": self,
        }

    def _resolve_rating(self, context: dict) -> Optional[float]:
        """Resolve a normalized rating from multiple Vorbis fields using normalized values and conflict strategy."""
        raw = context.get("raw", {})
        ratings: dict[str, float] = {}

        for field, raw_value in raw.items():
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                continue

            normalized = self._normalize_value(field, value)
            if normalized is not None:
                ratings[field] = normalized

        if not ratings:
            return None
        if len(set(ratings.values())) == 1:
            return next(iter(ratings.values()))

        return self._resolve_normalized_conflict(ratings)

    def _normalize_value(self, field: str, value: float) -> Optional[float]:
        """Normalize a value using already-inferred scale."""
        if field == VorbisField.FMPS_RATING:
            scale = self.fmps_rating_scale
        elif field == VorbisField.RATING:
            scale = self.rating_scale
        else:
            return None

        if not scale:
            return None

        return self._normalize_by_scale(value, scale)

    def finalize_rating_strategy(self, conflicts: list[dict]) -> None:
        self._print_summary()

        def resolve_scale_for(field: str) -> Optional[RatingScale]:
            field_stats = self.mgr.stats.get(f"VorbisHandler::scale_inferred::{field}")
            if not field_stats:
                return None
            max_count = max(field_stats.values())
            tied = [RatingScale[key] for key, count in field_stats.items() if count == max_count]
            sorted_scales = sorted(tied, key=lambda s: s != RatingScale.NORMALIZED)
            return sorted_scales[0]

        self.fmps_rating_scale = resolve_scale_for(VorbisField.FMPS_RATING)
        self.rating_scale = resolve_scale_for(VorbisField.RATING)
        self.aggressive_inference = True

    def apply_tags(self, audio_file: mutagen.FileType, metadata: Optional[dict] = None, rating: Optional[float] = None) -> mutagen.FileType:
        """Write metadata and ratings to Vorbis comments."""
        if metadata:
            for key, value in metadata.items():
                if value:
                    audio_file[key] = value

        if rating is not None:
            fmps_rating_value = self._native_rating(rating, VorbisField.FMPS_RATING)
            rating_value = self._native_rating(rating, VorbisField.RATING)

            audio_file[VorbisField.FMPS_RATING] = str(fmps_rating_value)
            audio_file[VorbisField.RATING] = str(rating_value)
            self.logger.info(f"Successfully wrote ratings: FMPS_RATING={fmps_rating_value}, RATING={rating_value}")

        return audio_file

    def _normalize_by_scale(self, value: float, scale: RatingScale) -> float:
        if scale == RatingScale.ZERO_TO_HUNDRED:
            return round(value / 100.0, 3)
        if scale == RatingScale.ZERO_TO_FIVE:
            return round(value / 5.0, 3)
        return round(value, 3)

    def _infer_scale(self, value: float, field: str) -> Optional[RatingScale]:
        """
        Infer the likely scale used for a raw rating value.
        Increments usage stats but does not assign the scale.
        """
        inferred = None
        if value > 10:
            inferred = RatingScale.ZERO_TO_HUNDRED
        elif 0 < value <= 1:
            inferred = RatingScale.NORMALIZED
        elif 1 < value <= 5:
            inferred = RatingScale.ZERO_TO_FIVE
        elif self.aggressive_inference:
            if value in {0.5, 1.0}:
                inferred = RatingScale.NORMALIZED
            elif value == 5.0:
                inferred = RatingScale.ZERO_TO_FIVE

        if inferred:
            self.mgr.stats.increment(f"VorbisHandler::scale_inferred::{field}::{inferred.name}")

        return inferred

    def _normalize_value(self, field: str, value: float) -> Optional[float]:
        """
        Normalize a rating value using the known scale if available.
        If not set, infer the scale for this single use (without setting global state),
        and track the inference for later resolution.
        """
        if field == VorbisField.FMPS_RATING:
            scale = self.fmps_rating_scale
        elif field == VorbisField.RATING:
            scale = self.rating_scale
        else:
            return None

        # If scale is known, apply it
        if scale:
            return self._normalize_by_scale(value, scale)

        # Else infer one just for this value
        inferred = self._infer_scale(value, field)
        return self._normalize_by_scale(value, inferred) if inferred else None

    def _native_rating(self, normalized: float, field: str) -> str:
        """Convert normalized 0-1 rating to appropriate scale for the target field."""
        scale = self.fmps_rating_scale if field.upper() == VorbisField.FMPS_RATING else self.rating_scale

        if scale == RatingScale.NORMALIZED:
            return round(normalized, 3)
        elif scale == RatingScale.ZERO_TO_FIVE:
            return round(normalized * 5.0, 1)
        elif scale == RatingScale.ZERO_TO_HUNDRED:
            return round(normalized * 100.0)
        else:
            self.logger.warning(f"Unknown scale for {field}, writing as normalized")
            return round(normalized, 3)

    def _print_summary(self) -> None:
        print("\nFLAC, OGG (and other Vorbis formats) Ratings:")

        # Inferred Scales
        rating_counts = self.mgr.stats.get("VorbisHandler::scale_inferred::RATING")
        fmps_counts = self.mgr.stats.get("VorbisHandler::scale_inferred::FMPS_RATING")

        if rating_counts or fmps_counts:
            print("  Inferred Rating Scales:")
            for field, counts in [("RATING", rating_counts), ("FMPS_RATING", fmps_counts)]:
                if not counts:
                    continue
                print(f"    {field}:")
                for scale, count in counts.items():
                    print(f"      - {scale.title()}: {count}")

        # TODO: any stats that need summarized should be collected with mgr.stats
        # ambiguous = deferral_info.get("ambiguous", 0)
        # conflict = deferral_info.get("conflict", 0)
        # if ambiguous:
        #     print(f"  Files with ambiguous ratings: {ambiguous}")
        # if conflict:
        #     print(f"  Files with conflicting ratings: {conflict}")

        if self.conflict_resolution_strategy:
            print("\n  Conflict Resolution Strategy:")
            print(f"    {self.conflict_resolution_strategy.display}")


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

    def __init__(self, tagging_policy: Optional[dict] = None, **kwargs):
        super().__init__(tagging_policy=tagging_policy, **kwargs)

        self.discovered_rating_tags = set()
        self.tag_registry = ID3TagRegistry()

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

    def _manual_conflict_resolution(self, ratings: dict[str, float]) -> Optional[float]:
        print("\nConflicting ratings detected (ID3):")
        for tag_key, rating in ratings.items():
            print(f"  {self.tag_registry.display_name_for_id3_tag(tag_key):<30} : {self.get_5star_rating(rating):<5}")
        while True:
            choice = input("Select the correct rating (0-5, half-star increments allowed): ").strip()
            try:
                selected = float(choice)
                if 0 <= selected <= 5 and (selected * 2).is_integer():
                    return round(selected / 5, 3)
            except ValueError:
                pass
            print("Invalid input. Please enter a number between 0 and 5 in half-star increments.")

    def _extract_rating_value(self, id3_tag: str, frame: Union[POPM, TXXX]) -> Optional[float]:
        if isinstance(frame, POPM):
            return self.rating_from_popm(frame.rating) if frame.rating else None
        elif isinstance(frame, TXXX) and id3_tag.upper() == "TXXX:RATING":
            try:
                return float(frame.text[0]) / 5
            except (ValueError, IndexError):
                return None
        return None

    # ------------------------------
    # Rating Conflict Handling
    # ------------------------------
    def _resolve_rating(self, context: dict) -> Optional[float]:
        """Resolve normalized rating from multiple ID3 tags using conflict strategy."""
        ratings: dict[str, float] = {}

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
        track = AudioTag.from_id3(audio_file.tags, audio_file.filename, duration=int(audio_file.info.length))

        raw: dict[str, Optional[str]] = {}
        normalized: dict[str, Optional[float]] = {}

        for tag_key, frame in audio_file.tags.items():
            if not (tag_key.startswith("POPM:") or tag_key == "TXXX:RATING"):
                continue

            key = self.tag_registry.register(tag_key)
            self.discovered_rating_tags.add(tag_key)
            self.mgr.stats.increment(f"FileSystemPlayer::tags_used::{key}")

            if isinstance(frame, POPM):
                raw_value = str(frame.rating)
            elif isinstance(frame, TXXX):
                raw_value = frame.text[0] if frame.text else None
            else:
                raw_value = None

            raw[key] = raw_value
            normalized[key] = self._normalize_raw_rating(tag_key, raw_value)

        return track, {
            "raw": raw,
            "normalized": normalized,
            "handler": self,
        }

    def _normalize_raw_rating(self, tag_key: str, raw_value: Optional[str]) -> Optional[float]:
        """Normalize a raw rating using tag type awareness (TXXX or POPM)."""
        if raw_value is None:
            return None

        try:
            val = float(raw_value)
        except (ValueError, TypeError):
            return None

        tag_upper = tag_key.upper()
        if tag_upper == "TXXX:RATING":
            return round(val / 5.0, 3)

        if tag_upper.startswith("POPM:"):
            return self.rating_from_popm(int(val))

        self.logger.warning(f"Unknown tag type for normalization: {tag_key}")
        return None

    def apply_tags(self, audio_file: mutagen.FileType, metadata: dict, rating: Optional[float] = None) -> mutagen.FileType:
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

    def apply_rating(self, audio_file: mutagen.FileType, rating: float, tag_keys: set[str]) -> mutagen.FileType:
        self.logger.debug(f"Applying normalized rating {rating} to file: {audio_file.filename}")

        for tag_key in tag_keys:
            if not tag_key:
                continue  # Invalid key; no log needed

            tag = self.tag_registry.get_id3_tag_for_key(tag_key)
            if not tag:
                self.logger.warning(f"Tag key '{tag_key}' has no registered tag string; skipping.")
                continue

            # Determine tag type and apply rating
            if tag.upper() == "TXXX:RATING":
                txt_rating = str(self.get_5star_rating(rating))
                if tag in audio_file.tags:
                    self.logger.debug(f"Updating TXXX:RATING ({tag_key}) to value: {txt_rating}")
                    audio_file.tags[tag].text = [txt_rating]
                else:
                    self.logger.debug(f"Creating TXXX:RATING ({tag_key}) with value: {txt_rating}")
                    new_txxx = TXXX(encoding=1, desc="RATING", text=[txt_rating])
                    audio_file.tags.add(new_txxx)

            elif tag.upper().startswith("POPM:"):
                popm_rating = int(self.rating_to_popm(rating))
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

    # ------------------------------
    # Utility Methods
    # ------------------------------

    def can_handle(self, file: mutagen.FileType) -> bool:
        return isinstance(file, ID3FileType) or (hasattr(file, "tags") and any(tag.startswith("TXXX:") or tag.startswith("POPM:") for tag in file.tags.keys()))

    def _print_summary(self) -> None:
        print("\nMP3 Ratings:")

        tag_keys_used = self.mgr.stats.get("FileSystemPlayer::tags_used")
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
        tag_keys_used = self.mgr.stats.get("FileSystemPlayer::tags_used")
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
                                print(f"\t{self.tag_registry.display_name_for_id3_tag(tag_key):<30} : {self.get_5star_rating(rating):<5}")
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

        self.logger = logging.getLogger("PlexSync.FileSystemProvider")
        self._audio_files = []
        self._playlist_files = []
        self.deferred_tracks = []

        self.id3_handler = ID3Handler(tagging_policy=self.mgr.config.to_dict())
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
        """Finalize the scan by letting handlers resolve strategies and processing deferred tracks."""
        for handler in self._handlers:
            handler.finalize_rating_strategy(self.deferred_tracks)

        resolved_tracks = []
        bar = self.mgr.status.start_phase("Resolving rating conflicts", total=len(self.deferred_tracks)) if len(self.deferred_tracks) > 100 else None

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

    def update_metadata_in_file(self, file_path: Union[Path, str], metadata: Optional[dict] = None, rating: Optional[float] = None) -> Optional[mutagen.File]:
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
