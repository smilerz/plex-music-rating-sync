import abc
import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

    # ------------------------------
    # Lifecycle Phase 0: Capability
    # ------------------------------
    @abc.abstractmethod
    def can_handle(self, file: mutagen.FileType) -> bool:
        """Return True if this handler supports the given file."""
        raise NotImplementedError("can_handle() not implemented")

    # ------------------------------
    # Phase 1: Extraction
    # ------------------------------
    @abc.abstractmethod
    def extract_metadata(self, audio_file: mutagen.FileType) -> Tuple[AudioTag, Dict[str, Any]]:
        """
        Read *only* raw rating‐tag values from the file.
        Return (AudioTag(without rating), {tag_key: raw_value, …})
        """
        raise NotImplementedError("extract_metadata() not implemented")

    # ------------------------------
    # Phase 2: Resolution
    # ------------------------------
    def resolve_rating(self, raw_ratings: Dict[str, Union[str, float]], track: AudioTag) -> Optional[Rating]:
        """Attempt to resolve a final rating based on raw values and context."""
        normalized: Dict[str, Rating] = {}
        failed: set[str] = set()

        for tag_key, raw_value in raw_ratings.items():
            rating = self._try_normalize(raw_value, tag_key)
            if rating is not None:
                normalized[tag_key] = rating
            else:
                failed.add(tag_key)

        if not normalized:
            # all failed = unrated
            return Rating.unrated()

        if failed:
            # some succeeded, some failed = ambiguous
            self.logger.debug(f"Rating ambiguity in {track.ID}: could not normalize {failed}, deferring resolution")
            return None

        unique_ratings = set(normalized.values())
        if len(unique_ratings) == 1:
            return next(iter(unique_ratings))

        # genuine conflict
        return self._resolve_conflict(normalized, track)

    @abc.abstractmethod
    def _try_normalize(self, raw_value: Union[str, float], tag_key: str) -> Optional[Rating]:
        """
        Attempt to convert a single raw tag_value into a Rating.
        Subclasses implement scale inference or parsing here.
        """
        raise NotImplementedError("Subclasses must implement _try_normalize()")

    def _resolve_conflict(self, ratings_by_tag: Dict[str, Rating], track: AudioTag) -> Optional[Rating]:
        strat = self.conflict_resolution_strategy
        if not strat:
            return None

        if not self.is_strategy_supported(strat):
            self.logger.warning(f"Unsupported strategy {strat}; defaulting to HIGHEST")
            strat = ConflictResolutionStrategy.HIGHEST

        if strat == ConflictResolutionStrategy.CHOICE:
            return self._manual_conflict_resolution(ratings_by_tag, track)

        if strat == ConflictResolutionStrategy.PRIORITIZED_ORDER:
            if not self.tag_priority_order:
                self.logger.warning("No tag_priority_order for PRIORITIZED_ORDER")
                raise ValueError("No tag_priority_order for PRIORITIZED_ORDER")
            for key in self.tag_priority_order:
                if key in ratings_by_tag:
                    return ratings_by_tag[key]
            return Rating.unrated()

        if strat == ConflictResolutionStrategy.HIGHEST:
            return max(ratings_by_tag.values())

        if strat == ConflictResolutionStrategy.LOWEST:
            return min(ratings_by_tag.values())

        if strat == ConflictResolutionStrategy.AVERAGE:
            vals = [r.to_float() for r in ratings_by_tag.values()]
            return Rating(sum(vals) / len(vals)) if vals else None

        self.logger.warning(f"Unknown conflict strategy {strat}")
        return None

    def _manual_conflict_resolution(self, ratings_by_tag: Dict[str, Rating], track: AudioTag) -> Optional[Rating]:
        """
        Default interactive chooser: list each tag and rating, let user pick or skip.
        Subclasses may override for custom UI.
        """
        items = list(ratings_by_tag.items())
        print(f"\nConflicting ratings for {track.artist} – {track.album} – {track.title}:")
        for idx, (key, rating) in enumerate(items, start=1):
            print(f"  {idx}) {key:<20} : {rating.to_display()}")
        print(f"  {len(items)+1}) Skip / defer")

        while True:
            choice = input(f"Select [1-{len(items)+1}]: ").strip()
            if choice.isdigit():
                i = int(choice)
                if 1 <= i <= len(items):
                    return items[i - 1][1]
                if i == len(items) + 1:
                    return None
            print("Invalid choice; try again.")

    def is_strategy_supported(self, strategy: ConflictResolutionStrategy) -> bool:
        """Override if a strategy isn’t valid for this format."""
        return True

    # ------------------------------
    # Phase 3: Orchestration
    # ------------------------------
    def read_tags(self, audio_file: mutagen.FileType) -> Tuple[AudioTag, Optional[Dict[str, Any]]]:
        """
        Full tag‐reading workflow:
          1. extract_metadata()
          2. resolve_rating()
          3. return (AudioTag, None) if we got a rating,
             or (AudioTag, raw_ratings) to defer if unresolved.
        """
        track, raw = self.extract_metadata(audio_file)
        rating = self.resolve_rating(raw, track)
        if rating is not None and not rating.is_unrated:
            track.rating = rating
            return track, None
        return track, raw

    # ------------------------------
    # Phase 4: Writing
    # ------------------------------
    @abc.abstractmethod
    def apply_tags(self, audio_file: mutagen.FileType, metadata: Optional[Dict[str, Any]], rating: Optional[Rating] = None) -> mutagen.FileType:
        """Write metadata fields and the resolved rating back to the file."""
        raise NotImplementedError("apply_tags() not implemented")

    # ------------------------------
    # Phase 5: Finalization
    # ------------------------------
    @abc.abstractmethod
    def finalize_rating_strategy(self, conflicts: List[dict]) -> None:
        """
        After a full scan, infer dominant scales or prompt for strategy settings.
        """
        raise NotImplementedError("finalize_rating_strategy() not implemented")

    # ------------------------------
    # Utilities
    # ------------------------------
    def _show_conflicts(self, conflicts: List[dict]) -> None:
        print("\nConflicts:")
        for conflict in conflicts:
            if conflict.get("handler") is not self:
                continue
            track = conflict["track"]
            raw = conflict.get("raw", {})
            print(f"\n  {track.artist} — {track.album} — {track.title}")
            for tag_key, raw_val in raw.items():
                rating = self._try_normalize(raw_val, tag_key)
                label = self._get_label(tag_key)
                print(f"    {label:<30} : {rating.to_display() if rating else str(raw_val)}")

    def _get_label(self, tag_key: str) -> str:
        """Override in subclass to map tag keys to user-friendly labels."""
        return tag_key


class VorbisHandler(AudioTagHandler):
    def __init__(self, tagging_policy: Optional[dict] = None, **kwargs):
        super().__init__(tagging_policy=tagging_policy, **kwargs)
        self.fmps_rating_scale: Optional[RatingScale] = None
        self.rating_scale: Optional[RatingScale] = None
        self.aggressive_inference = False
        self.conflict_resolution_strategy = ConflictResolutionStrategy.HIGHEST

    # ------------------------------
    # Capability Detection
    # ------------------------------
    def can_handle(self, file: mutagen.FileType) -> bool:
        return hasattr(file, "tags") and (VorbisField.FMPS_RATING.value in file.tags or VorbisField.RATING.value in file.tags)

    # ------------------------------
    # Phase 1: Metadata Extraction
    # ------------------------------
    def extract_metadata(self, audio_file: mutagen.FileType) -> Tuple[AudioTag, Dict[str, Any]]:
        tag = self._get_audiotag(audio_file, audio_file.filename)
        raw: Dict[str, Any] = {}
        for field in (VorbisField.FMPS_RATING, VorbisField.RATING):
            key = field.value
            if key in audio_file.tags:
                raw[key] = audio_file.get(key, [None])[0]
        return tag, raw

    # ------------------------------
    # Phase 2: Normalization Hook
    # ------------------------------
    def _try_normalize(self, raw_value: Union[str, float], tag_key: str) -> Optional[Rating]:
        scale = self.fmps_rating_scale if tag_key == VorbisField.FMPS_RATING.value else self.rating_scale
        rating = Rating.try_create(raw_value, scale=scale, aggressive=self.aggressive_inference)

        if not scale and rating:
            self.stats_mgr.increment(f"VorbisHandler::scale_inferred::{tag_key}::{rating.scale.name}")

        if rating is not None:
            return rating

        if self.aggressive_inference:
            return Rating.unrated()

        return None

    def is_strategy_supported(self, strategy: ConflictResolutionStrategy) -> bool:
        return strategy != ConflictResolutionStrategy.PRIORITIZED_ORDER

    # ------------------------------
    # Phase 3: Finalization / Scale Inference
    # ------------------------------
    def finalize_rating_strategy(self, conflicts: List[dict]) -> None:
        conflicts = [c for c in conflicts if c.get("handler") is self]
        self._print_summary()

        def pick_scale(field: VorbisField) -> Optional[RatingScale]:
            stats = self.stats_mgr.get(f"VorbisHandler::scale_inferred::{field.value}")
            if not stats:
                return None
            max_count = max(stats.values())
            tied = [RatingScale[k] for k, v in stats.items() if v == max_count]
            return sorted(tied, key=lambda s: s != RatingScale.NORMALIZED)[0]

        self.fmps_rating_scale = pick_scale(VorbisField.FMPS_RATING)
        self.rating_scale = pick_scale(VorbisField.RATING)
        self.aggressive_inference = True

    def _print_summary(self) -> None:
        """Print the stats of inferred scales and current strategy."""
        print("\nFLAC/OGG (Vorbis) Ratings Summary:")
        for field in (VorbisField.RATING, VorbisField.FMPS_RATING):
            stats = self.stats_mgr.get(f"VorbisHandler::scale_inferred::{field.value}")
            if stats:
                print(f"  {field.value} scale usage:")
                for scale, count in stats.items():
                    print(f"    - {scale}: {count}")
        if self.conflict_resolution_strategy:
            print(f"\n  Conflict Strategy: {self.conflict_resolution_strategy.display}\n")

    # ------------------------------
    # Phase 4: Writing
    # ------------------------------
    def apply_tags(self, audio_file: mutagen.FileType, metadata: Optional[Dict[str, Any]] = None, rating: Optional[Rating] = None) -> mutagen.FileType:
        """
        Write metadata fields and resolved rating back into the Vorbis tags.
        """
        if metadata:
            for k, v in metadata.items():
                if v is not None:
                    audio_file[k] = v

        if rating is not None:
            if self.fmps_rating_scale is not None:
                audio_file[VorbisField.FMPS_RATING.value] = rating.to_str(self.fmps_rating_scale)
            if self.rating_scale is not None:
                audio_file[VorbisField.RATING.value] = rating.to_str(self.rating_scale)

        return audio_file

    # ------------------------------
    # Utility
    # ------------------------------
    def _get_audiotag(self, audio_file: mutagen.FileType, file_path: str) -> AudioTag:
        track_num = audio_file.get(VorbisField.TRACKNUMBER.value, ["0"])[0]
        duration = getattr(getattr(audio_file, "info", None), "length", -1)
        return AudioTag(
            artist=audio_file.get(VorbisField.ARTIST.value, [""])[0],
            album=audio_file.get(VorbisField.ALBUM.value, [""])[0],
            title=audio_file.get(VorbisField.TITLE.value, [""])[0],
            file_path=file_path,
            ID=file_path.lower(),
            track=int(track_num.split("/")[0]),
            duration=int(duration or -1),
            rating=None,
        )


class ID3Handler(AudioTagHandler):
    def __init__(self, tagging_policy: Optional[dict] = None, **kwargs):
        super().__init__(tagging_policy=tagging_policy, **kwargs)
        self.tag_registry = ID3TagRegistry()
        self.discovered_rating_tags: set[str] = set()

    # ------------------------------
    # Capability Detection
    # ------------------------------
    def can_handle(self, file: mutagen.FileType) -> bool:
        return isinstance(file, ID3FileType) or (hasattr(file, "tags") and any(key.startswith("POPM:") or key == "TXXX:RATING" for key in getattr(file, "tags", {}).keys()))

    # ------------------------------
    # Phase 1: Metadata Extraction
    # ------------------------------
    def extract_metadata(self, audio_file: mutagen.FileType) -> Tuple[AudioTag, Dict[str, Any]]:
        # Build the AudioTag skeleton
        track = self._get_audiotag(
            audio_file.tags,
            audio_file.filename,
            duration=int(getattr(audio_file.info, "length", -1)),
        )

        raw: Dict[str, Any] = {}
        for frame_key, frame in audio_file.tags.items():
            # Only pull rating frames
            if not (frame_key.startswith("POPM:") or frame_key == "TXXX:RATING"):
                continue

            tag_key = self.tag_registry.register(frame_key)
            self.discovered_rating_tags.add(tag_key)
            self.stats_mgr.increment(f"ID3Handler::tags_used::{tag_key}")

            if isinstance(frame, POPM):
                raw[tag_key] = str(frame.rating)
            elif isinstance(frame, TXXX):
                raw[tag_key] = frame.text[0] if frame.text else None

        return track, raw

    # ------------------------------
    # Phase 2: Normalization Hook
    # ------------------------------
    def _try_normalize(self, raw_value: Union[str, float], tag_key: str) -> Optional[Rating]:
        id3_tag = self.tag_registry.get_id3_tag_for_key(tag_key) or ""
        if id3_tag.upper().startswith("POPM:"):
            scale = RatingScale.POPM
        else:
            scale = RatingScale.ZERO_TO_FIVE
        return Rating.try_create(raw_value, scale=scale) or Rating.unrated()

    # ------------------------------
    # Phase 3: Finalization / Scale Inference
    # ------------------------------
    def finalize_rating_strategy(self, conflicts: List[dict]) -> None:
        conflicts = [c for c in conflicts if c.get("handler") is self]
        tag_counts = self.stats_mgr.get("ID3Handler::tags_used") or {}
        unique = [
            {
                "key": k,
                "tag": self.tag_registry.get_id3_tag_for_key(k),
                "player": self.tag_registry.get_player_name_for_key(k),
            }
            for k in tag_counts
        ]
        has_multiple = len(unique) > 1
        has_conflicts = len(conflicts) > 0
        needs_save = False

        self._print_summary()

        # Step 1: conflict strategy
        if has_conflicts and self.conflict_resolution_strategy is None:
            while True:
                print("\nChoose conflict resolution strategy:")
                for i, strat in enumerate(ConflictResolutionStrategy, start=1):
                    print(f"  {i}) {strat.display}")
                print(f"  {len(ConflictResolutionStrategy)+1}) Show conflicts")
                print(f"  {len(ConflictResolutionStrategy)+2}) Ignore conflicts")
                choice = input(f"Select [1-{len(ConflictResolutionStrategy)+2}]: ").strip()
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(ConflictResolutionStrategy):
                        self.conflict_resolution_strategy = list(ConflictResolutionStrategy)[idx - 1]
                        needs_save = True
                        break
                    if idx == len(ConflictResolutionStrategy) + 1:
                        self._show_conflicts(conflicts)
                        continue
                    if idx == len(ConflictResolutionStrategy) + 2:
                        break
                print("Invalid choice.")

        # Step 2: priority order
        if self.conflict_resolution_strategy == ConflictResolutionStrategy.PRIORITIZED_ORDER and has_multiple and not self.tag_priority_order:
            while True:
                print("\nEnter tag priority (highest-first), comma-separated:")
                for i, info in enumerate(unique, start=1):
                    print(f"  {i}) {info['player']}")
                sel = input("Your order: ").strip()
                try:
                    idxs = [int(x) for x in sel.split(",")]
                    if all(1 <= x <= len(unique) for x in idxs):
                        self.tag_priority_order = [unique[x - 1]["key"] for x in idxs]
                        needs_save = True
                        break
                except ValueError:
                    pass
                print("Invalid input.")

        # Step 3: write strategy
        if has_multiple and not self.tag_write_strategy:
            while True:
                print("\nChoose write strategy:")
                for i, strat in enumerate(TagWriteStrategy, start=1):
                    print(f"  {i}) {strat.display}")
                c = input(f"Select [1-{len(TagWriteStrategy)}]: ").strip()
                if c.isdigit() and 1 <= int(c) <= len(TagWriteStrategy):
                    self.tag_write_strategy = list(TagWriteStrategy)[int(c) - 1]
                    needs_save = True
                    break
                print("Invalid choice.")

        # Step 4: default tag
        if self.tag_write_strategy and self.tag_write_strategy.requires_default_tag() and not self.default_tag:
            while True:
                print("\nSelect default tag for writing:")
                for i, info in enumerate(unique, start=1):
                    print(f"  {i}) {info['player']}")
                c = input(f"Select [1-{len(unique)}]: ").strip()
                if c.isdigit() and 1 <= int(c) <= len(unique):
                    self.default_tag = unique[int(c) - 1]["key"]
                    needs_save = True
                    break
                print("Invalid choice.")

        # Step 5: save config
        cfg = get_manager().get_config_manager()
        cfg.conflict_resolution_strategy = self.conflict_resolution_strategy
        cfg.tag_write_strategy = self.tag_write_strategy
        cfg.default_tag = self.default_tag
        cfg.tag_priority_order = self.tag_priority_order
        if needs_save:
            while True:
                yn = input("\nSave settings to config.ini? (y/n): ").strip().lower()
                if yn in ("y", "yes"):
                    cfg.save_config()
                    break
                if yn in ("n", "no"):
                    break
                print("Please enter y or n.")

    def _print_summary(self) -> None:
        print("\nMP3 Ratings Summary:")
        used = self.stats_mgr.get("ID3Handler::tags_used") or {}
        for key, cnt in used.items():
            player = self.tag_registry.get_player_name_for_key(key) or key
            print(f"  {player}: {cnt}")
        if self.conflict_resolution_strategy:
            print(f"\nConflict strategy: {self.conflict_resolution_strategy.display}")
        if self.tag_write_strategy:
            print(f"Write strategy: {self.tag_write_strategy.display}")
        if self.default_tag:
            dt = self.tag_registry.get_player_name_for_key(self.default_tag) or self.default_tag
            print(f"Default tag: {dt}")

    def _get_label(self, tag_key: str) -> str:
        return self.tag_registry.display_name(tag_key)

    # ------------------------------
    # Phase 4: Writing
    # ------------------------------
    def apply_tags(self, audio_file: mutagen.FileType, metadata: Optional[Dict[str, Any]] = None, rating: Optional[Rating] = None) -> mutagen.FileType:
        if metadata:
            for k, v in metadata.items():
                if v is not None:
                    audio_file[k] = v

        if rating is not None:
            if self.tag_write_strategy == TagWriteStrategy.OVERWRITE_DEFAULT:
                self._remove_existing_id3_tags(audio_file)
                to_write = {self.default_tag}
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_ALL:
                to_write = self.discovered_rating_tags
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_EXISTING:
                existing = {self.tag_registry.register(fk) for fk, fr in audio_file.tags.items() if isinstance(fr, POPM) or (isinstance(fr, TXXX) and fk == "TXXX:RATING")}
                to_write = existing or {self.default_tag}
            elif self.tag_write_strategy == TagWriteStrategy.WRITE_DEFAULT:
                to_write = {self.default_tag}
            else:
                to_write = set()

            if to_write:
                audio_file = self._apply_rating(audio_file, rating, to_write)

        return audio_file

    def _apply_rating(self, audio_file: mutagen.FileType, rating: Rating, tag_keys: set[str]) -> mutagen.FileType:
        for key in tag_keys:
            id3_tag = self.tag_registry.get_id3_tag_for_key(key)
            if not id3_tag:
                self.logger.warning(f"No tag for key '{key}'")
                continue

            if id3_tag == "TXXX:RATING":
                txt = rating.to_str(RatingScale.ZERO_TO_FIVE)
                if "TXXX:RATING" in audio_file.tags:
                    audio_file.tags["TXXX:RATING"].text = [txt]
                else:
                    audio_file.tags.add(TXXX(encoding=1, desc="RATING", text=[txt]))

            elif id3_tag.upper().startswith("POPM:"):
                pr = rating.to_int(RatingScale.POPM)
                email = self.tag_registry.get_popm_email_for_key(key) or ""
                if id3_tag in audio_file.tags:
                    audio_file.tags[id3_tag].rating = pr
                else:
                    audio_file.tags.add(POPM(email=email, rating=pr, count=0))

        return audio_file

    def _remove_existing_id3_tags(self, audio_file: mutagen.FileType) -> None:
        for frame in list(audio_file.keys()):
            if frame == "TXXX:RATING" or frame.startswith("POPM:"):
                del audio_file[frame]

    # ------------------------------
    # Utility
    # ------------------------------
    def _get_audiotag(self, audio_file: mutagen.FileType, file_path: str, duration: int = -1) -> AudioTag:
        def _safe(field: str) -> str:
            return audio_file.get(field).text[0] if audio_file.get(field) else ""

        track = _safe("TRCK") or "0"
        return AudioTag(
            artist=_safe("TPE1"),
            album=_safe("TALB"),
            title=_safe("TIT2"),
            file_path=file_path,
            rating=None,
            ID=file_path.lower(),
            track=int(track.split("/")[0]),
            duration=duration,
        )


class FileSystemProvider:
    """Adapter class for handling filesystem operations for audio files and playlists."""

    AUDIO_EXT = {".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aac"}
    PLAYLIST_EXT = {".m3u", ".m3u8", ".pls"}

    def __init__(self):
        self.logger = logging.getLogger("PlexSync.FileSystemProvider")
        mgr = get_manager()
        self.config_mgr = mgr.get_config_manager()
        self.status_mgr = mgr.get_status_manager()
        self._media_files = []
        self._playlist_title_map: dict[str, Path] = {}
        self.deferred_tracks = []

        self.id3_handler = ID3Handler(tagging_policy=self.config_mgr.to_dict())
        self.vorbis_handler = VorbisHandler()
        self._handlers = [self.id3_handler, self.vorbis_handler]

    # ------------------------------
    # Core Handler Dispatch
    # ------------------------------
    def _get_handler(self, audio_file: mutagen.FileType) -> Optional[AudioTagHandler]:
        """Determine the appropriate handler for the given audio file."""
        return next((handler for handler in self._handlers if handler.can_handle(audio_file)), None)

    # ------------------------------
    # File Handling (Low-Level Helpers)
    # ------------------------------
    def _open_audio_file(self, file_path: Union[Path, str]) -> Optional[mutagen.FileType]:
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

    def _save_audio_file(self, audio_file: mutagen.FileType) -> bool:
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
    def scan_media_files(self) -> None:
        """Scan configured paths for audio and playlist files without duplication or scope leakage."""
        self._media_files = []

        self.path = audio_root = Path(self.config_mgr.path).resolve()
        self.playlist_path = playlist_root = Path(self.config_mgr.playlist_path).resolve() if self.config_mgr.playlist_path else audio_root

        scanned_files: set[Path] = set()
        bar = self.status_mgr.start_phase("Scanning media files", total=None)

        # Track whether playlist_root is fully covered by audio_root
        playlist_scanned_during_audio = playlist_root.is_relative_to(audio_root)

        # Scan audio_root for audio and playlists (if they fall within playlist_root)
        for file_path in audio_root.rglob("*"):
            resolved = file_path.resolve()
            if not file_path.is_file() or resolved in scanned_files:
                continue
            scanned_files.add(resolved)

            ext = file_path.suffix.lower()
            if ext in self.AUDIO_EXT:
                self._media_files.append(resolved)
            if ext in self.PLAYLIST_EXT and file_path.is_relative_to(playlist_root):
                self._media_files.append(resolved)

            bar.update()

        # If playlist_root wasn't already covered, scan it independently (playlists only)
        if not playlist_scanned_during_audio:
            for file_path in playlist_root.rglob("*"):
                resolved = file_path.resolve()
                if not file_path.is_file() or resolved in scanned_files:
                    continue
                scanned_files.add(resolved)

                ext = file_path.suffix.lower()
                if ext in self.PLAYLIST_EXT:
                    self._media_files.append(file_path)

                bar.update()

        bar.close()

        self.logger.info(f"Found {len(self.get_tracks())} audio files")
        self.logger.info(f"Found {len(self.get_playlists())} playlist files")

    def get_tracks(self) -> List[Path]:
        """Return all discovered audio files."""
        return [t for t in self._media_files if t.suffix.lower() in self.AUDIO_EXT]

    def get_playlists(self, title: Optional[str] = None, path: Optional[Union[str, Path]] = None) -> List[Playlist]:
        """Return all discovered Playlist objects, optionally filtered by title or resolved path."""
        playlist_paths = [p for p in self._media_files if p.suffix.lower() in self.PLAYLIST_EXT]

        if title:
            matched_path = self._playlist_title_map.get(title)
            playlist_paths = [matched_path] if matched_path else []

        elif path:
            resolved = Path(path).resolve()
            playlist_paths = [p for p in playlist_paths if p.resolve() == resolved]

        # Convert paths to Playlist objects
        playlists = []
        for p in playlist_paths:
            playlist = self.read_playlist_metadata(p)
            if playlist:
                playlists.append(playlist)

        return playlists

    def finalize_scan(self) -> List[AudioTag]:
        """Finalize the scan by letting handlers resolve strategies and processing deferred tracks."""
        # give each handler a chance to configure conflict strategy / infer scales
        for handler in self._handlers:
            handler.finalize_rating_strategy(self.deferred_tracks)

        resolved_tracks: List[AudioTag] = []
        total = len(self.deferred_tracks)
        bar = self.status_mgr.start_phase("Resolving rating conflicts", total=total) if total > 100 else None

        for entry in self.deferred_tracks:
            track = entry["track"]
            handler = entry["handler"]
            raw = entry["raw"]

            self.logger.debug(f"Resolving deferred rating for {track.artist} | {track.album} | {track.title}")
            # new lifecycle: normalize & resolve from raw only
            rating = handler.resolve_rating(raw, track)
            track.rating = rating

            # write back whichever rating (or unrated) was chosen
            self.update_track_metadata(track.file_path, rating=track.rating)
            resolved_tracks.append(track)

            bar.update() if bar else None

        bar.close() if bar else None

        return resolved_tracks

    # ------------------------------
    # Metadata Access and Update
    # ------------------------------
    def _get_playlist_title(self, path: Path, base_title: Optional[str]) -> str:
        candidate = base_title
        rel_path = path.relative_to(self.playlist_path)
        folders = list(reversed(rel_path.parts[:-1]))  # exclude file

        parts = [base_title]
        while candidate in self._playlist_title_map and self._playlist_title_map[candidate] != path:
            if not folders:
                self.logger.warning(f"Could not disambiguate duplicate playlist title: {base_title}")
                break
            parts.insert(0, folders.pop(0))
            candidate = ".".join(parts)

        return candidate

    def read_track_metadata(self, file_path: Union[Path, str]) -> Optional[AudioTag]:
        audio_file = self._open_audio_file(file_path)
        if not audio_file:
            return None

        handler = self._get_handler(audio_file)
        if not handler:
            self.logger.warning(f"No supported metadata found for {file_path}")
            return None

        tag, raw_ratings = handler.read_tags(audio_file)
        if raw_ratings:
            # only raw, no normalized any more
            self.deferred_tracks.append(
                {
                    "track": tag,
                    "handler": handler,
                    "raw": raw_ratings,
                }
            )

        self.logger.debug(f"Successfully read metadata for {file_path}")
        return tag

    def update_track_metadata(self, file_path: Union[Path, str], metadata: Optional[dict] = None, rating: Optional[Rating] = None) -> Optional[mutagen.File]:
        """Update metadata and/or rating in the audio file."""
        # TODO: add logic to handle relative paths
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        audio_file = self._open_audio_file(file_path)
        if not audio_file:
            self.logger.warning(f"Failed to open file for metadata update: {file_path}")
            return None

        handler = self._get_handler(audio_file)
        if not handler:
            self.logger.warning(f"Cannot update metadata for unsupported format: {file_path}")
            return None

        updated_file = handler.apply_tags(audio_file, metadata, rating)
        if metadata or rating is not None:
            if self._save_audio_file(updated_file):
                self.logger.info(f"Successfully updated metadata for file: {file_path}")
                return updated_file
        return None

    # ------------------------------
    # Playlist Operations
    # ------------------------------
    def _open_playlist(self, path: Path, mode: str = "r", encoding: str = "utf-8"):
        try:
            if "r" in mode and not path.exists():
                raise FileNotFoundError(f"Playlist file not found: {path}")
            return path.open(mode, encoding=encoding)
        except Exception as e:
            self.logger.error(f"Failed to open playlist file {path} with mode '{mode}': {e}")
            return None

    def create_playlist(self, title: str, is_extm3u: bool = False) -> Playlist:
        """Create a new M3U playlist file."""
        if self.config_mgr.dry:
            self.logger.info("Dry run enabled. Playlist creation of {title} skipped.")
            return None

        playlist_path = self.playlist_path / f"{title}.m3u"
        try:
            with playlist_path.open("w", encoding="utf-8") as file:
                if is_extm3u:
                    file.write("#EXTM3U\n")
                    file.write(f"#PLAYLIST:{title}\n")
            self.logger.info(f"Created playlist: {playlist_path}")
        except Exception as e:
            self.logger.error(f"Failed to create playlist {playlist_path}: {e}")
        return self.read_playlist_metadata(str(playlist_path))

    def read_playlist_metadata(self, playlist_path: Path) -> Optional[Playlist]:
        playlist_path = playlist_path.resolve()
        title = playlist_path.stem
        is_extm3u = False

        with self._open_playlist(playlist_path) as file:
            for line in file:
                line = line.strip()
                if line.startswith("#EXTM3U"):
                    is_extm3u = True
                elif not is_extm3u:
                    break
                elif line.startswith("#PLAYLIST:"):
                    title = line.split(":", 1)[1].strip()
                    break

        title = self._get_playlist_title(playlist_path, title)
        self._playlist_title_map[title] = playlist_path

        playlist = Playlist(name=title, ID=str(playlist_path).lower(), player=None)
        playlist.file_path = str(playlist_path)
        playlist.is_extm3u = is_extm3u
        return playlist

    def get_tracks_from_playlist(self, playlist_path: Union[str, Path]) -> List[Path]:
        """Retrieve resolved track paths from a playlist file, scoped to self.path."""
        playlist_path = Path(playlist_path)
        tracks = []

        with self._open_playlist(playlist_path) as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                candidate_path = (self.path / line).resolve()

                # Must exist and be within root path
                if candidate_path.exists() and candidate_path.is_relative_to(self.path):
                    tracks.append(candidate_path)
                else:
                    self.logger.debug(f"Ignored invalid or out-of-scope track in playlist: {candidate_path}")
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
