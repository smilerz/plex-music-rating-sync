import logging
import os
import pickle
from typing import Optional, Type

import pandas as pd

from MediaPlayer import MediaMonkey, MediaPlayer, PlexPlayer
from sync_items import AudioTag


class CacheManager:
    """Handles caching for metadata and track matches, supporting multiple caching modes."""

    KNOWN_PLAYERS = [MediaMonkey, PlexPlayer]  # Explicitly defined known players
    MATCH_CACHE_FILE = "matches_cache.pkl"
    METADATA_CACHE_FILE = "metadata_cache.pkl"

    def __init__(self, mode: str) -> None:
        """
        Initialize the cache manager.

        :param mode: Cache mode ('metadata', 'matches', 'matches-only', 'disabled').
        """
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.metadata_cache: Optional[pd.DataFrame] = None  # Will be created on first access
        self.match_cache: Optional[pd.DataFrame] = None  # Will be created on first access

        # Validate mode
        valid_modes = {"metadata", "matches", "matches-only", "disabled"}
        if mode not in valid_modes:
            self.logger.error(f"Invalid cache mode: {mode}. Defaulting to 'matches'.")
            self.mode = "matches"

        # Load caches based on mode
        if self.mode in {"matches", "matches-only"}:
            self.match_cache = self._load_match_cache()
        if self.mode in {"metadata", "matches"}:
            self.metadata_cache = self._load_metadata_cache()

        self.logger.debug(f"Cache initialized in mode: {self.mode}")

    def _load_match_cache(self) -> pd.DataFrame:
        """Load match cache from disk if allowed."""
        if os.path.exists(self.MATCH_CACHE_FILE):
            try:
                with open(self.MATCH_CACHE_FILE, "rb") as f:
                    match_cache = pickle.load(f)
                self.logger.info("Match cache loaded from disk.")
                return match_cache
            except (pickle.UnpicklingError, EOFError) as e:
                self.logger.error(f"Failed to load match cache: {e}. Resetting cache.")

        return self._initialize_match_cache()

    def _load_metadata_cache(self) -> pd.DataFrame:
        """Load metadata cache from disk if allowed."""
        if os.path.exists(self.METADATA_CACHE_FILE):
            try:
                with open(self.METADATA_CACHE_FILE, "rb") as f:
                    metadata_cache = pickle.load(f)
                self.logger.info("Metadata cache loaded from disk.")
                return metadata_cache
            except (pickle.UnpicklingError, EOFError) as e:
                self.logger.error(f"Failed to load metadata cache: {e}. Resetting cache.")

        return None  # Metadata cache is initialized on first access

    def _initialize_match_cache(self) -> pd.DataFrame:
        """Create an empty match cache with explicitly defined player columns."""
        player_names = [player.name for player in self.KNOWN_PLAYERS]
        return pd.DataFrame(columns=player_names)

    def _initialize_metadata_cache(self) -> pd.DataFrame:
        """Create metadata cache on first access, defining columns from AudioTag attributes."""
        self.logger.info("Initializing metadata cache with AudioTag fields.")
        columns = ["track_player", "track_id"] + list(AudioTag.get_fields())  # Dynamic column list
        return pd.DataFrame(columns=columns).set_index(["track_player", "track_id"])

    def invalidate(self) -> None:
        """Invalidate both match and metadata caches if applicable."""
        if self.mode in {"matches", "matches-only"} and os.path.exists(self.MATCH_CACHE_FILE):
            os.remove(self.MATCH_CACHE_FILE)
            self.logger.info("Match cache invalidated.")

        if self.mode in {"metadata", "matches"} and os.path.exists(self.METADATA_CACHE_FILE):
            os.remove(self.METADATA_CACHE_FILE)
            self.logger.info("Metadata cache invalidated.")

    ### MATCH CACHING (PERSISTENT) ###
    def get_match(self, source_player: Type[MediaPlayer], source_track_id: str, destination_player: Type[MediaPlayer]) -> Optional[str]:
        """Retrieve a cached match for a track."""
        if self.mode not in {"matches", "matches-only"} or self.match_cache is None:
            return None

        if source_track_id in self.match_cache[source_player.name].values:
            row = self.match_cache[self.match_cache[source_player.name] == source_track_id]
            if not row.empty:
                return row[destination_player.name].values[0]  # Return matched ID

        return None

    def set_match(self, source_player: Type[MediaPlayer], source_track_id: str, destination_player: Type[MediaPlayer], destination_track_id: str) -> None:
        """Store a track match, ensuring existing matches are updated correctly."""
        if self.mode not in {"matches", "matches-only"}:
            return

        if self.match_cache is None:
            self.match_cache = self._initialize_match_cache()

        new_row = dict.fromkeys(self.match_cache.columns)
        new_row[source_player.name] = source_track_id
        new_row[destination_player.name] = destination_track_id
        self.match_cache = self.match_cache.append(new_row, ignore_index=True)

        self.save_match_cache()

    ### METADATA CACHING (PERSISTENT) ###
    def get_metadata(self, track_player: Type[MediaPlayer], track_id: str) -> Optional[AudioTag]:
        """Retrieve cached metadata, initializing cache if necessary."""
        if self.mode not in {"metadata", "matches"} or self.metadata_cache is None:
            return None

        if (track_player.name, track_id) in self.metadata_cache.index:
            return AudioTag.from_dict(self.metadata_cache.loc[(track_player.name, track_id)].to_dict())

        return None

    def set_metadata(self, track_player: Type[MediaPlayer], track_id: str, metadata: AudioTag) -> None:
        """Store metadata, initializing cache if necessary."""
        if self.mode not in {"metadata", "matches"}:
            return

        if self.metadata_cache is None:
            self.metadata_cache = self._initialize_metadata_cache()

        self.metadata_cache.loc[(track_player.name, track_id)] = metadata.to_dict()
        self.save_metadata_cache()

    ### PERIODIC SAVING ###
    def save_match_cache(self) -> None:
        """Save the match cache to disk if mode allows."""
        if self.mode in {"matches", "matches-only"} and self.match_cache is not None:
            try:
                with open(self.MATCH_CACHE_FILE, "wb") as f:
                    pickle.dump(self.match_cache, f)
                self.logger.info("Match cache saved to disk.")
            except Exception as e:
                self.logger.error(f"Failed to save match cache: {e}")

    def save_metadata_cache(self) -> None:
        """Save the metadata cache to disk if mode allows."""
        if self.mode in {"metadata", "matches"} and self.metadata_cache is not None:
            try:
                with open(self.METADATA_CACHE_FILE, "wb") as f:
                    pickle.dump(self.metadata_cache, f)
                self.logger.info("Metadata cache saved to disk.")
            except Exception as e:
                self.logger.error(f"Failed to save metadata cache: {e}")
