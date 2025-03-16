import logging
import os
import pickle
from typing import Optional, Type

import pandas as pd

# Import player classes from MediaPlayer
from MediaPlayer import MediaMonkey, MediaPlayer, PlexPlayer

# Import AudioTag from sync_items
from sync_items import AudioTag


class CacheManager:
    """Handles caching for metadata and track matches, supporting multiple caching modes."""

    KNOWN_PLAYERS = [MediaMonkey, PlexPlayer]
    MATCH_CACHE_FILE = "matches_cache.pkl"
    METADATA_CACHE_FILE = "metadata_cache.pkl"
    SAVE_THRESHOLD = 100

    def __init__(self, mode: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.mode = self._validate_mode(mode)
        self.metadata_cache: Optional[pd.DataFrame] = None
        self.match_cache: Optional[pd.DataFrame] = None
        self._update_count = 0

        if self.mode == "disabled":
            self.logger.info("Cache disabled.")
            return None

        self._initialize_caches()

    def _validate_mode(self, mode: str) -> str:
        """Validate and normalize cache mode."""
        valid_modes = {"metadata", "matches", "matches-only", "disabled"}
        if mode not in valid_modes:
            self.logger.error(f"Invalid cache mode: {mode}. Defaulting to 'matches'.")
            return "matches"
        return mode

    def _initialize_caches(self) -> None:
        """Initialize both caches based on current mode."""
        if self.mode in {"matches", "matches-only"}:
            self.match_cache = self._load_cache(self.MATCH_CACHE_FILE, self._initialize_match_cache, "match")

        if self.mode in {"metadata", "matches"}:
            self.metadata_cache = self._load_cache(self.METADATA_CACHE_FILE, self._initialize_metadata_cache, "metadata")

    def _load_cache(self, filepath: str, init_func: callable, cache_type: str) -> Optional[pd.DataFrame]:
        """Generic cache loading with error handling."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    cache = pickle.load(f)
                self.logger.info(f"{cache_type.title()} cache loaded: {len(cache)} entries")
                return cache
            except (pickle.UnpicklingError, EOFError) as e:
                self.logger.error(f"Failed to load {cache_type} cache: {e}")
                self.logger.info(f"Initializing new {cache_type} cache")
        return init_func()

    def _save_cache(self, cache: pd.DataFrame, filepath: str, cache_type: str) -> bool:
        """Generic cache saving with error handling."""
        try:
            with open(filepath, "wb") as f:
                pickle.dump(cache, f)
            self.logger.info(f"{cache_type.title()} cache saved to disk")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save {cache_type} cache: {e}")
            return False

    def _delete_cache(self, filepath: str, cache_type: str) -> bool:
        """Generic cache deletion with error handling."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                self.logger.info(f"{cache_type.title()} cache deleted from disk")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete {cache_type} cache: {e}")
            return False

    def _initialize_match_cache(self) -> pd.DataFrame:
        """Create an empty match cache with explicitly defined player columns."""
        player_names = [player.name for player in self.KNOWN_PLAYERS]
        return pd.DataFrame(columns=player_names)

    def _initialize_metadata_cache(self) -> pd.DataFrame:
        """Create metadata cache on first access, defining columns from AudioTag attributes."""
        self.logger.info("Initializing metadata cache.")
        columns = ["track_player", *AudioTag.get_fields()]
        return pd.DataFrame(columns=columns).set_index(["track_player", "ID"])

    def invalidate(self) -> None:
        """Invalidate both match and metadata caches."""
        if self.mode in {"matches", "matches-only"}:
            self._delete_cache(self.MATCH_CACHE_FILE, "match")
            self.match_cache = None

        if self.mode in {"metadata", "matches"}:
            self._delete_cache(self.METADATA_CACHE_FILE, "metadata")
            self.metadata_cache = None

    ### MATCH CACHING (PERSISTENT) ###
    def get_match(self, source_id: str) -> Optional[str]:
        """Retrieve a cached match for a track."""
        if self.mode not in {"matches", "matches-only"}:
            return None

        # Check both columns since we only have 2 players
        for col in self.match_cache.columns:
            if source_id in self.match_cache[col].values:
                row = self.match_cache[self.match_cache[col] == source_id]
                if not row.empty:
                    # Return ID from other column
                    other_col = next(c for c in row.columns if c != col)
                    return row[other_col].values[0]
        return None

    def set_match(self, source_id: str, dest_id: str, source_name: str, dest_name: str) -> None:
        """Store a track match between source and destination."""
        if self.mode not in {"matches", "matches-only"}:
            return

        new_row = dict.fromkeys(self.match_cache.columns)
        new_row[source_name] = source_id
        new_row[dest_name] = dest_id
        self.match_cache = self.match_cache.append(new_row, ignore_index=True)

        self._trigger_auto_save()

    ### METADATA CACHING (NON-PERSISTENT) ###
    def get_metadata(self, track_player: Type[MediaPlayer], track_id: str) -> Optional[AudioTag]:
        """Retrieve cached metadata"""
        if self.mode not in {"metadata", "matches"} or self.metadata_cache is None:
            return None

        key = (track_player, track_id)
        if key in self.metadata_cache.index:
            data = self.metadata_cache.loc[key].to_dict()
            return AudioTag.from_dict(data)
        return None

    def set_metadata(self, track_player: Type[MediaPlayer], track_id: str, metadata: AudioTag) -> None:
        """Store metadata"""
        if self.mode not in {"metadata", "matches"}:
            return

        if self.metadata_cache is None:
            self.metadata_cache = self._initialize_metadata_cache()

        self.metadata_cache.loc[(track_player, track_id)] = metadata.to_dict()
        self._trigger_auto_save()

    ### PERIODIC SAVING ###
    def _trigger_auto_save(self) -> None:
        """Handle periodic cache saving."""
        self._update_count += 1
        if self._update_count >= self.SAVE_THRESHOLD:
            self.force_save()
            self._update_count = 0

    def force_save(self) -> None:
        """Save both caches to disk if enabled."""
        if self.mode in {"matches", "matches-only"}:
            self._save_cache(self.match_cache, self.MATCH_CACHE_FILE, "match")

        if self.mode in {"metadata", "matches"} and self.metadata_cache is not None:
            self._save_cache(self.metadata_cache, self.METADATA_CACHE_FILE, "metadata")

    def cleanup(self) -> None:
        """Clean up cache files from disk."""
        if self.mode in {"metadata", "matches"}:
            self._delete_cache(self.METADATA_CACHE_FILE, "metadata")
