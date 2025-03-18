import logging
import os
import pickle
import warnings
from typing import Optional

import pandas as pd

from MediaPlayer import MediaMonkey, PlexPlayer
from sync_items import AudioTag

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class CacheManager:
    """Handles caching for metadata and track matches, supporting multiple caching modes."""

    KNOWN_PLAYERS = [MediaMonkey, PlexPlayer]
    MATCH_CACHE_FILE = "matches_cache.pkl"
    METADATA_CACHE_FILE = "metadata_cache.pkl"
    SAVE_THRESHOLD = 100

    def __init__(self, mode: str) -> None:
        """Initialize cache manager

        Args:
            mode: Cache mode to use
            logger: Logger instance to use for messages
        """
        self.logger = logging.getLogger("PlexSync.CacheManager")
        self.mode = mode
        self.metadata_cache: Optional[pd.DataFrame] = None
        self.match_cache: Optional[pd.DataFrame] = None
        self._metadata_update_count = 0
        self._match_update_count = 0

        if self.mode == "disabled":
            self.logger.info("Cache disabled.")
            return

        self._initialize_caches()

    def _initialize_caches(self) -> None:
        """Initialize both caches based on current mode."""
        self.logger.debug(f"Initializing caches for mode: {self.mode}")
        if self.mode in {"matches", "matches-only"}:
            self.logger.debug("Initializing match cache")
            self.match_cache = self._load_cache(self.MATCH_CACHE_FILE, self._initialize_match_cache, "match")

        if self.mode in {"metadata", "matches"}:
            self.logger.debug("Initializing metadata cache")
            self.metadata_cache = self._load_cache(self.METADATA_CACHE_FILE, self._initialize_metadata_cache, "metadata")

    def _load_cache(self, filepath: str, init_func: callable, cache_type: str) -> Optional[pd.DataFrame]:
        """Generic cache loading with error handling."""
        self.logger.debug(f"Attempting to load {cache_type.title()}Cache from {filepath}")
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    cache = pickle.load(f)
                self.logger.info(f"{cache_type.title()} cache loaded: {len(cache)} entries")
                return cache
            except (pickle.UnpicklingError, EOFError) as e:
                self.logger.error(f"Failed to load {cache_type.title()}Cache from {filepath}: {e}")
                self.logger.warning(f"Falling back to new {cache_type.title()}Cache initialization")
        else:
            self.logger.debug(f"No existing {cache_type.title()}Cache found at {filepath}")
        return init_func()

    def _save_cache(self, cache: pd.DataFrame, filepath: str, cache_type: str) -> bool:
        """Generic cache saving with error handling."""
        if cache is None:
            return False

        try:
            with open(filepath, "wb") as f:
                pickle.dump(cache, f)
            self.logger.info(f"{cache_type.title()}Cache saved successfully ({len(cache)} entries)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save {cache_type.title()}Cache to {filepath}: {e}")
            return False

    def _delete_cache(self, filepath: str, cache_type: str) -> bool:
        """Generic cache deletion with error handling."""
        self.logger.debug(f"Attempting to delete {cache_type.title()}Cache at {filepath}")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                self.logger.debug(f"{cache_type.title()}Cache deleted successfully")
            else:
                self.logger.debug(f"No {cache_type.title()} cache file found to delete")
            return True
        except PermissionError as e:
            self.logger.error(f"Permission denied while deleting {cache_type.title()}Cache: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error deleting {cache_type.title()}Cache: {e}")
            return False

    def _initialize_match_cache(self) -> pd.DataFrame:
        """Create a match cache with explicitly defined player columns and pre-allocated empty rows."""
        player_names = [player.name() for player in self.KNOWN_PLAYERS]
        self.logger.debug(f"Match cache initialized with columns: {player_names}")

        # Create a DataFrame with empty string values (or use NaN if preferred)
        df = pd.DataFrame(
            data=None,  # No actual data yet
            index=range(self.SAVE_THRESHOLD + 1),  # Auto-incrementing index
            columns=player_names,
        ).astype("object")  # Ensure all values are stored as objects (strings, None, etc.)

        self.logger.info(f"Match cache initialized with {self.SAVE_THRESHOLD + 1} empty rows")
        return df

    def _initialize_metadata_cache(self) -> pd.DataFrame:
        """Create metadata cache on first access, defining columns from AudioTag attributes."""
        # Ensure "player_name" is included, and remove duplicate column names
        fields = AudioTag.get_fields()
        columns = list(dict.fromkeys(["player_name", *fields]))  # Preserve order, remove duplicates

        self.logger.debug(f"Metadata cache columns: {columns}")

        # Create DataFrame with pre-allocated empty rows and an auto-incrementing integer index
        df = pd.DataFrame(
            data=None,
            index=range(self.SAVE_THRESHOLD + 1),  # Auto-numbered index
            columns=columns,
        ).astype({col: "str" if col == "ID" else "object" for col in columns})

        self.logger.info(f"Metadata cache initialized with {self.SAVE_THRESHOLD + 1} empty rows")
        return df

    def _resize_cache(self, cache: pd.DataFrame, cache_type: str, additional_rows: int = 100) -> pd.DataFrame:
        """Expands a cache DataFrame by adding more empty rows."""
        if cache is None:
            return None

        # Get the next available index
        start_index = cache.index.max() + 1 if not cache.empty else 0
        new_index = range(start_index, start_index + additional_rows)

        # Efficiently reindex without unnecessary copies
        cache = cache.reindex(cache.index.union(new_index))

        self.logger.debug(f"{cache_type.title()} cache expanded by {additional_rows} rows. New size: {len(cache)}")
        return cache

    ### PERIODIC SAVING ###
    def _trigger_auto_save(self) -> None:
        """Handle periodic cache saving for each cache type independently."""
        # Check and save match cache if needed
        if self.mode in {"matches", "matches-only"} and self._match_update_count >= self.SAVE_THRESHOLD:
            self.match_cache = self.match_cache.dropna(how="all").copy()
            self._save_cache(self.match_cache.reset_index(drop=True), self.MATCH_CACHE_FILE, "match")
            self._match_update_count = 0

        # Check and save metadata cache if needed
        if self.mode in {"metadata", "matches"} and self._metadata_update_count >= self.SAVE_THRESHOLD:
            self.metadata_cache = self.metadata_cache.dropna(how="all").copy()
            self._save_cache(self.metadata_cache.reset_index(drop=True), self.METADATA_CACHE_FILE, "metadata")
            self._metadata_update_count = 0

    ### MATCH CACHING (PERSISTENT) ###
    def get_match(self, source_id: str, source_name: str, dest_name: str) -> Optional[str]:
        """Retrieve a cached match for a track."""
        if self.mode not in {"matches", "matches-only"}:
            return None

        if self.match_cache is None or self.match_cache.empty:
            return None

        # Find rows where source_id is stored in the source_name column
        row = self.match_cache[self.match_cache[source_name] == source_id]

        if row.empty:
            return None  # No match found

        # Retrieve the corresponding match from the destination column
        match = row[dest_name].iloc[0]  # Get first match if multiple exist
        self.logger.debug(f"Match Cache hit: {match} for source_id: {source_id}")
        return match

    def set_match(self, source_id: str, dest_id: str, source_name: str, dest_name: str) -> None:
        """Store a track match between source and destination."""
        if self.mode not in {"matches", "matches-only"}:
            return
        if self.match_cache is None or self.match_cache.empty:
            return

        # Check if this match already exists to prevent duplicates
        existing_match = self.get_match(source_id, source_name, dest_name)
        if existing_match == dest_id:
            self.logger.debug(f"Match already exists: {source_name}:{source_id} <-> {dest_name}:{dest_id}")
            return

        # Find the next available empty row
        empty_row_idx = self.match_cache.index[self.match_cache.isna().all(axis=1)][0] if self.match_cache.isna().all(axis=1).any() else None

        if empty_row_idx is None:
            self.logger.warning("No empty rows left in match cache! Resizing...")
            self.match_cache = self._resize_cache(self.match_cache, "match")
            empty_row_idx = self.match_cache.index[self.match_cache.isna().all(axis=1)][0]  # Get new empty row

        # Assign new values in-place
        self.match_cache.loc[empty_row_idx, source_name] = source_id
        self.match_cache.loc[empty_row_idx, dest_name] = dest_id

        self._match_update_count += 1
        self._trigger_auto_save()

    def get_metadata(self, player_name: str, track_id: str) -> Optional[AudioTag]:
        """Retrieve cached metadata by player name and track ID.

        Args:
            player_name: Name of the player
            track_id: Track identifier

        Returns:
            AudioTag if found, None otherwise
        """
        if self.mode not in {"metadata", "matches"}:
            return None

        if self.metadata_cache is None or self.metadata_cache.empty:
            return None

        # Find row matching player name and track ID
        row = self.metadata_cache[(self.metadata_cache["player_name"] == player_name) & (self.metadata_cache["ID"] == track_id)]

        if row.empty:
            return None

        # Convert row data to AudioTag
        data = row.iloc[0].to_dict()
        self.logger.debug(f"Metadata cache hit for {player_name}:{track_id}")
        return AudioTag.from_dict(data)

    ### METADATA CACHING (NON-PERSISTENT) ###
    def set_metadata(self, player_name: str, track_id: str, metadata: AudioTag) -> None:
        """Store metadata in the pre-allocated cache, resizing if needed."""
        if self.mode not in {"metadata", "matches"}:
            return

        if self.metadata_cache is None or self.metadata_cache.empty:
            self.metadata_cache = self._initialize_metadata_cache()

        # Check if metadata already exists for this track
        existing_row = self.metadata_cache[(self.metadata_cache["player_name"] == player_name) & (self.metadata_cache["ID"] == track_id)]

        if not existing_row.empty:
            # Update the existing row
            row_index = existing_row.index[0]
            self.metadata_cache.update(pd.DataFrame(metadata.to_dict(), index=[row_index]))
        else:
            # Find the next available empty row (first row where all columns are NaN)
            empty_row_idx = self.metadata_cache.index[self.metadata_cache.isna().all(axis=1)][0] if self.metadata_cache.isna().all(axis=1).any() else None

            if empty_row_idx is None:
                self.logger.warning("No empty rows left in metadata cache! Resizing...")
                self.metadata_cache = self._resize_cache(self.metadata_cache, "metadata")
                empty_row_idx = self.metadata_cache.index[self.metadata_cache.isna().all(axis=1)][0]

            # Store new metadata in the available row
            self.metadata_cache.loc[empty_row_idx, "player_name"] = player_name
            self.metadata_cache.loc[empty_row_idx, "ID"] = track_id
            for key, value in metadata.to_dict().items():
                if key in self.metadata_cache.columns:
                    self.metadata_cache.loc[empty_row_idx, key] = value
        self._metadata_update_count += 1
        self._trigger_auto_save()

    def cleanup(self) -> None:
        """Clean up cache files from disk."""
        if self.mode in {"metadata", "matches"}:
            self._delete_cache(self.METADATA_CACHE_FILE, "metadata")

    def invalidate(self) -> None:
        """Invalidate both match and metadata caches."""
        if self.mode in {"matches", "matches-only"}:
            self._delete_cache(self.MATCH_CACHE_FILE, "match")
            self.match_cache = None

        if self.mode in {"metadata", "matches"}:
            self._delete_cache(self.METADATA_CACHE_FILE, "metadata")
            self.metadata_cache = None
