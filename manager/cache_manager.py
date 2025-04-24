import logging
import os
import pickle
import warnings
from datetime import datetime
from typing import List, Optional

import pandas as pd

from manager import get_manager
from MediaPlayer import FileSystem, MediaMonkey, Plex
from sync_items import AudioTag

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class Cache:
    """Generic Cache class to handle common operations for caching."""

    def __init__(self, filepath: str, columns: list, dtype: dict, save_threshold: int = 100, max_age_hours: Optional[int] = 720) -> None:
        self.filepath = filepath
        self.max_age_hours = max_age_hours
        self.columns = columns
        self.dtype = dtype
        self.save_threshold = save_threshold
        self.logger = logging.getLogger("PlexSync.Cache")
        self.cache: pd.DataFrame = self._initialize_cache()
        self.update_count = 0

    def _initialize_cache(self) -> pd.DataFrame:
        """Initialize a new cache with the required columns."""
        self.logger.debug(f"Initializing cache with columns: {self.columns}")
        df = pd.DataFrame(data=None, index=range(self.save_threshold + 1), columns=self.columns).astype(self.dtype)
        self.logger.info(f"Cache initialized with {self.save_threshold + 1} empty rows")
        return df

    def load(self) -> None:
        """Load the cache from the file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "rb") as f:
                    self.cache = pickle.load(f)
                self.logger.info(f"Cache loaded from {self.filepath}: {len(self.cache)} entries")
                self._ensure_columns()
            except (pickle.UnpicklingError, EOFError) as e:
                self.logger.error(f"Failed to load cache from {self.filepath}: {e}")
                self.logger.warning("Falling back to new cache initialization")
                self.cache = self._initialize_cache()
                # 👇 New: check age of cache

            if self.max_age_hours is not None:
                file_mtime = os.path.getmtime(self.filepath)
                now = datetime.now().timestamp()
                age_seconds = now - file_mtime
                age_hours = age_seconds / (60 * 60)

                if age_hours > self.max_age_hours:
                    self.logger.warning(f"Cache file {self.filepath} is {age_hours:.1f} hours old — exceeding max age of {self.max_age_hours} hours.")
                    self.logger.warning("Discarding cache and reinitializing.")
                    self.cache = self._initialize_cache()
                    return

        else:
            self.logger.debug(f"No existing cache found at {self.filepath}. Initializing new cache.")
            self.cache = self._initialize_cache()

    def save(self) -> None:
        """Save the cache to the file."""
        try:
            with open(self.filepath, "wb") as f:
                pickle.dump(self.cache, f)
            self.logger.info(f"Cache saved to {self.filepath}: {len(self.cache)} entries")
        except Exception as e:
            self.logger.error(f"Failed to save cache to {self.filepath}: {e}")

    def delete(self) -> None:
        """Delete the cache file."""
        try:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
                self.logger.info(f"Cache file {self.filepath} deleted successfully")
            else:
                self.logger.debug(f"No cache file found at {self.filepath} to delete")
        except Exception as e:
            self.logger.error(f"Failed to delete cache file {self.filepath}: {e}")

    def _ensure_columns(self) -> None:
        """Ensure all required columns are present in the cache."""
        for column in self.columns:
            if column not in self.cache.columns:
                self.logger.warning(f"Missing column '{column}' in cache. Adding it with NaN values.")
                self.cache[column] = pd.NA

    def resize(self, additional_rows: int = 100) -> None:
        """Resize the cache by adding more empty rows."""
        start_index = self.cache.index.max() + 1 if not self.cache.empty else 0
        new_index = range(start_index, start_index + additional_rows)
        self.cache = self.cache.reindex(self.cache.index.union(new_index))
        self.logger.debug(f"Cache resized by {additional_rows} rows. New size: {len(self.cache)}")

    def auto_save(self) -> None:
        """Trigger an auto-save if the update count exceeds the threshold."""
        if self.update_count >= self.save_threshold:
            self.cache = self.cache.dropna(how="all").copy()
            self.save()
            self.update_count = 0

    def is_empty(self) -> bool:
        """Check if the cache DataFrame is empty."""
        return self.cache.empty


class CacheManager:
    """Handles caching for metadata and track matches, supporting multiple caching modes."""

    KNOWN_PLAYERS = [MediaMonkey, Plex, FileSystem]
    MATCH_CACHE_FILE = "matches_cache.pkl"
    METADATA_CACHE_FILE = "metadata_cache.pkl"
    SAVE_THRESHOLD = 100

    def __init__(self) -> None:
        """Initialize cache manager"""
        mgr = get_manager()
        self.config_mgr = mgr.get_config_manager()
        self.stats_mgr = mgr.get_stats_manager()
        self.logger = logging.getLogger("PlexSync.CacheManager")
        self.mode = self.config_mgr.cache_mode
        self.metadata_cache = None
        self.match_cach = None

        if self.mode == "disabled":
            self.logger.info("Cache disabled.")
            return

        self._initialize_caches()

    def is_match_cache_enabled(self) -> bool:
        """Return True if match caching is enabled based on current mode."""
        return self.mode in {"matches", "matches-only"}

    def is_metadata_cache_enabled(self) -> bool:
        """Return True if metadata caching is enabled based on current mode."""
        return self.mode in {"metadata", "matches"}

    def _initialize_caches(self) -> None:
        """Initialize both caches based on current mode."""
        self.logger.debug(f"Initializing caches for mode: {self.mode}")
        if self.is_match_cache_enabled():
            self.logger.debug("Initializing match cache")
            self.match_cache = Cache(filepath=self.MATCH_CACHE_FILE, columns=self._get_match_cache_columns(), dtype="object", save_threshold=self.SAVE_THRESHOLD)
            self.match_cache.load()

        if self.is_metadata_cache_enabled():
            self.logger.debug("Initializing metadata cache")
            self.metadata_cache = Cache(
                filepath=self.METADATA_CACHE_FILE, columns=self._get_metadata_cache_columns(), dtype=object, save_threshold=self.SAVE_THRESHOLD, max_age_hours=24
            )
            self.metadata_cache.load()

    def _safe_get_value(self, row: pd.DataFrame, column_name: str) -> Optional[object]:
        """Safely extract a value from a pandas row, converting NaN to None."""
        value = row[column_name].iloc[0]
        return None if pd.isna(value) else value

    def _get_match_cache_columns(self) -> list:
        """Get the required columns for the match cache."""
        return [player.name() for player in self.KNOWN_PLAYERS] + ["score"]

    def _get_metadata_cache_columns(self) -> list:
        """Get the required columns for the metadata cache."""
        return list(dict.fromkeys(["player_name", *AudioTag.get_fields()]))

    def cleanup(self) -> None:
        """Clean up cache files from disk."""
        if self.metadata_cache:
            self.metadata_cache.delete()

        if self.match_cache and not self.is_match_cache_enabled():
            self.match_cache.delete()

    def invalidate(self) -> None:
        """Invalidate both match and metadata caches."""
        if self.is_match_cache_enabled() and self.match_cache:
            self.match_cache.delete()
            self.match_cache = None

        if self.is_metadata_cache_enabled() and self.metadata_cache:
            self.metadata_cache.delete()
            self.metadata_cache = None

    def _trigger_auto_save(self) -> None:
        """Handle periodic cache saving for each cache type independently."""
        # Check and save match cache if needed
        if self.is_match_cache_enabled() and self.match_cache and self._match_update_count >= self.SAVE_THRESHOLD:
            self.match_cache.auto_save()
            self._match_update_count = 0

        # Check and save metadata cache if needed
        if self.is_metadata_cache_enabled() and self.metadata_cache and self._metadata_update_count >= self.SAVE_THRESHOLD:
            self.metadata_cache.auto_save()
            self._metadata_update_count = 0

    ### MATCH CACHING (PERSISTENT) ###
    def get_match(self, source_id: str, source_name: str, dest_name: str) -> Optional[str]:
        """Retrieve a cached match for a track."""
        if not self.is_match_cache_enabled() or self.match_cache is None or self.match_cache.is_empty():
            return None, None

        row = self.match_cache.cache[self.match_cache.cache[source_name] == source_id]
        if row.empty:
            return None, None

        match = self._safe_get_value(row, dest_name)
        score = self._safe_get_value(row, "score")
        self.logger.trace(f"Match cache hit for {source_name}:{source_id} and {dest_name}:{match}")
        self.stats_mgr.increment("cache_hits")
        return match, score

    def set_match(self, source_id: str, dest_id: str, source_name: str, dest_name: str, score: Optional[float] = None) -> None:
        """Store or update a track match between source and destination, including a score."""
        if not self.is_match_cache_enabled() or self.match_cache is None or self.match_cache.is_empty():
            return

        # Check if this match already exists
        existing_row = self.match_cache.cache[(self.match_cache.cache[source_name] == source_id) & (self.match_cache.cache[dest_name] == dest_id)]

        if not existing_row.empty:
            # Update the score if it differs
            row_idx = existing_row.index[0]
            if self._safe_get_value(existing_row, "score") != score:
                self.match_cache.cache.loc[row_idx, "score"] = score
                self.logger.debug(f"Updated score for existing match: {source_name}:{source_id} <-> {dest_name}:{dest_id}")
                self.match_cache.update_count += 1
                self.match_cache.auto_save()
            return

        # Find the next available empty row or resize if needed
        empty_row_idx = self.match_cache.cache.index[self.match_cache.cache.isna().all(axis=1)][0] if self.match_cache.cache.isna().all(axis=1).any() else None
        if empty_row_idx is None:
            self.match_cache.resize()
            empty_row_idx = self.match_cache.cache.index[self.match_cache.cache.isna().all(axis=1)][0]

        # Insert new match
        self.match_cache.cache.loc[empty_row_idx, [source_name, dest_name, "score"]] = [source_id, dest_id, score]
        self.match_cache.update_count += 1
        self.match_cache.auto_save()

    def get_metadata(self, player_name: str, track_id: str, force_enable: bool = False) -> Optional[AudioTag]:
        """Retrieve cached metadata by player name and track ID."""

        if not (force_enable or self.is_metadata_cache_enabled()) or self.metadata_cache is None or self.metadata_cache.is_empty():
            return None

        player_name = player_name.strip().lower()
        track_id = str(track_id).strip().lower()

        # Find row matching player name and track ID
        row = self.metadata_cache.cache[(self.metadata_cache.cache["player_name"] == player_name) & (self.metadata_cache.cache["ID"] == track_id)]

        if row.empty:
            return None

        # Convert row data to AudioTag
        data = {key: self._safe_get_value(row, key) for key in row.columns}
        self.logger.trace(f"Metadata cache hit for {player_name}:{track_id}")
        self.stats_mgr.increment("cache_hits")
        return AudioTag.from_dict(data)

    ### METADATA CACHING (NON-PERSISTENT) ###
    def set_metadata(self, player_name: str, track_id: str, metadata: AudioTag, force_enable: bool = False) -> None:
        """Store metadata in the pre-allocated cache, resizing if needed."""
        if not (force_enable or self.is_metadata_cache_enabled()):
            return

        player_name = player_name.strip().lower()
        track_id = str(track_id).strip().lower()

        if self.metadata_cache is None or self.metadata_cache.is_empty():
            self.metadata_cache = Cache(
                filepath=self.METADATA_CACHE_FILE,
                columns=self._get_metadata_cache_columns(),
                dtype={col: "str" if col == "ID" else "object" for col in self._get_metadata_cache_columns()},
                save_threshold=self.SAVE_THRESHOLD,
            )
            self.metadata_cache.load()

        # Check if metadata already exists for this track
        existing_row = self.metadata_cache.cache[(self.metadata_cache.cache["player_name"] == player_name) & (self.metadata_cache.cache["ID"] == str(track_id))]

        if not existing_row.empty:
            # Update the existing row
            row_index = existing_row.index[0]
            metadata_dict = metadata.to_dict()
            metadata_dict["ID"] = str(metadata_dict["ID"]) if metadata_dict["ID"] is not None else None
            self.metadata_cache.cache.update(pd.DataFrame(metadata_dict, index=[row_index]))
        else:
            # Find the next available empty row (first row where all columns are NaN)
            empty_row_idx = self.metadata_cache.cache.index[self.metadata_cache.cache.isna().all(axis=1)][0] if self.metadata_cache.cache.isna().all(axis=1).any() else None

            if empty_row_idx is None:
                self.metadata_cache.resize()
                empty_row_idx = self.metadata_cache.cache.index[self.metadata_cache.cache.isna().all(axis=1)][0]

            # Store new metadata in the available row
            self.metadata_cache.cache.loc[empty_row_idx, "player_name"] = player_name
            for key, value in metadata.to_dict().items():
                if key in self.metadata_cache.cache.columns:
                    if key == "ID" and value is not None:
                        self.metadata_cache.cache.loc[empty_row_idx, key] = str(value)
                    elif value is None:
                        self.metadata_cache.cache.loc[empty_row_idx, key] = pd.NA
                    else:
                        self.metadata_cache.cache.loc[empty_row_idx, key] = value
        self.metadata_cache.update_count += 1
        self.metadata_cache.auto_save()

    def get_tracks_by_filter(self, filter_mask: pd.Series) -> List[AudioTag]:
        """Convert filtered DataFrame rows into AudioTag objects"""
        # Apply filter to get matching rows
        matching_rows = self.metadata_cache.cache[filter_mask]

        if matching_rows.empty:
            return []

        # Convert each row to an AudioTag
        tracks = []
        for _idx, row in matching_rows.iterrows():
            data = {col: None if pd.isna(row[col]) else row[col] for col in matching_rows.columns}
            track = AudioTag.from_dict(data)
            tracks.append(track)
        return tracks
