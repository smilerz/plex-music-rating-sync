from typing import Optional

from tqdm import tqdm


class StatusBarHandler:
    """Manages multiple tqdm progress bars, allowing parallel phase tracking and returning bar objects for direct manipulation."""

    def __init__(self):
        self.bars = {}  # Stores progress bars by phase name

    def start_phase(self, desc: str, initial: int = 0, total: Optional[int] = None) -> tqdm:
        """Start a new phase with a separate progress bar and return it for direct manipulation."""
        if desc in self.bars:
            self.bars[desc].close()  # Close existing bar if restarting phase

        bar = tqdm(
            desc=desc,
            unit=" track",
            leave=True,
            dynamic_ncols=True,
            initial=initial,
            total=total,
            miniters=1,
            position=len(self.bars),  # Ensures new bars appear below existing ones
        )
        self.bars[desc] = bar
        return bar  # Return the tqdm instance

    def get_phase(self, desc: str) -> Optional[tqdm]:
        """Retrieve an active progress bar object for direct updates."""
        return self.bars.get(desc, None)

    def close_phase(self, desc: str) -> None:
        """Close a progress bar for a completed or canceled phase."""
        if desc in self.bars:
            self.bars[desc].close()
            del self.bars[desc]
        else:
            print(f"Warning: Tried to close nonexistent phase '{desc}'")

    def close_all(self) -> None:
        """Close all active progress bars."""
        for desc in list(self.bars.keys()):
            self.close_phase(desc)
        self.bars.clear()


class StatsManager:
    """Centralized stats tracking for sync operations"""

    def __init__(self):
        self.stats = {
            "tracks_processed": 0,
            "tracks_matched": 0,
            "tracks_updated": 0,
            "tracks_conflicts": 0,
            "playlists_processed": 0,
            "playlists_matched": 0,
            "playlists_updated": 0,
            "perfect_matches": 0,  # 100% match score
            "good_matches": 0,  # 80-99% match score
            "poor_matches": 0,  # 30-79% match score
            "no_matches": 0,  # <30% match score or no match found
            "cache_hits": 0,  # Tracks found in cache
        }
        self.status_handler = StatusBarHandler()

    def increment(self, stat_name: str, amount: int = 1) -> None:
        """Increment a stat counter"""
        if stat_name in self.stats:
            self.stats[stat_name] += amount

    def get(self, stat_name: str) -> int:
        """Get current value of a stat"""
        return self.stats.get(stat_name, 0)

    def set(self, stat_name: str, value: int) -> None:
        """Get current value of a stat"""
        self.stats[stat_name] = value

    def get_status_handler(self) -> StatusBarHandler:
        """Returns the status bar handler instance"""
        return self.status_handler
