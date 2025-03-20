from typing import Optional

from tqdm import tqdm


class StatusBarHandler:
    """Manages multiple tqdm progress bars, allowing parallel phase tracking and returning bar objects for direct manipulation."""

    def __init__(self):
        self.bars = {}

    class ManagedProgressBar(tqdm):
        """Custom progress bar that removes itself from handler when closed"""

        def __init__(self, handler: "StatusBarHandler", desc: str, **kwargs):
            super().__init__(**kwargs)
            self.handler = handler
            self.desc = desc

        def close(self) -> None:
            """Override close to remove the bar from handler's dictionary"""
            super().close()
            if self.desc in self.handler.bars:
                del self.handler.bars[self.desc]

    def start_phase(self, desc: str, initial: int = 0, total: Optional[int] = None) -> tqdm:
        """Start a new phase with a separate progress bar and return it for direct manipulation."""
        if desc in self.bars:
            self.bars[desc].close()  # Close existing bar if restarting phase

        bar = self.ManagedProgressBar(
            handler=self,
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
        return bar


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
