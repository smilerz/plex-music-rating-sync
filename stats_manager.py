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

    def increment(self, stat_name: str, amount: int = 1) -> None:
        """Increment a stat counter"""
        if stat_name in self.stats:
            self.stats[stat_name] += amount

    def get(self, stat_name: str) -> int:
        """Get current value of a stat"""
        return self.stats.get(stat_name, 0)
