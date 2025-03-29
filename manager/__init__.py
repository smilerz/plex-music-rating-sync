from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache_manager import CacheManager
    from .config_manager import ConfigManager
    from .log_manager import LogManager
    from .stats_manager import StatsManager, StatusManager


class Manager:
    _instance = None

    def __new__(self) -> "Manager":
        if self._instance is None:
            self._instance = super(Manager, self).__new__(self)
        return self._instance

    def initialize(self) -> None:
        from .cache_manager import CacheManager
        from .config_manager import ConfigManager
        from .log_manager import LogManager
        from .stats_manager import StatsManager, StatusManager

        self.config = ConfigManager()
        self.log = LogManager()
        self.stats = StatsManager()
        self.status = StatusManager()
        self.cache = CacheManager()

        self.logger = self.log.setup_logging(self.config.log)

    def get_log_manager(self) -> "LogManager":
        return self.log

    def get_stats_manager(self) -> "StatsManager":
        return self.stats

    def get_status_manager(self) -> "StatusManager":
        return self.status

    def get_cache_manager(self) -> "CacheManager":
        return self.cache

    def get_config_manager(self) -> "ConfigManager":
        return self.config


# Singleton instance
manager = Manager()
