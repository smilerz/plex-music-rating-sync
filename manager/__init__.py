from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache_manager import CacheManager
    from .config_manager import ConfigManager
    from .log_manager import LogManager
    from .stats_manager import StatsManager, StatusManager


class Manager:
    _instance = None

    def __new__(cls) -> "Manager":
        if cls._instance is None:
            cls._instance = super(Manager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        from .cache_manager import CacheManager
        from .config_manager import ConfigManager
        from .log_manager import LogManager
        from .stats_manager import StatsManager, StatusManager

        self.config = ConfigManager()
        self.log = LogManager()
        self.stats = StatsManager()
        self.status = StatusManager()
        self.cache = CacheManager(mode=self.config.cache_mode, stats_manager=self.stats)

    def get_log_manager(self) -> "LogManager":
        return self.log

    def get_stats_manager(self) -> "StatsManager":
        return self.stats

    def get_status_manager(self) -> "StatusManager":
        return self.status

    def get_cache_manager(self) -> "CacheManager":
        return self.cache

    def get_config_manager(self) -> "ConfigManager":
        return self.args


# Singleton instance
manager = Manager()
