from enum import StrEnum

import configargparse


class ConfigEnum(StrEnum):
    """Base class for case-insensitive string enums."""

    def __eq__(self, other: "ConfigEnum") -> bool:
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value.lower())

    @classmethod
    def find(cls, value: str) -> "ConfigEnum":
        """Find an enum by its value, case-insensitive."""
        if value is None:
            return None

        for item in cls:
            if item.value.lower() == value.lower() or item.name.lower() == value.lower():
                return item
        return None


class LogLevel(ConfigEnum):
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class PlayerType(ConfigEnum):
    PLEX = "plex"
    MEDIAMONKEY = "mediamonkey"
    FILESYSTEM = "filesystem"


class SyncItem(ConfigEnum):
    TRACKS = "tracks"
    PLAYLISTS = "playlists"


class CacheMode(ConfigEnum):
    METADATA = "metadata"
    MATCHES = "matches"
    MATCHES_ONLY = "matches-only"
    DISABLED = "disabled"


class TagWriteStrategy(ConfigEnum):
    WRITE_ALL = "write_all"
    WRITE_EXISTING = "write_existing"
    WRITE_DEFAULT = "write_default"
    OVERWRITE_DEFAULT = "overwrite_default"

    @property
    def description(self) -> str:
        return {
            self.WRITE_ALL: "Update ratings for all discovered media players.",
            self.WRITE_EXISTING: "Only update ratings for media players in each file; fallback to default player if none exist.",
            self.WRITE_DEFAULT: "Update ratings only for the default player; do not remove other ratings.",
            self.OVERWRITE_DEFAULT: "Update ratings only for the default player and delete all other ratings.",
        }[self]

    def requires_default_tag(self) -> bool:
        return self in {
            TagWriteStrategy.WRITE_DEFAULT,
            TagWriteStrategy.OVERWRITE_DEFAULT,
            TagWriteStrategy.WRITE_EXISTING,
        }

    def __str__(self) -> str:
        friendly_name = self.name.replace("_", " ").title()
        return f"{friendly_name:<20} : {self.description}"


class ConflictResolutionStrategy(ConfigEnum):
    PRIORITIZED_ORDER = "prioritized_order"
    HIGHEST = "highest"
    LOWEST = "lowest"
    AVERAGE = "average"

    @property
    def description(self) -> str:
        return {
            self.PRIORITIZED_ORDER: "Prioritized order of media players.",
            self.HIGHEST: "Use the highest rating.",
            self.LOWEST: "Use the lowest rating.",
            self.AVERAGE: "Use the average of all ratings.",
        }[self]


class ConfigManager:
    def __init__(self) -> None:
        """Initialize the configuration manager with default settings."""
        self.parser = configargparse.ArgumentParser(default_config_files=["./config.ini"], description="Synchronizes ID3 music ratings with a Plex media-server")
        self.config = self.parse_args()
        self._initialize_attributes()

    def parse_args(self) -> configargparse.Namespace:
        # General arguments
        self.parser.add_argument("--dry", action="store_true", help="Does not apply any changes")
        self.parser.add_argument("--source", type=str, required=True, help=f"Source player ({', '.join(PlayerType)})")
        self.parser.add_argument("--destination", type=str, required=True, help=f"Destination player ({', '.join(PlayerType)})")
        self.parser.add_argument("--sync", nargs="*", required=True, help=f"Selects which items to sync: one or more of {list(SyncItem)}")
        self.parser.add_argument("--log", default=LogLevel.WARNING, help=f"Sets the logging level ({', '.join(LogLevel)})")
        self.parser.add_argument("--clear-cache", action="store_true", help="Clear existing cache files before starting")
        self.parser.add_argument(
            "--cache-mode",
            type=str,
            choices=list(CacheMode),
            default=CacheMode.METADATA,
            help=f"Cache mode: {CacheMode.METADATA} (in-memory only), {CacheMode.MATCHES} (both), {CacheMode.MATCHES_ONLY} (persistent matches), {CacheMode.DISABLED}",
        )
        # Plex specific arguments
        self.parser.add_argument("--server", type=str, help="The name of the plex media server")
        self.parser.add_argument("--username", type=str, help="The plex username")
        self.parser.add_argument("--passwd", type=str, help="The password for the plex user. NOT RECOMMENDED TO USE!")
        self.parser.add_argument(
            "--token",
            type=str,
            help="Plex API token.  See https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/ for information on how to find your token",
        )
        # Filesystem specific arguments
        self.parser.add_argument("--path", type=str, help="Path to music directory for filesystem player")
        self.parser.add_argument("--playlist-path", type=str, help="Path to playlists directory for filesystem player")

        # Filesystem metadata arguments
        self.parser.add_argument("--tag-write-strategy", type=str, choices=[strategy.value for strategy in TagWriteStrategy], help="Strategy for writing rating tags to files")
        self.parser.add_argument("--default-tag", type=str, help="Canonical tag to use for writing ratings (e.g., MEDIAMONKEY, WINDOWSMEDIAPLAYER, MUSICBEE, WINAMP, TEXT)")
        self.parser.add_argument(
            "--conflict-resolution-strategy",
            type=str,
            choices=[strategy.value for strategy in ConflictResolutionStrategy],
            help="Strategy for resolving conflicting rating values",
        )
        self.parser.add_argument("--tag-priority-order", type=str, nargs="+", help="Ordered list of tag identifiers for resolving conflicts")
        return self.parser.parse_args()

    def _initialize_attributes(self) -> None:
        """Create attributes for each argument and convert to enums where applicable."""
        for key, value in vars(self.config).items():
            sanitized_key = key.replace("-", "_")  # Replace hyphens with underscores

            # Check if the attribute corresponds to a ConfigEnum class
            enum_classes = {
                "log": LogLevel,
                "source": PlayerType,
                "destination": PlayerType,
                "sync": SyncItem,
                "cache_mode": CacheMode,
                "tag_write_strategy": TagWriteStrategy,
                "conflict_resolution_strategy": ConflictResolutionStrategy,
            }

            if sanitized_key in enum_classes and value is not None:
                # Convert to the appropriate enum using the `find()` method
                enum_class = enum_classes[sanitized_key]
                if isinstance(value, list):  # Handle lists (e.g., sync items)
                    converted_values = []
                    for v in value:
                        enum_value = enum_class.find(v)
                        if enum_value is None:
                            raise ValueError(f"Invalid value '{v}' for '{sanitized_key}'. Valid options are: {', '.join([e.value for e in enum_class])}")
                        converted_values.append(enum_value)
                    setattr(self, sanitized_key, converted_values)
                else:
                    enum_value = enum_class.find(value)
                    if enum_value is None:
                        raise ValueError(f"Invalid value '{value}' for '{sanitized_key}'. Valid options are: {', '.join([e.value for e in enum_class])}")
                    setattr(self, sanitized_key, enum_value)
            else:
                # Set the attribute directly for non-enum values
                setattr(self, sanitized_key, value)

        # Enforce default_tag when required by tag_write_strategy
        if self.tag_write_strategy in {"write_default", "overwrite_default", "write_existing"} and not self.default_tag:
            raise ValueError(f"default_tag must be set when using the '{self.tag_write_strategy}' strategy.")
