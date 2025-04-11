import os
from enum import StrEnum
from typing import List, Union

import configargparse
from configupdater import ConfigUpdater, Section


class ConfigEnum(StrEnum):
    """Base class for case-insensitive string enums."""

    def __eq__(self, other: "ConfigEnum") -> bool:
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value.lower())

    @property
    def display(self) -> str:
        """Return the display name of the enum value."""
        name = self.name.replace("_", " ").title()
        return f"{name:<20} : {self.description}" if hasattr(self, "description") else name

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
    CONFIG_FILE = "./config.ini"
    _ENUM_FIELDS = {
        "log": LogLevel,
        "source": PlayerType,
        "destination": PlayerType,
        "sync": SyncItem,
        "cache_mode": CacheMode,
        "tag_write_strategy": TagWriteStrategy,
        "conflict_resolution_strategy": ConflictResolutionStrategy,
    }

    def __init__(self) -> None:
        """Initialize the configuration manager with default settings."""
        self.parser = configargparse.ArgumentParser(default_config_files=[self.CONFIG_FILE], description="Synchronizes ID3 and Vorbis music ratings between media players")
        self.config = self.parse_args()
        self._initialize_attributes()

    def parse_args(self) -> configargparse.Namespace:
        # Main section arguments
        main_group = self.parser.add_argument_group("Main Configuration")
        main_group.add_argument("-c", action="store_true", dest="clear_cache", help="Clear existing cache files before starting")
        main_group.add_argument("-d", "--dry", action="store_true", help="Does not apply any changes")
        main_group.add_argument("--source", type=str, required=True, help=f"Source player ({', '.join(PlayerType)})")
        main_group.add_argument("--destination", type=str, required=True, help=f"Destination player ({', '.join(PlayerType)})")
        main_group.add_argument("--sync", nargs="*", required=True, default=["tracks"], help=f"Selects which items to sync: one or more of {list(SyncItem)}")
        main_group.add_argument("--log", type=str, default=LogLevel.WARNING, help=f"Sets the logging level ({', '.join(LogLevel)})")
        main_group.add_argument("--cache-mode", type=str, choices=list(CacheMode), default=CacheMode.METADATA, help="Cache mode: metadata, matches, matches-only, disabled")

        # Plex section arguments
        plex_group = self.parser.add_argument_group("Plex Configuration")
        plex_group.add_argument("--server", type=str, help="The name of the Plex media server")
        plex_group.add_argument("--username", type=str, help="The Plex username")
        plex_group.add_argument("--passwd", type=str, help="The password for the Plex user. NOT RECOMMENDED TO USE!")
        plex_group.add_argument("--token", type=str, help="Plex API token. See Plex documentation for details")

        # Filesystem section arguments
        filesystem_group = self.parser.add_argument_group("Filesystem Configuration")
        filesystem_group.add_argument("--path", type=str, help="Path to music directory for filesystem player")
        filesystem_group.add_argument("--playlist-path", type=str, help="Path to playlists directory for filesystem player")
        filesystem_group.add_argument("--tag-write-strategy", type=str, choices=[strategy.value for strategy in TagWriteStrategy], help="Strategy for writing rating tags to files")
        filesystem_group.add_argument("--default-tag", type=str, help="Canonical tag to use for writing ratings (e.g., MEDIAMONKEY, WINDOWSMEDIAPLAYER, MUSICBEE, WINAMP, TEXT)")
        filesystem_group.add_argument(
            "--conflict-resolution-strategy",
            type=str,
            choices=[strategy.value for strategy in ConflictResolutionStrategy],
            help="Strategy for resolving conflicting rating values",
        )
        filesystem_group.add_argument("--tag-priority-order", type=str, nargs="+", help="Ordered list of tag identifiers for resolving conflicts")

        return self.parser.parse_args()

    def _initialize_attributes(self) -> None:
        """Set attributes from parsed config, including enum coercion and validation."""
        for key, value in vars(self.config).items():
            sanitized_key = key.replace("-", "_")

            if sanitized_key in self._ENUM_FIELDS:
                setattr(self, sanitized_key, self._parse_enum_field(sanitized_key, value))
            else:
                setattr(self, sanitized_key, value)

        self._validate_config_requirements()

    def _parse_enum_field(self, key: str, value: ConfigEnum) -> ConfigEnum | list[ConfigEnum] | None:
        """Convert CLI string(s) to corresponding ConfigEnum value(s), with error checking."""
        enum_class = self._ENUM_FIELDS[key]
        if value is None:
            return None

        if isinstance(value, list):
            parsed = []
            for v in value:
                enum_value = enum_class.find(v)
                if enum_value is None:
                    raise ValueError(f"Invalid value '{v}' for '{key}'. Valid options: {', '.join(e.value for e in enum_class)}")
                parsed.append(enum_value)
            return parsed
        else:
            enum_value = enum_class.find(value)
            if enum_value is None:
                raise ValueError(f"Invalid value '{value}' for '{key}'. Valid options: {', '.join(e.value for e in enum_class)}")
            return enum_value

    def _validate_config_requirements(self) -> None:
        """Perform validations that require multiple config values."""
        strategy = getattr(self, "tag_write_strategy", None)
        requires_default = strategy in {
            TagWriteStrategy.WRITE_DEFAULT,
            TagWriteStrategy.OVERWRITE_DEFAULT,
            TagWriteStrategy.WRITE_EXISTING,
        }

        if requires_default and not getattr(self, "default_tag", None):
            raise ValueError(f"default_tag must be set when using the '{strategy}' strategy.")

    def save_config(self) -> None:
        """Update or insert config values into the config file using full ConfigUpdater features."""

        def stringify_value(value: Union[str, bool, List], expected_type: type) -> str:
            if expected_type is list:
                return "[" + ", ".join(value) + "]"
            if expected_type is bool:
                return "true" if value else "false"
            return str(value)

        def normalize_section_name(name: str) -> str:
            return name.lower().replace(" configuration", "").strip()

        def update_config_value(section: Section, option: str, new_value: str, original_lhs: str, comment: str) -> None:
            """
            Replaces the value while preserving any existing spacing and inline comment.
            - `original_lhs` includes any padding before the '#' in the original line.
            - `comment` is everything after the '#' (may be empty).
            """
            # Preserve original padding if the new value fits
            section[option] = None
            option_obj = section[option]

            original_value = original_lhs.rstrip()
            padding = original_lhs[len(original_value) :]  # everything after the trimmed value
            new_value = str(new_value)
            if len(new_value) > len(original_value):
                padded_value = new_value
            else:
                padded_value = new_value + padding

            if comment:
                option_obj.value = f"{padded_value}#{comment}"
            else:
                option_obj.value = new_value

        def cast_config_option(value: str, expected_type: type) -> Union[str, bool]:
            """Casts a config file string value into the expected Python type."""
            raw = value.strip()

            if expected_type is bool:
                return raw.lower() in {"1", "true", "yes", "on"}
            elif expected_type is list:
                raw = raw.replace("[", "").replace("]", "")
                return sorted(item.strip().lower() for item in raw.split(",") if item.strip())
            elif expected_type is str:
                return raw.lower()
            else:
                raise NotImplementedError(f"Unsupported type in cast_config_option: {expected_type}")

        def needs_update(runtime_val: Union[str, bool, List], file_value: str, expected_type: type, default: Union[str, bool, List]) -> bool:
            """Determines whether the value from runtime should be written to the config file."""

            file_val_casted = cast_config_option(file_value, expected_type)

            if expected_type is list:
                runtime_normalized = sorted(v.strip().lower() for v in runtime_val) if runtime_val else []
            elif expected_type is str:
                runtime_normalized = runtime_val.strip().lower() if runtime_val else ""
            elif expected_type is bool:
                runtime_normalized = runtime_val is not None and runtime_val
            else:
                raise NotImplementedError(f"Unsupported type in needs_update: {expected_type}")

            # Skip writing if file is empty and runtime value equals default
            if file_value.strip() == "" and runtime_normalized == default:
                return False

            return runtime_normalized != file_val_casted

        updater = ConfigUpdater()
        # section_lookup: dest -> section_name
        config_parser = {}
        for group in self.parser._action_groups:
            section_name = normalize_section_name(group.title)
            for action in group._group_actions:
                if action.default == "==SUPPRESS==":
                    continue
                if not action.option_strings or not any(opt.startswith("--") for opt in action.option_strings):
                    continue
                elif action.dest:
                    if action.nargs in ("*", "+"):
                        t = list
                    elif isinstance(action.default, bool):
                        t = bool
                    elif action.type is str:
                        t = str
                    else:
                        raise NotImplementedError(f"Type {action.type} not implemented")
                    config_parser[action.dest] = {"section": section_name, "default": action.default, "type": t}
        config_file_values = {}

        if os.path.exists(self.CONFIG_FILE):
            updater.read(self.CONFIG_FILE)
            for section in updater.sections():
                for option in updater[section]:
                    line = updater[section][option].value.partition("#")
                    config_file_values[option] = {"section": section, "value": line[0], "comment": line[2]}
        else:
            # If the file doesn't exist, create it and write default values
            for section in self.parser._action_groups:
                if section._group_actions:
                    section_name = normalize_section_name(section.title)
                    updater.add_section(section_name)

        for dest_key, _value in self.config._get_kwargs():
            if dest_key not in config_parser:
                continue  # Skip unexpected/positional arguments

            meta = config_parser[dest_key]
            runtime_value = getattr(self, dest_key)
            dest_key = dest_key.replace("_", "-")
            file_val = config_file_values.get(dest_key)

            # Determine which section to use
            normalized_section = meta["section"].lower()
            matched_section = next((sect for sect in updater.sections() if sect.lower() == normalized_section), None)
            section_name = matched_section or meta["section"]

            # Option exists in config file — check if update is needed
            if not needs_update(
                runtime_value,
                file_value=file_val.get("value") if file_val else "",
                expected_type=meta["type"],
                default=meta["default"],
            ):
                continue
            # Update required
            update_config_value(
                updater[section_name],
                option=dest_key,
                new_value=stringify_value(runtime_value, meta["type"]),
                original_lhs=file_val.get("value") if file_val else "",
                comment=file_val["comment"] if file_val else "",
            )
        updater.update_file()
