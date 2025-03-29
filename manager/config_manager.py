import configargparse


class ConfigManager:
    def __init__(self) -> None:
        """Initialize the configuration manager with default settings."""
        self.parser = configargparse.ArgumentParser(default_config_files=["./config.ini"], description="Synchronizes ID3 music ratings with a Plex media-server")
        self.config = self.parse_args()
        self._initialize_attributes()

    def parse_args(self) -> configargparse.Namespace:
        self.parser.add_argument("--dry", action="store_true", help="Does not apply any changes")
        self.parser.add_argument("--source", type=str, default="mediamonkey", help="Source player (plex or [mediamonkey])")
        self.parser.add_argument("--destination", type=str, default="plex", help="Destination player ([plex] or mediamonkey)")
        self.parser.add_argument("--sync", nargs="*", default=["tracks"], help="Selects which items to sync: one or more of [tracks, playlists]")
        self.parser.add_argument("--log", default="warning", help="Sets the logging level (critical, error, [warning], info, debug)")
        self.parser.add_argument("--server", type=str, required=True, help="The name of the plex media server")
        self.parser.add_argument("--username", type=str, required=True, help="The plex username")
        self.parser.add_argument("--passwd", type=str, help="The password for the plex user. NOT RECOMMENDED TO USE!")
        self.parser.add_argument(
            "--token",
            type=str,
            help="Plex API token.  See https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/ for information on how to find your token",
        )
        self.parser.add_argument("--path", type=str, help="Path to music directory for filesystem player")
        self.parser.add_argument("--playlist-path", type=str, help="Path to playlists directory for filesystem player")
        self.parser.add_argument("--album-playlist", action="store_true", help="Sync album playlists")
        self.parser.add_argument(
            "--cache-mode",
            type=str,
            choices=["metadata", "matches", "matches-only", "disabled"],
            default="metadata",
            help="Cache mode: [metadata] (in-memory only), matches (both), matches-only (persistent matches), disabled",
        )
        self.parser.add_argument("--clear-cache", action="store_true", help="Clear existing cache files before starting")
        self.parser.add_argument(
            "--tag-write-strategy", type=str, choices=["write_all", "write_existing", "write_standard", "overwrite_standard"], help="Strategy for writing rating tags to files"
        )
        self.parser.add_argument(
            "--standard-tag", type=str, choices=["MEDIAMONKEY", "WINDOWSMEDIAPLAYER", "MUSICBEE", "WINAMP", "TEXT"], help="Canonical tag to use for writing ratings"
        )
        self.parser.add_argument(
            "--conflict-resolution-strategy", type=str, choices=["prioritized_order", "highest", "lowest", "average"], help="Strategy for resolving conflicting rating values"
        )
        self.parser.add_argument("--tag-priority-order", type=str, nargs="+", help="Ordered list of tag identifiers for resolving conflicts")
        return self.parser.parse_args()

    def _initialize_attributes(self) -> None:
        """Create attributes for each argument."""
        for key, value in vars(self.config).items():
            sanitized_key = key.replace("-", "_")  # Replace hyphens with underscores
            setattr(self, sanitized_key, value)
