#!/usr/bin/env python3
import locale
import time
from datetime import timedelta
from typing import List

import configargparse

from cache_manager import CacheManager
from log_manager import LoggingManager
from MediaPlayer import FileSystemPlayer, MediaMonkey, MediaPlayer, PlexPlayer
from stats_manager import StatsManager
from sync_items import AudioTag
from sync_pair import PlaylistPair, SyncState, TrackPair


class PlexSync:
    def __init__(self, options: configargparse.Namespace) -> None:
        self.logger = None
        self.options = options
        self.setup_logging()

        self.stats_manager = StatsManager()
        self.cache_manager = CacheManager(options.cache_mode, self.stats_manager)
        self._status = None

        try:
            self.source_player = self._create_player(self.options.source)
            self.destination_player = self._create_player(self.options.destination)
        except ValueError:
            exit(1)

        self.conflicts = []
        self.updates = []
        self.start_time = time.time()

    @property
    def status(self) -> object:
        """Lazily initialize status handler when needed"""
        if self._status is None:
            self._status = self.stats_manager.get_status_handler()
        return self._status

    def _create_player(self, player_type: str) -> MediaPlayer:
        """Create and configure a media player instance"""
        player_type = player_type.lower()
        player_map = {"plex": PlexPlayer, "mediamonkey": MediaMonkey, "filesystem": FileSystemPlayer}

        if player_type not in player_map:
            self.logger.error(f"Invalid player type: {player_type}")
            self.logger.error(f"Supported players: {', '.join(player_map.keys())}")
            raise ValueError(f"Invalid player type: {player_type}")

        player = player_map[player_type](cache_manager=self.cache_manager, stats_manager=self.stats_manager)
        player.dry_run = self.options.dry
        return player

    def setup_logging(self) -> None:
        """Initializes custom logging with log rotation and multi-level console output."""
        logging_manager = LoggingManager()
        self.logger = logging_manager.setup_logging(self.options.log)

    def _connect_player(self, player: MediaPlayer) -> None:
        """Connect a player with appropriate parameters based on player type."""
        if isinstance(player, PlexPlayer):
            if not self.options.token and not self.options.passwd:
                self.logger.error("Plex token or password is required for Plex player")
                raise
            if not self.options.server or not self.options.username:
                self.logger.error("Plex server and username are required for Plex player")
                raise
            player.connect(server=self.options.server, username=self.options.username, password=self.options.passwd, token=self.options.token)
        elif isinstance(player, FileSystemPlayer):
            if not (path := self.options.path) or not (playlist_path := self.options.playlist_path):
                self.logger.error("Path and playlist path are required for filesystem player")
                raise
            player.connect(path=path, playlist_path=playlist_path)
        else:
            player.connect()

    def sync(self) -> None:
        if self.options.clear_cache:
            self.cache_manager.invalidate()

        # Connect players with appropriate parameters based on player type
        for player in [self.source_player, self.destination_player]:
            self._connect_player(player)

        for sync_item in self.options.sync:
            if sync_item.lower() == "tracks":
                self.logger.info(f"Starting to sync track ratings from {self.source_player.name()} to {self.destination_player.name()}")
                self.sync_tracks()
            elif sync_item.lower() == "playlists":
                self.logger.info(f"Starting to sync playlists from {self.source_player.name()} to {self.destination_player.name()}")
                self.sync_playlists()
            else:
                raise ValueError(f"Invalid sync item selected: {sync_item}")

    def _get_match_summary(self) -> str:
        """Generate a summary of match quality statistics."""
        return (
            f"100%: {self.stats_manager.get('perfect_matches')} | "
            f"Good: {self.stats_manager.get('good_matches')} | "
            f"Poor: {self.stats_manager.get('poor_matches')} | "
            f"None: {self.stats_manager.get('no_matches')}"
        )

    def _match_tracks(self, tracks: List[AudioTag]) -> List[TrackPair]:
        """Match tracks between source and destination players

        Args:
            tracks: List of tracks from source player

        Returns:
            List[TrackPair]: List of matched track pairs
        """
        bar = self.status.start_phase("Matching tracks", total=len(tracks))
        sync_pairs = []
        for track in tracks:
            pair = TrackPair(self.source_player, self.destination_player, track)
            sync_pairs.append(pair)
            pair.match()
            bar.set_description_str(self._get_match_summary())
            bar.update()
        bar.close()
        return sync_pairs

    def _sync_matched_tracks(self, sync_pairs: List[TrackPair]) -> None:
        """Sync tracks that need updates

        Args:
            sync_pairs: List of matched track pairs
        """
        pairs_need_update = [pair for pair in sync_pairs if pair.sync_state is SyncState.NEEDS_UPDATE]
        self.logger.info(f"Synchronizing {len(pairs_need_update)} matching tracks without conflicts")

        bar = self.status.start_phase("Syncing tracks", total=len(pairs_need_update))
        for pair in pairs_need_update:
            pair.sync()
            bar.update()
        bar.close()

    def _display_conflict_options(self) -> str:
        """Display conflict resolution options and get user choice.

        Returns:
            str: The selected option
        """
        prompt = {
            "1": f"Keep all ratings from {self.source_player.name()} and update {self.destination_player.name()}",
            "2": f"Keep all ratings from {self.destination_player.name()} and update {self.source_player.name()}",
            "3": "Choose rating for each track",
            "4": "Display all conflicts",
            "5": "Display track match details",
            "6": "Don't resolve conflicts",
        }
        print("\n")
        for key in prompt:
            print(f"\t[{key}]: {prompt[key]}")
        return input("Select how to resolve conflicting rating: ")

    def _apply_ratings(self, pairs_conflicting: List[TrackPair], source_to_destination: bool) -> None:
        """Apply ratings from one player to another for all conflicting track pairs."""
        if not pairs_conflicting:
            return

        bar = self.status.start_phase("Resolving conflicts", total=len(pairs_conflicting))
        for pair in pairs_conflicting:
            pair.sync(force=True, source_to_destination=source_to_destination)
            bar.update()
        bar.close()

    def _display_track_details(self, sync_pairs: List[TrackPair]) -> None:
        """Display track match details based on user selection."""
        valid_choices = {"U", "G", "P", "N"}
        while True:
            sub_choice = input("Select tracks to display: [U]pdated, [G]ood, [P]oor, [N]one: ").strip().upper()
            if sub_choice in valid_choices:
                break
            print("Invalid choice. Please select [A], [G], [P], or [N].")

        filters = {
            "U": lambda pair: pair.score is not None and pair.sync_state is not SyncState.UP_TO_DATE,
            "G": lambda pair: pair.score is not None and pair.score >= 80,
            "P": lambda pair: pair.score is not None and 30 <= pair.score < 80,
            "N": lambda pair: pair.score is None or pair.score < 30,
        }
        TrackPair.display_pair_details(f"{sub_choice} Matches", [pair for pair in sync_pairs if filters[sub_choice](pair)])

    def _resolve_conflicts(self, pairs_conflicting: List[TrackPair], sync_pairs: List[TrackPair]) -> None:
        """Resolve conflicts between source and destination ratings"""
        self.stats_manager.set("tracks_conflicts", len(pairs_conflicting))

        while True:
            choice = self._display_conflict_options()
            if choice == "1":
                self._apply_ratings(pairs_conflicting, source_to_destination=True)
                break
            elif choice == "2":
                self._apply_ratings(pairs_conflicting, source_to_destination=False)
                break
            elif choice == "3":
                for i, pair in enumerate(pairs_conflicting, start=1):
                    result = pair.resolve_conflict(counter=i, total=len(pairs_conflicting))
                    if not result:
                        break
                break
            elif choice == "4":
                TrackPair.display_pair_details("Conflicting Matches", pairs_conflicting)
            elif choice == "5":
                self._display_track_details(sync_pairs)
            elif choice == "6":
                break
            else:
                print(f"{choice} is not a valid choice, please try again.")

    def sync_tracks(self) -> None:
        tracks = self.source_player.search_tracks(key="rating", value=True)

        self.stats_manager.increment("tracks_processed", len(tracks))
        self.logger.info(f"Attempting to match {len(tracks)} tracks")

        sync_pairs = self._match_tracks(tracks)
        self._sync_matched_tracks(sync_pairs)

        pairs_conflicting = [pair for pair in sync_pairs if pair.sync_state is SyncState.CONFLICTING]

        if pairs_conflicting:
            self._resolve_conflicts(pairs_conflicting, sync_pairs)

    def sync_playlists(self) -> None:
        # Start discovery phase
        print(f"Discovering playlists from {self.source_player.name()}")
        playlists = self.source_player.read_playlists()

        playlist_pairs = [PlaylistPair(self.source_player, self.destination_player, pl) for pl in playlists if not pl.is_auto_playlist]
        self.stats_manager.increment("playlists_processed", len(playlist_pairs))

        if self.options.dry:
            self.logger.info("Running a DRY RUN. No changes will be propagated!")

        self.logger.info(f"Matching {self.source_player.name()} playlists with {self.destination_player.name()}")

        # Start playlist matching phase
        bar = self.status.start_phase("Matching playlists", total=len(playlist_pairs))
        for pair in playlist_pairs:
            pair.match()
            bar.update()
        bar.close()

        # Start playlist sync phase
        bar = self.status.start_phase("Syncing playlists", total=len(playlist_pairs))
        for pair in playlist_pairs:
            pair.sync()
            bar.update()
        bar.close()

    def print_summary(self) -> None:
        elapsed = time.time() - self.start_time
        elapsed_time = str(timedelta(seconds=int(elapsed)))

        print("\nSync Summary:")
        print("-" * 50)
        print(f"Total time: {elapsed_time}")

        if "tracks" in self.options.sync:
            print("Tracks:")
            print(f"- Processed: {self.stats_manager.get('tracks_processed')}")
            print(f"- Matched: {self.stats_manager.get('tracks_matched')}")
            print(f"- Updated: {self.stats_manager.get('tracks_updated')}")
            print(f"- Conflicts: {self.stats_manager.get('tracks_conflicts')}")

            print("\nMatch Quality:")
            print(f"- Perfect matches (100%): {self.stats_manager.get('perfect_matches')}")
            print(f"- Good matches (80-99%): {self.stats_manager.get('good_matches')}")
            print(f"- Poor matches (30-79%): {self.stats_manager.get('poor_matches')}")
            print(f"- No matches (<30%): {self.stats_manager.get('no_matches')}")
            if self.options.log == "DEBUG":
                print(f"- Cache hits: {self.stats_manager.get('cache_hits')}")

        if "playlists" in self.options.sync:
            print("\nPlaylists:")
            print(f"- Processed: {self.stats_manager.get('playlists_processed')}")
            print(f"- Matched: {self.stats_manager.get('playlists_matched')}")
            print(f"- Updated: {self.stats_manager.get('playlists_updated')}")

        if self.options.dry:
            print("\nThis was a DRY RUN - no changes were actually made.")
        print("-" * 50)


def parse_args() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser(default_config_files=["./config.ini"], description="Synchronizes ID3 music ratings with a Plex media-server")
    parser.add_argument("--dry", action="store_true", help="Does not apply any changes")
    parser.add_argument("--source", type=str, default="mediamonkey", help="Source player (plex or [mediamonkey])")
    parser.add_argument("--destination", type=str, default="plex", help="Destination player ([plex] or mediamonkey)")
    parser.add_argument("--sync", nargs="*", default=["tracks"], help="Selects which items to sync: one or more of [tracks, playlists]")
    parser.add_argument("--log", default="warning", help="Sets the logging level (critical, error, [warning], info, debug)")
    parser.add_argument("--server", type=str, required=True, help="The name of the plex media server")
    parser.add_argument("--username", type=str, required=True, help="The plex username")
    parser.add_argument("--passwd", type=str, help="The password for the plex user. NOT RECOMMENDED TO USE!")
    parser.add_argument(
        "--token",
        type=str,
        help="Plex API token.  See https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/ for information on how to find your token",
    )
    parser.add_argument("--path", type=str, help="Path to music directory for filesystem player")
    parser.add_argument("--playlist-path", type=str, help="Path to playlists directory for filesystem player")
    parser.add_argument("--album-playlist", action="store_true", help="Sync album playlists")
    parser.add_argument(
        "--cache-mode",
        type=str,
        choices=["metadata", "matches", "matches-only", "disabled"],
        default="metadata",
        help="Cache mode: [metadata] (in-memory only), matches (both), matches-only (persistent matches), disabled",
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear existing cache files before starting")
    parser.add_argument(
        "--tag-write-strategy", type=str, choices=["write_all", "write_existing", "write_standard", "overwrite_standard"], help="Strategy for writing rating tags to files"
    )
    parser.add_argument("--standard-tag", type=str, choices=["MEDIAMONKEY", "WINDOWSMEDIAPLAYER", "MUSICBEE", "WINAMP", "TEXT"], help="Canonical tag to use for writing ratings")
    parser.add_argument(
        "--conflict-resolution-strategy", type=str, choices=["prioritized_order", "highest", "lowest", "average"], help="Strategy for resolving conflicting rating values"
    )
    parser.add_argument("--tag-priority-order", type=str, nargs="+", help="Ordered list of tag identifiers for resolving conflicts")
    return parser.parse_args()


if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, "")
    args = parse_args()
    sync_agent = PlexSync(args)
    if args.clear_cache:
        sync_agent.cache_manager.invalidate()
    sync_agent.sync()
    sync_agent.print_summary()
    sync_agent.cache_manager.cleanup()
