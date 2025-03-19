#!/usr/bin/env python3
import locale
import logging
import os
import sys
import time
from datetime import timedelta

import configargparse
from tqdm import tqdm

from cache_manager import CacheManager
from MediaPlayer import MediaMonkey, MediaPlayer, PlexPlayer
from stats_manager import StatsManager
from sync_pair import PlaylistPair, SyncState, TrackPair


class PlexSync:
    log_levels = {"CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR, "WARNING": logging.WARNING, "INFO": logging.INFO, "DEBUG": logging.DEBUG}
    LOG_DIR = "logs"
    LOG_FILE = "sync_ratings.log"
    MAX_BACKUPS = 5

    def _create_player(self, player_type: str) -> MediaPlayer:
        """Create and configure a media player instance"""
        player_type = player_type.lower()
        player_map = {"plex": PlexPlayer, "mediamonkey": MediaMonkey}

        if player_type not in player_map:
            self.logger.error(f"Invalid player type: {player_type}")
            self.logger.error(f"Supported players: {', '.join(player_map.keys())}")
            raise ValueError(f"Invalid player type: {player_type}")

        player = player_map[player_type](cache_manager=self.cache_manager, stats_manager=self.stats_manager)
        player.dry_run = self.options.dry
        return player

    def __init__(self, options) -> None:
        self.logger = logging.getLogger("PlexSync")
        self.options = options
        self.setup_logging()

        self.stats_manager = StatsManager()
        self.cache_manager = CacheManager(options.cache_mode, self.stats_manager)

        try:
            self.source_player = self._create_player(self.options.source)
            self.destination_player = self._create_player(self.options.destination)
        except ValueError:
            exit(1)

        self.conflicts = []
        self.updates = []
        self.start_time = time.time()

    def get_player(self) -> None:
        """Removed as no longer needed"""
        pass

    def setup_logging(self) -> None:
        """Initializes custom logging with log rotation and multi-level console output without mixing debug in console."""

        class InfoFilter(logging.Filter):
            def filter(self, rec):
                return rec.levelno in (logging.DEBUG, logging.INFO)

        class TqdmLoggingHandler(logging.Handler):
            """
            A logging handler that writes log messages via tqdm.write(),
            preventing them from overwriting or corrupting any active tqdm bars.
            """

            def __init__(self, stream=sys.stdout, level=logging.INFO):
                super().__init__(level)
                self.stream = stream

            def emit(self, record):
                try:
                    msg = self.format(record)
                    tqdm.write(msg, file=self.stream)
                    self.flush()
                except Exception:
                    self.handleError(record)

        class LevelBasedFormatter(logging.Formatter):
            def format(self, record):
                if record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
                    self._style._fmt = "[%(asctime)s] %(levelname)s " "[%(name)s.%(funcName)s:%(lineno)d] %(message)s"
                else:
                    self._style._fmt = "[%(asctime)s] %(levelname)s: %(message)s"
                return super().format(record)

        def _rotate_logs():
            """Rotates log files within self.LOG_DIR, handling up to MAX_BACKUPS."""
            base_log = os.path.join(self.LOG_DIR, "sync_ratings.log")
            for i in range(self.MAX_BACKUPS, 0, -1):
                old_log = f"{base_log}.{i}"
                new_log = f"{base_log}.{i + 1}"
                if os.path.exists(old_log):
                    if i == self.MAX_BACKUPS:
                        os.remove(old_log)
                    else:
                        os.rename(old_log, new_log)

            if os.path.exists(base_log):
                os.rename(base_log, f"{base_log}.1")

        os.makedirs(self.LOG_DIR, exist_ok=True)
        _rotate_logs()

        self.logger.setLevel(logging.DEBUG)

        # File Handler: Always at DEBUG, dynamic formatting
        file_formatter = LevelBasedFormatter(datefmt="%H:%M:%S")
        log_file_path = os.path.join(self.LOG_DIR, self.LOG_FILE)
        fh = logging.FileHandler(filename=log_file_path, encoding="utf-8", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_formatter)
        self.logger.addHandler(fh)

        # Console: Warning+ => stderr with TqdmLoggingHandler
        warn_format = logging.Formatter("%(levelname)s [%(funcName)s:%(lineno)d]: %(message)s")
        ch_err = TqdmLoggingHandler(stream=sys.stderr, level=logging.WARNING)
        ch_err.setFormatter(warn_format)
        self.logger.addHandler(ch_err)

        # Console: Info => stdout with TqdmLoggingHandler
        info_format = logging.Formatter("%(levelname)s: %(message)s")
        ch_std = TqdmLoggingHandler(stream=sys.stdout)
        ch_std.setFormatter(info_format)
        ch_std.addFilter(InfoFilter())
        self.logger.addHandler(ch_std)

        level = -1
        if isinstance(self.options.log, str):
            try:
                level = self.log_levels[self.options.log.upper()]
            except KeyError:
                pass
        elif isinstance(self.options.log, int):
            if 0 <= self.options.log <= 50:
                level = self.options.log

        if level < 0:
            print("Valid logging levels specified by either key or value:\n\t" + "\n\t".join(f"{key}: {value}" for key, value in self.log_levels.items()))
            raise RuntimeError(f"Invalid logging level selected: {level}")
        ch_err.setLevel(level)
        ch_std.setLevel(level)
        self.logger.info("Logging initialized with custom settings.")

    def sync(self) -> None:
        if self.options.clear_cache:
            self.cache_manager.invalidate()

        # Connect players appropriately based on which is Plex
        # TODO: Refactor this to be more dynamic
        if isinstance(self.source_player, PlexPlayer):
            self.source_player.connect(server=self.options.server, username=self.options.username, password=self.options.passwd, token=self.options.token)
            self.destination_player.connect()
        elif isinstance(self.destination_player, PlexPlayer):
            self.destination_player.connect(server=self.options.server, username=self.options.username, password=self.options.passwd, token=self.options.token)
            self.source_player.connect()

        for sync_item in self.options.sync:
            if sync_item.lower() == "tracks":
                self.logger.info(f"Starting to sync track ratings from {self.source_player.name()} to {self.destination_player.name()}")
                self.sync_tracks()
            elif sync_item.lower() == "playlists":
                self.logger.info(f"Starting to sync playlists from {self.source_player.name()} to {self.destination_player.name()}")
                self.sync_playlists()
            else:
                raise ValueError(f"Invalid sync item selected: {sync_item}")

    def sync_tracks(self) -> None:
        # Start discovery phase
        status = self.stats_manager.get_status_handler()
        tracks = self.source_player.search_tracks(key="rating", value=True)

        self.stats_manager.increment("tracks_processed", len(tracks))
        self.logger.info(f"Attempting to match {len(tracks)} tracks")

        # Start matching phase
        status = self.stats_manager.get_status_handler()
        bar = status.start_phase("Matching tracks", total=len(tracks))
        sync_pairs = []
        for track in tracks:
            pair = TrackPair(self.source_player, self.destination_player, track)
            sync_pairs.append(pair)
            pair.match()
            # Update progress description with current stats
            perfect = self.stats_manager.get("perfect_matches")
            good = self.stats_manager.get("good_matches")
            poor = self.stats_manager.get("poor_matches")
            no_match = self.stats_manager.get("no_matches")
            bar.set_description_str(f"100%: {perfect} | Good: {good} | Poor: {poor} | None: {no_match}")
            bar.update()
        bar.close()

        pairs_need_update = [pair for pair in sync_pairs if pair.sync_state is SyncState.NEEDS_UPDATE]
        self.logger.info(f"Synchronizing {len(pairs_need_update)} matching tracks without conflicts")

        bar = status.start_phase("Syncing tracks", total=len(pairs_need_update))
        for pair in pairs_need_update:
            pair.sync()
            bar.update()
        bar.close()

        pairs_conflicting = [pair for pair in sync_pairs if pair.sync_state is SyncState.CONFLICTING]
        choose = True
        while choose:
            choose = False
            if len(pairs_conflicting) > 0:
                self.stats_manager.set("tracks_conflicts", len(pairs_conflicting))
                prompt = {
                    "1": f"Keep all ratings from {pair.source_player.name()} and update {pair.destination_player.name()}",
                    "2": f"Keep all ratings from {pair.destination_player.name()} and update {pair.source_player.name()}",
                    "3": "Choose rating for each track",
                    "4": "Display all conflicts",
                    "5": "Display track match details",
                    "6": "Don't resolve conflicts",
                }
                for key in prompt:
                    print(f"\t[{key}]: {prompt[key]}")
                choice = input("Select how to resolve conflicting rating: ")
                if choice == "1":
                    if len(pairs_conflicting) > 0:
                        status = self.stats_manager.get_status_handler()
                        bar = status.start_phase("Resolving conflicts", total=len(pairs_conflicting))
                        for pair in pairs_conflicting:
                            pair.sync(force=True)
                            bar.update()
                        bar.close()
                elif choice == "2":
                    if len(pairs_conflicting) > 0:
                        status = self.stats_manager.get_status_handler()
                        bar = status.start_phase("Resolving conflicts", total=len(pairs_conflicting))
                        for pair in pairs_conflicting:
                            # Reverse source and destination assignment
                            (pair.source, pair.source_player, pair.rating_source, pair.destination, pair.destination_player, pair.rating_destination) = (
                                pair.destination,
                                pair.destination_player,
                                pair.rating_destination,
                                pair.source,
                                pair.source_player,
                                pair.rating_source,
                            )
                            pair.sync(force=True)
                            bar.update()
                        bar.close()
                elif choice == "3":
                    for i, pair in enumerate(pairs_conflicting, start=1):
                        result = pair.resolve_conflict(counter=i, total=len(pairs_conflicting))
                        if not result:
                            break
                elif choice == "4":
                    for pair in pairs_conflicting:
                        print(
                            f"Conflict: {pair.source} \
                                (Source - {pair.source_player.name()}: {pair.rating_source} \
                                    | Destination - {pair.destination_player.name()}: {pair.rating_destination})"
                        )
                    choose = True
                elif choice == "5":
                    sub_choice = input("Select tracks to display: [A]ll, [G]ood, [P]oor, [N]one: ").strip().upper()
                    if sub_choice == "A":
                        TrackPair.display_pair_details("All Matches", [pair for pair in sync_pairs if pair.score is not None and pair.sync_state is not SyncState.UP_TO_DATE])
                    elif sub_choice == "G":
                        TrackPair.display_pair_details("Good Matches", [pair for pair in sync_pairs if pair.score is not None and pair.score >= 80])
                    elif sub_choice == "P":
                        TrackPair.display_pair_details("Poor Matches", [pair for pair in sync_pairs if pair.score is not None and 30 <= pair.score < 80])
                    elif sub_choice == "N":
                        TrackPair.display_pair_details("No Matches", [pair for pair in sync_pairs if pair.score is None or pair.score < 30])
                    else:
                        print("Invalid choice. Please select [A], [G], [P], or [N].")
                    choose = True
                elif choice != "6":
                    print(f"{choice} is not a valid choice, please try again.")
                    choose = True

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
        status = self.stats_manager.get_status_handler()
        bar = status.start_phase("Matching playlists", total=len(playlist_pairs))
        for pair in playlist_pairs:
            pair.match()
            bar.update()
        bar.close()

        # Start playlist sync phase
        bar = status.start_phase("Syncing playlists", total=len(playlist_pairs))
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
            print(f"- Conflicts: {self.stats_manager.get('tracks_conflicts')}")  # TODO: this showed zero after showing conflicts to be updated

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
    parser.add_argument("--passwd", type=str, help="The password for the plex user. NOT RECOMMENDED TO USE!")
    parser.add_argument("--server", type=str, required=True, help="The name of the plex media server")
    parser.add_argument("--username", type=str, required=True, help="The plex username")
    parser.add_argument(
        "--token",
        type=str,
        help="Plex API token.  See https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/ for information on how to find your token",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        choices=["metadata", "matches", "matches-only", "disabled"],
        default="metadata",
        help="Cache mode: [metadata] (in-memory only), matches (both), matches-only (persistent matches), disabled",
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear existing cache files before starting")
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
