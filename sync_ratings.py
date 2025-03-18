#!/usr/bin/env python3
import locale
import logging
import sys
import time
from datetime import timedelta

import configargparse

from cache_manager import CacheManager
from MediaPlayer import MediaMonkey, MediaPlayer, PlexPlayer
from stats_manager import StatsManager
from sync_pair import PlaylistPair, SyncState, TrackPair


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO)


class PlexSync:
    log_levels = {"CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR, "WARNING": logging.WARNING, "INFO": logging.INFO, "DEBUG": logging.DEBUG}

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

        self.cache_manager = CacheManager(options.cache_mode)
        self.stats_manager = StatsManager()

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
        self.logger.setLevel(logging.DEBUG)

        formatter_brief = logging.Formatter(fmt="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
        formatter_explicit = logging.Formatter(fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s", datefmt="%H:%M:%S")

        # Set up the file logger
        fh = logging.FileHandler(filename="sync_ratings.log", encoding="utf-8", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter_explicit)
        self.logger.addHandler(fh)

        # Set up the error / warning command line logger
        ch_err = logging.StreamHandler(stream=sys.stderr)
        ch_err.setFormatter(formatter_explicit)
        ch_err.setLevel(logging.WARNING)
        self.logger.addHandler(ch_err)

        # Set up the verbose info / debug command line logger
        ch_std = logging.StreamHandler(stream=sys.stdout)
        ch_std.setFormatter(formatter_brief)
        ch_std.addFilter(InfoFilter())
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
        else:
            ch_std.setLevel(level)
            self.logger.addHandler(ch_std)

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
        tracks = self.source_player.search_tracks(key="rating", value=True)
        self.stats_manager.increment("tracks_processed", len(tracks))
        self.logger.info(f"Attempting to match {len(tracks)} tracks")
        sync_pairs = [TrackPair(self.source_player, self.destination_player, track) for track in tracks]

        self.logger.info("Matching source tracks with destination player")
        matched = 0
        for pair in sync_pairs:
            if pair.match():
                matched += 1
        self.logger.info(f"Matched {matched}/{len(sync_pairs)} tracks")

        if self.options.dry:
            self.logger.info("Running a DRY RUN. No changes will be propagated!")
        pairs_need_update = [pair for pair in sync_pairs if pair.sync_state is SyncState.NEEDS_UPDATE]
        self.logger.info(f"Synchronizing {len(pairs_need_update)} matching tracks without conflicts")
        for pair in pairs_need_update:
            pair.sync()

        pairs_conflicting = [pair for pair in sync_pairs if pair.sync_state is SyncState.CONFLICTING]
        self.logger.info(f"{len(pairs_conflicting)} pairs have conflicting ratings")
        self.stats_manager.increment("tracks_conflicts", len(pairs_conflicting))

        choose = True
        while choose:
            choose = False
            if len(pairs_conflicting) > 0:
                prompt = {
                    "1": f"Keep all ratings from {pair.source_player.name()} and update {pair.destination_player.name()}",
                    "2": f"Keep all ratings from {pair.destination_player.name()} and update {pair.source_player.name()}",
                    "3": "Choose rating for each track",
                    "4": "Display all conflicts",
                    "5": "Don't resolve conflicts",
                }
                for key in prompt:
                    print(f"\t[{key}]: {prompt[key]}")
                choice = input("Select how to resolve conflicting rating: ")
                if choice == "1":
                    for pair in pairs_conflicting:
                        pair.sync(force=True)
                elif choice == "2":
                    for pair in pairs_conflicting:
                        # reverse source and destination assignment
                        (pair.source, pair.source_player, pair.rating_source, pair.destination, pair.destination_player, pair.rating_destination) = (
                            pair.destination,
                            pair.destination_player,
                            pair.rating_destination,
                            pair.source,
                            pair.source_player,
                            pair.rating_source,
                        )
                        pair.sync(force=True)
                elif choice == "3":
                    for pair in pairs_conflicting:
                        result = pair.resolve_conflict()
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
                elif choice != "5":
                    print(f"{choice} is not a valid choice, please try again.")
                    choose = True

    def sync_playlists(self) -> None:
        playlists = self.source_player.read_playlists()
        playlist_pairs = [PlaylistPair(self.source_player, self.destination_player, pl) for pl in playlists if not pl.is_auto_playlist]
        self.stats_manager.increment("playlists_processed", len(playlist_pairs))

        if self.options.dry:
            self.logger.info("Running a DRY RUN. No changes will be propagated!")

        self.logger.info(f"Matching {self.source_player.name()} playlists with {self.destination_player.name()}")
        for pair in playlist_pairs:
            pair.match()

        self.logger.info(f"Synchronizing {len(playlist_pairs)} matching playlists")
        for pair in playlist_pairs:
            pair.sync()

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
    parser.add_argument("--source", type=str, default="mediamonkey", help="Source player (plex or mediamonkey)")
    parser.add_argument("--destination", type=str, default="plex", help="Destination player (plex or mediamonkey)")
    parser.add_argument("--sync", nargs="*", default=["tracks"], help="Selects which items to sync: one or more of [tracks, playlists]")
    parser.add_argument("--log", default="info", help="Sets the logging level")
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
        help="Cache mode: metadata (in-memory only), matches (both), matches-only (persistent matches), disabled",
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
