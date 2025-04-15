#!/usr/bin/env python3
import locale
import logging
import time
from datetime import timedelta
from typing import List

from manager import get_manager
from manager.config_manager import PlayerType, SyncItem
from MediaPlayer import FileSystemPlayer, MediaMonkey, MediaPlayer, PlexPlayer
from sync_items import AudioTag
from sync_pair import PlaylistPair, SyncState, TrackPair


class PlexSync:
    def __init__(self) -> None:
        self.logger = logging.getLogger("PlexSync.PlaylistPair")
        mgr = get_manager()
        self.config_mgr = mgr.get_config_manager()
        self.stats_mgr = mgr.get_stats_manager()
        self.status_mgr = mgr.get_status_manager()

        try:
            self.source_player = self._create_player(self.config_mgr.source)
            self.destination_player = self._create_player(self.config_mgr.destination)
        except ValueError:
            exit(1)

        self.conflicts = []
        self.updates = []
        self.start_time = time.time()

    def _create_player(self, player_type: str) -> MediaPlayer:
        """Create and configure a media player instance"""
        player_map = {PlayerType.PLEX: PlexPlayer, PlayerType.MEDIAMONKEY: MediaMonkey, PlayerType.FILESYSTEM: FileSystemPlayer}

        if player_type not in player_map:
            self.logger.error(f"Invalid player type: {player_type}")
            self.logger.error(f"Supported players: {', '.join(player_map.keys())}")
            raise ValueError(f"Invalid player type: {player_type}")

        player = player_map[player_type]()
        player.dry_run = self.config_mgr.dry
        return player

    def sync(self) -> None:
        # Connect players with appropriate parameters based on player type
        for player in [self.source_player, self.destination_player]:
            player.connect()

        for sync_item in self.config_mgr.sync:
            if sync_item == SyncItem.TRACKS:
                self.logger.info(f"Starting to sync track ratings from {self.source_player.name()} to {self.destination_player.name()}")
                self.sync_tracks()
            elif sync_item == SyncItem.PLAYLISTS:
                self.logger.info(f"Starting to sync playlists from {self.source_player.name()} to {self.destination_player.name()}")
                self.sync_playlists()
            else:
                raise ValueError(f"Invalid sync item selected: {sync_item}")

    def _get_match_summary(self) -> str:
        """Generate a summary of match quality statistics."""
        return (
            f"100%: {self.stats_mgr.get('perfect_matches')} | "
            f"Good: {self.stats_mgr.get('good_matches')} | "
            f"Poor: {self.stats_mgr.get('poor_matches')} | "
            f"None: {self.stats_mgr.get('no_matches')}"
        )

    def _match_tracks(self, tracks: List[AudioTag]) -> List[TrackPair]:
        """Match tracks between source and destination players"""
        print(f"Matching tracks between from {self.source_player} to {self.destination_player}")
        bar = self.status_mgr.start_phase("Matching tracks", total=len(tracks))
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
        """Sync tracks that need updates"""
        pairs_need_update = [pair for pair in sync_pairs if pair.sync_state is SyncState.NEEDS_UPDATE]
        self.logger.info(f"Synchronizing {len(pairs_need_update)} matching tracks without conflicts")

        bar = self.status_mgr.start_phase("Syncing tracks", total=len(pairs_need_update))
        for pair in pairs_need_update:
            pair.sync()
            bar.update()
        bar.close()

    def _display_conflict_options(self) -> str:
        """Display conflict resolution options and get user choice."""
        # TODO: add only update perfect and only update good & perfect matches, selection would loop back to this menu
        #     it add would replace selected option with All and replace the word all in option two with selected option
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

        bar = self.status_mgr.start_phase("Resolving conflicts", total=len(pairs_conflicting))
        for pair in pairs_conflicting:
            pair.sync(force=True, source_to_destination=source_to_destination)
            bar.update()
        bar.close()

    def _display_track_details(self, sync_pairs: List[TrackPair]) -> None:
        """Display track match details based on user selection."""
        valid_choices = {"G", "P", "N"}
        choice_labels = {"G": "Good", "P": "Poor", "N": "No"}

        while True:
            sub_choice = input("Select tracks to display: To Be [G]ood Matches, [P]oor Matches, [N]o Matches: ").strip().upper()
            if sub_choice in valid_choices:
                break
            print("Invalid choice. Please select [G], [P], or [N].")

        filters = {
            "G": lambda pair: pair.score is not None and pair.score >= 80,
            "P": lambda pair: pair.score is not None and 30 <= pair.score < 80,
            "N": lambda pair: pair.score is None or pair.score < 30,
        }
        TrackPair.display_pair_details(f"{choice_labels[sub_choice]} Matches", [pair for pair in sync_pairs if filters[sub_choice](pair)])

    def _resolve_conflicts(self, pairs_conflicting: List[TrackPair], sync_pairs: List[TrackPair]) -> None:
        """Resolve conflicts between source and destination ratings"""
        self.stats_mgr.set("tracks_conflicts", len(pairs_conflicting))

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
                    print(f"\nResolving conflict {i} of {len(pairs_conflicting)}:")
                    result = pair.resolve_conflict()
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

        self.stats_mgr.increment("tracks_processed", len(tracks))
        if len(tracks) == 0:
            self.logger.warning("No tracks found")
            return
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
        if not playlists:
            self.logger.warning("No playlists found")
            return

        playlist_pairs = [PlaylistPair(self.source_player, self.destination_player, pl) for pl in playlists if not pl.is_auto_playlist]
        self.stats_mgr.increment("playlists_processed", len(playlist_pairs))

        if self.config_mgr.dry:
            self.logger.info("Running a DRY RUN. No changes will be propagated!")

        self.logger.info(f"Matching {self.source_player.name()} playlists with {self.destination_player.name()}")

        # Start playlist matching phase
        bar = None
        for pair in playlist_pairs:
            if bar is None:
                bar = self.status_mgr.start_phase("Matching playlists", total=len(playlist_pairs))
            pair.match()
            bar.update()
        if bar is not None:
            bar.close()

        # Start playlist sync phase
        bar = None
        for pair in playlist_pairs:
            if bar is None:
                bar = self.status_mgr.start_phase("Syncing playlists", total=len(playlist_pairs))
            pair.sync()
            bar.update()
        if bar is not None:
            bar.close()

    def print_summary(self) -> None:
        elapsed = time.time() - self.start_time
        elapsed_time = str(timedelta(seconds=int(elapsed)))

        print("\nSync Summary:")
        print("-" * 50)
        print(f"Total time: {elapsed_time}")

        if SyncItem.TRACKS in self.config_mgr.sync:  # Use enum directly instead of string
            print("Tracks:")
            print(f"- Processed: {self.stats_mgr.get('tracks_processed')}")
            print(f"- Matched: {self.stats_mgr.get('tracks_matched')}")
            print(f"- Updated: {self.stats_mgr.get('tracks_updated')}")
            print(f"- Conflicts: {self.stats_mgr.get('tracks_conflicts')}")

            print("\nMatch Quality:")
            print(f"- Perfect matches (100%): {self.stats_mgr.get('perfect_matches')}")
            print(f"- Good matches (80-99%): {self.stats_mgr.get('good_matches')}")
            print(f"- Poor matches (30-79%): {self.stats_mgr.get('poor_matches')}")
            print(f"- No matches (<30%): {self.stats_mgr.get('no_matches')}")
            if self.config_mgr.log == "DEBUG":
                print(f"- Cache hits: {self.stats_mgr.get('cache_hits')}")

        if SyncItem.PLAYLISTS in self.config_mgr.sync:  # Use enum directly instead of string
            print("\nPlaylists:")
            print(f"- Processed: {self.stats_mgr.get('playlists_processed')}")
            print(f"- Matched: {self.stats_mgr.get('playlists_matched')}")
            print(f"- Created: {self.stats_mgr.get('playlists_created')}")
            print(f"- Updated: {self.stats_mgr.get('playlists_updated')}")

        if self.config_mgr.dry:
            print("\nThis was a DRY RUN - no changes were actually made.")
        print("-" * 50)


if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, "")

    get_manager().initialize()
    config_mgr = get_manager().get_config_manager()
    cache_mgr = get_manager().get_cache_manager()
    if config_mgr.clear_cache:
        cache_mgr.invalidate()

    sync_agent = PlexSync()
    sync_agent.sync()
    sync_agent.print_summary()

    cache_mgr.cleanup()
