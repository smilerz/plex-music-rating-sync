#!/usr/bin/env python3
import locale
import logging
import time
from datetime import timedelta
from typing import List

from manager import get_manager
from manager.config_manager import PlayerType, SyncItem
from MediaPlayer import FileSystemPlayer, MediaMonkey, MediaPlayer, PlexPlayer
from ratings import Rating, RatingScale
from sync_items import AudioTag
from sync_pair import MatchThresholds, PlaylistPair, SyncState, TrackPair
from ui.help import ShowHelp
from ui.prompt import UserPrompt


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

    def _sync_ratings(self, pairs: List[TrackPair]) -> None:
        """Apply ratings for oriented TrackPairs. Assumes each pair is already in the correct sync direction."""

        if not pairs:
            print("No applicable tracks to update.")
            return

        source = pairs[0].source_player.name()
        destination = pairs[0].destination_player.name()
        label = f"{source} → {destination}"

        self.logger.info(f"Syncing ratings ({label}) for {len(pairs)} tracks.")

        bar = self.status_mgr.start_phase(f"Syncing ratings ({label})", total=len(pairs))
        for pair in pairs:
            pair.sync()
            bar.update()
        bar.close()

    def _display_matches_by_category(self, sync_pairs: List[TrackPair], category_filters: dict, prompt: UserPrompt) -> None:
        # Filter to only categories with matches
        nonempty_filters = {k: fn for k, fn in category_filters.items() if any(fn(p) for p in sync_pairs)}

        if not nonempty_filters:
            print("No matching categories to display.")
            return

        options = ["Show all", *list(nonempty_filters.keys())]
        selection = prompt.choice("Which match categories would you like to view?", options)
        selected_keys = list(nonempty_filters.keys()) if selection == "Show all" else [selection]

        for label in selected_keys:
            filter_fn = category_filters[label]
            matches = [p for p in sync_pairs if filter_fn(p)]
            if not matches:
                continue

            print(f"\n{label}: {len(matches)} tracks")
            header_line = AudioTag.DISPLAY_HEADER

            for i, pair in enumerate(matches, 1):
                if (i - 1) % 100 == 0:
                    print("-" * 137)
                    print(header_line)
                    print("-" * 137)

                print(pair.source.details(pair.source_player))
                if pair.destination:
                    print(pair.destination.details(pair.destination_player))
                else:
                    print("(No destination track matched)")
                print("-" * 137)

                if i % 100 == 0 and i != len(matches):
                    if not prompt.confirm_continue(f"[{i}/{len(matches)}] in {label} — Press Enter to continue or 'q' to quit: "):
                        break

    def sync_tracks(self) -> None:
        SYNC_OPTIONS = {
            "unrated": "Update only unrated destination tracks",
            "src_to_dst": "Overwrite destination ratings with source ratings",
            "dst_to_src": "Overwrite source ratings with destination ratings",
            "manual": "Manually resolve all ratings",
            "summary": "View a summary of match quality and conflict stats",
            "actionable_details": "Show details for tracks with pending updates (unrated or conflicting)",
            "all_match_details": "Show details for all tracks (matched, unmatched, or unchanged)",
            "cancel": "Cancel and exit without syncing",
        }
        prompt = UserPrompt()

        tracks = self.source_player.search_tracks(key="rating", value=True)
        self.stats_mgr.increment("tracks_processed", len(tracks))

        if not tracks:
            self.logger.warning("No tracks found")
            return

        self.logger.info(f"Attempting to match {len(tracks)} tracks")
        sync_pairs = self._match_tracks(tracks)

        # Build summary counts
        conflicts = [p for p in sync_pairs if p.sync_state == SyncState.CONFLICTING]
        unrated = [p for p in sync_pairs if p.sync_state == SyncState.NEEDS_UPDATE]
        if not conflicts and not unrated:
            return
        options = list(SYNC_OPTIONS.values())

        if not unrated:
            options.remove(SYNC_OPTIONS["unrated"])

        if not (conflicts or unrated):
            options.remove(SYNC_OPTIONS["src_to_dst"])

        if not conflicts:
            options.remove(SYNC_OPTIONS["dst_to_src"])
            options.remove(SYNC_OPTIONS["manual"])

        while True:
            if self.config_mgr.dry:
                print("[DRY RUN] No changes will be written.")
            choice = prompt.choice("How would you like to proceed?", options, help_text=ShowHelp.SyncOptions)

            if choice == SYNC_OPTIONS["unrated"]:
                self._sync_ratings(unrated)
                break

            elif choice == SYNC_OPTIONS["src_to_dst"]:
                self._sync_ratings(conflicts + unrated)
                break

            elif choice == SYNC_OPTIONS["dst_to_src"]:
                reversed_conflicts = [pair.reversed() for pair in conflicts]
                self._sync_ratings(reversed_conflicts)
                break

            elif choice == SYNC_OPTIONS["manual"]:
                for i, pair in enumerate(conflicts, 1):
                    src_desc = f"{pair.source_player.name():<20}: ({pair.source.title}) - Rating: {pair.rating_source.to_display()}"
                    dst_desc = f"{pair.destination_player.name():<20}: ({pair.destination.title}) - Rating: {pair.rating_destination.to_display()}"

                    manual_options = {
                        "src_to_dst": f"{src_desc}",
                        "dst_to_src": f"{dst_desc}",
                        "manual": "Enter a new rating",
                        "skip": "Skip this track",
                        "cancel": "Cancel conflict resolution",
                    }

                    print(f"Resolving conflict {i} of {len(conflicts)}:")
                    prompt_choice = prompt.choice("Choose how to resolve this conflict:", list(manual_options.values()))

                    if prompt_choice == manual_options["src_to_dst"]:
                        pair.sync()
                    elif prompt_choice == manual_options["dst_to_src"]:
                        pair.reversed().sync()
                    elif prompt_choice == manual_options["manual"]:
                        rating_input = prompt.text(
                            "Enter a new rating (0-5, half-star increments):",
                            validator=lambda val: Rating.validate(val, scale=RatingScale.ZERO_TO_FIVE) is not None,
                            help_text="Allowed values: 0, 0.5, 1, ..., 5",
                        )
                        if rating_input is not None:
                            rating = Rating(rating_input, scale=RatingScale.ZERO_TO_FIVE)
                            pair.source_player.update_rating(pair.source, rating)
                            pair.destination_player.update_rating(pair.destination, rating)
                    elif prompt_choice == manual_options["skip"]:
                        continue
                    elif prompt_choice == manual_options["cancel"]:
                        print("Manual resolution canceled.")
                        break
                break

            elif choice == SYNC_OPTIONS["summary"]:
                print("\n=== Match Summary ===")
                print(f"Unrated destination tracks: {len(unrated)}")
                print(f"Conflicts (perfect matches): {len([p for p in conflicts if p.score == MatchThresholds.PERFECT_MATCH])}")
                print(f"Conflicts (good matches):    {len([p for p in conflicts if MatchThresholds.GOOD_MATCH <= (p.score or 0) < MatchThresholds.PERFECT_MATCH])}")
                print(f"Conflicts (poor matches):    {len([p for p in conflicts if MatchThresholds.POOR_MATCH <= (p.score or 0) < MatchThresholds.GOOD_MATCH])}")
                print(f"No matches:                  {len([p for p in sync_pairs if (p.score or 0) < MatchThresholds.POOR_MATCH])}")

            elif choice == SYNC_OPTIONS["actionable_details"]:
                sync_category_filters = {
                    "Unrated": lambda p: p.sync_state == SyncState.NEEDS_UPDATE,
                    "Perfect": lambda p: p.sync_state == SyncState.CONFLICTING and p.score == MatchThresholds.PERFECT_MATCH,
                    "Good": lambda p: p.sync_state == SyncState.CONFLICTING and MatchThresholds.GOOD_MATCH <= (p.score or 0) < MatchThresholds.PERFECT_MATCH,
                    "Poor": lambda p: p.sync_state == SyncState.CONFLICTING and MatchThresholds.POOR_MATCH <= (p.score or 0) < MatchThresholds.GOOD_MATCH,
                    "No Match": lambda p: p.sync_state == SyncState.CONFLICTING and (p.score or 0) < MatchThresholds.POOR_MATCH,
                }
                print("\n=== Rating Details by Sync State and Quality ===")
                self._display_matches_by_category(sync_pairs, sync_category_filters, prompt)

            elif choice == SYNC_OPTIONS["all_match_details"]:
                score_category_filters = {
                    "Unrated": lambda p: p.sync_state == SyncState.NEEDS_UPDATE,
                    "Perfect": lambda p: p.score == MatchThresholds.PERFECT_MATCH,
                    "Good": lambda p: MatchThresholds.GOOD_MATCH <= (p.score or 0) < MatchThresholds.PERFECT_MATCH,
                    "Poor": lambda p: MatchThresholds.POOR_MATCH <= (p.score or 0) < MatchThresholds.GOOD_MATCH,
                    "No Match": lambda p: (p.score or 0) < MatchThresholds.POOR_MATCH,
                }
                print("\n=== Match Details by Match Quality ===")
                self._display_matches_by_category(sync_pairs, score_category_filters, prompt)
            elif choice == SYNC_OPTIONS["cancel"]:
                print("Sync tracks canceled.")
                break
        self.print_summary()

    def sync_playlists(self) -> None:
        # Start discovery phase
        print(f"Discovering playlists from {self.source_player.name()}")
        playlists = self.source_player.search_playlists("all")
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
        """Print and log the sync summary."""
        elapsed = time.time() - self.start_time
        elapsed_time = str(timedelta(seconds=int(elapsed)))

        summary_lines = [
            "\nSync Summary:",
            "-" * 50,
            f"Total time: {elapsed_time}",
        ]

        if SyncItem.TRACKS in self.config_mgr.sync:
            summary_lines.append("Tracks:")
            summary_lines.append(f"- Processed: {self.stats_mgr.get('tracks_processed')}")
            summary_lines.append(f"- Matched: {self.stats_mgr.get('tracks_matched')}")
            summary_lines.append(f"- Updated: {self.stats_mgr.get('tracks_updated')}")
            summary_lines.append(f"- Conflicts: {self.stats_mgr.get('tracks_conflicts')}")

            summary_lines.append("\nMatch Quality:")
            summary_lines.append(f"- Perfect matches (100%): {self.stats_mgr.get('perfect_matches')}")
            summary_lines.append(f"- Good matches (80-99%): {self.stats_mgr.get('good_matches')}")
            summary_lines.append(f"- Poor matches (30-79%): {self.stats_mgr.get('poor_matches')}")
            summary_lines.append(f"- No matches (<30%): {self.stats_mgr.get('no_matches')}")
            if self.config_mgr.log == "DEBUG":
                summary_lines.append(f"- Cache hits: {self.stats_mgr.get('cache_hits')}")

        if SyncItem.PLAYLISTS in self.config_mgr.sync:
            summary_lines.append("\nPlaylists:")
            summary_lines.append(f"- Processed: {self.stats_mgr.get('playlists_processed')}")
            summary_lines.append(f"- Matched: {self.stats_mgr.get('playlists_matched')}")
            summary_lines.append(f"- Created: {self.stats_mgr.get('playlists_created')}")
            summary_lines.append(f"- Updated: {self.stats_mgr.get('playlists_updated')}")

        if self.config_mgr.dry:
            summary_lines.append("\nThis was a DRY RUN - no changes were actually made.")
        summary_lines.append("-" * 50)

        # Log the summary to the debug log
        for line in summary_lines:
            self.logger.debug(line)

        # Print the summary only if the logger level is not DEBUG
        if self.config_mgr.log != logging.DEBUG:
            for line in summary_lines:
                print(line)


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
