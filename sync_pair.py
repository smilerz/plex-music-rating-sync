import abc
import logging
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
from fuzzywuzzy import fuzz

from MediaPlayer import MediaPlayer
from sync_items import AudioTag, Playlist


class SyncState(Enum):
    UNKNOWN = auto()
    UP_TO_DATE = auto()
    NEEDS_UPDATE = auto()
    CONFLICTING = auto()
    ERROR = auto()


class MatchThresholds:
    MINIMUM_ACCEPTABLE = 30
    POOR_MATCH = MINIMUM_ACCEPTABLE
    GOOD_MATCH = 80
    PERFECT_MATCH = 100


class ConflictResolutionOptions:
    PROMPT = {
        "1": "Apply source rating to destination",
        "2": "Apply destination rating to source",
        "3": "Enter a new rating",
        "4": "Skip this track",
        "5": "Cancel resolving conflicts",
    }
    MIN_RATING = 0
    MAX_RATING = 10


class TruncateDefaults:
    MAX_ARTIST_LENGTH = 20
    MAX_ALBUM_LENGTH = 25
    MAX_TITLE_LENGTH = 35
    MAX_FILE_PATH_LENGTH = 50


NO_MATCHES_HEADER = (
    f"{'':<2} {'Source Artist':<{TruncateDefaults.MAX_ARTIST_LENGTH}} "
    f"{'Source Album':<{TruncateDefaults.MAX_ALBUM_LENGTH}} "
    f"{'Source Title':<{TruncateDefaults.MAX_TITLE_LENGTH}} {'Score/Status':<15}"
)
MATCHES_HEADER = (
    f"{'':<2} {'Track #':>5} {'Artist':<{TruncateDefaults.MAX_ARTIST_LENGTH}} "
    f"{'Album':<{TruncateDefaults.MAX_ALBUM_LENGTH}} "
    f"{'Title':<{TruncateDefaults.MAX_TITLE_LENGTH}} "
    f"{'File Path':<{TruncateDefaults.MAX_FILE_PATH_LENGTH}}"
)


class SyncPair(abc.ABC):
    source = None
    destination = None
    sync_state = SyncState.UNKNOWN

    def __init__(self, source_player: MediaPlayer, destination_player: MediaPlayer) -> None:
        self.source_player: MediaPlayer = source_player
        self.destination_player: MediaPlayer = destination_player
        self.stats_manager = source_player.stats_manager

    def _record_match_quality(self, score: int) -> None:
        """Track match quality statistics based on the score."""
        if score >= MatchThresholds.PERFECT_MATCH:
            self.stats_manager.increment("perfect_matches")
        elif score >= MatchThresholds.GOOD_MATCH:
            self.stats_manager.increment("good_matches")
        elif score >= MatchThresholds.POOR_MATCH:
            self.stats_manager.increment("poor_matches")
        else:
            self.stats_manager.increment("no_matches")

    @abc.abstractmethod
    def match(self) -> bool:
        """Tries to find a match on the destination player that matches the source replica as good as possible"""

    @abc.abstractmethod
    def resolve_conflict(self, counter: int, total: int) -> bool:
        """Tries to resolve a conflict as good as possible and optionally prompts the user to resolve it manually"""

    @abc.abstractmethod
    def similarity(self, candidate: AudioTag) -> float:
        """Determines the similarity of the source replica with the candidate replica"""

    @abc.abstractmethod
    def sync(self, force: bool = False) -> bool:
        """
        Synchronizes the source and destination replicas
        :return flag indicating success
        :rtype: bool
        """


class TrackPair(SyncPair):
    rating_source = 0.0
    rating_destination = 0.0
    score = None

    def __init__(self, source_player: MediaPlayer, destination_player: MediaPlayer, source_track: AudioTag) -> None:
        super(TrackPair, self).__init__(source_player, destination_player)
        self.logger = logging.getLogger("PlexSync.TrackPair")
        self.source = source_track
        self.stats_manager = source_player.stats_manager

    @staticmethod
    def truncate(value: str, length: int, from_end: bool = True) -> str:
        """Truncate a string to the specified length, adding '...' at the start or end."""
        if len(value) <= length:
            return value
        return f"{value[:length - 3]}..." if from_end else f"...{value[-(length - 3):]}"

    @staticmethod
    def display_pair_details(category: str, sync_pairs: List["TrackPair"]) -> None:
        """Display track details in a tabular format."""

        def safe_get(attr: str, default: str = "N/A"):
            """Safely retrieve an attribute or return default if not present."""
            return attr if attr else default

        if not sync_pairs:
            print(f"\nNo tracks found for {category}.\n")
            return

        print(f"\n{category}:\n{'-' * 50}")
        if category == "No Matches":
            print(NO_MATCHES_HEADER)
            print(f"{'-' * 92}")
            for pair in sync_pairs:
                print(
                    f"{pair.source_player.abbr:<2} {safe_get(pair.source.artist):<20} {safe_get(pair.source.album):<20} "
                    f"{safe_get(pair.source.title):<30} {pair.sync_state.name if pair.score is None else pair.score:<15}"
                )
        else:
            print(MATCHES_HEADER)
            print(f"{'-' * 137}")

            for pair in sync_pairs:
                source_abbr = pair.source_player.abbr
                source_track = safe_get(pair.source.track)
                source_artist = TrackPair.truncate(safe_get(pair.source.artist), TruncateDefaults.MAX_ARTIST_LENGTH)
                source_album = TrackPair.truncate(safe_get(pair.source.album), TruncateDefaults.MAX_ALBUM_LENGTH)
                source_title = TrackPair.truncate(safe_get(pair.source.title), TruncateDefaults.MAX_TITLE_LENGTH)
                source_path = TrackPair.truncate(safe_get(pair.source.file_path), TruncateDefaults.MAX_FILE_PATH_LENGTH, from_end=False)

                print(f"{source_abbr:<2} {source_track:>5} {source_artist:<20} {source_album:<25} {source_title:<35} {source_path:<50}")

                if pair.destination:
                    dest_abbr = pair.destination_player.abbr
                    dest_track = safe_get(pair.destination.track)
                    dest_artist = TrackPair.truncate(safe_get(pair.destination.artist), TruncateDefaults.MAX_ARTIST_LENGTH)
                    dest_album = TrackPair.truncate(safe_get(pair.destination.album), TruncateDefaults.MAX_ALBUM_LENGTH)
                    dest_title = TrackPair.truncate(safe_get(pair.destination.title), TruncateDefaults.MAX_TITLE_LENGTH)
                    dest_path = TrackPair.truncate(safe_get(pair.destination.file_path), TruncateDefaults.MAX_FILE_PATH_LENGTH, from_end=False)

                    print(f"{dest_abbr:<2} {dest_track:>5} {dest_artist:<20} {dest_album:<25} {dest_title:<35} {dest_path:<50}")
                print("-" * 137)

        print("\n")

    def albums_similarity(self, destination: Optional[AudioTag] = None) -> int:
        """
        Determines how similar two album names are. It takes into account different conventions for empty album names.
        :type destination: str
                        optional album title to compare the album name of the source track with
        :returns a similarity rating [0, 100]
        :rtype: int
        """
        if destination is None:
            destination = self.destination
        if self.both_albums_empty(destination=destination):
            return MatchThresholds.PERFECT_MATCH
        else:
            return fuzz.ratio(self.source.album, destination.album)

    def both_albums_empty(self, destination: Optional[AudioTag] = None) -> bool:
        if destination is None:
            destination = self.destination

        return self.source_player.album_empty(self.source.album) and self.destination_player.album_empty(destination.album)

    def _get_cache_match(self) -> Optional[AudioTag]:
        """Attempt to retrieve a cached match for the current source track.

        Returns:
            The matched AudioTag with full metadata if needed, otherwise None.
        """
        if not self.source_player.cache_manager:
            return None

        cached_id, cached_score = self.source_player.cache_manager.get_match(self.source.ID, source_name=self.source_player.name(), dest_name=self.destination_player.name())

        if not cached_id:
            return None

        candidates = self.destination_player.search_tracks(key="id", value=cached_id)
        if candidates:
            self.stats_manager.increment("cache_hits")
            destination_track = candidates[0]

            if cached_score is None:
                cached_score = self.similarity(destination_track)
                self.source_player.cache_manager.set_match(self.source.ID, destination_track.ID, self.source_player.name(), self.destination_player.name(), cached_score)

            self.score = cached_score
            return destination_track
        return None

    def _search_candidates(self) -> List[AudioTag]:
        """Search for track candidates matching the source track in the destination player."""
        try:
            candidates = self.destination_player.search_tracks(key="title", value=self.source.title)
            return candidates
        except ValueError as e:
            self.logger.error(f"Failed to search tracks for track '{self.source}' stored at {self.source.file_path}.")
            raise e

    def _get_best_match(self, candidates: List[AudioTag], match_threshold: int = MatchThresholds.MINIMUM_ACCEPTABLE) -> Tuple[Optional[AudioTag], int]:
        """Find the best matching track from a list of candidates based on similarity score."""
        if not candidates:
            return None, 0

        scores = np.array([self.similarity(candidate) for candidate in candidates])
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < match_threshold:
            self.logger.debug(f"Best candidate score too low: {best_score} < {match_threshold}")
            return None, best_score

        best_match = candidates[best_idx]
        return best_match, best_score

    def _set_cache_match(self, best_match: AudioTag, score: Optional[float] = None) -> None:
        """Store a successful match in the cache along with its score.

        Args:
            best_match: The best matching track found.
            score: The similarity score of the match (optional).
        """
        if not self.source_player.cache_manager:
            return

        self.source_player.cache_manager.set_match(self.source.ID, best_match.ID, self.source_player.name(), self.destination_player.name(), score)

    def find_best_match(self, candidates: Optional[List[AudioTag]] = None, match_threshold: int = MatchThresholds.MINIMUM_ACCEPTABLE) -> Tuple[Optional[AudioTag], int]:
        """Find the best matching track from candidates or by searching."""
        cached_match = self._get_cache_match()
        if cached_match:
            return cached_match, self.score

        if candidates is None:
            candidates = self._search_candidates()

        if not candidates:
            self.logger.warning(f"No candidates found for {self.source}")
            return None, 0

        best_match, best_score = self._get_best_match(candidates, match_threshold)

        if best_match:
            self._set_cache_match(best_match, best_score)

            self.logger.debug(f"Found match with score {best_score} for {self.source} - : - {best_match}")
            if best_score != MatchThresholds.PERFECT_MATCH:
                self.logger.info(f"Found match with score {best_score} for {self.source}: {best_match}")
            if best_score < MatchThresholds.GOOD_MATCH:
                self.logger.debug(f"Source: {self.source}")
                self.logger.debug(f"Best Match: {best_match}")

        return best_match, best_score

    def match(self, candidates: Optional[List[AudioTag]] = None, match_threshold: int = MatchThresholds.MINIMUM_ACCEPTABLE) -> int:
        """Find matching track on destination player"""
        if self.source is None:
            raise RuntimeError("Source track not set")

        best_match, score = self.find_best_match(candidates, match_threshold)
        self._record_match_quality(score)

        if not best_match:
            self.sync_state = SyncState.ERROR
            return score

        self.destination = best_match
        self.stats_manager.increment("tracks_matched")

        self.rating_source = self.source.rating
        self.rating_destination = self.destination.rating

        if self.rating_source == self.rating_destination:
            self.sync_state = SyncState.UP_TO_DATE
        elif self.rating_source == 0.0 or self.rating_destination == 0.0:
            self.sync_state = SyncState.NEEDS_UPDATE
        elif self.rating_source != self.rating_destination:
            self.sync_state = SyncState.CONFLICTING
            self.logger.warning(f"Found match with conflicting ratings: {self.source} (Source: {self.rating_source} | Destination: {self.rating_destination})")

        self.score = score
        return self.score

    def _get_new_rating(self) -> Optional[float]:
        """Prompt for and validate a new rating."""
        new_rating_input = input(f"Please enter a rating between {ConflictResolutionOptions.MIN_RATING} and {ConflictResolutionOptions.MAX_RATING}: ")
        try:
            new_rating = int(new_rating_input) / 10
            if new_rating < ConflictResolutionOptions.MIN_RATING / 10:
                raise Exception("Ratings below 0 not allowed")
            elif new_rating > ConflictResolutionOptions.MAX_RATING / 10:
                raise Exception("Ratings above 10 not allowed")
            return new_rating
        except Exception as e:
            print("Error:", e)
            print(
                f"Rating {new_rating_input} is not a valid rating, please choose an integer between {ConflictResolutionOptions.MIN_RATING} "
                f"and {ConflictResolutionOptions.MAX_RATING}"
            )
            return None

    def _apply_resolution(self, choice: str, counter: int, total: int) -> bool:
        """Display conflict menu and apply the selected resolution."""

        if not choice:
            print(f"\nResolving conflict {counter} of {total}:")
            print("\n".join(f"\t[{key}]: {value}" for key, value in ConflictResolutionOptions.PROMPT.items()))
            return False

        if choice == "1":
            self.destination_player.update_rating(self.destination, self.rating_source)
            return True
        elif choice == "2":
            self.source_player.update_rating(self.source, self.rating_destination)
            return True
        elif choice == "3":
            new_rating = self._get_new_rating()
            if new_rating is not None:
                self.destination_player.update_rating(self.destination, new_rating)
                self.source_player.update_rating(self.source, new_rating)
                return True
            return False
        elif choice == "4":
            return True
        elif choice == "5":
            return False
        else:
            print(f"{choice} is not a valid choice, please try again.")
            return False

    def resolve_conflict(self, counter: int, total: int) -> bool:
        choice = ""
        while True:
            # Call _apply_resolution with empty choice to display menu first time
            if not choice:
                self._apply_resolution(choice, counter, total)

            choice = input("Select how to resolve conflicting rating: ")
            resolution_applied = self._apply_resolution(choice, counter, total)

            if resolution_applied or choice == "5":
                return resolution_applied

            # Reset choice to empty string if we need to show the menu again
            if choice not in ConflictResolutionOptions.PROMPT:
                choice = ""

    def similarity(self, candidate: AudioTag) -> float:
        """Determines the matching similarity of @candidate with the source query track"""
        scores = np.array(
            [
                fuzz.ratio(self.source.title, candidate.title),
                fuzz.ratio(self.source.artist, candidate.artist),
                MatchThresholds.PERFECT_MATCH if self.source.track == candidate.track else 0,
                self.albums_similarity(destination=candidate),
            ]
        )
        return np.average(scores)

    def sync(self, force: bool = False, source_to_destination: bool = False) -> bool:
        """Synchronizes the source and destination replicas.:return: True if synchronization was successful, False otherwise"""
        if self.rating_destination <= 0.0 or force:
            if source_to_destination:
                self.destination_player.update_rating(self.destination, self.rating_source)
            else:
                self.source_player.update_rating(self.source, self.rating_destination)
            self.stats_manager.increment("tracks_updated")
            return True
        return False


class PlaylistPair(SyncPair):
    source: Playlist
    destination: Optional[Playlist] = None

    def __init__(self, source_player: MediaPlayer, destination_player: MediaPlayer, source_playlist: Playlist) -> None:
        """Initialize playlist pair with consistent naming"""
        super(PlaylistPair, self).__init__(source_player, destination_player)
        self.logger = logging.getLogger("PlexSync.PlaylistPair")
        self.source = source_playlist
        self.stats_manager = source_player.stats_manager

    def match(self) -> bool:
        """Find matching playlist on destination player based on name."""
        self.destination = self.destination_player.find_playlist(title=self.source.name)

        if self.destination is None:
            self.sync_state = SyncState.NEEDS_UPDATE
            self.logger.info(f"Playlist {self.source.name} needs to be created")
            return False

        missing = self.destination.missing_tracks(self.source)
        if missing:
            self.sync_state = SyncState.NEEDS_UPDATE
            self.logger.info(f"Playlist {self.source.name} needs {len(missing)} tracks added")
        else:
            self.sync_state = SyncState.UP_TO_DATE
            self.logger.info(f"Playlist {self.source.name} is up to date")

        self.stats_manager.increment("playlists_matched")
        return True

    def resolve_conflict(self) -> bool:
        """Resolves conflicts for playlists."""
        raise NotImplementedError

    def similarity(self, candidate: Playlist) -> float:
        """Determines the similarity of the playlist with a candidate."""
        raise NotImplementedError

    def _create_new_playlist(self, track_pairs: List[TrackPair]) -> bool:
        """Create a new playlist on the destination player with matched tracks."""
        destination_tracks = [pair.destination for pair in track_pairs]
        self.destination = self.destination_player.create_playlist(self.source.name, destination_tracks)
        self.logger.info(f"Created new playlist {self.source.name} with {len(destination_tracks)} tracks")
        self.stats_manager.increment("playlists_updated")
        return True

    def _update_existing_playlist(self, track_pairs: List[TrackPair]) -> bool:
        """Update an existing playlist on the destination player with missing tracks."""
        updates = []
        if len(track_pairs) > 0:
            status = self.stats_manager.get_status_handler()
            bar = status.start_phase(f"Updating playlist '{self.source.name}'", total=len(track_pairs))
            for pair in track_pairs:
                if not self.destination.has_track(pair.destination):
                    updates.append((pair.destination, True))
                bar.update()
            bar.close()

        if updates:
            self.logger.debug(f"Adding {len(updates)} missing tracks to playlist {self.source.name}")
            self.destination_player.sync_playlist(self.destination, updates)
            self.stats_manager.increment("playlists_updated")
            return True
        else:
            self.logger.debug(f"Playlist {self.source.name} is up to date")
            return False

    def sync(self, force: bool = False) -> bool:
        """Non-destructive one-way sync from source to destination."""
        if self.sync_state is SyncState.UP_TO_DATE:
            return True

        self.logger.info(f"Synchronizing playlist {self.source.name}")
        self.logger.debug(f"Source playlist has {len(self.source.tracks)} tracks")

        # Match tracks from source playlist to destination player (merged from _match_tracks)
        track_pairs = []
        unmatched = []
        if len(self.source.tracks) > 0:
            for track in self.source.tracks:
                pair = TrackPair(self.source_player, self.destination_player, track)
                pair.match()
                if pair.destination is not None:
                    track_pairs.append(pair)
                else:
                    unmatched.append(track)

        self.logger.info(f"Matched {len(track_pairs)}/{len(self.source.tracks)} tracks for playlist {self.source.name}")
        if unmatched:
            self.logger.warning(f"Failed to match {len(unmatched)} tracks:")
            for track in unmatched:
                self.logger.warning(f"  - {track}")

        if not track_pairs:
            self.logger.warning(f"No tracks could be matched for playlist {self.source.name}")
            return False

        if self.destination is None:
            return self._create_new_playlist(track_pairs)
        else:
            return self._update_existing_playlist(track_pairs)
