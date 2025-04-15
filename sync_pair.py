import abc
import logging
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
from fuzzywuzzy import fuzz

from manager import get_manager
from MediaPlayer import MediaPlayer
from ratings import Rating, RatingScale
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


class SyncPair(abc.ABC):
    source = None
    destination = None
    sync_state = SyncState.UNKNOWN

    def __init__(self, source_player: MediaPlayer, destination_player: MediaPlayer) -> None:
        mgr = get_manager()
        self.stats_mgr = mgr.get_stats_manager()
        self.cache_mgr = mgr.get_cache_manager()
        self.source_player: MediaPlayer = source_player
        self.destination_player: MediaPlayer = destination_player

    @abc.abstractmethod
    def match(self, *args, **kwargs) -> bool:
        """Tries to find a match on the destination player that matches the source replica as good as possible"""

    @abc.abstractmethod
    def resolve_conflict(self) -> bool:
        """Tries to resolve a conflict as good as possible and optionally prompts the user to resolve it manually"""

    @abc.abstractmethod
    def similarity(self, candidate: AudioTag) -> float:
        """Determines the similarity of the source replica with the candidate replica"""

    @abc.abstractmethod
    def sync(self, force: bool = False) -> bool:
        """Synchronizes the source and destination replicas"""


class TrackPair(SyncPair):
    rating_source: Optional[Rating] = None
    rating_destination: Optional[Rating] = None
    score: Optional[int] = None

    def __init__(self, source_player: MediaPlayer, destination_player: MediaPlayer, source_track: AudioTag) -> None:
        super(TrackPair, self).__init__(source_player, destination_player)
        self.logger = logging.getLogger("PlexSync.TrackPair")
        self.source = source_track

    def _record_match_quality(self, score: int) -> None:
        """Track match quality statistics based on the score."""
        if score >= MatchThresholds.PERFECT_MATCH:
            self.stats_mgr.increment("perfect_matches")
        elif score >= MatchThresholds.GOOD_MATCH:
            self.stats_mgr.increment("good_matches")
        elif score >= MatchThresholds.POOR_MATCH:
            self.stats_mgr.increment("poor_matches")
        else:
            self.stats_mgr.increment("no_matches")

    @staticmethod
    def display_pair_details(category: str, sync_pairs: List["TrackPair"]) -> None:
        """Display track details in a tabular format."""
        if not sync_pairs:
            print(f"\nNo tracks found for {category}.")
            return

        separator = "-" * 137
        print(f"\n{category}:\n{separator}")
        print(AudioTag.DISPLAY_HEADER)
        print(separator)

        for pair in sync_pairs:
            print(pair.source.details(pair.source_player))
            if pair.destination:
                print(pair.destination.details(pair.destination_player))
                print(separator)

    def albums_similarity(self, destination: Optional[AudioTag] = None) -> int:
        """Determines how similar two album names are. It takes into account different conventions for empty album names."""
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
        """Attempt to retrieve a cached match for the current source track."""
        if not self.cache_mgr.match_cache:
            return None

        cached_id, cached_score = self.cache_mgr.get_match(self.source.ID, source_name=self.source_player.name(), dest_name=self.destination_player.name())

        if not cached_id:
            return None

        candidates = self.destination_player.search_tracks(key="id", value=cached_id)
        if candidates:
            self.stats_mgr.increment("cache_hits")
            destination_track = candidates[0]

            if cached_score is None:
                cached_score = self.similarity(destination_track)
                self.cache_mgr.set_match(self.source.ID, destination_track.ID, self.source_player.name(), self.destination_player.name(), cached_score)

            self.score = cached_score
            return destination_track
        return None

    def _search_candidates(self) -> List[AudioTag]:
        """Search for track candidates matching the source track in the destination player."""
        if not self.source.title:
            self.logger.error(f"Source track has no title: {self.source.file_path}")
            return []
        try:
            candidates = self.destination_player.search_tracks(key="title", value=self.source.title)
            return candidates
        except ValueError as e:
            self.logger.error(f"Search failed for '{self.source.title}.")
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
        """Store a successful match in the cache along with its score."""
        if not self.cache_mgr.match_cache:
            return

        self.cache_mgr.set_match(self.source.ID, best_match.ID, self.source_player.name(), self.destination_player.name(), score)

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

    def match(self, candidates: Optional[List[AudioTag]] = None, match_threshold: int = MatchThresholds.MINIMUM_ACCEPTABLE) -> bool:
        """Find matching track on destination player"""
        if self.source is None:
            raise RuntimeError("Source track not set")

        best_match, score = self.find_best_match(candidates, match_threshold)
        self._record_match_quality(score)

        if not best_match:
            self.sync_state = SyncState.ERROR
            return False

        self.destination = best_match
        self.stats_mgr.increment("tracks_matched")

        src = self.rating_source = self.source.rating
        dst = self.rating_destination = self.destination.rating

        if src and dst and src == dst:
            self.sync_state = SyncState.UP_TO_DATE
        elif src.is_unrated or dst.is_unrated:
            self.sync_state = SyncState.NEEDS_UPDATE
        elif src != dst:
            self.sync_state = SyncState.CONFLICTING
            self.logger.warning(f"Found match with conflicting ratings: {self.source} " f"(Source: {src.to_display()} | " f"Destination: {dst.to_display()})")

        self.score = score
        return True

    def _get_new_rating(self) -> Optional[Rating]:
        """Prompt for and validate a new rating."""
        choice = input("Select the correct rating (0-5, half-star increments allowed): ").strip()
        validated = Rating.validate(choice, scale=RatingScale.ZERO_TO_FIVE)
        if validated is not None:
            return Rating(choice, scale=RatingScale.ZERO_TO_FIVE)
        print("Invalid input. Please enter a number between 0 and 5 in half-star increments.")
        return None

    def _apply_resolution(self, choice: str) -> bool:
        """Display conflict menu and apply the selected resolution."""

        if not choice:
            self.display_pair_details("Conflicting Tracks", [self])
            print("".join(f"\t[{key}]: {value}\n" for key, value in ConflictResolutionOptions.PROMPT.items()))
            return False

        if choice == "1":
            self.logger.info(f"Applying source rating {self.rating_source.to_display()}  to destination track {self.destination}")
            self.destination_player.update_rating(self.destination, self.rating_source)
            return True
        elif choice == "2":
            self.logger.info(f"Applying destination rating {self.rating_destination.to_display()} to source track {self.source}")
            self.source_player.update_rating(self.source, self.rating_destination)
            return True
        elif choice == "3":
            new_rating = self._get_new_rating()
            if new_rating is not None:
                self.logger.info(f"Applying new rating {new_rating.to_display()}  to both source and destination tracks")
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

    def resolve_conflict(self) -> bool:
        choice = ""
        while True:
            # Call _apply_resolution with empty choice to display menu first time
            if not choice:
                self._apply_resolution(choice)

            choice = input("Select how to resolve conflicting rating: ")
            resolution_applied = self._apply_resolution(choice)

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
        destination_unrated = self.rating_destination is None or self.rating_destination.is_unrated
        if destination_unrated or force:
            if source_to_destination:
                self.destination_player.update_rating(self.destination, self.rating_source)
            else:
                self.source_player.update_rating(self.source, self.rating_destination)
            self.stats_mgr.increment("tracks_updated")
            return True
        return False


class PlaylistPair(SyncPair):
    source: Playlist
    destination: Optional[Playlist] = None

    def __init__(self, source_player: MediaPlayer, destination_player: MediaPlayer, source_playlist: Playlist) -> None:
        """Initialize playlist pair with consistent naming"""
        super(PlaylistPair, self).__init__(source_player, destination_player)
        self.status_mgr = get_manager().get_status_manager()
        self.logger = logging.getLogger("PlexSync.PlaylistPair")
        self.source = source_playlist

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

        self.stats_mgr.increment("playlists_matched")
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
        self.stats_mgr.increment("playlists_created")
        return True

    def _update_existing_playlist(self, track_pairs: List[TrackPair]) -> bool:
        """Update an existing playlist on the destination player with missing tracks."""
        updates = []
        if len(track_pairs) > 0:
            bar = self.status_mgr.start_phase(f"Updating playlist '{self.source.name}'", total=len(track_pairs))
            for pair in track_pairs:
                if not self.destination.has_track(pair.destination):
                    updates.append((pair.destination, True))
                bar.update()
            bar.close()

        if updates:
            self.logger.debug(f"Adding {len(updates)} missing tracks to playlist {self.source.name}")
            self.destination_player.sync_playlist(self.destination, updates)
            self.stats_mgr.increment("playlists_updated")
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

        track_pairs = []
        unmatched = []
        if len(self.source.tracks) == 0:
            self.source_player.read_playlist_tracks(self.source)
        if len(self.source.tracks) > 0:
            bar = None
            if len(self.source.tracks) > 50:
                bar = self.status_mgr.start_phase(f"Matching tracks for playlist '{self.source.name}'", total=len(self.source.tracks))
            for track in self.source.tracks:
                bar.update() if bar else None
                pair = TrackPair(self.source_player, self.destination_player, track)
                pair.match()
                if pair.destination is not None:
                    track_pairs.append(pair)
                else:
                    unmatched.append(track)
            bar.close() if bar else None

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
