import abc
import logging
from enum import Enum, auto
from typing import Optional, Tuple

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


class SyncPair(abc.ABC):
    source = None
    destination = None
    sync_state = SyncState.UNKNOWN

    def __init__(self, source_player, destination_player) -> None:
        # """
        # TODO: this is no longer true - not sure if it matters
        # :type local_player: MediaPlayer.MediaPlayer
        # :type remote_player: MediaPlayer.PlexPlayer
        # """
        self.source_player: MediaPlayer = source_player
        self.destination_player: MediaPlayer = destination_player

    @abc.abstractmethod
    def match(self) -> bool:
        """Tries to find a match on the destination player that matches the source replica as good as possible"""

    @abc.abstractmethod
    def resolve_conflict(self) -> bool:
        """Tries to resolve a conflict as good as possible and optionally prompts the user to resolve it manually"""

    @abc.abstractmethod
    def similarity(self, candidate) -> float:
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

    def __init__(self, source_player, destination_player, source_track: AudioTag) -> None:
        super(TrackPair, self).__init__(source_player, destination_player)
        self.logger = logging.getLogger("PlexSync.TrackPair")
        self.source = source_track
        self.stats_manager = source_player.stats_manager

    def albums_similarity(self, destination=None) -> int:
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
            return 100
        else:
            return fuzz.ratio(self.source.album, destination.album)

    def both_albums_empty(self, destination=None) -> bool:
        if destination is None:
            destination = self.destination

        return self.source_player.album_empty(self.source.album) and self.destination_player.album_empty(destination.album)

    def find_best_match(self, candidates=None, match_threshold=30) -> Tuple[Optional[AudioTag], int]:
        """Find the best matching track from candidates or by searching

        First checks the cache for a previously matched track. If not found,
        searches for candidates and scores them by similarity.

        Args:
            candidates: Optional list of tracks to match against. If None, will search.
            match_threshold: Minimum similarity score required to match

        Returns:
            Tuple of (best_match, score) or (None, 0) if no match found
        """
        # Check cache first if available
        if self.source_player.cache_manager:
            cached_id = self.source_player.cache_manager.get_match(self.source.ID, source_name=self.source_player.name(), dest_name=self.destination_player.name())
            if cached_id:
                candidates = self.destination_player.search_tracks(key="id", value=cached_id)
                if candidates:
                    self.stats_manager.increment("cache_hits")
                    return candidates[0], 100  # Cache hit implies perfect match

        # Search for candidates if not provided
        if candidates is None:
            try:
                candidates = self.destination_player.search_tracks(key="title", value=self.source.title)
            except ValueError as e:
                self.logger.error(f"Failed to search tracks for track '{self.source}' stored at {self.source.file_path}.")
                raise e

        if not candidates:
            self.stats_manager.increment("no_matches")
            self.logger.warning(f"No candidates found for {self.source}")
            return None, 0

        # Calculate similarity scores
        scores = np.array([self.similarity(candidate) for candidate in candidates])
        ranks = scores.argsort()
        best_score = scores[ranks[-1]]

        # Track match quality
        if best_score < match_threshold:
            self.stats_manager.increment("no_matches")
            self.logger.debug(f"Best candidate score too low: {best_score} < {match_threshold}")
            return None, best_score
        elif best_score == 100:
            self.stats_manager.increment("perfect_matches")
        elif best_score >= 80:
            self.stats_manager.increment("good_matches")
        else:
            self.stats_manager.increment("poor_matches")

        best_match = candidates[ranks[-1]]

        # Cache successful match if available
        if self.source_player.cache_manager:
            self.source_player.cache_manager.set_match(self.source.ID, best_match.ID, self.source_player.name(), self.destination_player.name())

        self.logger.debug(f"Found match with score {best_score} for {self.source} - : - {best_match}")
        if best_score != 100:
            self.logger.info(f"Found match with score {best_score} for {self.source}: {best_match}")
        if best_score < 80:
            self.logger.debug(f"Source: {self.source}")
            self.logger.debug(f"Best Match: {best_match}")

        return best_match, best_score

    def match(self, candidates=None, match_threshold=30) -> int:
        """Find matching track on destination player

        :param candidates: Optional list of tracks to match against
        :param match_threshold: Minimum similarity score required to match
        :return: Match score or 0 if no match found
        """
        if self.source is None:
            raise RuntimeError("Source track not set")

        # Find best matching track
        best_match, score = self.find_best_match(candidates, match_threshold)
        if not best_match:
            self.sync_state = SyncState.ERROR
            return score

        self.destination = best_match
        self.stats_manager.increment("tracks_matched")

        # Compare ratings
        self.rating_source = self.source.rating
        self.rating_destination = self.destination.rating

        # Set sync state based on ratings
        if self.rating_source == self.rating_destination:
            self.sync_state = SyncState.UP_TO_DATE
        elif self.rating_source == 0.0 or self.rating_destination == 0.0:
            self.sync_state = SyncState.NEEDS_UPDATE
        elif self.rating_source != self.rating_destination:
            self.sync_state = SyncState.CONFLICTING
            self.logger.warning(f"Found match with conflicting ratings: {self.source} (Source: {self.rating_source} | Destination: {self.rating_destination})")

        return score

    def resolve_conflict(self) -> bool:
        prompt = {
            "1": f"{self.source_player.name()}: ({self.source}) - Rating: {self.rating_source}",
            "2": f"{self.destination_player.name()}: ({self.destination}) - Rating: {self.rating_destination}",
            "3": "New rating",
            "4": "Skip",
            "5": "Cancel resolving conflicts",
        }
        choose = True
        while choose:
            choose = False
            for key in prompt:
                print(f"\t[{key}]: {prompt[key]}")

            choice = input("Select how to resolve conflicting rating: ")
            if choice == "1":
                # apply source rating to destination
                self.destination_player.update_rating(self.destination, self.rating_source)
                return True
            elif choice == "2":
                # apply destination rating to source
                self.source_player.update_rating(self.source, self.rating_destination)
                return True
            elif choice == "3":
                # apply new rating to source and destination
                new_rating = input("Please enter a rating between 0 and 10: ")
                try:
                    new_rating = int(new_rating) / 10
                    if new_rating < 0:
                        raise Exception("Ratings below 0 not allowed")
                    elif new_rating > 1:
                        raise Exception("Ratings above 10 not allowed")
                    self.destination_player.update_rating(self.destination, new_rating)
                    self.source_player.update_rating(self.source, new_rating)
                    return True
                except Exception as e:
                    print("Error:", e)
                    print(f"Rating {new_rating} is not a valid rating, please choose an integer between 0 and 10")
                    choose = True
            elif choice == "4":
                return True
            elif choice == "5":
                return False
            else:
                print(f"{choice} is not a valid choice, please try again.")
                choose = True

        print(f"you chose {choice} which is {prompt[choice]}")

    def similarity(self, candidate) -> float:
        """
        Determines the matching similarity of @candidate with the source query track
        :type candidate: Track
        :returns a similarity rating [0.0, 100.0]
        :rtype: float
        """
        scores = np.array(
            [
                fuzz.ratio(self.source.title, candidate.title),
                fuzz.ratio(self.source.artist, candidate.artist),
                100.0 if self.source.track == candidate.track else 0.0,
                self.albums_similarity(destination=candidate),
            ]
        )
        return np.average(scores)

    def sync(self, force: bool = False) -> bool:
        if self.rating_destination <= 0.0 or force:
            # Propagate the rating of the source track to the destination track
            self.destination_player.update_rating(self.destination, self.rating_source)
            self.stats_manager.increment("tracks_updated")
            return True
        return False


class PlaylistPair(SyncPair):
    source: Playlist
    destination: Optional[Playlist] = None

    def __init__(self, source_player, destination_player, source_playlist: Playlist) -> None:
        """
        Initialize playlist pair with consistent naming
        :type source_player: MediaPlayer.MediaPlayer
        :type destination_player: MediaPlayer.MediaPlayer
        :type source_playlist: Playlist
        """
        super(PlaylistPair, self).__init__(source_player, destination_player)
        self.logger = logging.getLogger("PlexSync.PlaylistPair")
        self.source = source_playlist
        self.stats_manager = source_player.stats_manager

    def match(self) -> bool:
        """
        Find matching playlist on destination player based on name
        Sets sync state based on match results
        """
        self.destination = self.destination_player.find_playlist(title=self.source.name)

        if self.destination is None:
            self.sync_state = SyncState.NEEDS_UPDATE
            self.logger.info(f"Playlist {self.source.name} needs to be created")
            return False

        # Compare tracks
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
        # Playlist conflicts not implemented yet
        raise NotImplementedError

    def similarity(self, candidate) -> float:
        # Direct name matching used instead
        raise NotImplementedError

    def sync(self, force: bool = False) -> bool:
        """
        Non-destructive one-way sync from source to destination.
        Creates destination playlist if needed, adds any missing tracks.
        :return: True if sync successful
        :rtype: bool
        """
        if self.sync_state is SyncState.UP_TO_DATE:
            return True

        self.logger.info(f"Synchronizing playlist {self.source.name}")
        self.logger.debug(f"Source playlist has {len(self.source.tracks)} tracks")

        # Match all tracks from source
        track_pairs = []
        unmatched = []
        for track in self.source.tracks:
            pair = TrackPair(self.source_player, self.destination_player, track)
            pair.match()
            if pair.destination is not None:
                track_pairs.append(pair)
            else:
                unmatched.append(track)

        if not track_pairs:
            self.logger.warning(f"No tracks could be matched for playlist {self.source.name}")
            return False

        self.logger.info(f"Matched {len(track_pairs)}/{len(self.source.tracks)} tracks for playlist {self.source.name}")
        if unmatched:
            self.logger.warning(f"Failed to match {len(unmatched)} tracks:")
            for track in unmatched:
                self.logger.warning(f"  - {track}")

        # Create new playlist or update existing
        if self.destination is None:
            # Create new playlist with all tracks
            destination_tracks = [pair.destination for pair in track_pairs]
            self.destination = self.destination_player.create_playlist(self.source.name, destination_tracks)
            self.logger.info(f"Created new playlist {self.source.name} with {len(destination_tracks)} tracks")
            self.stats_manager.increment("playlists_updated")
            return True
        else:
            # Get current state
            orig_track_count = len(self.destination.tracks)
            self.logger.debug(f"Destination playlist has {orig_track_count} tracks")

            # Create list of updates - just additions for now since we're doing non-destructive sync
            updates = []
            for pair in track_pairs:
                if not self.destination.has_track(pair.destination):
                    updates.append((pair.destination, True))

            if updates:
                self.logger.debug(f"Adding {len(updates)} missing tracks to playlist {self.source.name}")
                self.destination_player.sync_playlist(self.destination, updates)
                self.stats_manager.increment("playlists_updated")
                return True
            else:
                self.logger.debug(f"Playlist {self.source.name} is up to date")
                return False
