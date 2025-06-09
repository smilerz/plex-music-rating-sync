import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ratings import Rating, RatingScale
from sync_items import AudioTag


class SQLQueryProcessor:
    """Basic SQL query parsing and filtering logic for MediaMonkey track queries."""

    def process_query(self, sql_query, tracks):

        # Simulate title queries with escaping
        if "SongTitle" in sql_query:
            # Handle escaped quotes in MediaMonkey format: "" becomes "
            # Pattern matches: SongTitle = "content with possible ""escaped"" quotes"
            m = re.search(r'SongTitle = "(.+)"$', sql_query)
            if m:
                title = m.group(1).replace('""', '"')
                return [t for t in tracks if getattr(t, "Title", None) == title]

        # Simulate ID queries (handle both "ID=" and "ID = " formats)
        if "ID" in sql_query and "=" in sql_query:
            m = re.search(r"ID\s*=\s*(\d+)", sql_query)
            if m:
                tid = int(m.group(1))
                return [t for t in tracks if getattr(t, "ID", None) == tid]

        # Simulate rating queries with comparison operators
        if "Rating" in sql_query:
            # Handle Rating > 0, Rating >= 50, Rating = 75, Rating < 100, etc.
            rating_patterns = [
                (r"Rating\s*>\s*(\d+(?:\.\d+)?)", lambda track_rating, value: track_rating > float(value)),
                (r"Rating\s*>=\s*(\d+(?:\.\d+)?)", lambda track_rating, value: track_rating >= float(value)),
                (r"Rating\s*=\s*(\d+(?:\.\d+)?)", lambda track_rating, value: track_rating == float(value)),
                (r"Rating\s*<=\s*(\d+(?:\.\d+)?)", lambda track_rating, value: track_rating <= float(value)),
                (r"Rating\s*<\s*(\d+(?:\.\d+)?)", lambda track_rating, value: track_rating < float(value)),
                (r"Rating\s*!=\s*(\d+(?:\.\d+)?)", lambda track_rating, value: track_rating != float(value)),
            ]

            for pattern, comparator in rating_patterns:
                m = re.search(pattern, sql_query)
                if m:
                    threshold = m.group(1)
                    return [t for t in tracks if hasattr(t, "Rating") and comparator(getattr(t, "Rating", 0), threshold)]

        return []


@pytest.fixture
def mm_track_factory():
    """Factory for MediaMonkey track objects with ID, Title, Rating, Artist.Name, Album.Name."""

    def _factory(ID=1, Title="Track", Rating=0, ArtistName="Artist", AlbumName="Album", Path="/mock/path.mp3", TrackOrder=1, SongLength=180000):
        track = SimpleNamespace()
        track.ID = ID
        track.Title = Title
        track.Rating = Rating
        track.Artist = SimpleNamespace(Name=ArtistName)
        track.Album = SimpleNamespace(Name=AlbumName)
        track.Path = Path
        track.TrackOrder = TrackOrder
        track.SongLength = SongLength  # in milliseconds
        return track

    return _factory


@pytest.fixture
def mm_playlist_factory():
    """Factory for MediaMonkey playlist objects with ID, Title, and required MediaMonkey attributes."""

    def _factory(ID=1, Title="Playlist", isAutoplaylist=False, tracks=None, children=None):
        class TracksCollection:
            """Mock MediaMonkey Tracks collection with Count property and list-like behavior."""

            def __init__(self, tracks_list=None):
                self._tracks = tracks_list or []

            @property
            def Count(self):
                return len(self._tracks)

            def __getitem__(self, index):
                return self._tracks[index]

            def __len__(self):
                return len(self._tracks)

            def append(self, track):
                self._tracks.append(track)

            def remove(self, track):
                self._tracks.remove(track)

        playlist = SimpleNamespace()
        playlist.ID = ID
        playlist.Title = Title
        playlist.isAutoplaylist = isAutoplaylist
        playlist.Tracks = TracksCollection(tracks)
        playlist.ChildPlaylists = children or []
        playlist.AddTrack = lambda track: playlist.Tracks.append(track)
        playlist.RemoveTrack = lambda track: playlist.Tracks.remove(track)
        playlist.CreateChildPlaylist = lambda title: _factory(ID=ID * 10 + len(playlist.ChildPlaylists) + 1, Title=title)
        return playlist

    return _factory


# --- Validation Test for Factories ---


@pytest.fixture
def mm_api():
    """
    Pythonic, maintainable SDB/COM API mock for MediaMonkey player integration tests.
    - Uses SimpleNamespace for tracks/playlists, Python lists for collections.
    - Only QueryResultIterator simulates COM-style iteration for QuerySongs.
    - All test-driven setup is via set_tracks/set_playlists.
    - Side effects can be tracked by adding attributes to objects as needed.
    """

    class QueryResultIterator:
        """Simulates MediaMonkey database result iterator with EOF, Item, Next."""

        def __init__(self, items):
            self._items = list(items)
            self._index = 0
            self.EOF = len(self._items) == 0
            self.Item = self._items[self._index] if self._items else None

        def Next(self):
            if self._index + 1 < len(self._items):
                self._index += 1
                self.Item = self._items[self._index]
                self.EOF = False
            else:
                self._index = len(self._items)
                self.Item = None
                self.EOF = True

    class DatabaseMock:
        def __init__(self, sdb):
            self._sdb = sdb
            # Use MagicMock with side_effect for mockable behavior
            self.QuerySongs = MagicMock(side_effect=self._query_songs)

        def _query_songs(self, sql_query):
            """Default implementation for QuerySongs - can be overridden by tests."""
            if self._sdb.QuerySongs_raise:
                raise self._sdb.QuerySongs_raise
            if self._sdb.QuerySongs_return is not None:
                return self._sdb.QuerySongs_return
            return QueryResultIterator(SQLQueryProcessor().process_query(sql_query, self._sdb._tracks))

    class SDBMock:
        def __init__(self):
            self.QuerySongs_raise = None
            self.QuerySongs_return = None
            self._tracks = []
            self._playlists = []
            self.Database = DatabaseMock(self)
            self._playlist_by_id = {}
            self._playlist_by_title = {}
            self._root_playlist = self._make_playlist(ID=0, Title="", children=[])

            # Use MagicMock with side_effect for mockable behavior
            self.PlaylistByTitle = MagicMock(side_effect=self._playlist_by_title_impl)
            self.PlaylistByID = MagicMock(side_effect=self._playlist_by_id_impl)

        def set_tracks(self, tracks):
            self._tracks = tracks

        def set_playlists(self, playlists):
            self._playlists = playlists
            self._playlist_by_id = {pl.ID: pl for pl in playlists}
            self._playlist_by_title = {pl.Title.lower(): pl for pl in playlists}
            self._root_playlist.ChildPlaylists = playlists

        def _playlist_by_title_impl(self, title):
            """Default implementation for PlaylistByTitle - can be overridden by tests."""
            if title == "":
                return self._root_playlist
            return self._playlist_by_title.get(title.lower())

        def _playlist_by_id_impl(self, id):
            """Default implementation for PlaylistByID - can be overridden by tests."""
            return self._playlist_by_id.get(int(id))

        def _make_playlist(self, ID=1, Title="Playlist", isAutoplaylist=False, tracks=None, children=None):
            class TracksCollection:
                """Mock MediaMonkey Tracks collection with Count property and list-like behavior."""

                def __init__(self, tracks_list=None):
                    self._tracks = tracks_list or []

                @property
                def Count(self):
                    return len(self._tracks)

                def __getitem__(self, index):
                    return self._tracks[index]

                def __len__(self):
                    return len(self._tracks)

                def append(self, track):
                    self._tracks.append(track)

                def remove(self, track):
                    self._tracks.remove(track)

            pl = SimpleNamespace()
            pl.ID = ID
            pl.Title = Title
            pl.isAutoplaylist = isAutoplaylist
            pl.Tracks = TracksCollection(tracks)
            pl.ChildPlaylists = children or []
            pl.AddTrack = lambda track: pl.Tracks.append(track)
            pl.RemoveTrack = lambda track: pl.Tracks.remove(track)
            pl.CreateChildPlaylist = lambda title: self._make_playlist(ID=ID * 10 + len(pl.ChildPlaylists) + 1, Title=title)
            return pl

    # Factory-driven setup
    sdb = SDBMock()
    # Initialize error injection attributes (like plex_api pattern)
    sdb.connect_raise = None
    # Allow test to inject tracks/playlists after fixture creation
    sdb.set_tracks([])
    sdb.set_playlists([])
    return sdb


@pytest.fixture
def mm_player(monkeypatch, request, mm_api):
    """
    MediaMonkey player fixture that wires up the real MediaMonkey instance with:
    - The mm_api fixture for all SDB/COM API mocking and test-driven scenario setup.
    - Manager, config, and system dependency mocks.
    - Patches win32com.client.Dispatch to return mm_api, so connect() does not launch the real app.
    - All scenario logic (tracks, playlists, error simulation) is handled in mm_api, not here.
    Mirrors the plex_player fixture pattern for maintainability and realism.
    """

    # --- Patch win32com.client.Dispatch with error injection support ---
    def dispatch_side_effect(*args, **kwargs):
        # Check for connection-level error injection (like plex_api.connect_raise)
        exc = getattr(mm_api, "connect_raise", None)
        if exc is not None:
            raise exc
        return mm_api

    monkeypatch.setattr("win32com.client.Dispatch", dispatch_side_effect)

    # --- Manager and config mocks ---
    fake_manager = MagicMock()
    config_defaults = {
        "server": "mock_server",
        "username": "mock_user",
        "passwd": "mock_password",
        "token": None,
    }
    config_overrides = getattr(request, "param", {}).get("config", {})
    config = MagicMock(**{**config_defaults, **config_overrides})
    fake_manager.get_config_manager.return_value = config
    fake_manager.get_cache_manager.return_value = MagicMock()
    fake_manager.get_status_manager.return_value = MagicMock()
    monkeypatch.setattr("manager.get_manager", lambda: fake_manager)

    # --- Import after patching ---
    from MediaPlayer import MediaMonkey

    player = MediaMonkey()
    player.config_mgr = config
    player.logger = MagicMock()
    player.status_mgr = MagicMock()
    player.cache_mgr = MagicMock()
    # Configure cache manager to return None for cache misses (realistic behavior)
    player.cache_mgr.get_metadata.return_value = None
    player.cache_mgr.set_metadata.return_value = None
    player.sdb = mm_api
    # Allow test to override config attributes
    for k, v in config_overrides.items():
        setattr(player.config_mgr, k, v)
    return player


class TestConnect:
    """Tests for MediaMonkey.connect behavior and error handling."""

    def test_connect_success(self, mm_player):
        """Test that MediaMonkey.connect logs info when successful."""
        mm_player.connect()
        mm_player.logger.info.assert_called()

    def test_connect_raises(self, mm_player, mm_api):
        """Test that MediaMonkey.connect raises an exception on connection failure and logs error."""
        # Set the raise condition at the connection level (not QuerySongs level)
        mm_api.connect_raise = RuntimeError("Connection failed")
        with pytest.raises(RuntimeError, match="Connection failed"):
            mm_player.connect()
        mm_player.logger.error.assert_called()


class TestPlaylistSearch:
    """Tests for MediaMonkey playlist search and retrieval."""

    @pytest.mark.parametrize(
        "key,value,return_native,setup_playlists,expect_error,expect_empty",
        [
            ("all", None, False, [{"ID": 1, "Title": "My Playlist"}, {"ID": 2, "Title": "Another Playlist"}], False, False),
            ("title", "My Playlist", False, [{"ID": 1, "Title": "My Playlist"}], False, False),
            ("id", 1, False, [{"ID": 1, "Title": "My Playlist"}], False, False),
            ("badkey", None, False, [], True, False),
            ("id", 9999, False, [{"ID": 1, "Title": "My Playlist"}], False, True),
            ("title", "Nonexistent", False, [{"ID": 1, "Title": "My Playlist"}], False, True),
        ],
        ids=[
            "search_all_returns_multiple_playlists",
            "search_by_title_finds_matching_playlist",
            "search_by_id_finds_matching_playlist",
            "search_invalid_key_raises_error",
            "search_nonexistent_id_returns_empty",
            "search_nonexistent_title_returns_empty",
        ],
    )
    def test_search_parametrized_queries(self, mm_player, mm_playlist_factory, key, value, return_native, setup_playlists, expect_error, expect_empty):
        """Test playlist search handles different search parameters correctly."""
        # Set up test playlists
        playlists = [mm_playlist_factory(**pl_data) for pl_data in setup_playlists]
        mm_player.sdb.set_playlists(playlists)

        if expect_error:
            with pytest.raises(ValueError, match="Invalid search key"):
                mm_player.search_playlists(key, value, return_native)
        else:
            result = mm_player.search_playlists(key, value, return_native)
            assert isinstance(result, list)

            if expect_empty:
                assert len(result) == 0
            else:
                assert len(result) > 0
                if key == "all":
                    assert len(result) == len(setup_playlists)
                elif key in ["title", "id"]:
                    assert len(result) == 1
                    if return_native:
                        # Native playlist should have MediaMonkey-specific attributes
                        assert hasattr(result[0], "ID")
                        assert hasattr(result[0], "Title")
                    else:
                        # Converted playlist should be Playlist object
                        assert hasattr(result[0], "name")
                        assert hasattr(result[0], "ID")

    @pytest.mark.parametrize(
        "return_native,playlist_id,title,expected_attributes",
        [
            (True, 1003, "Test Playlist", ["ID", "Title"]),
            (False, 1004, "Converted Playlist", ["name", "ID"]),
        ],
        ids=["native_return", "converted_return"],
    )
    def test_search_return_native_parameter(self, mm_player, mm_playlist_factory, return_native, playlist_id, title, expected_attributes):
        """Test searching playlists with return_native parameter returns appropriate object types."""
        playlist_data = {"ID": playlist_id, "Title": title}
        native_playlist = mm_playlist_factory(**playlist_data)
        mm_player.sdb.set_playlists([native_playlist])

        result = mm_player.search_playlists("id", playlist_id, return_native=return_native)

        assert isinstance(result, list)
        assert len(result) == 1

        playlist = result[0]
        for attr in expected_attributes:
            assert hasattr(playlist, attr)

        if return_native:
            # Native MediaMonkey objects should have original SimpleNamespace structure
            assert getattr(playlist, "ID", None) == playlist_id
            assert getattr(playlist, "Title", None) == title
        else:
            # Converted objects should be Playlist instances
            assert playlist.ID == playlist_id
            assert playlist.name == title

    def test_search_title_case_insensitive(self, mm_player, mm_playlist_factory):
        """Test that title search is case-insensitive."""
        playlist_data = {"ID": 1005, "Title": "CaseSensitive Playlist"}
        native_playlist = mm_playlist_factory(**playlist_data)
        mm_player.sdb.set_playlists([native_playlist])

        # Test various case combinations
        for search_title in ["casesensitive playlist", "CASESENSITIVE PLAYLIST", "CaseSensitive Playlist"]:
            result = mm_player.search_playlists("title", search_title)
            assert len(result) == 1
            assert result[0].name == "CaseSensitive Playlist"

    def test_search_nested_hierarchy(self, mm_player, mm_playlist_factory):
        """Test searching for nested playlists with dot notation."""
        playlists = [
            mm_playlist_factory(ID=1, Title="Parent"),
            mm_playlist_factory(ID=2, Title="Parent.Child"),
            mm_playlist_factory(ID=3, Title="Parent.Child.Grandchild"),
        ]
        mm_player.sdb.set_playlists(playlists)

        # Search for nested playlist
        result = mm_player.search_playlists("title", "Parent.Child")
        assert len(result) == 1
        assert result[0].name == "Parent.Child"
        assert result[0].ID == 2

    def test_search_all_preserves_order(self, mm_player, mm_playlist_factory):
        """Test that search all returns playlists in the order they were set."""
        playlists = [
            mm_playlist_factory(ID=3, Title="Third"),
            mm_playlist_factory(ID=1, Title="First"),
            mm_playlist_factory(ID=2, Title="Second"),
        ]
        mm_player.sdb.set_playlists(playlists)

        result = mm_player.search_playlists("all")
        assert len(result) == 3
        # Should preserve the order from set_playlists
        expected_order = ["Third", "First", "Second"]
        actual_order = [pl.name for pl in result]
        assert actual_order == expected_order

    def test_search_empty_returns_empty(self, mm_player):
        """Test searching playlists when no playlists exist returns empty list."""
        mm_player.sdb.set_playlists([])

        result = mm_player.search_playlists("all")
        assert isinstance(result, list)
        assert len(result) == 0


class TestPlaylistCreation:
    """Tests for MediaMonkey playlist creation and conversion."""

    def test_create_empty_title_raises(self, mm_player, mm_track_factory):
        """Test that _create_playlist raises ValueError when title is empty."""
        tracks = [mm_track_factory(ID=1, Title="Test Track")]

        with pytest.raises(ValueError, match="Title and tracks cannot be empty"):
            mm_player._create_playlist("", tracks)

    def test_create_empty_tracks_raises(self, mm_player):
        """Test that _create_playlist raises ValueError when tracks list is empty."""
        with pytest.raises(ValueError, match="Title and tracks cannot be empty"):
            mm_player._create_playlist("Test Playlist", [])

    def test_create_simple_success(self, mm_player, mm_track_factory):
        """Test that _create_playlist successfully creates a simple playlist with tracks."""
        # Set up tracks in the database
        track1 = mm_track_factory(ID=1, Title="Track One")
        track2 = mm_track_factory(ID=2, Title="Track Two")
        mm_player.sdb.set_tracks([track1, track2])

        # Create AudioTag objects for the tracks
        from ratings import Rating, RatingScale
        from sync_items import AudioTag

        audio_tracks = [
            AudioTag(ID="1", title="Track One", artist="Artist", album="Album", track=1, rating=Rating(0.8, RatingScale.NORMALIZED)),
            AudioTag(ID="2", title="Track Two", artist="Artist", album="Album", track=2, rating=Rating(0.6, RatingScale.NORMALIZED)),
        ]

        # Mock status manager for progress bar
        mm_player.status_mgr.start_phase.return_value = MagicMock()

        # Call method under test
        result = mm_player._create_playlist("Simple Playlist", audio_tracks)

        # Verify playlist was created and returned
        assert result is not None
        assert hasattr(result, "Title")
        assert result.Title == "Simple Playlist"

        # Verify progress bar was used
        mm_player.status_mgr.start_phase.assert_called_once_with("Adding tracks to playlist Simple Playlist", total=2)

    def test_create_nested_with_dots(self, mm_player, mm_track_factory):
        """Test that _create_playlist creates nested playlists using dot notation."""
        # Set up tracks in the database
        track1 = mm_track_factory(ID=1, Title="Nested Track")
        mm_player.sdb.set_tracks([track1])

        # Create AudioTag
        from ratings import Rating, RatingScale
        from sync_items import AudioTag

        audio_tracks = [AudioTag(ID="1", title="Nested Track", artist="Artist", album="Album", track=1, rating=Rating(0.5, RatingScale.NORMALIZED))]

        # Mock status manager
        mm_player.status_mgr.start_phase.return_value = MagicMock()

        # Call method under test with nested playlist name
        result = mm_player._create_playlist("Parent.Child.Grandchild", audio_tracks)

        # Verify nested playlist was created
        assert result is not None
        assert hasattr(result, "Title")
        assert result.Title == "Grandchild"

    def test_create_existing_reused(self, mm_player, mm_track_factory):
        """Test that _create_playlist reuses existing playlists instead of creating duplicates."""
        # Set up tracks
        track1 = mm_track_factory(ID=1, Title="Existing Track")
        mm_player.sdb.set_tracks([track1])

        # Pre-create a playlist using the mock infrastructure
        existing_playlist = mm_player.sdb._make_playlist(ID=100, Title="Existing Playlist")
        mm_player.sdb.set_playlists([existing_playlist])

        # Create AudioTag
        from ratings import Rating, RatingScale
        from sync_items import AudioTag

        audio_tracks = [AudioTag(ID="1", title="Existing Track", artist="Artist", album="Album", track=1, rating=Rating(0.7, RatingScale.NORMALIZED))]

        # Mock status manager
        mm_player.status_mgr.start_phase.return_value = MagicMock()

        # Call method under test with existing playlist name
        result = mm_player._create_playlist("Existing Playlist", audio_tracks)

        # Verify existing playlist was reused
        assert result is not None
        assert result.ID == 100  # Should be the pre-existing playlist

    def test_create_track_not_found_logs(self, mm_player):
        """Test that _create_playlist logs warning when tracks are not found in MediaMonkey database."""
        # Don't set any tracks in database - simulate missing tracks
        mm_player.sdb.set_tracks([])

        # Create AudioTag for non-existent track
        from ratings import Rating, RatingScale
        from sync_items import AudioTag

        audio_tracks = [AudioTag(ID="999", title="Missing Track", artist="Artist", album="Album", track=1, rating=Rating(0.5, RatingScale.NORMALIZED))]

        # Mock status manager and QuerySongs to return empty result
        mm_player.status_mgr.start_phase.return_value = MagicMock()

        # Mock QuerySongs to return None/empty for missing track
        empty_result = MagicMock()
        empty_result.Item = None
        mm_player.sdb.Database.QuerySongs.return_value = empty_result

        # Call method under test
        mm_player._create_playlist("Test Playlist", audio_tracks)

        # Verify warning was logged for missing track
        mm_player.logger.warning.assert_called_with("Track with ID 999 not found in MediaMonkey database")

    def test_create_track_error_continues(self, mm_player, mm_track_factory):
        """Test that _create_playlist handles track addition errors gracefully."""
        # Set up track in database
        track1 = mm_track_factory(ID=1, Title="Problematic Track")
        mm_player.sdb.set_tracks([track1])

        # Create AudioTag
        from ratings import Rating, RatingScale
        from sync_items import AudioTag

        audio_tracks = [AudioTag(ID="1", title="Problematic Track", artist="Artist", album="Album", track=1, rating=Rating(0.5, RatingScale.NORMALIZED))]

        # Mock status manager
        mm_player.status_mgr.start_phase.return_value = MagicMock()

        # Create a mock playlist that will cause AddTrack to fail
        error_playlist = mm_player.sdb._make_playlist(ID=1, Title="Error Playlist")
        error_playlist.AddTrack = MagicMock(side_effect=RuntimeError("AddTrack failed"))

        # Set up the playlist in the mock so search_playlists can find it
        mm_player.sdb.set_playlists([error_playlist])

        # Override the QuerySongs behavior to return a valid track
        # We need to override the QuerySongs_return attribute used by the side_effect
        mm_player.sdb.QuerySongs_return = MagicMock()
        mm_player.sdb.QuerySongs_return.Item = track1

        # Call method under test
        result = mm_player._create_playlist("Error Playlist", audio_tracks)

        # Verify error was logged but playlist creation continued
        mm_player.logger.error.assert_called_with("Failed to add track ID 1 to playlist: AddTrack failed")
        assert result is not None

    @pytest.mark.parametrize(
        "track_count,expect_progress_bar",
        [
            (1, True),  # Even small playlists get progress bars in _create_playlist
            (150, True),
        ],
        ids=["single_track", "large_playlist"],
    )
    def test_create_uses_progress_bar(self, mm_player, mm_track_factory, track_count, expect_progress_bar):
        """Test that _create_playlist uses progress bar for track addition regardless of playlist size."""
        # Set up tracks in database
        tracks = [mm_track_factory(ID=i, Title=f"Track {i}") for i in range(1, track_count + 1)]
        mm_player.sdb.set_tracks(tracks)

        # Create AudioTag objects
        from ratings import Rating, RatingScale
        from sync_items import AudioTag

        audio_tracks = [
            AudioTag(ID=str(i), title=f"Track {i}", artist="Artist", album="Album", track=i, rating=Rating(0.5, RatingScale.NORMALIZED)) for i in range(1, track_count + 1)
        ]

        # Mock status manager
        progress_bar_mock = MagicMock()
        mm_player.status_mgr.start_phase.return_value = progress_bar_mock

        # Call method under test
        result = mm_player._create_playlist("Progress Test", audio_tracks)

        # Verify progress bar behavior
        if expect_progress_bar:
            mm_player.status_mgr.start_phase.assert_called_once_with("Adding tracks to playlist Progress Test", total=track_count)
            assert progress_bar_mock.update.call_count == track_count
            progress_bar_mock.close.assert_called_once()

        assert result is not None

    def test_create_integration_with_search(self, mm_player, mm_track_factory):
        """Test integration between _create_playlist, track search, and playlist conversion."""
        # Set up tracks in database
        track1 = mm_track_factory(ID=1, Title="Integration Track 1", ArtistName="Artist A")
        track2 = mm_track_factory(ID=2, Title="Integration Track 2", ArtistName="Artist B")
        mm_player.sdb.set_tracks([track1, track2])

        # Create AudioTag objects
        from ratings import Rating, RatingScale
        from sync_items import AudioTag

        audio_tracks = [
            AudioTag(ID="1", title="Integration Track 1", artist="Artist A", album="Album", track=1, rating=Rating(0.8, RatingScale.NORMALIZED)),
            AudioTag(ID="2", title="Integration Track 2", artist="Artist B", album="Album", track=2, rating=Rating(0.6, RatingScale.NORMALIZED)),
        ]

        # Mock status manager
        mm_player.status_mgr.start_phase.return_value = MagicMock()

        # Call method under test
        result = mm_player._create_playlist("Integration Test", audio_tracks)

        # Verify playlist creation and track addition
        assert result is not None
        assert result.Title == "Integration Test"

        # Verify tracks were found and added (2 QuerySongs calls for 2 tracks)
        assert mm_player.sdb.Database.QuerySongs.call_count == 2

        # Verify all tracks were added to playlist by checking the playlist's track count
        # Since AddTrack is a lambda that appends to Tracks, check the actual tracks
        assert len(result.Tracks) == 2


class TestPlaylistTracks:
    """Tests for MediaMonkey playlist track reading and counting."""

    def test_read_returns_converted_tracks(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test that _read_native_playlist_tracks converts all playlist tracks to AudioTag objects."""
        # Create mock tracks with MediaMonkey-specific attributes
        track1 = mm_track_factory(ID=1, Title="Track One", ArtistName="Artist A", Rating=80)
        track2 = mm_track_factory(ID=2, Title="Track Two", ArtistName="Artist B", Rating=60)
        track3 = mm_track_factory(ID=3, Title="Track Three", ArtistName="Artist C", Rating=100)

        # Create playlist with tracks
        playlist = mm_playlist_factory(Title="Test Playlist", tracks=[track1, track2, track3])

        # Call method under test
        result = mm_player._read_native_playlist_tracks(playlist)

        # Verify return type and count
        assert isinstance(result, list)
        assert len(result) == 3

        # Verify all items are AudioTag objects
        for track in result:
            assert isinstance(track, AudioTag)

        # Verify track conversion correctness
        assert result[0].title == "Track One"
        assert result[0].artist == "Artist A"
        assert result[0].rating.to_float(RatingScale.ZERO_TO_FIVE) == 4  # MediaMonkey 80 -> 5-star scale 4

        assert result[1].title == "Track Two"
        assert result[1].artist == "Artist B"
        assert result[1].rating.to_float(RatingScale.ZERO_TO_FIVE) == 3  # MediaMonkey 60 -> 5-star scale 3

        assert result[2].title == "Track Three"
        assert result[2].artist == "Artist C"
        assert result[2].rating.to_float(RatingScale.ZERO_TO_FIVE) == 5  # MediaMonkey 100 -> 5-star scale 5

    def test_read_handles_empty_playlist(self, mm_player, mm_playlist_factory):
        """Test that _read_native_playlist_tracks handles empty playlists correctly."""
        # Create empty playlist
        playlist = mm_playlist_factory(Title="Empty Playlist")

        # Call method under test
        result = mm_player._read_native_playlist_tracks(playlist)

        # Verify empty list returned
        assert isinstance(result, list)
        assert len(result) == 0

    def test_read_iterates_by_count(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test that _read_native_playlist_tracks uses Count property for iteration."""
        # Create tracks but set Count differently than actual length
        track1 = mm_track_factory(ID=1, Title="Track One")
        track2 = mm_track_factory(ID=2, Title="Track Two")
        track3 = mm_track_factory(ID=3, Title="Track Three")

        playlist = mm_playlist_factory(Title="Test Playlist", tracks=[track1, track2])
        # Manually override tracks list to have 3 tracks but access only first 2
        playlist.Tracks._tracks = [track1, track2, track3]
        playlist.Tracks._tracks = playlist.Tracks._tracks[:2]  # Simulate Count = 2 behavior

        # Call method under test
        result = mm_player._read_native_playlist_tracks(playlist)

        # Verify only Count number of tracks processed
        assert len(result) == 2
        assert result[0].title == "Track One"
        assert result[1].title == "Track Two"

    def test_read_preserves_order(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test that _read_native_playlist_tracks preserves track order from playlist."""
        # Create tracks in specific order
        track_z = mm_track_factory(ID=3, Title="Z Track", ArtistName="Z Artist")
        track_a = mm_track_factory(ID=1, Title="A Track", ArtistName="A Artist")
        track_m = mm_track_factory(ID=2, Title="M Track", ArtistName="M Artist")

        playlist = mm_playlist_factory(Title="Ordered Playlist", tracks=[track_z, track_a, track_m])

        # Call method under test
        result = mm_player._read_native_playlist_tracks(playlist)

        # Verify order preservation
        assert len(result) == 3
        assert result[0].title == "Z Track"  # First in playlist order
        assert result[1].title == "A Track"  # Second in playlist order
        assert result[2].title == "M Track"  # Third in playlist order

    @pytest.mark.parametrize(
        "count,description",
        [
            (0, "empty playlists"),
            (99, "normal playlists"),
            (101, "large playlists"),
        ],
    )
    def test_read_track_count(self, mm_player, mm_playlist_factory, count, description):
        """Test that _get_native_playlist_track_count returns the Count property value for various playlist sizes."""
        # Create dummy tracks to match the count
        dummy_tracks = [SimpleNamespace(ID=i, Title=f"Track {i}") for i in range(count)]
        playlist = mm_playlist_factory(Title="Test Playlist", tracks=dummy_tracks)

        # Call method under test
        result = mm_player._get_native_playlist_track_count(playlist)

        # Verify correct count returned
        assert result == count
        assert isinstance(result, int)

    def test_read_methods_integration(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test integration between _get_native_playlist_track_count and _read_native_playlist_tracks."""
        # Create playlist with tracks
        track1 = mm_track_factory(ID=1, Title="Integration Track 1")
        track2 = mm_track_factory(ID=2, Title="Integration Track 2")
        track3 = mm_track_factory(ID=3, Title="Integration Track 3")

        playlist = mm_playlist_factory(Title="Integration Playlist", tracks=[track1, track2, track3])

        # Get count and read tracks
        count = mm_player._get_native_playlist_track_count(playlist)
        tracks = mm_player._read_native_playlist_tracks(playlist)

        # Verify count matches actual tracks read
        assert count == len(tracks)
        assert count == 3

        # Verify all tracks properly converted
        for track in tracks:
            assert isinstance(track, AudioTag)
            assert track.title.startswith("Integration Track")


class TestTrackSearch:
    """Tests for MediaMonkey track search by title, ID, and rating."""

    def test_search_by_id_returns_track(self, mm_player, mm_track_factory):
        """Test that MediaMonkey.search_tracks returns the correct track by ID using fixture-based dependency injection."""
        track1 = mm_track_factory(ID=1, Title="A")
        track2 = mm_track_factory(ID=2, Title="B")
        mm_player.sdb.set_tracks([track1, track2])
        results = mm_player.search_tracks("id", 2)
        assert isinstance(results, list)
        assert any(getattr(t, "ID", None) == 2 for t in results)
        assert all(getattr(t, "ID", None) == 2 for t in results)

    def test_search_query_raises_exception(self, mm_player, mm_track_factory):
        """Test that MediaMonkey.search_tracks raises if the underlying query raises, using fixture-based error simulation."""
        mm_player.sdb.QuerySongs_raise = RuntimeError("Query failed")
        with pytest.raises(RuntimeError, match="Query failed"):
            mm_player.search_tracks("id", 1)

    def test_search_by_title_returns_matching(self, mm_player, mm_track_factory):
        """Test that MediaMonkey.search_tracks returns tracks matching the exact title."""
        track1 = mm_track_factory(ID=1, Title="My Song")
        track2 = mm_track_factory(ID=2, Title="Another Song")
        track3 = mm_track_factory(ID=3, Title="My Song")  # Duplicate title
        mm_player.sdb.set_tracks([track1, track2, track3])

        results = mm_player.search_tracks("title", "My Song")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(getattr(t, "title", None) == "My Song" for t in results)

    def test_search_title_escapes_quotes(self, mm_player, mm_track_factory):
        """Test that MediaMonkey.search_tracks properly escapes quotes in title search."""
        track1 = mm_track_factory(ID=1, Title='Song with "quotes"')
        track2 = mm_track_factory(ID=2, Title="Regular Song")
        mm_player.sdb.set_tracks([track1, track2])

        results = mm_player.search_tracks("title", 'Song with "quotes"')
        assert isinstance(results, list)
        assert len(results) == 1
        assert getattr(results[0], "title", None) == 'Song with "quotes"'

    @pytest.mark.parametrize(
        "search_value,track_setup,expected_count,expected_ids",
        [
            # Test rating > 0 search
            (
                True,
                [
                    {"ID": 1, "Title": "Unrated", "Rating": 0},
                    {"ID": 2, "Title": "Rated Low", "Rating": 25},
                    {"ID": 3, "Title": "Rated High", "Rating": 85},
                ],
                2,
                {2, 3},
            ),
            # Test specific rating value search
            (
                "= 50",
                [
                    {"ID": 1, "Title": "Low Rating", "Rating": 25},
                    {"ID": 2, "Title": "Medium Rating", "Rating": 50},
                    {"ID": 3, "Title": "High Rating", "Rating": 75},
                    {"ID": 4, "Title": "Also Medium", "Rating": 50},
                ],
                2,
                {2, 4},
            ),
        ],
        ids=["rating_greater_than_zero", "specific_rating_value"],
    )
    def test_search_by_rating_criteria(self, mm_player, mm_track_factory, search_value, track_setup, expected_count, expected_ids):
        """Test that MediaMonkey.search_tracks returns correct tracks for various rating search criteria."""
        tracks = [mm_track_factory(**track_data) for track_data in track_setup]
        mm_player.sdb.set_tracks(tracks)

        results = mm_player.search_tracks("rating", search_value)

        assert isinstance(results, list)
        assert len(results) == expected_count
        matched_ids = {getattr(t, "ID", None) for t in results}
        assert matched_ids == expected_ids

    def test_search_invalid_key_raises(self, mm_player):
        """Test that MediaMonkey.search_tracks raises KeyError for invalid search key."""
        with pytest.raises(KeyError, match="Invalid search mode"):
            mm_player.search_tracks("invalid_key", "value")

    def test_search_empty_value_raises(self, mm_player):
        """Test that MediaMonkey.search_tracks raises ValueError for empty search value."""
        with pytest.raises(ValueError, match="value can not be empty"):
            mm_player.search_tracks("id", "")

    def test_search_return_native_bypasses_conversion(self, mm_player, mm_track_factory):
        """Test that MediaMonkey.search_tracks with return_native=True returns native track objects."""
        track1 = mm_track_factory(ID=1, Title="Test Track")
        mm_player.sdb.set_tracks([track1])

        results = mm_player.search_tracks("id", 1, return_native=True)
        assert isinstance(results, list)
        assert len(results) == 1
        # Native track should have the original SimpleNamespace structure
        assert hasattr(results[0], "ID")
        assert hasattr(results[0], "Title")
        assert getattr(results[0], "ID", None) == 1
        assert getattr(results[0], "Title", None) == "Test Track"


class TestTrackMetadata:
    """Tests for MediaMonkey track metadata reading and rating updates."""

    @pytest.mark.parametrize(
        "cache_scenario,track_data,expected_cache_calls",
        [
            # Cache hit scenario
            (
                "hit",
                {"ID": 123, "Title": "Cached Track", "Rating": 75},
                {"get_called": True, "set_called": False, "returns_cached": True},
            ),
            # Cache miss scenario
            (
                "miss",
                {
                    "ID": 456,
                    "Title": "New Track",
                    "Rating": 50,
                    "ArtistName": "Test Artist",
                    "AlbumName": "Test Album",
                    "Path": "/path/to/song.mp3",
                    "TrackOrder": 3,
                    "SongLength": 240000,
                },
                {"get_called": True, "set_called": True, "returns_cached": False},
            ),
        ],
        ids=["cache_hit", "cache_miss"],
    )
    def test_metadata_with_cache_hit_and_miss(self, mm_player, mm_track_factory, cache_scenario, track_data, expected_cache_calls):
        """Test that _read_track_metadata handles cache hits and misses correctly."""
        native_track = mm_track_factory(**track_data)

        if cache_scenario == "hit":
            # Configure cache hit - return cached AudioTag
            cached_tag = AudioTag(ID=str(track_data["ID"]), title=track_data["Title"], artist="Cached Artist", album="Cached Album", track=1)
            mm_player.cache_mgr.get_metadata.return_value = cached_tag
        else:
            # Configure cache miss
            mm_player.cache_mgr.get_metadata.return_value = None

        result = mm_player._read_track_metadata(native_track)

        # Verify cache method calls
        if expected_cache_calls["get_called"]:
            mm_player.cache_mgr.get_metadata.assert_called_once_with("MediaMonkey", track_data["ID"])

        if expected_cache_calls["set_called"]:
            mm_player.cache_mgr.set_metadata.assert_called_once_with("MediaMonkey", track_data["ID"], result)
        else:
            mm_player.cache_mgr.set_metadata.assert_not_called()

        # Verify return behavior
        if expected_cache_calls["returns_cached"]:
            assert result is mm_player.cache_mgr.get_metadata.return_value
        else:
            # Verify new AudioTag was created with correct data
            assert isinstance(result, AudioTag)
            assert result.ID == track_data["ID"]
            assert result.title == track_data["Title"]

    @pytest.mark.parametrize(
        "mm_rating,expected_normalized,is_unrated",
        [
            (85, 0.85, False),  # Valid rating
            (0, None, True),  # Unrated track
        ],
        ids=["valid_rating", "unrated_track"],
    )
    def test_metadata_rating_conversion(self, mm_player, mm_track_factory, mm_rating, expected_normalized, is_unrated):
        """Test that _read_track_metadata correctly converts MediaMonkey ratings to normalized scale."""
        native_track = mm_track_factory(ID=789, Rating=mm_rating)
        mm_player.cache_mgr.get_metadata.return_value = None

        result = mm_player._read_track_metadata(native_track)

        assert isinstance(result.rating, Rating)
        if is_unrated:
            assert result.rating.is_unrated
        else:
            assert result.rating.to_float(RatingScale.NORMALIZED) == expected_normalized
            assert not result.rating.is_unrated

    def test_metadata_missing_song_length_defaults(self, mm_player, mm_track_factory):
        """Test that _read_track_metadata handles missing SongLength gracefully."""
        native_track = mm_track_factory(ID=111, SongLength=None)
        mm_player.cache_mgr.get_metadata.return_value = None

        result = mm_player._read_track_metadata(native_track)

        assert result.duration == -1

    def test_metadata_update_rating_success(self, mm_player, mm_track_factory):
        """Test that _update_rating successfully updates rating and calls UpdateDB."""
        # Create AudioTag and native track
        audio_tag = AudioTag(ID="555", title="Update Track", artist="Artist", album="Album", track=1)
        native_track = mm_track_factory(ID=555, Title="Update Track")

        # Add UpdateDB method to native track
        native_track.UpdateDB = MagicMock()

        # Configure search to return the native track
        mm_player.sdb.set_tracks([native_track])

        new_rating = Rating(0.9, scale=RatingScale.NORMALIZED)  # Should convert to 90/100
        mm_player._update_rating(audio_tag, new_rating)

        # Verify rating was set correctly (converted to MediaMonkey scale)
        assert native_track.Rating == 90.0
        native_track.UpdateDB.assert_called_once()
        mm_player.logger.debug.assert_called_with("Updating rating for Artist - Album - Update Track to 4.5")

    @pytest.mark.parametrize(
        "error_scenario,track_id,track_title,setup_error,expected_exception,error_check",
        [
            # Track not found scenario
            (
                "track_not_found",
                "999",
                "Missing Track",
                "empty_tracks",
                IndexError,
                lambda error_msg: "Failed to update rating:" in error_msg,
            ),
            # UpdateDB failure scenario
            (
                "updatedb_failure",
                "777",
                "Failing Track",
                "updatedb_exception",
                RuntimeError,
                lambda error_msg: "Database update failed" in str(error_msg),
            ),
        ],
        ids=["track_not_found", "updatedb_failure"],
    )
    def test_metadata_update_rating_handles_errors(self, mm_player, mm_track_factory, error_scenario, track_id, track_title, setup_error, expected_exception, error_check):
        """Test that _update_rating handles various error scenarios correctly."""
        audio_tag = AudioTag(ID=track_id, title=track_title, artist="Artist", album="Album", track=1)
        new_rating = Rating(0.5, scale=RatingScale.NORMALIZED)

        if setup_error == "empty_tracks":
            # Configure search to return empty list (track not found)
            mm_player.sdb.set_tracks([])
        elif setup_error == "updatedb_exception":
            # Configure UpdateDB to raise an exception
            native_track = mm_track_factory(ID=int(track_id), Title=track_title)
            update_error = RuntimeError("Database update failed")
            native_track.UpdateDB = MagicMock(side_effect=update_error)
            mm_player.sdb.set_tracks([native_track])

        # Test that appropriate exception is raised
        with pytest.raises(expected_exception):
            mm_player._update_rating(audio_tag, new_rating)

        # Verify error logging occurred
        mm_player.logger.error.assert_called()

        if setup_error == "empty_tracks":
            error_call_args = mm_player.logger.error.call_args[0][0]
            assert error_check(error_call_args)
        elif setup_error == "updatedb_exception":
            # For updatedb failure, also verify rating was set before failure
            assert native_track.Rating == 50.0  # 0.5 normalized -> 50 MediaMonkey scale
            native_track.UpdateDB.assert_called_once()

    def test_metadata_update_converts_scale(self, mm_player, mm_track_factory):
        """Test that _update_rating correctly converts ratings from different scales to MediaMonkey scale."""
        audio_tag = AudioTag(ID="333", title="Scale Test", artist="Artist", album="Album", track=1)
        native_track = mm_track_factory(ID=333)
        native_track.UpdateDB = MagicMock()
        mm_player.sdb.set_tracks([native_track])

        # Test key conversion scenarios
        test_cases = [
            (Rating(0.0, scale=RatingScale.NORMALIZED), 0.0),  # Unrated
            (Rating(0.5, scale=RatingScale.NORMALIZED), 50.0),  # Middle rating
            (Rating(1.0, scale=RatingScale.NORMALIZED), 100.0),  # Max rating
        ]

        for input_rating, expected_mm_rating in test_cases:
            # Reset the native track rating
            native_track.Rating = 0
            native_track.UpdateDB.reset_mock()

            mm_player._update_rating(audio_tag, input_rating)

            assert native_track.Rating == expected_mm_rating
            native_track.UpdateDB.assert_called_once()

    def test_metadata_integration_all_fields(self, mm_player, mm_track_factory):
        """Test that _read_track_metadata correctly processes all track fields in integration."""
        native_track = mm_track_factory(
            ID=12345,
            Title="Complete Track Info",
            Rating=75,
            ArtistName="Integration Artist",
            AlbumName="Integration Album",
            Path="/full/path/to/track.mp3",
            TrackOrder=7,
            SongLength=195000,  # 3:15 in milliseconds
        )
        mm_player.cache_mgr.get_metadata.return_value = None

        result = mm_player._read_track_metadata(native_track)

        # Verify all fields are correctly mapped and converted
        assert result.ID == 12345
        assert result.title == "Complete Track Info"
        assert result.artist == "Integration Artist"
        assert result.album == "Integration Album"
        assert result.file_path == "/full/path/to/track.mp3"
        assert result.track == 7
        assert result.duration == 195  # Converted from ms to seconds
        assert result.rating.to_float(RatingScale.NORMALIZED) == 0.75  # 75/100 converted to normalized


class TestLoadPlaylistTracks:
    """Tests for MediaMonkey.load_playlist_tracks integration and error handling."""

    def test_load_successful_track_loading(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test that load_playlist_tracks successfully loads tracks from a normal playlist."""
        # Set up tracks and playlist
        track1 = mm_track_factory(ID=1, Title="Load Track 1", ArtistName="Artist A", Rating=60)
        track2 = mm_track_factory(ID=2, Title="Load Track 2", ArtistName="Artist B", Rating=80)
        native_playlist = mm_playlist_factory(ID=100, Title="Load Test Playlist", tracks=[track1, track2])
        mm_player.sdb.set_playlists([native_playlist])

        # Create target playlist object
        from sync_items import Playlist

        playlist = Playlist(ID=100, name="Load Test Playlist")
        playlist.is_auto_playlist = False
        playlist.tracks = []

        # Call method under test
        mm_player.load_playlist_tracks(playlist)

        # Verify tracks were loaded correctly
        assert len(playlist.tracks) == 2
        assert playlist.tracks[0].title == "Load Track 1"
        assert playlist.tracks[0].artist == "Artist A"
        assert playlist.tracks[1].title == "Load Track 2"
        assert playlist.tracks[1].artist == "Artist B"

    def test_load_missing_playlist_warning(self, mm_player):
        """Test that load_playlist_tracks logs warning when playlist is not found."""
        # Don't set any playlists - simulate missing playlist
        mm_player.sdb.set_playlists([])

        # Create target playlist
        from sync_items import Playlist

        playlist = Playlist(ID=999, name="Missing Playlist")
        playlist.is_auto_playlist = False
        playlist.tracks = []

        # Call method under test
        mm_player.load_playlist_tracks(playlist)

        # Verify warning was logged and no tracks were added
        mm_player.logger.warning.assert_called_with("Playlist 'Missing Playlist' not found")
        assert len(playlist.tracks) == 0

    def test_load_auto_playlist_skips(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test that load_playlist_tracks skips track loading for auto-playlists."""
        # Set up auto-playlist with tracks
        track1 = mm_track_factory(ID=1, Title="Auto Track")
        native_playlist = mm_playlist_factory(ID=200, Title="Auto Playlist", isAutoplaylist=True, tracks=[track1])
        mm_player.sdb.set_playlists([native_playlist])

        # Create target auto-playlist
        from sync_items import Playlist

        playlist = Playlist(ID=200, name="Auto Playlist")
        playlist.is_auto_playlist = True
        playlist.tracks = []

        # Call method under test
        mm_player.load_playlist_tracks(playlist)

        # Verify no tracks were loaded for auto-playlist
        assert len(playlist.tracks) == 0

    def test_load_empty_playlist_gracefully(self, mm_player, mm_playlist_factory):
        """Test that load_playlist_tracks handles empty playlists correctly."""
        # Create empty playlist
        native_playlist = mm_playlist_factory(ID=300, Title="Empty Playlist", tracks=[])
        mm_player.sdb.set_playlists([native_playlist])

        # Create target playlist
        from sync_items import Playlist

        playlist = Playlist(ID=300, name="Empty Playlist")
        playlist.is_auto_playlist = False
        playlist.tracks = []

        # Call method under test
        mm_player.load_playlist_tracks(playlist)

        # Verify empty playlist handled correctly
        assert len(playlist.tracks) == 0

    @pytest.mark.parametrize(
        "track_count,expect_progress_bar",
        [
            (99, False),  # Just under threshold
            (101, True),  # Over threshold - progress bar expected
        ],
        ids=["under_threshold", "over_threshold"],
    )
    def test_load_progress_bar_threshold(self, mm_player, mm_playlist_factory, mm_track_factory, track_count, expect_progress_bar):
        """Test that load_playlist_tracks creates progress bar based on track count threshold."""
        # Create playlist with specified number of tracks
        tracks = [mm_track_factory(ID=i, Title=f"Progress Track {i}") for i in range(1, track_count + 1)]
        native_playlist = mm_playlist_factory(ID=400, Title="Progress Test", tracks=tracks)
        mm_player.sdb.set_playlists([native_playlist])

        # Mock status manager
        progress_bar_mock = MagicMock()
        mm_player.status_mgr.start_phase.return_value = progress_bar_mock

        # Create target playlist
        from sync_items import Playlist

        playlist = Playlist(ID=400, name="Progress Test")
        playlist.is_auto_playlist = False
        playlist.tracks = []

        # Call method under test
        mm_player.load_playlist_tracks(playlist)

        # Verify progress bar behavior
        if expect_progress_bar:
            mm_player.status_mgr.start_phase.assert_called_once_with("Reading tracks from playlist Progress Test", total=track_count)
            assert progress_bar_mock.update.call_count == track_count
            progress_bar_mock.close.assert_called_once()
        else:
            mm_player.status_mgr.start_phase.assert_not_called()
            progress_bar_mock.update.assert_not_called()
            progress_bar_mock.close.assert_not_called()

        # Verify all tracks were loaded regardless of progress bar
        assert len(playlist.tracks) == track_count

    def test_load_metadata_conversion_error_handling(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test that load_playlist_tracks handles metadata conversion errors gracefully."""
        # Create playlist with tracks
        track1 = mm_track_factory(ID=1, Title="Good Track")
        track2 = mm_track_factory(ID=2, Title="Bad Track")  # This will cause error
        native_playlist = mm_playlist_factory(ID=500, Title="Error Test", tracks=[track1, track2])
        mm_player.sdb.set_playlists([native_playlist])

        # Mock _read_track_metadata to raise exception on second track
        original_read_track_metadata = mm_player._read_track_metadata
        call_count = 0

        def mock_read_track_metadata(track):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call raises exception
                raise RuntimeError("Metadata conversion failed")
            return original_read_track_metadata(track)

        mm_player._read_track_metadata = mock_read_track_metadata

        # Create target playlist
        from sync_items import Playlist

        playlist = Playlist(ID=500, name="Error Test")
        playlist.is_auto_playlist = False
        playlist.tracks = []

        # Call method under test - should raise exception
        with pytest.raises(RuntimeError, match="Metadata conversion failed"):
            mm_player.load_playlist_tracks(playlist)

        # Verify first track was processed before error
        assert len(playlist.tracks) == 1
        assert playlist.tracks[0].title == "Good Track"

    def test_load_search_by_title_not_id(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test that load_playlist_tracks searches playlist by title, not ID."""
        # Create playlist
        track1 = mm_track_factory(ID=1, Title="Title Search Track")
        native_playlist = mm_playlist_factory(ID=600, Title="Title Search Test", tracks=[track1])
        mm_player.sdb.set_playlists([native_playlist])

        # Create target playlist with different name but same ID
        from sync_items import Playlist

        playlist = Playlist(ID=600, name="Title Search Test")  # name matches playlist title
        playlist.is_auto_playlist = False
        playlist.tracks = []

        # Spy on search_playlists to verify search method
        original_search = mm_player.search_playlists
        search_calls = []

        def spy_search_playlists(*args, **kwargs):
            search_calls.append((args, kwargs))
            return original_search(*args, **kwargs)

        mm_player.search_playlists = spy_search_playlists

        # Call method under test
        mm_player.load_playlist_tracks(playlist)

        # Verify search was called with title, not ID
        assert len(search_calls) == 1
        assert search_calls[0][0] == ("title", "Title Search Test")
        assert search_calls[0][1]["return_native"] is True

        # Verify track was loaded
        assert len(playlist.tracks) == 1
        assert playlist.tracks[0].title == "Title Search Track"

    def test_load_integration_with_metadata(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test integration between load_playlist_tracks and _read_track_metadata."""
        # Set up complex track data to verify metadata reading
        track1 = mm_track_factory(
            ID=1,
            Title="Integration Track",
            ArtistName="Integration Artist",
            AlbumName="Integration Album",
            Rating=75,
            TrackOrder=3,
            SongLength=210000,  # 3.5 minutes
            Path="/integration/path.mp3",
        )
        native_playlist = mm_playlist_factory(ID=700, Title="Integration Test", tracks=[track1])
        mm_player.sdb.set_playlists([native_playlist])

        # Create target playlist
        from sync_items import Playlist

        playlist = Playlist(ID=700, name="Integration Test")
        playlist.is_auto_playlist = False
        playlist.tracks = []

        # Call method under test
        mm_player.load_playlist_tracks(playlist)

        # Verify comprehensive metadata integration
        assert len(playlist.tracks) == 1
        loaded_track = playlist.tracks[0]

        assert loaded_track.title == "Integration Track"
        assert loaded_track.artist == "Integration Artist"
        assert loaded_track.album == "Integration Album"
        assert loaded_track.track == 3
        assert loaded_track.duration == 210  # Converted from milliseconds to seconds
        assert loaded_track.file_path == "/integration/path.mp3"

        # Verify rating conversion (MediaMonkey 75 -> normalized scale)
        from ratings import RatingScale

        assert loaded_track.rating.to_float(RatingScale.ZERO_TO_FIVE) == 3.75  # 75/100 * 5

    def test_load_debug_logging_with_progress(self, mm_player, mm_playlist_factory, mm_track_factory):
        """Test that load_playlist_tracks logs debug messages when progress bar is active."""
        # Create large playlist to trigger progress bar and debug logging
        tracks = [mm_track_factory(ID=i, Title=f"Debug Track {i}") for i in range(1, 151)]  # 150 tracks
        native_playlist = mm_playlist_factory(ID=800, Title="Debug Test", tracks=tracks)
        mm_player.sdb.set_playlists([native_playlist])

        # Mock status manager
        progress_bar_mock = MagicMock()
        mm_player.status_mgr.start_phase.return_value = progress_bar_mock

        # Create target playlist
        from sync_items import Playlist

        playlist = Playlist(ID=800, name="Debug Test")
        playlist.is_auto_playlist = False
        playlist.tracks = []

        # Call method under test
        mm_player.load_playlist_tracks(playlist)

        # Verify debug logging occurred for each track when progress bar was active
        assert mm_player.logger.debug.call_count == 150

        # Verify debug message format
        debug_calls = mm_player.logger.debug.call_args_list
        assert "Reading track" in debug_calls[0][0][0]
        assert "from playlist Debug Test" in debug_calls[0][0][0]
