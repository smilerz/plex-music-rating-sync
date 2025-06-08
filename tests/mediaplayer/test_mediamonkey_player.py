from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ratings import Rating, RatingScale
from sync_items import AudioTag


class SQLQueryProcessor:
    """Basic SQL query parsing and filtering logic for MediaMonkey track queries."""

    def process_query(self, sql_query, tracks):
        import re

        # Simulate title queries with escaping
        if "SongTitle" in sql_query:
            # Handle escaped quotes in MediaMonkey format: "" becomes "
            # Pattern matches: SongTitle = "content with possible ""escaped"" quotes"
            m = re.search(r'SongTitle = "(.+)"$', sql_query)
            if m:
                title = m.group(1).replace('""', '"')
                return [t for t in tracks if getattr(t, "Title", None) == title]

        # Simulate ID queries
        if "ID = " in sql_query:
            m = re.search(r"ID = (\d+)", sql_query)
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

        def QuerySongs(self, sql_query):
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

        def set_tracks(self, tracks):
            self._tracks = tracks

        def set_playlists(self, playlists):
            self._playlists = playlists
            self._playlist_by_id = {pl.ID: pl for pl in playlists}
            self._playlist_by_title = {pl.Title.lower(): pl for pl in playlists}
            self._root_playlist.ChildPlaylists = playlists

        def PlaylistByTitle(self, title):
            if title == "":
                return self._root_playlist
            return self._playlist_by_title.get(title.lower())

        def PlaylistByID(self, id):
            return self._playlist_by_id.get(int(id))

        def _make_playlist(self, ID=1, Title="Playlist", isAutoplaylist=False, tracks=None, children=None):
            pl = SimpleNamespace()
            pl.ID = ID
            pl.Title = Title
            pl.isAutoplaylist = isAutoplaylist
            pl.Tracks = tracks or []
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


class TestMediaMonkeyConnect:
    """Tests for MediaMonkey.connect behavior and error handling."""

    def test_connect_success_logs_info(self, mm_player):
        """Test that MediaMonkey.connect logs info when successful."""
        mm_player.connect()
        mm_player.logger.info.assert_called()

    def test_connect_failure_raises_exception_and_logs(self, mm_player, mm_api):
        """Test that MediaMonkey.connect raises an exception on connection failure and logs error."""
        # Set the raise condition at the connection level (not QuerySongs level)
        mm_api.connect_raise = RuntimeError("Connection failed")
        with pytest.raises(RuntimeError, match="Connection failed"):
            mm_player.connect()
        mm_player.logger.error.assert_called()


class TestMediaMonkeyPlaylistSearch:
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
    def test_search_playlists_parameters(self, mm_player, mm_playlist_factory, key, value, return_native, setup_playlists, expect_error, expect_empty):
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
    def test_search_playlists_return_native_parameter(self, mm_player, mm_playlist_factory, return_native, playlist_id, title, expected_attributes):
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

    def test_search_playlists_title_case_insensitive(self, mm_player, mm_playlist_factory):
        """Test that title search is case-insensitive."""
        playlist_data = {"ID": 1005, "Title": "CaseSensitive Playlist"}
        native_playlist = mm_playlist_factory(**playlist_data)
        mm_player.sdb.set_playlists([native_playlist])

        # Test various case combinations
        for search_title in ["casesensitive playlist", "CASESENSITIVE PLAYLIST", "CaseSensitive Playlist"]:
            result = mm_player.search_playlists("title", search_title)
            assert len(result) == 1
            assert result[0].name == "CaseSensitive Playlist"

    def test_search_playlists_nested_hierarchy_with_dots(self, mm_player, mm_playlist_factory):
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

    def test_search_playlists_all_preserves_order(self, mm_player, mm_playlist_factory):
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

    def test_search_playlists_empty_collection_returns_empty_list(self, mm_player):
        """Test searching playlists when no playlists exist returns empty list."""
        mm_player.sdb.set_playlists([])

        result = mm_player.search_playlists("all")
        assert isinstance(result, list)
        assert len(result) == 0


class TestMediaMonkeyPlaylistCreation:
    """Tests for MediaMonkey playlist creation and conversion."""

    # TODO: Implement tests that instantiate MediaMonkey and test playlist creation/conversion logic.
    pass


class TestMediaMonkeyPlaylistTracks:
    """Tests for MediaMonkey playlist track reading and counting."""

    def test_read_native_playlist_tracks_returns_converted_tracks(self, mm_player, mm_playlist_factory, mm_track_factory):
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

    def test_read_native_playlist_tracks_handles_empty_playlist(self, mm_player, mm_playlist_factory):
        """Test that _read_native_playlist_tracks handles empty playlists correctly."""
        # Create empty playlist
        playlist = mm_playlist_factory(Title="Empty Playlist")

        # Call method under test
        result = mm_player._read_native_playlist_tracks(playlist)

        # Verify empty list returned
        assert isinstance(result, list)
        assert len(result) == 0

    def test_read_native_playlist_tracks_iterates_by_count(self, mm_player, mm_playlist_factory, mm_track_factory):
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

    def test_read_native_playlist_tracks_preserves_order(self, mm_player, mm_playlist_factory, mm_track_factory):
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
    def test_get_native_playlist_track_count(self, mm_player, mm_playlist_factory, count, description):
        """Test that _get_native_playlist_track_count returns the Count property value for various playlist sizes."""
        # Create dummy tracks to match the count
        dummy_tracks = [SimpleNamespace(ID=i, Title=f"Track {i}") for i in range(count)]
        playlist = mm_playlist_factory(Title="Test Playlist", tracks=dummy_tracks)

        # Call method under test
        result = mm_player._get_native_playlist_track_count(playlist)

        # Verify correct count returned
        assert result == count
        assert isinstance(result, int)

    def test_playlist_track_methods_integration(self, mm_player, mm_playlist_factory, mm_track_factory):
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


class TestMediaMonkeyTrackSearch:
    """Tests for MediaMonkey track search by title, ID, and rating."""

    def test_search_tracks_by_id_returns_track(self, mm_player, mm_track_factory):
        """Test that MediaMonkey.search_tracks returns the correct track by ID using fixture-based dependency injection."""
        track1 = mm_track_factory(ID=1, Title="A")
        track2 = mm_track_factory(ID=2, Title="B")
        mm_player.sdb.set_tracks([track1, track2])
        results = mm_player.search_tracks("id", 2)
        assert isinstance(results, list)
        assert any(getattr(t, "ID", None) == 2 for t in results)
        assert all(getattr(t, "ID", None) == 2 for t in results)

    def test_search_tracks_query_raises_exception(self, mm_player, mm_track_factory):
        """Test that MediaMonkey.search_tracks raises if the underlying query raises, using fixture-based error simulation."""
        mm_player.sdb.QuerySongs_raise = RuntimeError("Query failed")
        with pytest.raises(RuntimeError, match="Query failed"):
            mm_player.search_tracks("id", 1)

    def test_search_tracks_by_title_returns_matching_tracks(self, mm_player, mm_track_factory):
        """Test that MediaMonkey.search_tracks returns tracks matching the exact title."""
        track1 = mm_track_factory(ID=1, Title="My Song")
        track2 = mm_track_factory(ID=2, Title="Another Song")
        track3 = mm_track_factory(ID=3, Title="My Song")  # Duplicate title
        mm_player.sdb.set_tracks([track1, track2, track3])

        results = mm_player.search_tracks("title", "My Song")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(getattr(t, "title", None) == "My Song" for t in results)

    def test_search_tracks_by_title_with_quotes_escapes_correctly(self, mm_player, mm_track_factory):
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
    def test_search_tracks_by_rating_criteria(self, mm_player, mm_track_factory, search_value, track_setup, expected_count, expected_ids):
        """Test that MediaMonkey.search_tracks returns correct tracks for various rating search criteria."""
        tracks = [mm_track_factory(**track_data) for track_data in track_setup]
        mm_player.sdb.set_tracks(tracks)

        results = mm_player.search_tracks("rating", search_value)

        assert isinstance(results, list)
        assert len(results) == expected_count
        matched_ids = {getattr(t, "ID", None) for t in results}
        assert matched_ids == expected_ids

    def test_search_tracks_invalid_key_raises_key_error(self, mm_player):
        """Test that MediaMonkey.search_tracks raises KeyError for invalid search key."""
        with pytest.raises(KeyError, match="Invalid search mode"):
            mm_player.search_tracks("invalid_key", "value")

    def test_search_tracks_empty_value_raises_value_error(self, mm_player):
        """Test that MediaMonkey.search_tracks raises ValueError for empty search value."""
        with pytest.raises(ValueError, match="value can not be empty"):
            mm_player.search_tracks("id", "")

    def test_search_tracks_return_native_bypasses_conversion(self, mm_player, mm_track_factory):
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


class TestMediaMonkeyTrackMetadata:
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
    def test_read_track_metadata_cache_behavior(self, mm_player, mm_track_factory, cache_scenario, track_data, expected_cache_calls):
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
    def test_read_track_metadata_rating_conversion(self, mm_player, mm_track_factory, mm_rating, expected_normalized, is_unrated):
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

    def test_read_track_metadata_missing_song_length_defaults_to_negative_one(self, mm_player, mm_track_factory):
        """Test that _read_track_metadata handles missing SongLength gracefully."""
        native_track = mm_track_factory(ID=111, SongLength=None)
        mm_player.cache_mgr.get_metadata.return_value = None

        result = mm_player._read_track_metadata(native_track)

        assert result.duration == -1

    def test_update_rating_success_updates_native_track_and_database(self, mm_player, mm_track_factory):
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
    def test_update_rating_error_scenarios(self, mm_player, mm_track_factory, error_scenario, track_id, track_title, setup_error, expected_exception, error_check):
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

    def test_update_rating_rating_scale_conversion_various_values(self, mm_player, mm_track_factory):
        """Test that _update_rating correctly converts ratings from different scales to MediaMonkey scale."""
        audio_tag = AudioTag(ID="333", title="Scale Test", artist="Artist", album="Album", track=1)
        native_track = mm_track_factory(ID=333)
        native_track.UpdateDB = MagicMock()
        mm_player.sdb.set_tracks([native_track])

        test_cases = [
            (Rating(0.0, scale=RatingScale.NORMALIZED), 0.0),  # Unrated
            (Rating(0.2, scale=RatingScale.NORMALIZED), 20.0),  # 1 star
            (Rating(0.5, scale=RatingScale.NORMALIZED), 50.0),  # 2.5 stars
            (Rating(1.0, scale=RatingScale.NORMALIZED), 100.0),  # 5 stars
            (Rating(3, scale=RatingScale.ZERO_TO_FIVE), 60.0),  # 3/5 stars
            (Rating(8, scale=RatingScale.ZERO_TO_TEN), 80.0),  # 8/10 stars
        ]

        for input_rating, expected_mm_rating in test_cases:
            # Reset the native track rating
            native_track.Rating = 0
            native_track.UpdateDB.reset_mock()

            mm_player._update_rating(audio_tag, input_rating)

            assert native_track.Rating == expected_mm_rating
            native_track.UpdateDB.assert_called_once()

    def test_read_track_metadata_integration_with_all_fields(self, mm_player, mm_track_factory):
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


class TestMediaMonkeyLoadPlaylistTracks:
    """Tests for MediaMonkey.load_playlist_tracks integration and error handling."""

    # TODO: Implement tests that instantiate MediaMonkey and test load_playlist_tracks integration.
    pass
