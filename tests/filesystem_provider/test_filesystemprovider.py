import io
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from filesystem_provider import FileSystemProvider
from ratings import Rating
from sync_items import AudioTag, Playlist


# --- Fixtures & Factories ---
@pytest.fixture
def fsp_instance(request, tmp_path):
    """Fixture for a FileSystemProvider instance with mocked config and status managers."""
    cfg = getattr(request, "param", {})
    audio_root = tmp_path / "music"
    playlist_root = tmp_path / "playlists"

    fsp = FileSystemProvider()
    # override config_mgr
    config = MagicMock()
    config.path = audio_root
    config.playlist_path = playlist_root
    config.dry = cfg.get("dry", False)
    fsp.config_mgr = config
    # override status_mgr
    status = MagicMock()
    status.start_phase.return_value = MagicMock()
    fsp.status_mgr = status
    # Ensure path and playlist_path are always set (mimic scan_media_files)
    fsp.path = audio_root
    fsp.playlist_path = playlist_root
    return fsp


@pytest.fixture
def file_factory():
    """Factory fixture to create minimal fake file objects with specified suffix and filename."""

    def _factory(suffix, filename):
        file = type("FakeFile", (), {})()
        file.suffix = suffix
        file.filename = filename
        return file

    return _factory


@pytest.fixture
def playlist_factory():
    """
    Factory fixture for in-memory playlist file creation and file IO spying.
    Usage:
        - Requires playlist_root (directory for playlists) and filename (playlist file name).
        - Optionally, lines, is_extm3u, title, exists, and track_exists_map can be provided.
        - Returns (WriteSpy, playlist_path).
    Args:
        playlist_root: Directory in which the playlist file will be created. Must be provided explicitly for clarity and test isolation.
        filename: Name of the playlist file (e.g., 'mylist.m3u').
        lines: List of track lines (str or dict). If dict, supports EXT3 per-track metadata.
        is_extm3u: If True, include #EXTM3U header.
        title: Optional playlist title (EXT3 only).
        exists: If True, Path.exists() returns True for the playlist file; if False, returns False. Other paths are unaffected.
        track_exists_map: Optional dict mapping str(path) to bool for track file existence.
    Example:
        with playlist_factory(playlist_root=tmp_path / 'playlists', filename='mylist.m3u', lines=[...], track_exists_map={...}) as (playlist_content, playlist_path):
            ...
    """

    class WriteSpy(io.StringIO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.write_calls = []
            self.writelines_calls = []

        def write(self, s):
            self.write_calls.append(s)
            return super().write(s)

        def writelines(self, lines):
            self.writelines_calls.append(list(lines))
            return super().writelines(lines)

        def open_for_mode(self, mode):
            if "w" in mode:
                self.seek(0)
                self.truncate(0)
            elif "a" in mode:
                self.seek(0, io.SEEK_END)
            else:
                self.seek(0)
            return self

    @contextmanager
    def _factory(playlist_root, filename, lines, is_extm3u=False, title=None, exists=True, track_exists_map=None):
        if playlist_root is None or filename is None:
            raise ValueError("playlist_root and filename are required for playlist_factory")
        if lines is None:
            raise ValueError("lines must be provided for playlist_factory")

        playlist_root = Path(playlist_root)
        playlist_root.mkdir(parents=True, exist_ok=True)
        playlist_path = playlist_root / filename
        playlist_content = WriteSpy()
        playlist_content.close = lambda: None
        if track_exists_map is None:
            track_exists_map = {}

        # Patch Path.open
        def open_patch(path_self, mode="r", encoding=None, *args, **kwargs):
            if path_self == playlist_path:
                return playlist_content.open_for_mode(mode)
            return Path.__original_open__(path_self, mode, encoding, *args, **kwargs)

        # Patch Path.exists
        def exists_patch(path_self):
            if path_self == playlist_path:
                return exists
            if str(path_self) in track_exists_map:
                return track_exists_map[str(path_self)]
            return Path.exists(path_self)

        open_patcher = patch.object(Path, "open", open_patch)
        exists_patcher = patch.object(Path, "exists", exists_patch)
        open_patcher.start()
        exists_patcher.start()
        try:
            if lines is not None or is_extm3u or title:
                header = []
                if is_extm3u:
                    header.append("#EXTM3U")
                    if title:
                        header.append(f"#PLAYLIST:{title}")
                default_lines = [
                    "music/track1.mp3",
                    "music/track2.mp3",
                    "other/track3.mp3",
                    "other/track4.mp3",
                    "# This is a comment",
                    "",
                ]
                content_lines = header
                for entry in lines if lines is not None else default_lines:
                    if isinstance(entry, dict):
                        extinf = entry.get("extinf")
                        path = entry.get("path")
                        if is_extm3u and extinf:
                            content_lines.append(f"#EXTINF:{extinf}")
                        if path:
                            content_lines.append(path)
                    else:
                        content_lines.append(entry)
                content = "\n".join(content_lines)
                playlist_content.write(content + "\n" if content and not content.endswith("\n") else content)
                playlist_content.seek(0)
            yield playlist_content, playlist_path
        finally:
            open_patcher.stop()
            exists_patcher.stop()

    return _factory


# --- Tests ---
class TestScanMediaFiles:
    """Test scanning for audio and playlist files with various extensions."""

    @pytest.mark.parametrize(
        "suffixes,expected_audio,expected_playlists,playlist_location",
        [
            # All playlists in audio_root, none in playlist_root
            ([".mp3", ".flac", ".ogg", ".m3u", ".m3u8"], 3, 0, "audio_root"),
            # All playlists in playlist_root, none in audio_root
            ([".mp3", ".flac", ".ogg", ".m3u", ".m3u8"], 3, 2, "playlist_root"),
            # Mixed: playlists in both roots, with some overlap
            (
                [".mp3", ".flac", ".ogg", ".m3u", ".m3u8", ".pls", ".m3u"],  # 3 audio, 4 playlists
                3,
                2,  # Only playlists in playlist_root and those in audio_root that are also under playlist_root
                "mixed",
            ),
        ],
    )
    def test_scan_media_files_various(self, fsp_instance, suffixes, expected_audio, expected_playlists, playlist_location):
        playlist_suffixes = {".m3u", ".m3u8", ".pls", ".log"}
        audio_root = fsp_instance.path
        playlist_root = fsp_instance.playlist_path

        def make_mock_file(suffix, name, parent_dir, is_relative_to_val):
            p = MagicMock(spec=Path)
            p.suffix = suffix
            p.is_file.return_value = True
            p.resolve.return_value = p
            p.name = name
            p.exists.return_value = True
            p.is_relative_to.side_effect = lambda other: is_relative_to_val
            p.__str__.return_value = str(parent_dir / name)
            p.parent = parent_dir
            return p

        dummy_files = []
        resolved_paths = {}
        for i, suf in enumerate(suffixes):
            is_playlist = suf in playlist_suffixes
            name = f"test{i}{suf}" if is_playlist else f"track{i}{suf}"
            # Decide parent_dir based on playlist_location
            if is_playlist:
                if playlist_location == "audio_root":
                    parent_dir = audio_root
                elif playlist_location == "playlist_root":
                    parent_dir = playlist_root
                elif playlist_location == "mixed":
                    parent_dir = audio_root if i % 2 == 0 else playlist_root
            else:
                parent_dir = audio_root
            is_relative_to_val = parent_dir == playlist_root
            key = None if is_playlist else f"audio{suf}"
            if not is_playlist and key in resolved_paths:
                p = resolved_paths[key]
            else:
                p = make_mock_file(suf, name, parent_dir, is_relative_to_val)
                if not is_playlist:
                    resolved_paths[key] = p
            dummy_files.append(p)

        # Patch Path.rglob to only return files under the correct root
        def rglob_side_effect(self, pattern):
            return [f for f in dummy_files if str(f.parent) == str(self)]

        with patch.object(Path, "rglob", rglob_side_effect):
            fsp_instance.scan_media_files()
        assert len(fsp_instance.get_tracks()) == expected_audio
        assert len(fsp_instance._get_playlist_paths()) == expected_playlists


class TestReadTrackMetadata:
    """Test reading track metadata for supported and unsupported formats, and deferred resolution."""

    @pytest.mark.parametrize(
        "file_args,expect",
        [
            ((".mp3", "fake.mp3"), True),
            ((".flac", "fake.flac"), True),
            ((".ogg", "fake.ogg"), True),
            ((".txt", "fake.txt"), False),
            ((".nonsense", "fake.nonsense"), False),
        ],
    )
    def test_read_track_metadata_dispatch(self, fsp_instance, file_factory, file_args, expect):
        """Test correct handler dispatch for each file type. Only handler.can_handle/read_tags are patched.
        Asserts type/value correctness and that deferred_tracks is not mutated for non-deferred cases.
        """
        audio_file = file_factory(*file_args)
        file_path = Path(audio_file.filename)
        # Patch only the correct handler's can_handle/read_tags
        handler = None
        if audio_file.suffix == ".mp3":
            handler = fsp_instance.id3_handler
        elif audio_file.suffix in [".flac", ".ogg"]:
            handler = fsp_instance.vorbis_handler
        for h in [fsp_instance.id3_handler, fsp_instance.vorbis_handler]:
            h.can_handle = lambda f, h=h: h is handler
            h.read_tags = lambda f, h=h: (AudioTag(ID="x"), None) if h is handler and expect else (None, None)
        # Patch _open_audio_file to return the fake file
        with patch.object(fsp_instance, "_open_audio_file", return_value=audio_file):
            original_deferred = list(fsp_instance.deferred_tracks)
            result = fsp_instance.read_track_metadata(file_path)
        if expect:
            assert isinstance(result, AudioTag)
        else:
            assert result is None
        # Assert deferred_tracks is not mutated for non-deferred cases
        assert fsp_instance.deferred_tracks == original_deferred

    def test_read_track_metadata_open_error(self, fsp_instance):
        """Test that None is returned if mutagen.File returns None (open error)."""
        with patch("mutagen.File", return_value=None):
            assert fsp_instance.read_track_metadata(Path("x.mp3")) is None

    def test_read_track_metadata_deferred(self, fsp_instance, file_factory):
        """Test that deferred_tracks is populated when raw_ratings is present."""
        audio_file = file_factory(".mp3", "fake.mp3")
        file_path = Path(audio_file.filename)
        # Patch only id3_handler for this test
        fsp_instance.id3_handler.can_handle = lambda f: True
        fsp_instance.vorbis_handler.can_handle = lambda f: False
        fsp_instance.id3_handler.read_tags = lambda f: (AudioTag(ID="x"), {"TXXX:RATING": "5"})
        with patch.object(fsp_instance, "_open_audio_file", return_value=audio_file):
            fsp_instance.deferred_tracks = []
            result = fsp_instance.read_track_metadata(file_path)
        assert isinstance(result, AudioTag)
        assert len(fsp_instance.deferred_tracks) == 1


class ConfigurableFakeHandler:
    """
    A fake handler for testing FileSystemProvider logic. Configurable to simulate all handler behaviors.
    """

    def __init__(self, can_handle_return=True, can_handle_exception=None, apply_tags_return=None, apply_tags_exception=None):
        self.can_handle_return = can_handle_return
        self.can_handle_exception = can_handle_exception
        self.apply_tags_return = apply_tags_return
        self.apply_tags_exception = apply_tags_exception
        self.apply_tags_called = False
        self.can_handle_called = False
        self.last_apply_tags_args = None
        self.last_can_handle_args = None

    def can_handle(self, f):
        self.can_handle_called = True
        self.last_can_handle_args = (f,)
        if self.can_handle_exception:
            raise self.can_handle_exception
        return self.can_handle_return

    def apply_tags(self, audio_file, audio_tag, rating):
        self.apply_tags_called = True
        self.last_apply_tags_args = (audio_file, audio_tag, rating)
        if self.apply_tags_exception:
            raise self.apply_tags_exception
        return self.apply_tags_return


class TestUpdateTrackMetadata:
    """
    Test updating track metadata for various file existence, handler, and save scenarios, including all handler behaviors.
    Only assert handler state when the handler is expected to be used (file exists).
    """

    @pytest.mark.parametrize(
        "file_args,exists,save_success,can_handle_return,can_handle_exception,apply_tags_exception,expect_none",
        [
            # Success: handler supports, apply_tags returns file
            ((".mp3", "fake.mp3"), True, True, True, None, None, False),
            # Handler does not support
            ((".mp3", "fake.mp3"), True, True, False, None, None, True),
            # File does not exist (handler should not be called)
            ((".mp3", "fake.mp3"), False, True, True, None, None, True),
            # Both handlers return False (fallback logic)
            ((".nonsense", "fake.nonsense"), True, True, False, None, None, True),
        ],
    )
    def test_update_track_metadata_branches(
        self, fsp_instance, file_factory, file_args, exists, save_success, can_handle_return, can_handle_exception, apply_tags_exception, expect_none
    ):
        """
        Test update_track_metadata for all handler behaviors using ConfigurableFakeHandler.
        Only assert handler call state when file exists and handler is selected for the file type.
        """
        audio_file = file_factory(*file_args)
        file_path = Path(audio_file.filename)
        fsp_instance.id3_handler = ConfigurableFakeHandler(
            can_handle_return=can_handle_return,
            can_handle_exception=can_handle_exception,
            apply_tags_return=audio_file,
            apply_tags_exception=apply_tags_exception,
        )
        fsp_instance.vorbis_handler.can_handle = lambda f: False
        fsp_instance._handlers = [fsp_instance.id3_handler, fsp_instance.vorbis_handler]
        with patch.object(Path, "exists", return_value=exists):
            with patch.object(fsp_instance, "_open_audio_file", return_value=audio_file):
                # Only inject handler if file exists and handler is selected (e.g., .mp3)
                if not exists and audio_file.suffix != ".mp3":
                    fsp_instance.id3_handler.can_handle = lambda f: False
                # Patch save method on the audio file
                if save_success:
                    audio_file.save = MagicMock(return_value=None)
                else:
                    audio_file.save = MagicMock(side_effect=Exception("fail"))
                original_tags = getattr(audio_file, "tags", None)
                result = fsp_instance.update_track_metadata(file_path, audio_tag=AudioTag(), rating=Rating(1))
        if expect_none:
            assert result is None
            assert getattr(audio_file, "tags", None) == original_tags
            assert not (hasattr(audio_file.save, "called") and audio_file.save.called and save_success)
        else:
            assert result == audio_file
            assert audio_file.save.called
        # Only assert handler call state if file exists and handler is selected
        if exists and audio_file.suffix == ".mp3" and (can_handle_return or can_handle_exception):
            if can_handle_exception is None:
                assert fsp_instance.id3_handler.can_handle_called
                if can_handle_return:
                    assert fsp_instance.id3_handler.apply_tags_called or apply_tags_exception is not None


class TestFinalizeScan:
    """Test finalize_scan for small and large deferred_tracks (bar/no-bar)."""

    @pytest.mark.parametrize("count,use_bar", [(3, False), (150, True)])
    def test_finalize_scan_small_vs_large(self, fsp_instance, count, use_bar):
        # prepare deferred_tracks
        handler = MagicMock()
        handler.resolve_rating.return_value = Rating(2)
        fsp_instance.deferred_tracks = [{"track": AudioTag(ID=str(i), title="t", artist="a", album="b", track=1), "handler": handler, "raw": {}} for i in range(count)]
        fsp_instance.update_track_metadata = MagicMock()

        result = fsp_instance.finalize_scan()
        assert len(result) == count
        if use_bar:
            fsp_instance.status_mgr.start_phase.assert_called()
        else:
            fsp_instance.status_mgr.start_phase.assert_not_called()
        assert fsp_instance.update_track_metadata.call_count == count


class TestPlaylistCreation:
    """Test create_playlist for dry-run, mkdir error, open error, and success."""

    @pytest.mark.parametrize(
        "dry,mkdir_fail,write_fail,expect_none",
        [(False, False, False, False), (True, False, False, True), (False, True, False, True), (False, False, True, True)],
    )
    def test_create_playlist_success_and_failures(self, fsp_instance, dry, mkdir_fail, write_fail, expect_none):
        """Test create_playlist for all error and success branches."""
        fsp_instance.config_mgr.dry = dry
        m_open = mock_open()
        if write_fail:
            m_open.side_effect = OSError()
        if mkdir_fail:
            # Only patch Path.mkdir when simulating failure
            with patch.object(Path, "mkdir", side_effect=OSError()):
                with patch("pathlib.Path.open", m_open):
                    result = fsp_instance.create_playlist("pl", is_extm3u=True)
        elif not expect_none:
            # For the success branch, use a real file (no mock_open)
            result = fsp_instance.create_playlist("pl", is_extm3u=True)
        else:
            with patch("pathlib.Path.open", m_open):
                result = fsp_instance.create_playlist("pl", is_extm3u=True)
        if expect_none:
            assert result is None
        else:
            assert isinstance(result, Playlist)


class TestReadPlaylistMetadata:
    """
    Test read_playlist_metadata for various playlist content and header/title combinations.
    All playlist IO is handled in-memory via playlist_factory.
    - lines: List of lines for the playlist file (including headers/comments as needed)
    - is_extm3u: Whether to treat as EXT3 format (adds #EXTM3U if True)
    - title: Optional playlist title (for EXT3)
    - expect_none: If True, expect None result; else expect Playlist
    - expected_name: Expected playlist name if not None
    """

    @pytest.mark.parametrize(
        "lines,is_extm3u,title,expect_none,expected_name",
        [
            (["music/track1.mp3"], False, None, False, "x"),
            (["music/track1.mp3"], True, None, False, "x"),
            (["#PLAYLIST:MyList", "music/track1.mp3"], True, "MyList", False, "MyList"),
            (["#EXTM3U", "# This is a comment", ""], True, None, False, "x"),
        ],
    )
    def test_read_playlist_metadata_variations(self, fsp_instance, playlist_factory, lines, is_extm3u, title, expect_none, expected_name):
        """
        Test read_playlist_metadata for various playlist content and header/title combinations.
        All playlist IO is handled in-memory via playlist_factory.
        """
        with playlist_factory(
            playlist_root=fsp_instance.playlist_path,
            filename="x.m3u",
            lines=lines,
            is_extm3u=is_extm3u,
            title=title,
        ) as (playlist_content, playlist_path):
            result = fsp_instance.read_playlist_metadata(playlist_path)
            if expect_none:
                assert result is None
            else:
                assert isinstance(result, Playlist)
                assert result.is_extm3u == is_extm3u
                assert result.name == expected_name


class TestGetTracksFromPlaylist:
    """
    Test get_tracks_from_playlist for filtering of comments, blank, in-scope, and out-of-scope lines.
    Uses playlist_factory for all playlist content injection and parameterization for edge cases.
    """

    @pytest.mark.parametrize(
        "playlist_case",
        [
            # Default: in-scope, out-of-scope, comment, blank
            [
                {"line": "music/track1.mp3", "exists": True, "in_scope": True, "expected": "included"},
                {"line": "music/track2.mp3", "exists": True, "in_scope": True, "expected": "included"},
                {"line": "other/track3.mp3", "exists": False, "in_scope": False, "expected": "excluded"},
                {"line": "other/track4.mp3", "exists": False, "in_scope": False, "expected": "excluded"},
                {"line": "# This is a comment", "expected": "excluded"},
                {"line": "", "expected": "excluded"},
            ],
            # Only comments and blanks
            [
                {"line": "#c", "expected": "excluded"},
                {"line": "", "expected": "excluded"},
            ],
            # One valid, one invalid
            [
                {"line": "a.mp3", "exists": True, "in_scope": True, "expected": "included"},
                {"line": "b.mp3", "exists": False, "in_scope": True, "expected": "excluded"},
            ],
            # Only comment
            [
                {"line": "# only comment", "expected": "excluded"},
            ],
            # Only blank
            [
                {"line": "", "expected": "excluded"},
            ],
            # Mixed valid/invalid
            [
                {"line": "music/track1.mp3", "exists": True, "in_scope": True, "expected": "included"},
                {"line": "invalid/path", "exists": False, "in_scope": False, "expected": "excluded"},
            ],
        ],
        ids=[
            "Default playlist: in-scope, out-of-scope, comment, blank",
            "Only comments and blanks",
            "One valid, one invalid",
            "Only comment",
            "Only blank",
            "Mixed valid/invalid",
        ],
    )
    def test_get_tracks_from_playlist_various(self, fsp_instance, playlist_factory, playlist_case):
        """
        Parameterized test for get_tracks_from_playlist using a list of dicts per playlist.
        Each dict represents a line and its expected inclusion/exclusion.
        """
        lines = [entry["line"] for entry in playlist_case]
        with playlist_factory(
            playlist_root=fsp_instance.playlist_path,
            filename="test_playlist.m3u",
            lines=lines,
        ) as (playlist_content, playlist_path):
            # Build maps for exists and in_scope
            exists_map = {}
            scope_map = {}
            for entry in playlist_case:
                line = entry["line"]
                if "exists" in entry and line and not line.strip().startswith("#"):
                    candidate_path = (playlist_path.parent / line) if not Path(line).is_absolute() else Path(line)
                    exists_map[str(candidate_path)] = entry["exists"]
                if "in_scope" in entry and line and not line.strip().startswith("#"):
                    candidate_path = (playlist_path.parent / line) if not Path(line).is_absolute() else Path(line)
                    scope_map[str(candidate_path)] = entry["in_scope"]

            real_exists = Path.exists
            real_is_relative_to = Path.is_relative_to

            def exists_side_effect(self):
                if self == playlist_path:
                    return real_exists(self)
                return exists_map.get(str(self), False)

            def is_relative_to_side_effect(self, other):
                if self == playlist_path:
                    return real_is_relative_to(self, other)
                return scope_map.get(str(self), False)

            with patch.object(Path, "exists", exists_side_effect), patch.object(Path, "is_relative_to", is_relative_to_side_effect):
                result = fsp_instance.get_tracks_from_playlist(playlist_path)

            result_paths = {str(Path(p).resolve()) for p in result}

            for entry in playlist_case:
                line = entry["line"]
                expected = entry["expected"]
                if not line or line.strip().startswith("#"):
                    # Comments and blanks are always excluded
                    if expected == "included":
                        pytest.fail(f"Comment or blank line '{line}' should not be included.")
                    continue
                candidate_path = (playlist_path.parent / line) if not Path(line).is_absolute() else Path(line)
                candidate_abs = str(candidate_path.resolve())
                if expected == "included":
                    assert candidate_abs in result_paths, f"Expected '{line}' to be included, but it was not."
                else:
                    assert candidate_abs not in result_paths, f"Did not expect '{line}' to be included, but it was."


class TestAddTrackToPlaylist:
    """Test add_track_to_playlist for success and file open error."""

    def test_add_track_to_playlist_success(self, fsp_instance, playlist_factory):
        """Test that add_track_to_playlist writes EXTINF and track path to the playlist file, and logs writes."""
        track1_path = fsp_instance.path / "test_track1.mp3"
        track2_path = fsp_instance.path.parent / "test_track2.mp3"
        track1 = AudioTag(file_path=str(track1_path), artist="A", title="T", duration=123)
        track2 = AudioTag(file_path=str(track2_path), artist="B", title="U", duration=123)
        with playlist_factory(
            playlist_root=fsp_instance.playlist_path,
            filename="test_playlist.m3u",
            lines=["#EXTM3U", "# This is a comment", "music/track1.mp3"],
            is_extm3u=True,
        ) as (playlist_content, playlist_path):
            playlist_content.seek(0)
            original_content = playlist_content.read()
            assert str(track1_path) not in original_content
            assert str(track2_path) not in original_content

            fsp_instance.add_track_to_playlist(playlist_path, track1, is_extm3u=True)
            fsp_instance.add_track_to_playlist(playlist_path, track2, is_extm3u=True)
            playlist_content.seek(0)
            content = playlist_content.read()
            # Assert on write calls
            assert any("#EXTINF:123,A - T" in call for call in playlist_content.write_calls)
            assert any("test_track1.mp3" in call for call in playlist_content.write_calls)
            assert "#EXTINF:123,A - T" in content
            assert "test_track1.mp3" in content
            assert not any("#EXTINF:123,B - U" in call for call in playlist_content.write_calls)
            assert not any("test_track2.mp3" in call for call in playlist_content.write_calls)
            assert "#EXTINF:123,B - U" not in content
            assert "test_track2.mp3" not in content


class TestRemoveTrackFromPlaylist:
    """Test remove_track_from_playlist for success and file open error."""

    @pytest.mark.parametrize(
        "initial_lines,remove_track,expected_lines",
        [
            (["music/track1.mp3", "music/track2.mp3"], "music/track1.mp3", ["music/track2.mp3"]),
            (["music/track1.mp3", "music/track2.mp3"], "music/track2.mp3", ["music/track1.mp3"]),
            (["music/track1.mp3"], "music/track1.mp3", []),
            (["music/track1.mp3", "music/track2.mp3"], "music/track3.mp3", ["music/track1.mp3", "music/track2.mp3"]),
        ],
    )
    def test_remove_track_from_playlist_success(self, fsp_instance, playlist_factory, initial_lines, remove_track, expected_lines):
        """
        Test that remove_track_from_playlist removes the correct track and logs writes, using playlist_factory and WriteSpy.
        Uses playlist_factory for playlist file setup to ensure test isolation and clarity.
        """
        with playlist_factory(
            playlist_root=fsp_instance.playlist_path,
            filename="test_playlist.m3u",
            lines=initial_lines,
        ) as (playlist_content, playlist_path):
            playlist_content.seek(0)
            original_content = playlist_content.read()

            track = playlist_path.parent / remove_track
            fsp_instance.remove_track_from_playlist(playlist_path, track)
            playlist_content.seek(0)
            new_content = playlist_content.read()
            # Normalize written and expected lines to relative paths for comparison
            written_lines = [str(Path(line.strip())) for lines in playlist_content.writelines_calls for line in lines if line.strip()]
            expected_lines_stripped = [str(Path(l.strip())) for l in expected_lines if l.strip()]
            assert written_lines == expected_lines_stripped or (not written_lines and not expected_lines_stripped)
            # Assert file content matches expected
            if remove_track in initial_lines:
                assert original_content != new_content
            else:
                assert original_content == new_content


class TestGetPlaylistTitle:
    """Test _get_playlist_title for disambiguation when all folders are exhausted (warning branch)."""

    def test_get_playlist_title_exhausted(self, fsp_instance, tmp_path):
        """Test that _get_playlist_title returns a disambiguated title when all folders are exhausted."""
        pl_path = tmp_path / "a" / "b" / "MyList.m3u"
        pl_path.parent.mkdir(parents=True)
        fsp_instance.playlist_path = tmp_path
        # Simulate duplicate title already mapped
        fsp_instance._playlist_title_map["MyList"] = pl_path.parent / "other" / "MyList.m3u"
        # Should warn and return a candidate
        title = fsp_instance._get_playlist_title(pl_path, "MyList")
        assert "MyList" in title


class TestReadPlaylistMetadataOpenError:
    """Test read_playlist_metadata for file open failure (returns None)."""

    def test_read_playlist_metadata_open_error(self, fsp_instance, tmp_path):
        """Test that read_playlist_metadata returns None if file open fails."""
        pl_path = tmp_path / "fail.m3u"
        with patch.object(fsp_instance, "_open_playlist", return_value=None):
            result = fsp_instance.read_playlist_metadata(pl_path)
        assert result is None
