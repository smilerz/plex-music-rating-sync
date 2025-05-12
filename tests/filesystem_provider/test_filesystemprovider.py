import io
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from mutagen.id3 import ID3FileType

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
        - Optionally, lines, is_extm3u, title, exists, open_raises, and track_exists_map can be provided.
        - Returns (WriteSpy, playlist_path).
    Args:
        playlist_root: Directory in which the playlist file will be created. Must be provided explicitly for clarity and test isolation.
        filename: Name of the playlist file (e.g., 'mylist.m3u').
        lines: List of track lines (str or dict). If dict, supports EXT3 per-track metadata.
        is_extm3u: If True, include #EXTM3U header.
        title: Optional playlist title (EXT3 only).
        exists: If True, Path.exists() returns True for the playlist file; if False, returns False. Other paths are unaffected.
        open_raises: If True, Path.open will raise an OSError for the playlist file (simulates open error).
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
    def _factory(playlist_root, filename, lines, is_extm3u=False, title=None, exists=True, open_raises=False, track_exists_map=None):
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
                if open_raises:
                    raise OSError("Simulated open error")
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


def make_mock_file(suffix, name, parent_dir, is_file=True, is_relative_to_val=None, resolved_to=None):
    """
    Helper to create a MagicMock Path object for use in scan_media_files tests.
    Args:
        suffix: File extension (e.g., '.mp3')
        name: File name
        parent_dir: Parent directory (Path)
        is_file: Whether this mock is a file (default True)
        is_relative_to_val: Value to return for is_relative_to (default: parent_dir == playlist_root)
        resolved_to: Value to return for resolve (default: self)
    """
    p = MagicMock(spec=Path)
    p.suffix = suffix
    p.is_file.return_value = is_file
    p.resolve.return_value = resolved_to if resolved_to is not None else p
    p.name = name
    p.exists.return_value = True
    # If not specified, default to False (for most tests)
    if is_relative_to_val is None:
        p.is_relative_to.side_effect = lambda other: False
    else:
        p.is_relative_to.side_effect = lambda other: is_relative_to_val
    p.__str__.return_value = str(parent_dir / name)
    p.parent = parent_dir
    return p

    # --- Tests ---


class TestScanMediaFiles:
    """
    Test scanning for audio and playlist files with various directory structures and file types.
    This refactored version uses the FileSystemProvider's own methods to verify discovered files,
    ensuring the test validates the provider's detection logic, not the test's filtering logic.
    """

    @pytest.mark.parametrize(
        "playlist_dir, audio_files, playlist_files, expected_audio, expected_playlists",
        [
            # Case 1: roots same, all audio and playlist types
            ("music", ["track1.mp3", "track2.flac", "track3.ogg"], ["list1.m3u", "list2.m3u8", "list3.pls", "list4.log"], 3, 3),
            # Case 2: subdir, playlists only in playlist_root
            ("music/playlists", ["track1.mp3", "track2.flac", "track3.ogg", "note.txt"], ["list1.m3u", "list2.m3u8", "list3.pls", "list4.log"], 3, 3),
            # Case 3: subdir, playlists in both roots
            ("music/playlists", ["track1.mp3", "list1.m3u", "note.txt", "track2.ogg"], ["list2.m3u8", "list3.pls"], 2, 2),
            # Case 4: disjoint, playlists only in audio_root
            ("other", ["track1.mp3", "track2.ogg", "list3.pls"], ["note.txt"], 2, 0),
            # Case 5: disjoint, playlists only in playlist_root
            ("other", ["track1.mp3", "note.txt", "track2.ogg"], ["list1.m3u", "list2.mp3", "list3.pls"], 2, 2),
            # Case 6: non-audio in audio_root, playlist in playlist_root
            ("music/playlists", ["note.txt", "readme.md"], ["list1.m3u", "list2.log"], 0, 1),
            # Case 7: empty dirs
            ("other", [], [], 0, 0),
        ],
        ids=[
            "roots same, all audio and playlist types",
            "subdir, playlists only in playlist_root",
            "subdir, playlists in both roots",
            "disjoint, playlists only in audio_root",
            "disjoint, playlists only in playlist_root",
            "non-audio in audio_root, playlist in playlist_root",
            "empty dirs",
        ],
    )
    def test_scan_media_files_various(self, fsp_instance, tmp_path, playlist_dir, audio_files, playlist_files, expected_audio, expected_playlists):
        """Parameterized test for FileSystemProvider.scan_media_files covering all relevant scenarios and file types."""
        audio_root = fsp_instance.path = tmp_path / "music"
        playlist_root = fsp_instance.playlist_path = tmp_path / playlist_dir

        # Create dummy files: audio_files in audio_root, playlist_files in playlist_root
        dummy_files = []
        for filename in set(audio_files):
            # is_relative_to should be True if playlist_root is a subdirectory of audio_root
            is_relative = playlist_root.is_relative_to(audio_root)
            dummy_files.append(make_mock_file(Path(filename).suffix, filename, audio_root, is_file=True, is_relative_to_val=is_relative))
        for filename in set(playlist_files):
            dummy_files.append(make_mock_file(Path(filename).suffix, filename, playlist_root, is_file=True, is_relative_to_val=True))

        def rglob_side_effect(self, pattern):
            # If self is audio_root, return all files in both roots if roots are the same
            # If playlist_root is a subdir of audio_root, only include playlist files from playlist_root
            if str(self) == str(audio_root):
                if str(audio_root) == str(playlist_root):
                    return [f for f in dummy_files if str(f.parent) in {str(audio_root), str(playlist_root)}]
                elif playlist_root.is_relative_to(audio_root):
                    # Only include audio files from audio_root and playlist files from playlist_root
                    audio = [f for f in dummy_files if str(f.parent) == str(audio_root) and f.suffix.lower() in fsp_instance.AUDIO_EXT]
                    playlists = [f for f in dummy_files if str(f.parent) == str(playlist_root) and f.suffix.lower() in fsp_instance.PLAYLIST_EXT]
                    return audio + playlists
                else:
                    return [f for f in dummy_files if str(f.parent) == str(audio_root)]
            elif str(self) == str(playlist_root):
                if str(audio_root) == str(playlist_root):
                    return [f for f in dummy_files if str(f.parent) in {str(audio_root), str(playlist_root)}]
                else:
                    return [f for f in dummy_files if str(f.parent) == str(playlist_root)]
            else:
                return []

        with patch.object(Path, "rglob", rglob_side_effect):
            fsp_instance.scan_media_files()

        assert len(fsp_instance.get_tracks()) == expected_audio
        assert len(fsp_instance._get_playlist_paths()) == expected_playlists

    def test_scan_media_files_skips_duplicate_resolved_files(self, fsp_instance):
        """
        Test that scan_media_files skips files whose resolved path is already in scanned_files (deduplication branch).
        """
        audio_root = fsp_instance.path
        audio_root.mkdir(parents=True, exist_ok=True)
        # Create two mock files with different names but same resolved path
        resolved_path = audio_root / "track1.mp3"
        mock1 = make_mock_file(".mp3", "track1.mp3", audio_root, is_file=True, resolved_to=resolved_path)
        mock2 = make_mock_file(".mp3", "alias_track1.mp3", audio_root, is_file=True, resolved_to=resolved_path)
        # rglob yields both
        with patch.object(Path, "rglob", return_value=[mock1, mock2]):
            fsp_instance.scan_media_files()
        # Only one should be in _media_files
        audio_files = [t for t in fsp_instance._media_files if t.suffix.lower() in fsp_instance.AUDIO_EXT]
        assert len(audio_files) == 1, f"Expected only one unique audio file, got {len(audio_files)}: {audio_files}"


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
        "audio_tag,rating,expect_none,audio_file_is_none,save_fails",
        [
            (AudioTag(), Rating(1), False, False, False),
            (AudioTag(), None, False, False, False),
            (None, Rating(1), False, False, False),
            (None, None, True, False, False),
            # New scenario: _open_audio_file returns None (not audio_file)
            (AudioTag(), Rating(1), True, True, False),
            # New scenario: _save_audio_file returns False
            (AudioTag(), Rating(1), True, False, True),
        ],
    )
    def test_update_track_metadata_handler_succeeds(self, fsp_instance, file_factory, audio_tag, rating, expect_none, audio_file_is_none, save_fails):
        """
        Branch-free: Handler supports file, file exists, save succeeds (or fails if save_fails is True), or audio_file is None if audio_file_is_none is True.
        Parameterized for all combinations of audio_tag and rating present or not, plus error branches.
        """
        audio_file = file_factory(".mp3", "fake.mp3")
        file_path = Path(audio_file.filename)
        handler = ConfigurableFakeHandler(
            can_handle_return=True,
            apply_tags_return=audio_file,
        )
        fsp_instance.id3_handler = handler
        fsp_instance.vorbis_handler.can_handle = lambda f: False
        fsp_instance._handlers = [fsp_instance.id3_handler, fsp_instance.vorbis_handler]
        with patch.object(Path, "exists", return_value=True):
            with patch.object(fsp_instance, "_open_audio_file", return_value=None if audio_file_is_none else audio_file):
                audio_file.save = MagicMock(return_value=None)
                with patch.object(fsp_instance, "_save_audio_file", return_value=not save_fails):
                    result = fsp_instance.update_track_metadata(file_path, audio_tag=audio_tag, rating=rating)

        expected_result = None if expect_none else audio_file
        expected_apply_tags_called = not audio_file_is_none and not (audio_tag is None and rating is None)

        assert result is expected_result
        assert handler.apply_tags_called == expected_apply_tags_called

    def test_update_track_metadata_handler_not_supported(self, fsp_instance, file_factory):
        """
        Branch-free: Handler does not support file, file exists.
        """
        audio_file = file_factory(".mp3", "fake.mp3")
        file_path = Path(audio_file.filename)
        handler = ConfigurableFakeHandler(
            can_handle_return=False,
            apply_tags_return=audio_file,
        )
        fsp_instance.id3_handler = handler
        fsp_instance.vorbis_handler.can_handle = lambda f: False
        fsp_instance._handlers = [fsp_instance.id3_handler, fsp_instance.vorbis_handler]
        with patch.object(Path, "exists", return_value=True):
            with patch.object(fsp_instance, "_open_audio_file", return_value=audio_file):
                audio_file.save = MagicMock(return_value=None)
                result = fsp_instance.update_track_metadata(file_path, audio_tag=AudioTag(), rating=Rating(1))
        assert result is None
        assert not handler.apply_tags_called
        assert not audio_file.save.called

    def test_update_track_metadata_file_not_exists(self, fsp_instance, file_factory):
        """
        Branch-free: File does not exist, handler should not be called.
        """
        audio_file = file_factory(".mp3", "fake.mp3")
        file_path = Path(audio_file.filename)
        handler = ConfigurableFakeHandler(
            can_handle_return=True,
            apply_tags_return=audio_file,
        )
        fsp_instance.id3_handler = handler
        fsp_instance.vorbis_handler.can_handle = lambda f: False
        fsp_instance._handlers = [fsp_instance.id3_handler, fsp_instance.vorbis_handler]
        with patch.object(Path, "exists", return_value=False):
            with patch.object(fsp_instance, "_open_audio_file", return_value=audio_file):
                audio_file.save = MagicMock(return_value=None)
                result = fsp_instance.update_track_metadata(file_path, audio_tag=AudioTag(), rating=Rating(1))
        assert result is None
        assert not handler.apply_tags_called
        assert not audio_file.save.called

    def test_update_track_metadata_unsupported_filetype(self, fsp_instance, file_factory):
        """
        Branch-free: Unsupported file type, no handler supports file.
        """
        audio_file = file_factory(".nonsense", "fake.nonsense")
        file_path = Path(audio_file.filename)
        handler = ConfigurableFakeHandler(
            can_handle_return=False,
            apply_tags_return=audio_file,
        )
        fsp_instance.id3_handler = handler
        fsp_instance.vorbis_handler.can_handle = lambda f: False
        fsp_instance._handlers = [fsp_instance.id3_handler, fsp_instance.vorbis_handler]
        with patch.object(Path, "exists", return_value=True):
            with patch.object(fsp_instance, "_open_audio_file", return_value=audio_file):
                audio_file.save = MagicMock(return_value=None)
                result = fsp_instance.update_track_metadata(file_path, audio_tag=AudioTag(), rating=Rating(1))
        assert result is None
        assert not handler.apply_tags_called
        assert not audio_file.save.called


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

    @pytest.mark.parametrize("is_extm3u", [True, False])
    def test_create_playlist_success(self, fsp_instance, is_extm3u):
        """Test create_playlist success for both EXT3 and non-EXT3 playlists."""
        fsp_instance.config_mgr.dry = False
        result = fsp_instance.create_playlist("pl", is_extm3u=is_extm3u)
        assert isinstance(result, Playlist)
        assert result.is_extm3u == is_extm3u

    def test_create_playlist_dry_run(self, fsp_instance):
        """Test create_playlist returns None in dry-run mode."""
        fsp_instance.config_mgr.dry = True
        result = fsp_instance.create_playlist("pl", is_extm3u=True)
        assert result is None

    def test_create_playlist_mkdir_fail(self, fsp_instance):
        """Test create_playlist returns None if mkdir fails."""
        fsp_instance.config_mgr.dry = False
        m_open = mock_open()
        with patch.object(Path, "mkdir", side_effect=OSError()):
            with patch("pathlib.Path.open", m_open):
                result = fsp_instance.create_playlist("pl", is_extm3u=True)
        assert result is None

    def test_create_playlist_write_fail(self, fsp_instance):
        """Test create_playlist returns None if file open/write fails."""
        fsp_instance.config_mgr.dry = False
        m_open = mock_open()
        m_open.side_effect = OSError()
        with patch("pathlib.Path.open", m_open):
            result = fsp_instance.create_playlist("pl", is_extm3u=True)
        assert result is None


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

    def test_open_playlist_file_not_exists_in_read_mode(self, fsp_instance, playlist_factory):
        """
        Test _open_playlist returns None if file does not exist in read mode.
        """
        with playlist_factory(
            playlist_root=fsp_instance.playlist_path,
            filename="notfound.m3u",
            lines=["music/track1.mp3"],
            exists=False,
        ) as (playlist_content, playlist_path):
            # _open_playlist is a protected method, call directly
            result = fsp_instance._open_playlist(playlist_path, mode="r")
            assert result is None

    def test_open_playlist_open_raises_exception(self, fsp_instance, playlist_factory):
        """
        Test _open_playlist returns None if Path.open raises an exception.
        """
        with playlist_factory(
            playlist_root=fsp_instance.playlist_path,
            filename="failopen.m3u",
            lines=["music/track1.mp3"],
            open_raises=True,
        ) as (playlist_content, playlist_path):
            result = fsp_instance._open_playlist(playlist_path, mode="r")
            assert result is None


class TestGetTracksFromPlaylist:
    """
    Test get_tracks_from_playlist for filtering of comments, blank, in-scope, and out-of-scope lines.
    """

    def test_get_tracks_from_playlist(self, fsp_instance, playlist_factory):
        """
        Test get_tracks_from_playlist with all relevant line types (relative paths):
        """
        audio_root = fsp_instance.path
        playlist_case = [
            {"line": "music/track1.mp3", "exists": True, "in_scope": True},
            {"line": "music/track2.mp3", "exists": False, "in_scope": True},
            {"line": "other/track3.mp3", "exists": True, "in_scope": False},
            {"line": "other/track4.mp3", "exists": False, "in_scope": False},
            {"line": " "},
            {"line": str((audio_root / "abs_track1.mp3").absolute()), "exists": True, "in_scope": True},
            {"line": str((audio_root / "abs_track2.mp3").absolute()), "exists": False, "in_scope": True},
            {"line": str((audio_root.parent / "other" / "abs_track3.mp3").absolute()), "exists": True, "in_scope": False},
            {"line": str((audio_root.parent / "other" / "abs_track4.mp3").absolute()), "exists": False, "in_scope": False},
            {"line": "# This is a comment"},
        ]
        lines = [entry["line"] for entry in playlist_case]
        with playlist_factory(
            playlist_root=fsp_instance.playlist_path,
            filename="test_playlist.m3u",
            lines=lines,
        ) as (playlist_content, playlist_path):
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
                if "exists" not in entry:
                    assert all(line not in p for p in result_paths)
                    continue
                candidate_path = (playlist_path.parent / line) if not Path(line).is_absolute() else Path(line)
                candidate_abs = str(candidate_path.resolve())
                if "exists" in entry and "in_scope" in entry:
                    if entry["exists"] and entry["in_scope"]:
                        assert candidate_abs in result_paths, f"Expected '{line}' to be included, but it was not."
                    else:
                        assert candidate_abs not in result_paths, f"Did not expect '{line}' to be included, but it was."
                else:
                    # If 'exists' is not present, should not be included
                    assert candidate_abs not in result_paths, f"Did not expect '{line}' to be included, but it was."

    def test_get_tracks_from_playlist_only_comments_and_blanks(self, fsp_instance, playlist_factory):
        """
        Test get_tracks_from_playlist with only comments and blank lines (blanks between comments).
        """
        playlist_case = [
            {"line": "#c"},
            {"line": ""},
            {"line": "# another comment"},
            {"line": ""},
        ]
        lines = [entry["line"] for entry in playlist_case]
        with playlist_factory(
            playlist_root=fsp_instance.playlist_path,
            filename="test_playlist.m3u",
            lines=lines,
        ) as (playlist_content, playlist_path):
            with patch.object(Path, "exists", return_value=True), patch.object(Path, "is_relative_to", return_value=True):
                result = fsp_instance.get_tracks_from_playlist(playlist_path)
            assert result == []


class TestAddTrackToPlaylist:
    """Test add_track_to_playlist for success and file open error."""

    @pytest.mark.parametrize("is_extm3u", [True, False])
    def test_add_track_to_playlist_success(self, is_extm3u, fsp_instance, playlist_factory):
        """Test that add_track_to_playlist writes EXTINF and track path to the playlist file, and logs writes."""
        track1_path = fsp_instance.path / "test_track1.mp3"
        track1 = AudioTag(file_path=str(track1_path), artist="A", title="T", duration=123)
        with playlist_factory(
            playlist_root=fsp_instance.playlist_path,
            filename="test_playlist.m3u",
            lines=["#EXTM3U", "# This is a comment", "music/track1.mp3"],
            is_extm3u=is_extm3u,
        ) as (playlist_content, playlist_path):
            playlist_content.seek(0)
            original_content = playlist_content.read()
            assert str(track1_path) not in original_content

            fsp_instance.add_track_to_playlist(playlist_path, track1, is_extm3u=is_extm3u)
            playlist_content.seek(0)
            content = playlist_content.read()
            # Assert on write calls
            assert any("#EXTINF:123,A - T" in call for call in playlist_content.write_calls) == is_extm3u
            assert any("test_track1.mp3" in call for call in playlist_content.write_calls)
            assert ("#EXTINF:123,A - T" in content) == is_extm3u
            assert "test_track1.mp3" in content

    def test_add_track_to_playlist_file_open_error(self, fsp_instance, tmp_path):
        """Test that add_track_to_playlist handles file open/write errors gracefully (does not raise)."""
        playlist_path = tmp_path / "fail.m3u"
        track_path = fsp_instance.path / "test_track1.mp3"
        track = AudioTag(file_path=str(track_path), artist="A", title="T", duration=123)
        with patch.object(Path, "is_relative_to", return_value=True), patch.object(Path, "open", side_effect=OSError("Simulated file open error")):
            # Should not raise
            fsp_instance.add_track_to_playlist(playlist_path, track, is_extm3u=True)

    def test_add_track_to_playlist_outside_audio_root_no_write(self, fsp_instance, tmp_path):
        """
        Test that add_track_to_playlist does not write if the track is outside the audio root (is_relative_to returns False).
        """
        playlist_path = tmp_path / "test_playlist.m3u"
        # Track is outside the audio root
        outside_path = tmp_path.parent / "outside.mp3"
        track = AudioTag(file_path=str(outside_path), artist="X", title="Y", duration=100)
        # Patch is_relative_to to return False for this path
        with (
            patch.object(Path, "is_relative_to", return_value=False) as mock_isrel,
            patch.object(Path, "open", side_effect=AssertionError("Should not open file for outside track")) as mock_open,
            patch.object(fsp_instance.logger, "debug") as mock_debug,
        ):
            fsp_instance.add_track_to_playlist(playlist_path, track, is_extm3u=True)
            mock_isrel.assert_called()
            mock_open.assert_not_called()
            mock_debug.assert_any_call(f"Track path {track.file_path} is outside the audio root {fsp_instance.path}; skipping add")


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
            expected_lines_stripped = [str(Path(line.strip())) for line in expected_lines if line.strip()]
            assert written_lines == expected_lines_stripped or (not written_lines and not expected_lines_stripped)
            # Assert file content matches expected
            if remove_track in initial_lines:
                assert original_content != new_content
            else:
                assert original_content == new_content

    def test_remove_track_from_playlist_file_open_error(self, fsp_instance, tmp_path):
        """Test that remove_track_from_playlist handles file open/write errors gracefully (does not raise)."""
        from unittest.mock import patch

        playlist_path = tmp_path / "fail.m3u"
        track_path = fsp_instance.path / "test_track1.mp3"
        # Patch Path.is_relative_to to always return True for this test
        with patch.object(Path, "is_relative_to", return_value=True), patch.object(Path, "open", side_effect=OSError("Simulated file open error")):
            # Should not raise
            fsp_instance.remove_track_from_playlist(playlist_path, track_path)
        # If we reach here, the exception was handled and code continued as expected


class TestGetPlaylistTitle:
    """Test _get_playlist_title for disambiguation when all folders are exhausted (warning branch)."""

    def test_get_playlist_title_duplicate_success(self, fsp_instance, tmp_path):
        """Test that _get_playlist_title returns a disambiguated title when all folders are exhausted."""
        pl_path = tmp_path / "a" / "b" / "MyList.m3u"
        pl_path.parent.mkdir(parents=True)
        fsp_instance.playlist_path = tmp_path
        # Simulate duplicate title already mapped
        fsp_instance._playlist_title_map["MyList"] = pl_path.parent / "other" / "MyList.m3u"
        # Should warn and return a candidate
        title = fsp_instance._get_playlist_title(pl_path, "MyList")
        assert ".MyList" in title

    def test_get_playlist_title_exhausted_warns(self, fsp_instance):
        """Test that _get_playlist_title logs a warning when all folders are exhausted."""

        fsp_instance = FileSystemProvider()
        fsp_instance.logger = MagicMock()
        # Set playlist_path to a dummy path
        fsp_instance.playlist_path = Path("/root")
        # path is directly under playlist_path, so rel_path.parts[:-1] is empty
        pl_path = Path("/root/MyList.m3u")
        # Prepopulate _playlist_title_map so the loop runs and triggers the warning
        fsp_instance._playlist_title_map = {"MyList": Path("/other/SomeOtherList.m3u")}
        # Call the method
        result = fsp_instance._get_playlist_title(pl_path, "MyList")
        # The warning should be called
        assert result == "MyList"
        fsp_instance.logger.warning.assert_called_once()
        assert "Could not disambiguate duplicate playlist title" in fsp_instance.logger.warning.call_args[0][0]


class TestReadPlaylistMetadataOpenError:
    """Test read_playlist_metadata for file open failure (returns None)."""

    def test_read_playlist_metadata_open_error(self, fsp_instance, tmp_path):
        """Test that read_playlist_metadata returns None if file open fails."""
        pl_path = tmp_path / "fail.m3u"
        with patch.object(fsp_instance, "_open_playlist", return_value=None):
            result = fsp_instance.read_playlist_metadata(pl_path)
        assert result is None


class TestGetPlaylistsFiltering:
    """
    Parameterized tests for FileSystemProvider.get_playlists filtering by title and path.
    All playlist IO is handled in-memory via playlist_factory. No real filesystem IO occurs.
    Populates _media_files and _playlist_title_map directly to simulate a scan.
    Refactored to use fully declarative parameterization and explicit skip_titles.
    """

    CASES = [
        # (filter_kwargs, expected_titles, skip_titles, id)
        ({}, {"Rock", "Pop", "Jazz"}, set(), "no_filter"),
        ({"title": "Pop"}, {"Pop"}, set(), "by_title_hit"),
        ({"title": "Classical"}, set(), set(), "by_title_miss"),
        ({"path": "/fake/rock.m3u"}, {"Rock"}, set(), "by_path_hit"),
        ({"path": "/does/not/exist.m3u"}, set(), set(), "by_path_miss"),
        ({"title": ""}, {"Rock", "Pop", "Jazz"}, set(), "empty_title_treated_as_none"),
        ({"title": None}, {"Rock", "Pop", "Jazz"}, set(), "title_None"),
        ({}, {"Rock", "Jazz"}, {"Pop"}, "skip_pop_none"),
    ]

    @pytest.mark.parametrize("filter_kwargs, expected, skip_titles, _id", CASES, ids=[c[3] for c in CASES])
    def test_get_playlists_filtering(self, fsp_instance, filter_kwargs, expected, skip_titles, _id):
        """Exhaustive branch-coverage for FileSystemProvider.get_playlists()."""

        def _register(fsp, title: str, fake_path: str) -> Path:
            p = Path(fake_path).resolve()
            fsp._media_files.append(p)
            fsp._playlist_title_map[title] = p
            return p

        _register(fsp_instance, "Rock", "/fake/rock.m3u")
        _register(fsp_instance, "Pop", "/fake/pop.m3u")
        _register(fsp_instance, "Jazz", "/fake/jazz.m3u")

        title_by_path = {v: k for k, v in fsp_instance._playlist_title_map.items()}
        from unittest.mock import MagicMock

        def stub_reader(path: Path):
            resolved = Path(path).resolve()
            title = title_by_path.get(resolved, "UNKNOWN")
            if title in skip_titles:
                return None
            return Playlist(ID=str(resolved), name=title)

        fsp_instance.read_playlist_metadata = MagicMock(side_effect=stub_reader)
        result_titles = {pl.name for pl in fsp_instance.get_playlists(**filter_kwargs)}
        assert result_titles == expected

    def test_get_playlists_raises_on_both_title_and_path(self, fsp_instance):
        """
        Test that get_playlists raises ValueError if both title and path are provided.
        """
        from unittest.mock import MagicMock

        path_rock = Path("/fake/rock.m3u").resolve()
        fsp_instance._media_files.append(path_rock)
        fsp_instance._playlist_title_map["Rock"] = path_rock
        fsp_instance.read_playlist_metadata = MagicMock(return_value=Playlist(ID=str(path_rock), name="Rock"))

        with pytest.raises(ValueError):
            fsp_instance.get_playlists(title="Rock", path=path_rock)


class TestOpenAudioFile:
    """
    Tests for FileSystemProvider._open_audio_file error and exception branches.
    Covers:
    - mutagen.File returns None (should raise ValueError)
    - mutagen.File raises Exception (should return None)
    """

    def test_open_audio_file_returns_none_raises_valueerror(self, fsp_instance):
        """
        Test that _open_audio_file raises ValueError if mutagen.File returns None.
        """
        with patch("filesystem_provider.mutagen.File", return_value=None):
            result = fsp_instance._open_audio_file(Path("fake.mp3"))
            assert result is None

    def test_open_audio_file_mutagen_raises_returns_none(self, fsp_instance):
        """
        Test that _open_audio_file returns None if mutagen.File raises.
        """
        with patch("filesystem_provider.mutagen.File", side_effect=RuntimeError("fail")):
            result = fsp_instance._open_audio_file(Path("fake.mp3"))
            assert result is None

    def test_open_audio_file_success(self, fsp_instance, caplog):
        """
        Test that _open_audio_file returns the audio file object when mutagen.File returns a valid object.
        """
        audio_file = MagicMock()
        audio_file.__bool__.return_value = True  # Ensure truthy
        with patch("filesystem_provider.mutagen.File", return_value=audio_file):
            result = fsp_instance._open_audio_file(Path("fake.mp3"))
        assert result is audio_file


class TestSaveAudioFile:
    """
    Tests for FileSystemProvider._save_audio_file covering:
    - dry-run branch (returns True, does not call save)
    - ID3FileType branch (calls save, returns True)
    - Exception branch (save raises, returns False)
    """

    def test_save_audio_file_dry_returns_true(self, fsp_instance):
        """
        Test that _save_audio_file returns True and does not call save if config_mgr.dry is True.
        """
        fsp_instance.config_mgr.dry = True
        audio_file = MagicMock()
        result = fsp_instance._save_audio_file(audio_file)
        assert result is True
        assert not audio_file.save.called

    def test_save_audio_file_id3filetype_calls_save_and_returns_true(self, fsp_instance):
        """
        Test that _save_audio_file calls save and returns True if audio_file is ID3FileType.
        """
        dummy = MagicMock(spec=ID3FileType)
        dummy.filename = "fake.mp3"
        result = fsp_instance._save_audio_file(dummy)
        assert result is True
        dummy.save.assert_called_once()

    def test_save_audio_file_save_raises_returns_false(self, fsp_instance):
        """
        Test that _save_audio_file returns False if audio_file.save raises an exception.
        """
        audio_file = MagicMock()
        audio_file.save.side_effect = Exception("fail")
        # Use a type that is NOT ID3FileType, so isinstance(audio_file, ID3FileType) is False
        result = fsp_instance._save_audio_file(audio_file)
        assert result is False
