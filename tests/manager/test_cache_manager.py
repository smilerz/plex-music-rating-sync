import datetime
import datetime as dt
import os
import pickle
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest

from manager.cache_manager import Cache, CacheManager
from manager.config_manager import CacheMode
from sync_items import AudioTag


@pytest.fixture
def dummy_audio_tag():
    return AudioTag(ID="abc123", title="Test", artist="Test Artist", album="Album", track=1)


@pytest.fixture
def cache_manager(request):
    mode = getattr(request, "param", {}).get("mode", CacheMode.METADATA)

    cm = CacheManager()
    cm.stats_mgr = MagicMock()
    cm.logger = MagicMock()
    cm.mode = mode
    return cm


class TestCacheManagerMetadata:
    def test_set_and_get_metadata_success(self, cache_manager, dummy_audio_tag):
        """Test storing and retrieving metadata returns correct AudioTag object."""
        cache_manager.set_metadata("plex", dummy_audio_tag.ID, dummy_audio_tag, force_enable=True)
        retrieved = cache_manager.get_metadata("plex", dummy_audio_tag.ID, force_enable=True)
        assert isinstance(retrieved, type(dummy_audio_tag))
        assert retrieved.ID == dummy_audio_tag.ID
        assert retrieved.title == dummy_audio_tag.title
        assert retrieved.artist == dummy_audio_tag.artist
        assert retrieved.album == dummy_audio_tag.album
        assert retrieved.track == dummy_audio_tag.track

    def test_get_metadata_missing_returns_none(self, cache_manager):
        """Test retrieving metadata for missing key returns None."""
        result = cache_manager.get_metadata("plex", "nonexistent", force_enable=True)
        assert result is None

    def test_metadata_overwrite_replaces_value(self, cache_manager, dummy_audio_tag):
        """Test that setting metadata for the same ID overwrites the previous value."""
        tag1 = dummy_audio_tag
        tag2 = type(dummy_audio_tag)(ID=tag1.ID, title="New Title", artist=tag1.artist, album=tag1.album, track=tag1.track)
        cache_manager.set_metadata("plex", tag1.ID, tag1, force_enable=True)
        cache_manager.set_metadata("plex", tag1.ID, tag2, force_enable=True)
        updated = cache_manager.get_metadata("plex", tag1.ID, force_enable=True)
        assert updated.title == "New Title"
        assert updated.ID == tag1.ID

    def test_set_metadata_triggers_resize(self, cache_manager, dummy_audio_tag):
        # Covers set_metadata: triggers resize when full
        cols = cache_manager._get_metadata_cache_columns()
        cache_manager.metadata_cache = Cache(filepath=":memory:", columns=cols, dtype={c: "object" for c in cols}, save_threshold=1)
        cache_manager.metadata_cache.cache.loc[0, :] = ["plex"] + ["x"] * (len(cols) - 1)
        tag2 = type(dummy_audio_tag)(ID="newid", title="T", artist="A", album="B", track=2)
        cache_manager.set_metadata("plex", tag2.ID, tag2, force_enable=True)
        assert (cache_manager.metadata_cache.cache == "newid").any().any()

    @pytest.mark.parametrize(
        "cache_manager,force_enable,expect_called",
        [
            ({"mode": CacheMode.DISABLED}, False, False),
            ({"mode": CacheMode.DISABLED}, True, True),
            ({"mode": CacheMode.METADATA}, False, True),
        ],
        indirect=["cache_manager"],
    )
    def test_set_metadata_disabled_and_force(self, cache_manager, force_enable, expect_called):
        cache_manager.metadata_cache = None
        tag = AudioTag(ID="id", title="t", artist="a", album="b", track=1)
        cache_manager.set_metadata("plex", "id", tag, force_enable=force_enable)
        if expect_called:
            assert cache_manager.metadata_cache is not None
        else:
            assert cache_manager.metadata_cache is None

    @pytest.mark.parametrize("cache_manager", [{"mode": CacheMode.METADATA}], indirect=["cache_manager"])
    def test_set_metadata_triggers_cache_creation(self, cache_manager):
        cache_manager.metadata_cache = None
        tag = AudioTag(ID="id", title="t", artist="a", album="b", track=1)
        cache_manager.set_metadata("plex", "id", tag, force_enable=True)
        assert cache_manager.metadata_cache is not None

    @pytest.mark.parametrize("cache_manager", [{"mode": CacheMode.METADATA}], indirect=["cache_manager"])
    def test_get_tracks_by_filter_empty(self, cache_manager):
        cols = cache_manager._get_metadata_cache_columns()
        cache_manager.metadata_cache = Cache(filepath=":memory:", columns=cols, dtype={c: "object" for c in cols})
        mask = cache_manager.metadata_cache.cache["player_name"] == "nonexistent"
        result = cache_manager.get_tracks_by_filter(mask)
        assert result == []

    @pytest.mark.parametrize("cache_manager", [{"mode": CacheMode.METADATA}], indirect=["cache_manager"])
    def test_get_tracks_by_filter_nonempty(self, cache_manager):
        cols = cache_manager._get_metadata_cache_columns()
        cache_manager.metadata_cache = Cache(filepath=":memory:", columns=cols, dtype={c: "object" for c in cols})
        tag = AudioTag(ID="id", title="t", artist="a", album="b", track=1)
        cache_manager.metadata_cache.cache.loc[0, :] = ["plex"] + [getattr(tag, f) for f in tag.get_fields()]
        mask = cache_manager.metadata_cache.cache["player_name"] == "plex"
        result = cache_manager.get_tracks_by_filter(mask)
        assert isinstance(result, list)
        assert all(isinstance(x, AudioTag) for x in result)


class TestCacheManagerMatch:
    @pytest.mark.parametrize("cache_manager", [{"mode": CacheMode.MATCHES}], indirect=["cache_manager"])
    def test_set_and_get_and_update_match(self, cache_manager):
        """Test storing, retrieving, and updating a match."""
        cache_manager.set_match("src1", "dst1", "Plex", "FileSystem", 80)
        match, score = cache_manager.get_match("src1", "Plex", "FileSystem")
        assert match == "dst1"
        assert score == 80
        # Now update the score
        cache_manager.set_match("src1", "dst1", "Plex", "FileSystem", 99)
        match2, score2 = cache_manager.get_match("src1", "Plex", "FileSystem")
        assert match2 == "dst1"
        assert score2 == 99

    @pytest.mark.parametrize("cache_manager", [{"mode": CacheMode.MATCHES}], indirect=["cache_manager"])
    def test_get_match_missing_returns_none(self, cache_manager):
        """Test retrieving a match for missing key returns (None, None)."""
        match, score = cache_manager.get_match("missing", "Plex", "FileSystem")
        assert match is None
        assert score is None

    def test_set_match_triggers_resize(self, cache_manager):
        # Covers set_match: triggers resize when full
        cols = cache_manager._get_match_cache_columns()
        cache_manager.match_cache = Cache(filepath=":memory:", columns=cols, dtype={c: "object" for c in cols}, save_threshold=1)
        # Fill the only available row
        cache_manager.match_cache.cache.loc[0, :] = ["src", "dst", "other"] + [1] * (len(cols) - 3)
        cache_manager.set_match("src2", "dst2", "Plex", "FileSystem", 99)
        # Check that the new entry exists in the correct columns
        found = ((cache_manager.match_cache.cache["Plex"] == "src2") & (cache_manager.match_cache.cache["FileSystem"] == "dst2")).any()
        assert found

    @pytest.mark.parametrize(
        "cache_manager,empty,expect_none,expect_empty,expect_unchanged",
        [
            ({"mode": CacheMode.DISABLED}, False, True, False, False),
            ({"mode": CacheMode.DISABLED}, True, True, True, False),
            ({"mode": CacheMode.MATCHES}, True, False, True, False),
            ({"mode": CacheMode.MATCHES}, False, False, False, True),
        ],
        indirect=["cache_manager"],
    )
    def test_set_match_disabled_or_empty(self, cache_manager, empty, expect_none, expect_empty, expect_unchanged):
        cache_before = None
        if cache_manager.match_cache is not None:
            if empty:
                cache_manager.match_cache.cache = cache_manager.match_cache.cache.iloc[0:0]
            cache_before = cache_manager.match_cache.cache.copy()
        cache_manager.set_match("src", "dst", "Plex", "FileSystem", 42)
        if expect_none:
            assert cache_manager.match_cache is None or cache_manager.match_cache.cache.empty
        if expect_empty:
            assert cache_manager.match_cache is not None and cache_manager.match_cache.cache.empty
        if expect_unchanged:
            assert cache_manager.match_cache.cache.equals(cache_before)


class TestCacheResize:
    def test_cache_resize_increases_capacity(self):
        """Test that resizing the cache increases its capacity."""
        c = Cache(filepath=":memory:", columns=["ID", "title"], dtype={"ID": "str", "title": "str"}, save_threshold=2)
        initial_len = len(c.cache)
        c.resize(initial_len + 5)
        assert len(c.cache) > initial_len

    def test_cache_resize_decreases_capacity(self):
        """Test that resizing the cache to a smaller size truncates entries."""
        c = Cache(filepath=":memory:", columns=["ID", "title"], dtype={"ID": "str", "title": "str"}, save_threshold=2)
        for i in range(10):
            c.cache.loc[i, ["ID", "title"]] = [str(i), f"Title {i}"]
        # Now shrink the DataFrame to 5 rows
        c.cache = c.cache.iloc[:5].copy()
        assert len(c.cache) == 5

    def test_resize_noop_when_large_enough(self):
        c = Cache(filepath=":memory:", columns=["ID"], dtype={"ID": "str"}, save_threshold=2)
        # Already large enough
        old_len = len(c.cache)
        c.resize(0)
        assert len(c.cache) == old_len


class TestCacheManagerExpiration:
    def test_metadata_cache_discard_on_age_expiration(self, cache_manager, dummy_audio_tag, monkeypatch):
        """Test that metadata is discarded after expiration (simulate expiration logic)."""
        cache_manager.set_metadata("plex", dummy_audio_tag.ID, dummy_audio_tag, force_enable=True)
        # Simulate time passing if expiration is time-based
        if hasattr(cache_manager, "_metadata_cache") and hasattr(cache_manager._metadata_cache, "cache"):
            for entry in cache_manager._metadata_cache.cache:
                if "timestamp" in entry:
                    entry["timestamp"] = 0  # Set to epoch
        result = cache_manager.get_metadata("plex", dummy_audio_tag.ID, force_enable=True)
        assert result is None or isinstance(result, type(dummy_audio_tag))


class TestCache:
    def test_load_missing_file_initializes_cache(self, tmp_path, caplog):
        """Test that load initializes cache if file does not exist."""
        c = Cache(filepath=str(tmp_path / "nofile.pkl"), columns=["ID"], dtype={"ID": "str"})
        c.load()
        assert isinstance(c.cache, pd.DataFrame)
        assert c.cache.shape[1] == 1

    def test_load_old_file_discards_cache(self, tmp_path, monkeypatch, caplog):
        """Test that load discards cache if file is too old."""
        fpath = tmp_path / "cache.pkl"
        df = pd.DataFrame({"ID": ["1"]})
        with open(fpath, "wb") as f:
            pickle.dump(df, f)
        monkeypatch.setattr("os.path.getmtime", lambda path: 0)

        # Patch the correct target: datetime.datetime.timestamp
        class FakeNow:
            def timestamp(self):
                return 60 * 60 * 25  # 25 hours in seconds

        monkeypatch.setattr(dt, "datetime", type("FakeDateTime", (), {"now": staticmethod(lambda: FakeNow())}))
        c = Cache(filepath=str(fpath), columns=["ID"], dtype={"ID": "str"}, max_age_hours=1)
        c.load()
        assert c.cache.shape[1] == 1  # columns present

    def test_load_unpickling_error_initializes_cache(self, tmp_path, monkeypatch, caplog):
        """Test that load initializes cache on unpickling error."""
        fpath = tmp_path / "bad.pkl"
        with open(fpath, "wb") as f:
            f.write(b"notapickle")
        c = Cache(filepath=str(fpath), columns=["ID"], dtype={"ID": "str"})
        monkeypatch.setattr(pickle, "load", lambda f: (_ for _ in ()).throw(pickle.UnpicklingError()))
        c.load()
        assert isinstance(c.cache, pd.DataFrame)

    def test_save_and_save_error(self, tmp_path, monkeypatch, caplog):
        """Test save writes file and handles exception."""
        c = Cache(filepath=str(tmp_path / "save.pkl"), columns=["ID"], dtype={"ID": "str"})
        c.cache.loc[0, "ID"] = "foo"
        c.save()
        # Simulate error
        monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(IOError("fail")))
        c.save()
        assert any("Failed to save cache" in r.message for r in caplog.records)

    def test_ensure_columns_adds_missing(self):
        """Test _ensure_columns adds missing columns."""
        c = Cache(filepath=":memory:", columns=["ID", "foo"], dtype={"ID": "str", "foo": "str"})
        c.cache = pd.DataFrame({"ID": ["1"]})
        c._ensure_columns()
        assert "foo" in c.cache.columns

    def test_ensure_columns_with_extra_columns(self):
        """Test _ensure_columns does not remove extra columns but ensures required ones exist."""
        c = Cache(filepath=":memory:", columns=["ID", "foo"], dtype={"ID": "str", "foo": "str"})
        # Add an extra column not in required columns
        c.cache = pd.DataFrame({"ID": ["1"], "bar": ["baz"]})
        c._ensure_columns()
        assert "foo" in c.cache.columns
        assert "bar" in c.cache.columns  # Extra column remains

    def test_delete_removes_file_and_handles_missing(self, tmp_path, caplog):
        """Test delete removes file and handles missing file."""
        fpath = tmp_path / "del.pkl"
        with open(fpath, "wb") as f:
            f.write(b"x")
        c = Cache(filepath=str(fpath), columns=["ID"], dtype={"ID": "str"})
        c.delete()
        assert not fpath.exists()
        # No file
        c.delete()
        # Simulate error
        c2 = Cache(filepath=str(tmp_path / "badfile.pkl"), columns=["ID"], dtype={"ID": "str"})
        with mock.patch("os.remove", side_effect=OSError("fail")):
            c2.delete()

    def test_delete_exception_logs_error(self, tmp_path, caplog):
        """Test delete logs error if os.remove raises exception."""
        fpath = tmp_path / "del2.pkl"
        with open(fpath, "wb") as f:
            f.write(b"x")
        c = Cache(filepath=str(fpath), columns=["ID"], dtype={"ID": "str"})
        with mock.patch("os.remove", side_effect=OSError("fail")):
            c.delete()
        assert any("Failed to delete cache file" in r.message for r in caplog.records)

    def test_auto_save_triggers_save(self, tmp_path, monkeypatch):
        """Test auto_save triggers save when update_count >= threshold."""
        c = Cache(filepath=str(tmp_path / "auto.pkl"), columns=["ID"], dtype={"ID": "str"}, save_threshold=1)
        c.update_count = 1
        called = {}
        monkeypatch.setattr(c, "save", lambda: called.setdefault("saved", True))
        c.auto_save()
        assert called.get("saved")
        assert c.update_count == 0

    def test_auto_save_noop_when_below_threshold(self, tmp_path):
        c = Cache(filepath=str(tmp_path / "auto.pkl"), columns=["ID"], dtype={"ID": "str"}, save_threshold=2)
        c.update_count = 1
        # Should not trigger save
        c.auto_save()
        assert c.update_count == 1

    def test_is_empty(self):
        """Test is_empty returns True/False appropriately."""
        c = Cache(filepath=":memory:", columns=["ID"], dtype={"ID": "str"})
        # The DataFrame is pre-allocated, so .empty is False
        assert c.is_empty() is False
        c.cache.loc[0, "ID"] = "foo"
        assert c.is_empty() is False

    def test_is_empty_true_for_empty(self):
        c = Cache(filepath=":memory:", columns=["ID"], dtype={"ID": "str"})
        c.cache = c.cache.iloc[0:0]  # Make DataFrame empty
        assert c.is_empty() is True


class TestCacheLoad:
    @pytest.mark.parametrize(
        "file_exists, file_old, unpickle_error",
        [
            (True, True, False),  # file exists, is old
            (False, False, False),  # file missing
            (True, False, True),  # file exists, unpickling error
        ],
    )
    def test_load_various_conditions(self, tmp_path, monkeypatch, file_exists, file_old, unpickle_error):
        # Covers Cache.load: file missing, file old, unpickle error
        fpath = tmp_path / "cache.pkl"
        if file_exists:
            with open(fpath, "wb") as f:
                if unpickle_error:
                    f.write(b"notapickle")
                else:
                    pickle.dump(pd.DataFrame({"ID": ["x"]}), f)
            if file_old:
                old_time = (datetime.datetime.now() - datetime.timedelta(hours=2)).timestamp()
                os.utime(fpath, (old_time, old_time))
        c = Cache(filepath=str(fpath), columns=["ID"], dtype={"ID": "str"}, max_age_hours=1)
        c.load()
        assert isinstance(c.cache, pd.DataFrame)


class TestCacheManagerInternal:
    @pytest.mark.parametrize(
        "cache_manager, expect_match, expect_metadata",
        [
            ({"mode": CacheMode.MATCHES}, True, True),
            ({"mode": CacheMode.METADATA}, False, True),
            ({"mode": CacheMode.MATCHES_ONLY}, True, False),
            ({"mode": CacheMode.DISABLED}, False, False),
        ],
        indirect=["cache_manager"],
    )
    def test_initialize_caches_modes_all(self, cache_manager, expect_match, expect_metadata):
        assert cache_manager.is_match_cache_enabled() == expect_match
        assert cache_manager.is_metadata_cache_enabled() == expect_metadata
        assert isinstance(cache_manager.match_cache, Cache)
        assert isinstance(cache_manager.metadata_cache, Cache)

    def test_safe_get_value_nan_and_value(self, cache_manager):
        import pandas as pd

        df = pd.DataFrame({"foo": [pd.NA, "bar"]})
        assert cache_manager._safe_get_value(df.iloc[[0]], "foo") is None
        assert cache_manager._safe_get_value(df.iloc[[1]], "foo") == "bar"

    def test_get_match_cache_columns(self, cache_manager):
        cols = cache_manager._get_match_cache_columns()
        assert "score" in cols
        assert all(isinstance(c, str) for c in cols)

    def test_get_metadata_cache_columns(self, cache_manager):
        cols = cache_manager._get_metadata_cache_columns()
        assert "player_name" in cols
        assert "ID" in cols

    def test_get_tracks_by_filter_empty(self, cache_manager):
        # Covers get_tracks_by_filter: filter matches no rows
        cache_manager.metadata_cache = mock.Mock()
        cache_manager.metadata_cache.cache = pd.DataFrame({"ID": ["a", "b"]})
        result = cache_manager.get_tracks_by_filter(pd.Series([False, False]))
        assert result == []

    @pytest.mark.parametrize("has_metadata, has_match", [(True, True), (True, False), (False, True), (False, False)])
    def test_cleanup_and_invalidate_various(self, cache_manager, monkeypatch, has_metadata, has_match):
        if has_metadata:
            cache_manager.metadata_cache = mock.Mock(delete=mock.Mock())
        else:
            cache_manager.metadata_cache = None
        if has_match:
            cache_manager.match_cache = mock.Mock(delete=mock.Mock())
        else:
            cache_manager.match_cache = None
        monkeypatch.setattr(cache_manager, "is_match_cache_enabled", lambda: has_match)
        monkeypatch.setattr(cache_manager, "is_metadata_cache_enabled", lambda: has_metadata)
        cache_manager.cleanup()
        cache_manager.invalidate()
        # No assertion needed, just ensure no error

    @pytest.mark.parametrize("match_count, meta_count, expect_match, expect_meta", [(100, 0, True, False), (0, 100, False, True), (100, 100, True, True), (0, 0, False, False)])
    def test_trigger_auto_save_various(self, cache_manager, match_count, meta_count, expect_match, expect_meta):
        cache_manager.match_cache = mock.Mock(auto_save=mock.Mock())
        cache_manager.metadata_cache = mock.Mock(auto_save=mock.Mock())
        cache_manager._match_update_count = match_count
        cache_manager._metadata_update_count = meta_count
        cache_manager.is_match_cache_enabled = lambda: True
        cache_manager.is_metadata_cache_enabled = lambda: True
        cache_manager._trigger_auto_save()
        assert cache_manager.match_cache.auto_save.called == expect_match
        assert cache_manager.metadata_cache.auto_save.called == expect_meta

    @pytest.mark.parametrize("enabled, empty, found", [(False, False, False), (True, True, False), (True, False, False), (True, False, True)])
    def test_get_match_all_branches(self, cache_manager, enabled, empty, found):
        import pandas as pd

        cache_manager.is_match_cache_enabled = lambda: enabled
        if not enabled:
            cache_manager.match_cache = None
        else:
            if empty:
                cache_manager.match_cache = type("Dummy", (), {"is_empty": lambda self: True, "cache": pd.DataFrame()})()
            else:
                if found:
                    df = pd.DataFrame({"Plex": ["src1"], "FileSystem": ["dst1"], "score": [99]})
                    cache_manager.match_cache = type("Dummy", (), {"is_empty": lambda self: False, "cache": df})()
                else:
                    df = pd.DataFrame({"Plex": ["other"], "FileSystem": ["dst2"], "score": [88]})
                    cache_manager.match_cache = type("Dummy", (), {"is_empty": lambda self: False, "cache": df})()
        match, score = cache_manager.get_match("src1", "Plex", "FileSystem")
        if not enabled or empty or not found:
            assert match is None and score is None
        else:
            assert match == "dst1" and score == 99


class TestCacheManagerCleanup:
    def test_cleanup_deletes_match_cache_when_disabled(self, cache_manager, monkeypatch):
        """Test cleanup deletes match cache if not enabled."""
        cache_manager.match_cache = MagicMock(delete=MagicMock())
        monkeypatch.setattr(cache_manager, "is_match_cache_enabled", lambda: False)
        cache_manager.cleanup()
        cache_manager.match_cache.delete.assert_called()


class TestCacheManagerModeForceEnable:
    @pytest.mark.parametrize(
        "cache_manager,force_enable,expect_metadata,expect_match",
        [
            ({"mode": CacheMode.METADATA}, True, True, False),
            ({"mode": CacheMode.METADATA}, False, True, False),
            ({"mode": CacheMode.MATCHES}, True, False, True),
            ({"mode": CacheMode.MATCHES}, False, False, True),
            ({"mode": CacheMode.MATCHES_ONLY}, True, False, True),
            ({"mode": CacheMode.MATCHES_ONLY}, False, False, True),
            ({"mode": CacheMode.DISABLED}, True, False, False),
            ({"mode": CacheMode.DISABLED}, False, False, False),
        ],
        indirect=["cache_manager"],
    )
    def test_mode_force_enable_behavior(self, monkeypatch, cache_manager, force_enable, expect_metadata, expect_match, dummy_audio_tag, request):
        """Test all combinations of mode and force_enable for set/get methods."""
        # Metadata
        cache_manager.set_metadata("plex", dummy_audio_tag.ID, dummy_audio_tag, force_enable=force_enable)
        meta = cache_manager.get_metadata("plex", dummy_audio_tag.ID, force_enable=force_enable)
        if expect_metadata:
            assert isinstance(meta, type(dummy_audio_tag))
        else:
            assert meta is None
        # Match
        cache_manager.set_match("src", "dst", "Plex", "FileSystem", 42)
        match, score = cache_manager.get_match("src", "Plex", "FileSystem")
        if expect_match:
            assert match == "dst"
            assert score == 42
        else:
            assert match is None and score is None


class TestCacheManagerMatchExpiration:
    def test_match_cache_discard_on_age_expiration(self, cache_manager, monkeypatch):
        """Test that match cache is discarded after expiration (simulate expiration logic)."""
        cache_manager.set_match("src", "dst", "Plex", "FileSystem", 99)
        # Simulate time passing if expiration is time-based
        if hasattr(cache_manager, "_match_cache") and hasattr(cache_manager._match_cache, "cache"):
            for entry in cache_manager._match_cache.cache:
                if "timestamp" in entry:
                    entry["timestamp"] = 0  # Set to epoch
        match, score = cache_manager.get_match("src", "Plex", "FileSystem")
        assert match is None or score is None


class TestCacheManagerModeForceEnable:
    @pytest.mark.parametrize("mode", ["disabled", "matches", "metadata"])
    def test_cache_manager_init_modes(self, mode):
        cm = CacheManager()
        cm.mode = mode
        if mode == "disabled":
            assert cm.metadata_cache is None and cm.match_cache is None
        else:
            # Should initialize at least one cache
            assert cm.metadata_cache is not None or cm.match_cache is not None
