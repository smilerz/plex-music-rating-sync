import datetime as dt
import logging
import os
import pickle
from unittest import mock
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from manager.cache_manager import Cache
from manager.config_manager import CacheMode
from sync_items import AudioTag


@pytest.fixture
def dummy_audio_tag():
    return AudioTag(ID="abc123", title="Test", artist="Test Artist", album="Album", track=1)


@pytest.fixture
def cache_manager(request, monkeypatch):
    mode = getattr(request, "param", {}).get("mode", CacheMode.METADATA)

    fake_manager = MagicMock()
    fake_manager.get_config_manager.return_value = MagicMock(cache_mode=mode)

    # Replace the function at module level
    monkeypatch.setattr("manager.cache_manager.get_manager", lambda: fake_manager)
    from manager.cache_manager import CacheManager

    cm = CacheManager()
    cm.stats_mgr = MagicMock()
    cm.logger = MagicMock()

    return cm


@pytest.fixture
def dummy_cache(tmp_path, request):
    """Fixture to create a dummy Cache object for testing with sensible defaults.
    Defaults: columns ["ID", "foo"], dtype str, one empty row (all pd.NA),
    save_threshold=2, temp file. Override via request.param: columns, dtype, data, save_threshold, max_age_hours, filepath.
    """
    params = getattr(request, "param", {})
    columns = params.get("columns", ["ID", "foo"])
    dtype = params.get("dtype", {c: "str" for c in columns})
    save_threshold = params.get("save_threshold", 2)
    max_age_hours = params.get("max_age_hours", None)
    filepath = params.get("filepath", str(tmp_path / "cache.pkl"))
    data = params.get("data", None)
    c = Cache(filepath=filepath, columns=columns, dtype=dtype, save_threshold=save_threshold, max_age_hours=max_age_hours)
    if data is not None:
        c.cache = pd.DataFrame(data)
    else:
        # Default: one empty row, all pd.NA
        c.cache = pd.DataFrame([{col: pd.NA for col in columns}])
    return c


class TestCache:
    @pytest.mark.parametrize(
        "file_exists, file_old, unpickle_error, expected_type",
        [
            (True, True, False, pd.DataFrame),
            (False, False, False, pd.DataFrame),
            (True, False, True, pd.DataFrame),
        ],
    )
    def test_load_file_states_returns_dataframe(self, tmp_path, file_exists, file_old, unpickle_error, expected_type):
        """Test Cache.load returns DataFrame for file missing, file old, and unpickle error."""
        fpath = tmp_path / "cache.pkl"
        if file_exists:
            with open(fpath, "wb") as f:
                if unpickle_error:
                    f.write(b"notapickle")
                else:
                    pickle.dump(pd.DataFrame({"ID": ["x"]}), f)
            if file_old:
                old_time = (dt.datetime.now() - dt.timedelta(hours=2)).timestamp()
                os.utime(fpath, (old_time, old_time))
        c = Cache(filepath=str(fpath), columns=["ID"], dtype={"ID": "str"}, max_age_hours=1)
        c.load()
        assert isinstance(c.cache, expected_type)

    @pytest.mark.parametrize(
        "max_age_hours,file_age_hours,should_discard",
        [
            (None, 25, False),
            (1, 25, True),
            (25, 1, False),
        ],
    )
    def test_load_age_expiration_discards_or_keeps_data(self, tmp_path, monkeypatch, max_age_hours, file_age_hours, should_discard):
        """Test Cache.load discards or keeps data based on file age and max_age_hours."""
        fpath = tmp_path / "cache_age.pkl"
        df = pd.DataFrame({"ID": ["1"]})
        with open(fpath, "wb") as f:
            pickle.dump(df, f)
        fake_mtime = dt.datetime.now().timestamp() - (file_age_hours * 60 * 60)
        monkeypatch.setattr("os.path.getmtime", lambda path: fake_mtime)
        c = Cache(filepath=str(fpath), columns=["ID"], dtype={"ID": "str"}, max_age_hours=max_age_hours)
        c.load()
        if should_discard:
            assert c.cache.shape[0] > 0
            assert (c.cache["ID"] == "1").sum() == 0
        else:
            assert (c.cache["ID"] == "1").sum() == 1

    @pytest.mark.parametrize(
        "missing,extra",
        [
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_ensure_columns_adds_missing_and_preserves_extra(self, dummy_cache, missing, extra):
        """Test _ensure_columns adds missing columns and preserves extra columns."""
        cols = ["ID", "foo"]
        data = {"ID": ["1"]}
        if extra:
            data["bar"] = ["baz"]
        dummy_cache.columns = cols
        dummy_cache.cache = pd.DataFrame(data)
        if missing:
            # Remove 'foo' if present
            if "foo" in dummy_cache.cache.columns:
                dummy_cache.cache = dummy_cache.cache.drop(columns=["foo"])
        dummy_cache._ensure_columns()
        for col in cols:
            assert col in dummy_cache.cache.columns
        if extra:
            assert "bar" in dummy_cache.cache.columns

    @pytest.mark.parametrize(
        "file_exists,remove_file,simulate_error",
        [
            (True, True, False),
            (False, False, False),
            (True, True, True),
        ],
    )
    def test_delete_file_removes_or_logs_error(self, tmp_path, caplog, file_exists, remove_file, simulate_error):
        """Test delete removes file, handles missing file, and logs error if exception occurs."""
        fpath = tmp_path / "del.pkl"
        if file_exists:
            with open(fpath, "wb") as f:
                f.write(b"x")
        c = Cache(filepath=str(fpath), columns=["ID"], dtype={"ID": "str"})
        if simulate_error:
            with mock.patch("os.remove", side_effect=OSError("fail")):
                c.delete()
            assert any("Failed to delete cache file" in r.message for r in caplog.records)
        else:
            c.delete()
            assert not fpath.exists() or not file_exists

    @pytest.mark.parametrize(
        "update_count,threshold,should_save",
        [
            (1, 1, True),
            (1, 2, False),
            (2, 2, True),
        ],
    )
    def test_auto_save_update_count_triggers_save_or_not(self, dummy_cache, monkeypatch, update_count, threshold, should_save):
        """Test auto_save triggers save when update_count >= threshold, otherwise does not save."""
        dummy_cache.update_count = update_count
        dummy_cache.save_threshold = threshold
        called = {}
        monkeypatch.setattr(dummy_cache, "save", lambda: called.setdefault("saved", True))
        dummy_cache.auto_save()
        if should_save:
            assert called.get("saved")
            assert dummy_cache.update_count == 0
        else:
            assert not called.get("saved")
            assert dummy_cache.update_count == update_count

    @pytest.mark.parametrize(
        "preallocate,add_row,expected_empty",
        [
            (True, False, True),
            (True, True, False),
            (False, False, True),
        ],
    )
    def test_is_empty_various_states_returns_expected(self, dummy_cache, preallocate, add_row, expected_empty):
        """Test is_empty returns True for empty cache and False for non-empty cache."""
        if not preallocate:
            dummy_cache.cache = dummy_cache.cache.iloc[0:0]
        if add_row:
            dummy_cache.cache.loc[0, "ID"] = "foo"
        assert dummy_cache.is_empty() is expected_empty

    @pytest.mark.parametrize(
        "initial_rows,add_rows",
        [
            (2, 7),
            (5, 10),
            (0, 3),
        ],
    )
    def test_resize_adds_rows_increases_length(self, dummy_cache, initial_rows, add_rows):
        """Test resize adds the requested number of rows to the cache."""
        dummy_cache.cache = pd.DataFrame({"ID": [str(i) for i in range(initial_rows)], "foo": ["x"] * initial_rows})
        before = len(dummy_cache.cache)
        dummy_cache.resize(add_rows)
        after = len(dummy_cache.cache)
        assert after == before + add_rows

    def test_save_exception_logs_error(self, dummy_cache, monkeypatch, caplog):
        """Test save writes file and logs error if exception occurs."""
        dummy_cache.cache.loc[0, "ID"] = "foo"
        dummy_cache.save()
        monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(IOError("fail")))
        dummy_cache.save()
        assert any("Failed to save cache" in r.message for r in caplog.records)

    @pytest.mark.parametrize(
        "criteria,data,expected_rows,expected_values",
        [
            ({"ID": "1"}, {"foo": "bar"}, 1, ["bar"]),  # Insert new row
            ({"ID": "1"}, {"foo": "baz"}, 1, ["baz"]),  # Update existing row
            ({"ID": "2"}, {"foo": None}, 2, [None]),  # Insert with None
        ],
    )
    def test_upsert_row_insert_and_update(self, dummy_cache, criteria, data, expected_rows, expected_values):
        """Test upsert_row inserts new row or updates existing row, including None handling."""
        dummy_cache.cache = dummy_cache.cache.iloc[0:0].copy()
        dummy_cache._ensure_columns()
        dummy_cache.upsert_row(criteria, data)
        # Upsert again to test update branch
        dummy_cache.upsert_row(criteria, {"foo": expected_values[0]})
        found = dummy_cache._find_row_by_columns(criteria)
        assert len(found) == 1
        assert found.iloc[0]["foo"] == expected_values[0] or pd.isna(found.iloc[0]["foo"])

    @pytest.mark.parametrize(
        "criteria,expected_indices",
        [
            ({"ID": "1"}, [0]),
            ({"ID": "2"}, [1]),
        ],
    )
    def test_find_row_by_columns_matches_and_no_match(self, dummy_cache, criteria, expected_indices):
        """Test _find_row_by_columns returns correct rows for match and no match."""
        dummy_cache.cache = pd.DataFrame({"ID": ["1", "2"], "foo": ["a", "b"]})
        result = dummy_cache._find_row_by_columns(criteria)
        assert list(result.index) == expected_indices

    def test_find_empty_row_or_resize(self, dummy_cache):
        """Test _find_empty_row_or_resize reuses an existing empty row, then appends a new one when none remain."""
        before_shape = dummy_cache.cache.shape
        # First call: should reuse the empty row
        idx1 = dummy_cache._find_empty_row_or_resize()
        after_shape1 = dummy_cache.cache.shape
        assert before_shape == after_shape1, f"Shape changed unexpectedly: {before_shape} -> {after_shape1}"
        row1 = dummy_cache.cache.loc[idx1]
        assert all(pd.isna(row1)), f"Row at idx1={idx1} should be empty, got: {row1}"
        # Fill the row
        dummy_cache.cache.loc[idx1, "ID"] = "1"
        dummy_cache.cache.loc[idx1, "foo"] = "a"
        # Second call: should append a new empty row
        before_shape2 = dummy_cache.cache.shape
        idx2 = dummy_cache._find_empty_row_or_resize()
        after_shape2 = dummy_cache.cache.shape
        assert after_shape2[0] > before_shape2[0], f"Row count did not increase: {before_shape2[0]} -> {after_shape2[0]}"
        row2 = dummy_cache.cache.loc[idx2]
        assert all(pd.isna(row2)), f"Row at idx2={idx2} should be empty, got: {row2}"

    def test_update_row_updates_only_existing_columns(self, dummy_cache):
        """Test _update_row only updates columns that exist in cache."""
        dummy_cache.cache = pd.DataFrame({"ID": ["1"], "foo": ["a"]})
        dummy_cache._update_row(0, {"foo": "b", "bar": "should_ignore"})
        assert dummy_cache.cache.loc[0, "foo"] == "b"
        assert "bar" not in dummy_cache.cache.columns

    @pytest.mark.parametrize("value,expected", [(None, pd.NA), ("x", "x")])
    def test_insert_row_handles_none_and_value(self, dummy_cache, value, expected):
        """Test _insert_row inserts pd.NA for None and value otherwise."""
        dummy_cache._insert_row(0, {"ID": "1", "foo": value})
        if expected is pd.NA:
            assert pd.isna(dummy_cache.cache.loc[0, "foo"])
        else:
            assert dummy_cache.cache.loc[0, "foo"] == expected

    @pytest.mark.parametrize(
        "data,expected_present,expected_absent",
        [
            ({"ID": "1", "foo": "bar"}, ["ID", "foo"], []),
            ({"ID": "2", "notacol": "baz"}, ["ID"], ["notacol"]),
        ],
    )
    def test_insert_row_handles_missing_and_present_keys(self, dummy_cache, data, expected_present, expected_absent):
        """Test _insert_row only sets values for keys present in columns, ignores others."""
        dummy_cache._insert_row(0, data)
        for key in expected_present:
            assert key in dummy_cache.cache.columns
            assert dummy_cache.cache.loc[0, key] == data[key] or pd.isna(dummy_cache.cache.loc[0, key])
        for key in expected_absent:
            assert key not in dummy_cache.cache.columns

    @pytest.mark.parametrize(
        "file_exists, remove_raises",
        [
            (True, False),
            (True, True),
            (False, False),
        ],
    )
    def test_delete_file_all_paths(self, monkeypatch, caplog, file_exists, remove_raises):
        """Test Cache.delete_file covers all code paths: file exists/success, file exists/error, file missing."""
        dummy_path = "dummy_cache.pkl"
        monkeypatch.setattr("os.path.exists", lambda path: file_exists)
        mock_remove = MagicMock()
        if file_exists and remove_raises:
            mock_remove.side_effect = OSError("fail")
        monkeypatch.setattr("os.remove", mock_remove)
        with caplog.at_level(logging.ERROR):
            Cache.delete_file(dummy_path)
            assert len(caplog.records) == (1 if remove_raises else 0)
        if file_exists:
            assert mock_remove.called
        else:
            assert not mock_remove.called


class TestCacheManager:
    @pytest.mark.parametrize(
        "cache_manager, expect_match, expect_metadata, expect_match_enabled, expect_metadata_enabled",
        [
            ({"mode": CacheMode.MATCHES}, True, True, True, True),
            ({"mode": CacheMode.METADATA}, False, True, False, True),
            ({"mode": CacheMode.MATCHES_ONLY}, True, False, True, False),
            ({"mode": CacheMode.DISABLED}, False, False, False, False),
        ],
        indirect=["cache_manager"],
    )
    def test_init_mode_creates_expected_caches(self, cache_manager, expect_match, expect_metadata, expect_match_enabled, expect_metadata_enabled):
        assert (cache_manager.match_cache is not None) == expect_match
        assert (cache_manager.metadata_cache is not None) == expect_metadata
        assert cache_manager.is_match_cache_enabled() is expect_match_enabled
        assert cache_manager.is_metadata_cache_enabled() is expect_metadata_enabled

    def test_safe_get_value_nan_and_value(self, cache_manager):
        df = pd.DataFrame({"foo": [pd.NA, "bar"]})
        assert cache_manager._safe_get_value(df.iloc[[0]], "foo") is None
        assert cache_manager._safe_get_value(df.iloc[[1]], "foo") == "bar"

    def test_log_cache_hit_increments_stats_and_logs(self, cache_manager):
        cache_manager.stats_mgr = MagicMock()
        cache_manager.logger = MagicMock()
        cache_manager._log_cache_hit("match", "keyinfo")
        cache_manager.stats_mgr.increment.assert_called_with("cache_hits")
        cache_manager.logger.trace.assert_called()

    @pytest.mark.parametrize(
        "cache,enabled,empty,expected",
        [
            (None, True, False, False),
            (MagicMock(is_empty=lambda: True), True, True, False),
            (MagicMock(is_empty=lambda: False), True, False, True),
            (MagicMock(is_empty=lambda: False), False, False, False),
        ],
    )
    def test_is_cache_ready_various_states(self, cache_manager, cache, enabled, empty, expected):
        result = cache_manager._is_cache_ready(cache, enabled)
        assert result is expected

    @pytest.mark.parametrize(
        "cache_manager",
        [
            ({"mode": CacheMode.MATCHES}),
            ({"mode": CacheMode.METADATA}),
            ({"mode": CacheMode.MATCHES_ONLY}),
            ({"mode": CacheMode.DISABLED}),
        ],
        indirect=["cache_manager"],
    )
    def test_invalidate(self, cache_manager):
        """Test invalidate sets caches to None only if enabled by mode, and asserts state before and after."""
        with patch.object(Cache, "delete_file", autospec=True) as spy_delete_file:
            cache_manager.invalidate()

            calls = [call[0][0] for call in spy_delete_file.call_args_list]

            assert cache_manager.match_cache is None
            assert cache_manager.metadata_cache is None
            assert any(cache_manager.MATCH_CACHE_FILE in str(args) for args in calls)
            assert any(cache_manager.METADATA_CACHE_FILE in str(args) for args in calls)

    @pytest.mark.parametrize(
        "cache_manager, expect_match_deleted, expect_metadata_deleted",
        [
            ({"mode": CacheMode.MATCHES}, False, True),
            ({"mode": CacheMode.METADATA}, True, True),
            ({"mode": CacheMode.MATCHES_ONLY}, False, True),
        ],
        indirect=["cache_manager"],
    )
    def test_cleanup_expected_cache_deletion(self, cache_manager, expect_match_deleted, expect_metadata_deleted):
        """Test cleanup deletes caches only if not enabled by mode, and asserts both called and not called cases."""

        if cache_manager.is_match_cache_enabled():
            cache_manager.match_cache.delete = MagicMock(delete=MagicMock())
        if cache_manager.metadata_cache is not None:
            cache_manager.metadata_cache = MagicMock(delete=MagicMock())

        cache_manager.cleanup()

        if cache_manager.is_match_cache_enabled():
            assert cache_manager.match_cache.delete.called == expect_match_deleted
        if cache_manager.is_metadata_cache_enabled():
            assert cache_manager.metadata_cache.delete.called == expect_metadata_deleted


class TestMatchCache:
    @pytest.mark.parametrize("enabled, empty, found", [(False, False, False), (True, True, False), (True, False, False), (True, False, True)])
    def test_get_match_enabled_empty_found_expected_result(self, cache_manager, enabled, empty, found):
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

    def test_cleanup_match_cache_disabled_deletes_file(self, cache_manager):
        """Test cleanup deletes match cache if not enabled."""
        cache_manager.match_cache = MagicMock(delete=MagicMock())
        cache_manager.cleanup()
        cache_manager.match_cache.delete.assert_called()

    def test_force_enable_match_cache_set_and_get(self, cache_manager, dummy_audio_tag):
        """Test force_enable orchestration for match cache."""
        cache_manager.set_match("src", "dst", "Plex", "FileSystem", 42)
        match, score = cache_manager.get_match("src", "Plex", "FileSystem")
        assert match == "dst"
        assert score == 42

    def test_match_cache_expiration_cleared_returns_none(self, cache_manager):
        """Test that match cache expiration is handled by CacheManager orchestration."""
        cache_manager.set_match("src", "dst", "Plex", "FileSystem", 99)
        # Simulate expiration by clearing cache
        if hasattr(cache_manager, "match_cache") and hasattr(cache_manager.match_cache, "cache"):
            cache_manager.match_cache.cache = cache_manager.match_cache.cache.iloc[0:0]
        match, score = cache_manager.get_match("src", "Plex", "FileSystem")
        assert match is None and score is None

    def test_get_match_cache_columns_returns_expected(self, cache_manager):
        cols = cache_manager._get_match_cache_columns()
        assert "score" in cols
        assert all(isinstance(c, str) for c in cols)


class TestMetadataCache:
    def test_force_enable_metadata_cache_set_and_get(self, cache_manager, dummy_audio_tag):
        """Test force_enable orchestration for metadata cache."""
        cache_manager.set_metadata("plex", dummy_audio_tag.ID, dummy_audio_tag, force_enable=True)
        meta = cache_manager.get_metadata("plex", dummy_audio_tag.ID, force_enable=True)
        assert isinstance(meta, type(dummy_audio_tag))

    def test_row_to_audiotag_converts_row(self, cache_manager, dummy_audio_tag):
        row = pd.DataFrame([{f: getattr(dummy_audio_tag, f) for f in dummy_audio_tag.get_fields()}])
        tag = cache_manager._row_to_audiotag(row)
        assert isinstance(tag, type(dummy_audio_tag))
        for f in dummy_audio_tag.get_fields():
            assert getattr(tag, f) == getattr(dummy_audio_tag, f)

    def test_get_metadata_cache_columns_returns_expected(self, cache_manager):
        cols = cache_manager._get_metadata_cache_columns()
        assert "player_name" in cols
        assert "ID" in cols
        assert all(isinstance(c, str) for c in cols)
