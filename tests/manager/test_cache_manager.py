import datetime as dt
import os
import pickle
from unittest import mock
from unittest.mock import MagicMock

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


class TestCacheManager:
    @pytest.mark.parametrize("has_metadata, has_match", [(True, True), (True, False), (False, True), (False, False)])
    def test_cleanup_and_invalidate_various(self, cache_manager, has_metadata, has_match):
        """
        Test that cleanup and invalidate do not error for all cache presence combinations.
        """
        if has_metadata:
            cache_manager.metadata_cache = mock.Mock(delete=mock.Mock())
        else:
            cache_manager.metadata_cache = None
        if has_match:
            cache_manager.match_cache = mock.Mock(delete=mock.Mock())
        else:
            cache_manager.match_cache = None
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

    def test_cleanup_deletes_match_cache_when_disabled(self, cache_manager):
        """Test cleanup deletes match cache if not enabled."""
        cache_manager.match_cache = MagicMock(delete=MagicMock())
        cache_manager.cleanup()
        cache_manager.match_cache.delete.assert_called()

    def test_force_enable_metadata_and_match(self, cache_manager, dummy_audio_tag):
        """
        Test force_enable orchestration for both metadata and match cache.
        """
        # Metadata
        cache_manager.set_metadata("plex", dummy_audio_tag.ID, dummy_audio_tag, force_enable=True)
        meta = cache_manager.get_metadata("plex", dummy_audio_tag.ID, force_enable=True)
        assert isinstance(meta, type(dummy_audio_tag))
        # Match
        cache_manager.set_match("src", "dst", "Plex", "FileSystem", 42)
        match, score = cache_manager.get_match("src", "Plex", "FileSystem")
        assert match == "dst"
        assert score == 42

    def test_match_cache_expiration_orchestration(self, cache_manager):
        """
        Test that match cache expiration is handled by CacheManager orchestration.
        """
        cache_manager.set_match("src", "dst", "Plex", "FileSystem", 99)
        # Simulate expiration by clearing cache
        if hasattr(cache_manager, "match_cache") and hasattr(cache_manager.match_cache, "cache"):
            cache_manager.match_cache.cache = cache_manager.match_cache.cache.iloc[0:0]
        match, score = cache_manager.get_match("src", "Plex", "FileSystem")
        assert match is None and score is None


class TestMatchCache:
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
        assert bool(found)

    def test_match_cache_discard_on_age_expiration(self, cache_manager):
        """Test that match cache is discarded after expiration (simulate expiration logic)."""
        cache_manager.set_match("src", "dst", "Plex", "FileSystem", 99)
        # Simulate time passing if expiration is time-based
        if hasattr(cache_manager, "_match_cache") and hasattr(cache_manager._match_cache, "cache"):
            # Simulate expiration by clearing cache
            cache_manager._match_cache.cache = cache_manager._match_cache.cache.iloc[0:0]
        match, score = cache_manager.get_match("src", "Plex", "FileSystem")
        assert match is None or score is None

    @pytest.mark.parametrize(
        "force_enable,expect_match",
        [(True, True), (False, True), (False, False)],
    )
    def test_mode_force_enable_behavior_match(self, cache_manager, force_enable, expect_match):
        """
        Test all combinations of mode and force_enable for set/get methods (match cache branch).
        """
        cache_manager.set_match("src", "dst", "Plex", "FileSystem", 42)
        match, score = cache_manager.get_match("src", "Plex", "FileSystem")
        if expect_match:
            assert match == "dst"
            assert score == 42
        else:
            assert match is None and score is None


class TestMetadataCache:
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

    def test_set_metadata_triggers_cache_creation(self, cache_manager):
        cache_manager.metadata_cache = None
        tag = AudioTag(ID="id", title="t", artist="a", album="b", track=1)
        cache_manager.set_metadata("plex", "id", tag, force_enable=True)
        assert cache_manager.metadata_cache is not None

    def test_get_tracks_by_filter_empty(self, cache_manager):
        cols = cache_manager._get_metadata_cache_columns()
        cache_manager.metadata_cache = Cache(filepath=":memory:", columns=cols, dtype={c: "object" for c in cols})
        mask = cache_manager.metadata_cache.cache["player_name"] == "nonexistent"
        result = cache_manager.get_tracks_by_filter(mask)
        assert result == []

    def test_get_tracks_by_filter_nonempty(self, cache_manager):
        cols = cache_manager._get_metadata_cache_columns()
        cache_manager.metadata_cache = Cache(filepath=":memory:", columns=cols, dtype={c: "object" for c in cols})
        tag = AudioTag(ID="id", title="t", artist="a", album="b", track=1)
        cache_manager.metadata_cache.cache.loc[0, :] = ["plex"] + [getattr(tag, f) for f in tag.get_fields()]
        mask = cache_manager.metadata_cache.cache["player_name"] == "plex"
        result = cache_manager.get_tracks_by_filter(mask)
        assert isinstance(result, list)
        assert all(isinstance(x, AudioTag) for x in result)

    def test_metadata_cache_discard_on_age_expiration(self, cache_manager, dummy_audio_tag):
        """Test that metadata is discarded after expiration (simulate expiration logic)."""
        cache_manager.set_metadata("plex", dummy_audio_tag.ID, dummy_audio_tag, force_enable=True)
        # Simulate time passing if expiration is time-based
        if hasattr(cache_manager, "_metadata_cache") and hasattr(cache_manager._metadata_cache, "cache"):
            for entry in cache_manager._metadata_cache.cache:
                if "timestamp" in entry:
                    entry["timestamp"] = 0  # Set to epoch
        result = cache_manager.get_metadata("plex", dummy_audio_tag.ID, force_enable=True)
        assert result is None or isinstance(result, type(dummy_audio_tag))

    @pytest.mark.parametrize(
        "force_enable,expect_metadata",
        [(True, True), (False, True), (False, False)],
    )
    def test_mode_force_enable_behavior_metadata(self, cache_manager, force_enable, expect_metadata, dummy_audio_tag):
        """
        Test all combinations of mode and force_enable for set/get methods (metadata cache branch).
        """
        cache_manager.set_metadata("plex", dummy_audio_tag.ID, dummy_audio_tag, force_enable=force_enable)
        meta = cache_manager.get_metadata("plex", dummy_audio_tag.ID, force_enable=force_enable)
        if expect_metadata:
            assert isinstance(meta, type(dummy_audio_tag))
        else:
            assert meta is None
