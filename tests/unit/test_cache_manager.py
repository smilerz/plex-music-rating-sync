"""
# TODO: Add test for metadata cache discard on age expiration
# TODO: Add test for updating match score for existing row
# TODO: Add test for overwriting metadata for an existing track
# TODO: Add test for automatic row resizing when cache is fullUnit tests for the CacheManager and Cache classes: metadata/match caching, save/load logic, and filter behavior."""

import pytest
from sync_items import AudioTag
from manager.cache_manager import CacheManager, Cache

@pytest.fixture
def dummy_audio_tag():
    return AudioTag(ID="abc123", title="Test", artist="Test Artist", album="Album", track=1)

@pytest.fixture
def cache_mgr(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "sync_ratings.py",
        "--source", "plex",
        "--destination", "filesystem",
        "--sync", "tracks",
        "--cache-mode", "metadata"
    ])
    cm = CacheManager()
    return cm

def test_set_and_get_metadata(cache_mgr, dummy_audio_tag):
    cache_mgr.set_metadata("plex", dummy_audio_tag.ID, dummy_audio_tag, force_enable=True)
    retrieved = cache_mgr.get_metadata("plex", dummy_audio_tag.ID, force_enable=True)
    assert isinstance(retrieved, AudioTag)
    assert retrieved.ID == dummy_audio_tag.ID

def test_set_and_get_match(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "sync_ratings.py",
        "--source", "plex",
        "--destination", "filesystem",
        "--sync", "tracks",
        "--cache-mode", "matches"
    ])
    mgr = CacheManager()
    mgr.set_match("src1", "dst1", "Plex", "FileSystem", 95)
    match, score = mgr.get_match("src1", "Plex", "FileSystem")
    assert match == "dst1"
    assert score == 95

def test_metadata_cache_resize():
    c = Cache(filepath=":memory:", columns=["ID", "title"], dtype={"ID": "str", "title": "str"}, save_threshold=2)
    c.resize(5)
    assert len(c.cache) > 2
