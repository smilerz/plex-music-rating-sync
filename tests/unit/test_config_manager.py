"""
# TODO: Add test for invalid enum values in CLI args
# TODO: Add test for multi-valued enums like --sync with multiple values
# TODO: Add test for dry-run preventing config save
# TODO: Add test for _find_section_for_key() and config file updatesUnit tests for the ConfigManager: parsing, enum coercion, validation, and config file persistence."""

import pytest
from manager.config_manager import ConfigManager, LogLevel, PlayerType, TagWriteStrategy, ConflictResolutionStrategy

@pytest.fixture
def raw_args(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "sync_ratings.py",
        "--source", "plex",
        "--destination", "filesystem",
        "--sync", "tracks",
        "--log", "debug",
        "--cache-mode", "metadata",
        "--tag-write-strategy", "write_default",
        "--default-tag", "MEDIAMONKEY",
        "--conflict-resolution-strategy", "highest"
    ])

def test_enum_coercion_and_parsing(raw_args):
    cfg = ConfigManager()
    assert cfg.source == PlayerType.PLEX
    assert cfg.destination == PlayerType.FILESYSTEM
    assert TagWriteStrategy.WRITE_DEFAULT == cfg.tag_write_strategy
    assert ConflictResolutionStrategy.HIGHEST == cfg.conflict_resolution_strategy
    assert cfg.default_tag == "MEDIAMONKEY"
    assert LogLevel.DEBUG == cfg.log

def test_validate_config_requirements_missing_default(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "sync_ratings.py",
        "--source", "plex",
        "--destination", "filesystem",
        "--sync", "tracks",
        "--tag-write-strategy", "write_default"
    ])
    with pytest.raises(ValueError, match="default_tag must be set"):
        ConfigManager()

def test_to_dict_structure(raw_args):
    cfg = ConfigManager()
    result = cfg.to_dict()
    assert "source" in result
    assert "destination" in result
    assert "sync" in result
    assert "log" in result

# NOTE: save_config and config file mutation logic should be tested in integration (mocked I/O)
