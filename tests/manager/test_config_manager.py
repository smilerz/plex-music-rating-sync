from unittest.mock import patch

import pytest

from manager.config_manager import ConflictResolutionStrategy, LogLevel, PlayerType, TagWriteStrategy


@pytest.fixture(autouse=True)
def initialize_manager(monkeypatch, tmp_path):
    # supress conftest intitilization of manager
    pass


@pytest.fixture(autouse=True)
def ConfigManager(monkeypatch):
    monkeypatch.setattr("manager.config_manager.ConfigManager.CONFIG_FILE", "./does_not_exist.ini")
    from manager.config_manager import ConfigManager

    return ConfigManager


@pytest.fixture
def config_manager(request, monkeypatch, ConfigManager):
    """Fixture to provide a ConfigManager instance with CLI args set via rational defaults, overridden by request.param['argv'] if provided.
    Accepts overrides as a dict of key:value pairs.
    """

    default_args = {
        "source": "plex",
        "destination": "filesystem",
        "sync": "tracks",
        "log": "debug",
        "cache-mode": "metadata",
        "tag-write-strategy": "write_default",
        "default-tag": "MEDIAMONKEY",
        "conflict-resolution-strategy": "highest",
    }

    override_args = getattr(request, "param", {}).get("argv", {})

    merged_args = {**default_args, **override_args}

    argv = ["sync_ratings.py"]
    for k, v in merged_args.items():
        argv.extend([f"--{k}", v])
    monkeypatch.setattr("sys.argv", argv)

    return ConfigManager()


# --- Test Classes ---
class TestConfigEnum:
    def test_enum_coercion_source_destination(self, config_manager):
        assert config_manager.source == PlayerType.PLEX
        assert config_manager.destination == PlayerType.FILESYSTEM

    def test_enum_coercion_tag_write_strategy(self, config_manager):
        assert config_manager.tag_write_strategy == TagWriteStrategy.WRITE_DEFAULT

    def test_enum_coercion_conflict_resolution_strategy(self, config_manager):
        assert config_manager.conflict_resolution_strategy == ConflictResolutionStrategy.HIGHEST

    def test_enum_coercion_log_level(self, config_manager):
        assert config_manager.log == LogLevel.DEBUG


class TestTagWriteStrategy:
    def test_enum_members_exist_and_are_unique(self):
        # Ensure all expected members exist and are unique
        members = list(TagWriteStrategy)
        names = [m.name for m in members]
        values = [m.value for m in members]
        # Check for duplicates
        assert len(set(names)) == len(names)
        assert len(set(values)) == len(values)
        # Check expected members (update as needed)
        expected = {"WRITE_ALL", "WRITE_DEFAULT", "OVERWRITE_DEFAULT"}
        actual = set(names)
        assert expected.issubset(actual)

    @pytest.mark.parametrize(
        "enum_member,expected_value",
        [
            (TagWriteStrategy.WRITE_ALL, "write_all"),
            (TagWriteStrategy.WRITE_DEFAULT, "write_default"),
            (TagWriteStrategy.OVERWRITE_DEFAULT, "overwrite_default"),
        ],
    )
    def test_enum_member_values(self, enum_member, expected_value):
        assert str(enum_member) == expected_value
        assert repr(enum_member) == f"<TagWriteStrategy.{enum_member.name}: '{expected_value}'>"

    @pytest.mark.parametrize(
        "value,expected_member",
        [
            ("write_all", TagWriteStrategy.WRITE_ALL),
            ("write_default", TagWriteStrategy.WRITE_DEFAULT),
            ("overwrite_default", TagWriteStrategy.OVERWRITE_DEFAULT),
        ],
    )
    def test_enum_from_value(self, value, expected_member):
        assert TagWriteStrategy(value) == expected_member

    def test_enum_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TagWriteStrategy("not_a_valid_strategy")


class TestConflictResolutionStrategy:
    def test_enum_members_exist_and_are_unique(self):
        # Ensure all expected members exist and are unique
        members = list(ConflictResolutionStrategy)
        names = [m.name for m in members]
        values = [m.value for m in members]
        # Check for duplicates
        assert len(set(names)) == len(names)
        assert len(set(values)) == len(values)
        # Check expected members (update as needed)
        expected = {"HIGHEST", "LOWEST"}
        actual = set(names)
        assert expected.issubset(actual)

    @pytest.mark.parametrize(
        "enum_member,expected_value",
        [
            (ConflictResolutionStrategy.HIGHEST, "highest"),
            (ConflictResolutionStrategy.LOWEST, "lowest"),
        ],
    )
    def test_enum_member_values(self, enum_member, expected_value):
        # str(enum_member) should return the value
        assert str(enum_member) == expected_value
        # repr(enum_member) should return the qualified name and value
        assert repr(enum_member) == f"<ConflictResolutionStrategy.{enum_member.name}: '{expected_value}'>"

    @pytest.mark.parametrize(
        "value,expected_member",
        [
            ("highest", ConflictResolutionStrategy.HIGHEST),
            ("lowest", ConflictResolutionStrategy.LOWEST),
        ],
    )
    def test_enum_from_value(self, value, expected_member):
        assert ConflictResolutionStrategy(value) == expected_member

    def test_enum_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ConflictResolutionStrategy("not_a_valid_strategy")


class TestParseArgs:
    def test_parse_args_valid(self, config_manager):
        assert config_manager.source == PlayerType.PLEX
        assert config_manager.destination == PlayerType.FILESYSTEM
        assert config_manager.default_tag == "MEDIAMONKEY"

    def test_parse_args_invalid_enum_value_raises(self, monkeypatch):
        argv = [
            "sync_ratings.py",
            "--source",
            "plex",
            "--destination",
            "invalid",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit):
            from manager.config_manager import ConfigManager

            ConfigManager()

    def test_parse_args_missing_required_raises(self, monkeypatch, ConfigManager):
        argv = [
            "sync_ratings.py",
            "--source",
            "plex",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit):
            ConfigManager()


class TestInitialize:
    def test_initialize_with_valid_args(self, config_manager):
        assert config_manager.default_tag == "MEDIAMONKEY"

    def test_initialize_missing_default_tag_raises(self, monkeypatch):
        argv = [
            "sync_ratings.py",
            "--source",
            "plex",
            "--destination",
            "filesystem",
            "--sync",
            "tracks",
            "--tag-write-strategy",
            "write_default",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(ValueError):
            from manager.config_manager import ConfigManager

            ConfigManager()


class TestConfigManager:
    def test_to_dict_structure(self, config_manager):
        result = config_manager.to_dict()
        assert "source" in result
        assert "destination" in result
        assert "sync" in result
        assert "log" in result

    def test_config_manager_persistence_behavior(self, config_manager, ConfigManager):
        # Mock file I/O for config save
        with patch.object(ConfigManager, "save_config", return_value=None) as mock_save:
            config_manager.save_config()
            mock_save.assert_called_once()


class TestStringifyValue:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (PlayerType.PLEX, "plex"),
            (TagWriteStrategy.WRITE_DEFAULT, "write_default"),
            ("teststring", "teststring"),
            (["a", "b"], "[a, b]"),
        ],
    )
    def test_stringify_value(self, value, expected):
        from manager.config_manager import stringify_value

        result = stringify_value(value)
        assert str(result) == expected
