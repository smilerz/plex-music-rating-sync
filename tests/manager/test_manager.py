import importlib
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def initialize_manager():
    from manager import Manager

    # Save original state
    original_instance = Manager._instance
    original_initialized = Manager._initialized

    # Reset for tests in this file
    Manager._instance = None
    Manager._initialized = None

    yield

    # Restore for other tests
    Manager._instance = original_instance
    Manager._initialized = original_initialized


@pytest.fixture()
def get_manager(monkeypatch, config_args):
    import sys

    sys.argv = config_args

    monkeypatch.setattr("manager.log_manager.LogManager", MagicMock())
    monkeypatch.setattr("manager.cache_manager.CacheManager", type("CacheManager", (), {}))
    monkeypatch.setattr("manager.config_manager.ConfigManager", type("ConfigManager", (), {"log": None}))
    monkeypatch.setattr("manager.stats_manager.StatsManager", type("StatsManager", (), {}))
    monkeypatch.setattr("manager.stats_manager.StatusManager", type("StatusManager", (), {}))

    import manager as mgr_module

    importlib.reload(mgr_module)
    return mgr_module.get_manager


class TestManager:
    def test_get_manager_returns_manager_instance(self, monkeypatch, get_manager):
        """Test that get_manager() returns a Manager instance."""
        from manager import Manager

        m = get_manager()
        assert isinstance(m, Manager)

    def test_get_manager_returns_same_instance(self, monkeypatch, get_manager):
        """Test that get_manager() always returns the same singleton instance."""
        m1 = get_manager()
        assert m1._initialized is None
        m1.initialize()
        assert m1._initialized is True

        m2 = get_manager()
        assert m2._initialized is True
        m2.initialize()
        assert m1 is m2

    def test_manager_module_variable_is_singleton(self, monkeypatch, get_manager):
        """Test that the module-level 'manager' is the same as get_manager()."""

        m = get_manager()
        assert m._initialized is None
        m.initialize()
        assert m._initialized is True
        from manager import manager

        assert manager is m

    @pytest.mark.parametrize(
        "accessor,expected_type_name",
        [
            ("get_stats_manager", "StatsManager"),
            ("get_status_manager", "StatusManager"),
            ("get_cache_manager", "CacheManager"),
            ("get_config_manager", "ConfigManager"),
        ],
    )
    def test_manager_accessors_return_expected_types_after_initialize(self, accessor, expected_type_name, monkeypatch, get_manager):
        """Test that after initialize(), accessors return the expected types by name."""

        m = get_manager()
        assert m._initialized is None
        with pytest.raises(AttributeError):
            getattr(m, accessor)()

        m.initialize()

        result_after = getattr(m, accessor)()
        assert type(result_after).__name__ == expected_type_name
