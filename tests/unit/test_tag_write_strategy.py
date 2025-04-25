"""Unit tests for TagWriteStrategy enum and related write behavior assumptions."""

from manager.config_manager import TagWriteStrategy

def test_requires_default_tag():
    assert TagWriteStrategy.WRITE_DEFAULT.requires_default_tag() is True
    assert TagWriteStrategy.OVERWRITE_DEFAULT.requires_default_tag() is True
    assert TagWriteStrategy.WRITE_ALL.requires_default_tag() is False

def test_descriptions_present():
    for strategy in TagWriteStrategy:
        assert isinstance(strategy.description, str)
