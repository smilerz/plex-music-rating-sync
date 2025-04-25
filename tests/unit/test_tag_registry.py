"""
# TODO: Add test for display_name lookup from player name
# TODO: Add test for resolving unknown tag with fallback name
# TODO: Add test for consistent round-trip between tag_key and id3_tagUnit tests for the ID3TagRegistry: tag registration, reverse lookup, and config value generation."""

from filesystem_provider import ID3TagRegistry

def test_register_and_lookup():
    registry = ID3TagRegistry()
    key = registry.register("POPM:test@example.com", tag_key="TEST", player_name="TestPlayer")
    assert registry.get_id3_tag_for_key("TEST") == "POPM:test@example.com"
    assert registry.get_player_name_for_key("TEST") == "TestPlayer"
    assert registry.get_key_for_id3_tag("POPM:test@example.com") == "TEST"

def test_unknown_tag_defaults():
    registry = ID3TagRegistry()
    unknown_key = registry.register("POPM:custom@example.com")
    assert unknown_key.startswith("UNKNOWN")
    assert registry.get_config_value(unknown_key) == "POPM:custom@example.com"
