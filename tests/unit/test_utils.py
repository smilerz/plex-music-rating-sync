"""
# TODO: Add test for padding restoration when new value is shorter than existing
# TODO: Add test for boolean stringification in presence of comment
# TODO: Add test for stringify_value with no commentUnit tests for standalone helper functions such as stringify_value()."""

from manager.config_manager import stringify_value

def test_stringify_value_preserves_comment_and_padding():
    new_value = "true"
    existing_line = "false     # keep this"
    result = stringify_value(new_value, existing_line)
    assert result.startswith("true")
    assert "# keep this" in result

def test_stringify_list_value():
    value = ["a", "b", "c"]
    result = stringify_value(value)
    assert result == "[a, b, c]"
