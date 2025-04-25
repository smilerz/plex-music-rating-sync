"""
# TODO: Add test for multiple-choice selection
# TODO: Add test for help prompt behavior with '?'
# TODO: Add test for retry logic on invalid input
# TODO: Add test for text() validator rejection and retryUnit tests for the UserPrompt class: choice, yes_no, text input, and confirm_continue behavior."""

import pytest
from ui.prompt import UserPrompt

@pytest.mark.parametrize("user_input,expected", [
    ("1", "Option A"),
    ("2", "Option B"),
])
def test_choice(monkeypatch, user_input, expected):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = UserPrompt().choice("Choose one", ["Option A", "Option B"])
    assert result == expected

@pytest.mark.parametrize("user_input,expected", [
    ("y", True),
    ("n", False),
    ("", True),  # default True
])
def test_yes_no(monkeypatch, user_input, expected):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = UserPrompt().yes_no("Proceed?", default=True)
    assert result == expected

@pytest.mark.parametrize("user_input,expected", [
    ("custom text", "custom text"),
])
def test_text(monkeypatch, user_input, expected):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = UserPrompt().text("Enter value", validator=lambda x: True)
    assert result == expected

@pytest.mark.parametrize("user_input,expected", [
    ("", True),
    ("q", False),
])
def test_confirm_continue(monkeypatch, user_input, expected):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = UserPrompt().confirm_continue()
    assert result == expected
