"""
# TODO: Add test for CHOICE strategy returning None (user skips)
# TODO: Add test for CHOICE strategy with only one rating present
# TODO: Add test for CHOICE fallback when tag registry returns unknownIntegration test for manual conflict resolution flow using CHOICE strategy."""

import pytest
from filesystem_provider import ID3Handler
from sync_items import AudioTag
from ratings import Rating
from manager import get_manager


def test_manual_conflict_choice(monkeypatch):
    handler = ID3Handler(tagging_policy={"conflict_resolution_strategy": "choice"})

    monkeypatch.setattr("builtins.input", lambda _: "2")

    track = AudioTag(ID="song.mp3", title="Track", artist="Artist", album="Album", track=1)
    raw = {
        "MEDIAMONKEY": "3.0",
        "TEXT": "4.5"
    }

    handler.tag_registry.register("POPM:no@email", tag_key="MEDIAMONKEY", player_name="MediaMonkey")
    handler.tag_registry.register("TXXX:RATING", tag_key="TEXT", player_name="Text")

    result = handler.resolve_rating(raw, track)
    assert isinstance(result, Rating)
    assert result.to_float() == Rating(4.5).to_float()


def test_invalid_then_valid_input(monkeypatch):
    inputs = iter(["x", "5", "1"])
    handler = ID3Handler(tagging_policy={"conflict_resolution_strategy": "choice"})

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    track = AudioTag(ID="song.mp3", title="Track", artist="Artist", album="Album", track=1)
    raw = {
        "MEDIAMONKEY": "3.5",
        "TEXT": "2.0"
    }

    handler.tag_registry.register("POPM:no@email", tag_key="MEDIAMONKEY", player_name="MediaMonkey")
    handler.tag_registry.register("TXXX:RATING", tag_key="TEXT", player_name="Text")

    result = handler.resolve_rating(raw, track)
    assert isinstance(result, Rating)
    assert result.to_float() == Rating(3.5).to_float()
