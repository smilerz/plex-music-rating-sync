"""
# TODO: Add test for nested stat overwrite via set()
# TODO: Add test for starting multiple progress bars simultaneously
# TODO: Add test for bar re-creation with the same descriptionUnit tests for the StatsManager and StatusManager: stat tracking, nested key logic, and progress bar management."""

import pytest
from manager.stats_manager import StatsManager, StatusManager

def test_increment_and_get_simple_key():
    stats = StatsManager()
    stats.increment("tracks_processed")
    assert stats.get("tracks_processed") == 1

    stats.increment("tracks_processed", 2)
    assert stats.get("tracks_processed") == 3


def test_increment_and_get_nested_key():
    stats = StatsManager()
    stats.increment("VorbisHandler::inferred_scale::RATING::ZERO_TO_FIVE")
    stats.increment("VorbisHandler::inferred_scale::RATING::ZERO_TO_FIVE")
    stats.increment("VorbisHandler::inferred_scale::RATING::POPM")

    assert stats.get("VorbisHandler::inferred_scale::RATING::ZERO_TO_FIVE") == 2
    assert stats.get("VorbisHandler::inferred_scale::RATING::POPM") == 1


def test_set_value():
    stats = StatsManager()
    stats.set("custom_metric", 42)
    assert stats.get("custom_metric") == 42


def test_status_manager_lifecycle():
    sm = StatusManager()
    bar = sm.start_phase("Testing Phase", total=10)
    assert "Testing Phase" in sm.bars
    bar.update(1)
    bar.close()
    assert "Testing Phase" not in sm.bars
