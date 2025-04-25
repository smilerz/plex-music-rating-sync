import pytest

from ratings import Rating, RatingScale


def test_rating_creation_normalized():
    r = Rating(0.75)
    assert r.to_float(RatingScale.NORMALIZED) == pytest.approx(0.75)


def test_rating_comparisons():
    r1 = Rating(0.8)
    r2 = Rating(0.6)
    assert r1 > r2
    assert r1 != r2


def test_rating_to_display():
    r = Rating(0.9)
    assert isinstance(r.to_display(), str)


def test_rating_to_int_scale_popm():
    r = Rating(1.0)
    val = r.to_int(RatingScale.POPM)
    assert 0 <= val <= 255


def test_rating_to_str_scale_zero_to_five():
    r = Rating(1.0)
    val = r.to_str(RatingScale.ZERO_TO_FIVE)
    assert val in {"0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"}
