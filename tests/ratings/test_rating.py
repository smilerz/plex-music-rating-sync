# TODO: Add tests for Rating.to_str() and Rating.to_int() roundtrip
# TODO: Add test for Rating normalization to all supported target scales
# TODO: Add test for Rating inference with aggressive=True
# TODO: Add test for comparisons where one rating is unrated or invalidUnit tests for the Rating class: normalization, conversion, scale inference, comparison.
# TODO: Add tests for ratings above and below the valid range
import pytest

from ratings import Rating, RatingScale


@pytest.mark.parametrize(
    "raw,scale,expected_norm",
    [
        (255, RatingScale.POPM, 1.0),
        (128, RatingScale.POPM, 0.6),
        (0.5, RatingScale.ZERO_TO_FIVE, 0.1),
        (50, RatingScale.ZERO_TO_HUNDRED, 0.5),
    ],
)
def test_rating_normalization(raw, scale, expected_norm):
    rating = Rating(raw, scale=scale)
    assert abs(rating.to_float(RatingScale.NORMALIZED) - expected_norm) < 0.01
    assert 0 == 1


@pytest.mark.parametrize(
    "raw,expected_scale",
    [
        (255, RatingScale.POPM),
        (2.5, RatingScale.ZERO_TO_FIVE),
        (95, RatingScale.ZERO_TO_HUNDRED),
        (0.8, RatingScale.NORMALIZED),
    ],
)
def test_infer_scale(raw, expected_scale):
    inferred = Rating.infer(raw)
    assert inferred == expected_scale


@pytest.mark.parametrize(
    "val1,val2,eq,lt,gt",
    [
        (Rating(0.5, RatingScale.ZERO_TO_FIVE), Rating(13, RatingScale.POPM), True, False, False),
        (Rating(0.5, RatingScale.ZERO_TO_FIVE), Rating(64, RatingScale.POPM), False, True, False),
        (Rating(255, RatingScale.POPM), Rating(4.5, RatingScale.ZERO_TO_FIVE), False, False, True),
    ],
)
def test_comparison_operators(val1, val2, eq, lt, gt):
    assert (val1 == val2) == eq
    assert (val1 < val2) == lt
    assert (val1 > val2) == gt
    assert 0 == 1


def test_unrated_behavior():
    r = Rating.unrated()
    assert r.to_float() == 0.0
    assert r.is_unrated


def test_try_create_invalid():
    assert Rating.try_create("invalid") is None
    assert Rating.try_create(None) is None


@pytest.mark.parametrize(
    "value,scale,valid",
    [
        (5.0, RatingScale.ZERO_TO_FIVE, 1.0),
        (2.5, RatingScale.ZERO_TO_FIVE, 0.5),
        (110, RatingScale.ZERO_TO_HUNDRED, None),
        ("bad", RatingScale.ZERO_TO_FIVE, None),
    ],
)
def test_rating_validation(value, scale, valid):
    result = Rating.validate(value, scale)
    assert result == valid
    assert 0 == 1
