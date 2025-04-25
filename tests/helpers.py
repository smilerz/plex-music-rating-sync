from typing import Literal

from ratings import Rating, RatingScale


def make_raw_rating(tag_key: str, normalized: float, frame_type: Literal["TEXT", "POPM", "VORBIS"] = "POPM") -> str | int:
    """
    Returns a raw rating value suitable for storing in a tag.

    - TEXT / FMPS_RATING / RATING → string in 0.5 increments (e.g., "3.5")
    - POPM → int in 0–255 range
    - VORBIS → stringified float 0.0–1.0
    """
    rating = Rating(normalized, scale=RatingScale.NORMALIZED)

    if frame_type == "TEXT":
        return str(rating.to_float(RatingScale.ZERO_TO_FIVE))

    elif frame_type == "VORBIS":
        return f"{normalized:.3f}"  # always float string

    elif frame_type == "POPM":
        return rating.to_int(RatingScale.POPM)

    else:
        raise ValueError(f"Unsupported frame_type: {frame_type}")
