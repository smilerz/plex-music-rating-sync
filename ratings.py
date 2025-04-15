from enum import Enum
from typing import Optional, Union


class RatingScale(Enum):
    NORMALIZED = 1.0
    ZERO_TO_FIVE = 5.0
    ZERO_TO_TEN = 10.0
    ZERO_TO_HUNDRED = 100.0
    POPM = 255


POPM_MAP = [
    (0.0, 0),
    (0.1, 13),
    (0.2, 32),
    (0.3, 54),
    (0.4, 64),
    (0.5, 118),
    (0.6, 128),
    (0.7, 186),
    (0.8, 196),
    (0.9, 242),
    (1.0, 255),
]


class Rating:
    def __init__(self, raw: Union[float, str], scale: Optional[RatingScale] = None, *, aggressive: bool = False) -> None:
        self.raw = raw
        value = float(self.raw)
        self.scale = scale or self.infer(value, aggressive=aggressive)
        if not self.scale:
            raise ValueError(f"Unable to infer rating scale from value: {raw}")

        self._normalized = self._normalize(value, self.scale)

    @property
    def is_unrated(self) -> bool:
        return self._normalized <= 0.0

    @staticmethod
    def infer(value: float, aggressive: bool = False) -> Optional[RatingScale]:
        popm_values = {byte for _, byte in POPM_MAP[1:]}  # exclude zero
        if value in popm_values:
            return RatingScale.POPM
        elif 0 < value <= 1:
            return RatingScale.NORMALIZED
        elif 1 < value <= 5:
            return RatingScale.ZERO_TO_FIVE
        elif value > 10:
            return RatingScale.ZERO_TO_HUNDRED
        elif aggressive:
            if value in {0.5, 1.0}:
                return RatingScale.NORMALIZED
            if value == 5.0:
                return RatingScale.ZERO_TO_FIVE
        return None

    @classmethod
    def try_create(cls, raw: Union[str, float], scale: Optional[RatingScale] = None, *, aggressive: bool = False) -> Optional["Rating"]:
        if raw is None:
            return None
        try:
            return cls(raw, scale=scale, aggressive=aggressive)
        except ValueError:
            return None

    def to_float(self, scale: Optional[RatingScale] = None) -> float:
        """Return this rating as a float in the given scale (default: self.scale)."""
        target_scale = scale or self.scale
        if target_scale == RatingScale.POPM:
            return self._to_popm(self._normalized)
        return round(self._normalized * target_scale.value, 3)

    def to_str(self, scale: Optional[RatingScale] = None) -> str:
        """Convert the rating to a string for writing to file (e.g., POPM, Vorbis)."""
        target = scale or self.scale
        return str(self.to_float(target))

    def to_int(self, scale: Optional[RatingScale] = None) -> int:
        """Return the rating as a rounded int in the target scale (e.g., for POPM)."""
        return round(self.to_float(scale))

    def to_display(self, scale: RatingScale = RatingScale.ZERO_TO_FIVE) -> str:
        """Return a user-friendly rating string in a 5-star style, e.g., '4.5 / 5.0'."""
        value = self.to_float(scale)
        return f"{round(value, 1)}"

    def update(self, other: "Rating") -> None:
        """Update this rating with another's normalized value."""
        self._normalized = other._normalized

    @staticmethod
    def validate(value: Union[str, float], scale: RatingScale = RatingScale.ZERO_TO_FIVE) -> Optional[float]:
        """
        Validate that a value is within the expected scale and aligned to 0.5 steps (if applicable).
        Returns the normalized value or None if invalid.
        """
        try:
            val = float(value)
            if 0 <= val <= scale.value and (val * 2).is_integer():
                return round(val / scale.value, 3)
        except (ValueError, TypeError):
            pass
        return None

    def _normalize(self, value: float, scale: RatingScale) -> float:
        if scale == RatingScale.POPM:
            return self._from_popm(value)
        return round(value / scale.value, 3)

    def _from_popm(self, byte: float) -> float:
        if byte <= 0:
            return 0.0
        best_match = min(POPM_MAP[1:], key=lambda pair: abs(pair[1] - byte))
        return best_match[0]

    def _to_popm(self, normalized: float) -> int:
        for rating, byte in POPM_MAP[1:]:
            if normalized <= rating:
                return byte
        return 255

    def __float__(self):
        raise NotImplementedError("Use `to_float()` with an optional scale parameter.")

    def __int__(self):
        raise NotImplementedError("Use `to_int()` with an optional scale parameter.")

    def __hash__(self):
        return hash(self._normalized)

    def __eq__(self, other: Union["Rating", float]) -> bool:
        if isinstance(other, Rating):
            return self._normalized == other._normalized
        elif isinstance(other, (int, float)):
            return self._normalized == other
        return NotImplemented

    def __lt__(self, other: Union["Rating", float]) -> bool:
        if isinstance(other, Rating):
            return self._normalized < other._normalized
        elif isinstance(other, (int, float)):
            return self._normalized < other
        return NotImplemented

    def __le__(self, other: Union["Rating", float]) -> bool:
        if isinstance(other, Rating):
            return self._normalized <= other._normalized
        elif isinstance(other, (int, float)):
            return self._normalized <= other
        return NotImplemented

    def __gt__(self, other: Union["Rating", float]) -> bool:
        if isinstance(other, Rating):
            return self._normalized > other._normalized
        elif isinstance(other, (int, float)):
            return self._normalized > other
        return NotImplemented

    def __ge__(self, other: Union["Rating", float]) -> bool:
        if isinstance(other, Rating):
            return self._normalized >= other._normalized
        elif isinstance(other, (int, float)):
            return self._normalized >= other
        return NotImplemented

    def __str__(self) -> str:
        raise NotImplementedError("Use `to_str()` or `to_display()` for explicit rating string output.")

    def __repr__(self) -> str:
        return f"<Rating normalized={self._normalized:.3f}>"
