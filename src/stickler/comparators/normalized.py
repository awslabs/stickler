"""Configurable equality after Unicode-aware text normalization."""

import unicodedata
from typing import Any, Dict, Optional

from stickler.comparators.base import BaseComparator


class NormalizedComparator(BaseComparator):
    """Compare text after applying an explicit set of normalizations.

    By default this restores the formatting-insensitive equality behavior that
    ``ExactComparator`` provided before 0.7.0: case, whitespace, and punctuation
    differences are ignored. Unlike the old implementation, punctuation means
    every character in a Unicode ``P*`` category and whitespace is identified
    with :meth:`str.isspace`. Unicode symbols such as ``$``, ``±``, and emoji
    are retained, as are combining marks. Text is normalized to NFC so composed
    and decomposed spellings compare consistently.

    Each transform can be disabled independently. This makes the normalization
    policy visible in code and in serialized comparator configuration instead
    of hiding it behind a single ambiguous "strip punctuation" operation.

    Args:
        threshold: Similarity threshold (default 1.0). This comparator returns
            only 0.0 or 1.0.
        case_sensitive: Preserve case differences when True (default False).
        ignore_whitespace: Remove Unicode whitespace when True (default True).
        ignore_punctuation: Remove Unicode punctuation when True (default True).
    """

    DEFAULT_THRESHOLD = 1.0

    def __init__(
        self,
        threshold: Optional[float] = None,
        case_sensitive: bool = False,
        ignore_whitespace: bool = True,
        ignore_punctuation: bool = True,
    ):
        super().__init__(threshold=threshold)
        self.case_sensitive = case_sensitive
        self.ignore_whitespace = ignore_whitespace
        self.ignore_punctuation = ignore_punctuation

    @property
    def name(self) -> str:
        """Return the comparator's short name."""
        return "normalized"

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Return non-default options for JSON-friendly serialization."""
        config: Dict[str, Any] = {}
        if self.case_sensitive:
            config["case_sensitive"] = True
        if not self.ignore_whitespace:
            config["ignore_whitespace"] = False
        if not self.ignore_punctuation:
            config["ignore_punctuation"] = False
        return config or None

    def _normalize(self, value: Any) -> str:
        """Convert a value to text and apply the configured transforms."""
        text = unicodedata.normalize("NFC", str(value))
        if not self.case_sensitive:
            text = text.casefold()
        if self.ignore_whitespace or self.ignore_punctuation:
            text = "".join(
                character
                for character in text
                if not (self.ignore_whitespace and character.isspace())
                and not (
                    self.ignore_punctuation
                    and unicodedata.category(character).startswith("P")
                )
            )
        return text

    def _compare(self, str1: Any, str2: Any) -> float:
        """Return 1.0 when normalized values are equal, otherwise 0.0."""
        return 1.0 if self._normalize(str1) == self._normalize(str2) else 0.0

    def __repr__(self) -> str:
        """Return a representation containing every effective option."""
        return (
            "NormalizedComparator("
            f"threshold={self.threshold}, "
            f"case_sensitive={self.case_sensitive}, "
            f"ignore_whitespace={self.ignore_whitespace}, "
            f"ignore_punctuation={self.ignore_punctuation})"
        )
