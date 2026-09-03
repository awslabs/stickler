"""Exact string comparison comparator."""

from collections.abc import Mapping as abc_Mapping
from typing import Any, Dict, Optional

from stickler.comparators.base import BaseComparator
from stickler.utils.canonical import canonicalize_json_sorted


class ExactComparator(BaseComparator):
    """Comparator that checks for exact string equality.

    Returns 1.0 if the two values are identical, 0.0 otherwise. By default,
    comparison is case-sensitive: ``"Hello"`` does not match ``"hello"``.
    Set ``case_sensitive=False`` for case-insensitive matching using Unicode
    case folding.

    No normalization (punctuation or whitespace stripping) is performed.
    ``"SHP-2024-001"`` does not match ``"SHP 2024 001"`` — they are different
    strings. This is the correct behavior for identifiers, codes, and any
    field where the exact character sequence matters.

    Example:
        ```python
        # Default: case-sensitive exact matching
        comparator = ExactComparator()
        comparator.compare("Hello", "Hello")  # Returns 1.0
        comparator.compare("Hello", "hello")  # Returns 0.0
        comparator.compare("ID-123", "ID 123")  # Returns 0.0

        # Case-insensitive matching (uses Unicode casefold)
        ci = ExactComparator(case_sensitive=False)
        ci.compare("Hello", "hello")  # Returns 1.0
        ci.compare("STRASSE", "straße")  # Returns 1.0 (casefold handles this)
        ```

    Args:
        threshold: Similarity threshold (default 1.0). Since this comparator
            only returns 0.0 or 1.0, values below 1.0 effectively accept any
            non-null value as a match.
        case_sensitive: If True (default), comparison distinguishes case.
            If False, uses ``str.casefold()`` for Unicode-aware case folding.

    .. versionchanged:: 0.7.0
        Default changed to ``case_sensitive=True``. Punctuation and whitespace
        stripping removed — use ``LevenshteinComparator`` or ``FuzzyComparator``
        for normalized text matching. This makes ``ExactComparator`` truly exact,
        fixing #199 where ``"SHP-2024-001"`` incorrectly matched ``"shp 2024 001"``.
    """

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = True):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (default 1.0)
            case_sensitive: Whether comparison is case sensitive (default True)
        """
        super().__init__(threshold=threshold)
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        """Return the name of the comparator."""
        return "exact"

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Return configuration parameters for serialization.

        Only includes non-default values to keep serialized output minimal.
        """
        cfg: Dict[str, Any] = {}
        if not self.case_sensitive:
            cfg["case_sensitive"] = False
        return cfg if cfg else None


    def _compare(self, str1: Any, str2: Any) -> float:
        """Compare two values with exact string matching.

        Args:
            str1: First value
            str2: Second value

        Returns:
            1.0 if the strings match exactly, 0.0 otherwise
        """
        # A mapping is canonicalised, not stringified. `str(dict)` preserves
        # insertion order, so two mappings with identical content scored 0.0
        # whenever their keys happened to be ordered differently -- and 1.0 when
        # they happened to agree, which is worse than a consistent answer because
        # it depends on how the JSON arrived.
        if isinstance(str1, abc_Mapping) or isinstance(str2, abc_Mapping):
            str1 = canonicalize_json_sorted(str1)
            str2 = canonicalize_json_sorted(str2)

        # Convert to strings if they aren't already
        str1 = str(str1)
        str2 = str(str2)

        # Apply case folding if case-insensitive
        if not self.case_sensitive:
            str1 = str1.casefold()
            str2 = str2.casefold()

        return 1.0 if str1 == str2 else 0.0

    def __repr__(self) -> str:
        """Detailed string representation."""
        parts = [f"threshold={self.threshold}"]
        if not self.case_sensitive:
            parts.append("case_sensitive=False")
        return f"ExactComparator({', '.join(parts)})"
