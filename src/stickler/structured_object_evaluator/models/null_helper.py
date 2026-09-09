"""Helper class for null value checking and validation.

This module provides utility methods for checking various null conditions
used throughout the comparison process.
"""

from typing import Any


class NullHelper:
    """Helper class for null value checking and validation."""

    @staticmethod
    def is_truly_null(val: Any) -> bool:
        """Check if a value is truly null (None).

        Args:
            val: Value to check

        Returns:
            True if the value is None, False otherwise
        """
        return val is None

    @staticmethod
    def is_effectively_null_for_lists(val: Any) -> bool:
        """Check if a list value is effectively null (None or empty list).

        Args:
            val: Value to check

        Returns:
            True if the value is None or an empty list, False otherwise
        """
        return val is None or (isinstance(val, list) and len(val) == 0)

    @staticmethod
    def is_effectively_null_for_primitives(val: Any) -> bool:
        """Check if a non-list value is effectively null.

        Treats ``None``, an empty string and an empty dict as equivalent, which
        is the rule the docs state: "Empty strings (``""``), empty lists
        (``[]``), and empty objects (``{}``) are treated as null." Empty lists
        are the one case this does *not* answer, because list-ness is decided
        from the annotation rather than the value -- see
        :meth:`is_effectively_null_for_lists`, which ``ComparisonDispatcher``
        reaches first for any field ``_is_list_field`` recognises.

        ``{}`` was missing here, and its absence was visible in the metrics
        rather than benign: two **identical** objects each holding ``{}`` in a
        dict field classified as a false discovery, the same contradiction
        #233 reports for lists. It was also unreachable to score, because no
        comparator accepts a dict -- ``LevenshteinComparator`` raises
        ``TypeError`` -- so a populated dict field made the pair uncomparable
        instead of merely mismatched. Reading ``{}`` as absent gives the three
        empty/null combinations a true negative and turns the fourth
        (``{}`` against populated) into a false alarm rather than a crash.

        Args:
            val: Value to check

        Returns:
            True if the value is None, an empty string or an empty dict
        """
        return val is None or (isinstance(val, (str, dict)) and len(val) == 0)
