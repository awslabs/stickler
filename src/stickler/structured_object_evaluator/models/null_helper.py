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
        """Check if a primitive value is effectively null.

        Treats empty strings and None as equivalent for string fields.

        Args:
            val: Value to check

        Returns:
            True if the value is None or an empty string, False otherwise
        """
        return val is None or (isinstance(val, str) and val == "")
