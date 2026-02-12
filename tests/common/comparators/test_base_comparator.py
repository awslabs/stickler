"""Unit tests for the BaseComparator class."""

from typing import Any

from stickler.comparators.base import BaseComparator


class MockComparator(BaseComparator):
    """Mock implementation of BaseComparator for testing."""

    def compare(self, str1: Any, str2: Any) -> float:
        """Compare two values and return a similarity score.

        For testing, this implementation returns:
        - 1.0 if strings are identical
        - 0.5 if one string is contained within the other
        - 0.0 otherwise
        """
        if str1 is None and str2 is None:
            return 1.0
        if str1 is None or str2 is None:
            return 0.0

        str1 = str(str1)
        str2 = str(str2)

        if str1 == str2:
            return 1.0
        elif str1 in str2 or str2 in str1:
            return 0.5
        else:
            return 0.0


class TestBaseComparator:
    """Test the BaseComparator abstract class and its methods."""

    def setup_method(self):
        """Set up test environment."""
        # Create a mock comparator with default threshold (0.7)
        self.comparator = MockComparator()

        # Create mock comparators with different thresholds
        self.high_threshold = MockComparator(threshold=0.9)
        self.low_threshold = MockComparator(threshold=0.4)

    def test_compare_method(self):
        """Test that the compare method works as expected."""
        assert self.comparator.compare("test", "test") == 1.0
        assert self.comparator.compare("test", "testing") == 0.5
        assert self.comparator.compare("test", "different") == 0.0

    def test_call_interface(self):
        """Test that the __call__ interface works correctly."""
        assert self.comparator("test", "test") == 1.0
        assert self.comparator("test", "testing") == 0.5
        assert self.comparator("test", "different") == 0.0

    def test_binary_compare(self):
        """Test that binary_compare returns correct tuples based on threshold."""
        # Default threshold (0.7)
        assert self.comparator.binary_compare("test", "test") == (1, 0)  # 1.0 >= 0.7
        assert self.comparator.binary_compare("test", "testing") == (0, 1)  # 0.5 < 0.7
        assert self.comparator.binary_compare("test", "different") == (
            0,
            1,
        )  # 0.0 < 0.7

        # High threshold (0.9)
        assert self.high_threshold.binary_compare("test", "test") == (
            1,
            0,
        )  # 1.0 >= 0.9
        assert self.high_threshold.binary_compare("test", "testing") == (
            0,
            1,
        )  # 0.5 < 0.9

        # Low threshold (0.4)
        assert self.low_threshold.binary_compare("test", "test") == (1, 0)  # 1.0 >= 0.4
        assert self.low_threshold.binary_compare("test", "testing") == (
            1,
            0,
        )  # 0.5 >= 0.4
        assert self.low_threshold.binary_compare("test", "different") == (
            0,
            1,
        )  # 0.0 < 0.4

    def test_none_values(self):
        """Test handling of None values."""
        assert self.comparator.binary_compare(None, None) == (1, 0)
        assert self.comparator.binary_compare(None, "test") == (0, 1)
        assert self.comparator.binary_compare("test", None) == (0, 1)
