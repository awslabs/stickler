"""Tests for NumericComparator."""

import unittest
from decimal import Decimal

from stickler.comparators import NumericComparator


class TestNumericComparator(unittest.TestCase):
    """Test the NumericComparator."""

    def setUp(self):
        """Set up test fixtures."""
        # Default comparator with exact matching
        self.comparator = NumericComparator()

        # Comparator with 10% relative tolerance
        self.relative_comparator = NumericComparator(relative_tolerance=0.1)

        # Comparator with 5 absolute tolerance
        self.absolute_comparator = NumericComparator(absolute_tolerance=5)

        # Comparator with both tolerances
        self.combined_comparator = NumericComparator(
            relative_tolerance=0.1, absolute_tolerance=5
        )

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        self.assertEqual(self.comparator.compare("123", "123"), 1.0)
        self.assertEqual(self.comparator.compare(456, 456), 1.0)
        self.assertEqual(self.comparator.compare("789.0", 789), 1.0)
        self.assertEqual(self.comparator.compare("0", "0"), 1.0)

    def test_numeric_formatting(self):
        """Test that different numeric formats are handled correctly."""
        self.assertEqual(self.comparator.compare("123", "123.0"), 1.0)
        self.assertEqual(self.comparator.compare("$123", "123"), 1.0)
        self.assertEqual(self.comparator.compare("123,456", "123456"), 1.0)
        self.assertEqual(self.comparator.compare("-123", "-123"), 1.0)

    def test_non_matching(self):
        """Test that non-matching numbers return 0.0."""
        self.assertEqual(self.comparator.compare("123", "456"), 0.0)
        self.assertEqual(self.comparator.compare(789, 987), 0.0)
        self.assertEqual(self.comparator.compare("1.23", "1.24"), 0.0)

    def test_non_numeric_values(self):
        """Test that non-numeric values are handled correctly."""
        self.assertEqual(self.comparator.compare("abc", "123"), 0.0)
        self.assertEqual(self.comparator.compare("123", "abc"), 0.0)
        self.assertEqual(self.comparator.compare("abc", "def"), 0.0)

    def test_none_values(self):
        """Test that None values are handled correctly."""
        self.assertEqual(self.comparator.compare(None, None), 1.0)
        self.assertEqual(self.comparator.compare("123", None), 0.0)
        self.assertEqual(self.comparator.compare(None, "456"), 0.0)

    def test_relative_tolerance(self):
        """Test that relative tolerance works correctly."""
        # Within 10% tolerance
        self.assertEqual(self.relative_comparator.compare(100, 109), 1.0)
        self.assertEqual(self.relative_comparator.compare(100, 91), 1.0)

        # Outside 10% tolerance - 11% difference
        self.assertEqual(self.relative_comparator.compare(100, 111), 0.0)
        self.assertEqual(self.relative_comparator.compare(100, 89), 0.0)

        # Edge cases
        self.assertEqual(self.relative_comparator.compare(0, 0), 1.0)
        self.assertEqual(
            self.relative_comparator.compare(0, 0.05), 1.0
        )  # Special case for zero
        self.assertEqual(self.relative_comparator.compare(1000000, 1090000), 1.0)

    def test_absolute_tolerance(self):
        """Test that absolute tolerance works correctly."""
        # Within tolerance of 5
        self.assertEqual(self.absolute_comparator.compare(100, 104), 1.0)
        self.assertEqual(self.absolute_comparator.compare(100, 96), 1.0)

        # Outside tolerance of 5
        self.assertEqual(self.absolute_comparator.compare(100, 106), 0.0)
        self.assertEqual(self.absolute_comparator.compare(100, 94), 0.0)

        # Edge cases
        self.assertEqual(self.absolute_comparator.compare(0, 4), 1.0)
        self.assertEqual(self.absolute_comparator.compare(0, -4), 1.0)

    def test_combined_tolerance(self):
        """Test that combined tolerances work correctly."""
        # Passes absolute tolerance (within 5)
        self.assertEqual(self.combined_comparator.compare(10, 14), 1.0)

        # Passes relative tolerance (within 10%)
        self.assertEqual(self.combined_comparator.compare(100, 109), 1.0)

        # Fails both tolerances
        self.assertEqual(self.combined_comparator.compare(100, 116), 0.0)

    def test_binary_compare_match(self):
        """Test that binary_compare returns (1, 0) for match."""
        self.assertEqual(self.comparator.binary_compare("123", "123"), (1, 0))
        self.assertEqual(self.relative_comparator.binary_compare(100, 109), (1, 0))

    def test_binary_compare_no_match(self):
        """Test that binary_compare returns (0, 1) for no match."""
        self.assertEqual(self.comparator.binary_compare("123", "456"), (0, 1))
        # Use 112 which is 12% different, clearly outside the 10% tolerance
        self.assertEqual(self.relative_comparator.binary_compare(100, 112), (0, 1))

    def test_custom_threshold(self):
        """Test that the threshold is applied correctly."""
        # This shouldn't affect the behavior for NumericComparator
        # since it only returns 0.0 or 1.0
        comparator = NumericComparator(threshold=0.5)
        self.assertEqual(comparator.binary_compare("123", "123"), (1, 0))
        self.assertEqual(comparator.binary_compare("123", "456"), (0, 1))


if __name__ == "__main__":
    unittest.main()
