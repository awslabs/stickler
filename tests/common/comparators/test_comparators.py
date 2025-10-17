"""Unit tests for the common comparators.

This module tests the functionality of all comparators in the common module
to ensure they work correctly and maintain compatibility with existing code.
"""

import unittest

from stickler.comparators import (
    LevenshteinComparator,
    NumericComparator,
)

# Try to import FuzzyComparator if available
try:
    from stickler.comparators import FuzzyComparator

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


class TestLevenshteinComparator(unittest.TestCase):
    """Test the LevenshteinComparator implementation."""

    def setUp(self):
        """Set up test environment."""
        self.comparator = LevenshteinComparator()

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        self.assertEqual(self.comparator.compare("test", "test"), 1.0)
        self.assertEqual(
            self.comparator("test", "test"), 1.0
        )  # Test __call__ interface

    def test_no_match(self):
        """Test that completely different strings return low scores."""
        self.assertLess(self.comparator.compare("test", "completely different"), 0.5)

    def test_partial_match(self):
        """Test that similar strings return partial scores."""
        self.assertGreater(self.comparator.compare("testing", "test"), 0.5)
        self.assertLess(self.comparator.compare("testing", "test"), 1.0)

    def test_case_insensitive(self):
        """Test that case doesn't affect comparison by default."""
        self.assertEqual(self.comparator.compare("Test", "test"), 1.0)
        self.assertEqual(self.comparator.compare("TEST", "test"), 1.0)

    def test_whitespace_handling(self):
        """Test that whitespace is normalized."""
        self.assertEqual(self.comparator.compare("test  string", "test string"), 1.0)
        self.assertEqual(self.comparator.compare(" test ", "test"), 1.0)

    def test_none_values(self):
        """Test that None values are handled properly."""
        self.assertEqual(self.comparator.compare(None, None), 1.0)
        self.assertEqual(self.comparator.compare(None, "test"), 0.0)
        self.assertEqual(self.comparator.compare("test", None), 0.0)

    def test_empty_strings(self):
        """Test that empty strings are handled properly."""
        self.assertEqual(self.comparator.compare("", ""), 1.0)
        self.assertEqual(self.comparator.compare("", "test"), 0.0)
        self.assertEqual(self.comparator.compare("test", ""), 0.0)

    def test_binary_compare(self):
        """Test binary_compare returns correct (tp, fp) tuples."""
        # Exact match should return (1, 0)
        self.assertEqual(self.comparator.binary_compare("test", "test"), (1, 0))

        # Completely different should return (0, 1)
        self.assertEqual(
            self.comparator.binary_compare("test", "completely different"), (0, 1)
        )

        # Test with different thresholds
        high_threshold = LevenshteinComparator(threshold=0.9)
        self.assertEqual(high_threshold.binary_compare("testing", "test"), (0, 1))

        low_threshold = LevenshteinComparator(threshold=0.5)
        self.assertEqual(low_threshold.binary_compare("testing", "test"), (1, 0))


class TestNumericComparator(unittest.TestCase):
    """Test the NumericComparator implementation."""

    def setUp(self):
        """Set up test environment."""
        self.comparator = NumericComparator()
        self.tolerance_comparator = NumericComparator(relative_tolerance=0.1)

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        self.assertEqual(self.comparator.compare(123, 123), 1.0)
        self.assertEqual(self.comparator(123, 123), 1.0)  # Test __call__ interface

    def test_numeric_string_match(self):
        """Test that numeric strings are properly converted."""
        self.assertEqual(self.comparator.compare("123", 123), 1.0)
        self.assertEqual(self.comparator.compare(123, "123"), 1.0)
        self.assertEqual(self.comparator.compare("123", "123"), 1.0)

    def test_currency_formatting(self):
        """Test that currency formatting is handled."""
        self.assertEqual(self.comparator.compare("$1,234.56", 1234.56), 1.0)
        self.assertEqual(self.comparator.compare("$1,234.56", "$1,234.56"), 1.0)
        self.assertEqual(self.comparator.compare("$1,234.56", "1,234.56"), 1.0)

    def test_negative_numbers(self):
        """Test that negative numbers are handled properly."""
        self.assertEqual(self.comparator.compare("-123", -123), 1.0)
        self.assertEqual(self.comparator.compare("(123)", -123), 1.0)

    def test_tolerance(self):
        """Test that tolerance is respected."""
        # Without tolerance
        self.assertEqual(self.comparator.compare(100, 109), 0.0)
        self.assertEqual(self.comparator.compare(100, 91), 0.0)

        # With 10% tolerance
        self.assertEqual(
            self.tolerance_comparator.compare(100, 109), 1.0
        )  # Within tolerance
        self.assertEqual(
            self.tolerance_comparator.compare(100, 111), 0.0
        )  # Outside tolerance

    def test_none_values(self):
        """Test that None values are handled properly."""
        self.assertEqual(self.comparator.compare(None, None), 1.0)
        self.assertEqual(self.comparator.compare(None, 123), 0.0)
        self.assertEqual(self.comparator.compare(123, None), 0.0)

    def test_non_numeric_strings(self):
        """Test that non-numeric strings are handled gracefully."""
        self.assertEqual(self.comparator.compare("abc", "abc"), 0.0)
        self.assertEqual(self.comparator.compare("abc", 123), 0.0)

    def test_binary_compare(self):
        """Test binary_compare returns correct (tp, fp) tuples."""
        # Exact match should return (1, 0)
        self.assertEqual(self.comparator.binary_compare(123, 123), (1, 0))
        self.assertEqual(self.comparator.binary_compare("$1,234.56", 1234.56), (1, 0))

        # Non-match should return (0, 1)
        self.assertEqual(self.comparator.binary_compare(100, 200), (0, 1))

        # Test with tolerance
        self.assertEqual(self.tolerance_comparator.binary_compare(100, 109), (1, 0))
        self.assertEqual(self.tolerance_comparator.binary_compare(100, 111), (0, 1))


# Only run FuzzyComparator tests if the thefuzz library is available
if FUZZY_AVAILABLE:

    class TestFuzzyComparator(unittest.TestCase):
        """Test the FuzzyComparator implementation."""

        def setUp(self):
            """Set up test environment."""
            self.comparator = FuzzyComparator()

        def test_exact_match(self):
            """Test that exact matches return 1.0."""
            self.assertEqual(self.comparator.compare("test", "test"), 1.0)
            self.assertEqual(
                self.comparator("test", "test"), 1.0
            )  # Test __call__ interface

        def test_partial_match(self):
            """Test that fuzzy matching works for similar strings."""
            self.assertGreater(self.comparator.compare("saturday", "sunday"), 0.5)
            self.assertGreater(self.comparator.compare("kitten", "sitting"), 0.5)

        def test_token_methods(self):
            """Test different fuzzy matching methods."""
            token_sort = FuzzyComparator(method="token_sort_ratio")
            token_set = FuzzyComparator(method="token_set_ratio")

            # Test token sorting (order doesn't matter)
            self.assertEqual(
                token_sort.compare("python is great", "great is python"), 1.0
            )

            # Test token set (common tokens matter)
            self.assertGreater(
                token_set.compare("python is great and fast", "python is fast"), 0.8
            )

        def test_binary_compare(self):
            """Test binary_compare returns correct (tp, fp) tuples."""
            # Exact match should return (1, 0)
            self.assertEqual(self.comparator.binary_compare("test", "test"), (1, 0))

            # Test partial matches with different thresholds
            high_threshold = FuzzyComparator(threshold=0.9)
            self.assertEqual(
                high_threshold.binary_compare("saturday", "sunday"), (0, 1)
            )

            low_threshold = FuzzyComparator(threshold=0.5)
            self.assertEqual(low_threshold.binary_compare("saturday", "sunday"), (1, 0))

            # Test token methods with binary comparison
            token_set = FuzzyComparator(method="token_set_ratio", threshold=0.7)
            self.assertEqual(
                token_set.binary_compare("python is great and fast", "python is fast"),
                (1, 0),
            )


if __name__ == "__main__":
    unittest.main()
