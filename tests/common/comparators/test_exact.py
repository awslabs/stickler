"""Tests for ExactComparator."""

import unittest

from stickler.comparators import ExactComparator


class TestExactComparator(unittest.TestCase):
    """Test the ExactComparator."""

    def setUp(self):
        """Set up test fixtures."""
        self.comparator = ExactComparator()
        self.case_sensitive_comparator = ExactComparator(case_sensitive=True)

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        self.assertEqual(self.comparator.compare("hello", "hello"), 1.0)
        self.assertEqual(self.comparator.compare("123", "123"), 1.0)
        self.assertEqual(self.comparator.compare("", ""), 1.0)

    def test_case_insensitive(self):
        """Test that case is ignored by default."""
        self.assertEqual(self.comparator.compare("Hello", "hello"), 1.0)
        self.assertEqual(self.comparator.compare("WORLD", "world"), 1.0)
        self.assertEqual(self.comparator.compare("Hello World", "hello world"), 1.0)

    def test_case_sensitive(self):
        """Test that case-sensitive comparator respects case."""
        self.assertEqual(self.case_sensitive_comparator.compare("hello", "hello"), 1.0)
        self.assertEqual(self.case_sensitive_comparator.compare("Hello", "hello"), 0.0)
        self.assertEqual(self.case_sensitive_comparator.compare("WORLD", "world"), 0.0)

    def test_punctuation_and_spaces(self):
        """Test that punctuation and spaces are ignored."""
        self.assertEqual(self.comparator.compare("hello, world", "hello world"), 1.0)
        self.assertEqual(self.comparator.compare("hello-world", "hello world"), 1.0)
        self.assertEqual(self.comparator.compare("hello.world", "hello world"), 1.0)
        self.assertEqual(self.comparator.compare("hello  world", "hello world"), 1.0)

    def test_non_matching(self):
        """Test that non-matching strings return 0.0."""
        self.assertEqual(self.comparator.compare("hello", "world"), 0.0)
        self.assertEqual(self.comparator.compare("123", "456"), 0.0)
        self.assertEqual(self.comparator.compare("hello", "hello world"), 0.0)

    def test_none_values(self):
        """Test that None values are handled correctly."""
        self.assertEqual(self.comparator.compare(None, None), 1.0)
        self.assertEqual(self.comparator.compare("hello", None), 0.0)
        self.assertEqual(self.comparator.compare(None, "world"), 0.0)

    def test_non_string_values(self):
        """Test that non-string values are converted to strings."""
        self.assertEqual(self.comparator.compare(123, "123"), 1.0)
        self.assertEqual(self.comparator.compare("456", 456), 1.0)
        self.assertEqual(self.comparator.compare(789, 789), 1.0)

    def test_binary_compare_match(self):
        """Test that binary_compare returns (1, 0) for match."""
        self.assertEqual(self.comparator.binary_compare("hello", "hello"), (1, 0))
        self.assertEqual(self.comparator.binary_compare("Hello", "hello"), (1, 0))

    def test_binary_compare_no_match(self):
        """Test that binary_compare returns (0, 1) for no match."""
        self.assertEqual(self.comparator.binary_compare("hello", "world"), (0, 1))
        self.assertEqual(
            self.case_sensitive_comparator.binary_compare("Hello", "hello"), (0, 1)
        )

    def test_custom_threshold(self):
        """Test that the threshold is applied correctly.

        This is a silly test for ExactComparator since it only returns 0.0 or 1.0,
        but it demonstrates how thresholds work with binary_compare.
        """
        # Create a comparator with a threshold of 0.5, it shouldn't matter
        # since ExactComparator only returns 0.0 or 1.0
        comparator = ExactComparator(threshold=0.5)
        self.assertEqual(comparator.binary_compare("hello", "hello"), (1, 0))
        self.assertEqual(comparator.binary_compare("hello", "world"), (0, 1))


if __name__ == "__main__":
    unittest.main()
