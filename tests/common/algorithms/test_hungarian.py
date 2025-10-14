"""Unit tests for the Hungarian algorithm implementation.

This module tests the functionality of the Hungarian algorithm implementation
to ensure it works correctly and maintains compatibility with existing code.
"""

import unittest

from stickler.comparators import LevenshteinComparator, NumericComparator
from stickler.algorithms import HungarianMatcher


class TestHungarianMatcher(unittest.TestCase):
    """Test the HungarianMatcher implementation."""

    def setUp(self):
        """Set up test environment."""
        self.matcher = HungarianMatcher()  # Default exact matching
        self.levenshtein_matcher = HungarianMatcher(comparator=LevenshteinComparator())
        self.numeric_matcher = HungarianMatcher(comparator=NumericComparator())

    def test_exact_match(self):
        """Test exact matching between lists."""
        list1 = ["apple", "banana", "cherry"]
        list2 = ["banana", "cherry", "apple"]

        # Test the direct match method
        indices, _ = self.matcher.match(list1, list2)
        self.assertEqual(len(indices), 3)  # All items should be matched

        # Test the metrics calculation
        metrics = self.matcher.calculate_metrics(list1, list2)
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_partial_match(self):
        """Test partial matching between lists."""
        list1 = ["apple", "banana", "cherry"]
        list2 = ["banana", "cherry", "date"]

        # Test metrics calculation
        metrics = self.matcher.calculate_metrics(list1, list2)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_no_match(self):
        """Test no matching between lists."""
        list1 = ["apple", "banana", "cherry"]
        list2 = ["date", "fig", "grape"]

        # Test metrics calculation
        metrics = self.matcher.calculate_metrics(list1, list2)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 3)
        self.assertEqual(metrics["fn"], 3)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_different_length_lists(self):
        """Test matching with different length lists."""
        # Ground truth longer
        list1 = ["apple", "banana", "cherry", "date"]
        list2 = ["banana", "cherry", "apple"]

        metrics = self.matcher.calculate_metrics(list1, list2)
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 0.75)
        self.assertAlmostEqual(metrics["f1"], 0.8571428571428571)

        # Prediction longer
        list1 = ["banana", "cherry", "apple"]
        list2 = ["apple", "banana", "cherry", "date"]

        metrics = self.matcher.calculate_metrics(list1, list2)
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["f1"], 0.8571428571428571)

    def test_empty_lists(self):
        """Test handling of empty lists."""
        # Both empty
        metrics = self.matcher.calculate_metrics([], [])
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

        # Ground truth empty
        metrics = self.matcher.calculate_metrics([], ["apple", "banana"])
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 2)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 0.0)

        # Prediction empty
        metrics = self.matcher.calculate_metrics(["apple", "banana"], [])
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 2)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_levenshtein_matching(self):
        """Test fuzzy matching using LevenshteinComparator."""
        list1 = ["apple", "banana", "cherry"]
        list2 = ["apples", "banana", "cherrys"]

        # Should get high scores for similar strings
        metrics = self.levenshtein_matcher.calculate_metrics(list1, list2)
        self.assertEqual(
            metrics["tp"], 3
        )  # All items should match with fuzzy comparison
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

    def test_numeric_matching(self):
        """Test numeric matching using NumericComparator."""
        list1 = [100, 250, 375.5]
        list2 = ["$100", "$250.00", "$375.50"]

        # Should match numeric values despite formatting
        metrics = self.numeric_matcher.calculate_metrics(list1, list2)
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

    def test_legacy_interface(self):
        """Test the legacy __call__ interface for backward compatibility."""
        list1 = ["apple", "banana", "cherry"]
        list2 = ["banana", "cherry", "apple"]

        # Test the traditional tp, fp return format
        tp, fp = self.matcher(list1, list2)
        self.assertEqual(tp, 3)
        self.assertEqual(fp, 0)

        # Test with partial match
        tp, fp = self.matcher(list1, ["banana", "cherry", "date"])
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)

    def test_single_item_lists(self):
        """Test the optimization for single-item lists."""
        # Matching single items
        metrics = self.matcher.calculate_metrics("apple", "apple")
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

        # Non-matching single items
        metrics = self.matcher.calculate_metrics("apple", "banana")
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)

    def test_string_list_parsing(self):
        """Test parsing of string representations of lists."""
        # String representation of lists
        list1_str = '["apple", "banana", "cherry"]'
        list2 = ["banana", "cherry", "apple"]

        metrics = self.matcher.calculate_metrics(list1_str, list2)
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)


if __name__ == "__main__":
    unittest.main()
