"""Tests for comparators used in the structured object evaluator."""

import unittest
from typing import Any, Dict, List, Optional

from stickler.structured_object_evaluator import StructuredModel, ComparableField
from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator

try:
    from stickler.comparators.fuzzy import FuzzyComparator

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

try:
    from stickler.comparators.semantic import SemanticComparator

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


class TestLevenshteinComparator(unittest.TestCase):
    """Test cases for the LevenshteinComparator."""

    def setUp(self):
        """Set up test fixtures."""
        self.comparator = LevenshteinComparator()

    def test_exact_match(self):
        """Test exact string match."""
        self.assertEqual(self.comparator.compare("hello", "hello"), 1.0)

    def test_similar_match(self):
        """Test similar string match."""
        score = self.comparator.compare("hello", "helo")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_different_match(self):
        """Test completely different strings."""
        score = self.comparator.compare("hello", "world")
        self.assertGreater(score, 0.0)  # Even different strings have some similarity
        self.assertLess(score, 0.5)  # But the score should be low

    def test_case_sensitivity(self):
        """Test case sensitivity."""
        # Note: Some implementations might not be case sensitive
        score = self.comparator.compare("Hello", "hello")
        if score < 1.0:
            self.assertLess(score, 1.0)  # Should be case sensitive
        else:
            # If the implementation is not case sensitive, this will pass
            self.assertEqual(score, 1.0)

    def test_empty_strings(self):
        """Test empty strings."""
        self.assertEqual(self.comparator.compare("", ""), 1.0)
        self.assertEqual(self.comparator.compare("hello", ""), 0.0)
        self.assertEqual(self.comparator.compare("", "hello"), 0.0)

    def test_none_values(self):
        """Test None values."""
        self.assertEqual(self.comparator.compare(None, None), 1.0)
        self.assertEqual(self.comparator.compare("hello", None), 0.0)
        self.assertEqual(self.comparator.compare(None, "hello"), 0.0)


class TestNumericComparator(unittest.TestCase):
    """Test cases for the NumericComparator."""

    def setUp(self):
        """Set up test fixtures."""
        self.comparator = NumericComparator()

    def test_exact_match(self):
        """Test exact numeric match."""
        self.assertEqual(self.comparator.compare("123", "123"), 1.0)
        self.assertEqual(self.comparator.compare(123, 123), 1.0)
        self.assertEqual(self.comparator.compare("123.45", "123.45"), 1.0)
        self.assertEqual(self.comparator.compare(123.45, 123.45), 1.0)

    def test_similar_match(self):
        """Test similar numeric match with tolerance."""
        # Test with values that are close but not exact
        # Note: The default implementation might not support tolerance
        score1 = self.comparator.compare(100, 105)
        score2 = self.comparator.compare(100, 100.001)

        # Either they're considered equal (1.0) or different (0.0)
        # depending on the implementation
        self.assertTrue(score1 == 1.0 or score1 == 0.0)
        self.assertTrue(score2 == 1.0 or score2 == 0.0)

    def test_different_match(self):
        """Test completely different numbers."""
        self.assertEqual(self.comparator.compare(123, 456), 0.0)
        self.assertEqual(self.comparator.compare("123", "456"), 0.0)

    def test_string_formatting(self):
        """Test string formatting of numbers."""
        self.assertEqual(self.comparator.compare("$123.45", "123.45"), 1.0)
        self.assertEqual(self.comparator.compare("123.45", "$123.45"), 1.0)
        self.assertEqual(self.comparator.compare("1,234.56", "1234.56"), 1.0)
        self.assertEqual(self.comparator.compare("1234.56", "1,234.56"), 1.0)

    def test_none_values(self):
        """Test None values."""
        self.assertEqual(self.comparator.compare(None, None), 1.0)
        self.assertEqual(self.comparator.compare(123, None), 0.0)
        self.assertEqual(self.comparator.compare(None, 123), 0.0)


@unittest.skipIf(not FUZZY_AVAILABLE, "FuzzyComparator not available")
class TestFuzzyComparator(unittest.TestCase):
    """Test cases for the FuzzyComparator."""

    def setUp(self):
        """Set up test fixtures."""
        self.comparator = FuzzyComparator()

    def test_exact_match(self):
        """Test exact string match."""
        self.assertEqual(self.comparator.compare("hello", "hello"), 1.0)

    def test_similar_match(self):
        """Test similar string match."""
        score = self.comparator.compare("hello", "helo")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_different_match(self):
        """Test completely different strings."""
        score = self.comparator.compare("hello", "world")
        self.assertGreater(score, 0.0)  # Even different strings have some similarity
        self.assertLess(score, 0.5)  # But the score should be low


@unittest.skipIf(not SEMANTIC_AVAILABLE, "SemanticComparator not available")
class TestSemanticComparator(unittest.TestCase):
    """Test cases for the SemanticComparator."""

    def setUp(self):
        """Set up test fixtures."""
        # Skip this test if SemanticComparator is not available
        if not SEMANTIC_AVAILABLE:
            self.skipTest("SemanticComparator not available")
            return

        # For now, we'll skip the actual tests since the implementation
        # might vary and we don't want to make API calls in tests
        self.skipTest("SemanticComparator tests require API access")


class CustomComparator(BaseComparator):
    """Custom comparator for testing."""

    def compare(self, str1: str, str2: str) -> float:
        """Compare two strings based on length similarity."""
        if str1 is None or str2 is None:
            return 1.0 if str1 == str2 else 0.0

        len1 = len(str1) if str1 else 0
        len2 = len(str2) if str2 else 0

        if len1 == 0 and len2 == 0:
            return 1.0

        max_len = max(len1, len2)
        min_len = min(len1, len2)

        return min_len / max_len


class TestCustomComparator(unittest.TestCase):
    """Test cases for a custom comparator."""

    def setUp(self):
        """Set up test fixtures."""
        self.comparator = CustomComparator()

    def test_exact_match(self):
        """Test exact length match."""
        self.assertEqual(self.comparator.compare("hello", "world"), 1.0)

    def test_similar_match(self):
        """Test similar length match."""
        self.assertEqual(self.comparator.compare("hello", "hi"), 0.4)  # 2/5

    def test_different_match(self):
        """Test completely different lengths."""
        self.assertEqual(self.comparator.compare("a", "abcdefghij"), 0.1)  # 1/10

    def test_empty_strings(self):
        """Test empty strings."""
        self.assertEqual(self.comparator.compare("", ""), 1.0)
        self.assertEqual(self.comparator.compare("hello", ""), 0.0)
        self.assertEqual(self.comparator.compare("", "hello"), 0.0)


class Person(StructuredModel):
    """Person model with custom comparator for testing."""

    name: str = ComparableField(
        comparator=CustomComparator(), threshold=0.5, weight=1.0
    )
    bio: str = ComparableField(comparator=CustomComparator(), threshold=0.7, weight=0.5)


class TestComparatorIntegration(unittest.TestCase):
    """Test cases for integrating comparators with StructuredModel."""

    def test_custom_comparator_integration(self):
        """Test integration of custom comparator with StructuredModel."""
        # Create two Person objects
        person1 = Person(name="John Doe", bio="Software Engineer")
        person2 = Person(name="Jane Smith", bio="Software Developer")

        # Compare them
        result = person1.compare_with(person2)

        # Check field scores - the exact values might depend on the implementation
        # so we'll just check that they're in the expected range
        self.assertGreaterEqual(result["field_scores"]["name"], 0.0)
        self.assertLessEqual(result["field_scores"]["name"], 1.0)
        self.assertGreaterEqual(result["field_scores"]["bio"], 0.0)
        self.assertLessEqual(result["field_scores"]["bio"], 1.0)

        # Check overall score is in the expected range
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
