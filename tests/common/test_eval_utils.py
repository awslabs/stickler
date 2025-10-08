import unittest
import sys
import os
import pandas as pd
import numpy as np
import string

# Import utility functions from the evaluation module
from stickler.utils.text_normalizers import lowercase, strip_punctuation_space


class TestEvaluationUtils(unittest.TestCase):
    """Test cases for utility functions in the common evaluation module."""

    def test_lowercase(self):
        """Test the lowercase function."""
        # Basic functionality tests
        self.assertEqual(lowercase("TEST"), "test")
        self.assertEqual(lowercase("Test String"), "test string")
        self.assertEqual(lowercase(None), None)
        self.assertEqual(lowercase(""), "")

        # Test with mixed case
        self.assertEqual(lowercase("MiXeD cAsE"), "mixed case")

        # Test with numbers and special characters
        self.assertEqual(lowercase("ABC123!@#"), "abc123!@#")

        # Test with non-English characters
        self.assertEqual(lowercase("CAFÉ"), "café")
        self.assertEqual(lowercase("ÑANDÚ"), "ñandú")

        # Test with whitespace
        self.assertEqual(lowercase("  SPACES  "), "  spaces  ")

    def test_strip_punctuation_space(self):
        """Test the strip_punctuation_space function."""
        # Basic functionality tests
        self.assertEqual(strip_punctuation_space("hello!"), "hello")
        self.assertEqual(strip_punctuation_space("test, string."), "teststring")
        self.assertEqual(strip_punctuation_space(None), None)
        self.assertEqual(strip_punctuation_space(""), "")

        # Test with multiple punctuation marks
        self.assertEqual(
            strip_punctuation_space("a!b@c#d$e%f^g&h*i(j)k"), "abcdefghijk"
        )

        # Test with spaces and punctuation
        self.assertEqual(strip_punctuation_space("hello, world!"), "helloworld")
        self.assertEqual(
            strip_punctuation_space("  spaces  and  punctuation!  "),
            "spacesandpunctuation",
        )

        # Test with all punctuation characters
        all_punct = string.punctuation
        self.assertEqual(strip_punctuation_space(all_punct), "")

        # Test with numbers
        self.assertEqual(strip_punctuation_space("123,456.78"), "12345678")

        # Test with mixed content
        self.assertEqual(
            strip_punctuation_space("Phone: (123) 456-7890"), "Phone1234567890"
        )


if __name__ == "__main__":
    unittest.main()
