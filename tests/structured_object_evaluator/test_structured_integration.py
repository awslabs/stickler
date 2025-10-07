"""Test integration of structured models with the main anls_score function."""

import unittest
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stickler.structured_object_evaluator.utils.anls_score import anls_score
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator


class SimpleDocument(StructuredModel):
    """Simple document model for testing."""

    title: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )

    author: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)

    content: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.7)


class NestedDocument(StructuredModel):
    """Document with nested structure for testing."""

    metadata: SimpleDocument

    tags: list[str] = ComparableField(comparator=LevenshteinComparator(), threshold=0.9)


class TestStructuredIntegration(unittest.TestCase):
    """Test integration of structured models with anls_score."""

    def test_simple_document_comparison(self):
        """Test comparison of simple document models."""
        # Define ground truth
        gt = SimpleDocument(
            title="Introduction to ANLS*",
            author="John Smith",
            content="ANLS* is a metric for evaluating document processing tasks.",
        )

        # Test exact match
        exact_match = SimpleDocument(
            title="Introduction to ANLS*",
            author="John Smith",
            content="ANLS* is a metric for evaluating document processing tasks.",
        )

        score = anls_score(gt, exact_match)
        self.assertEqual(score, 1.0)

        # Test close match
        close_match = SimpleDocument(
            title="Introduction to ANLS",  # Missing * but still close
            author="John Smith",
            content="ANLS* is a metric for document processing evaluation.",  # Reworded but similar
        )

        score = anls_score(gt, close_match)
        self.assertGreater(score, 0.7)  # Should be reasonably high score

        # Test with return_key_scores
        score, field_scores = anls_score(gt, close_match, return_key_scores=True)
        # Field scores are directly returned in the dictionary
        self.assertIn("title", field_scores)
        self.assertIn("author", field_scores)
        self.assertIn("content", field_scores)

        # Test with return_gt
        score, returned_gt = anls_score(gt, close_match, return_gt=True)
        self.assertEqual(returned_gt, gt)  # Should return the original GT

    def test_nested_document_comparison(self):
        """Test comparison of nested document models."""
        # Define ground truth
        gt = NestedDocument(
            metadata=SimpleDocument(
                title="Advanced ANLS* Applications",
                author="Jane Doe",
                content="This document explores advanced uses of ANLS*.",
            ),
            tags=["nlp", "metrics", "evaluation"],
        )

        # Test with variations
        variation = NestedDocument(
            metadata=SimpleDocument(
                title="Advanced Applications of ANLS*",  # Reworded but similar
                author="J. Doe",  # Abbreviated but still close
                content="This document explores advanced applications of the ANLS* metric.",  # Reworded
            ),
            tags=["nlp", "evaluation", "metrics"],  # Reordered
        )

        # Since we're testing the integration, just make sure it runs without errors
        score = anls_score(gt, variation)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test with return_key_scores and return_gt
        score, returned_gt, field_scores = anls_score(
            gt, variation, return_gt=True, return_key_scores=True
        )
        self.assertEqual(returned_gt, gt)  # Should return the original GT

        # Field scores are directly returned in the dictionary
        self.assertIn("metadata", field_scores)
        self.assertIn("tags", field_scores)

    def test_dictionary_comparison(self):
        """Test that anls_score still works with traditional types."""
        # Define a regular dict ground truth
        gt_dict = {
            "title": "ANLS* Guide",
            "authors": ["John Smith", "Jane Doe"],
            "sections": {
                "intro": "Introduction to ANLS*",
                "methods": "How ANLS* works",
            },
        }

        # Define a prediction
        pred_dict = {
            "title": "ANLS* User Guide",
            "authors": ["J. Smith", "J. Doe"],
            "sections": {
                "intro": "An Introduction to the ANLS* Metric",
                "methods": "How ANLS* is implemented",
            },
        }

        # Make sure regular dict comparison still works
        score = anls_score(gt_dict, pred_dict)
        self.assertGreater(score, 0.5)  # Should have a reasonable score


if __name__ == "__main__":
    unittest.main()
