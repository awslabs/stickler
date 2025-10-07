"""BERT-based semantic comparator."""

import evaluate
from typing import Any

from stickler.utils import strip_punctuation_space
from stickler.comparators.base import BaseComparator

# Load BERT model globally
try:
    model = evaluate.load("bertscore", model_type="distilbert-base-uncased")
except Exception as e:
    print(f"Warning: Could not load BERT model: {str(e)}")
    model = None


class BERTComparator(BaseComparator):
    """Comparator that uses BERT embeddings for semantic similarity.

    This comparator uses the BERTScore metric to calculate semantic similarity
    between strings, returning the f1 score as the similarity measure.

    Example:
        ```python
        comparator = BERTComparator(threshold=0.8)

        # Returns similarity score based on semantic similarity
        score = comparator.compare("The cat sat on the mat", "A feline was sitting on a rug")
        ```
    """

    def __init__(self, threshold: float = 0.7):
        """Initialize the BERTComparator.

        Args:
            threshold: Similarity threshold (0.0-1.0)
        """
        super().__init__(threshold=threshold)
        if model is None:
            raise ImportError(
                "BERTScore model could not be loaded. Please install 'evaluate' package."
            )

    def compare(self, str1: Any, str2: Any) -> float:
        """Compare two strings using BERT semantic similarity.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0.0 and 1.0 based on BERTScore f1
        """
        if str1 is None or str2 is None:
            return 0.0

        # Convert to strings if they aren't already
        str1 = str(str1)
        str2 = str(str2)

        # Strip punctuation and whitespace
        str1_clean = strip_punctuation_space(str1)
        str2_clean = strip_punctuation_space(str2)

        # Handle empty strings
        if not str1_clean or not str2_clean:
            return 1.0 if str1_clean == str2_clean else 0.0

        try:
            # Calculate BERT score
            result = model.compute(
                predictions=[str1_clean], references=[str2_clean], lang="en"
            )

            # Return f1 score
            return result["f1"][0]
        except Exception as e:
            # Fallback to direct comparison
            print(f"BERT comparison error: {str(e)}")
            return 1.0 if str1_clean == str2_clean else 0.0
