"""Semantic comparator for embedding-based similarity."""

from functools import partial
from scipy import spatial
from typing import Callable, Dict, Any, Optional

from stickler.comparators.base import BaseComparator
from stickler.comparators.utils import generate_bedrock_embedding


class SemanticComparator(BaseComparator):
    """Comparator that uses embeddings for semantic similarity.

    This comparator uses embeddings from a model (default: Titan) to calculate
    semantic similarity between strings.

    Attributes:
        SIMILARITY_FUNCTIONS: Dictionary of similarity functions
        bc: BedrockClient instance
        model_id: Model ID to use for embeddings
        embedding_function: Function to generate embeddings
        sim_function: Name of the similarity function to use
        similarity_function: The actual similarity function
    """

    SIMILARITY_FUNCTIONS = {
        "cosine_similarity": lambda x, y: 1 - spatial.distance.cosine(x, y)
    }

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        sim_function: str = "cosine_similarity",
        embedding_function: Optional[Callable] = None,
        threshold: float = 0.7,
    ):
        """Initialize the SemanticComparator.

        Args:
            model_id: Model ID to use for embeddings
            sim_function: Name of the similarity function to use
            embedding_function: Optional custom embedding function
            threshold: Similarity threshold (0.0-1.0)

        Raises:
            ImportError: If BedrockClient is not available and no embedding_function is provided
        """
        super().__init__(threshold=threshold)

        if embedding_function is not None:
            self.embedding_function = embedding_function
        else:
            self.model_id = (model_id,)
            self.embedding_function = partial(
                generate_bedrock_embedding, model_id=model_id
            )

        self.sim_function = sim_function
        self.similarity_function = self.SIMILARITY_FUNCTIONS[self.sim_function]

    def compare(self, str1: str, str2: str) -> float:
        """Compare two strings using semantic similarity.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if str1 is None or str2 is None:
            return 0.0

        try:
            x, y = self.embedding_function(str1), self.embedding_function(str2)
            return self.similarity_function(x, y)
        except Exception as e:
            # Fallback to string equality if embedding fails
            return 1.0 if str1 == str2 else 0.0
