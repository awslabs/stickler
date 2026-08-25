"""Semantic comparator for embedding-based similarity."""

import logging
import sys
from functools import partial
from typing import Callable, Optional

from stickler.comparators.base import BaseComparator
from stickler.comparators.utils import generate_bedrock_embedding

logger = logging.getLogger(__name__)


def _cosine_distance(x, y) -> float:
    """Cosine distance between two embedding vectors.

    scipy is imported here rather than at module scope: ``SemanticComparator``
    is a top-level export, so a module-level import would put scipy on the
    ``import stickler`` path for every user, including those who never compute
    an embedding.

    Raises:
        ImportError: If scipy is not installed.
    """
    try:
        from scipy import spatial
    except ImportError as exc:  # pragma: no cover - exercised by the extras gate
        raise ImportError(
            "SemanticComparator's cosine similarity requires scipy. Install it "
            'with: pip install "stickler-eval[semantic]"'
        ) from exc

    return spatial.distance.cosine(x, y)


def _embedding_function_name(embedding_function: Callable) -> str:
    """Return a useful name for custom callables and functools.partial wrappers."""
    if isinstance(embedding_function, partial):
        return getattr(
            embedding_function.func,
            "__name__",
            type(embedding_function.func).__name__,
        )
    return getattr(embedding_function, "__name__", type(embedding_function).__name__)


def _input_length(value) -> int:
    """Return a log-safe length for primitive values routed to semantic compare."""
    try:
        return len(value)
    except TypeError:
        return len(str(value))


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
        "cosine_similarity": lambda x, y: 1 - _cosine_distance(x, y)
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

        self.model_id = model_id
        if embedding_function is not None:
            self.embedding_function = embedding_function
        else:
            self.embedding_function = partial(
                generate_bedrock_embedding, model_id=model_id
            )

        self.sim_function = sim_function
        self.similarity_function = self.SIMILARITY_FUNCTIONS[self.sim_function]

    def _compare(self, str1: str, str2: str) -> float:
        """Compare two values using semantic similarity.

        If embedding generation fails, this logs the model ID, embedding function,
        input lengths, similarity function, and exception type before falling back
        to raw equality.

        Args:
            str1: First value
            str2: Second value

        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            x, y = self.embedding_function(str1), self.embedding_function(str2)
            return self.similarity_function(x, y)
        except Exception:
            logger.exception(
                "Semantic embedding comparison failed; falling back to string equality",
                extra={
                    "embedding_function": _embedding_function_name(
                        self.embedding_function
                    ),
                    "model_id": getattr(self, "model_id", None),
                    "input_1_length": _input_length(str1),
                    "input_2_length": _input_length(str2),
                    "similarity_function": self.sim_function,
                    "exception_type": type(sys.exc_info()[1]).__name__,
                },
            )
            # Fallback to string equality if embedding fails
            return 1.0 if str1 == str2 else 0.0
