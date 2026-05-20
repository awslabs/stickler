"""
Base class for post-comparison metric accumulators.

Post-comparison accumulators extract metadata from comparison results
(via prediction_raw), accumulate data across documents, and compute
aggregate metrics. The BulkStructuredModelEvaluator delegates to a
list of accumulators at each step.

Current implementations:
    - ConfidenceAccumulator (confidence/accumulator.py): AUROC, ECE, Brier, ECARB

Future implementations:
    - BBoxMAPAccumulator: mean Average Precision from bounding box data
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PostComparisonAccumulator(ABC):
    """Interface for accumulating post-comparison metrics across documents.

    Each accumulator extracts its specific metadata from comparison results,
    accumulates data across documents, and computes aggregate metrics.
    The bulk evaluator orchestrates multiple accumulators without knowing
    their internals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Key used in ProcessEvaluation for this accumulator's metrics."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state."""
        ...

    @abstractmethod
    def accumulate(
        self,
        comparison_result: Dict[str, Any],
        prediction_raw: Optional[Dict[str, Any]],
    ) -> None:
        """Extract and accumulate data from a single comparison result.

        Args:
            comparison_result: The full compare_with() result dict.
                Must contain "field_comparisons" for most accumulators.
            prediction_raw: The original prediction JSON (pre-unwrapping),
                or None if the prediction had no rich value metadata.
        """
        ...

    @abstractmethod
    def compute(self) -> Optional[Dict[str, Any]]:
        """Compute aggregate metrics from accumulated data.

        Returns:
            Dict of metrics, or None if no data was accumulated.
        """
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return serializable state for checkpointing."""
        ...

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        ...

    @abstractmethod
    def merge_state(self, other_state: Dict[str, Any]) -> None:
        """Merge state from another accumulator instance (distributed eval)."""
        ...
