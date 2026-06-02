"""
Aggregate confusion-matrix accumulator for bulk evaluation.

Public API:
    AggregateConfusionMatrixAccumulator - rolls up per-document confusion
        matrix aggregates into corpus-level totals.
"""

from stickler.structured_object_evaluator.models.aggregate.accumulator import (
    AggregateConfusionMatrixAccumulator,
)

__all__ = ["AggregateConfusionMatrixAccumulator"]
