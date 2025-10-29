"""Models package for structured object evaluation.

This package contains the core models and helper classes for structured comparison.
The refactored architecture uses a delegation pattern where StructuredModel delegates
to specialized helper classes for different aspects of comparison.

Public API:
    - StructuredModel: Main model class for structured comparison
    - ComparableField: Field configuration for comparison
    - NonMatchField: Representation of non-matching fields
    - NonMatchType: Enumeration of non-match types

Internal Helper Classes (not part of public API):
    - ModelFactory: Creates dynamic StructuredModel subclasses from JSON
    - ComparisonEngine: Orchestrates the comparison process
    - ComparisonDispatcher: Routes field comparisons to appropriate handlers
    - FieldComparator: Compares primitive and structured fields
    - PrimitiveListComparator: Compares lists of primitive values
    - StructuredListComparator: Compares lists of StructuredModel instances
    - NonMatchCollector: Collects non-matching fields during comparison
    - ConfusionMatrixBuilder: Builds complete confusion matrices
    - ConfusionMatrixCalculator: Calculates confusion matrix metrics
    - AggregateMetricsCalculator: Calculates aggregate metrics
    - DerivedMetricsCalculator: Calculates derived metrics (F1, precision, recall)
"""

# Public API exports
from .structured_model import StructuredModel
from .comparable_field import ComparableField
from .non_match_field import NonMatchField, NonMatchType

# Internal helper classes (not exported in __all__ but available for import)
from .model_factory import ModelFactory
from .comparison_engine import ComparisonEngine
from .comparison_dispatcher import ComparisonDispatcher
from .field_comparator import FieldComparator
from .primitive_list_comparator import PrimitiveListComparator
from .structured_list_comparator import StructuredListComparator
from .non_match_collector import NonMatchCollector
from .confusion_matrix_builder import ConfusionMatrixBuilder
from .confusion_matrix_calculator import ConfusionMatrixCalculator
from .aggregate_metrics_calculator import AggregateMetricsCalculator
from .derived_metrics_calculator import DerivedMetricsCalculator

# Only export public API classes
__all__ = [
    "StructuredModel",
    "ComparableField",
    "NonMatchField",
    "NonMatchType",
]
