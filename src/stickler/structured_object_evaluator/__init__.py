"""Structured object evaluator package.

This package provides tools for evaluating structured objects using configurable
comparison metrics and displaying the results in a user-friendly format.
"""

from .models.structured_model import StructuredModel
from .models.comparable_field import ComparableField
from .models.non_match_field import NonMatchField, NonMatchType
from .utils.anls_score import compare_structured_models, anls_score, compare_json
from .utils.key_scores import ScoreNode, construct_nested_dict, merge_and_calculate_mean
from .evaluator import StructuredModelEvaluator
from .utils.pretty_print import print_confusion_matrix, print_confusion_matrix_html

__all__ = [
    "StructuredModel",
    "ComparableField",
    "NonMatchField",
    "NonMatchType",
    "compare_structured_models",
    "anls_score",
    "compare_json",
    "ScoreNode",
    "construct_nested_dict",
    "merge_and_calculate_mean",
    "StructuredModelEvaluator",
    "print_confusion_matrix",
    "print_confusion_matrix_html",
]
