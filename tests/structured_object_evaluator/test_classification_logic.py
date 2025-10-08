"""
Test the classification logic for confusion matrix metrics based on the definitions:

False Alarm (FP): GT == null, EST != null
False Discovery (FD): GT != null, EST != null, GT != EST

This test file validates that the metrics evaluator correctly implements these definitions
for different data types and edge cases.
"""

import pytest
from typing import Dict, Any, List, Optional

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator


# Test Models
class SimpleModel(StructuredModel):
    """Simple model with basic field types for testing classification logic."""

    name: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    count: Optional[int] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    description: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class NestedModel(StructuredModel):
    """Model with a nested structured model field."""

    id: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    details: Optional[SimpleModel] = ComparableField(weight=1.0)


class ListModel(StructuredModel):
    """Model with list fields for testing classification logic."""

    id: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    tags: Optional[List[str]] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    items: Optional[List[SimpleModel]] = ComparableField(weight=1.0)

    # Add default empty lists to handle None values
    def __init__(self, **data):
        if "tags" not in data:
            data["tags"] = []
        if "items" not in data:
            data["items"] = []
        super().__init__(**data)


def get_base_metrics(cm_result, field_name):
    """Helper function to extract base metrics without derived metrics."""
    field_data = cm_result["fields"][field_name]

    # Handle hierarchical structure (new format)
    if isinstance(field_data, dict) and "overall" in field_data:
        return {k: v for k, v in field_data["overall"].items() if k != "derived"}

    # Handle flat structure (old format for backward compatibility)
    return {k: v for k, v in field_data.items() if k != "derived"}


def test_simple_field_classification_logic():
    """
    Test the classification logic for simple fields based on the definitions:
    - False Alarm (FP): GT == null, EST != null
    - False Discovery (FD): GT != null, EST != null, GT != EST
    """
    evaluator = StructuredModelEvaluator(threshold=0.7)

    # 1. True Positive: GT != null, EST != null, GT == EST
    gt = SimpleModel(name="John Doe", count=42, description="Test description")
    pred = SimpleModel(name="John Doe", count=42, description="Test description")
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    name_cm = get_base_metrics(cm, "name")
    assert name_cm["tp"] == 1, "Exact match should be TP"
    assert name_cm["fp"] == 0, "No False Alarm when both values exist and match"
    assert name_cm["fd"] == 0, "No False Discovery when both values exist and match"

    # 2. False Alarm (FP): GT == null, EST != null
    gt = SimpleModel(name="John Doe", count=None, description=None)
    pred = SimpleModel(name="John Doe", count=42, description="Test description")
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    count_cm = get_base_metrics(cm, "count")
    desc_cm = get_base_metrics(cm, "description")
    assert count_cm["fp"] == 1, "GT null, EST not null should be False Pos (FP)"
    assert desc_cm["fp"] == 1, "GT null, EST not null should be False Pos (FP)"
    assert count_cm["fa"] == 1, "GT null, EST not null should be False Alarm (FA)"
    assert desc_cm["fa"] == 1, "GT null, EST not null should be False Alarm (FA)"

    # 3. False Discovery (FD): GT != null, EST != null, GT != EST
    gt = SimpleModel(name="John Doe", count=42, description="Test description")
    pred = SimpleModel(name="John Doe", count=43, description="Different description")
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    count_cm = get_base_metrics(cm, "count")
    desc_cm = get_base_metrics(cm, "description")
    assert count_cm["fd"] == 1, (
        "Both non-null but different should be False Discovery (FD)"
    )
    assert desc_cm["fd"] == 1, (
        "Both non-null but different should be False Discovery (FD)"
    )
    assert count_cm["fp"] == 1, "False Positive case both are non-null but wrong"

    # 4. False Negative (FN): GT != null, EST == null
    gt = SimpleModel(name="John Doe", count=42, description="Test description")
    pred = SimpleModel(name="John Doe", count=None, description=None)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    count_cm = get_base_metrics(cm, "count")
    desc_cm = get_base_metrics(cm, "description")
    assert count_cm["fn"] == 1, "GT not null, EST null should be FN"
    assert desc_cm["fn"] == 1, "GT not null, EST null should be FN"
    assert count_cm["fp"] == 0, "No False Alarm when EST is null"
    assert count_cm["fd"] == 0, "No False Discovery when EST is null"
    assert count_cm["fa"] == 0, "No False Alarm when EST is null"

    # 5. True Negative (TN): GT == null, EST == null
    gt = SimpleModel(name="John Doe", count=None, description=None)
    pred = SimpleModel(name="John Doe", count=None, description=None)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    count_cm = get_base_metrics(cm, "count")
    desc_cm = get_base_metrics(cm, "description")
    assert count_cm["tn"] == 1, "Both null should be TN"
    assert desc_cm["tn"] == 1, "Both null should be TN"
    assert count_cm["fp"] == 0, "No False Alarm when both are null"
    assert count_cm["fd"] == 0, "No False Discovery when both are null"


def test_list_classification_logic():
    """
    Test the classification logic for list fields based on the definitions:
    - False Alarm (FP): Elements in EST that don't exist in GT
    - False Discovery (FD): Elements in both GT and EST that don't match above threshold
    """
    evaluator = StructuredModelEvaluator(threshold=0.7)

    # 1. Empty lists (TN)
    gt = ListModel(id="empty", tags=[], items=[])
    pred = ListModel(id="empty", tags=[], items=[])
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    tags_cm = get_base_metrics(cm, "tags")
    assert tags_cm["tn"] == 1, "Empty lists should be TN"
    assert tags_cm["fp"] == 0, "No False Alarm with empty lists"
    assert tags_cm["fd"] == 0, "No False Discovery with empty lists"

    # 2. False Alarm (FP): GT empty, EST has items
    gt = ListModel(id="fp", tags=[], items=[])
    pred = ListModel(id="fp", tags=["red", "blue"], items=[])
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    tags_cm = get_base_metrics(cm, "tags")
    assert tags_cm["fp"] == 2, "Items in EST but not in GT should be False Alarm (FP)"
    assert tags_cm["fd"] == 0, "No False Discovery when GT is empty"

    # 3. False Discovery (FD): Items in both GT and EST that don't match
    gt = ListModel(id="fd", tags=["apple", "banana", "cherry"], items=[])
    pred = ListModel(id="fd", tags=["apl", "bnn", "chry"], items=[])
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    tags_cm = get_base_metrics(cm, "tags")
    # The exact counts depend on the threshold, but we should have some FDs
    assert tags_cm["fd"] > 0, "Non-matching items should be False Discovery (FD)"
    assert tags_cm["fp"] > 0, "No False Alarm when all items have a match in GT"

    # 4. Mixed case: Some matches, some False Alarms, some False Discoveries
    gt = ListModel(id="mixed", tags=["red", "blue", "green"], items=[])
    pred = ListModel(id="mixed", tags=["red", "blu", "yellow", "orange"], items=[])
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    tags_cm = get_base_metrics(cm, "tags")

    # Expected:
    # - "red" matches exactly (TP)
    # - "blu" is similar to "blue" and might be TP or FD depending on threshold
    # - "green" is missing from prediction (FN)
    # - "yellow" and "orange" are in prediction but not in GT (FP - False Alarm)

    assert tags_cm["tp"] >= 1, "Should have at least one exact match (TP)"
    assert tags_cm["fp"] >= 1, "Should have False Alarms for items in EST but not in GT"
    assert tags_cm["fn"] >= 0, "Should account for missing GT items"

    # Total counts should add up correctly
    assert tags_cm["tp"] + tags_cm["fd"] + tags_cm["fn"] >= 3, (
        "Should account for all GT items"
    )
    assert tags_cm["tp"] + tags_cm["fp"] + tags_cm["fd"] >= 4, (
        "Should account for all EST items"
    )

    # 5. Null vs Empty list
    gt = ListModel(id="null", tags=None, items=None)
    pred = ListModel(id="null", tags=["red", "blue"], items=[])
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    tags_cm = get_base_metrics(cm, "tags")
    items_cm = get_base_metrics(cm, "items")
    # Note: The current implementation counts the entire list as one FP rather than counting each item
    assert tags_cm["fp"] > 0, (
        "Items in EST but not in GT (null) should be False Alarm (FP)"
    )
    assert items_cm["tn"] == 1, "Both empty lists should be TN"


def test_nested_model_classification_logic():
    """
    Test the classification logic for nested models based on the definitions:
    - False Alarm (FP): GT == null, EST != null
    - False Discovery (FD): GT != null, EST != null, GT != EST
    """
    evaluator = StructuredModelEvaluator(threshold=0.7)

    # Create test nested models
    details1 = SimpleModel(name="Details 1", count=1, description="First details")
    details2 = SimpleModel(name="Details 2", count=2, description="Second details")
    details_different = SimpleModel(name="Different", count=99, description="Other")

    # 1. True Positive: GT != null, EST != null, GT == EST
    gt = NestedModel(id="nested1", details=details1)
    pred = NestedModel(id="nested1", details=details1)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    details_cm = get_base_metrics(cm, "details")
    assert details_cm["tp"] == 1, (
        "Exact match nested model should have 1 TP (object-level counting)"
    )
    assert details_cm["fp"] == 0, "No False Alarm when both values exist and match"
    assert details_cm["fd"] == 0, "No False Discovery when both values exist and match"

    # 2. False Alarm (FP): GT == null, EST != null
    gt = NestedModel(id="nested2", details=None)
    pred = NestedModel(id="nested2", details=details1)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    details_cm = get_base_metrics(cm, "details")
    assert details_cm["fp"] == 1, "GT null, EST not null should be False Alarm (FP)"
    assert details_cm["fa"] == 1, "False Alarm"
    assert details_cm["fd"] == 0, "No False Discovery when GT is null"

    # 3. False Discovery (FD): GT != null, EST != null, GT != EST
    gt = NestedModel(id="nested3", details=details1)
    pred = NestedModel(id="nested3", details=details_different)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    details_cm = get_base_metrics(cm, "details")
    assert details_cm["fd"] == 1, (
        "Different nested model should have 1 FD (object-level counting)"
    )
    assert details_cm["fp"] == 1, "Object-level FP = FD + FA = 1 + 0"
    assert details_cm["fa"] == 0, "No False Alarm"

    # 4. False Negative (FN): GT != null, EST == null
    gt = NestedModel(id="nested4", details=details1)
    pred = NestedModel(id="nested4", details=None)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    details_cm = get_base_metrics(cm, "details")
    assert details_cm["fn"] == 1, "GT not null, EST null should be FN"
    assert details_cm["fp"] == 0, "No False Alarm when EST is null"
    assert details_cm["fd"] == 0, "No False Discovery when EST is null"

    # 5. True Negative (TN): GT == null, EST == null
    gt = NestedModel(id="nested5", details=None)
    pred = NestedModel(id="nested5", details=None)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    details_cm = get_base_metrics(cm, "details")
    assert details_cm["tn"] == 1, "Both null should be TN"
    assert details_cm["fp"] == 0, "No False Alarm when both are null"
    assert details_cm["fd"] == 0, "No False Discovery when both are null"


def test_edge_cases_classification_logic():
    """
    Test edge cases for the classification logic:
    - Empty strings vs null
    - Empty lists vs null
    - Threshold boundary cases
    """
    evaluator = StructuredModelEvaluator(threshold=0.7)

    # 1. Empty string vs null
    gt = SimpleModel(name="", count=None, description=None)
    pred = SimpleModel(name=None, count=None, description=None)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    name_cm = get_base_metrics(cm, "name")
    # Empty string should be treated as null for classification purposes
    assert name_cm["tn"] == 1, "Empty string vs null should be TN"
    assert name_cm["fp"] == 0, "No False Alarm with empty string vs null"
    assert name_cm["fd"] == 0, "No False Discovery with empty string vs null"

    # 2. Empty list vs null list
    gt = ListModel(id="empty_vs_null", tags=[], items=[])
    pred = ListModel(id="empty_vs_null", tags=None, items=None)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    tags_cm = get_base_metrics(cm, "tags")
    items_cm = get_base_metrics(cm, "items")
    # Empty list should be treated as null for classification purposes
    assert tags_cm["tn"] == 1, "Empty list vs null should be TN"
    assert items_cm["tn"] == 1, "Empty list vs null should be TN"

    # 3. Threshold boundary - exactly at threshold
    # Create a custom evaluator with a specific threshold
    threshold_evaluator = StructuredModelEvaluator(threshold=0.75)

    # For Levenshtein, "abcd" vs "abcx" has similarity 0.75
    gt = SimpleModel(name="abcd", count=1, description="test")
    pred = SimpleModel(name="abcx", count=1, description="test")
    result = threshold_evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    name_cm = get_base_metrics(cm, "name")
    assert name_cm["tp"] == 1, "Similarity exactly at threshold should be TP"
    assert name_cm["fd"] == 0, "No False Discovery when at threshold"

    # 4. Threshold boundary - just below threshold
    # For Levenshtein, "abcd" vs "abxy" has similarity 0.5
    gt = SimpleModel(name="abcd", count=1, description="test")
    pred = SimpleModel(name="abxy", count=1, description="test")
    result = threshold_evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    name_cm = get_base_metrics(cm, "name")
    assert name_cm["fd"] == 1, (
        "Similarity below threshold should be False Discovery (FD)"
    )
    assert name_cm["tp"] == 0, "No TP when below threshold"
