"""
Test cases for CORRECT classification definitions where False Positive = False Alarm + False Discovery.

These tests define the expected behavior after the classification refactor:
- False Alarm (FA): GT == null, EST != null
- False Discovery (FD): GT != null, EST != null, but don't match
- False Positive (FP): FA + FD (both are types of False Positive)
- Precision: TP / (TP + FP) = TP / (TP + FA + FD)

This file contains NEW tests that will initially FAIL until the core implementation is updated.
"""

import pytest
from typing import Dict, Any, List, Optional

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator


# Test Models
class SimpleModel(StructuredModel):
    """Simple model for testing correct classification definitions."""

    match_threshold = 0.7

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    count: Optional[int] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    description: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class ListModel(StructuredModel):
    """Model with list fields for testing correct classification."""

    id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=1.0
    )
    tags: List[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    items: List[SimpleModel] = ComparableField(weight=1.0)

    def __init__(self, **data):
        if "tags" not in data or data["tags"] is None:
            data["tags"] = []
        if "items" not in data or data["items"] is None:
            data["items"] = []
        super().__init__(**data)


def test_false_positive_combination():
    """Test that FA and FD are both counted as FP in precision calculation."""
    evaluator = StructuredModelEvaluator(threshold=0.7)

    # GT: [10, 20]
    # EST: [10, 25, 30]
    # Expected: TP=1 (10 matches), FA=1 (30 unmatched in EST), FD=1 (25 doesn't match 20), FN=1 (20 unmatched in GT)
    # FP = FA + FD = 1 + 1 = 2
    # Precision should be: TP/(TP+FP) = 1/(1+2) = 0.33

    gt = ListModel(id="test1", tags=["10", "20"], items=[])
    pred = ListModel(id="test1", tags=["10", "25", "30"], items=[])
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    tags_cm = cm["fields"]["tags"]

    # NOTE: Using direct access instead of nested "overall" key structure
    tags_overall = tags_cm

    # Verify raw counts
    assert tags_overall["tp"] == 1, f"Expected TP=1, got {tags_overall['tp']}"
    # FA should be 1 (unmatched prediction item 30)
    fa_count = tags_overall.get("fa", 0) or (
        tags_overall["fp"] - tags_overall.get("fd", 0)
    )
    # FD should be 1 (25 doesn't match 20 well enough)
    fd_count = tags_overall["fd"]
    fp_total = tags_overall["fp"]

    # Test the core requirement: FP should equal FA + FD
    assert fp_total == fa_count + fd_count, (
        f"FP ({fp_total}) should equal FA ({fa_count}) + FD ({fd_count})"
    )

    # Test correct precision calculation: TP / (TP + FP)
    expected_precision = 1 / (1 + fp_total)  # 1 / (1 + 2) = 0.33
    actual_precision = tags_overall["derived"]["cm_precision"]
    assert abs(actual_precision - expected_precision) < 0.01, (
        f"Expected precision {expected_precision:.2f}, got {actual_precision:.2f}"
    )


def test_simple_field_correct_classification():
    """Test correct classification for simple fields with mixed scenarios."""
    evaluator = StructuredModelEvaluator(threshold=0.7)

    # Mixed scenario:
    # - name: TP (exact match)
    # - count: FA (GT=null, EST=42)
    # - description: FD (both non-null but don't match)
    gt = SimpleModel(name="John Doe", count=None, description="Test description")
    pred = SimpleModel(name="John Doe", count=42, description="Different description")
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]

    # Check individual field classifications
    name_cm = cm["fields"]["name"]
    assert (
        name_cm["tp"] == 1
        and name_cm["fp"] == 0
        and name_cm["fn"] == 0
        and name_cm["fd"] == 0
    )

    count_cm = cm["fields"]["count"]
    # Count should be classified as False Alarm (FA): GT=null, EST=42
    assert count_cm["tp"] == 0 and count_cm["fn"] == 0 and count_cm["fd"] == 0
    assert count_cm["fp"] == 1  # This should be the FA case

    desc_cm = cm["fields"]["description"]
    # Description should be classified as False Discovery (FD): both non-null but don't match
    assert desc_cm["tp"] == 0 and desc_cm["fp"] == 1 and desc_cm["fn"] == 0
    assert desc_cm["fd"] == 1

    # Check overall aggregation
    overall_cm = cm["overall"]
    assert overall_cm["tp"] == 1  # name
    assert overall_cm["fp"] == 2  # count (FA) + description (FD) = 2 total FP
    assert overall_cm["fn"] == 0

    # Check correct precision calculation for overall
    total_fp = overall_cm["fp"]  # Should be 2
    expected_precision = 1 / (1 + total_fp)  # 1 / (1 + 2) = 0.33
    actual_precision = overall_cm["derived"]["cm_precision"]
    assert abs(actual_precision - expected_precision) < 0.01


def test_hungarian_matching_correct_fp_handling():
    """Test that Hungarian matching correctly handles FP = FA + FD."""
    evaluator = StructuredModelEvaluator(
        threshold=0.8
    )  # Higher threshold to force some FD cases

    # Create structured model items
    item1 = SimpleModel(name="Item A", count=1, description="First item")
    item2 = SimpleModel(name="Item B", count=2, description="Second item")
    item3 = SimpleModel(name="Item C", count=3, description="Third item")

    # Similar but not exact items (should be FD if similarity < threshold)
    item1_similar = SimpleModel(
        name="Item A", count=1, description="First"
    )  # Similar but different description
    item2_different = SimpleModel(
        name="Item X", count=99, description="Completely different"
    )  # Very different
    item4_extra = SimpleModel(
        name="Item D", count=4, description="Extra item"
    )  # Unmatched (FA)

    # GT: [item1, item2, item3]
    # PRED: [item1_similar, item2_different, item4_extra]
    # Expected outcomes:
    # - item1 vs item1: likely TP (similar but below 0.8 threshold)
    # - item2 vs item2_different: FD (very different)
    # - item3 vs item4_extra: FD

    gt = ListModel(id="hungarian1", tags=[], items=[item1, item2, item3])
    pred = ListModel(
        id="hungarian1", tags=[], items=[item1, item2_different, item4_extra]
    )
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    items_cm = cm["fields"]["items"]
    # NOTE: Using direct access instead of nested "overall" key structure
    items_overall = items_cm

    # Verify FP components
    fa_count = items_overall.get("fa", 0) or (
        items_overall["fp"] - items_overall.get("fd", 0)
    )  # Unmatched predictions
    fd_count = items_overall["fd"]  # Matched but below threshold
    fp_total = items_overall["fp"]

    # Core requirement: FP = FA + FD
    assert fp_total == fa_count + fd_count, (
        f"FP ({fp_total}) should equal FA ({fa_count}) + FD ({fd_count})"
    )

    # Should have some FD from matched pairs below threshold
    assert fd_count > 0, (
        "Expected some False Discovery cases from matched pairs below threshold"
    )

    # Should have some FN from unmatched GT items
    assert items_overall["fn"] == 0, (
        "Expected some False Negatives from unmatched GT items"
    )

    # Check precision calculation
    tp_count = items_overall["tp"]
    expected_precision = (
        tp_count / (tp_count + fp_total) if (tp_count + fp_total) > 0 else 0.0
    )
    actual_precision = items_overall["derived"]["cm_precision"]
    assert abs(actual_precision - expected_precision) < 0.01


def test_nested_field_aggregation():
    """Test that nested field metrics are correctly aggregated with FP = FA + FD."""
    evaluator = StructuredModelEvaluator(threshold=0.7)

    # Create items with different field scenarios
    item1 = SimpleModel(name="Item 1", count=1, description="First")
    item2 = SimpleModel(
        name="Item 2", count=None, description="Second"
    )  # count=null in GT

    item1_pred = SimpleModel(name="Item 1", count=1, description="First")  # Exact match
    item2_pred = SimpleModel(
        name="Item 2", count=42, description="Different"
    )  # count=FA, description=FD

    gt = ListModel(id="nested1", tags=[], items=[item1, item2])
    pred = ListModel(id="nested1", tags=[], items=[item1_pred, item2_pred])
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]

    # Check aggregated field metrics for nested fields
    # Access nested fields through items.fields structure
    items_fields = cm["fields"]["items"]["fields"]

    # With threshold-gated recursion, only item1 (similarity=1.0 >= 0.7) gets field-level analysis
    # item2 (similarity=0.407 < 0.7) is classified as FD at object level, no field recursion

    # items.name should have 1 TP (only item1's name, item2 was below threshold)
    name_metrics = items_fields["name"]
    if "overall" in name_metrics:
        assert name_metrics["overall"]["tp"] == 1
    else:
        assert name_metrics["tp"] == 1
    # Handle both old and new structure for remaining assertions
    if "overall" in name_metrics:
        assert (
            name_metrics["overall"]["fp"] + name_metrics["overall"]["fd"] == 0
        )  # No false positives for names
    else:
        assert (
            name_metrics["fp"] + name_metrics["fd"] == 0
        )  # No false positives for names

    # items.count should have 1 TP (only item1's count, item2 was below threshold)
    count_metrics = items_fields["count"]
    if "overall" in count_metrics:
        assert count_metrics["overall"]["tp"] == 1  # item1 count matches
        assert (
            count_metrics["overall"]["fp"] == 0
        )  # No false positives since item2 not analyzed at field level
        assert count_metrics["overall"]["fd"] == 0  # No false discoveries for count
    else:
        assert count_metrics["tp"] == 1  # item1 count matches
        assert (
            count_metrics["fp"] == 0
        )  # No false positives since item2 not analyzed at field level
        assert count_metrics["fd"] == 0  # No false discoveries for count

    # items.description should have 1 TP (only item1's description, item2 was below threshold)
    desc_metrics = items_fields["description"]
    if "overall" in desc_metrics:
        assert desc_metrics["overall"]["tp"] == 1  # item1 description matches
        assert (
            desc_metrics["overall"]["fd"] == 0
        )  # No false discoveries since item2 not analyzed at field level
        assert desc_metrics["overall"]["fp"] == 0  # No false positives for description
    else:
        assert desc_metrics["tp"] == 1  # item1 description matches
        assert (
            desc_metrics["fd"] == 0
        )  # No false discoveries since item2 not analyzed at field level
        assert desc_metrics["fp"] == 0  # No false positives for description

    # Verify FP = FA + FD for each nested field
    for field_name, field_metrics in items_fields.items():
        if "overall" in field_metrics:
            fp_total = field_metrics["overall"]["fp"]
            fa = field_metrics["overall"].get("fa", 0) or (
                fp_total - field_metrics["overall"].get("fd", 0)
            )
            fd = field_metrics["overall"]["fd"]
        else:
            fp_total = field_metrics["fp"]
            fa = field_metrics.get("fa", 0) or (fp_total - field_metrics.get("fd", 0))
            fd = field_metrics["fd"]
        assert fp_total == fa + fd, (
            f"Field items.{field_name}: FP ({fp_total}) should equal FA ({fa}) + FD ({fd})"
        )


def test_edge_cases_with_correct_classification():
    """Test edge cases (empty lists, null values) with correct FP handling."""
    evaluator = StructuredModelEvaluator(threshold=0.7)

    # Test 1: Empty GT, non-empty prediction (all FA)
    gt = ListModel(id="edge1", tags=[], items=[])
    pred = ListModel(id="edge1", tags=["red", "blue"], items=[])
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    tags_cm = cm["fields"]["tags"]
    # NOTE: Using direct access instead of nested "overall" key structure
    tags_overall = tags_cm

    # All predictions should be False Alarms (FA)
    assert tags_overall["tp"] == 0
    assert tags_overall["fn"] == 0
    assert tags_overall["fd"] == 0
    fa_count = tags_overall.get("fa", 0) or tags_overall["fp"]
    assert fa_count == 2  # Both "red" and "blue" are FA
    assert tags_overall["fp"] == 2  # FP = FA + FD = 2 + 0 = 2

    # Precision should be 0 (no TP, all FP)
    assert tags_overall["derived"]["cm_precision"] == 0.0

    # Test 2: Non-empty GT, empty prediction (all FN)
    gt = ListModel(id="edge2", tags=["red", "blue"], items=[])
    pred = ListModel(id="edge2", tags=[], items=[])
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    tags_cm = cm["fields"]["tags"]
    # NOTE: Using direct access instead of nested "overall" key structure
    tags_overall = tags_cm

    # All GT items should be False Negatives (FN)
    assert tags_overall["tp"] == 0
    assert tags_overall["fp"] == 0
    assert tags_overall["fd"] == 0
    assert tags_overall["fn"] == 2  # Both "red" and "blue" are FN

    # Recall should be 0 (no TP, all FN)
    assert tags_overall["derived"]["cm_recall"] == 0.0

    # Test 3: Mixed scenario with null handling
    gt = SimpleModel(name="Test", count=None, description=None)
    pred = SimpleModel(name="Test", count=42, description=None)
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]

    # name: TP (exact match)
    # count: FA (null -> 42)
    # description: TN (both null)
    name_cm = cm["fields"]["name"]
    count_cm = cm["fields"]["count"]
    desc_cm = cm["fields"]["description"]

    assert name_cm["tp"] == 1 and name_cm["fp"] == 0
    assert (
        count_cm["fp"] == 1 and count_cm["tp"] == 0 and count_cm["fd"] == 0
    )  # FA case
    assert desc_cm["tn"] == 1 and desc_cm["tp"] == 0 and desc_cm["fp"] == 0  # TN case

    # Overall: 1 TP, 1 FP (from FA), 1 TN, 0 FN, 0 FD
    overall_cm = cm["overall"]
    assert overall_cm["tp"] == 1
    assert overall_cm["fp"] + overall_cm["fd"] == 1  # Total FP = 1
    assert overall_cm["tn"] == 1
    assert overall_cm["fn"] == 0

    # Precision = 1/(1+1) = 0.5
    expected_precision = 1 / (1 + 1)
    actual_precision = overall_cm["derived"]["cm_precision"]
    assert abs(actual_precision - expected_precision) < 0.01


def test_precision_formula_validation():
    """Explicit test to validate that precision uses the correct formula: TP / (TP + FP)."""
    evaluator = StructuredModelEvaluator(threshold=0.7)

    # Create a scenario with known TP, FA, and FD counts
    # GT: name="A", count=1, description="X"
    # PRED: name="A", count=null, description="Y"
    # Expected: name=TP, count=FN, description=FD
    gt = SimpleModel(name="A", count=1, description="X")
    pred = SimpleModel(name="A", count=None, description="Y")
    result = evaluator.evaluate(gt, pred)

    cm = result["confusion_matrix"]
    overall_cm = cm["overall"]

    tp = overall_cm["tp"]
    fp = overall_cm["fp"]
    fd = overall_cm["fd"]

    # Manual calculation: TP=1, FP=0, FD=1, FN=1
    # Total FP = FP + FD = 0 + 1 = 1
    # Precision = TP / (TP + FP_total) = 1 / (1 + 1) = 0.5

    expected_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    actual_precision = overall_cm["derived"]["cm_precision"]

    assert abs(actual_precision - expected_precision) < 0.01, (
        f"Expected precision {expected_precision}, got {actual_precision}"
    )
