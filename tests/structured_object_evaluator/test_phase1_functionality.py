

"""Test Phase 1 refactor functionality for StructuredModel enhanced compare_with method."""

import pytest
from typing import Optional, List
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


class Phase1TestModel(StructuredModel):
    """Test model for Phase 1 functionality."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=1.0,
    )

    age: Optional[int] = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=0.5,
    )

    tags: Optional[List[str]] = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.8,
        weight=0.5,
    )


def test_compare_with_confusion_matrix():
    """Test compare_with with confusion matrix enabled."""
    model1 = Phase1TestModel(name="John Doe", age=30, tags=["developer", "python"])
    model2 = Phase1TestModel(name="John Smith", age=30, tags=["developer", "java"])

    # Test with confusion matrix enabled
    result = model1.compare_with(model2, include_confusion_matrix=True)

    # Check basic structure is preserved
    assert "field_scores" in result
    assert "overall_score" in result
    assert "all_fields_matched" in result

    # Check confusion matrix is included
    assert "confusion_matrix" in result
    assert "fields" in result["confusion_matrix"]
    assert "overall" in result["confusion_matrix"]

    # Check field-level confusion matrix data
    cm_fields = result["confusion_matrix"]["fields"]
    assert "name" in cm_fields
    assert "age" in cm_fields
    assert "tags" in cm_fields

    # Check each field has the required confusion matrix structure
    for field_name, field_cm in cm_fields.items():
        # Helper function to get metrics from new aggregate structure
        def get_metrics_dict(field_data):
            if "overall" in field_data and "fields" in field_data:
                # List field with hierarchical structure
                return field_data["overall"]
            elif "overall" in field_data:
                # New aggregate structure - metrics are in "overall"
                return field_data["overall"]
            else:
                # Simple field with flat structure (fallback)
                return field_data

        metrics = get_metrics_dict(field_cm)

        assert "tp" in metrics
        assert "fa" in metrics
        assert "fd" in metrics
        assert "fp" in metrics
        assert "tn" in metrics
        assert "fn" in metrics
        assert "derived" in metrics

        # Check derived metrics
        derived = metrics["derived"]
        assert "cm_precision" in derived
        assert "cm_recall" in derived
        assert "cm_f1" in derived
        assert "cm_accuracy" in derived

    # Check overall confusion matrix
    overall_cm = result["confusion_matrix"]["overall"]
    assert "tp" in overall_cm
    assert "fa" in overall_cm
    assert "fd" in overall_cm
    assert "fp" in overall_cm
    assert "tn" in overall_cm
    assert "fn" in overall_cm
    assert "derived" in overall_cm

    print("✓ Confusion matrix test passed")


def test_compare_with_non_matches():
    """Test compare_with with non-match documentation enabled."""
    model1 = Phase1TestModel(name="John Doe", age=30, tags=["developer"])
    model2 = Phase1TestModel(
        name="Jane Smith", age=None, tags=["designer"]
    )  # Different values

    # Test with non-match documentation enabled
    result = model1.compare_with(model2, document_non_matches=True)

    # Check basic structure is preserved
    assert "field_scores" in result
    assert "overall_score" in result
    assert "all_fields_matched" in result

    # Check non-matches are included
    assert "non_matches" in result
    assert isinstance(result["non_matches"], list)

    # Should have non-matches since the models are different
    assert len(result["non_matches"]) > 0

    # Check structure of non-match entries
    for non_match in result["non_matches"]:
        assert "field_path" in non_match
        assert "non_match_type" in non_match
        assert "ground_truth_value" in non_match
        assert "prediction_value" in non_match
        # similarity_score is optional

    print("✓ Non-match documentation test passed")


def test_compare_with_both_features():
    """Test compare_with with both confusion matrix and non-match documentation."""
    model1 = Phase1TestModel(name="Alice", age=25, tags=["analyst"])
    model2 = Phase1TestModel(name="Bob", age=30, tags=["developer"])

    # Test with both features enabled
    result = model1.compare_with(
        model2, include_confusion_matrix=True, document_non_matches=True
    )

    # Check all features are present
    assert "field_scores" in result
    assert "overall_score" in result
    assert "all_fields_matched" in result
    assert "confusion_matrix" in result
    assert "non_matches" in result

    print("✓ Combined features test passed")


def test_compare_with_backward_compatibility():
    """Test that compare_with maintains backward compatibility when no optional params used."""
    model1 = Phase1TestModel(name="Test", age=25, tags=["tag1"])
    model2 = Phase1TestModel(name="Test", age=25, tags=["tag1"])

    # Test default behavior (no optional parameters)
    result = model1.compare_with(model2)

    # Should have standard structure only
    assert "field_scores" in result
    assert "overall_score" in result
    assert "all_fields_matched" in result

    # Should NOT have optional features
    assert "confusion_matrix" not in result
    assert "non_matches" not in result

    print("✓ Backward compatibility test passed")


def test_perfect_match_confusion_matrix():
    """Test confusion matrix with perfect match."""
    model1 = Phase1TestModel(name="Perfect", age=42, tags=["exact"])
    model2 = Phase1TestModel(name="Perfect", age=42, tags=["exact"])

    result = model1.compare_with(model2, include_confusion_matrix=True)

    # Should have high true positives, low false positives/negatives
    overall_cm = result["confusion_matrix"]["overall"]
    assert overall_cm["tp"] > 0  # Should have true positives
    assert (
        overall_cm["fp"] == 0 or overall_cm["fp"] < overall_cm["tp"]
    )  # Should have few/no false positives

    print("✓ Perfect match confusion matrix test passed")


if __name__ == "__main__":
    test_compare_with_confusion_matrix()
    test_compare_with_non_matches()
    test_compare_with_both_features()
    test_compare_with_backward_compatibility()
    test_perfect_match_confusion_matrix()
    print("All Phase 1 functionality tests passed!")
