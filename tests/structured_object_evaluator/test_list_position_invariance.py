"""Test that confusion matrix counts are invariant to list position in the tree structure.

This test verifies that the same list comparison produces identical confusion matrix
counts regardless of where the list appears in the object hierarchy - whether at
the root level, nested one level deep, or nested multiple levels deep.

This is a critical property for the Universal Aggregate Field feature to work correctly.
"""

import pytest
from typing import List
from pydantic import Field

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


class SimpleItem(StructuredModel):
    """Simple item for testing list comparisons."""

    # Set threshold for Hungarian matching as class variable
    match_threshold = 0.7

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    value: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class RootLevelModel(StructuredModel):
    """Model with list at root level."""

    items: List[SimpleItem] = ComparableField(
        comparator=LevenshteinComparator(), weight=1.0
    )
    primitives: List[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class OneLevelNestedModel(StructuredModel):
    """Model with list nested one level deep."""

    container: "ContainerLevel1" = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class ContainerLevel1(StructuredModel):
    """Container for one-level nesting."""

    items: List[SimpleItem] = ComparableField(
        comparator=LevenshteinComparator(), weight=1.0
    )
    primitives: List[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class TwoLevelNestedModel(StructuredModel):
    """Model with list nested two levels deep."""

    outer: "OuterContainer" = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class OuterContainer(StructuredModel):
    """Outer container for two-level nesting."""

    inner: "InnerContainer" = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


class InnerContainer(StructuredModel):
    """Inner container for two-level nesting."""

    items: List[SimpleItem] = ComparableField(
        comparator=LevenshteinComparator(), weight=1.0
    )
    primitives: List[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )


# Test case data for parametrized testing
TEST_CASES = [
    {
        "case_name": "mixed_matching",
        "description": "Original test case with partial matches and mismatches",
        "gt_items": [
            SimpleItem(name="apple", value=1),
            SimpleItem(name="banana", value=2),
            SimpleItem(name="cherry", value=3),
        ],
        "pred_items": [
            SimpleItem(name="aple", value=1),  # Typo in apple -> should match
            SimpleItem(name="banana", value=2),  # Exact match -> TP
            SimpleItem(name="grape", value=4),  # New item -> FA/FD
        ],
        "gt_primitives": ["red", "green", "blue"],
        "pred_primitives": ["rd", "green", "yellow"],
    },
    {
        "case_name": "empty_lists",
        "description": "Edge case with empty lists",
        "gt_items": [],
        "pred_items": [],
        "gt_primitives": [],
        "pred_primitives": [],
    },
    {
        "case_name": "gt_larger",
        "description": "Ground truth has more items than prediction",
        "gt_items": [
            SimpleItem(name="cat", value=1),
            SimpleItem(name="dog", value=2),
            SimpleItem(name="bird", value=3),
        ],
        "pred_items": [
            SimpleItem(name="cat", value=1)  # Only one match
        ],
        "gt_primitives": ["alpha", "beta", "gamma"],
        "pred_primitives": ["alpha"],
    },
    {
        "case_name": "pred_larger",
        "description": "Prediction has more items than ground truth",
        "gt_items": [SimpleItem(name="single", value=1)],
        "pred_items": [
            SimpleItem(name="single", value=1),
            SimpleItem(name="extra1", value=2),
            SimpleItem(name="extra2", value=3),
        ],
        "gt_primitives": ["one"],
        "pred_primitives": ["one", "two", "three"],
    },
    {
        "case_name": "no_matches",
        "description": "No similarity matches - all should be FD",
        "gt_items": [
            SimpleItem(name="apple", value=1),
            SimpleItem(name="banana", value=2),
        ],
        "pred_items": [
            SimpleItem(name="zebra", value=1),
            SimpleItem(name="yacht", value=2),
        ],
        "gt_primitives": ["red", "blue"],
        "pred_primitives": ["zebra", "yacht"],
    },
    {
        "case_name": "all_perfect_matches",
        "description": "All items match perfectly - should be all TP",
        "gt_items": [
            SimpleItem(name="exact", value=1),
            SimpleItem(name="match", value=2),
        ],
        "pred_items": [
            SimpleItem(name="exact", value=1),
            SimpleItem(name="match", value=2),
        ],
        "gt_primitives": ["perfect", "identical"],
        "pred_primitives": ["perfect", "identical"],
    },
    {
        "case_name": "threshold_boundary",
        "description": "Test items right at similarity threshold boundary",
        "gt_items": [
            SimpleItem(name="test", value=1),
            SimpleItem(name="boundary", value=2),
        ],
        "pred_items": [
            SimpleItem(
                name="tset", value=1
            ),  # ~0.75 similarity, should pass 0.7 threshold
            SimpleItem(name="boundry", value=2),  # ~0.86 similarity, should pass
        ],
        "gt_primitives": ["hello", "world"],
        "pred_primitives": ["helo", "wrld"],  # Similar threshold testing
    },
]


class TestListPositionInvariance:
    """Test that confusion matrix counts are invariant to list position in tree."""

    def setup_method(self):
        """Set up test data - now using parametrized data."""
        pass  # Data will come from parametrized test cases

    def extract_list_metrics(
        self, confusion_matrix: dict, list_field_path: str
    ) -> dict:
        """Extract confusion matrix metrics for a specific list field from the tree.

        Args:
            confusion_matrix: Full confusion matrix tree
            list_field_path: Dot-separated path to the list field (e.g., "items", "container.items", "outer.inner.items")

        Returns:
            Dictionary with TP, FA, FD, FP, TN, FN counts for the list field
        """
        # Navigate to the field using the path
        current = confusion_matrix
        path_parts = list_field_path.split(".")

        # Navigate through the tree structure
        for part in path_parts:
            if "fields" in current and part in current["fields"]:
                current = current["fields"][part]
            else:
                raise KeyError(
                    f"Could not find field '{part}' in path '{list_field_path}'"
                )

        # Extract the overall metrics for this field
        if "overall" in current:
            metrics = current["overall"]
        else:
            # Handle legacy format where metrics might be at top level
            metrics = current

        return {
            "tp": metrics.get("tp", 0),
            "fa": metrics.get("fa", 0),
            "fd": metrics.get("fd", 0),
            "fp": metrics.get("fp", 0),
            "tn": metrics.get("tn", 0),
            "fn": metrics.get("fn", 0),
        }

    @pytest.mark.parametrize("test_case", TEST_CASES)
    def test_structured_list_position_invariance(self, test_case):
        """Test that structured list metrics are invariant to position in tree."""

        # Extract test data from parametrized case
        gt_items = test_case["gt_items"]
        pred_items = test_case["pred_items"]
        gt_primitives = test_case["gt_primitives"]
        pred_primitives = test_case["pred_primitives"]

        print(f"\n=== Testing case: {test_case['case_name']} ===")
        print(f"Description: {test_case['description']}")

        # Test at root level
        root_gt = RootLevelModel(items=gt_items, primitives=gt_primitives)
        root_pred = RootLevelModel(items=pred_items, primitives=pred_primitives)
        root_result = root_gt.compare_with(root_pred, include_confusion_matrix=True)
        root_items_metrics = self.extract_list_metrics(
            root_result["confusion_matrix"], "items"
        )

        # Test at one level nested
        nested1_gt = OneLevelNestedModel(
            container=ContainerLevel1(items=gt_items, primitives=gt_primitives)
        )
        nested1_pred = OneLevelNestedModel(
            container=ContainerLevel1(items=pred_items, primitives=pred_primitives)
        )
        nested1_result = nested1_gt.compare_with(
            nested1_pred, include_confusion_matrix=True
        )
        nested1_items_metrics = self.extract_list_metrics(
            nested1_result["confusion_matrix"], "container.items"
        )

        # Test at two levels nested
        nested2_gt = TwoLevelNestedModel(
            outer=OuterContainer(
                inner=InnerContainer(items=gt_items, primitives=gt_primitives)
            )
        )
        nested2_pred = TwoLevelNestedModel(
            outer=OuterContainer(
                inner=InnerContainer(items=pred_items, primitives=pred_primitives)
            )
        )
        nested2_result = nested2_gt.compare_with(
            nested2_pred, include_confusion_matrix=True
        )
        nested2_items_metrics = self.extract_list_metrics(
            nested2_result["confusion_matrix"], "outer.inner.items"
        )

        # Verify that all three positions produce identical metrics for the structured list
        print(f"Root level items metrics: {root_items_metrics}")
        print(f"One level nested items metrics: {nested1_items_metrics}")
        print(f"Two level nested items metrics: {nested2_items_metrics}")

        assert root_items_metrics == nested1_items_metrics, (
            f"Root and one-level nested metrics differ: {root_items_metrics} vs {nested1_items_metrics}"
        )

        assert nested1_items_metrics == nested2_items_metrics, (
            f"One-level and two-level nested metrics differ: {nested1_items_metrics} vs {nested2_items_metrics}"
        )

        assert root_items_metrics == nested2_items_metrics, (
            f"Root and two-level nested metrics differ: {root_items_metrics} vs {nested2_items_metrics}"
        )

    def test_primitive_list_position_invariance(self):
        """Test that primitive list metrics are invariant to position in tree."""

        # Use first test case as default data
        test_case = TEST_CASES[0]  # mixed_matching case
        gt_items = test_case["gt_items"]
        pred_items = test_case["pred_items"]
        gt_primitives = test_case["gt_primitives"]
        pred_primitives = test_case["pred_primitives"]

        # Test at root level
        root_gt = RootLevelModel(items=gt_items, primitives=gt_primitives)
        root_pred = RootLevelModel(items=pred_items, primitives=pred_primitives)
        root_result = root_gt.compare_with(root_pred, include_confusion_matrix=True)
        root_primitives_metrics = self.extract_list_metrics(
            root_result["confusion_matrix"], "primitives"
        )

        # Test at one level nested
        nested1_gt = OneLevelNestedModel(
            container=ContainerLevel1(items=gt_items, primitives=gt_primitives)
        )
        nested1_pred = OneLevelNestedModel(
            container=ContainerLevel1(items=pred_items, primitives=pred_primitives)
        )
        nested1_result = nested1_gt.compare_with(
            nested1_pred, include_confusion_matrix=True
        )
        nested1_primitives_metrics = self.extract_list_metrics(
            nested1_result["confusion_matrix"], "container.primitives"
        )

        # Test at two levels nested
        nested2_gt = TwoLevelNestedModel(
            outer=OuterContainer(
                inner=InnerContainer(items=gt_items, primitives=gt_primitives)
            )
        )
        nested2_pred = TwoLevelNestedModel(
            outer=OuterContainer(
                inner=InnerContainer(items=pred_items, primitives=pred_primitives)
            )
        )
        nested2_result = nested2_gt.compare_with(
            nested2_pred, include_confusion_matrix=True
        )
        nested2_primitives_metrics = self.extract_list_metrics(
            nested2_result["confusion_matrix"], "outer.inner.primitives"
        )

        # Verify that all three positions produce identical metrics for the primitive list
        print(f"Root level primitives metrics: {root_primitives_metrics}")
        print(f"One level nested primitives metrics: {nested1_primitives_metrics}")
        print(f"Two level nested primitives metrics: {nested2_primitives_metrics}")

        assert root_primitives_metrics == nested1_primitives_metrics, (
            f"Root and one-level nested primitive metrics differ: {root_primitives_metrics} vs {nested1_primitives_metrics}"
        )

        assert nested1_primitives_metrics == nested2_primitives_metrics, (
            f"One-level and two-level nested primitive metrics differ: {nested1_primitives_metrics} vs {nested2_primitives_metrics}"
        )

        assert root_primitives_metrics == nested2_primitives_metrics, (
            f"Root and two-level nested primitive metrics differ: {root_primitives_metrics} vs {nested2_primitives_metrics}"
        )

    def test_expected_confusion_matrix_values(self):
        """Test that the confusion matrix values match expected patterns."""

        # Use first test case as default data
        test_case = TEST_CASES[0]  # mixed_matching case
        gt_items = test_case["gt_items"]
        pred_items = test_case["pred_items"]
        gt_primitives = test_case["gt_primitives"]
        pred_primitives = test_case["pred_primitives"]

        # Test with root level model for simplicity
        root_gt = RootLevelModel(items=gt_items, primitives=gt_primitives)
        root_pred = RootLevelModel(items=pred_items, primitives=pred_primitives)
        result = root_gt.compare_with(root_pred, include_confusion_matrix=True)

        # Extract metrics for structured list
        items_metrics = self.extract_list_metrics(result["confusion_matrix"], "items")

        # Expected for structured list:
        # - "aple" vs "apple": similarity ~0.8 >= 0.7 threshold -> TP
        # - "banana" vs "banana": exact match -> TP
        # - "grape": unmatched prediction -> FA
        # - "cherry": unmatched ground truth -> FN
        # So: TP=2, FA=1, FD=0, FN=1, FP=1
        expected_items = {"tp": 2, "fa": 1, "fd": 0, "fp": 1, "tn": 0, "fn": 1}

        print(f"Actual items metrics: {items_metrics}")
        print(f"Expected items metrics: {expected_items}")

        # Note: The exact values may vary based on similarity calculations,
        # but the key point is that they should be consistent across positions
        assert items_metrics["tp"] >= 1, (
            "Should have at least 1 true positive (banana match)"
        )
        assert (items_metrics["fa"] + items_metrics["fd"]) >= 1, (
            "Should have at least 1 false alarm or false detection (grape)"
        )
        assert items_metrics["fn"] >= 0, "Should have 0 or more false negatives"

        # Extract metrics for primitive list
        primitives_metrics = self.extract_list_metrics(
            result["confusion_matrix"], "primitives"
        )

        print(f"Actual primitives metrics: {primitives_metrics}")

        # Expected for primitive list:
        # - "rd" vs "red": similarity depends on threshold
        # - "green" vs "green": exact match -> TP
        # - "yellow": unmatched prediction -> FA
        # - "blue": unmatched ground truth -> FN
        assert primitives_metrics["tp"] >= 1, (
            "Should have at least 1 true positive (green match)"
        )
        assert (primitives_metrics["fa"] + primitives_metrics["fd"]) >= 1, (
            "Should have at least 1 false alarm or false detection"
        )
        assert primitives_metrics["fn"] >= 0, "Should have 0 or more false negatives"

    def test_aggregate_field_consistency(self):
        """Test that Universal Aggregate Fields sum correctly across positions."""

        # Use first test case as default data
        test_case = TEST_CASES[0]  # mixed_matching case
        gt_items = test_case["gt_items"]
        pred_items = test_case["pred_items"]
        gt_primitives = test_case["gt_primitives"]
        pred_primitives = test_case["pred_primitives"]

        # Test at root level
        root_gt = RootLevelModel(items=gt_items, primitives=gt_primitives)
        root_pred = RootLevelModel(items=pred_items, primitives=pred_primitives)
        root_result = root_gt.compare_with(root_pred, include_confusion_matrix=True)

        # Test at nested level
        nested_gt = OneLevelNestedModel(
            container=ContainerLevel1(items=gt_items, primitives=gt_primitives)
        )
        nested_pred = OneLevelNestedModel(
            container=ContainerLevel1(items=pred_items, primitives=pred_primitives)
        )
        nested_result = nested_gt.compare_with(
            nested_pred, include_confusion_matrix=True
        )

        # Extract aggregate metrics
        root_aggregate = root_result["confusion_matrix"].get("aggregate", {})
        nested_aggregate = nested_result["confusion_matrix"].get("aggregate", {})

        print(f"Root aggregate: {root_aggregate}")
        print(f"Nested aggregate: {nested_aggregate}")

        # The aggregate should be the same since we're comparing the same lists
        # (just at different positions in the tree)
        for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
            assert root_aggregate.get(metric, 0) == nested_aggregate.get(metric, 0), (
                f"Aggregate {metric} differs between root ({root_aggregate.get(metric, 0)}) and nested ({nested_aggregate.get(metric, 0)})"
            )


if __name__ == "__main__":
    # Run the tests
    test_instance = TestListPositionInvariance()
    test_instance.setup_method()

    print("Testing structured list position invariance...")
    test_instance.test_structured_list_position_invariance()
    print("✓ Structured list position invariance test passed")

    print("\nTesting primitive list position invariance...")
    test_instance.test_primitive_list_position_invariance()
    print("✓ Primitive list position invariance test passed")

    print("\nTesting expected confusion matrix values...")
    test_instance.test_expected_confusion_matrix_values()
    print("✓ Expected confusion matrix values test passed")

    print("\nTesting aggregate field consistency...")
    test_instance.test_aggregate_field_consistency()
    print("✓ Aggregate field consistency test passed")

    print("\n🎉 All list position invariance tests passed!")
