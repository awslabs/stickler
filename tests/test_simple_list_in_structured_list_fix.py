"""
Tests for simple list in structured list bug fix.

This test file verifies that simple lists (List[primitive]) within structured lists
are correctly counted element-by-element rather than as single items.

Bug Report: https://github.com/your-repo/issues/33
Expected: LineItemDays with ['M', 'T', 'W', 'Th', 'F'] + ['Su'] should give tp=6
Current Bug: Gives tp=2 (counts entire lists as single items)
"""

import pytest
from typing import Optional, List, Any

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField


class LineItemsInfo(StructuredModel):
    """Model for line items with simple list field."""
    LineItemDays: Optional[List[str]] | Any = ComparableField(weight=1.0)
    match_threshold = 1.0


class Invoice(StructuredModel):
    """Model for invoice with structured list of line items."""
    LineItems: Optional[List[LineItemsInfo]] | Any = ComparableField(weight=1.0)


class TestSimpleListBugFix:
    """Test suite for the simple list bug fix."""

    def test_simple_list_element_counting_self_comparison(self):
        """Test that simple lists count each element when comparing against itself.
        
        This is the core bug: simple lists should count elements, not the list as 1 item.
        """
        gt_data = {
            "LineItems": [
                {'LineItemDays': ['M', 'T', 'W', 'Th', 'F']},  # 5 elements
                {'LineItemDays': ['Su']}                        # 1 element
            ]
        }
        pred_data = gt_data
        
        gt_model = Invoice(**gt_data)
        pred_model = Invoice(**pred_data)
        
        comparison_result = gt_model.compare_with(
            pred_model,
            include_confusion_matrix=True,
            document_non_matches=False,
        )
        
        aggregate = comparison_result['confusion_matrix']['aggregate']
        
        # Expected: 6 TP (5 from first item + 1 from second item)
        # Bug was: 2 TP (counting entire lists as single items)
        assert aggregate['tp'] == 6, f"Expected tp=6, got tp={aggregate['tp']}"
        assert aggregate['fa'] == 0
        assert aggregate['fd'] == 0
        assert aggregate['fp'] == 0
        assert aggregate['tn'] == 0
        assert aggregate['fn'] == 0

    def test_simple_list_partial_match(self):
        """Test simple list counting with partial matches.
        
        Note: With match_threshold=1.0, a partial match (similarity=0.667) is below threshold
        and treated as a bad match (FD at object level). To test field-level counting for
        partial matches, we need a lower threshold.
        """
        # Create a model with lower threshold to allow partial matches
        class LineItemsInfoLowThreshold(StructuredModel):
            LineItemDays: Optional[List[str]] | Any = ComparableField(weight=1.0)
            match_threshold = 0.5  # Allow partial matches
        
        class InvoiceLowThreshold(StructuredModel):
            LineItems: Optional[List[LineItemsInfoLowThreshold]] | Any = ComparableField(weight=1.0)
        
        gt_data = {
            "LineItems": [
                {'LineItemDays': ['M', 'T', 'W']},  # 3 elements
            ]
        }
        pred_data = {
            "LineItems": [
                {'LineItemDays': ['M', 'T', 'X']},  # 2 match, 1 mismatch
            ]
        }
        
        gt_model = InvoiceLowThreshold(**gt_data)
        pred_model = InvoiceLowThreshold(**pred_data)
        
        comparison_result = gt_model.compare_with(
            pred_model,
            include_confusion_matrix=True,
            document_non_matches=False,
        )
        
        aggregate = comparison_result['confusion_matrix']['aggregate']
        
        # Should count individual element matches (similarity=0.667 >= threshold=0.5)
        assert aggregate['tp'] == 2, f"Expected tp=2 (M, T match), got tp={aggregate['tp']}"
        assert aggregate['fd'] == 1, f"Expected fd=1 (W vs X mismatch), got fd={aggregate['fd']}"
        assert aggregate['fp'] == 1, f"Expected fp=1, got fp={aggregate['fp']}"

    def test_simple_list_unmatched_gt_items(self):
        """Test that unmatched GT items count each list element as FN."""
        gt_data = {
            "LineItems": [
                {'LineItemDays': ['M', 'T', 'W']},      # Object 1 (matches)
                {'LineItemDays': ['Th', 'F']}           # Object 2 (unmatched)
            ]
        }
        pred_data = {
            "LineItems": [
                {'LineItemDays': ['M', 'T', 'W']},      # Object 1 (matches)
            ]
        }
        
        gt_model = Invoice(**gt_data)
        pred_model = Invoice(**pred_data)
        
        comparison_result = gt_model.compare_with(
            pred_model,
            include_confusion_matrix=True,
            document_non_matches=False,
        )
        
        aggregate = comparison_result['confusion_matrix']['aggregate']
        
        # Object 1: 3 TP (M, T, W match)
        # Object 2: 2 FN (Th, F missing from prediction)
        assert aggregate['tp'] == 3, f"Expected tp=3, got tp={aggregate['tp']}"
        assert aggregate['fn'] == 2, f"Expected fn=2 (Th, F missing), got fn={aggregate['fn']}"

    def test_simple_list_unmatched_pred_items(self):
        """Test that unmatched pred items count each list element as FA/FP."""
        gt_data = {
            "LineItems": [
                {'LineItemDays': ['M', 'T', 'W']},      # Object 1 (matches)
            ]
        }
        pred_data = {
            "LineItems": [
                {'LineItemDays': ['M', 'T', 'W']},      # Object 1 (matches)
                {'LineItemDays': ['Sa', 'Su']}          # Object 2 (unmatched)
            ]
        }
        
        gt_model = Invoice(**gt_data)
        pred_model = Invoice(**pred_data)
        
        comparison_result = gt_model.compare_with(
            pred_model,
            include_confusion_matrix=True,
            document_non_matches=False,
        )
        
        aggregate = comparison_result['confusion_matrix']['aggregate']
        
        # Object 1: 3 TP (M, T, W match)
        # Object 2: 2 FA (Sa, Su are false alarms)
        assert aggregate['tp'] == 3, f"Expected tp=3, got tp={aggregate['tp']}"
        assert aggregate['fa'] == 2, f"Expected fa=2 (Sa, Su false alarms), got fa={aggregate['fa']}"
        assert aggregate['fp'] == 2, f"Expected fp=2, got fp={aggregate['fp']}"

    def test_simple_list_empty_list(self):
        """Test handling of empty simple lists."""
        gt_data = {
            "LineItems": [
                {'LineItemDays': []},  # Empty list
            ]
        }
        pred_data = gt_data
        
        gt_model = Invoice(**gt_data)
        pred_model = Invoice(**pred_data)
        
        comparison_result = gt_model.compare_with(
            pred_model,
            include_confusion_matrix=True,
            document_non_matches=False,
        )
        
        aggregate = comparison_result['confusion_matrix']['aggregate']
        
        # Empty lists should result in TN
        assert aggregate['tn'] == 1, f"Expected tn=1 (empty list), got tn={aggregate['tn']}"
        assert aggregate['tp'] == 0

    def test_simple_list_multiple_objects_complex(self):
        """Test complex scenario with multiple objects and various list sizes."""
        gt_data = {
            "LineItems": [
                {'LineItemDays': ['M', 'T', 'W', 'Th', 'F']},  # 5 elements
                {'LineItemDays': ['Sa']},                       # 1 element
                {'LineItemDays': ['Su']},                       # 1 element
            ]
        }
        pred_data = gt_data
        
        gt_model = Invoice(**gt_data)
        pred_model = Invoice(**pred_data)
        
        comparison_result = gt_model.compare_with(
            pred_model,
            include_confusion_matrix=True,
            document_non_matches=False,
        )
        
        aggregate = comparison_result['confusion_matrix']['aggregate']
        
        # Total: 5 + 1 + 1 = 7 elements
        assert aggregate['tp'] == 7, f"Expected tp=7, got tp={aggregate['tp']}"
        assert aggregate['fa'] == 0
        assert aggregate['fn'] == 0


class TestBadMatchesHandling:
    """Test suite for bad matches (similarity < threshold) handling."""

    def test_bad_matches_are_processed(self):
        """Test that bad matches (below threshold) are still processed for field metrics."""
        # Create models with different thresholds to force bad matches
        class LineItemLowThreshold(StructuredModel):
            LineItemDays: Optional[List[str]] | Any = ComparableField(weight=1.0)
            Description: Optional[str] | Any = ComparableField(weight=1.0)
            match_threshold = 0.9  # High threshold to create bad matches

        class InvoiceLowThreshold(StructuredModel):
            LineItems: Optional[List[LineItemLowThreshold]] | Any = ComparableField(weight=1.0)

        gt_data = {
            "LineItems": [
                {
                    'LineItemDays': ['M', 'T'],
                    'Description': 'Item 1'
                },
                {
                    'LineItemDays': ['W'],
                    'Description': 'Item 2'
                }
            ]
        }
        pred_data = {
            "LineItems": [
                {
                    'LineItemDays': ['M', 'T'],
                    'Description': 'Different'  # Will cause low similarity
                },
                {
                    'LineItemDays': ['W'],
                    'Description': 'Also Different'  # Will cause low similarity
                }
            ]
        }
        
        gt_model = InvoiceLowThreshold(**gt_data)
        pred_model = InvoiceLowThreshold(**pred_data)
        
        comparison_result = gt_model.compare_with(
            pred_model,
            include_confusion_matrix=True,
            document_non_matches=False,
        )
        
        # Even with bad matches, field-level metrics should be calculated
        # LineItemDays should still count elements correctly
        fields = comparison_result['confusion_matrix']['fields']
        
        # Verify that LineItems field has nested field metrics
        assert 'LineItems' in fields
        assert 'fields' in fields['LineItems']


class TestHelperMethods:
    """Test suite for new helper methods."""

    def test_merge_field_metrics(self):
        """Test _merge_field_metrics aggregates correctly."""
        from stickler.structured_object_evaluator.models.structured_list_comparator import StructuredListComparator
        
        # Create a dummy parent model
        parent = Invoice(LineItems=[])
        comparator = StructuredListComparator(parent)
        
        target = {
            'field1': {'overall': {'tp': 1, 'fa': 0, 'fd': 0, 'fp': 0, 'tn': 0, 'fn': 0}}
        }
        # Source should have the structure returned by compare_recursive
        source = {
            'overall': {},  # Not used by _merge_field_metrics
            'fields': {
                'field1': {'overall': {'tp': 2, 'fa': 1, 'fd': 0, 'fp': 1, 'tn': 0, 'fn': 0}},
                'field2': {'overall': {'tp': 3, 'fa': 0, 'fd': 0, 'fp': 0, 'tn': 0, 'fn': 0}}
            }
        }
        
        comparator._merge_field_metrics(target, source)
        
        # field1 should be aggregated
        assert target['field1']['overall']['tp'] == 3  # 1 + 2
        assert target['field1']['overall']['fa'] == 1  # 0 + 1
        
        # field2 should be added
        assert 'field2' in target
        assert target['field2']['overall']['tp'] == 3

    def test_convert_tp_to_fn(self):
        """Test _convert_tp_to_fn converts TP to FN correctly."""
        from stickler.structured_object_evaluator.models.structured_list_comparator import StructuredListComparator
        
        parent = Invoice(LineItems=[])
        comparator = StructuredListComparator(parent)
        
        # Metrics should have the structure returned by compare_recursive
        metrics = {
            'overall': {},  # Not used by _convert_tp_to_fn
            'fields': {
                'field1': {'overall': {'tp': 5, 'fa': 0, 'fd': 0, 'fp': 0, 'tn': 0, 'fn': 0}},
                'field2': {'overall': {'tp': 3, 'fa': 0, 'fd': 0, 'fp': 0, 'tn': 0, 'fn': 0}}
            }
        }
        
        converted = comparator._convert_tp_to_fn(metrics)
        
        # TP should be converted to FN
        assert converted['field1']['overall']['tp'] == 0
        assert converted['field1']['overall']['fn'] == 5
        assert converted['field2']['overall']['tp'] == 0
        assert converted['field2']['overall']['fn'] == 3

    def test_convert_tp_to_fa_fp(self):
        """Test _convert_tp_to_fa_fp converts TP to FA/FP correctly."""
        from stickler.structured_object_evaluator.models.structured_list_comparator import StructuredListComparator
        
        parent = Invoice(LineItems=[])
        comparator = StructuredListComparator(parent)
        
        # Metrics should have the structure returned by compare_recursive
        metrics = {
            'overall': {},  # Not used by _convert_tp_to_fa_fp
            'fields': {
                'field1': {'overall': {'tp': 5, 'fa': 0, 'fd': 0, 'fp': 0, 'tn': 0, 'fn': 0}},
                'field2': {'overall': {'tp': 3, 'fa': 0, 'fd': 0, 'fp': 0, 'tn': 0, 'fn': 0}}
            }
        }
        
        converted = comparator._convert_tp_to_fa_fp(metrics)
        
        # TP should be converted to FA and FP
        assert converted['field1']['overall']['tp'] == 0
        assert converted['field1']['overall']['fa'] == 5
        assert converted['field1']['overall']['fp'] == 5
        assert converted['field2']['overall']['tp'] == 0
        assert converted['field2']['overall']['fa'] == 3
        assert converted['field2']['overall']['fp'] == 3


class TestRegressionPrevention:
    """Test suite to ensure existing functionality still works."""

    def test_hierarchical_lists_still_work(self):
        """Test that List[StructuredModel] (hierarchical) still works correctly."""
        class NestedItem(StructuredModel):
            value: Optional[str] | Any = ComparableField(weight=1.0)

        class Container(StructuredModel):
            items: Optional[List[NestedItem]] | Any = ComparableField(weight=1.0)

        gt_data = {
            "items": [
                {'value': 'A'},
                {'value': 'B'}
            ]
        }
        pred_data = gt_data
        
        gt_model = Container(**gt_data)
        pred_model = Container(**pred_data)
        
        comparison_result = gt_model.compare_with(
            pred_model,
            include_confusion_matrix=True,
            document_non_matches=False,
        )
        
        aggregate = comparison_result['confusion_matrix']['aggregate']
        
        # Should count objects, not fields
        assert aggregate['tp'] == 2  # 2 objects matched

    def test_primitive_fields_still_work(self):
        """Test that primitive fields still work correctly."""
        class SimpleModel(StructuredModel):
            name: Optional[str] | Any = ComparableField(weight=1.0)
            age: Optional[int] | Any = ComparableField(weight=1.0)

        gt_data = {"name": "Alice", "age": 30}
        pred_data = gt_data
        
        gt_model = SimpleModel(**gt_data)
        pred_model = SimpleModel(**pred_data)
        
        comparison_result = gt_model.compare_with(
            pred_model,
            include_confusion_matrix=True,
            document_non_matches=False,
        )
        
        aggregate = comparison_result['confusion_matrix']['aggregate']
        
        # 2 fields matched
        assert aggregate['tp'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
