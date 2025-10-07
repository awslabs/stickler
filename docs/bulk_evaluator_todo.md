# Bulk Evaluator TODO

## Issue Summary

The StructuredObjectBulkEvaluator has inconsistencies in how it handles nested fields in structured models. Tests expect nested fields like "transactions.date" to be directly accessible in result.field_metrics, but the current implementation doesn't properly expose these nested fields.

## Debug Findings

When running tests like `test_evaluate_perfect_matches` from `test_bulk_evaluator.py`, we observed that:

1. The test expects nested fields like "transactions.date" to be directly accessible:
   ```python
   assert "transactions.date" in result.field_metrics
   ```

2. However, the bulk evaluator only aggregates top-level fields:
   ```
   DEBUG - Current field_metrics structure:
   {'accountNumber': {...}, 'period': {...}, 'transactions': {...}}
   ```

3. The nested field metrics are not being properly extracted from the single evaluator results.

## Potential Fix

We started implementing a fix in the `_aggregate_field_metrics` method to handle nested fields:

```python
# Handle nested_fields structure from the new implementation
if isinstance(field_data, dict) and "nested_fields" in field_data:
    for nested_field_name, nested_metrics in field_data["nested_fields"].items():
        nested_path = f"{current_path}.{nested_field_name}" if not nested_field_name.startswith(current_path) else nested_field_name
        
        if nested_path not in aggregated_cm:
            aggregated_cm[nested_path] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "fd": 0, "fa": 0}
            
        for metric in ["tp", "fp", "tn", "fn", "fd", "fa"]:
            if metric in nested_metrics:
                aggregated_cm[nested_path][metric] += nested_metrics[metric]
```

This approach would ensure that nested fields from the single evaluator results are properly exposed in the bulk evaluator results.

## Affected Tests

Priority 2 tests that need this fix:

1. `test_evaluate_perfect_matches` - Expecting nested fields like "transactions.date" to be directly accessible
2. `test_confusion_matrix_structure_matches_single_evaluator` - Same issue with nested field access

## Next Steps

1. Re-implement and test the fix for the `_aggregate_field_metrics` method
2. Ensure compatibility with existing code and that all tests pass
3. Consider adding explicit tests for the bulk evaluator's handling of nested fields
