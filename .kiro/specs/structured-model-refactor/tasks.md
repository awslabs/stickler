# Implementation Plan

This implementation plan breaks down the refactoring into discrete, manageable coding tasks organized by phase. Each task builds incrementally on previous tasks and includes specific requirements references.

## Phase 1: Foundation (Low Risk - ~250 lines saved)

- [x] 1. Create ModelFactory class
  - Create `src/stickler/structured_object_evaluator/models/model_factory.py`
  - Implement `ModelFactory` class with `create_model_from_json` static method
  - Implement `validate_config` static method for configuration validation
  - Import and use existing `field_converter` module functions
  - Add comprehensive docstrings explaining the factory pattern
  - _Requirements: 1, 7, 10_

- [x] 2. Extract model_from_json to ModelFactory
  - Copy logic from `StructuredModel.model_from_json` to `ModelFactory.create_model_from_json`
  - Update `StructuredModel.model_from_json` to delegate to `ModelFactory.create_model_from_json`
  - Ensure all validation logic is preserved
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_model_from_json.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_dynamic_model_creation.py -v` (if exists)
  - Verify all tests pass before proceeding
  - _Requirements: 1, 7, 10_

- [x] 3. Create NonMatchCollector class
  - Create `src/stickler/structured_object_evaluator/models/non_match_collector.py`
  - Implement `NonMatchCollector` class with `__init__` accepting StructuredModel instance
  - Import and reuse existing `NonMatchesHelper` class
  - Add method stubs for `collect_enhanced_non_matches` and `collect_non_matches`
  - _Requirements: 1, 6, 10_

- [x] 4. Extract non-match collection methods to NonMatchCollector
  - Move `_collect_enhanced_non_matches` logic to `NonMatchCollector.collect_enhanced_non_matches`
  - Move `_collect_non_matches` logic to `NonMatchCollector.collect_non_matches`
  - Update `StructuredModel.compare_with` to use `NonMatchCollector` when `document_non_matches=True`
  - Preserve all existing logic including recursive calls
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_non_match_documentation.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_structured_model.py::test_compare_with_non_matches -v`
  - Verify all tests pass before proceeding
  - _Requirements: 1, 6, 10_

- [x] 5. Create ConfusionMatrixCalculator class
  - Create `src/stickler/structured_object_evaluator/models/confusion_matrix_calculator.py`
  - Implement `ConfusionMatrixCalculator` class with `__init__` accepting StructuredModel instance
  - Add method stubs for `calculate_list_confusion_matrix`, `classify_field_for_confusion_matrix`, `calculate_nested_field_metrics`
  - _Requirements: 1, 4, 10_

- [x] 6. Extract list confusion matrix calculation to ConfusionMatrixCalculator
  - Move `_calculate_list_confusion_matrix` logic to `ConfusionMatrixCalculator.calculate_list_confusion_matrix`
  - Move `_classify_field_for_confusion_matrix` logic to `ConfusionMatrixCalculator.classify_field_for_confusion_matrix`
  - Move `_calculate_nested_field_metrics` logic to `ConfusionMatrixCalculator.calculate_nested_field_metrics`
  - Move `_calculate_single_nested_field_metrics` logic to `ConfusionMatrixCalculator.calculate_single_nested_field_metrics`
  - Update all callers in StructuredModel to use the calculator
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_confusion_matrix_definitions.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_list_confusion_matrix.py -v` (if exists)
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_nested_field_metrics.py -v` (if exists)
  - Verify all tests pass before proceeding
  - _Requirements: 1, 4, 10_

- [ ] 7. Phase 1 validation and cleanup
  - **Full Regression Test:** Run `pytest tests/structured_object_evaluator/ -v --tb=short`
  - **Performance Test:** Run comparison benchmark to ensure no regression
  - Verify StructuredModel class is ~250 lines smaller (from 2584 to ~2334 lines)
  - Check for any unused imports or dead code
  - Update any affected docstrings
  - Verify all tests pass with no failures or warnings
  - _Requirements: 1, 8, 9, 10_

## Phase 2: Core Comparison (Moderate Risk - ~400 lines saved)

- [ ] 8. Create FieldComparator class
  - Create `src/stickler/structured_object_evaluator/models/field_comparator.py`
  - Implement `FieldComparator` class with `__init__` accepting StructuredModel instance
  - Add method stubs for `compare_primitive_with_scores` and `compare_structured_field`
  - _Requirements: 1, 2, 3, 10_

- [ ] 9. Extract primitive field comparison to FieldComparator
  - Move `_compare_primitive_with_scores` logic to `FieldComparator.compare_primitive_with_scores`
  - Extract structured field comparison logic from `_dispatch_field_comparison` to `FieldComparator.compare_structured_field`
  - Update callers to use FieldComparator methods
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_structured_model_simple.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_primitive_field_comparison.py -v` (if exists)
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_comparator_scenarios.py -v`
  - Verify all tests pass before proceeding
  - _Requirements: 1, 2, 3, 10_

- [ ] 10. Create PrimitiveListComparator class
  - Create `src/stickler/structured_object_evaluator/models/primitive_list_comparator.py`
  - Implement `PrimitiveListComparator` class following the pattern of `StructuredListComparator`
  - Add `__init__` accepting StructuredModel instance
  - Add method stub for `compare_primitive_list_with_scores`
  - _Requirements: 1, 2, 5, 10_

- [ ] 11. Extract primitive list comparison to PrimitiveListComparator
  - Move `_compare_primitive_list_with_scores` logic to `PrimitiveListComparator.compare_primitive_list_with_scores`
  - Preserve hierarchical structure and Hungarian matching logic
  - Update callers to use PrimitiveListComparator
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_list_position_invariance.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_hungarian_matching_validation.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_primitive_list_comparison.py -v` (if exists)
  - Verify all tests pass before proceeding
  - _Requirements: 1, 2, 5, 10_

- [ ] 12. Create ComparisonDispatcher class
  - Create `src/stickler/structured_object_evaluator/models/comparison_dispatcher.py`
  - Implement `ComparisonDispatcher` class with `__init__` accepting StructuredModel instance
  - Initialize FieldComparator, PrimitiveListComparator, and StructuredListComparator instances
  - Add method stubs for `dispatch_field_comparison` and `handle_list_field_dispatch`
  - _Requirements: 1, 2, 3, 10_

- [ ] 13. Extract dispatch logic to ComparisonDispatcher
  - Move `_dispatch_field_comparison` logic to `ComparisonDispatcher.dispatch_field_comparison`
  - Move `_handle_list_field_dispatch` logic to `ComparisonDispatcher.handle_list_field_dispatch`
  - Preserve match-statement based dispatch for clarity
  - Update `compare_recursive` to use ComparisonDispatcher
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_comparator_scenarios.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_dispatch_logic.py -v` (if exists)
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_type_based_dispatch.py -v` (if exists)
  - Verify all tests pass before proceeding
  - _Requirements: 1, 2, 3, 10_

- [ ] 14. Create ComparisonEngine class
  - Create `src/stickler/structured_object_evaluator/models/comparison_engine.py`
  - Implement `ComparisonEngine` class with `__init__` accepting StructuredModel instance
  - Initialize ComparisonDispatcher, NonMatchCollector, and ConfusionMatrixCalculator instances
  - Add method stubs for `compare_recursive` and `compare_with`
  - _Requirements: 1, 2, 10_

- [ ] 15. Extract comparison orchestration to ComparisonEngine
  - Move `compare_recursive` logic to `ComparisonEngine.compare_recursive`
  - Move `compare_with` logic to `ComparisonEngine.compare_with`
  - Update StructuredModel methods to delegate to ComparisonEngine
  - Preserve single-traversal optimization
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_structured_model.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_recursive_comparison.py -v` (if exists)
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_single_traversal.py -v` (if exists)
  - Verify all tests pass before proceeding
  - _Requirements: 1, 2, 10_

- [ ] 16. Phase 2 validation and cleanup
  - **Full Regression Test:** Run `pytest tests/structured_object_evaluator/ -v --tb=short`
  - **Critical Tests:** Run `pytest tests/structured_object_evaluator/test_bulk_evaluator_parity.py -v`
  - **Critical Tests:** Run `pytest tests/structured_object_evaluator/test_threshold_gated_recursion.py -v`
  - **Performance Test:** Run comparison benchmark with large nested structures
  - Verify StructuredModel class is ~650 lines smaller than original (from 2584 to ~1934 lines)
  - Verify dispatch logic is clear and traceable with match statements
  - Check for any performance regressions (should be within 5% of baseline)
  - Update docstrings to reflect delegation pattern
  - Verify all tests pass with no failures or warnings
  - _Requirements: 1, 2, 3, 8, 9, 10_

## Phase 3: Metrics & Polish (Low Risk - ~350 lines saved)

- [ ] 17. Create AggregateMetricsCalculator class
  - Create `src/stickler/structured_object_evaluator/models/aggregate_metrics_calculator.py`
  - Implement `AggregateMetricsCalculator` class
  - Add method stub for `calculate_aggregate_metrics`
  - _Requirements: 1, 4, 10_

- [ ] 18. Extract aggregate metrics calculation to AggregateMetricsCalculator
  - Move `_calculate_aggregate_metrics` logic to `AggregateMetricsCalculator.calculate_aggregate_metrics`
  - Update callers to use AggregateMetricsCalculator
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_universal_aggregate_field_comprehensive.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_aggregate_metrics.py -v` (if exists)
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_metrics_rollup.py -v` (if exists)
  - Verify all tests pass before proceeding
  - _Requirements: 1, 4, 10_

- [ ] 19. Create DerivedMetricsCalculator class
  - Create `src/stickler/structured_object_evaluator/models/derived_metrics_calculator.py`
  - Implement `DerivedMetricsCalculator` class
  - Add method stub for `add_derived_metrics_to_result`
  - Consider merging with MetricsHelper if there's significant overlap
  - _Requirements: 1, 4, 10_

- [ ] 20. Extract derived metrics calculation to DerivedMetricsCalculator
  - Move `_add_derived_metrics_to_result` logic to `DerivedMetricsCalculator.add_derived_metrics_to_result`
  - Update callers to use DerivedMetricsCalculator
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_derived_metrics.py -v` (if exists)
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_f1_precision_recall.py -v` (if exists)
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_confusion_matrix_definitions.py -v`
  - Verify all tests pass before proceeding
  - _Requirements: 1, 4, 10_

- [ ] 21. Create ConfusionMatrixBuilder orchestrator
  - Create `src/stickler/structured_object_evaluator/models/confusion_matrix_builder.py`
  - Implement `ConfusionMatrixBuilder` class that orchestrates ConfusionMatrixCalculator, AggregateMetricsCalculator, and DerivedMetricsCalculator
  - Add method `build_confusion_matrix` that coordinates all metrics calculation
  - Update `ComparisonEngine.compare_with` to use ConfusionMatrixBuilder
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_confusion_matrix_definitions.py -v`
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_universal_aggregate_field_comprehensive.py -v`
  - Verify all tests pass before proceeding
  - _Requirements: 1, 4, 10_

- [ ] 22. Clean up remaining helper methods in StructuredModel
  - Review remaining private methods in StructuredModel
  - Extract any remaining large methods (>50 lines) to appropriate helpers
  - Ensure all helper methods are well-documented
  - Remove any dead code or unused imports
  - _Requirements: 1, 8, 9_

- [ ] 23. Add comprehensive docstrings and documentation
  - Add class-level docstrings to all new classes explaining their responsibility
  - Add method-level docstrings with Args, Returns, and Raises sections
  - Update StructuredModel class docstring to explain delegation pattern
  - Add inline comments for complex logic (especially in dispatch methods)
  - _Requirements: 9_

- [ ] 24. Final validation and performance testing
  - **Full Regression Test:** Run `pytest tests/ -v --tb=short`
  - **Critical Path Tests:** Run all tests in `tests/structured_object_evaluator/` individually to identify any failures
  - **Performance Benchmark:** Run comparison benchmark comparing original vs refactored code
  - **Performance Validation:** Verify no regression (should be within 5% of baseline)
  - **Line Count Validation:** Verify StructuredModel class is under 500 lines (target: ~400 lines)
  - **Coverage Check:** Run `pytest tests/structured_object_evaluator/ --cov=src/stickler/structured_object_evaluator/models/structured_model --cov-report=term-missing`
  - Verify all 80+ test files pass without modification
  - Document any test failures and root causes
  - _Requirements: 1, 8, 10_

- [ ] 25. Update module exports and imports
  - Update `__init__.py` files to export new classes if needed
  - Verify all public APIs remain unchanged
  - Check for any circular import issues
  - Run import tests to ensure clean module structure
  - **Regression Test:** Run `pytest tests/structured_object_evaluator/test_imports.py -v` (if exists)
  - **Import Test:** Run `python -c "from stickler.structured_object_evaluator.models import StructuredModel; print('Import successful')"`
  - Verify no import errors or warnings
  - _Requirements: 1, 9_

- [ ] 26. Create migration guide and documentation
  - Document the new architecture in a REFACTORING.md file
  - Explain the delegation pattern and how to extend the system
  - Provide examples of adding new field type comparators
  - Document the class responsibilities and interactions
  - Add architecture diagram showing component relationships
  - _Requirements: 9_

## Regression Testing Strategy

### After Each Task
- Run the specific test files mentioned in the task
- Verify all tests pass with no failures or warnings
- If any test fails, debug immediately before proceeding
- Do not move to the next task until all tests pass

### After Each Phase
- Run the full test suite: `pytest tests/structured_object_evaluator/ -v --tb=short`
- Run critical path tests individually to isolate any failures
- Run performance benchmarks to ensure no regression
- Verify line count reduction matches expectations
- Document any issues or unexpected behavior

### Critical Test Files to Monitor
These tests are particularly important and should be checked after major extractions:
- `test_structured_model.py` - Core functionality
- `test_bulk_evaluator_parity.py` - Bulk evaluation correctness
- `test_confusion_matrix_definitions.py` - Confusion matrix correctness
- `test_hungarian_matching_validation.py` - Hungarian matching correctness
- `test_threshold_gated_recursion.py` - Threshold gating logic
- `test_universal_aggregate_field_comprehensive.py` - Aggregate metrics
- `test_list_position_invariance.py` - List comparison correctness

### Performance Baseline
Before starting Phase 1, establish a performance baseline:
```bash
# Create a simple benchmark script
python -m pytest tests/structured_object_evaluator/test_structured_model.py::test_large_nested_comparison --durations=10
```
After each phase, re-run the benchmark and verify performance is within 5% of baseline.

### Rollback Procedure
If tests fail and cannot be fixed quickly:
1. Identify the specific commit that introduced the failure
2. Revert that commit: `git revert <commit-hash>`
3. Analyze the root cause
4. Adjust the approach and retry
5. Document the issue in the spec for future reference

## Notes

- Each task should be completed and tested before moving to the next
- Run the specific test file mentioned after each extraction to catch issues early
- After each phase, run the full test suite to ensure no regressions
- If any test fails, debug and fix before proceeding
- Commit after each successful task for easy rollback if needed
- The total expected line reduction is ~1000 lines (from 2584 to ~1584)
- Stretch goal is to get StructuredModel under 500 lines (~400 lines target)
- **CRITICAL:** Never skip regression tests - they are your safety net
