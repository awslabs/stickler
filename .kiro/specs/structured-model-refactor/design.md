# Design Document

## Overview

This design document outlines the refactoring of the 2584-line structured_model.py file into a modular, maintainable architecture. The refactoring follows a three-phase approach, extracting comparison logic into dedicated classes while maintaining complete backward compatibility and performance characteristics.

The core principle is **delegation over implementation** - the StructuredModel class will become a thin facade that delegates to specialized helper classes, each with a single, well-defined responsibility.

## Architecture

### Current Architecture (Before Refactoring)

```
StructuredModel (2584 lines)
├── Model definition & Pydantic integration
├── Comparison dispatch logic
├── Field comparison methods
├── List comparison methods
├── Confusion matrix calculation
├── Aggregate metrics calculation
├── Non-match documentation
├── Dynamic model creation
└── Evaluator formatting
```

### Target Architecture (After Refactoring)

```
StructuredModel (~400-500 lines)
├── Public API (compare, compare_with, compare_field)
├── Pydantic integration (__init_subclass__, model_json_schema)
└── Delegation to helpers

Helper Classes:
├── ModelFactory - Dynamic model creation
├── ComparisonEngine - Core comparison orchestration
│   ├── ComparisonDispatcher - Type-based routing
│   ├── FieldComparator - Primitive field comparison
│   ├── PrimitiveListComparator - List[primitive] comparison
│   └── StructuredListComparator (existing) - List[StructuredModel] comparison
├── ConfusionMatrixBuilder - Metrics calculation
│   ├── AggregateMetricsCalculator - Aggregate rollup
│   └── DerivedMetricsCalculator - F1, precision, recall
├── NonMatchCollector - Non-match documentation
└── EvaluatorFormatter (existing) - Output formatting
```

### Delegation Pattern

All helper classes will receive the StructuredModel instance as a parameter rather than inheriting from it. This avoids circular dependencies and keeps the architecture clean:

```python
class StructuredModel(BaseModel):
    def compare_with(self, other, **options):
        # Delegate to comparison engine
        engine = ComparisonEngine(self)
        return engine.compare_with(other, **options)
```

## Components and Interfaces

### Phase 1: Foundation Components

#### 1. ModelFactory

**Responsibility:** Create dynamic StructuredModel subclasses from JSON configuration

**Location:** `src/stickler/structured_object_evaluator/models/model_factory.py`

**Interface:**
```python
class ModelFactory:
    """Factory for creating dynamic StructuredModel subclasses from JSON configuration."""
    
    @staticmethod
    def create_model_from_json(config: Dict[str, Any]) -> Type[StructuredModel]:
        """Create a StructuredModel subclass from JSON configuration.
        
        Args:
            config: JSON configuration with fields, comparators, and model settings
            
        Returns:
            A fully functional StructuredModel subclass
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate model configuration before creation.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
```

**Extraction from StructuredModel:**
- `model_from_json` class method → `create_model_from_json`
- Validation logic → `validate_config`

**Dependencies:**
- `field_converter` module (existing)
- `pydantic.create_model`

#### 2. NonMatchCollector

**Responsibility:** Collect and document non-matching fields during comparison

**Location:** `src/stickler/structured_object_evaluator/models/non_match_collector.py`

**Interface:**
```python
class NonMatchCollector:
    """Collects non-matching fields during comparison for detailed analysis."""
    
    def __init__(self, model: StructuredModel):
        """Initialize collector with the ground truth model."""
        self.model = model
        self.helper = NonMatchesHelper()  # Reuse existing helper
    
    def collect_enhanced_non_matches(
        self, 
        recursive_result: dict, 
        other: StructuredModel
    ) -> List[Dict[str, Any]]:
        """Collect enhanced non-matches with object-level granularity.
        
        Args:
            recursive_result: Result from compare_recursive
            other: The predicted StructuredModel instance
            
        Returns:
            List of non-match dictionaries with enhanced information
        """
        pass
    
    def collect_non_matches(
        self, 
        other: StructuredModel, 
        base_path: str = ""
    ) -> List[NonMatchField]:
        """Collect non-matches for detailed analysis (legacy format).
        
        Args:
            other: Other model to compare with
            base_path: Base path for field naming
            
        Returns:
            List of NonMatchField objects
        """
        pass
```

**Extraction from StructuredModel:**
- `_collect_enhanced_non_matches` → `collect_enhanced_non_matches`
- `_collect_non_matches` → `collect_non_matches`

**Dependencies:**
- `NonMatchesHelper` (existing)
- `NonMatchField`, `NonMatchType` (existing)

#### 3. ConfusionMatrixCalculator

**Responsibility:** Calculate confusion matrix metrics for field comparisons

**Location:** `src/stickler/structured_object_evaluator/models/confusion_matrix_calculator.py`

**Interface:**
```python
class ConfusionMatrixCalculator:
    """Calculates confusion matrix metrics for field comparisons."""
    
    def __init__(self, model: StructuredModel):
        """Initialize calculator with the ground truth model."""
        self.model = model
    
    def calculate_list_confusion_matrix(
        self, 
        field_name: str, 
        other_list: List[Any]
    ) -> Dict[str, Any]:
        """Calculate confusion matrix for a list field.
        
        Args:
            field_name: Name of the list field
            other_list: Predicted list to compare with
            
        Returns:
            Dictionary with TP, FP, TN, FN, FD, FA counts and nested metrics
        """
        pass
    
    def classify_field_for_confusion_matrix(
        self,
        field_name: str,
        other_value: Any,
        threshold: float = None
    ) -> Dict[str, Any]:
        """Classify a field comparison according to confusion matrix rules.
        
        Args:
            field_name: Name of the field being compared
            other_value: Value to compare with
            threshold: Threshold for matching
            
        Returns:
            Dictionary with TP, FP, TN, FN, FD counts and derived metrics
        """
        pass
    
    def calculate_nested_field_metrics(
        self,
        list_field_name: str,
        gt_list: List[StructuredModel],
        pred_list: List[StructuredModel],
        threshold: float
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for fields within list items.
        
        Args:
            list_field_name: Name of the parent list field
            gt_list: Ground truth list
            pred_list: Predicted list
            threshold: Matching threshold
            
        Returns:
            Dictionary mapping nested field paths to their metrics
        """
        pass
```

**Extraction from StructuredModel:**
- `_calculate_list_confusion_matrix` → `calculate_list_confusion_matrix`
- `_classify_field_for_confusion_matrix` → `classify_field_for_confusion_matrix`
- `_calculate_nested_field_metrics` → `calculate_nested_field_metrics`
- `_calculate_single_nested_field_metrics` → `calculate_single_nested_field_metrics`

### Phase 2: Core Comparison Components

#### 4. ComparisonDispatcher

**Responsibility:** Route field comparisons to appropriate handlers based on type

**Location:** `src/stickler/structured_object_evaluator/models/comparison_dispatcher.py`

**Interface:**
```python
class ComparisonDispatcher:
    """Dispatches field comparisons to appropriate handlers based on field type."""
    
    def __init__(self, model: StructuredModel):
        """Initialize dispatcher with the ground truth model."""
        self.model = model
        self.field_comparator = FieldComparator(model)
        self.primitive_list_comparator = PrimitiveListComparator(model)
        self.structured_list_comparator = StructuredListComparator(model)
    
    def dispatch_field_comparison(
        self, 
        field_name: str, 
        gt_val: Any, 
        pred_val: Any
    ) -> dict:
        """Dispatch field comparison using match-based routing.
        
        This is the core dispatch logic that routes to the appropriate
        comparison handler based on field type and null states.
        
        Args:
            field_name: Name of the field being compared
            gt_val: Ground truth value
            pred_val: Predicted value
            
        Returns:
            Comparison result with metrics and scores
        """
        pass
    
    def handle_list_field_dispatch(
        self, 
        gt_val: Any, 
        pred_val: Any, 
        weight: float
    ) -> Optional[dict]:
        """Handle list field comparison with early exit for null cases.
        
        Args:
            gt_val: Ground truth list value
            pred_val: Predicted list value
            weight: Field weight for scoring
            
        Returns:
            Comparison result if early exit needed, None to continue processing
        """
        pass
```

**Extraction from StructuredModel:**
- `_dispatch_field_comparison` → `dispatch_field_comparison`
- `_handle_list_field_dispatch` → `handle_list_field_dispatch`

**Key Design Decision:** Use match statements for clear, traceable dispatch logic:

```python
def dispatch_field_comparison(self, field_name: str, gt_val: Any, pred_val: Any) -> dict:
    # Get field configuration
    info = self.model._get_comparison_info(field_name)
    weight = info.weight
    threshold = info.threshold
    
    # Check field type
    is_list_field = self.model._is_list_field(field_name)
    
    # Handle list fields with match statements
    if is_list_field:
        list_result = self.handle_list_field_dispatch(gt_val, pred_val, weight)
        if list_result is not None:
            return list_result
    
    # Type-based dispatch using match statements
    match (type(gt_val), type(pred_val)):
        case (str | int | float, str | int | float):
            return self.field_comparator.compare_primitive_with_scores(
                gt_val, pred_val, field_name
            )
        case (list, list):
            if gt_val and isinstance(gt_val[0], StructuredModel):
                return self.structured_list_comparator.compare_struct_list_with_scores(
                    gt_val, pred_val, field_name
                )
            else:
                return self.primitive_list_comparator.compare_primitive_list_with_scores(
                    gt_val, pred_val, field_name
                )
        case (StructuredModel, StructuredModel):
            return self.field_comparator.compare_structured_field(
                gt_val, pred_val, field_name, threshold
            )
        case _:
            # Mismatched types
            return self._create_mismatch_result(weight)
```

#### 5. FieldComparator

**Responsibility:** Compare primitive and structured fields

**Location:** `src/stickler/structured_object_evaluator/models/field_comparator.py`

**Interface:**
```python
class FieldComparator:
    """Compares primitive and structured fields."""
    
    def __init__(self, model: StructuredModel):
        """Initialize comparator with the ground truth model."""
        self.model = model
    
    def compare_primitive_with_scores(
        self, 
        gt_val: Any, 
        pred_val: Any, 
        field_name: str
    ) -> dict:
        """Compare primitive fields and return metrics + scores.
        
        Args:
            gt_val: Ground truth value
            pred_val: Predicted value
            field_name: Name of the field
            
        Returns:
            Dictionary with metrics and similarity scores
        """
        pass
    
    def compare_structured_field(
        self,
        gt_val: StructuredModel,
        pred_val: StructuredModel,
        field_name: str,
        threshold: float
    ) -> dict:
        """Compare nested StructuredModel fields.
        
        Args:
            gt_val: Ground truth StructuredModel
            pred_val: Predicted StructuredModel
            field_name: Name of the field
            threshold: Matching threshold
            
        Returns:
            Dictionary with object-level metrics and nested field details
        """
        pass
```

**Extraction from StructuredModel:**
- `_compare_primitive_with_scores` → `compare_primitive_with_scores`
- Structured field comparison logic from `_dispatch_field_comparison` → `compare_structured_field`

#### 6. PrimitiveListComparator

**Responsibility:** Compare lists of primitive values

**Location:** `src/stickler/structured_object_evaluator/models/primitive_list_comparator.py`

**Interface:**
```python
class PrimitiveListComparator:
    """Compares lists of primitive values using Hungarian matching."""
    
    def __init__(self, model: StructuredModel):
        """Initialize comparator with the ground truth model."""
        self.model = model
    
    def compare_primitive_list_with_scores(
        self,
        gt_list: List[Any],
        pred_list: List[Any],
        field_name: str
    ) -> dict:
        """Compare primitive lists and return hierarchical metrics.
        
        Args:
            gt_list: Ground truth list
            pred_list: Predicted list
            field_name: Name of the field
            
        Returns:
            Dictionary with hierarchical structure and metrics
        """
        pass
```

**Extraction from StructuredModel:**
- `_compare_primitive_list_with_scores` → `compare_primitive_list_with_scores`

**Design Note:** This follows the same pattern as StructuredListComparator (which already exists).

#### 7. ComparisonEngine

**Responsibility:** Orchestrate the overall comparison process

**Location:** `src/stickler/structured_object_evaluator/models/comparison_engine.py`

**Interface:**
```python
class ComparisonEngine:
    """Orchestrates the comparison process for StructuredModel instances."""
    
    def __init__(self, model: StructuredModel):
        """Initialize engine with the ground truth model."""
        self.model = model
        self.dispatcher = ComparisonDispatcher(model)
        self.non_match_collector = NonMatchCollector(model)
        self.metrics_builder = ConfusionMatrixBuilder(model)
    
    def compare_recursive(self, other: StructuredModel) -> dict:
        """The core recursive comparison function.
        
        Args:
            other: Another instance to compare with
            
        Returns:
            Dictionary with hierarchical comparison results
        """
        pass
    
    def compare_with(
        self,
        other: StructuredModel,
        include_confusion_matrix: bool = False,
        document_non_matches: bool = False,
        evaluator_format: bool = False,
        recall_with_fd: bool = False,
        add_derived_metrics: bool = True
    ) -> Dict[str, Any]:
        """Compare with another instance using single traversal.
        
        Args:
            other: Another instance to compare with
            include_confusion_matrix: Whether to include confusion matrix
            document_non_matches: Whether to document non-matches
            evaluator_format: Whether to format for evaluator
            recall_with_fd: Whether to include FD in recall
            add_derived_metrics: Whether to add derived metrics
            
        Returns:
            Dictionary with comparison results
        """
        pass
```

**Extraction from StructuredModel:**
- `compare_recursive` → `compare_recursive`
- `compare_with` → `compare_with`

### Phase 3: Metrics & Polish Components

#### 8. ConfusionMatrixBuilder

**Responsibility:** Build complete confusion matrices with aggregate and derived metrics

**Location:** `src/stickler/structured_object_evaluator/models/confusion_matrix_builder.py`

**Interface:**
```python
class ConfusionMatrixBuilder:
    """Builds complete confusion matrices with aggregate and derived metrics."""
    
    def __init__(self, model: StructuredModel):
        """Initialize builder with the ground truth model."""
        self.model = model
        self.calculator = ConfusionMatrixCalculator(model)
        self.aggregate_calculator = AggregateMetricsCalculator()
        self.derived_calculator = DerivedMetricsCalculator()
    
    def build_confusion_matrix(
        self,
        recursive_result: dict,
        add_derived_metrics: bool = True
    ) -> dict:
        """Build complete confusion matrix from recursive result.
        
        Args:
            recursive_result: Result from compare_recursive
            add_derived_metrics: Whether to add derived metrics
            
        Returns:
            Complete confusion matrix with aggregate and derived metrics
        """
        pass
```

#### 9. AggregateMetricsCalculator

**Responsibility:** Calculate aggregate metrics by rolling up child metrics

**Location:** `src/stickler/structured_object_evaluator/models/aggregate_metrics_calculator.py`

**Interface:**
```python
class AggregateMetricsCalculator:
    """Calculates aggregate metrics by rolling up child field metrics."""
    
    def calculate_aggregate_metrics(self, result: dict) -> dict:
        """Calculate aggregate metrics for all nodes in the result tree.
        
        Args:
            result: Result from compare_recursive
            
        Returns:
            Modified result with 'aggregate' fields added at each level
        """
        pass
```

**Extraction from StructuredModel:**
- `_calculate_aggregate_metrics` → `calculate_aggregate_metrics`

#### 10. DerivedMetricsCalculator

**Responsibility:** Calculate derived metrics (F1, precision, recall, accuracy)

**Location:** `src/stickler/structured_object_evaluator/models/derived_metrics_calculator.py`

**Interface:**
```python
class DerivedMetricsCalculator:
    """Calculates derived metrics from basic confusion matrix counts."""
    
    def add_derived_metrics_to_result(self, result: dict) -> dict:
        """Walk through result and add 'derived' fields.
        
        Args:
            result: Result with basic TP, FP, FN, etc. metrics
            
        Returns:
            Modified result with 'derived' fields added at each level
        """
        pass
```

**Extraction from StructuredModel:**
- `_add_derived_metrics_to_result` → `add_derived_metrics_to_result`

**Design Note:** This may be merged with MetricsHelper if there's significant overlap.

## Data Models

### Comparison Result Structure

The comparison result maintains the current hierarchical structure:

```python
{
    "overall": {
        "tp": int,
        "fa": int,
        "fd": int,
        "fp": int,
        "tn": int,
        "fn": int,
        "similarity_score": float,
        "all_fields_matched": bool
    },
    "fields": {
        "field_name": {
            "overall": {...},  # Same structure as above
            "fields": {...},   # Recursive for nested objects
            "raw_similarity_score": float,
            "similarity_score": float,
            "threshold_applied_score": float,
            "weight": float
        }
    },
    "aggregate": {
        "tp": int,
        "fa": int,
        "fd": int,
        "fp": int,
        "tn": int,
        "fn": int,
        "derived": {
            "precision": float,
            "recall": float,
            "f1": float,
            "accuracy": float
        }
    },
    "non_matches": [
        {
            "field_path": str,
            "non_match_type": NonMatchType,
            "ground_truth_value": Any,
            "prediction_value": Any,
            "similarity_score": Optional[float],
            "details": dict
        }
    ]
}
```

## Error Handling

### Validation Errors

All validation errors will be raised as `ValueError` with descriptive messages:

```python
# In ModelFactory
if not isinstance(config, dict):
    raise ValueError("Configuration must be a dictionary")

# In ComparisonDispatcher
if not hasattr(self.model, field_name):
    raise ValueError(f"Field '{field_name}' not found in model")
```

### Type Mismatches

Type mismatches during comparison will be handled gracefully:

```python
# In ComparisonDispatcher
case _:
    # Mismatched types - return FD result
    return {
        "overall": {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0},
        "fields": {},
        "raw_similarity_score": 0.0,
        "similarity_score": 0.0,
        "threshold_applied_score": 0.0,
        "weight": weight
    }
```

## Testing Strategy

### Unit Testing

Each extracted class will have dedicated unit tests:

```
tests/structured_object_evaluator/models/
├── test_model_factory.py
├── test_comparison_dispatcher.py
├── test_field_comparator.py
├── test_primitive_list_comparator.py
├── test_comparison_engine.py
├── test_non_match_collector.py
├── test_confusion_matrix_calculator.py
├── test_aggregate_metrics_calculator.py
└── test_derived_metrics_calculator.py
```

### Integration Testing

The existing test suite (80+ test files) will serve as integration tests. All tests must pass after each phase:

```bash
# After Phase 1
pytest tests/structured_object_evaluator/ -v

# After Phase 2
pytest tests/structured_object_evaluator/ -v

# After Phase 3
pytest tests/structured_object_evaluator/ -v
```

### Regression Testing

Key regression tests to validate:
1. `test_structured_model.py` - Core functionality
2. `test_bulk_evaluator_parity.py` - Bulk evaluation parity
3. `test_confusion_matrix_definitions.py` - Confusion matrix correctness
4. `test_hungarian_matching_validation.py` - Hungarian matching correctness
5. `test_threshold_gated_recursion.py` - Threshold gating logic

### Performance Testing

Create a simple performance benchmark to validate no regression:

```python
# tests/structured_object_evaluator/test_performance_regression.py
import time
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

def test_comparison_performance():
    """Ensure refactored code maintains performance."""
    # Create large nested structure
    # ... setup code ...
    
    start = time.time()
    result = gt_model.compare_with(pred_model, include_confusion_matrix=True)
    elapsed = time.time() - start
    
    # Should complete in reasonable time (adjust threshold as needed)
    assert elapsed < 1.0, f"Comparison took {elapsed}s, expected < 1.0s"
```

## Migration Path

### Phase 1: Foundation (Low Risk)

**Goal:** Extract ~250 lines with minimal risk

**Steps:**
1. Create `ModelFactory` class
2. Move `model_from_json` logic to `ModelFactory.create_model_from_json`
3. Update `StructuredModel.model_from_json` to delegate to factory
4. Run tests, verify no breakage
5. Create `NonMatchCollector` class
6. Move `_collect_enhanced_non_matches` and `_collect_non_matches` to collector
7. Update `compare_with` to use collector
8. Run tests, verify no breakage
9. Create `ConfusionMatrixCalculator` class
10. Move `_calculate_list_confusion_matrix` and related methods to calculator
11. Update callers to use calculator
12. Run tests, verify no breakage

**Validation:**
- All tests pass
- No performance regression
- Code coverage maintained

### Phase 2: Core Comparison (Moderate Risk)

**Goal:** Extract ~400 lines, refactor core comparison logic

**Steps:**
1. Create `FieldComparator` class
2. Move `_compare_primitive_with_scores` to comparator
3. Run tests, verify no breakage
4. Create `PrimitiveListComparator` class
5. Move `_compare_primitive_list_with_scores` to comparator
6. Run tests, verify no breakage
7. Create `ComparisonDispatcher` class
8. Move `_dispatch_field_comparison` to dispatcher
9. Update `compare_recursive` to use dispatcher
10. Run tests, verify no breakage
11. Create `ComparisonEngine` class
12. Move `compare_recursive` and `compare_with` to engine
13. Update `StructuredModel` to delegate to engine
14. Run tests, verify no breakage

**Validation:**
- All tests pass
- No performance regression
- Dispatch logic remains clear and traceable

### Phase 3: Metrics & Polish (Low Risk)

**Goal:** Extract ~350 lines, finalize refactoring

**Steps:**
1. Create `AggregateMetricsCalculator` class
2. Move `_calculate_aggregate_metrics` to calculator
3. Run tests, verify no breakage
4. Create `DerivedMetricsCalculator` class
5. Move `_add_derived_metrics_to_result` to calculator
6. Run tests, verify no breakage
7. Create `ConfusionMatrixBuilder` class to orchestrate metrics calculation
8. Update `compare_with` to use builder
9. Run tests, verify no breakage
10. Clean up remaining helper methods
11. Add comprehensive docstrings
12. Final test run and performance validation

**Validation:**
- All tests pass
- No performance regression
- StructuredModel class is under 500 lines
- Code is more readable and maintainable

## Rollback Strategy

Each phase can be rolled back independently:

1. **Git branching:** Each phase is a separate branch
2. **Incremental commits:** Each extraction is a separate commit
3. **Test validation:** Tests must pass before merging
4. **Feature flags:** If needed, use feature flags to toggle between old and new implementations

If a phase fails validation:
1. Identify the failing test
2. Debug the issue
3. If unfixable, revert the phase
4. Analyze root cause
5. Adjust design and retry

## Performance Considerations

### Delegation Overhead

Delegation adds minimal overhead (~1-2 function calls per comparison). This is acceptable given:
1. Comparison logic is already complex (not a hot loop)
2. The benefit in maintainability outweighs the minimal performance cost
3. Modern Python JIT compilers can optimize delegation

### Object Allocation

Avoid creating new objects in hot paths:
- Pass `self` (the StructuredModel instance) to helpers rather than copying data
- Reuse existing helper instances where possible
- Use generators for large collections

### Single Traversal

Maintain the single-traversal optimization:
- `compare_recursive` still does one pass through the tree
- Confusion matrix, aggregate metrics, and derived metrics are calculated from the same result
- No redundant Hungarian matching calculations

## Future Enhancements

After the refactoring is complete, the modular architecture enables:

1. **Pluggable comparators:** Easy to add new field type comparators
2. **Custom dispatch strategies:** Easy to modify dispatch logic
3. **Alternative metrics:** Easy to add new metrics calculations
4. **Performance optimizations:** Easy to optimize individual components
5. **Testing improvements:** Easy to test components in isolation

## Conclusion

This design provides a clear path to refactor the 2584-line structured_model.py file into a modular, maintainable architecture. The three-phase approach minimizes risk while delivering significant improvements in code quality and maintainability. The extensive test suite provides confidence that the refactoring will not break existing functionality.
