# StructuredModel Refactoring Guide

## Overview

This document describes the architectural refactoring of the `StructuredModel` class from a monolithic ~2584-line implementation to a modular architecture using the delegation pattern. `StructuredModel` is now ~1486 lines (down from 2584, a ~42% reduction), and most of the comparison, metrics, and dynamic-model logic now lives in dedicated modules under `src/stickler/structured_object_evaluator/models/`. The public `compare`, `compare_with`, `from_json`, `from_json_schema`, `model_from_json`, `to_json_schema`, and `to_stickler_config` APIs are preserved.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Delegation Pattern](#delegation-pattern)
- [Component Responsibilities](#component-responsibilities)
- [Extending the System](#extending-the-system)
- [Migration Guide](#migration-guide)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

### Before Refactoring

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

### After Refactoring

```
StructuredModel (~1486 lines)
├── Public API (compare, compare_with, compare_field_raw, from_json,
│              model_from_json, from_json_schema, to_json_schema,
│              to_stickler_config, model_json_schema)
├── Pydantic integration (__init_subclass__, model_post_init)
├── Confidence/extras accessors (get_field_confidence, get_all_confidences,
│                                get_field_extras, get_all_extras)
├── Thin delegating shims for legacy private methods
└── Delegation to specialized components

Specialized Components (selected — ~36 modules total in models/):
├── Dynamic-model creation
│   ├── ModelFactory                  - create_model_from_json / create_model_from_fields
│   ├── JsonSchemaFieldConverter      - JSON Schema → field definitions
│   └── FieldConverter                - Stickler config → field definitions
├── Comparison
│   ├── ComparisonEngine              - Orchestrates compare_recursive / compare_with
│   ├── ComparisonDispatcher          - Routes comparisons by type
│   ├── FieldComparator               - Primitive & nested-StructuredModel fields
│   ├── PrimitiveListComparator       - Lists of primitives
│   └── StructuredListComparator      - Lists of StructuredModels (Hungarian)
├── Metrics
│   ├── ConfusionMatrixBuilder        - Orchestrates the three calculators
│   ├── ConfusionMatrixCalculator     - TP/FP/TN/FN/FD/FA
│   ├── AggregateMetricsCalculator    - Rolls up child metrics
│   └── DerivedMetricsCalculator      - Precision / recall / F1 / accuracy
├── Reporting
│   ├── NonMatchCollector             - Object- and field-level non-matches
│   ├── FieldComparisonCollector      - document_field_comparisons output
│   └── EvaluatorFormatHelper         - evaluator_format output
├── Confidence (v0.4.0+)
│   └── models/confidence/            - AUROC and other calibration metrics
│       used when compare_with(add_confidence_metrics=True, ...)
└── Pre-existing helpers
    ├── HungarianHelper, ConfigurationHelper, ComparisonHelper
    ├── ThresholdHelper, RichValueHelper, NonMatchesHelper, FieldHelper
    ├── ResultHelper, NullHelper, TypeResolver, ComparatorRegistry
    └── PostComparisonAccumulator, ComparisonInfo, ComparisonHelperBase
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        StructuredModel                          │
│                       (Public API ~1486 lines)                  │
│  • compare()  • compare_with()  • compare_field_raw()           │
│  • from_json() / model_from_json() / from_json_schema()         │
│  • to_json_schema() / to_stickler_config()                      │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├────► ModelFactory / JsonSchemaFieldConverter
             │      (Dynamic model creation: model_from_json,
             │       from_json_schema)
             │
             └────► ComparisonEngine
                    (Orchestrates compare_recursive / compare_with;
                     also drives confidence-metrics path)
                            │
            ┌───────────────┼───────────────────────────┐
            │               │               │           │
            ▼               ▼               ▼           ▼
   ComparisonDispatcher  NonMatchCollector  FieldComparison  ConfusionMatrixBuilder
   (Type-based routing)  (Non-matches)      Collector        (Metrics orchestration)
            │                              (per-field log)        │
  ┌─────────┼─────────┐                                ┌──────────┼──────────┐
  ▼         ▼         ▼                                ▼          ▼          ▼
FieldCmp  PrimList  StructList                       Confusion  Aggregate  Derived
(prim &   Comparator Comparator                       Matrix    Metrics    Metrics
 nested)                                              Calc.     Calc.      Calc.
```

## Delegation Pattern

The refactored architecture uses **delegation over inheritance** to separate concerns.

### Key Principles

1. **Single Responsibility**: Each component has one clear purpose
2. **Composition**: Components are composed, not inherited
3. **Dependency Injection**: Components receive the `StructuredModel` instance
4. **Immutability**: Components don't modify the model
5. **Testability**: Each component can be tested in isolation
6. **Lazy imports**: To avoid circular dependencies, `StructuredModel` imports its delegates inside methods (e.g. `from .comparison_engine import ComparisonEngine`). You will see this pattern throughout the file — it is intentional, not legacy cruft.

### Example: Comparison Delegation

```python
class StructuredModel(BaseModel):
    """Main model class - delegates to specialized components."""

    def compare_with(
        self,
        other: "StructuredModel",
        include_confusion_matrix: bool = False,
        document_non_matches: bool = False,
        evaluator_format: bool = False,
        recall_with_fd: bool = False,
        add_derived_metrics: bool = True,
        document_field_comparisons: bool = False,
        add_confidence_metrics: bool = False,
        confidence_metrics: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Compare with another instance - delegates to ComparisonEngine."""
        from .comparison_engine import ComparisonEngine

        engine = ComparisonEngine(self)
        return engine.compare_with(
            other,
            include_confusion_matrix=include_confusion_matrix,
            document_non_matches=document_non_matches,
            evaluator_format=evaluator_format,
            recall_with_fd=recall_with_fd,
            add_derived_metrics=add_derived_metrics,
            document_field_comparisons=document_field_comparisons,
            add_confidence_metrics=add_confidence_metrics,
            confidence_metrics=confidence_metrics,
        )
```

### Legacy private methods are kept as delegating shims

Older callers (and many tests) reach for private methods on `StructuredModel` such as `_dispatch_field_comparison`, `compare_recursive`, `_calculate_list_confusion_matrix`, `_calculate_nested_field_metrics`, `_collect_enhanced_non_matches`, `_add_derived_metrics_to_result`, `_format_for_evaluator`, etc. These were **not removed** — they now exist as one-line wrappers that build the appropriate component and forward the call. They are retained for backward compatibility; new code should call the component directly.

## Component Responsibilities

### 1. StructuredModel (Core Interface)

**Location**: `src/stickler/structured_object_evaluator/models/structured_model.py`

**Responsibility**: Provide the public API and Pydantic integration

**Key Methods**:
- `compare(other)` - Returns a scalar weighted similarity score `[0.0, 1.0]`
- `compare_with(other, **options)` - Full comparison with metrics; supports `include_confusion_matrix`, `document_non_matches`, `evaluator_format`, `recall_with_fd`, `add_derived_metrics`, `document_field_comparisons`, `add_confidence_metrics`, `confidence_metrics`
- `compare_field_raw(field_name, other_value)` - Per-field raw similarity (no threshold applied)
- `from_json(json_data, process_rich_values=None, process_confidence=None)` - Construct an instance from JSON, optionally unwrapping rich values
- `model_from_json(config)` - Create a dynamic subclass from Stickler config (delegates to `ModelFactory`)
- `from_json_schema(schema)` - Create a dynamic subclass from a JSON Schema document with `x-aws-stickler-*` extensions
- `to_json_schema()` / `to_stickler_config()` - Round-trip exporters back to either format

**Size**: ~1486 lines (down from 2584)

### 2. ModelFactory and JSON-Schema path

**Location**:
- `src/stickler/structured_object_evaluator/models/model_factory.py`
- `src/stickler/structured_object_evaluator/models/json_schema_field_converter.py`
- `src/stickler/structured_object_evaluator/models/field_converter.py`

**Responsibility**: Create `StructuredModel` subclasses from configuration.

Two entry points are wired into `StructuredModel`:

```python
# 1. Stickler-native config
config = {
    "model_name": "Person",
    "fields": {
        "name": {"type": "str", "comparator": "LevenshteinComparator"},
        "age":  {"type": "int", "comparator": "NumericComparator"},
    },
}
PersonModel = StructuredModel.model_from_json(config)

# 2. Standard JSON Schema (draft-07+) with x-aws-stickler-* extensions
schema = {
    "type": "object",
    "x-aws-stickler-model-name": "Person",
    "properties": {
        "name": {"type": "string", "x-aws-stickler-comparator": "LevenshteinComparator"},
        "age":  {"type": "integer", "x-aws-stickler-comparator": "NumericComparator"},
    },
    "required": ["name"],
}
PersonModel = StructuredModel.from_json_schema(schema)
```

`from_json_schema` validates the schema, runs it through `JsonSchemaFieldConverter`, and then calls `ModelFactory.create_model_from_fields(...)`. The reverse operations `to_json_schema()` and `to_stickler_config()` round-trip the model back to either representation.

### 3. ComparisonEngine

**Location**: `src/stickler/structured_object_evaluator/models/comparison_engine.py`

**Responsibility**: Orchestrate the comparison process.

**Key Methods**:
- `compare_recursive(other)` - Single-pass recursive comparison (scores + confusion-matrix counts in one traversal)
- `compare_with(other, **options)` - Full comparison; layers on confusion matrix, non-matches, field-comparison logging, evaluator formatting, and confidence calibration metrics depending on flags

The engine lazily instantiates `ComparisonDispatcher`, `NonMatchCollector`, `FieldComparisonCollector`, and `ConfusionMatrixBuilder` on first access via properties.

### 4. ComparisonDispatcher

**Location**: `src/stickler/structured_object_evaluator/models/comparison_dispatcher.py`

**Responsibility**: Route field comparisons by type.

**Dispatch Logic**:
```python
match (type(gt_val), type(pred_val)):
    case (str | int | float, str | int | float):
        return self.field_comparator.compare_primitive_with_scores(...)
    case (list, list):
        # Route to PrimitiveListComparator or StructuredListComparator
    case (StructuredModel, StructuredModel):
        return self.field_comparator.compare_structured_field(...)
```

### 5-12. Other Components

See the docstring on `StructuredModel` (`structured_model.py`) for the in-code component map. The most relevant ones are:

- `FieldComparator` - primitive & nested-StructuredModel field comparison
- `PrimitiveListComparator` - lists of primitives via Hungarian matching
- `StructuredListComparator` - lists of StructuredModels via Hungarian + threshold-gated nested analysis
- `ConfusionMatrixBuilder` / `ConfusionMatrixCalculator` / `AggregateMetricsCalculator` / `DerivedMetricsCalculator`
- `NonMatchCollector` and `FieldComparisonCollector`
- `EvaluatorFormatHelper`
- `models/confidence/` - the confidence-calibration pipeline used when `compare_with(add_confidence_metrics=True, ...)`

## Extending the System

### Adding a New Field Type Comparator

Example: a basic datetime comparator.

#### Step 1: Create the Comparator

`BaseComparator` lives at `src/stickler/comparators/base.py` and exposes a 2-argument `compare(self, str1, str2)` signature. Configuration parameters belong on `__init__`, not on `compare(...)`:

```python
from datetime import datetime

from stickler.comparators.base import BaseComparator


class DateTimeComparator(BaseComparator):
    def __init__(self, max_diff_seconds: float = 86400.0, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.max_diff_seconds = max_diff_seconds

    def compare(self, gt: datetime, pred: datetime) -> float:
        if gt == pred:
            return 1.0

        diff = abs((gt - pred).total_seconds())
        return max(0.0, 1.0 - (diff / self.max_diff_seconds))
```

#### Step 2: Use in a Model

Comparators are configured directly on fields with `ComparableField()`. Each field carries its own configured instance.

```python
from datetime import datetime

from stickler import ComparableField, StructuredModel


class Event(StructuredModel):
    name: str = ComparableField()
    timestamp: datetime = ComparableField(
        comparator=DateTimeComparator(max_diff_seconds=3600),
        threshold=0.8,
        weight=1.0,
    )


event1 = Event(name="Meeting", timestamp=datetime(2024, 1, 1, 10, 0, 0))
event2 = Event(name="Meeting", timestamp=datetime(2024, 1, 1, 10, 30, 0))
result = event1.compare_with(event2)
print(result["field_scores"]["timestamp"])  # ~0.5 with the values above
```

#### Step 3: Update the Dispatcher (if needed)

If your new type is not already covered by the existing match arms in `ComparisonDispatcher.dispatch_field_comparison`, add a case so the dispatcher routes to your comparator. Most numeric/string/structured cases are already covered — datetimes typically fall through to the primitive path because `==` and the configured `compare()` already do the right thing.

### Adding Custom Metrics

```python
class CustomMetricsCalculator:
    def calculate_custom_metrics(self, result: dict) -> dict:
        if "aggregate" in result:
            tp = result["aggregate"].get("tp", 0)
            fp = result["aggregate"].get("fp", 0)
            weighted_score = (tp - 2 * fp) / (tp + fp) if (tp + fp) > 0 else 0
            result["custom_metrics"] = {"weighted_score": weighted_score}
        return result
```

## Migration Guide

### For Library Users

**No changes required.** The refactor preserves the public API.

```python
# All existing code works unchanged
from stickler import StructuredModel, ComparableField

class Person(StructuredModel):
    name: str
    age: int

person1 = Person(name="Alice", age=30)
person2 = Person(name="Alice", age=31)
result = person1.compare_with(person2)
```

The supported import paths are:

```python
# Preferred — re-exported from the package root
from stickler import StructuredModel, ComparableField

# Also works — re-exported from the subpackage
from stickler.structured_object_evaluator import StructuredModel, ComparableField

# Direct module imports also work, e.g.
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
```

> Note: `from stickler.structured_object_evaluator.models import StructuredModel` is **not** supported — `models/__init__.py` does not re-export. Use one of the paths above.

### For Contributors

#### Code Location Mapping

These methods still exist on `StructuredModel` as thin delegating shims for backward compatibility. New code should call the component directly:

| Method on `StructuredModel`           | Component the call delegates to                                  |
|---------------------------------------|------------------------------------------------------------------|
| `model_from_json`                     | `ModelFactory.create_model_from_json`                            |
| `from_json_schema`                    | `JsonSchemaFieldConverter` + `ModelFactory.create_model_from_fields` |
| `_dispatch_field_comparison`          | `ComparisonDispatcher.dispatch_field_comparison`                 |
| `_handle_list_field_dispatch`         | `ComparisonDispatcher.handle_list_field_dispatch`                |
| `compare_recursive`                   | `ComparisonEngine.compare_recursive`                             |
| `compare_with`                        | `ComparisonEngine.compare_with`                                  |
| `_classify_field_for_confusion_matrix`| `ConfusionMatrixCalculator.classify_field_for_confusion_matrix`  |
| `_calculate_list_confusion_matrix`    | `ConfusionMatrixCalculator.calculate_list_confusion_matrix`      |
| `_calculate_nested_field_metrics`     | `ConfusionMatrixCalculator.calculate_nested_field_metrics`       |
| `_calculate_single_nested_field_metrics` | `ConfusionMatrixCalculator.calculate_single_nested_field_metrics` |
| `_add_derived_metrics_to_result`      | `DerivedMetricsCalculator.add_derived_metrics_to_result`         |
| `_collect_enhanced_non_matches`       | `NonMatchCollector.collect_enhanced_non_matches`                 |
| `_format_for_evaluator`               | `EvaluatorFormatHelper.format_for_evaluator`                     |

## Performance Considerations

### Delegation Overhead

Delegation costs ~1-2 extra Python function calls per dispatch. CPython does not have a built-in JIT, but call dispatch is fast enough that this is dominated by the actual comparison work in every benchmark we have run. The maintainability benefit is the reason for the change; the overhead is not the bottleneck.

### Single Traversal Optimization

Maintained — one pass through the tree:

```python
result = engine.compare_recursive(other)        # Single traversal
if include_confusion_matrix:
    result = builder.build_confusion_matrix(result)  # No re-traversal
```

### Performance Validation

There is a benchmark at `tests/structured_object_evaluator/test_performance_benchmark.py`. Re-run it locally for current numbers — the timings depend strongly on hardware and Python version, so we do not pin specific numbers in this doc:

```bash
uv run pytest tests/structured_object_evaluator/test_performance_benchmark.py -v
```

## Troubleshooting

### Import Errors

```python
# ✅ Correct — public API
from stickler import StructuredModel, ComparableField

# ✅ Also correct — subpackage re-export
from stickler.structured_object_evaluator import StructuredModel, ComparableField

# ❌ Incorrect — models/__init__.py does not re-export
from stickler.structured_object_evaluator.models import StructuredModel

# ❌ Reaching into internal modules — works, but couples to internals
from stickler.structured_object_evaluator.models.comparison_engine import ComparisonEngine
```

### Test Failures

If tests fail after upgrade:

1. Check you're not testing internal implementation
2. Use the public API only
3. Avoid monkey-patching private methods

```python
# ❌ Bad: Testing internals
result = model._dispatch_field_comparison(...)

# ✅ Good: Testing public API
result = model.compare_with(other)
```

### Debugging

Step through the pipeline by exercising components directly:

```python
from stickler.structured_object_evaluator.models.comparison_engine import ComparisonEngine
from stickler.structured_object_evaluator.models.comparison_dispatcher import ComparisonDispatcher

engine = ComparisonEngine(model1)
recursive_result = engine.compare_recursive(model2)
print("Recursive result:", recursive_result)

dispatcher = ComparisonDispatcher(model1)
field_result = dispatcher.dispatch_field_comparison("field", val1, val2)
print("Field result:", field_result)
```

## Summary

The refactoring achieved:

- **Significant size reduction** in `StructuredModel`: 2584 → ~1486 lines (~42%), with the rest now spread across dedicated components under `models/`.
- **Delegation-over-inheritance architecture** with single-responsibility components for dynamic-model creation, comparison routing, list comparison, confusion-matrix metrics, derived metrics, non-match reporting, evaluator formatting, and confidence calibration.
- **100% public-API backward compatibility** — `compare`, `compare_with`, `from_json`, `model_from_json`, `from_json_schema`, `to_json_schema`, `to_stickler_config`, and the legacy private methods (as shims) all still work.
- **Single-traversal optimization** preserved.
- **Confidence-calibration pipeline** added (v0.4.0+) under `models/confidence/`, plumbed through `compare_with(add_confidence_metrics=True, ...)`.

The delegation pattern enables easy extension while keeping the public API clean.
