# Structured Object Evaluator

The Structured Object Evaluator provides tools for evaluating structured objects with nested fields using configurable comparison metrics.

## Overview

The Structured Object Evaluator is designed to compare complex, nested data structures using a tree-based approach. It calculates an ANLS (Average Normalized Levenshtein Similarity) score between two objects, taking into account the structure and field-specific comparison metrics.

Key components:
- **StructuredModel**: Base class for defining structured data models
- **ComparableField**: Field descriptor for configuring comparison behavior
- **ANLSTree**: Tree-based representation for structured objects
- **Utility functions**: For calculating ANLS scores and handling key scores

## Installation

The Structured Object Evaluator is part of the `stickler` package. Install it using pip:

```bash
pip install stickler
```

## Usage

### Defining Structured Models

```python
from stickler import StructuredModel, ComparableField, LevenshteinComparator
from pydantic import Field

# Define a simple structured model
class Address(StructuredModel):
    street: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=1.0
    )
    city: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=1.0
    )
    state: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=1.0
    )

# Define a nested structured model
class Person(StructuredModel):
    name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=2.0  # Name is twice as important as other fields
    )
    age: int = Field()  # Regular field, not comparable
    address: Address  # Nested structured model
```

### Comparing Structured Models

```python
# Create instances
address1 = Address(street="123 Main St", city="New York", state="NY")
address2 = Address(street="123 Main Street", city="New York", state="NY")

person1 = Person(name="John Doe", age=30, address=address1)
person2 = Person(name="Jon Doe", age=30, address=address2)

# Compare using the built-in method
result = person1.compare_with(person2)
print(f"Overall score: {result['overall_score']}")
print(f"Field scores: {result['field_scores']}")
```

### Using the ANLS Score Utility

```python
from stickler.structured_object_evaluator import anls_score

# Calculate ANLS score between two objects
score = anls_score(person1, person2)
print(f"ANLS score: {score}")

# Get more detailed information
score, closest_gt, key_scores = anls_score(
    person1, person2, return_gt=True, return_key_scores=True
)
print(f"ANLS score: {score}")
print(f"Key scores: {key_scores}")
```

### Comparing JSON Objects

```python
from stickler.structured_object_evaluator import compare_json

# JSON objects to compare
json1 = {
    "name": "John Doe",
    "age": 30,
    "address": {
        "street": "123 Main St",
        "city": "New York",
        "state": "NY"
    }
}

json2 = {
    "name": "Jon Doe",
    "age": 30,
    "address": {
        "street": "123 Main Street",
        "city": "New York",
        "state": "NY"
    }
}

# Compare using the Person model
result = compare_json(json1, json2, Person)
print(f"Overall score: {result['overall_score']}")
print(f"Field scores: {result['field_scores']}")
```

### Creating Models from JSON Schema

You can create StructuredModel classes directly from JSON Schema documents (Draft 7 compatible):

```python
from stickler.structured_object_evaluator import StructuredModel

# Define a JSON Schema
invoice_schema = {
    "type": "object",
    "title": "Invoice",
    "properties": {
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
        "customer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name"]
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "unit_price": {"type": "number"}
                },
                "required": ["description", "quantity", "unit_price"]
            }
        }
    },
    "required": ["invoice_number", "total", "customer", "items"]
}

# Create the model class from the schema
Invoice = StructuredModel.from_json_schema(invoice_schema)

# Use it like any other StructuredModel
ground_truth = Invoice(
    invoice_number="INV-001",
    total=150.00,
    customer={"name": "John Doe", "email": "john@example.com"},
    items=[
        {"description": "Widget A", "quantity": 2, "unit_price": 50.00},
        {"description": "Widget B", "quantity": 1, "unit_price": 50.00}
    ]
)

prediction = Invoice(
    invoice_number="INV-001",
    total=150.00,
    customer={"name": "John Doe", "email": "john@example.com"},
    items=[
        {"description": "Widget A", "quantity": 2, "unit_price": 50.00},
        {"description": "Widget C", "quantity": 1, "unit_price": 50.00}
    ]
)

# Compare with full metrics
result = ground_truth.compare_with(
    prediction,
    include_confusion_matrix=True
)
```

#### Custom Comparison Behavior with x-aws-stickler Extensions

Use `x-aws-stickler-*` extensions in your JSON Schema to customize comparison behavior:

```python
document_schema = {
    "type": "object",
    "title": "Document",
    "properties": {
        "title": {
            "type": "string",
            "x-aws-stickler-comparator": "FuzzyComparator",  # token-based string matching
            "x-aws-stickler-threshold": 0.8,                 # require 80% similarity
        },
        "priority": {
            "type": "integer",
            "x-aws-stickler-weight": 2.0,                    # double weight for priority field
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["title", "priority"],
}

Document = StructuredModel.from_json_schema(document_schema)

gt = Document(title="Quarterly Report", priority=1, tags=["finance"])
pred = Document(title="Quarterly Report (Final)", priority=1, tags=["finance"])
result = gt.compare_with(pred)
# title 0.8, overall 0.95 -- the same pair scores title 0.667, overall 0.889 without the
# extensions, which is how you check that yours are being applied rather than ignored.
```

**Field-level extensions**, valid on a property:

- `x-aws-stickler-comparator`: comparator **class name**, such as `"ExactComparator"`, `"FuzzyComparator"`, `"LevenshteinComparator"`, `"SemanticComparator"`. Lowercase aliases are not accepted and raise.
- `x-aws-stickler-comparator-config`: keyword arguments for that comparator's constructor
- `x-aws-stickler-threshold`: matching threshold, 0.0 to 1.0 (default: 0.5)
- `x-aws-stickler-weight`: field importance weight (default: 1.0)
- `x-aws-stickler-clip-under-threshold`: clip scores below threshold to 0.0 (boolean, default: true)

**Object-level extensions**, valid on the root schema or on any object-typed property:

- `x-aws-stickler-model-name`: the generated class name (default: `DynamicModel`)
- `x-aws-stickler-match-threshold`: the score at which two objects count as the same object (default: 0.7)

The two sets are not interchangeable, and position matters. A field-level key on
the root, or an object-level key on a scalar field, is not read, so it is
rejected rather than dropped. An unrecognized or misspelled `x-aws-stickler-*`
key raises and names the closest valid key for that position; unrelated `x-*`
extensions from other tooling are left alone.

Comparator names accepted by `x-aws-stickler-comparator` are the registered class names:
`LevenshteinComparator`, `ExactComparator`, `NormalizedComparator`, `PhoneComparator`,
`NumericComparator`, `DateComparator`, `FuzzyComparator`, `StructuredModelComparator`,
`ANLSStarComparator`, `BBoxIoUComparator`, `SemanticComparator`, plus `BERTComparator` and
`LLMComparator`, which need the `[bert]` and `[llm]` extras. Without the extra those two are not
registered, so naming one raises `ValueError: Invalid x-aws-stickler-comparator 'BERTComparator'
...` and the name is absent from the "Available" list the message prints.

With no comparator given, the choice is made in two steps, in `_default_comparison` and
`_evaluation_annotation` (`models/json_schema_importer.py`):

1. The schema library parses the property to a strict Python annotation — `format: date` becomes
   `date`, an `enum` becomes an `Enum` subclass, a single-value `const` becomes a `Literal`.
2. `_default_comparison` selects the comparator from *that* annotation, and the annotation is then
   widened back to the JSON value type so an invalid extraction is an ordinary zero-score candidate
   rather than a construction error.

So `format`, `enum` and `const` do participate — while field names never do — but the widening in
step 2 means the built field is a plain `str`. `model_fields` therefore shows `Optional[str]` for
every row below; `to_json_schema()["properties"]` is where the chosen comparator is legible.

| Schema | Parsed as (step 1) | Comparator | Threshold |
| --- | --- | --- | --- |
| `"string"` | `str` | `LevenshteinComparator` | `0.5` |
| `"number"` | `float` | `NumericComparator` | `0.5` |
| `"integer"` | `int` | `NumericComparator` | `0.5` |
| `"boolean"` | `bool` | `ExactComparator` | `0.5` |
| `"string"` + `"format": "date"` or `"date-time"` | `date` / `datetime` | `DateComparator` | `1.0` |
| `"string"` + `"enum"` or a single-value `const` | `Enum` / `Literal` | `ExactComparator` | `1.0` |

Any annotation not in that table falls back to `ExactComparator` at `1.0` — which is where a
`format` the schema library maps to a distinct type (`"uri"` → `AnyUrl`, `"uuid"` → `UUID`,
`"time"` → `time`) lands. A `format` it does not model (`"email"`, `"hostname"`, `"duration"`)
parses as `str`, so the field keeps `LevenshteinComparator` at `0.5`. `Decimal` is the one
non-primitive with its own entry, at `NumericComparator` `0.5`.

**Supported JSON Schema Features:**

- Primitive types: `string`, `number`, `integer`, `boolean`. A bare `{"type": "null"}` is
  *accepted*, and builds a `string` field at `ExactComparator` `1.0` — almost certainly not what
  the schema meant. Declare the nullable form `{"type": ["string", "null"]}` instead.
- Complex types: `object`, `array`
- Nested objects and arrays of objects
- Required fields via `required` array
- Optional fields (not in `required` array)
- JSON Schema Draft 7 compatibility

See `examples/scripts/json_schema_demo.py` for complete examples, and the
[extension reference](../../../README.md#json-schema-extensions-x-aws-stickler--complete-reference)
in the top-level README for per-extension detail.

## Field Comparison Configuration

The `ComparableField` descriptor allows you to configure how fields are compared:

```python
ComparableField(
    comparator=LevenshteinComparator(),  # comparison algorithm (default: LevenshteinComparator)
    threshold=0.7,                       # similarity threshold, 0.0-1.0 (default: 0.5)
    weight=1.0,                          # field weight for overall score (default: 1.0)
    clip_under_threshold=True,           # zero out scores below threshold (default: True)
    default=None,                        # field default; setting one does not make the field
                                         # required — is_required() is False either way, so a
                                         # missing prediction scores rather than failing validation
)
```

`ComparableField` passes any other keyword through to Pydantic's `Field`, so a name it does not
recognize is accepted without error and has no effect on comparison.

Available comparators:

- `LevenshteinComparator`: String similarity based on edit distance
- `ExactComparator`: Exact match comparison
- `FuzzyComparator`: Token-based fuzzy matching, order-independent
- `NumericComparator`: Numeric comparison with tolerance
- `DateComparator`: Date comparison across formats
- `PhoneComparator`: Phone-number comparison, normalizing formatting
- `SemanticComparator`: Semantic similarity using embeddings
- `StructuredModelComparator`: Recursive comparison of a nested model
- `BBoxIoUComparator`: Bounding-box overlap by intersection over union
- `BERTComparator`, `LLMComparator`: Contextual and LLM-judged similarity; need the `[bert]` and
  `[llm]` extras

## API Reference

### StructuredModel

```python
class StructuredModel(BaseModel):
    """Base class for structured data models with comparison capabilities."""
    
    def compare_with(self, other: 'StructuredModel') -> Dict[str, Any]:
        """
        Compare this model with another model.
        
        Args:
            other: Another StructuredModel to compare with
            
        Returns:
            Dictionary with comparison results
        """
    
    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]) -> 'StructuredModel':
        """
        Create a model instance from a JSON object.
        
        Args:
            json_obj: JSON object to convert
            
        Returns:
            StructuredModel instance
        """
    
    @classmethod
    def from_json_schema(cls, schema: Dict[str, Any]) -> Type['StructuredModel']:
        """
        Create a StructuredModel subclass from a JSON Schema document.
        
        Args:
            schema: JSON Schema document (Draft 7 compatible)
            
        Returns:
            New StructuredModel subclass
            
        Example:
            >>> schema = {
            ...     "type": "object",
            ...     "title": "Product",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "price": {"type": "number"}
            ...     },
            ...     "required": ["name", "price"]
            ... }
            >>> Product = StructuredModel.from_json_schema(schema)
            >>> product = Product(name="Widget", price=9.99)
        """
```

### Utility Functions

```python
def anls_score(
    gt: Any,
    pred: Any,
    return_gt: bool = False,
    return_key_scores: bool = False
) -> Union[float, Tuple[float, Any], Tuple[float, Any, Dict[str, Any]]]:
    """
    Calculate ANLS score between two objects.
    
    Args:
        gt: Ground truth object
        pred: Prediction object
        return_gt: Whether to return the closest ground truth
        return_key_scores: Whether to return detailed key scores
        
    Returns:
        Either just the overall score (float), or a tuple with the score and
        closest ground truth, or a tuple with the score, closest ground truth,
        and key scores.
    """

def compare_json(
    gt_json: Dict[str, Any],
    pred_json: Dict[str, Any],
    model_cls: Type[StructuredModel]
) -> Dict[str, Any]:
    """
    Compare JSON objects using a StructuredModel.
    
    Args:
        gt_json: Ground truth JSON
        pred_json: Prediction JSON
        model_cls: StructuredModel class to use for comparison
        
    Returns:
        Dictionary with comparison results
    """
```

## Performance Optimizations

### Hungarian Matching Optimization (2025)

The Structured Object Evaluator has been optimized to eliminate redundant Hungarian matching operations for list comparisons. Previously, the system performed 4-5 separate Hungarian matching calculations per list field, leading to O(n²) complexity for list-heavy comparisons.

**Optimization Details:**
- **Problem**: Multiple calls to `get_matched_pairs_with_scores()`, `get_assignments()`, and `get_unmatched_indices()` for the same list pairs
- **Solution**: Consolidated to single `get_complete_matching_info()` call that returns all required information
- **Performance Impact**: 60-75% improvement for comparisons with multiple list fields
- **Backward Compatibility**: All existing APIs maintained, optimization is transparent to users

**Files Modified:**
- `hungarian_helper.py`: Added unified `get_complete_matching_info()` method
- `structured_model.py`: Updated to use single Hungarian matching call
- `structured_list_comparator.py`: Optimized list comparison logic
- `comparison_helper.py`: Updated helper methods
- `evaluator_format_helper.py`: Updated evaluator formatting

**Testing**: All 374 existing tests pass, ensuring no regression in functionality.

### Post-Processing Architecture

The comparison pipeline uses a constructive building approach in `compare_recursive()` followed by lightweight post-processing transformations:

1. **Primary Traversal**: `compare_recursive()` builds the comparison tree structure efficiently
2. **Aggregate Metrics**: `AggregateMetricsCalculator.calculate_aggregate_metrics()` adds Universal Aggregate Field data
3. **Derived Metrics**: `_add_derived_metrics_to_result()` adds precision, recall, F1 scores

This architecture prioritizes correctness of the core comparison logic while keeping post-processing transformations simple and maintainable. The post-processing steps operate on already-computed confusion matrix data and represent a small fraction of total computation time.

## Examples

See the [examples](../../../examples/key_information_evaluation/structured_object_evaluator) directory for complete examples.

## Known Limitations

### Dictionary Fields
Dictionary fields (`Dict[str, Any]`) with text-based comparators like `LevenshteinComparator` are not supported as they would produce unpredictable results.

**Solution**: Create proper StructuredModel subclasses instead of using raw dictionaries.

**Example**:
```python
# ❌ Not supported - will raise an error
attributes: Dict[str, Any] = ComparableField(
    comparator=LevenshteinComparator(),
    threshold=0.7
)

# ✅ Correct approach - define a structured model
class CustomerAttributes(StructuredModel):
    loyalty_level: str = ComparableField(...)
    join_date: str = ComparableField(...)

class Customer(StructuredModel):
    attributes: CustomerAttributes = ComparableField(...)
```

This limitation ensures reliable, predictable comparison results.
