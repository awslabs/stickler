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
from stickler.structured_object_evaluator import StructuredModel, ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
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

#### Custom Comparison Behavior with x-stickler Extensions

Use `x-stickler-*` extensions in your JSON Schema to customize comparison behavior:

```python
document_schema = {
    "type": "object",
    "title": "Document",
    "properties": {
        "title": {
            "type": "string",
            "x-stickler-comparator": "fuzzy",  # Use fuzzy string matching
            "x-stickler-threshold": 0.8  # Require 80% similarity
        },
        "priority": {
            "type": "integer",
            "x-stickler-weight": 2.0  # Double weight for priority field
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["title", "priority"]
}

Document = StructuredModel.from_json_schema(document_schema)
```

**Available x-stickler Extensions:**
- `x-stickler-comparator`: Comparison algorithm (`"exact"`, `"fuzzy"`, `"levenshtein"`, `"semantic"`)
- `x-stickler-threshold`: Matching threshold (0.0 to 1.0, default: 0.5)
- `x-stickler-weight`: Field importance weight (default: 1.0)
- `x-stickler-clip-under-threshold`: Clip scores below threshold to 0.0 (boolean, default: false)

**Supported JSON Schema Features:**
- All primitive types: `string`, `number`, `integer`, `boolean`, `null`
- Complex types: `object`, `array`
- Nested objects and arrays of objects
- Required fields via `required` array
- Optional fields (not in `required` array)
- JSON Schema Draft 7 compatibility

See `examples/scripts/json_schema_demo.py` for complete examples.

## Field Comparison Configuration

The `ComparableField` descriptor allows you to configure how fields are compared:

```python
ComparableField(
    comparator=LevenshteinComparator(),  # Comparison algorithm
    threshold=0.7,  # Similarity threshold (0.0-1.0)
    weight=1.0,  # Field weight for overall score
    required=True  # Whether the field is required
)
```

Available comparators:
- `LevenshteinComparator`: String similarity based on edit distance
- `ExactComparator`: Exact match comparison
- `NumericComparator`: Numeric comparison with tolerance
- `SemanticComparator`: Semantic similarity using embeddings

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
2. **Aggregate Metrics**: `_calculate_aggregate_metrics()` adds Universal Aggregate Field data
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
