---
title: StructuredModel from JSON
---


# StructuredModel Dynamic Creation from JSON

This document describes how to create StructuredModel classes dynamically from JSON Schema or custom JSON configuration. This enables configuration-driven model creation with full comparison capabilities, perfect for runtime model generation, A/B testing, and integration with external systems.

## Overview

Stickler provides two methods for dynamic model creation:

1. **`from_json_schema()`** - Create models from standard JSON Schema documents (recommended)
2. **`model_from_json()`** - Create models from custom Stickler JSON configuration

Both methods produce fully functional StructuredModel classes with:
- Full comparison capabilities via `compare_with()`
- Nested StructuredModel hierarchies
- Custom comparators and thresholds
- Lists of StructuredModels with Hungarian matching
- Pydantic validation and serialization

## Method 1: JSON Schema (Recommended)

### Why JSON Schema?

- **Industry standard**: Works with existing JSON Schema tooling and validators
- **Interoperability**: Integrate with OpenAPI, AsyncAPI, and other schema-based systems
- **Documentation**: Built-in support for descriptions, examples, and validation
- **Extensibility**: Use `x-aws-stickler-*` extensions for comparison configuration

### Basic JSON Schema Example

```python
from stickler import StructuredModel
import json

# Define a standard JSON Schema with Stickler extensions
product_schema = {
    "type": "object",
    "title": "Product",
    "x-aws-stickler-model-name": "Product",
    "properties": {
        "name": {
            "type": "string",
            "description": "Product name",
            "x-aws-stickler-comparator": "LevenshteinComparator",
            "x-aws-stickler-threshold": 0.8,
            "x-aws-stickler-weight": 2.0
        },
        "price": {
            "type": "number",
            "description": "Product price in USD",
            "x-aws-stickler-comparator": "NumericComparator",
            "x-aws-stickler-threshold": 0.95,
            "x-aws-stickler-weight": 1.5
        },
        "in_stock": {
            "type": "boolean",
            "description": "Availability status"
        }
    },
    "required": ["name", "price"]
}

# Create the model class from JSON Schema
Product = StructuredModel.from_json_schema(product_schema)

# Use it with JSON data (typical usage)
ground_truth_json = {"name": "Laptop", "price": 999.99, "in_stock": True}
prediction_json = {"name": "Laptop Pro", "price": 999.99, "in_stock": True}

ground_truth = Product(**ground_truth_json)
prediction = Product(**prediction_json)

# Compare
result = ground_truth.compare_with(prediction)
print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Name Score: {result['field_scores']['name']:.3f}")
print(f"Price Score: {result['field_scores']['price']:.3f}")
```

### Nested Objects in JSON Schema

JSON Schema naturally supports nested objects, which become nested StructuredModels:

```python
invoice_schema = {
    "type": "object",
    "x-aws-stickler-model-name": "Invoice",
    "properties": {
        "invoice_number": {
            "type": "string",
            "x-aws-stickler-comparator": "ExactComparator",
            "x-aws-stickler-threshold": 1.0,
            "x-aws-stickler-weight": 3.0
        },
        "customer": {
            "type": "object",
            "description": "Customer information",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                    "x-aws-stickler-threshold": 0.8
                },
                "email": {
                    "type": "string",
                    "x-aws-stickler-comparator": "ExactComparator",
                    "x-aws-stickler-threshold": 1.0
                }
            },
            "required": ["name"]
        }
    },
    "required": ["invoice_number", "customer"]
}

Invoice = StructuredModel.from_json_schema(invoice_schema)

# Use with JSON data
invoice_json = {
    "invoice_number": "INV-001",
    "customer": {"name": "John Doe", "email": "john@example.com"}
}
invoice = Invoice(**invoice_json)
```

### Arrays in JSON Schema

Arrays of primitives and arrays of objects are both supported:

```python
order_schema = {
    "type": "object",
    "x-aws-stickler-model-name": "Order",
    "properties": {
        "order_id": {
            "type": "string",
            "x-aws-stickler-comparator": "ExactComparator"
        },
        "tags": {
            "type": "array",
            "description": "Simple array of strings",
            "items": {"type": "string"}
        },
        "line_items": {
            "type": "array",
            "description": "Array of objects - uses Hungarian matching",
            "items": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "x-aws-stickler-comparator": "LevenshteinComparator",
                        "x-aws-stickler-threshold": 0.8
                    },
                    "quantity": {
                        "type": "integer",
                        "x-aws-stickler-comparator": "NumericComparator"
                    },
                    "price": {
                        "type": "number",
                        "x-aws-stickler-comparator": "NumericComparator",
                        "x-aws-stickler-threshold": 0.95
                    }
                },
                "required": ["product", "quantity", "price"]
            }
        }
    },
    "required": ["order_id", "line_items"]
}

Order = StructuredModel.from_json_schema(order_schema)

# Use with JSON data - arrays are compared with Hungarian matching (order-independent)
order1_json = {
    "order_id": "ORD-001",
    "tags": ["electronics", "urgent"],
    "line_items": [
        {"product": "Widget A", "quantity": 2, "price": 50.00},
        {"product": "Widget B", "quantity": 1, "price": 100.00}
    ]
}

order2_json = {
    "order_id": "ORD-001",
    "tags": ["electronics", "urgent"],
    "line_items": [
        {"product": "Widget B", "quantity": 1, "price": 100.00},  # Reordered
        {"product": "Widget A", "quantity": 2, "price": 50.00}   # Reordered
    ]
}

order1 = Order(**order1_json)
order2 = Order(**order2_json)

result = order1.compare_with(order2)
# Line items will match perfectly despite reordering
```

### Complete JSON Schema Example

Here's a production-ready schema with all features:

```python
# Save this as invoice_schema.json
invoice_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "title": "Invoice",
    "description": "Invoice extraction schema with business-aligned comparison",
    "x-aws-stickler-model-name": "Invoice",
    "x-aws-stickler-match-threshold": 0.75,
    "properties": {
        "invoice_id": {
            "type": "string",
            "description": "Unique invoice identifier - must be exact",
            "examples": ["INV-2024-001", "INV-2024-002"],
            "x-aws-stickler-comparator": "ExactComparator",
            "x-aws-stickler-threshold": 1.0,
            "x-aws-stickler-weight": 3.0,
            "x-aws-stickler-clip-under-threshold": true
        },
        "customer_name": {
            "type": "string",
            "description": "Customer's full name - allow minor typos",
            "examples": ["John Smith", "Acme Corporation"],
            "x-aws-stickler-comparator": "LevenshteinComparator",
            "x-aws-stickler-threshold": 0.8,
            "x-aws-stickler-weight": 1.5
        },
        "total_amount": {
            "type": "number",
            "description": "Total invoice amount in USD",
            "examples": [1234.56, 99.99],
            "x-aws-stickler-comparator": "NumericComparator",
            "x-aws-stickler-threshold": 0.95,
            "x-aws-stickler-weight": 2.5
        },
        "line_items": {
            "type": "array",
            "description": "Individual line items",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "x-aws-stickler-comparator": "FuzzyComparator",
                        "x-aws-stickler-threshold": 0.7
                    },
                    "quantity": {
                        "type": "integer",
                        "x-aws-stickler-comparator": "NumericComparator",
                        "x-aws-stickler-threshold": 1.0
                    },
                    "unit_price": {
                        "type": "number",
                        "x-aws-stickler-comparator": "NumericComparator",
                        "x-aws-stickler-threshold": 0.95
                    }
                },
                "required": ["description", "quantity", "unit_price"]
            }
        },
        "metadata": {
            "type": "object",
            "description": "Additional metadata",
            "properties": {
                "processed_date": {"type": "string"},
                "processor_id": {"type": "string"}
            }
        }
    },
    "required": ["invoice_id", "customer_name", "total_amount", "line_items"]
}

# Load from file and create model
import json
with open('invoice_schema.json') as f:
    schema = json.load(f)

Invoice = StructuredModel.from_json_schema(schema)

# Use with JSON data (typical workflow)
ground_truth_json = {
    "invoice_id": "INV-2024-001",
    "customer_name": "Acme Corporation",
    "total_amount": 1250.00,
    "line_items": [
        {"description": "Widget A", "quantity": 10, "unit_price": 50.00},
        {"description": "Widget B", "quantity": 5, "unit_price": 100.00}
    ],
    "metadata": {"processed_date": "2024-01-15", "processor_id": "SYS-A"}
}

prediction_json = {
    "invoice_id": "INV-2024-001",
    "customer_name": "ACME Corp",  # Variation
    "total_amount": 1250.00,
    "line_items": [
        {"description": "Widget B", "quantity": 5, "unit_price": 100.00},  # Reordered
        {"description": "Widget A", "quantity": 10, "unit_price": 50.00}   # Reordered
    ],
    "metadata": {"processed_date": "2024-01-15", "processor_id": "SYS-B"}
}

ground_truth = Invoice(**ground_truth_json)
prediction = Invoice(**prediction_json)

result = ground_truth.compare_with(prediction)
print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Field Scores: {result['field_scores']}")
```

### JSON Schema Extension Reference

For complete documentation of all `x-aws-stickler-*` extensions, see the [README](../README.md#json-schema-extensions-x-aws-stickler--complete-reference).

**Quick Reference:**

| Extension | Purpose | Example Value |
|-----------|---------|---------------|
| `x-aws-stickler-comparator` | Comparison algorithm | `"LevenshteinComparator"` |
| `x-aws-stickler-threshold` | Match threshold (0.0-1.0) | `0.8` |
| `x-aws-stickler-weight` | Field importance | `2.0` |
| `x-aws-stickler-clip-under-threshold` | Zero low scores | `true` |
| `x-aws-stickler-aggregate` | Include in metrics | `true` |
| `x-aws-stickler-model-name` | Class name | `"Invoice"` |
| `x-aws-stickler-match-threshold` | Model threshold | `0.75` |

---

## Method 2: Custom JSON Configuration

For cases where you need more control or don't want to use JSON Schema, Stickler provides a custom configuration format via `model_from_json()`.

### Basic Custom Configuration

```python
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

# Define model configuration
person_config = {
    "model_name": "Person",
    "fields": {
        "name": {
            "type": "str",
            "comparator": "LevenshteinComparator",
            "threshold": 0.8,
            "weight": 1.0,
            "required": True
        },
        "age": {
            "type": "int",
            "comparator": "NumericComparator",
            "threshold": 0.9,
            "weight": 0.5,
            "required": True
        },
        "email": {
            "type": "str",
            "comparator": "ExactComparator",
            "threshold": 1.0,
            "weight": 1.5,
            "required": False,
            "default": None
        }
    }
}

# Create the model class
Person = StructuredModel.model_from_json(person_config)

# Use the model
person1 = Person(name="John Smith", age=30, email="john@example.com")
person2 = Person(name="Jon Smith", age=31, email="john@example.com")

result = person1.compare_with(person2)
print(f"Similarity: {result['overall_score']:.3f}")
```

### Custom Configuration Schema

#### Top-Level Configuration

```json
{
    "model_name": "string",           // Required: Name of the generated class
    "match_threshold": 0.7,           // Optional: Default threshold for list matching
    "fields": {                       // Required: Field definitions
        "field_name": { ... }
    }
}
```

#### Primitive Field Configuration

```json
{
    "type": "str|int|float|bool|list|dict",  // Required: Field type
    "comparator": "ComparatorName",          // Required: Comparator class name
    "comparator_config": { ... },            // Optional: Comparator configuration
    "threshold": 0.8,                        // Optional: Comparison threshold (0.0-1.0)
    "weight": 1.0,                          // Optional: Field weight (default: 1.0)
    "required": true,                        // Optional: Whether field is required
    "default": null,                         // Optional: Default value
    "aggregate": false,                      // Optional: Enable aggregation
    "clip_under_threshold": true,            // Optional: Clip scores under threshold
    "alias": "alternative_name",             // Optional: Field alias
    "description": "Field description",      // Optional: Field description
    "examples": ["example1", "example2"]     // Optional: Example values
}
```

#### Nested StructuredModel Fields

```json
{
    "type": "structured_model",              // Single nested model
    "threshold": 0.7,                        // Optional: Nested model threshold
    "weight": 1.0,                          // Optional: Field weight
    "fields": {                             // Required: Nested field definitions
        "nested_field": { ... }
    }
}
```

#### List of StructuredModels

```json
{
    "type": "list_structured_model",         // List of nested models
    "weight": 1.0,                          // Optional: Field weight
    "match_threshold": 0.7,                 // Optional: Hungarian matching threshold
    "fields": {                             // Required: Element field definitions
        "element_field": { ... }
    }
}
```

#### Optional StructuredModel Fields

```json
{
    "type": "optional_structured_model",     // Optional nested model
    "threshold": 0.7,                        // Optional: Nested model threshold
    "weight": 1.0,                          // Optional: Field weight
    "fields": {                             // Required: Nested field definitions
        "nested_field": { ... }
    }
}
```

## Supported Types

### Primitive Types
- `str`: String values
- `int`: Integer values  
- `float`: Floating-point values
- `bool`: Boolean values
- `list`: List of values
- `dict`: Dictionary/object values
- `tuple`: Tuple values
- `set`: Set values

### Generic Types
- `List`: Typed list (equivalent to `list`)
- `Dict`: Typed dictionary (equivalent to `dict`)
- `Tuple`: Typed tuple (equivalent to `tuple`)
- `Set`: Typed set (equivalent to `set`)
- `Optional`: Optional type wrapper
- `Union`: Union type
- `Any`: Any type

### StructuredModel Types
- `structured_model`: Single nested StructuredModel
- `list_structured_model`: List of StructuredModels
- `optional_structured_model`: Optional StructuredModel

## Available Comparators

### String Comparators
- `ExactComparator`: Exact string matching
- `LevenshteinComparator`: Edit distance-based comparison
- `FuzzyComparator`: Fuzzy string matching

### Numeric Comparators
- `NumericComparator`: Numeric value comparison with tolerance

### Structured Comparators
- `StructuredComparator`: For nested object comparison

### Configuration Examples

```json
{
    "comparator": "LevenshteinComparator",
    "comparator_config": {
        "case_sensitive": false
    }
}
```

```json
{
    "comparator": "NumericComparator", 
    "comparator_config": {
        "tolerance": 0.05  // 5% tolerance
    }
}
```

### Nested Models with Custom Configuration

#### Single Nested Model

```python
company_config = {
    "model_name": "Company",
    "fields": {
        "name": {
            "type": "str",
            "comparator": "LevenshteinComparator",
            "threshold": 0.8,
            "weight": 2.0
        },
        "ceo": {
            "type": "structured_model",
            "threshold": 0.7,
            "weight": 1.5,
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.8,
                    "weight": 1.0
                },
                "salary": {
                    "type": "float",
                    "comparator": "NumericComparator",
                    "threshold": 0.9,
                    "weight": 0.8
                }
            }
        }
    }
}

Company = StructuredModel.model_from_json(company_config)

# Use with JSON data
company_json = {
    "name": "TechCorp",
    "ceo": {"name": "Alice Johnson", "salary": 250000.0}
}
company = Company(**company_json)
```

#### List of Nested Models

```python
company_config = {
    "model_name": "Company",
    "fields": {
        "name": {
            "type": "str",
            "comparator": "LevenshteinComparator",
            "threshold": 0.8,
            "weight": 2.0
        },
        "employees": {
            "type": "list_structured_model",
            "weight": 1.0,
            "match_threshold": 0.7,
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.8,
                    "weight": 1.0
                },
                "department": {
                    "type": "str",
                    "comparator": "ExactComparator",
                    "threshold": 1.0,
                    "weight": 0.5
                },
                "salary": {
                    "type": "float",
                    "comparator": "NumericComparator",
                    "threshold": 0.95,
                    "weight": 0.7
                }
            }
        }
    }
}

Company = StructuredModel.model_from_json(company_config)

# Use with JSON data
company_json = {
    "name": "TechCorp",
    "employees": [
        {"name": "Bob Smith", "department": "Engineering", "salary": 85000.0},
        {"name": "Carol Davis", "department": "Marketing", "salary": 70000.0}
    ]
}
company = Company(**company_json)
```

## Loading from JSON Files

```python
import json
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

# Load configuration from file
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Create model class
MyModel = StructuredModel.model_from_json(config)

# Use the model
instance1 = MyModel(**data1)
instance2 = MyModel(**data2)
result = instance1.compare_with(instance2)
```

## Advanced Features

### Field Weights and Thresholds

```json
{
    "name": {
        "type": "str",
        "comparator": "LevenshteinComparator",
        "threshold": 0.8,    // Minimum similarity for match
        "weight": 2.0        // 2x importance in overall score
    },
    "optional_field": {
        "type": "str", 
        "comparator": "ExactComparator",
        "threshold": 1.0,
        "weight": 0.5,       // Half importance
        "required": false,
        "default": null
    }
}
```

### Aggregation Support

```json
{
    "score": {
        "type": "float",
        "comparator": "NumericComparator",
        "threshold": 0.9,
        "weight": 1.0,
        "aggregate": true    // Enable aggregation for this field
    }
}
```

### Threshold Clipping

```json
{
    "critical_field": {
        "type": "str",
        "comparator": "ExactComparator", 
        "threshold": 1.0,
        "weight": 3.0,
        "clip_under_threshold": true  // Set score to 0 if under threshold
    }
}
```

## Hungarian Matching for Lists

When using `list_structured_model`, the system automatically applies Hungarian matching to find the optimal pairing between list elements:

```python
# Lists are compared using Hungarian matching
company1 = Company(
    name="TechCorp",
    employees=[
        {"name": "Alice", "department": "Engineering"},
        {"name": "Bob", "department": "Marketing"}
    ]
)

company2 = Company(
    name="TechCorp", 
    employees=[
        {"name": "Bob", "department": "Marketing"},    # Reordered
        {"name": "Alice", "department": "Engineering"} # Reordered
    ]
)

# Hungarian matching finds optimal pairing despite reordering
result = company1.compare_with(company2)
```

## Error Handling

The system provides detailed error messages for configuration issues:

```python
try:
    Model = StructuredModel.model_from_json(config)
except ValueError as e:
    print(f"Configuration error: {e}")
    # Example: "Invalid type for field 'age': Unknown type: 'integer'"
    # Example: "Field 'name' missing required 'comparator' parameter"
```

## Best Practices

### 1. Field Naming
- Use descriptive field names
- Follow consistent naming conventions
- Avoid reserved Python keywords

### 2. Threshold Selection
- Start with default thresholds (0.7-0.8)
- Adjust based on data characteristics
- Use higher thresholds for critical fields

### 3. Weight Assignment
- Assign higher weights to more important fields
- Consider the relative importance in your domain
- Test with representative data

### 4. Nested Model Design
- Keep nesting levels reasonable (2-3 levels max)
- Group related fields into nested models
- Use meaningful names for nested model classes

### 5. List Matching
- Set appropriate `match_threshold` for list elements
- Consider the expected similarity of list items
- Test with various list sizes and orderings

## Performance Considerations

### Model Creation
- Model classes are created once and can be reused
- Cache created model classes for better performance
- Avoid recreating models unnecessarily

### Comparison Performance
- Nested models add computational overhead
- List comparisons use O(n³) Hungarian algorithm
- Consider field weights to optimize important comparisons

### Memory Usage
- Dynamic models have similar memory footprint to static models
- Nested models create additional object instances
- Large lists of nested models can consume significant memory

## Integration Examples

### Configuration-Driven Applications

```python
class ModelFactory:
    def __init__(self, config_dir):
        self.models = {}
        self.load_models(config_dir)
    
    def load_models(self, config_dir):
        for config_file in Path(config_dir).glob("*.json"):
            with open(config_file) as f:
                config = json.load(f)
            model_name = config["model_name"]
            self.models[model_name] = StructuredModel.model_from_json(config)
    
    def get_model(self, name):
        return self.models[name]

# Usage
factory = ModelFactory("model_configs/")
PersonModel = factory.get_model("Person")
CompanyModel = factory.get_model("Company")
```

### API Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/compare', methods=['POST'])
def compare_objects():
    data = request.json
    config = data['model_config']
    obj1_data = data['object1']
    obj2_data = data['object2']
    
    # Create model dynamically
    Model = StructuredModel.model_from_json(config)
    
    # Create instances and compare
    obj1 = Model(**obj1_data)
    obj2 = Model(**obj2_data)
    result = obj1.compare_with(obj2)
    
    return jsonify(result)
```

## Troubleshooting

### Common Issues

1. **"Unknown type" errors**: Check that the type string is in the supported types list
2. **"Missing comparator" errors**: Ensure primitive fields have comparator specified
3. **"Invalid threshold" errors**: Thresholds must be between 0.0 and 1.0
4. **Nested model validation errors**: Check that nested field configurations are valid

### Debugging Tips

1. Start with simple configurations and add complexity gradually
2. Use the validation methods to check configurations before creating models
3. Test with small datasets before scaling up
4. Check the generated model's `__annotations__` to verify field types

### Validation

```python
from stickler.structured_object_evaluator.models.field_converter import validate_fields_config

# Validate configuration before creating model
try:
    validate_fields_config(config['fields'])
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Example Scripts

Stickler includes complete working examples for both methods:

### JSON Schema Examples

**[`examples/scripts/json_schema_demo.py`](../examples/scripts/json_schema_demo.py)**

Comprehensive examples showing:
- Basic JSON Schema model creation
- Nested objects and arrays
- Custom `x-aws-stickler-*` extensions
- Real-world API response schemas
- Complete evaluation workflows

Run it:
```bash
python examples/scripts/json_schema_demo.py
```

### Custom Configuration Examples

**[`examples/scripts/model_from_json_demo.py`](../examples/scripts/model_from_json_demo.py)**

Comprehensive examples showing:
- Basic model creation from custom JSON
- Nested StructuredModels
- Lists of StructuredModels with Hungarian matching
- Custom comparator configuration
- Loading from JSON files

Run it:
```bash
python examples/scripts/model_from_json_demo.py
```

### Complete JSON-to-Evaluation Workflow

**[`examples/scripts/json_to_evaluation_demo.py`](../examples/scripts/json_to_evaluation_demo.py)**

End-to-end example showing:
- Loading model configuration from JSON
- Loading test data from JSON
- Creating models dynamically
- Running evaluations
- No Python object construction required

Run it:
```bash
python examples/scripts/json_to_evaluation_demo.py
```

## See Also

- [README: JSON Schema Extensions Reference](../README.md#json-schema-extensions-x-aws-stickler--complete-reference)
- [StructuredModel Advanced Functionality](StructuredModel_Advanced_Functionality.md)
- [Comparators Documentation](../src/stickler/comparators/Comparators.md)
