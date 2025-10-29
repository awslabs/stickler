---
title: StructuredModel from JSON
---


# StructuredModel Dynamic Creation from JSON

This document describes how to create StructuredModel classes dynamically from JSON configuration using the `model_from_json()` classmethod. This enables configuration-driven model creation with full comparison capabilities.

## Overview

The `StructuredModel.model_from_json()` method allows you to:

- Create StructuredModel classes from JSON configuration
- Define nested StructuredModel hierarchies
- Configure custom comparators and thresholds
- Support lists of StructuredModels with Hungarian matching
- Enable configuration-driven model creation for flexible applications

## Basic Usage

### Simple Model Creation

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

## Configuration Schema

### Top-Level Configuration

```json
{
    "model_name": "string",           // Required: Name of the generated class
    "match_threshold": 0.7,           // Optional: Default threshold for list matching
    "fields": {                       // Required: Field definitions
        "field_name": { ... }
    }
}
```

### Field Configuration

#### Primitive Fields

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

## Nested Model Examples

### Single Nested Model

```json
{
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
```

### List of Nested Models

```json
{
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

## See Also

- [StructuredModel Advanced Functionality](StructuredModel_Advanced_Functionality.md)
- [Comparators Documentation](../src/stickler/comparators/Comparators.md)
- [Examples](../examples/scripts/model_from_json_demo.py)
