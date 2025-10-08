# Universal Aggregate Field Feature

## Overview

The Universal Aggregate Field feature provides automatic aggregation of confusion matrix metrics at every node in the comparison result tree. This feature replaces the previous field-level `aggregate=True` parameter with a universal approach that provides field-level granularity across all structured comparisons.

## Key Features

### 1. Universal Coverage
- **Automatic**: Every node in the comparison result tree now includes an `aggregate` field
- **No Configuration Required**: Works automatically without any field-level configuration
- **Consistent Structure**: Provides uniform access pattern across all comparison results

### 2. Correct Placement
- **Sibling Relationship**: The `aggregate` field appears as a sibling of `overall` and `fields`, not nested within `overall`
- **Hierarchical Consistency**: Available at every level of the tree structure
- **Easy Access**: Simple access pattern: `result['confusion_matrix']['aggregate']` or `result['confusion_matrix']['fields']['contact']['aggregate']`

### 3. Comprehensive Metrics
- **Primitive Field Summation**: Aggregates all primitive field confusion matrices below each node
- **Derived Metrics Included**: Each aggregate includes precision, recall, F1, and accuracy calculations
- **Hierarchical Rollup**: Parent nodes sum metrics from all child primitive fields

## Usage Examples

### Basic Usage

```python
from src.stickler.structured_object_evaluator.models.structured_model import StructuredModel
from src.stickler.structured_object_evaluator.models.comparable_field import ComparableField
from src.stickler.comparators.exact import ExactComparator

class Contact(StructuredModel):
    phone: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    email: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

class Person(StructuredModel):
    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    contact: Contact = ComparableField(comparator=ExactComparator(), threshold=1.0)

# Create test instances
gt = Person(name="John", contact=Contact(phone="123", email="john@test.com"))
pred = Person(name="John", contact=Contact(phone="456", email="john@test.com"))

# Compare with confusion matrix
result = gt.compare_with(pred, include_confusion_matrix=True)
cm = result['confusion_matrix']

# Access aggregate metrics at different levels
print("Top-level aggregate:", cm['aggregate'])
print("Contact aggregate:", cm['fields']['contact']['aggregate'])
print("Phone aggregate:", cm['fields']['contact']['fields']['phone']['aggregate'])
```

### Output Structure

```json
{
  "confusion_matrix": {
    "overall": {
      "tp": 1, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0,
      "similarity_score": 0.5,
      "all_fields_matched": false,
      "derived": { "cm_precision": 0.5, "cm_recall": 1.0, "cm_f1": 0.67, "cm_accuracy": 0.5 }
    },
    "aggregate": {
      "tp": 2, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0,
      "derived": { "cm_precision": 0.67, "cm_recall": 1.0, "cm_f1": 0.8, "cm_accuracy": 0.67 }
    },
    "fields": {
      "name": {
        "overall": { "tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0 },
        "aggregate": { "tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0 },
        "fields": {}
      },
      "contact": {
        "overall": { "tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0 },
        "aggregate": { "tp": 1, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0 },
        "fields": {
          "phone": {
            "overall": { "tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0 },
            "aggregate": { "tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0 }
          },
          "email": {
            "overall": { "tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0 },
            "aggregate": { "tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0 }
          }
        }
      }
    }
  }
}
```

## Migration from Legacy `aggregate=True`

### Deprecated Parameter
The `aggregate=True` parameter in `ComparableField` is now deprecated and triggers a warning:

```python
# DEPRECATED - triggers warning
field = ComparableField(aggregate=True)

# NEW - automatic universal aggregation
field = ComparableField()  # aggregate field automatically available
```

### Migration Steps

1. **Remove `aggregate=True`**: Remove all `aggregate=True` parameters from `ComparableField` definitions
2. **Update Access Patterns**: Change from field-specific aggregate access to universal aggregate access
3. **Test Compatibility**: Verify that existing code works with the new universal aggregate structure

### Before (Legacy)
```python
class MyModel(StructuredModel):
    contact: Contact = ComparableField(aggregate=True)  # Only this field had aggregation

# Access was field-specific
if 'aggregate' in result['confusion_matrix']['fields']['contact']:
    contact_aggregate = result['confusion_matrix']['fields']['contact']['aggregate']
```

### After (Universal)
```python
class MyModel(StructuredModel):
    contact: Contact = ComparableField()  # All fields automatically have aggregation

# Access is universal and consistent
contact_aggregate = result['confusion_matrix']['fields']['contact']['aggregate']
top_aggregate = result['confusion_matrix']['aggregate']
```

## Technical Implementation

### Calculation Logic
1. **Leaf Nodes**: For primitive fields, `aggregate` equals `overall` metrics
2. **Parent Nodes**: Sum all `aggregate` metrics from child fields
3. **Hierarchical Rollup**: Each level aggregates metrics from all primitive fields below it

### Performance Considerations
- **Single Traversal**: Aggregate calculation happens during the main comparison traversal
- **Efficient Summation**: Simple integer addition for each metric type
- **Memory Overhead**: Minimal additional memory for aggregate dictionaries

### Backward Compatibility
- **Existing Code**: All existing code continues to work unchanged
- **Deprecation Warning**: Legacy `aggregate=True` usage shows deprecation warning
- **Gradual Migration**: Teams can migrate at their own pace

## Benefits

### 1. Universal Field-Level Granularity
- **Complete Coverage**: Every field comparison now provides aggregate metrics
- **Consistent Interface**: Same access pattern regardless of field type or nesting level
- **No Configuration**: Works automatically without any setup

### 2. Enhanced Analysis Capabilities
- **Drill-Down Analysis**: Easily identify which parts of nested structures contribute to overall metrics
- **Comparative Analysis**: Compare aggregate metrics across different levels of the hierarchy
- **Debugging Support**: Quickly identify problematic areas in complex nested structures

### 3. Improved Developer Experience
- **Predictable Structure**: Always know that `aggregate` field will be available
- **Simplified Code**: No need to check for aggregate field existence
- **Better Tooling**: IDE autocomplete and type checking work consistently

## Use Cases

### 1. Complex Nested Structure Analysis
```python
# Analyze which parts of a complex invoice structure have issues
invoice_result = gt_invoice.compare_with(pred_invoice, include_confusion_matrix=True)

# Check overall document accuracy
print("Document aggregate:", invoice_result['confusion_matrix']['aggregate'])

# Check line items accuracy
print("Line items aggregate:", invoice_result['confusion_matrix']['fields']['line_items']['aggregate'])

# Check individual product accuracy within line items
for i, item in enumerate(invoice_result['confusion_matrix']['fields']['line_items']['fields']):
    print(f"Item {i} aggregate:", item['aggregate'])
```

### 2. Model Performance Monitoring
```python
# Monitor different aspects of model performance
results = []
for test_case in test_cases:
    result = gt.compare_with(pred, include_confusion_matrix=True)
    results.append({
        'overall': result['confusion_matrix']['aggregate'],
        'contact_info': result['confusion_matrix']['fields']['contact']['aggregate'],
        'personal_info': result['confusion_matrix']['fields']['name']['aggregate']
    })

# Analyze trends across different field types
```

### 3. Hierarchical Reporting
```python
def print_hierarchical_metrics(node, path=""):
    """Print aggregate metrics for all levels of a comparison result"""
    if 'aggregate' in node:
        metrics = node['aggregate']
        precision = metrics.get('derived', {}).get('cm_precision', 0)
        recall = metrics.get('derived', {}).get('cm_recall', 0)
        f1 = metrics.get('derived', {}).get('cm_f1', 0)
        
        print(f"{path}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    if 'fields' in node:
        for field_name, field_data in node['fields'].items():
            new_path = f"{path}.{field_name}" if path else field_name
            print_hierarchical_metrics(field_data, new_path)

# Usage
result = gt.compare_with(pred, include_confusion_matrix=True)
print_hierarchical_metrics(result['confusion_matrix'])
```

## Conclusion

The Universal Aggregate Field feature provides comprehensive field-level granularity for all structured model comparisons. By automatically including aggregate metrics at every node, it enables powerful analysis capabilities while maintaining backward compatibility and providing a consistent, predictable interface for developers.
