---
title: StructuredModel Compare
---

# StructuredModel `compare_with` Method: A Layman's Guide

VALIDATED AGAINST ACTUAL OUTPUT - This documentation has been validated against runtime output to ensure accuracy. All output structures shown are exact representations of what the method returns.

## Quick Reference: Output Modes

| Parameters | Top-Level Keys | Use Case |
|------------|----------------|----------|
| Default (none) | `field_scores`, `overall_score`, `all_fields_matched` | Basic similarity scoring |
| `include_confusion_matrix=True` | + `confusion_matrix` | Detailed metrics and analysis |
| `document_non_matches=True` | + `non_matches` | Debugging field failures |
| `document_field_comparisons=True` | + `field_comparisons` | Comprehensive field-level analysis |
| `add_confidence_metrics=True` | + `auroc_confidence_metric` | Confidence calibration evaluation |
| `evaluator_format=True` | `overall`, `fields`, `confusion_matrix`, `non_matches` | Evaluation tool integration |

## Table of Contents
1. [What It Does](#what-it-does)
2. [Why It Matters](#why-it-matters)
3. [Complete Output Structure Reference](#complete-output-structure-reference)
4. [Important Notes](#important-notes)
5. [Usage Examples](#usage-examples)
6. [Appendix: How It Works](#appendix-how-it-works)
7. [Appendix: Flow Chart](#appendix-flow-chart)
8. [Appendix: Architecture Overview](#appendix-architecture-overview)

## What It Does

The `compare_with` method is like having a super-smart assistant that can compare two complex documents (like invoices, contracts, or product catalogs) and tell you:

- **How similar they are** (as a percentage score)
- **Which specific parts match or don't match** (field-by-field analysis)
- **Detailed statistics** about the comparison (like how many items were correct, incorrect, or missing)

Think of it like comparing two shopping receipts - but instead of just looking at totals, it can compare every line item, every date, every store detail, and even handle cases where items are in different orders or some information is missing.


## Complete Output Structure Reference

The `compare_with` method returns different structures based on the parameters provided.

### Output Mode 1: Basic Comparison (Default)

When called with no optional parameters:

```python
result = model1.compare_with(model2)
```

Returns three top-level keys:

```json
{
  "field_scores": {
    "field_name": 0.85,
    "another_field": 1.0
  },
  "overall_score": 0.92,
  "all_fields_matched": true
}
```

#### field_scores (dict)
Maps each field name to its similarity score (0.0 to 1.0). Uses `threshold_applied_score` which respects the `clip_under_threshold` setting. For nested objects, shows the weighted average of sub-fields.

#### overall_score (float)
Weighted average of all field scores (0.0 to 1.0). Calculated as: `sum(field_score * field_weight) / sum(field_weights)`

#### all_fields_matched (bool)
True when every field's `raw_similarity_score >= threshold`. False if any field falls below its configured threshold.

### Output Mode 2: With Confusion Matrix

When called with `include_confusion_matrix=True`:

```python
result = model1.compare_with(model2, include_confusion_matrix=True)
```

Returns all basic output keys plus a `confusion_matrix` key:
```json
{
  "field_scores": {...},
  "overall_score": 0.92,
  "all_fields_matched": true,
  "confusion_matrix": {
    "overall": {
      "tp": 5, "fa": 1, "fd": 2, "fp": 3, "tn": 0, "fn": 1,
      "similarity_score": 0.92,
      "all_fields_matched": true,
      "derived": {
        "cm_precision": 0.83,
        "cm_recall": 0.91,
        "cm_f1": 0.87,
        "cm_accuracy": 0.85
      }
    },
    "fields": {
      "field_name": {
        "overall": {
          "tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0,
          "similarity_score": 1.0,
          "all_fields_matched": true
        },
        "fields": {},
        "aggregate": {
          "tp": 1, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0,
          "derived": {
            "cm_precision": 1.0,
            "cm_recall": 1.0,
            "cm_f1": 1.0,
            "cm_accuracy": 1.0
          }
        }
      }
    },
    "non_matches": [],
    "aggregate": {
      "tp": 5, "fa": 1, "fd": 2, "fp": 3, "tn": 0, "fn": 1,
      "derived": {
        "cm_precision": 0.83,
        "cm_recall": 0.91,
        "cm_f1": 0.87,
        "cm_accuracy": 0.85
      }
    }
  }
}
```

#### Confusion Matrix Structure

The `confusion_matrix` object contains four keys:

**confusion_matrix.overall** - Object-level metrics for the current hierarchical level
**confusion_matrix.fields** - Field-by-field breakdown with nested structure  
**confusion_matrix.non_matches** - Empty list (populated at top level with `document_non_matches=True`)
**confusion_matrix.aggregate** - Sum of ALL primitive field metrics recursively below this node

#### Understanding overall vs aggregate

These two keys serve different purposes:

- **overall**: Counts objects/fields at the current level only
  - Example: 2 line items matched = `tp: 2`
- **aggregate**: Sums all primitive fields recursively below this node
  - Example: 2 items with 3 fields each = `tp: 6`

#### Confusion Matrix Metrics

Each confusion matrix node contains these integer counts:

- **tp** (True Positives): Fields that matched above threshold
- **fa** (False Alarms): Predicted fields that shouldn't exist (ground truth is null)
- **fd** (False Discoveries): Fields exist in both but don't match (below threshold)
- **fp** (False Positives): Sum of `fa + fd`
- **tn** (True Negatives): Correctly identified null/empty fields
- **fn** (False Negatives): Missing fields that should exist (prediction is null)

Plus these additional fields:

- **similarity_score** (float): Same as top-level `overall_score`
- **all_fields_matched** (bool): Same as top-level `all_fields_matched`

#### Derived Metrics

When `add_derived_metrics=True` (default), each node includes a `derived` object:

- **cm_precision**: `tp / (tp + fp)` - Accuracy of positive predictions
- **cm_recall**: `tp / (tp + fn)` or `tp / (tp + fn + fd)` if `recall_with_fd=True`
- **cm_f1**: `2 * (precision * recall) / (precision + recall)` - Harmonic mean
- **cm_accuracy**: `(tp + tn) / (tp + tn + fp + fn)` - Overall correctness

### Output Mode 3: With Non-Matches Documentation

When called with `document_non_matches=True`:

```python
result = model1.compare_with(model2, document_non_matches=True)
```

Adds a `non_matches` list to the top level containing details about fields that failed to match:

```json
{
  "field_scores": {...},
  "overall_score": 0.75,
  "all_fields_matched": false,
  "non_matches": [
    {
      "field_path": "contact.phone",
      "non_match_type": "false_discovery",
      "ground_truth_value": "555-123-4567",
      "prediction_value": "555-999-8888",
      "similarity_score": 0.3,
      "details": {
        "reason": "below threshold (0.300 < 1.0)"
      }
    }
  ]
}
```

#### Non-Match Entry Fields

- **field_path**: Dot-notation path to the field (e.g., "contact.phone")
- **non_match_type**: Type of mismatch
  - `false_discovery`: Field exists in both but doesn't match threshold
  - `false_alarm`: Field exists in prediction but not in ground truth
  - `false_negative`: Field exists in ground truth but not in prediction
- **ground_truth_value**: The expected value
- **prediction_value**: The predicted value
- **similarity_score**: Actual similarity score achieved
- **details**: Additional information including human-readable reason

### Output Mode 4: With Field Comparisons Documentation

When called with `document_field_comparisons=True`:

```python
result = model1.compare_with(model2, document_field_comparisons=True)
```

Adds a `field_comparisons` list to the top level containing detailed information about ALL field-level comparisons (both matches and non-matches):

```json
{
  "field_scores": {...},
  "overall_score": 0.85,
  "all_fields_matched": false,
  "field_comparisons": [
    {
      "expected_key": "invoice_id",
      "expected_value": "INV-2024-001",
      "actual_key": "invoice_id", 
      "actual_value": "INV-2024-001",
      "match": true,
      "score": 1.0,
      "weighted_score": 3.0,
      "reason": "exact match"
    },
    {
      "expected_key": "customer_name",
      "expected_value": "Acme Corporation",
      "actual_key": "customer_name",
      "actual_value": "ACME Corp",
      "match": true,
      "score": 0.85,
      "weighted_score": 1.275,
      "reason": "above threshold (0.850 >= 0.8)"
    },
    {
      "expected_key": "line_items[0].description",
      "expected_value": "Widget A",
      "actual_key": "line_items[1].description", 
      "actual_value": "Widget A",
      "match": true,
      "score": 1.0,
      "weighted_score": 1.0,
      "reason": "exact match"
    },
    {
      "expected_key": "contact.phone",
      "expected_value": "555-123-4567",
      "actual_key": "contact.phone",
      "actual_value": "555-999-8888", 
      "match": false,
      "score": 0.3,
      "weighted_score": 0.0,
      "reason": "below threshold (0.300 < 1.0)"
    }
  ]
}
```

#### Field Comparison Entry Fields

- **expected_key**: Field path in ground truth (dot notation for nested fields, bracket notation for lists). This will always have a value.
- **expected_value**: The ground truth value for this field
- **actual_key**: Field path in prediction (may differ for list items due to Hungarian matching). This will be None if the prediction does not exist.
- **actual_value**: The predicted value for this field
- **match**: Boolean indicating if this field comparison was considered a match (score >= threshold)
- **score**: Raw similarity score between the values (0.0 to 1.0)
- **weighted_score**: Score multiplied by the field's weight (used in overall score calculation)
- **reason**: Human-readable explanation of the comparison result

#### Key Features of Field Comparisons

**Comprehensive Coverage**: Documents ALL field comparisons, including:
- Primitive fields (strings, numbers, booleans)
- Nested object fields (with dot notation paths)
- List item fields (with bracket notation and Hungarian matching results)
- Both matches and non-matches

**Hungarian Matching Awareness**: For list fields, shows which specific items were matched together:
- `expected_key: "line_items[0].product"` 
- `actual_key: "line_items[2].product"`
- This indicates the first ground truth item was matched with the third prediction item

**Hierarchical Field Paths**: Uses intuitive path notation:
- Simple fields: `"customer_name"`
- Nested objects: `"address.street"`
- List items: `"line_items[0].description"`
- Deeply nested: `"customer.addresses[1].street"`

### Output Mode 5: Evaluator Format

When called with `evaluator_format=True`:

```python
result = model1.compare_with(model2, evaluator_format=True)
```

Returns a COMPLETELY DIFFERENT structure optimized for evaluation tools:

```json
{
  "overall": {
    "precision": 0.659,
    "recall": 1.0,
    "f1": 0.794,
    "accuracy": 0.659,
    "anls_score": 0.659
  },
  "fields": {
    "field_name": {
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "accuracy": 1.0,
      "anls_score": 1.0
    }
  },
  "confusion_matrix": {},
  "non_matches": []
}
```

#### Critical Difference: overall vs overall_score

**IMPORTANT:** The key naming changes based on format:

- Standard output (default): Uses `overall_score` (float)
- Evaluator format: Uses `overall` (dict with metrics)

The top-level `overall` key ONLY exists when `evaluator_format=True`.

#### Evaluator Format Fields

- **overall**: Dictionary containing aggregate evaluation metrics (precision, recall, f1, accuracy, anls_score)
- **fields**: Maps field names to their evaluation metrics (same structure as overall)
- **confusion_matrix**: Empty dict in evaluator format
- **non_matches**: Empty list in evaluator format

### Parameter Reference

```python
result = model1.compare_with(
    other,                           # Required: Model to compare against
    include_confusion_matrix=True,   # Include detailed metrics
    document_non_matches=True,       # Document what didn't match
    document_field_comparisons=True, # Document all field-level comparisons
    add_confidence_metrics=True,     # Add AUROC confidence calibration metric
    evaluator_format=False,          # Format for evaluation tools
    recall_with_fd=False,           # Include FD in recall calculation
    add_derived_metrics=True,       # Add precision/recall/F1 metrics
    document_field_comparisons=False # Document all field-level comparisons
)
```

## Universal Aggregate Fields (NEW FEATURE)

### What Are Aggregate Fields?

Every node in the confusion matrix now automatically includes an `aggregate` field that contains the sum of all primitive field confusion matrices below that node. This provides universal field-level granularity without any configuration.

### Structure

```json
{
  "confusion_matrix": {
    "aggregate": {
      "tp": 8, "fa": 2, "fd": 1, "fp": 3, "tn": 0, "fn": 1,
      "derived": {...}
    },
    "fields": {
      "contact": {
        "aggregate": {
          "tp": 2, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0,
          "derived": {...}
        },
        "fields": {
          "phone": {
            "aggregate": {
              "tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0,
              "derived": {...}
            }
          }
        }
      }
    }
  }
}
```

### Benefits

1. **Universal Access**: Get aggregate metrics at any level without configuration
2. **Hierarchical Analysis**: Understand which sections of your data have issues
3. **Automatic**: Works out of the box for all comparisons
4. **Consistent**: Every node has the same structure

### Usage Examples

```python
# Get total primitive field metrics across entire comparison
total_tp = result['confusion_matrix']['aggregate']['tp']

# Get metrics for a specific section (e.g., contact information)
contact_metrics = result['confusion_matrix']['fields']['contact']['aggregate']
contact_f1 = contact_metrics['derived']['cm_f1']

# Get metrics for deeply nested fields
address_metrics = result['confusion_matrix']['fields']['customer']['fields']['address']['aggregate']
```

## Using the Method

### Basic Usage
```python
# Simple comparison
result = model1.compare_with(model2)
print(f"Overall similarity: {result['overall_score']:.2%}")

# With detailed metrics
result = model1.compare_with(model2, include_confusion_matrix=True)
confusion_matrix = result['confusion_matrix']
print(f"True Positives: {confusion_matrix['overall']['tp']}")

# Access universal aggregate fields
total_aggregate = confusion_matrix['aggregate']
print(f"Total primitive TP: {total_aggregate['tp']}")
print(f"Overall F1: {total_aggregate['derived']['cm_f1']:.3f}")
```

### Advanced Options
```python
# Complete analysis with non-match documentation
result = model1.compare_with(
    model2,
    include_confusion_matrix=True,
    document_non_matches=True,
    evaluator_format=False
)

# Access field-level scores
field_scores = result['field_scores']
for field, score in field_scores.items():
    print(f"{field}: {score:.2%}")

# Access hierarchical aggregate metrics
cm = result['confusion_matrix']
for field_name, field_data in cm['fields'].items():
    if 'aggregate' in field_data:
        agg = field_data['aggregate']
        print(f"{field_name} section F1: {agg['derived']['cm_f1']:.3f}")

# Access non-matches for debugging
non_matches = result.get('non_matches', [])
for nm in non_matches:
    print(f"Field {nm['field_path']} failed: {nm['non_match_type']}")

# Comprehensive field-level analysis
result = model1.compare_with(
    model2,
    document_field_comparisons=True,
    include_confusion_matrix=True
)

# Access detailed field comparisons
field_comparisons = result.get('field_comparisons', [])
for fc in field_comparisons:
    status = "✓" if fc['match'] else "✗"
    print(f"{status} {fc['expected_key']}: {fc['score']:.3f} ({fc['reason']})")
    if fc['expected_key'] != fc['actual_key']:
        print(f"  → Matched with: {fc['actual_key']}")

# Filter for specific types of comparisons
matches = [fc for fc in field_comparisons if fc['match']]
non_matches = [fc for fc in field_comparisons if not fc['match']]
print(f"\nSummary: {len(matches)} matches, {len(non_matches)} non-matches")
```

### Aggregate Field Analysis
```python
# Analyze performance by data section
result = model1.compare_with(model2, include_confusion_matrix=True)
cm = result['confusion_matrix']

# Top-level summary
print("=== OVERALL PERFORMANCE ===")
total_agg = cm['aggregate']
print(f"Total Precision: {total_agg['derived']['cm_precision']:.3f}")
print(f"Total Recall: {total_agg['derived']['cm_recall']:.3f}")
print(f"Total F1: {total_agg['derived']['cm_f1']:.3f}")

# Section-by-section analysis
print("\n=== SECTION PERFORMANCE ===")
for section_name, section_data in cm['fields'].items():
    if 'aggregate' in section_data:
        agg = section_data['aggregate']
        f1 = agg['derived']['cm_f1']
        tp = agg['tp']
        total_errors = agg['fd'] + agg['fa'] + agg['fn']
        print(f"{section_name}: F1={f1:.3f}, TP={tp}, Errors={total_errors}")
```

## Important Notes

### The overall Key Has Multiple Meanings

The word "overall" appears in three different contexts with different meanings:

1. **confusion_matrix['overall']** - Object-level confusion matrix metrics (always present when confusion matrix is included)
2. **result['overall']** - Top-level evaluation metrics dictionary (ONLY exists when `evaluator_format=True`)
3. **fields[name]['overall']** - Field-level confusion matrix metrics

In standard output (default), there is NO top-level `overall` key. Use `overall_score` instead.

### Object-Level vs Primitive-Level Metrics

The confusion matrix provides two different views of the same data:

- **overall**: Counts objects/fields at the current hierarchical level only
- **aggregate**: Sums all primitive fields recursively below this node

Example: For 2 line items with 3 fields each:
- `line_items['overall']['tp'] = 2` means 2 line item objects matched
- `line_items['aggregate']['tp'] = 6` means 6 primitive fields matched total (2 items × 3 fields)

### Threshold Clipping Behavior

When `clip_under_threshold=True` (default):

- **raw_similarity_score**: Always shows the actual similarity value
- **threshold_applied_score**: Set to 0.0 if the score falls below the configured threshold
- **field_scores**: Uses `threshold_applied_score` for weighted averaging

This clipping behavior affects the final `overall_score` calculation.

### List Field Aggregation

For `List[StructuredModel]` fields, the nested `fields` section shows aggregated metrics across all matched items.

Example: If you have 2 matched line items:
```python
cm['fields']['line_items']['fields']['product']['overall']['tp'] = 2
```
This means the product field matched in BOTH line items.

---

## Appendix: How It Works

### The Big Picture

1. **Start**: You give it two structured objects to compare (like two invoices)
2. **Field-by-Field Analysis**: It looks at every piece of information in both objects
3. **Smart Matching**: For lists of items, it figures out which items should be compared to each other
4. **Scoring**: It calculates how similar each part is and combines them into an overall score
5. **Detailed Report**: It gives you both a simple score and detailed breakdown

### The Process Step-by-Step

1. **Preparation**: The method receives two objects and comparison options
2. **Recursive Traversal**: It walks through every field in both objects simultaneously
3. **Type-Specific Handling**: Different types of data get different comparison treatments:
   - Simple text/numbers: Direct comparison
   - Lists: Smart matching to pair up similar items
   - Nested objects: Recursive comparison of sub-fields
4. **Score Calculation**: Each comparison gets a similarity score (0.0 to 1.0)
5. **Aggregation**: All scores are combined using weighted averages
6. **Result Assembly**: Final results include scores, statistics, and detailed breakdowns

## Appendix: Validation

This documentation was validated against actual runtime output using:
- Python 3.12
- stickler-dev conda environment
- Test script: `test_output_validation.py`
- Date: 2024

All structures and examples shown are exact representations of actual output.

## Conclusion

The `compare_with` method represents a sophisticated document comparison system that balances accuracy, performance, and usability. It provides both simple similarity scores for basic use cases and detailed analytical data for advanced evaluation scenarios.

The method's strength lies in its ability to handle complex, nested data structures while maintaining interpretable results that can guide both automated systems and human reviewers in understanding how well document processing systems are performing.
