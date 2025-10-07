# StructuredModel `compare_with` Method: A Layman's Guide

## Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated 2025.

## Table of Contents
1. [What It Does](#what-it-does)
2. [Why It Matters](#why-it-matters)
3. [How It Works](#how-it-works)
4. [Flow Chart](#flow-chart)
5. [Key Concepts](#key-concepts)
6. [Examples](#examples)
7. [Architecture Overview](#architecture-overview)

## What It Does

The `compare_with` method is like having a super-smart assistant that can compare two complex documents (like invoices, contracts, or product catalogs) and tell you:

- **How similar they are** (as a percentage score)
- **Which specific parts match or don't match** (field-by-field analysis)
- **Detailed statistics** about the comparison (like how many items were correct, incorrect, or missing)

Think of it like comparing two shopping receipts - but instead of just looking at totals, it can compare every line item, every date, every store detail, and even handle cases where items are in different orders or some information is missing.

## Why It Matters

This system is crucial for:

- **Document Processing**: Automatically checking if extracted data matches expected formats
- **Quality Assurance**: Validating that AI systems correctly understand documents
- **Data Migration**: Ensuring data transfers preserve all important information
- **Compliance**: Proving that automated systems meet accuracy requirements

## How It Works

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

## Flow Chart

```
┌─────────────────────────┐
│   compare_with()        │ ← Entry Point
│   (Public Interface)    │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   compare_recursive()   │ ← Core Logic Engine
│   (Single Traversal)    │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   For Each Field:       │
│   _dispatch_field_      │ ← Field Processing Loop
│   _comparison()         │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   Field Type Check      │ ← Decision Point
└─────┬─────┬─────┬───────┘
      │     │     │
      ▼     ▼     ▼
┌─────────┐ ┌─────────┐ ┌─────────────┐
│Primitive│ │  List   │ │   Nested    │
│  Field  │ │  Field  │ │   Object    │
└────┬────┘ └────┬────┘ └──────┬──────┘
     │           │             │
     │           ▼             │
     │    ┌─────────────┐      │
     │    │  Hungarian  │      │ ← Smart List Matching
     │    │  Matching   │      │
     │    └──────┬──────┘      │
     │           │             │
     ▼           ▼             ▼
┌─────────────────────────────────┐
│     Score Calculation &        │ ← Similarity Assessment
│     Threshold Application      │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│    Result Aggregation &        │ ← Combine All Results
│    Score Percolation           │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│     Format Final Result        │ ← Output Generation
└─────────────────────────────────┘
```

## Pseudocode Logic

### Core `compare_with` Method

```
Algorithm: compare_with(other_model, options)
Input: other_model (StructuredModel), options (dict)

1. CALL recursive_result = compare_recursive(other_model)
2. EXTRACT field_scores from recursive_result
3. EXTRACT overall_score from recursive_result
4. EXTRACT all_fields_matched from recursive_result

5. CREATE result = {
     field_scores: field_scores,
     overall_score: overall_score,
     all_fields_matched: all_fields_matched
   }

6. IF options.include_confusion_matrix THEN
7.     ADD recursive_result to result.confusion_matrix
8. ENDIF

9. IF options.document_non_matches THEN
10.    ADD non_matches to result.non_matches
11. ENDIF

12. IF options.evaluator_format THEN
13.    TRANSFORM result for evaluator compatibility
14. ENDIF

15. RETURN result
```

### Core Recursive Comparison

```
Algorithm: compare_recursive(other_model)
Input: other_model (StructuredModel)

1. INITIALIZE total_score = 0, total_weight = 0
2. INITIALIZE overall_metrics = {tp: 0, fa: 0, fd: 0, fp: 0, tn: 0, fn: 0}
3. INITIALIZE field_results = {}

4. FOR each field_name IN model_fields DO
5.     GET gt_value = get_field(field_name)
6.     GET pred_value = other_model.get_field(field_name)
7.     
8.     CALL field_result = dispatch_field_comparison(field_name, gt_value, pred_value)
9.     
10.    SET field_results[field_name] = field_result
11.    AGGREGATE field_result INTO overall_metrics
12.    
13.    UPDATE total_score += field_result.score × field_result.weight
14.    UPDATE total_weight += field_result.weight
15. ENDFOR

16. CALCULATE overall_score = total_score / total_weight
17. RETURN {overall: overall_metrics, fields: field_results, overall_score: overall_score}
```

### Field Type Dispatch

```
Algorithm: dispatch_field_comparison(field_name, gt_val, pred_val)
Input: field_name (string), gt_val (any), pred_val (any)

1. GET field_config = get_field_config(field_name)
2. 
3. IF both_null(gt_val, pred_val) THEN
4.     RETURN true_negative_result()
5. ENDIF
6. 
7. IF one_null(gt_val, pred_val) THEN
8.     RETURN false_alarm_or_negative_result()
9. ENDIF
10. 
11. MATCH field_type:
12.     CASE primitive: RETURN compare_primitive(gt_val, pred_val, field_config)
13.     CASE list: RETURN compare_list(gt_val, pred_val, field_config)
14.     CASE nested_object: RETURN compare_nested(gt_val, pred_val, field_config)
15. ENDMATCH
```

### List Matching (Hungarian Algorithm)

```
Algorithm: hungarian_list_matching(gt_list, pred_list, config)
Input: gt_list, pred_list, config

1. IF both_empty(gt_list, pred_list) THEN
2.     RETURN perfect_match()
3. ENDIF

4. IF one_empty(gt_list, pred_list) THEN
5.     RETURN count_unmatched_items()
6. ENDIF

7. BUILD similarity_matrix FOR all pairs
8. RUN hungarian_algorithm(similarity_matrix)
9. GET matched_pairs WITH similarity_scores

10. CLASSIFY matches:
11.     IF similarity ≥ threshold THEN tp++
12.     ELSE fd++

13. COUNT unmatched_items:
14.     fn = unmatched_gt_items
15.     fa = unmatched_pred_items

16. CALCULATE overall_score = average_similarity × coverage_ratio
17. RETURN {tp, fd, fa, fn, overall_score, matched_pairs}
```

### Structured List Processing

```
Algorithm: compare_struct_list(gt_list, pred_list, field_name)
Input: gt_list, pred_list, field_name

1. GET match_threshold FROM gt_list[0].__class__
2. 
3. HANDLE empty_cases:
4.     IF both_empty THEN RETURN true_negative
5.     IF one_empty THEN RETURN false_alarms_or_negatives
6. 
7. RUN hungarian_matching(gt_list, pred_list)
8. GET matched_pairs WITH object_similarities
9. 
10. CLASSIFY objects:
11.     IF object_similarity ≥ match_threshold THEN tp_objects++
12.     ELSE fd_objects++
13. 
14. FOR good_matches_only DO
15.     FOR each sub_field IN object_fields DO
16.         COMPARE sub_field_values
17.         AGGREGATE sub_field_metrics
18.     ENDFOR
19. ENDFOR
20. 
21. RETURN {object_metrics, field_details, overall_score}
```

## Key Concepts

### 1. **Field Types**
- **Primitive Fields**: Simple data like names, dates, numbers
- **List Fields**: Collections of items (like transaction lists)
- **Nested Objects**: Complex structures containing other fields

### 2. **Hungarian Matching**
For lists of items (like products in an order), the system uses a sophisticated algorithm to figure out which items in list A should be compared to which items in list B. This handles cases where:
- Items are in different orders
- Some items are missing
- Lists have different lengths

**Example**: 
```
Ground Truth: [Apple, Banana, Cherry]
Prediction:   [Banana, Apple, Orange]
```
The algorithm pairs: Apple↔Apple, Banana↔Banana, and notes Cherry is missing while Orange is extra.

### 3. **Thresholds and Weights**
- **Threshold**: Minimum similarity score to consider two items as "matching" (e.g., 0.7 = 70%)
- **Weight**: How important each field is in the overall score (e.g., customer name might be weighted higher than order date)

### 4. **Confusion Matrix Metrics**
The system tracks detailed statistics:
- **True Positives (TP)**: Correctly identified matches
- **False Positives (FP)**: Incorrectly claimed matches  
- **False Negatives (FN)**: Missed actual matches
- **True Negatives (TN)**: Correctly identified non-matches

### 5. **Score Percolation**
Scores "bubble up" from detailed fields to overall scores:
```
Field Level:    name=0.9, date=1.0, amount=0.8
↓
Object Level:   Weighted average = 0.9
↓  
Document Level: Overall similarity = 0.9
```

## Examples

### Example 1: Simple Invoice Comparison

**Ground Truth Invoice**:
```json
{
  "invoice_number": "INV-001",
  "date": "2024-01-15",
  "amount": 150.00
}
```

**Predicted Invoice**:
```json
{
  "invoice_number": "INV-001",
  "date": "2024-01-15", 
  "amount": 155.00
}
```

**Result**:
- `invoice_number`: 1.0 (exact match)
- `date`: 1.0 (exact match)
- `amount`: 0.0 (differs by more than threshold)
- **Overall Score**: 0.67 (2 out of 3 fields match)

### Example 2: List Comparison with Hungarian Matching

**Ground Truth Products**:
```json
{
  "products": [
    {"name": "Widget A", "price": 10.00},
    {"name": "Widget B", "price": 20.00}
  ]
}
```

**Predicted Products**:
```json
{
  "products": [
    {"name": "Widget B", "price": 20.00},
    {"name": "Widget A", "price": 10.50}
  ]
}
```

**Hungarian Matching Process**:
1. Calculate similarity between all pairs
2. Find optimal assignment: GT[0]↔Pred[1], GT[1]↔Pred[0]
3. Compare matched pairs:
   - Widget A vs Widget A: High similarity (price slightly off)
   - Widget B vs Widget B: Perfect match

### Example 3: Nested Object Comparison

**Ground Truth Customer**:
```json
{
  "name": "John Doe",
  "address": {
    "street": "123 Main St",
    "city": "Springfield",
    "zip": "12345"
  }
}
```

The system recursively compares:
1. Top-level `name` field
2. Nested `address` object:
   - `address.street`
   - `address.city` 
   - `address.zip`

Each nested field contributes to the overall score.

## Architecture Overview

### Core Components

1. **StructuredModel**: The main class that defines comparable data structures
2. **ComparableField**: Configuration for how each field should be compared
3. **ComparisonHelper**: Utilities for field-level comparisons
4. **HungarianHelper**: Implements the Hungarian matching algorithm
5. **Various Helpers**: Specialized utilities for thresholds, metrics, formatting

### Helper Classes

- **MetricsHelper**: Calculates precision, recall, F1-score, etc.
- **ThresholdHelper**: Handles threshold application and edge cases  
- **NonMatchesHelper**: Documents what didn't match for debugging
- **FieldHelper**: Utilities for field type detection and null handling
- **ConfigurationHelper**: Manages field comparison settings

### Design Principles

1. **Single Traversal Optimization**: The system walks through the object structure only once, gathering both similarity scores and confusion matrix data simultaneously. This is much more efficient than separate passes.

2. **Type-Based Dispatch**: Different field types get different treatment:
   - **Primitives**: Direct comparator application with threshold checking
   - **Lists**: Hungarian matching for optimal pairing
   - **Nested Objects**: Recursive descent into sub-structures

3. **Threshold-Gated Recursion**: For complex nested structures, the system only performs detailed analysis on object pairs that meet similarity thresholds. Poor matches are treated as atomic failures.

4. **Hierarchical Result Structure**: Results maintain the same nested structure as the input objects, making it easy to understand which specific parts matched or failed.

### Method Flow Breakdown

#### 1. Entry Point: `compare_with()`
- **Purpose**: Public API that orchestrates the comparison
- **Parameters**: 
  - `other`: The object to compare against
  - `include_confusion_matrix`: Whether to include detailed metrics
  - `document_non_matches`: Whether to document failures
  - `evaluator_format`: Whether to format for evaluation tools
- **Returns**: Comprehensive comparison results

#### 2. Core Engine: `compare_recursive()`
- **Purpose**: Single-traversal comparison engine
- **Key Innovation**: Gathers similarity scores AND confusion matrix data in one pass
- **Output**: Hierarchical result structure with metrics at every level

#### 3. Field Dispatcher: `_dispatch_field_comparison()`
- **Purpose**: Routes each field to appropriate comparison logic based on type
- **Handles**: Null checks, type detection, and routing to specialized handlers
- **Uses Match Statements**: Clean, efficient type-based routing

#### 4. Specialized Handlers:

**`_compare_primitive_with_scores()`**:
- Handles simple fields (strings, numbers, dates)
- Applies comparators (Levenshtein, exact match, etc.)
- Converts similarity scores to binary classification metrics

**`_compare_primitive_list_with_scores()`**:
- Handles lists of simple items
- Uses Hungarian matching for optimal pairing
- Handles unmatched items as false positives/negatives

**`_compare_struct_list_with_scores()`**:
- Handles lists of complex objects
- Uses object-level Hungarian matching
- Supports threshold-gated recursion for field-level details
- Maintains separation between object-level and field-level metrics

#### 5. Hungarian Matching
- **Algorithm**: Optimal bipartite matching using the Hungarian algorithm
- **Handles**: Different list lengths, optimal pairing, similarity scoring
- **Output**: Matched pairs with similarity scores, plus unmatched items

#### 6. Score Calculation and Aggregation
- **Raw Scores**: Direct comparator outputs (0.0 to 1.0)
- **Threshold Application**: Convert to binary pass/fail based on thresholds
- **Weight Application**: Combine field scores using configured weights
- **Percolation**: Bubble up scores from leaf fields to parent objects

### Performance Optimizations

1. **Single Traversal**: Eliminates redundant object walks
2. **Lazy Evaluation**: Only calculates detailed metrics when requested
3. **Threshold Gating**: Skips expensive recursion for poor matches
4. **Efficient Matching**: Hungarian algorithm runs in O(n³) time

### Error Handling

- **Type Mismatches**: Graceful handling when field types don't match
- **Missing Fields**: Treats as null for comparison purposes
- **Circular References**: Prevented through proper object traversal
- **Memory Management**: Efficient result structure prevents memory bloat

## Complete Output Structure Reference

The `compare_with` method returns different structures based on the parameters provided. Here's the complete reference:

### 1. Basic Output (Default Parameters)

```python
result = model1.compare_with(model2)
```

**Structure**:
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

**Fields**:
- `field_scores`: Dictionary mapping field names to similarity scores (0.0-1.0)
- `overall_score`: Weighted average of all field scores (0.0-1.0)
- `all_fields_matched`: Boolean indicating if all fields met their thresholds

### 2. With Confusion Matrix

```python
result = model1.compare_with(model2, include_confusion_matrix=True)
```

**Structure**:
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

**Key Features**:
- **Universal Aggregate Fields**: Every node includes an `aggregate` field that sums all primitive field metrics below that node
- **Hierarchical Structure**: Maintains the same nested structure as your data models
- **Derived Metrics**: Automatic calculation of precision, recall, F1-score, and accuracy
- **Field-Level Granularity**: Access confusion matrix metrics at any level of nesting

### 3. With Non-Matches Documentation

```python
result = model1.compare_with(model2, document_non_matches=True)
```

**Additional Field**:
```json
{
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

**Non-Match Types**:
- `false_discovery`: Fields that don't meet similarity threshold
- `false_alarm`: Predicted fields that shouldn't exist
- `false_negative`: Missing fields that should exist

### 4. Evaluator Format

```python
result = model1.compare_with(model2, evaluator_format=True)
```

**Structure** (Optimized for evaluation tools):
```json
{
  "overall": {
    "precision": 0.83,
    "recall": 0.91,
    "f1": 0.87,
    "accuracy": 0.85,
    "anls_score": 0.89
  },
  "fields": {
    "field_name": {
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "accuracy": 1.0
    }
  },
  "confusion_matrix": {},
  "non_matches": []
}
```

### 5. Complete Parameter Reference

```python
result = model1.compare_with(
    other,                      # Required: Model to compare against
    include_confusion_matrix=True,   # Include detailed metrics
    document_non_matches=True,       # Document what didn't match
    evaluator_format=False,          # Format for evaluation tools
    recall_with_fd=False,           # Include FD in recall calculation
    add_derived_metrics=True        # Add precision/recall/F1 metrics
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

## Conclusion

The `compare_with` method represents a sophisticated document comparison system that balances accuracy, performance, and usability. It provides both simple similarity scores for basic use cases and detailed analytical data for advanced evaluation scenarios.

The method's strength lies in its ability to handle complex, nested data structures while maintaining interpretable results that can guide both automated systems and human reviewers in understanding how well document processing systems are performing.
