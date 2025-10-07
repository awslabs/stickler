# Confusion Matrix Pretty Print

This module provides functionality for displaying structured confusion matrix metrics in a visually appealing and readable format.

## Features

- **Visually Enhanced Metrics**: Displays confusion matrix metrics with colored bars and visual indicators
- **Hierarchical Field Structure**: Properly displays nested fields with hierarchical formatting
- **Filtering and Sorting**: Ability to filter fields by regex pattern and sort by various metrics
- **Terminal Color Support**: Automatic detection of color support with fallback to plain text
- **File Output**: Option to save the output to a file
- **Integration with Evaluator**: Directly accessible from the StructuredModelEvaluator class

## Usage

### Standalone Functions

```python
from stickler.structured_object_evaluator.utils.pretty_print import print_confusion_matrix

# Basic usage
print_confusion_matrix(results)

# Advanced usage with options
print_confusion_matrix(
    results,
    field_filter="transactions\\..*",  # Only show transaction fields
    sort_by="f1",                     # Sort fields by F1 score
    show_details=True,                # Show detailed field metrics
    use_color=True,                   # Use color in terminal output
    output_file="metrics_output.txt", # Save output to file
    nested_detail="detailed"          # Level of detail for nested objects
)
```

### Integrated with Evaluator

```python
from stickler.structured_object_evaluator import StructuredModelEvaluator

# Create an evaluator
evaluator = StructuredModelEvaluator()

# Run evaluation
results = evaluator.evaluate(ground_truth, predictions)

# Pretty print the results directly from the evaluator
evaluator.pretty_print_results(results)

# With options
evaluator.pretty_print_results(
    results,
    field_filter="transactions\\.date|transactions\\.credit",
    sort_by="precision",
    nested_detail="detailed"  # Show detailed metrics for nested objects
)
```

## Nested Object Display Options

When dealing with complex nested objects like list fields (e.g., "transactions" with many items), 
you can control the level of detail shown with the `nested_detail` parameter:

- **minimal**: Shows only top-level fields, hiding nested fields
- **standard** (default): Shows nested fields with basic metrics
- **detailed**: Shows comprehensive metrics for nested fields, including per-item breakdowns

This is especially helpful when you have list fields with many items and want to analyze
performance at different levels of detail.

```python
# Example for handling nested objects
print_confusion_matrix(
    results, 
    field_filter="transactions", 
    nested_detail="detailed"
)
```

The detailed view will show individual metrics for each item in list fields,
allowing you to identify specific problematic instances.

## Output Example

```
=== CONFUSION MATRIX SUMMARY ===

--- Raw Counts ---
Metric             Count
-------------------------
True Positive         53
False Positive         0
True Negative          0
False Negative         0
False Discovery        1

--- Derived Metrics ---
Metric               Value Visual                
--------------------------------------------------
Precision           98.15% ███████████████████░  
Recall              98.15% ███████████████████░  
F1 Score            98.15% ███████████████████░  
Accuracy            98.15% ███████████████████░  


=== FIELD-LEVEL METRICS ===

Field                            TP         FP         TN         FN         FD         F1 Visual                
-------------------------------------------------------------------------------------------------------------
accountNumber                     1          0          0          0          0    100.00% ████████████████████  

period                            1          0          0          0          0    100.00% ████████████████████  

transactions                     51          0          0          0          1     98.08% ███████████████████░  
  date                           52          0          0          0          0    100.00% ████████████████████  
  description                    52          0          0          0          0    100.00% ████████████████████  
  total                          52          0          0          0          0    100.00% ████████████████████  
  credit                         31         21          0          0          0     74.70% ██████████████░░░░░░  
  debit                          20         31          0          0          1     54.79% ██████████░░░░░░░░░░  


=== CONFUSION MATRIX VISUALIZATION ===

TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

Legend:
  T True Positive (TP): 53 (98.15%)
  N True Negative (TN): 0 (0.00%)
  F False Positive (FP): 0 (0.00%)
  M False Negative (FN): 0 (0.00%)
  D False Discovery (FD): 1 (1.85%)
```

## Demo

A comprehensive demonstration script is available at:
`examples/key_information_evaluation/structured_object_evaluator/pretty_print_demo.py`
