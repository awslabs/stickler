---
title: Understanding Results
---

# Understanding Results

This guide explains how to read and interpret the output of Stickler evaluations -- from single-document comparisons to bulk evaluation aggregates.

---

## Result Structure

A default call to `compare_with()` returns a dictionary with three keys:

```python
result = ground_truth.compare_with(prediction)
```

```json
{
  "field_scores": {
    "invoice_id": 1.0,
    "customer_name": 0.85,
    "total_amount": 1.0,
    "notes": 0.62
  },
  "overall_score": 0.92,
  "all_fields_matched": false
}
```

### `overall_score` (float)

A weighted average of all field scores, ranging from 0.0 to 1.0. Calculated as:

```
overall_score = sum(field_score * field_weight) / sum(field_weights)
```

Fields with `clip_under_threshold=True` (the default) contribute 0.0 if they score below their threshold, rather than their partial similarity.

### `field_scores` (dict)

Maps each field name to its similarity score (0.0 to 1.0). For nested objects, the value is the weighted average of the sub-fields. For lists, it reflects the Hungarian-matched aggregate.

### `all_fields_matched` (bool)

`True` only when every field's raw similarity score meets or exceeds its configured threshold. A single field below threshold makes this `False`.

---

## Confusion Matrix

When you pass `include_confusion_matrix=True`, the result gains a `confusion_matrix` key with detailed classification counts.

```python
result = ground_truth.compare_with(prediction, include_confusion_matrix=True)
cm = result['confusion_matrix']
```

### Classification Categories

Stickler uses five categories -- not the standard four. The False Positive category is split into two subcategories to distinguish between fundamentally different error types:

| Category | Abbreviation | When It Applies |
|----------|--------------|-----------------|
| True Positive | TP | Ground truth has a value, prediction has a value, and they match (similarity >= threshold). |
| False Alarm | FA | Ground truth is null/empty, but prediction has a value. The model hallucinated a field. |
| False Discovery | FD | Both ground truth and prediction have values, but they do not match (similarity < threshold). The model found the field but got the value wrong. |
| False Negative | FN | Ground truth has a value, but prediction is null/empty. The model missed the field entirely. |
| True Negative | TN | Both ground truth and prediction are null/empty. Correctly identified absence. |

False Positive (FP) is computed as the sum of FA and FD:

```
FP = FA + FD
```

The distinction between FA and FD is important for debugging:

- **FA (False Alarm)** points to hallucination problems -- the model is producing values where none should exist.
- **FD (False Discovery)** points to accuracy problems -- the model found the right field but extracted the wrong value.

### Confusion Matrix Structure

The `confusion_matrix` object has four keys:

- **`overall`** -- Object-level metrics for the current hierarchical level.
- **`fields`** -- Field-by-field breakdown, with nested structure for objects and lists.
- **`non_matches`** -- Populated when `document_non_matches=True` (empty otherwise).
- **`aggregate`** -- Sum of all primitive field metrics recursively below this node.

### `overall` vs `aggregate`

These serve different purposes:

- **`overall`**: Counts at the current level only. For a list of 2 line items that both matched, `tp = 2`.
- **`aggregate`**: Sums all primitive fields recursively below. For 2 line items with 3 fields each, all matching: `tp = 6`.

The `aggregate` field exists at every node in the confusion matrix hierarchy, making it easy to get field-level granularity at any depth.

```python
# Total primitive field metrics across the entire comparison
total = cm['aggregate']
print(f"Total TP: {total['tp']}, Total FP: {total['fp']}")

# Metrics for a specific section
contact = cm['fields']['contact']['aggregate']
print(f"Contact section F1: {contact['derived']['cm_f1']:.3f}")
```

---

## Derived Metrics

When the confusion matrix is included, each node automatically contains a `derived` object with four computed metrics:

### Precision

```
Precision = TP / (TP + FP)
```

Of all the values the model predicted, what fraction were correct? High precision means few false alarms and false discoveries.

### Recall

```
Recall = TP / (TP + FN)
```

Of all the values that should have been found, what fraction did the model find correctly? High recall means few missed fields.

Note: When `recall_with_fd=True` is passed to `compare_with()`, the formula changes to `TP / (TP + FN + FD)`, penalizing incorrect values in addition to missing ones.

### F1 Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The harmonic mean of precision and recall. This is typically the single best metric for overall extraction quality.

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Overall correctness, including correct identification of absent fields.

### Accessing Derived Metrics

```python
result = ground_truth.compare_with(prediction, include_confusion_matrix=True)

# Overall derived metrics
overall = result['confusion_matrix']['aggregate']['derived']
print(f"Precision: {overall['cm_precision']:.3f}")
print(f"Recall:    {overall['cm_recall']:.3f}")
print(f"F1:        {overall['cm_f1']:.3f}")
print(f"Accuracy:  {overall['cm_accuracy']:.3f}")

# Field-level derived metrics
for field_name, field_data in result['confusion_matrix']['fields'].items():
    if 'aggregate' in field_data and 'derived' in field_data['aggregate']:
        f1 = field_data['aggregate']['derived']['cm_f1']
        print(f"  {field_name}: F1 = {f1:.3f}")
```

---

## Non-Match Analysis

When you pass `document_non_matches=True`, the result includes a `non_matches` list containing detailed information about every field that failed to match. This is the primary tool for debugging extraction errors.

```python
result = ground_truth.compare_with(prediction, document_non_matches=True)
```

### Non-Match Entry Structure

Each entry in the `non_matches` list contains:

| Field | Type | Description |
|-------|------|-------------|
| `field_path` | string | Dot-notation path to the field (e.g., `"contact.phone"`, `"products[0].name"`). |
| `non_match_type` | string | One of `"false_discovery"`, `"false_alarm"`, or `"false_negative"`. |
| `ground_truth_value` | any | The expected value (null for false alarms). |
| `prediction_value` | any | The predicted value (null for false negatives). |
| `similarity_score` | float | The raw similarity score between the two values. |
| `details` | dict | Additional context, including a `"reason"` string (e.g., `"below threshold (0.300 < 1.0)"`). |

### Non-Match Types

- **`false_discovery`** -- Both values exist but the similarity is below threshold. The most common type; indicates the model found something but got the value wrong.
- **`false_alarm`** -- The prediction has a value but the ground truth is null. Indicates hallucination.
- **`false_negative`** -- The ground truth has a value but the prediction is null. Indicates the model missed the field.

### Debugging with Non-Matches

```python
result = ground_truth.compare_with(prediction, document_non_matches=True)

non_matches = result.get('non_matches', [])

# Group by type
false_discoveries = [nm for nm in non_matches if nm['non_match_type'] == 'false_discovery']
false_alarms = [nm for nm in non_matches if nm['non_match_type'] == 'false_alarm']
false_negatives = [nm for nm in non_matches if nm['non_match_type'] == 'false_negative']

print(f"False Discoveries: {len(false_discoveries)} (wrong values)")
print(f"False Alarms:      {len(false_alarms)} (hallucinated fields)")
print(f"False Negatives:   {len(false_negatives)} (missed fields)")

# Inspect the worst false discoveries
for nm in sorted(false_discoveries, key=lambda x: x['similarity_score']):
    print(f"  {nm['field_path']}: "
          f"expected={nm['ground_truth_value']!r}, "
          f"got={nm['prediction_value']!r}, "
          f"similarity={nm['similarity_score']:.3f}")
```

For list fields (e.g., products, line items), non-match entries can be at the object level. The `ground_truth_value` and `prediction_value` will be dictionaries representing the full object, allowing you to inspect which specific sub-fields caused the mismatch.

---

## Field Comparisons

When you pass `document_field_comparisons=True`, the result includes a `field_comparisons` list documenting every individual field comparison -- both matches and non-matches.

```python
result = ground_truth.compare_with(prediction, document_field_comparisons=True)

for fc in result['field_comparisons']:
    status = "MATCH" if fc['match'] else "MISS"
    print(f"  [{status}] {fc['expected_key']}: {fc['score']:.3f} ({fc['reason']})")
```

Each entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `expected_key` | string | Field path in ground truth. |
| `expected_value` | any | The ground truth value. |
| `actual_key` | string | Field path in prediction (may differ for list items due to Hungarian matching). |
| `actual_value` | any | The predicted value. |
| `match` | bool | Whether the score met the threshold. |
| `score` | float | Raw similarity score. |
| `weighted_score` | float | Score multiplied by the field's weight. |
| `reason` | string | Human-readable explanation. |

This is useful for comprehensive auditing of all comparisons, not just failures.

---

## HTML Reports

Stickler includes an `EvaluationHTMLReporter` that generates interactive HTML reports from evaluation results. The reporter supports both individual comparison results and `ProcessEvaluation` objects from `BulkStructuredModelEvaluator`.

```python
from stickler.reporting.html.html_reporter import EvaluationHTMLReporter

reporter = EvaluationHTMLReporter()
reporter.generate_report(
    evaluation_results=result,
    output_path="report.html",
    title="Invoice Extraction Evaluation",
)
```

The reporter accepts `ProcessEvaluation` objects from bulk evaluation, individual comparison result dictionaries, optional document file mappings for linking source documents, a `model_schema` parameter for extracting field thresholds, and a path to a JSONL file of individual results for per-document drill-down.

---

## Pretty Printing

For quick terminal output, Stickler provides `print_confusion_matrix()`:

```python
from stickler.structured_object_evaluator.utils.pretty_print import print_confusion_matrix

result = ground_truth.compare_with(prediction, include_confusion_matrix=True)
print_confusion_matrix(result, show_details=True)
```

This function works with any result format -- standard `compare_with()` output, evaluator format, or `ProcessEvaluation` from the bulk evaluator. It supports color output, visual progress bars, field filtering with regex patterns, and sorting by name, precision, recall, or F1.

For bulk evaluator results specifically, you can also use:

```python
evaluator.pretty_print_metrics()
```

This displays processing statistics (document count, throughput), overall confusion matrix counts and derived metrics, and field-level performance sorted by F1 score.

---

## Field-Level Aggregate Metrics

Every node in the confusion matrix automatically includes an `aggregate` field that sums all primitive field metrics recursively below that node. This gives you hierarchical analysis without any configuration:

```python
result = ground_truth.compare_with(prediction, include_confusion_matrix=True)
cm = result['confusion_matrix']

# Top-level aggregate: all primitive fields in the entire document
print(f"Total F1: {cm['aggregate']['derived']['cm_f1']:.3f}")

# Section-level aggregate: all primitive fields within a section
for section, data in cm['fields'].items():
    if 'aggregate' in data:
        f1 = data['aggregate']['derived']['cm_f1']
        errors = data['aggregate']['fd'] + data['aggregate']['fa'] + data['aggregate']['fn']
        print(f"  {section}: F1={f1:.3f}, Errors={errors}")
```

This is especially useful for identifying which sections of your data have the most extraction issues, without needing to manually aggregate individual field metrics.
