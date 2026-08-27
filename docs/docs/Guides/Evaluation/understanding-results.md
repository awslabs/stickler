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
  "overall_score": 0.92
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

### Asking whether anything failed

`overall_score` is the scalar summary, and for a single object-level verdict `stickler.evaluate()` returns an `EvalResult` whose `matched` attribute is `overall_score >= match_threshold`.

To ask whether anything at all went wrong, read **both** rollup nodes, because they scope different things:

```python
cm = result['confusion_matrix']
clean = (
    cm['aggregate']['fd'] + cm['aggregate']['fn'] == 0
    and cm['overall']['fd'] + cm['overall']['fn'] + cm['overall']['fa'] == 0
)
```

`aggregate` gives leaf detail for the objects that were comparable: `fd` is a leaf that scored below its threshold, `fn` a leaf absent from the prediction. `overall` gives the object verdicts, and it is the only node carrying `fa`, the fields the prediction invented, since those correspond to no ground truth leaf.

The second half of that check is what catches an object rejected outright. An object scoring below `match_threshold` is a spurious non-match, counted once as `fd` on `overall` and not descended into, so it contributes no leaf rows. If you want leaf detail for a marginal object, lower `match_threshold` until it qualifies as comparable. `field_comparisons` names the individual failures.

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

- **`overall`** -- Object-level metrics for the current hierarchical level. Counts item pairings, not leaves.
- **`fields`** -- Field-by-field breakdown, with nested structure for objects and lists.
- **`non_matches`** -- Populated when `document_non_matches=True` (empty otherwise).
- **`aggregate`** -- Primitive field metrics summed recursively below this node, for the subtrees the traversal reached. See the caveat below.

### `overall` vs `aggregate`

The two nodes answer two different questions, and you generally want both:

- **`overall`**: were these objects comparable at all? For a list of 5 line items that each paired above `match_threshold`, `tp = 5`.
- **`aggregate`**: among the objects that *were* comparable, how many individual leaves landed? For those same 5 items with 6 fields each, `tp` counts up to 30.

`match_threshold` is the line between them. An object scoring below it is classified as a single **false discovery**: a spurious non-match, counted once at the item level and not descended into. Leaf detail is not reported for it, because enumerating the parts of an object you have already rejected as a whole would be counting a thing you declared not comparable.

Two examples make the split concrete.

**A wrong leaf inside a comparable object.** Five line items of six fields each, one field of one item wrong. That item scores 5/6, clears the 0.7 threshold, and is comparable:

```
overall_score   0.9667

cm['overall']     tp=5   fd=0   P=1.0000  R=1.0000  F1=1.0000
cm['aggregate']   tp=29  fd=1   P=0.9667  R=1.0000  F1=0.9831
```

`overall` reports five comparable objects and no spurious matches, which is exactly what it measures. The wrong field is one of 30 leaves, and `aggregate` is where you see it.

**An object that was not comparable.** Two items of six fields, two fields of one item wrong. That item scores 4/6, below the threshold:

```
cm['overall']     tp=1   fd=1
cm['aggregate']   tp=6   fd=0
```

`overall` records the spurious non-match. `aggregate` reports the six leaves of the one comparable object, and none of them failed. The rejected object contributes no leaf rows.

#### Getting leaf detail for a marginal object

If you want those leaves counted, lower `match_threshold` so the object qualifies as comparable. Same data as above:

```
match_threshold   comparable?   overall            aggregate
0.70              no            tp=0 fd=1          tp=0 fd=1
0.66              yes           tp=1 fd=0          tp=4 fd=2
```

At `0.66` the object is comparable, so its six leaves are scored individually and the two bad ones appear as `fd`. This is the knob for how much leaf detail you get: `match_threshold` decides what counts as the same object, and leaf reporting follows from that.

#### Which node answers which question

| Question | Node |
|---|---|
| Were the objects comparable, and how many were spurious? | `overall` |
| Among comparable objects, which leaves landed? | `aggregate` |
| How many list items did the model find? | `overall` |
| Did anything at all fail? | both, see below |

Because the two nodes scope different things, a complete "did anything fail" check reads both:

```python
clean = (
    cm['aggregate']['fd'] + cm['aggregate']['fn'] == 0
    and cm['overall']['fd'] + cm['overall']['fn'] + cm['overall']['fa'] == 0
)
```

`overall` is the only node carrying `fa`, the fields the prediction invented, since those correspond to no ground truth leaf.

!!! note "`EvalResult.precision` is the object-level metric"

    `EvalResult.precision`, `.recall`, `.f1` and `.accuracy` from `stickler.evaluate()` come from `cm['overall']['derived']`, so they answer "how many objects were comparable rather than spurious". On the first example above `result.precision` is `1.0` while `result.overall_score` is `0.9667`: the five items were all comparable, and the score is a weighted mean over the leaves. Both numbers are right for what they measure. For leaf-level precision, read `result.confusion_matrix['aggregate']['derived']`.

    That two similar-looking numbers on one object mean different things is tracked in [#288](https://github.com/awslabs/stickler/issues/288), where the naming is under review for 1.0.

`aggregate` exists at every node, so the same access pattern gives field-level granularity at any depth:

```python
# Leaf-level counts across the entire comparison
total = cm['aggregate']
print(f"Total TP: {total['tp']}, Total FP: {total['fp']}")

# The same question, scoped to one section
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

Under the default (`recall_with_fd=False`), be aware that a pair *becoming* an FD moves recall in opposite directions depending on what it was before: a TP that becomes an FD lowers recall, while an FN that becomes an FD **raises** it. See [Two moves that push recall in opposite directions](../../Getting-Started/thresholds-and-metrics.md#two-moves-that-push-recall-in-opposite-directions) before comparing recall across releases or threshold changes.

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

## Dataset-Level Weighted Aggregate

The per-document `overall_score` above is a weighted average of field scores. When you aggregate across many documents with `BulkStructuredModelEvaluator.compute()`, the weight information is preserved via two keys:

- **`metrics["weighted_overall_score"]`** (float) -- Arithmetic mean of each document's `overall_score`. Prefer this over `cm_f1` whenever your schema uses non-uniform `ComparableField(weight=...)` values: `cm_f1` treats every field-match equally, while `weighted_overall_score` preserves the declared per-field weights across the dataset.
- **`field_metrics[path]["mean_score"]`** (float, when present) -- Arithmetic mean of each document's `threshold_applied_score` at that path, reported at every nested node that was actually scored. Paths with confusion-matrix counts but no score data (e.g., leaves inside `List[StructuredModel]`, where `compare_with()` only emits the score at the list parent) are surfaced without a `mean_score` key rather than as `0.0`.

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

evaluator = BulkStructuredModelEvaluator(target_schema=Invoice)
for gt, pred in dataset:
    evaluator.update(gt, pred)
result = evaluator.compute()

print(f"Weighted Score: {result.metrics['weighted_overall_score']:.3f}")
print(f"Aggregate F1:   {result.metrics['cm_f1']:.3f}")

for field_path, fm in result.field_metrics.items():
    mean = fm.get("mean_score")
    mean_str = f"{mean:.3f}" if mean is not None else "n/a"
    print(f"  {field_path}: mean={mean_str} | f1={fm.get('cm_f1', 0):.3f}")
```

Documents whose `overall_score` is missing or non-finite are excluded from the `weighted_overall_score` denominator (error docs are excluded from every aggregate). With zero eligible documents the score is `0.0`; disambiguate via `document_count` when that matters. See [Bulk Evaluation → Weighted Overall Score](bulk-evaluation.md#weighted-overall-score) for the full semantics.

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
