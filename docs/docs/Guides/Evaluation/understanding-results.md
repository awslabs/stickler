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
    cm['aggregate']['fp'] + cm['aggregate']['fn'] == 0
    and cm['overall']['fp'] + cm['overall']['fn'] == 0
)
```

`aggregate` gives leaf detail for the objects that were comparable: `fp` covers a leaf that scored below its threshold and a value invented where the ground truth was null, `fn` a leaf absent from the prediction. `overall` classifies the node's direct children, so on a list field it gives the per-item verdicts. Sum `fp` rather than `fa + fd`, since `FP = FA + FD` by construction and `fp` cannot go stale if a class is ever added.

The second half of that check is what catches an object rejected outright. A **list item** scoring below the element class's `match_threshold` is a spurious non-match, counted once as `fd` on `overall` and not descended into, so it contributes no leaf rows. If you want leaf detail for a marginal list item, lower `match_threshold` until it qualifies as comparable. `field_comparisons` names the individual failures.

A single nested `StructuredModel` field behaves differently and is worth knowing separately: it is never gated, so its leaves appear on `aggregate` even when the object itself is a false discovery, and `match_threshold` is not the knob; the field's own `threshold` is.

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

- **`overall`** -- Metrics for this node's direct children. Where the field is a list, those children are item pairings rather than leaves.
- **`fields`** -- Field-by-field breakdown, with nested structure for objects and lists.
- **`non_matches`** -- Populated when `document_non_matches=True` (empty otherwise).
- **`aggregate`** -- Primitive field metrics summed recursively below this node, excluding list items rejected at their element class's `match_threshold`. A single nested `StructuredModel` field is not excluded; its leaves are always counted. Where a list's items were *all* rejected the counts become object rows rather than primitive-field metrics, so check that before computing a leaf rate ([why](../../Advanced/aggregate-metrics.md#aggregate-counts-objects-for-an-all-rejected-list)). See [`overall` vs `aggregate`](#overall-vs-aggregate).

### `overall` vs `aggregate`

The two nodes are the two stages of the evaluation, and you generally want both:

- **`overall` is detection.** The unit is whatever this node's direct children are. On a list field whose 5 items each paired above `match_threshold`, that field's `overall` reads `tp = 5`. Read it on the list field, not at the root: the root's children are its own fields, so a document with 3 header fields beside that list reads `tp = 8` at the root -- 3 leaves plus 5 pairings, mixing the two units in one number.
- **`aggregate` is extraction.** The unit is the leaf, except on a node that has no leaves left to report because every one of its list items was rejected ([why](../../Advanced/aggregate-metrics.md#aggregate-counts-objects-for-an-all-rejected-list)). Among the objects established to be the same object, how many field values were correct? For those same 5 items with 6 fields each, `tp` counts up to 30.

`match_threshold` is the handoff, and it is really the definition of "the same object". Above it, the pair is the same thing, so grading its fields is meaningful. Below it, it is not the same thing, so grading its fields would be scoring the fields of a *different* object. Such an **item** is classified as a single **false discovery** and is not descended into. This gating is a property of `List[StructuredModel]` pairing, not of nested objects generally: a single nested `StructuredModel` field always reports its leaves, and is judged against the field's own `threshold` rather than `match_threshold`.

If that split is familiar, it should be: it is the structure of mean Average Precision, which Stickler also implements for bounding boxes. An IoU threshold decides whether a detection matched, and only matched pairs are scored further. See [Bounding Box mAP Metrics](../../Advanced/bbox-map-metrics.md#iou-calculation).

The analogy stops at recall. A below-threshold *detection* counts as both a false positive and a false negative, so recall falls. A below-threshold *object* is an `fd` only, so the unmatched ground-truth item leaves no trace and `overall` recall still reads `1.0`. Pass `recall_with_fd=True` to `compare_with()` for the mAP convention:

```
five items, one rejected

recall_with_fd=False   overall tp=4 fd=1 fn=0   R=1.0000
recall_with_fd=True    overall tp=4 fd=1 fn=0   R=0.8000
```

### Read the node whose children you mean

`overall` counts a node's direct children, so which node you read decides what the
number means. On a document with fields beside a list:

```python
class Doc(StructuredModel):
    invoice_id: str = ComparableField(...)
    vendor: str = ComparableField(...)
    date: str = ComparableField(...)
    lines: List[Line] = ComparableField(...)      # Line has 6 leaves
```

Five items, everything correct:

```
cm['overall']                    tp=8     3 header leaves + 5 item pairings
cm['fields']['lines']['overall'] tp=5     the item count
```

The root number is not wrong, it is a different question: it classifies the root's
own fields, and the list field contributes one row per item pairing rather than one
row for the field. But it is not a count of line items, and reading it as one
overstates by the number of header fields. For "how many items did we find", read the
list field's own node.

Two examples make the split concrete. Both drop the header fields and use a model whose **only** field is the list, so `cm['overall']` at the root is the list's own count and the mixing described just above does not apply. Add header fields back and the root numbers below gain one row per header field.

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

#### Getting leaf detail for a marginal list item

If you want those leaves counted, lower `match_threshold` so the **list item** qualifies as comparable. This is the knob for a list item only: a single nested `StructuredModel` field is never gated, so lowering `match_threshold` cannot change what it reports. Same two items as above, with the second still at 4/6:

```
match_threshold   comparable?   overall            aggregate
0.70              no            tp=1 fd=1          tp=6  fd=0
0.66              yes           tp=2 fd=0          tp=10 fd=2
```

At `0.66` the marginal item is comparable, so its six leaves join the first item's in `aggregate`: 12 leaves, of which the two wrong ones appear as `fd`. This is the knob for how much leaf detail you get: `match_threshold` decides what counts as the same object, and leaf reporting follows from that.

#### Which node answers which question

| Question | Node |
|---|---|
| Were the objects comparable, and how many were spurious? | that list field's own `overall`, e.g. `cm['fields']['lines']['overall']` -- not the root's, which also counts the root's own fields |
| Among comparable objects, which leaves landed? | that same field's `aggregate`, e.g. `cm['fields']['lines']['aggregate']` |
| How many list items did the model find? | the list field's `overall` again, `cm['fields']['lines']['overall']` |
| Did anything at all fail? | both `overall` and `aggregate`, on whichever node you are reading, see below |

Because the two nodes scope different things, a complete "did anything fail" check reads both:

```python
clean = (
    cm['aggregate']['fp'] + cm['aggregate']['fn'] == 0
    and cm['overall']['fp'] + cm['overall']['fn'] == 0
)
```

Sum `fp` rather than `fa + fd`. `FP = FA + FD` by construction, so the two are equivalent today, and reading `fp` cannot go stale if a class is ever added.

Both nodes carry `fa`, for different kinds of invention. A field invented where the ground truth has no value at all is `fa` at that leaf and rolls up into `aggregate`, while `overall` stays clean because the item still paired. An item invented wholesale is `fa` on `overall`. Naming only one node here is how a hallucinated value slips through:

```
one item, ground truth total=None, prediction total="INVENTED", five other leaves exact

overall     tp=1  fp=0  fn=0  fa=0  fd=0
aggregate   tp=5  fp=1  fn=0  fa=1  fd=0
```

!!! note "`EvalResult.precision` comes from the root `overall`"

    `EvalResult.precision`, `.recall`, `.f1` and `.accuracy` from `stickler.evaluate()` come from `cm['overall']['derived']` at the **root**, so they inherit the root's unit: they classify the root's direct children. On a model whose only field is the list that is a count of objects. Put three header fields beside it and it is a rate over 3 header leaves plus 5 item pairings, so it is not an object rate at all -- see [Read the node whose children you mean](#read-the-node-whose-children-you-mean). For an object rate, read the list field's own node.

    These are attributes, so they need the entry point that returns an `EvalResult`. Every other example on this page calls `compare_with()`, which returns a plain `dict`, so reaching for `.precision` on one of those raises `AttributeError`. The numbers below are the five-item, one-wrong-leaf document from [A wrong leaf inside a comparable object](#overall-vs-aggregate) above, evaluated through `stickler.evaluate()`:

    ```python
    import stickler

    # `ground_truth` and `prediction` are the five line items of six fields each,
    # with one field of one item wrong -- the first worked example above.
    result = stickler.evaluate(ground_truth, prediction)

    result.precision                                                # 1.0
    result.overall_score                                            # 0.9667
    result.confusion_matrix['aggregate']['derived']['cm_precision']  # 0.9667, per leaf
    ```

    The five items were all comparable, so the root `overall` rate is perfect; the score is a weighted mean over the leaves, and one of the thirty is wrong. Both numbers are right for what they measure.

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

Every node in the confusion matrix automatically includes an `aggregate` field that sums all primitive field metrics recursively below that node. This gives you hierarchical analysis without any configuration.

One caveat before you rank anything by these counts: where a list's items were *all* rejected, that node's `aggregate` reports one row per rejected object rather than its leaves, so counts from such a node are not comparable with leaf counts from another ([why](../../Advanced/aggregate-metrics.md#aggregate-counts-objects-for-an-all-rejected-list)).

**The unit cannot be derived from the confusion matrix.** You have to tell the snippet which fields are `List[StructuredModel]`, because the matrix does not carry that. A list of primitives with one element wrong and a list of objects whose only item was rejected are *identical* in shape -- both have `fields == {}` with non-zero `aggregate` counts -- and yet the first counts element comparisons, which are leaves, and the second counts one row per rejected object:

```
tags   List[str],   one of two elements wrong    fields={}   aggregate tp=1 fd=1   leaves
rows   List[Line],  its only item rejected       fields={}   aggregate tp=0 fd=1   object rows
```

Two earlier versions of this snippet tried to derive it and were wrong in opposite directions. `data['overall']['tp'] == 0` is also true of a primitive field that simply failed, which is one leaf and nothing else. `'fields' in data and not data['fields']` is also true of the primitive list above, and of a scalar that was null on both sides. So the schema has to supply the answer, and `overall['tp'] == 0` then means every item in that list was rejected.

```python
result = ground_truth.compare_with(prediction, include_confusion_matrix=True)
cm = result['confusion_matrix']

# The List[StructuredModel] fields of your model. Only you know these; the
# confusion matrix cannot tell them from a list of primitives.
OBJECT_LISTS = {'lines'}

# Top-level aggregate: all primitive fields in the entire document
print(f"Total F1: {cm['aggregate']['derived']['cm_f1']:.3f}")

# Section-level aggregate: all primitive fields within a section
for section, data in cm['fields'].items():
    if 'aggregate' in data:
        f1 = data['aggregate']['derived']['cm_f1']
        errors = data['aggregate']['fp'] + data['aggregate']['fn']
        # An object list whose items were ALL rejected reports its own rows here
        # rather than leaves, so say which unit the numbers are in instead of
        # ranking the two against each other.
        counts_objects = section in OBJECT_LISTS and data['overall']['tp'] == 0
        unit = 'object rows' if counts_objects else 'leaves'
        print(f"  {section}: F1={f1:.3f}, Errors={errors} ({unit})")
```

This is especially useful for identifying which sections of your data have the most extraction issues, without needing to manually aggregate individual field metrics. Sum `fp` rather than `fa + fd`, for the reason given under [Asking whether anything failed](#asking-whether-anything-failed): the two are equal by construction and `fp` cannot go stale if a class is ever added.
