---
title: Best Practices
---

# Best Practices

This section covers practical guidance for getting the most out of Stickler: tuning thresholds, assigning weights, optimizing performance, debugging evaluation results, and designing effective models.

---

## Threshold Tuning

The threshold determines the minimum similarity score required for a field comparison to be classified as a match. Choosing the right threshold depends on the field's role in your business process.

**Start with defaults, then adjust based on your data.** Run an initial evaluation with default thresholds, inspect the field-level scores, and tune from there.

| Field Category | Threshold Range | Examples |
|---|---|---|
| Critical identifiers | 0.95 -- 1.0 | Invoice IDs, PO numbers, account codes |
| Important operational fields | 0.8 -- 0.9 | Names, dates, addresses, amounts |
| Flexible text fields | 0.5 -- 0.7 | Descriptions, notes, comments, metadata |

**Guidelines:**

- **IDs and codes** should use `threshold=1.0` with `ExactComparator`. A partial match on an ID is effectively a mismatch.
- **Monetary amounts** benefit from `threshold=0.95` with `NumericComparator`. Small rounding differences are acceptable; large discrepancies are not.
- **Names and addresses** work well at `threshold=0.8` with `LevenshteinComparator`. This tolerates minor typos while catching significant errors.
- **Free-form text** can use `threshold=0.5-0.7` with `FuzzyComparator`. Variations in phrasing are expected and acceptable.

Use `clip_under_threshold=True` for binary pass/fail fields where partial similarity scores are meaningless:

```python
class Order(StructuredModel):
    order_id: str = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        clip_under_threshold=True,  # Score is 1.0 or 0.0, nothing in between
        weight=3.0,
    )
```

---

## Weight Assignment

Weights reflect **business importance**, not data complexity. A simple string field that drives logistics routing should be weighted higher than a complex nested object that serves as supplementary metadata.

| Priority | Weight Range | Use When |
|---|---|---|
| Critical | 2.5 -- 3.0 | Errors cause operational failures (IDs, amounts, status codes) |
| High | 1.5 -- 2.0 | Errors require manual correction (names, dates, addresses) |
| Normal | 1.0 | Standard fields with moderate impact |
| Low | 0.5 -- 0.8 | Nice-to-have fields, secondary information |
| Minimal | 0.1 -- 0.3 | Metadata, debug info, internal notes |

**How the weighted average works:**

```
overall_score = sum(field_score * weight) / sum(weight)
```

A field with `weight=3.0` has three times the influence on the overall score compared to a field with `weight=1.0`. This means a perfect score on a critical field compensates for imperfections in lower-priority fields.

**Example: Invoice with business-aligned weights**

```python
class Invoice(StructuredModel):
    invoice_id: str = ComparableField(weight=3.0)    # Wrong ID = wrong customer
    total_amount: float = ComparableField(weight=2.5) # Wrong amount = billing error
    vendor_name: str = ComparableField(weight=1.5)    # Typo = manual lookup needed
    line_items: List[LineItem] = ComparableField(weight=1.0)  # Detail verification
    internal_notes: str = ComparableField(weight=0.2) # No operational impact
```

---

## Performance Optimization

### Comparator speed ranking

Comparators vary significantly in computational cost. Choose the simplest comparator that meets your accuracy needs.

| Comparator | Relative Speed | When to Use |
|---|---|---|
| `ExactComparator` | Fastest | IDs, codes, booleans, enums |
| `NumericComparator` | Fast | Prices, quantities, measurements |
| `LevenshteinComparator` | Fast | Names, addresses, short text |
| `FuzzyComparator` | Moderate | Descriptions, token-order-independent text |
| `BERTComparator` | Slow | Deep semantic similarity |
| `SemanticComparator` | Slow | Embedding-based semantic comparison |
| `LLMComparator` | Slowest | Complex semantic evaluation with reasoning |

### Use bulk evaluation for large datasets

`BulkStructuredModelEvaluator` is designed for evaluating hundreds or thousands of document pairs. It avoids repeated model creation overhead and provides streaming progress metrics.

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import BulkStructuredModelEvaluator

evaluator = BulkStructuredModelEvaluator(target_schema=MyModel, verbose=True)

for gt, pred, doc_id in dataset:
    evaluator.update(gt, pred, doc_id)

result = evaluator.compute()
```

For very large datasets, use `update_batch()` to process groups of document pairs at once.

### General tips

- Avoid `LLMComparator` and `SemanticComparator` unless you genuinely need semantic understanding. `LevenshteinComparator` and `FuzzyComparator` handle most text comparison needs at a fraction of the cost.
- Use `BERTComparator` or `SemanticComparator` only for fields where meaning matters more than surface-level similarity.
- Profile your evaluation pipeline if performance is a concern. Field-level timing can reveal which comparators dominate execution time.

---

## Debugging Tips

### Enable detailed output flags

Stickler provides several flags to expose what is happening during evaluation:

- **`document_non_matches=True`** -- Shows which objects in a list could not be matched. Essential for diagnosing why list-level scores are low.
- **`document_field_comparisons=True`** -- Shows individual field comparison results for each pair. Reveals which specific fields are dragging scores down.
- **`include_confusion_matrix=True`** -- Produces TP/FP/FN/TN counts for classification analysis.

```python
result = ground_truth.compare_with(
    prediction,
    document_non_matches=True,
    document_field_comparisons=True,
    include_confusion_matrix=True,
)
```

### Debugging workflow

1. **Check overall score first.** If it is unexpectedly low, proceed to field-level analysis.
2. **Inspect `field_scores`** to find which fields have low scores.
3. **Enable `document_field_comparisons`** to see the raw similarity values for problem fields.
4. **For list fields**, enable `document_non_matches` to see which items were not paired.
5. **Check comparator and threshold** for the problem field. A common mistake is using `ExactComparator` where `LevenshteinComparator` would be appropriate, or setting a threshold too high for the natural variation in your data.

### Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Wrong comparator for field type | Unexpectedly low scores on text fields | Use `LevenshteinComparator` or `FuzzyComparator` instead of `ExactComparator` |
| Threshold too high | Many fields classified as non-matches | Lower the threshold to match acceptable variation |
| Threshold too low | Poor matches classified as matches | Raise the threshold to enforce stricter matching |
| Missing weights on critical fields | Overall score does not reflect business impact | Add weights proportional to field importance |
| Not using Hungarian matching | List items compared positionally instead of optimally | Use `List[StructuredModel]` type annotation for list fields |

---

## Model Design Patterns

### Keep models flat when possible

Flat models are easier to debug and reason about. Every field appears directly in the evaluation output.

```python
class FlatInvoice(StructuredModel):
    invoice_id: str = ComparableField(comparator=ExactComparator(), weight=3.0)
    vendor_name: str = ComparableField(comparator=LevenshteinComparator(), weight=1.5)
    vendor_address: str = ComparableField(comparator=LevenshteinComparator(), weight=1.0)
    total_amount: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=2.5)
```

### Use nested models for naturally hierarchical data

When data is genuinely hierarchical (e.g., an invoice with line items, an order with products), nested models with list fields use Hungarian matching to find optimal pairings.

```python
class LineItem(StructuredModel):
    match_threshold = 0.7  # Minimum similarity for Hungarian pairing

    description: str = ComparableField(comparator=FuzzyComparator(), weight=1.0)
    amount: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=1.5)

class Invoice(StructuredModel):
    invoice_id: str = ComparableField(comparator=ExactComparator(), weight=3.0)
    line_items: List[LineItem] = ComparableField(weight=2.0)
```

### Use JSON Schema for configuration-driven evaluation

When your evaluation criteria need to change without code modifications -- for example, across document types or during A/B testing of field configurations -- use JSON Schema with `x-aws-stickler-*` extensions.

```python
import json
from stickler import StructuredModel

with open("invoice_schema.json") as f:
    schema = json.load(f)

InvoiceModel = StructuredModel.from_json_schema(schema)

gt = InvoiceModel(**ground_truth_data)
pred = InvoiceModel(**prediction_data)
result = gt.compare_with(pred)
```

This pattern is particularly useful for:

- Supporting multiple document types with different evaluation criteria
- Allowing non-developers to adjust evaluation parameters
- Runtime configuration changes without redeployment
- Automated pipelines where schemas are generated programmatically

See the [Evaluation](../Evaluation/README.md) documentation for the complete extension reference.

