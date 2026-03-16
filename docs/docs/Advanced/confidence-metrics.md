---
title: Confidence Metrics
---

# Confidence Metrics

Stickler supports confidence scores alongside predicted field values. When a model reports how confident it is in each extraction, Stickler measures whether that confidence actually correlates with accuracy using AUROC (Area Under the ROC Curve).

## Why Confidence Matters

Confidence scores enable downstream systems to:

- Flag uncertain extractions for human review
- Prioritize high-confidence predictions for automated processing
- Provide transparency about prediction reliability
- Apply adaptive thresholds based on use-case requirements

## JSON Structure

### Standard Format (no confidence)

```json
{
  "name": "Widget",
  "price": 29.99
}
```

### Confidence Format

Each field wraps its value in an object with exactly two keys -- `"value"` and `"confidence"`:

```json
{
  "name": {"value": "Widget", "confidence": 0.95},
  "price": {"value": 29.99, "confidence": 0.8}
}
```

### Mixed Format

Fields with and without confidence can coexist:

```json
{
  "name": {"value": "Widget", "confidence": 0.95},
  "price": 29.99,
  "sku": {"value": "ABC123", "confidence": 0.7}
}
```

### Nested Structures

Confidence works with nested objects and arrays:

```json
{
  "customer": {
    "name": {"value": "John Doe", "confidence": 0.92},
    "address": {
      "street": {"value": "123 Main St", "confidence": 0.85},
      "city": "New York"
    }
  },
  "items": [
    {
      "product": {"value": "Laptop", "confidence": 0.89},
      "price": {"value": 1299.99, "confidence": 0.76}
    }
  ]
}
```

## Usage

### Creating Models with Confidence Data

```python
from stickler import StructuredModel, ComparableField

class Product(StructuredModel):
    name: str = ComparableField()
    price: float = ComparableField()
    sku: str = ComparableField()

ground_truth = Product(name="Widget Pro", price=29.99, sku="ABC123")

prediction = Product.from_json({
    "name": {"value": "Widget Pro", "confidence": 0.95},
    "price": {"value": 29.99, "confidence": 0.8},
    "sku": {"value": "XYZ789", "confidence": 0.3}
})
```

### Enabling Confidence Metrics

```python
result = ground_truth.compare_with(
    prediction,
    add_confidence_metrics=True,
    document_field_comparisons=True
)

auroc = result['auroc_confidence_metric']
print(f"Confidence calibration AUROC: {auroc:.3f}")
```

Both `add_confidence_metrics=True` and `document_field_comparisons=True` are required.

### Accessing Confidence Scores

```python
# Individual field
prediction.get_field_confidence("name")        # 0.95

# Nested field (dot notation)
prediction.get_field_confidence("address.street")

# Array element (bracket notation)
prediction.get_field_confidence("items[0].product")

# All confidence scores
all_conf = prediction.get_all_confidences()    # dict of path -> float
```

## AUROC Calculation

AUROC treats confidence evaluation as a binary classification problem:

- **Positive class**: fields where prediction matches ground truth
- **Negative class**: fields where prediction does not match
- **Score**: the model's confidence value

AUROC measures the probability that a randomly chosen correct prediction has higher confidence than a randomly chosen incorrect prediction.

| AUROC Range | Interpretation |
|-------------|---------------|
| 0.8 -- 1.0 | Well calibrated. High confidence correlates with correctness. |
| ~0.5 | Random. Confidence provides no signal. |
| 0.0 -- 0.3 | Inversely calibrated. High confidence correlates with errors. |

### Well-Calibrated Example

```python
# Correct predictions get high confidence; wrong predictions get low confidence
prediction = Product.from_json({
    "name": {"value": "Widget Pro", "confidence": 0.95},   # correct, high
    "price": {"value": 29.99, "confidence": 0.9},          # correct, high
    "sku": {"value": "WRONG", "confidence": 0.2}           # wrong, low
})
```

### Poorly-Calibrated Example

```python
# Wrong predictions get high confidence
prediction = Product.from_json({
    "name": {"value": "Wrong Name", "confidence": 0.95},   # wrong, high
    "price": {"value": 99.99, "confidence": 0.9},          # wrong, high
    "sku": {"value": "ABC123", "confidence": 0.1}          # correct, low
})
```

## Handling Edge Cases

```python
if 'auroc_confidence_metric' in result:
    auroc = result['auroc_confidence_metric']
    if auroc > 0.7:
        print("Good confidence calibration")
    elif auroc > 0.4:
        print("Weak calibration -- consider post-processing")
    else:
        print("Poor calibration -- confidence scores may be misleading")
```

AUROC is undefined when all predictions are correct (no negative class) or all are incorrect (no positive class). In those cases the key may be absent from the result.

## Comparing Model Versions

```python
models = [model_v1, model_v2, model_v3]

for i, pred in enumerate(models):
    result = ground_truth.compare_with(
        pred, add_confidence_metrics=True, document_field_comparisons=True
    )
    print(f"v{i+1} AUROC: {result['auroc_confidence_metric']:.3f}")
```

## See Also

- [Classification Logic](classification-logic.md) -- how match/no-match is determined for each field
- [Aggregate Metrics](aggregate-metrics.md) -- hierarchical metric rollup
