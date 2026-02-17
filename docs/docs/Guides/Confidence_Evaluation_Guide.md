---
title: Confidence Evaluation Guide
---

# Confidence Evaluation Guide

This guide shows how to use confidence scores in Stickler evaluations to measure prediction calibration quality.

## Basic Usage

### 1. Create Models with Confidence Data

```python
from stickler import StructuredModel, ComparableField

class Product(StructuredModel):
    name: str = ComparableField()
    price: float = ComparableField()
    sku: str = ComparableField()

# Ground truth (standard format)
ground_truth = Product(name="Widget Pro", price=29.99, sku="ABC123")

# Prediction with confidence scores
prediction = Product.from_json({
    "name": {"value": "Widget Pro", "confidence": 0.95},
    "price": {"value": 29.99, "confidence": 0.8},
    "sku": {"value": "XYZ789", "confidence": 0.3}
})
```

### 2. Enable Confidence Metrics

```python
result = ground_truth.compare_with(
    prediction,
    add_confidence_metrics=True,
    document_field_comparisons=True
)

# Access AUROC score
auroc = result['auroc_confidence_metric']
print(f"Confidence calibration AUROC: {auroc:.3f}")
```

## Complete Example

```python
from stickler import StructuredModel, ComparableField
from typing import List

class LineItem(StructuredModel):
    product: str = ComparableField()
    quantity: int = ComparableField()
    price: float = ComparableField()

class Invoice(StructuredModel):
    invoice_id: str = ComparableField()
    customer: str = ComparableField()
    items: List[LineItem] = ComparableField()

# Ground truth
gt_data = {
    "invoice_id": "INV-001",
    "customer": "Acme Corp",
    "items": [
        {"product": "Widget A", "quantity": 2, "price": 50.0},
        {"product": "Widget B", "quantity": 1, "price": 100.0}
    ]
}
ground_truth = Invoice(**gt_data)

# Prediction with mixed confidence
pred_data = {
    "invoice_id": {"value": "INV-001", "confidence": 0.99},
    "customer": {"value": "ACME Corporation", "confidence": 0.85},
    "items": [
        {
            "product": {"value": "Widget A", "confidence": 0.9},
            "quantity": {"value": 2, "confidence": 0.95},
            "price": {"value": 50.0, "confidence": 0.8}
        },
        {
            "product": {"value": "Widget C", "confidence": 0.4},  # Wrong, low confidence
            "quantity": 1,  # No confidence
            "price": {"value": 100.0, "confidence": 0.9}
        }
    ]
}
prediction = Invoice.from_json(pred_data)

# Evaluate with confidence metrics
result = ground_truth.compare_with(
    prediction,
    add_confidence_metrics=True,
    document_field_comparisons=True
)

print(f"Overall Score: {result['overall_score']:.3f}")
print(f"AUROC: {result['auroc_confidence_metric']:.3f}")
```

## Accessing Confidence Scores

```python
# Individual field confidence
invoice_conf = prediction.get_field_confidence("invoice_id")  # 0.99
customer_conf = prediction.get_field_confidence("customer")   # 0.85

# Nested field confidence
item0_product = prediction.get_field_confidence("items[0].product")  # 0.9
item1_price = prediction.get_field_confidence("items[1].price")      # 0.9

# All confidence scores
all_confidences = prediction.get_all_confidences()
for field_path, confidence in all_confidences.items():
    print(f"{field_path}: {confidence}")
```

## AUROC Interpretation

### Well-Calibrated Model (AUROC ≈ 0.8-1.0)
```python
# High confidence → Correct predictions
# Low confidence → Incorrect predictions
prediction_good = Product.from_json({
    "name": {"value": "Widget Pro", "confidence": 0.95},    # Correct, high conf
    "price": {"value": 29.99, "confidence": 0.9},          # Correct, high conf  
    "sku": {"value": "WRONG", "confidence": 0.2}           # Wrong, low conf
})
```

### Poorly-Calibrated Model (AUROC ≈ 0.0-0.3)
```python
# High confidence → Incorrect predictions
# Low confidence → Correct predictions  
prediction_bad = Product.from_json({
    "name": {"value": "Wrong Name", "confidence": 0.95},    # Wrong, high conf
    "price": {"value": 99.99, "confidence": 0.9},          # Wrong, high conf
    "sku": {"value": "ABC123", "confidence": 0.1}          # Correct, low conf
})
```

### Random Calibration (AUROC ≈ 0.5)
Confidence scores don't correlate with accuracy.

## Best Practices

### 1. Enable Required Parameters
```python
result = ground_truth.compare_with(
    prediction,
    add_confidence_metrics=True,        # Required for AUROC
    document_field_comparisons=True     # Required for field-level analysis
)
```

### 2. Handle Edge Cases
```python
# Check if AUROC is available
if 'auroc_confidence_metric' in result:
    auroc = result['auroc_confidence_metric']
    if auroc == 0.5:
        print("No confidence calibration detected")
    elif auroc > 0.7:
        print("Good confidence calibration")
    else:
        print("Poor confidence calibration")
```

### 3. Analyze Field-Level Results
```python
# Examine individual field comparisons
for comparison in result.get('field_comparisons', []):
    field_path = comparison['actual_key']
    is_match = comparison['match']
    confidence = prediction.get_field_confidence(field_path)
    
    if confidence is not None:
        print(f"{field_path}: match={is_match}, confidence={confidence}")
```

## Common Use Cases

### Model Evaluation
Use AUROC to compare different model versions:
```python
models = [model_v1, model_v2, model_v3]
auroc_scores = []

for model in models:
    result = ground_truth.compare_with(model, add_confidence_metrics=True, document_field_comparisons=True)
    auroc_scores.append(result['auroc_confidence_metric'])

best_model_idx = auroc_scores.index(max(auroc_scores))
print(f"Best model: v{best_model_idx + 1} (AUROC: {max(auroc_scores):.3f})")
```

### Confidence Threshold Tuning
Find optimal confidence thresholds for filtering predictions:
```python
confidences = prediction.get_all_confidences()
field_comparisons = result['field_comparisons']

# Analyze confidence vs accuracy relationship
for threshold in [0.5, 0.7, 0.8, 0.9]:
    high_conf_fields = [f for f, c in confidences.items() if c >= threshold]
    # Analyze accuracy of high-confidence fields
