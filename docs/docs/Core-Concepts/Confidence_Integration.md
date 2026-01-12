---
title: Confidence Integration
---

# Confidence Integration

Stickler supports confidence scores alongside field values to evaluate how well prediction confidence correlates with actual accuracy.

## JSON Structure

### Standard Format
```json
{
  "name": "Widget",
  "price": 29.99
}
```

### Confidence Format
```json
{
  "name": {"value": "Widget", "confidence": 0.95},
  "price": {"value": 29.99, "confidence": 0.8}
}
```

### Mixed Format
```json
{
  "name": {"value": "Widget", "confidence": 0.95},
  "price": 29.99,
  "sku": {"value": "ABC123", "confidence": 0.7}
}
```

## Confidence Structure Rules

A valid confidence structure must have exactly two keys:
- `"value"`: The actual field value
- `"confidence"`: Float between 0.0 and 1.0

## Nested Structures

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

## AUROC Calculation

**AUROC (Area Under ROC Curve)** measures confidence calibration quality:

- **1.0**: Perfect calibration (high confidence = correct, low confidence = incorrect)
- **0.5**: Random calibration (confidence doesn't correlate with accuracy)
- **0.0**: Inverse calibration (high confidence = incorrect, low confidence = correct)

### How It Works

1. For each field with confidence, determine if it matches ground truth
2. Create binary classification: match (1) or no-match (0)
3. Use confidence scores as prediction probabilities
4. Calculate ROC curve and area under it

### Requirements

- At least one field with confidence scores
- Both matches and non-matches in the comparison
- `document_field_comparisons=True` must be enabled

## Confidence Access API

```python
# Get individual field confidence
confidence = model.get_field_confidence("name")  # Returns float or None

# Get all confidences
all_confidences = model.get_all_confidences()  # Returns dict

# Nested field access
street_conf = model.get_field_confidence("address.street")

# Array field access  
item_conf = model.get_field_confidence("items[0].product")
```

## Field Path Format

Confidence paths use dot notation for nested fields and bracket notation for arrays:

- Simple field: `"name"`
- Nested field: `"address.street"`
- Array element: `"items[0].product"`
- Nested in array: `"orders[1].customer.name"`
