---
title: Confidence Integration
---

# Confidence Integration

Stickler supports confidence scores alongside field values to evaluate how well prediction confidence correlates with actual accuracy.

## AUROC for Confidence Calibration: Theoretical Foundation

### What is Confidence Estimation?

Confidence estimation in machine learning refers to a model's ability to assess the reliability of its own predictions. Rather than simply outputting a prediction, a well-calibrated model also provides a confidence score indicating how likely that prediction is to be correct. This is crucial for production systems where understanding prediction uncertainty enables better decision-making, error handling, and human-AI collaboration.

In structured data extraction tasks, confidence estimation allows systems to:
- Flag uncertain extractions for human review
- Prioritize high-confidence predictions for automated processing  
- Provide transparency about model reliability to end users
- Enable adaptive thresholding based on use case requirements

### Research Foundation and Motivation

The choice of AUROC for confidence evaluation is grounded in research on model calibration and confidence assessment. Other approaches to evaluating confidence, such as Expected Calibration Error (ECE), have significant limitations that motivated our selection of AUROC.

#### Limitations of Expected Calibration Error (ECE)

As noted in recent research (Jiang et al., 2025), Expected Calibration Error has a fundamental flaw:

> "Expected calibration error (ECE) (Guo et al., 2017a; Naeini et al., 2015) is also a standard metric to measure how closely a model's confidence matches its accuracy. However, ECE does not assess a model's ability to discriminate between correct and incorrect answers—a model with accuracy 0.5 can achieve perfect ECE by outputting a confidence of 0.5 for all of its answers. Therefore, we focus our results on the AUC."

This limitation means ECE can be "gamed" by models that output uniform confidence scores, making it unsuitable for evaluating whether confidence scores actually help distinguish between correct and incorrect predictions.

#### Why AUROC for Confidence Evaluation?

AUROC addresses the core question: **"Do higher confidence scores correspond to more accurate predictions?"** This discrimination ability is essential for practical applications where confidence scores guide decision-making.

**Mathematical Definition:**
AUROC measures the probability that a randomly chosen correct prediction has higher confidence than a randomly chosen incorrect prediction. It treats confidence estimation as a binary classification problem where:
- **Positive class**: Correct predictions (field matches ground truth)
- **Negative class**: Incorrect predictions (field doesn't match ground truth)  
- **Classifier score**: Model's confidence score

**Advantages over alternatives:**
1. **Discrimination focus**: Directly measures whether confidence correlates with accuracy
2. **Threshold independence**: Evaluates performance across all possible confidence thresholds
3. **Intuitive interpretation**: Values closer to 1.0 indicate better calibration
4. **Robust to class imbalance**: Works well even when correct/incorrect predictions are unbalanced

### AUROC Calculation in Stickler

**AUROC (Area Under ROC Curve)** measures confidence calibration quality:

- **1.0**: Perfect calibration (high confidence = correct, low confidence = incorrect)
- **0.5**: Random calibration (confidence doesn't correlate with accuracy)  
- **0.0**: Inverse calibration (high confidence = incorrect, low confidence = correct)

#### Implementation Details

1. For each field with confidence, determine if it matches ground truth
2. Create binary classification: match (1) or no-match (0)
3. Use confidence scores as prediction probabilities
4. Calculate ROC curve and area under it

#### Requirements

- At least one field with confidence scores
- Both matches and non-matches in the comparison
- `document_field_comparisons=True` must be enabled

### Interpretation Guidelines

**Well-Calibrated Models (AUROC ≈ 0.8-1.0):**
- High confidence predictions are predominantly correct
- Low confidence predictions are predominantly incorrect
- Confidence scores provide meaningful signal for decision-making

**Poorly-Calibrated Models (AUROC ≈ 0.0-0.3):**
- High confidence predictions are predominantly incorrect
- May indicate systematic overconfidence or model issues
- Confidence scores mislead rather than inform

**Random Calibration (AUROC ≈ 0.5):**
- Confidence scores don't correlate with accuracy
- Model may need confidence calibration techniques
- Consider post-processing methods like Platt scaling

### Limitations and Considerations

**Edge Cases:**
- **All predictions correct**: AUROC undefined (no negative class)
- **All predictions incorrect**: AUROC undefined (no positive class)
- **Uniform confidence**: AUROC = 0.5 regardless of accuracy

**Sample Size Requirements:**
- Reliable AUROC calculation requires sufficient examples of both correct and incorrect predictions
- Very small datasets may produce unstable AUROC estimates

**Interpretation Context:**
- AUROC should be interpreted alongside overall accuracy
- High AUROC with low accuracy may indicate good uncertainty awareness
- Low AUROC with high accuracy suggests overconfident correct predictions

### References and Further Reading

**Key Papers:**
- Jiang, Z., et al. (2025). "Confidence Estimation in Large Language Models." [arXiv:2502.01126](https://arxiv.org/pdf/2502.01126)
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML 2017
- Naeini, M. P., et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI 2015

**Additional Resources:**
- Scikit-learn ROC AUC documentation: [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

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
