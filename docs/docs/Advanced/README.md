---
title: Advanced Topics
---

# Advanced Topics

Reference for users who already compare models. Each page answers one question about how to configure, interpret, or extend an evaluation.

## Contents

- **[Rich Value Pattern](rich-value-pattern.md)** — How to attach confidence scores, bounding boxes, and other metadata to field values while preserving normal model validation and comparison.

- **[Classification Logic](classification-logic.md)** — How Stickler classifies comparison outcomes as true positives, false alarms, false negatives, false discoveries, and true negatives, and derives evaluation metrics from those counts.

- **[Hungarian Matching](hungarian-matching.md)** — How Stickler pairs list elements before classifying their comparison results.

- **[How Below-Threshold Pairs Are Classified](threshold-gated-evaluation.md)** — What happens when a matched pair falls below its similarity threshold and why it may not receive field-by-field breakdowns.

- **[Dynamic Model Creation](dynamic-models.md)** — How to create `StructuredModel` classes at runtime from JSON Schema or Stickler configuration.

- **[Confidence Metrics](confidence-metrics.md)** — How to measure whether prediction confidence scores correlate with actual accuracy using metrics such as AUROC.

- **[Bounding Box mAP Metrics](bbox-map-metrics.md)** — How to evaluate bounding-box localization using mean Average Precision and configurable IoU thresholds.

- **[Map/Reduce Evaluation](map-reduce-evaluation.md)** — How to compare documents individually, store their results, and combine them into bulk evaluation metrics.

- **[Aggregate Metrics](aggregate-metrics.md)** — How to understand hierarchical confusion-matrix counts. See [Understanding Results](../Guides/Evaluation/understanding-results.md) for the canonical explanation.

- **[Model Export](model-export.md)** — How to export and import model schemas using JSON Schema and Stickler configuration formats.
