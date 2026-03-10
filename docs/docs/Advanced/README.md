---
title: Advanced Topics
---

# Advanced Topics

This section provides deep dives into Stickler's internal algorithms and advanced features. These pages assume familiarity with the basics of defining a `StructuredModel`, running comparisons, and reading evaluation results.

## Contents

- **[Classification Logic](classification-logic.md)** -- How Stickler categorizes comparison results into True Positives, False Alarms, False Negatives, False Discoveries, and True Negatives, and how derived metrics (precision, recall, F1) are calculated.

- **[Hungarian Matching](hungarian-matching.md)** -- The optimal bipartite matching algorithm used to pair list elements before classification, including worked examples with `List[StructuredModel]`.

- **[Threshold-Gated Evaluation](threshold-gated-evaluation.md)** -- How recursive field-level evaluation is gated by a similarity threshold so that only well-matched object pairs receive detailed analysis.

- **[Dynamic Model Creation](dynamic-models.md)** -- Creating `StructuredModel` classes at runtime from JSON Schema or custom JSON configuration, enabling configuration-driven evaluation without writing Python model code.

- **[Confidence Metrics](confidence-metrics.md)** -- AUROC-based confidence calibration: attaching confidence scores to predictions, measuring how well confidence correlates with accuracy, and interpreting the results.

- **[Aggregate Metrics](aggregate-metrics.md)** -- Automatic hierarchical confusion-matrix aggregation at every node in the comparison result tree.

- **[Model Export](model-export.md)** -- Exporting and importing model schemas in JSON Schema and Stickler-config formats for round-trip serialization, version control, and cross-system interoperability.
