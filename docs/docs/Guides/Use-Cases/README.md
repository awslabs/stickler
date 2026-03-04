---
title: Use Cases
---

# Use Cases

Stickler is designed for any scenario where you need to compare structured JSON outputs against expected results. Whether you are evaluating GenAI extraction pipelines, validating ETL transformations, or monitoring data quality, Stickler provides field-level control over how comparisons are performed and scored.

Below are common patterns with model examples and comparator recommendations for each.

---

## Document Extraction

The primary use case. Extract structured data from documents (invoices, forms, receipts) and evaluate extraction accuracy against ground truth annotations.

**Comparator recommendations:**

- `ExactComparator` for IDs and codes (invoice numbers, PO numbers)
- `NumericComparator` for monetary amounts and quantities
- `LevenshteinComparator` for names, addresses, and short text
- `FuzzyComparator` for descriptions and free-form notes

```python
from typing import List
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, NumericComparator, LevenshteinComparator, FuzzyComparator

class LineItem(StructuredModel):
    description: str = ComparableField(comparator=FuzzyComparator(), weight=1.0)
    quantity: int = ComparableField(comparator=NumericComparator(tolerance=0), weight=1.2)
    unit_price: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=1.5)

class Invoice(StructuredModel):
    invoice_id: str = ComparableField(comparator=ExactComparator(), weight=3.0, threshold=1.0)
    vendor_name: str = ComparableField(comparator=LevenshteinComparator(), weight=1.5, threshold=0.8)
    total_amount: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=2.5)
    line_items: List[LineItem] = ComparableField(weight=2.0)
```

See the [Comparators](../Comparators/README.md) documentation for details on each comparator.

---

## OCR Evaluation

Compare OCR engine output against ground truth transcriptions. `LevenshteinComparator` is well suited here because it measures character-level edit distance, directly reflecting the kinds of errors OCR systems produce (substitutions, insertions, deletions).

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import LevenshteinComparator, NumericComparator

class OCRTextBlock(StructuredModel):
    text: str = ComparableField(
        comparator=LevenshteinComparator(), weight=3.0, threshold=0.7
    )
    x_position: float = ComparableField(
        comparator=NumericComparator(tolerance=5.0), weight=0.5
    )
    y_position: float = ComparableField(
        comparator=NumericComparator(tolerance=5.0), weight=0.5
    )
    page_number: int = ComparableField(weight=1.0, threshold=1.0)
```

Use `BulkStructuredModelEvaluator` when evaluating OCR across many documents. See [Bulk Evaluation](../Evaluation/bulk-evaluation.md) for guidance.

---

## Entity Extraction

Named entity recognition (NER) and entity extraction from text. Compare extracted entities against a labeled ground truth set. The Hungarian algorithm handles reordered entities automatically.

```python
from typing import List
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, LevenshteinComparator, NumericComparator

class Entity(StructuredModel):
    name: str = ComparableField(
        comparator=LevenshteinComparator(), weight=2.0, threshold=0.8
    )
    entity_type: str = ComparableField(
        comparator=ExactComparator(), weight=2.5, threshold=1.0
    )
    confidence: float = ComparableField(
        comparator=NumericComparator(tolerance=0.1), weight=0.3
    )

class ExtractionResult(StructuredModel):
    document_id: str = ComparableField(comparator=ExactComparator(), weight=1.0)
    entities: List[Entity] = ComparableField(weight=3.0)
```

When entity type must be exact but the entity name can tolerate minor variations (e.g., "John Smith" vs. "Jon Smith"), assign a higher weight and stricter threshold to `entity_type` and a more lenient threshold to `name`.

---

## ML Model Evaluation

Compare ML model predictions against ground truth labels. Useful for both regression outputs (use `NumericComparator` with tolerance) and classification outputs (use `ExactComparator` for labels).

```python
from typing import List
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, NumericComparator

class Prediction(StructuredModel):
    sample_id: str = ComparableField(comparator=ExactComparator(), weight=1.0)
    predicted_label: str = ComparableField(
        comparator=ExactComparator(), weight=3.0, threshold=1.0
    )
    predicted_score: float = ComparableField(
        comparator=NumericComparator(tolerance=0.05), weight=1.5
    )

class RegressionOutput(StructuredModel):
    sample_id: str = ComparableField(comparator=ExactComparator(), weight=1.0)
    predicted_value: float = ComparableField(
        comparator=NumericComparator(tolerance=0.1), weight=3.0
    )
```

Enable `include_confusion_matrix=True` when calling `compare_with()` to get precision, recall, and F1 metrics alongside the similarity scores.

---

## ETL Validation

Validate that ETL pipeline outputs match expected results. Stickler ensures data transformations produce the correct structured output by comparing field-by-field with appropriate tolerances.

```python
from typing import List
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, NumericComparator, LevenshteinComparator

class TransformedRecord(StructuredModel):
    record_id: str = ComparableField(
        comparator=ExactComparator(), weight=3.0, threshold=1.0
    )
    category: str = ComparableField(
        comparator=ExactComparator(), weight=2.0, threshold=1.0
    )
    normalized_name: str = ComparableField(
        comparator=LevenshteinComparator(), weight=1.5, threshold=0.9
    )
    computed_total: float = ComparableField(
        comparator=NumericComparator(tolerance=0.001), weight=2.5, threshold=0.99
    )

class ETLBatch(StructuredModel):
    batch_id: str = ComparableField(comparator=ExactComparator(), weight=1.0)
    records: List[TransformedRecord] = ComparableField(weight=3.0)
```

For ETL validation, use tight tolerances and high thresholds. Deterministic transformations should produce near-exact results, so the evaluation criteria should reflect that expectation.

---

## Data Quality Monitoring

Ongoing monitoring of data quality by comparing incoming data against baseline or expected patterns. Run evaluations periodically and track scores over time to catch regressions.

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, NumericComparator, LevenshteinComparator

class CustomerRecord(StructuredModel):
    customer_id: str = ComparableField(
        comparator=ExactComparator(), weight=3.0, threshold=1.0
    )
    full_name: str = ComparableField(
        comparator=LevenshteinComparator(), weight=1.5, threshold=0.85
    )
    account_balance: float = ComparableField(
        comparator=NumericComparator(tolerance=0.01), weight=2.0
    )
    status: str = ComparableField(
        comparator=ExactComparator(), weight=2.0, threshold=1.0
    )
```

Combine with `BulkStructuredModelEvaluator` and `save_metrics()` to produce evaluation reports. Compare metrics across runs to detect data quality drift. See [Bulk Evaluation](../Evaluation/bulk-evaluation.md) for batch processing patterns.

---

## Choosing the Right Pattern

| Use Case | Primary Comparators | Key Consideration |
|---|---|---|
| Document Extraction | Exact, Numeric, Levenshtein, Fuzzy | Weight fields by business impact |
| OCR Evaluation | Levenshtein | Character-level accuracy matters |
| Entity Extraction | Exact, Levenshtein | Entity type must be exact; names can be lenient |
| ML Model Evaluation | Exact, Numeric | Use confusion matrix for classification metrics |
| ETL Validation | Exact, Numeric | Tight tolerances for deterministic pipelines |
| Data Quality Monitoring | Exact, Numeric, Levenshtein | Track scores over time to detect drift |

For all use cases, define your `StructuredModel`, choose comparators based on field semantics, set thresholds and weights based on business impact, and call `compare_with()` or use `BulkStructuredModelEvaluator` for batch processing. See [Best Practices](../Best-Practices/README.md) for guidance on threshold tuning and weight assignment.
