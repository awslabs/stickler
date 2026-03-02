---
title: IDP Accelerator Integration
---

# IDP Accelerator Integration

The [AWS Intelligent Document Processing (IDP) Accelerator](https://github.com/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws) is an open-source AWS solution for building document processing pipelines. It provides a framework for extraction, classification, and enrichment of documents using AWS AI services.

Stickler serves as the **evaluation stage** in an IDP pipeline, measuring how accurately the pipeline extracts structured data from documents.

## How Stickler Fits

After documents are processed and data is extracted by the IDP Accelerator, Stickler evaluates the quality of that extraction:

1. **Compare extracted data against ground truth** -- Measure field-level accuracy for every document processed by the pipeline.
2. **Track accuracy across document types and fields** -- Identify which document types or specific fields have lower extraction quality.
3. **Use bulk evaluation for document batches** -- Process entire batches of documents with `BulkStructuredModelEvaluator` for aggregate metrics.
4. **Tune the pipeline based on evaluation results** -- Use field-level scores and confusion matrices to target improvements.

## Configuration for IDP

### Define models matching your extraction schema

Create `StructuredModel` classes that mirror the fields your IDP pipeline extracts. For example, if your pipeline extracts invoices:

```python
from typing import List
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, NumericComparator, LevenshteinComparator, FuzzyComparator

class IDPLineItem(StructuredModel):
    description: str = ComparableField(comparator=FuzzyComparator(), weight=1.0)
    quantity: int = ComparableField(comparator=NumericComparator(tolerance=0), weight=1.2)
    amount: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=1.5)

class IDPInvoice(StructuredModel):
    invoice_number: str = ComparableField(comparator=ExactComparator(), weight=3.0, threshold=1.0)
    vendor_name: str = ComparableField(comparator=LevenshteinComparator(), weight=1.5, threshold=0.8)
    invoice_date: str = ComparableField(comparator=ExactComparator(), weight=2.0, threshold=1.0)
    total_amount: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=2.5)
    line_items: List[IDPLineItem] = ComparableField(weight=2.0)
```

### Use JSON Schema for config-driven evaluation

For IDP pipelines where the extraction schema may change across document types, use JSON Schema with `x-aws-stickler-*` extensions. This lets you configure evaluation without modifying Python code:

```json
{
  "type": "object",
  "x-aws-stickler-model-name": "IDPInvoice",
  "properties": {
    "invoice_number": {
      "type": "string",
      "x-aws-stickler-comparator": "ExactComparator",
      "x-aws-stickler-threshold": 1.0,
      "x-aws-stickler-weight": 3.0
    },
    "total_amount": {
      "type": "number",
      "x-aws-stickler-comparator": "NumericComparator",
      "x-aws-stickler-threshold": 0.95,
      "x-aws-stickler-weight": 2.5
    }
  }
}
```

See the [Evaluation](../Evaluation/README.md) documentation for the full extension reference.

### Use bulk evaluation for batch processing

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import BulkStructuredModelEvaluator

evaluator = BulkStructuredModelEvaluator(
    target_schema=IDPInvoice,
    verbose=True,
    document_non_matches=True,
)

# Process each document pair from your IDP pipeline
for doc_id, ground_truth_data, extracted_data in idp_results:
    gt = IDPInvoice(**ground_truth_data)
    pred = IDPInvoice(**extracted_data)
    evaluator.update(gt, pred, doc_id)

result = evaluator.compute()
evaluator.save_metrics("idp_evaluation_metrics.json")
evaluator.pretty_print_metrics()
```

## Comparator Selection for Document Fields

| Field Type | Comparator | Rationale |
|---|---|---|
| Invoice/PO numbers | `ExactComparator` | Must be exact for routing and reconciliation |
| Dates | `ExactComparator` | Date formats should match exactly |
| Monetary amounts | `NumericComparator` | Allow small rounding tolerance |
| Names and addresses | `LevenshteinComparator` | Handle OCR-induced typos |
| Descriptions and notes | `FuzzyComparator` | Token-order-independent matching |

## Resources

- [IDP Accelerator on GitHub](https://github.com/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws)
- [Bulk Evaluation Guide](../Evaluation/bulk-evaluation.md)
- [JSON Schema Extensions](../Evaluation/README.md)
- [Use Cases Overview](README.md)
