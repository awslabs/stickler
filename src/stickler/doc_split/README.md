# doc_split — Document Packet Splitting Metrics

Evaluates how accurately a system splits a multi-page document packet into individual documents and classifies them.

Ported from the [GenAI IDP Accelerator](https://github.com/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws) `DocSplitClassificationMetrics` class, with S3/infrastructure dependencies removed.

## Reference

[DocSplit: A Comprehensive Benchmark Dataset and Evaluation Approach for Document Packet Recognition and Splitting](https://arxiv.org/abs/2602.15958) (arXiv:2602.15958)

## Metrics (Classical)

| Metric | What it measures |
|--------|-----------------|
| Page Level Accuracy | Per-page classification correctness |
| Split Accuracy (Without Order) | Correct page grouping + class, ignoring page order |
| Split Accuracy (With Order) | Correct page grouping + class + exact page order |

## Input Format

`load_sections()` accepts lists of dicts. Each dict needs:

- `section_id`: str identifier
- `document_class`: str class name, or `{"type": "invoice"}` dict
- `page_indices`: list of 0-based int page indices (or nested under `split_document.page_indices`)

## API

```python
from stickler.doc_split import DocSplitClassificationMetrics

m = DocSplitClassificationMetrics()
m.load_sections(gt_sections, pred_sections)
results = m.calculate_all_metrics()
report = m.generate_markdown_report(results)
```

## Future Work

The DocSplit paper proposes clustering-based metrics (V-measure + Rand Index for clustering, Kendall's Tau for ordering, combined Packet score) that provide continuous scoring and severity-aware penalties. These are planned for addition.
