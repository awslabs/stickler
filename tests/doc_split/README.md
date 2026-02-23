# doc_split Tests

Tests for `src/stickler/doc_split/` — document packet splitting metrics.

## test_doc_split_classification_metrics.py

Tests `DocSplitClassificationMetrics` across:

- Perfect match scenarios
- Paper edge cases (Table 3, Appendix A of arXiv:2602.15958): misclassification, wrong grouping, wrong ordering, split/merged groups, partial misclassification
- Input format variations: flat dicts, nested accelerator format, mixed formats, auto-generated section IDs
- Edge cases: empty inputs, single-page docs, empty page indices, string indices, non-sequential indices
- Markdown report generation
- Realistic multi-document lending packet scenarios

```bash
pytest tests/doc_split/ -v
```
