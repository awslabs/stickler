# doc_split Tests

Tests for `src/stickler/doc_split/` — document packet splitting metrics.

## test_doc_split_classification_metrics.py

Tests `DocSplitClassificationMetrics` (classical metrics) across:

- Perfect match, paper edge cases (Table 3/5, Appendix A of arXiv:2602.15958)
- Input format variations (flat, nested, mixed, auto-generated IDs)
- Edge cases (empty inputs, single-page, non-sequential indices)
- Markdown report generation
- Realistic lending packet scenarios

## test_packet_evaluation_metrics.py

Tests `evaluate_packet` (proposed metrics) across:

- All 10 edge cases from the paper (Table 4, Appendix A) with strict clustering
- Expected values verified against notebook outputs
- Individual functions: clustering score, ordering score, final score
- Input format flexibility: list-of-dicts, CSV path, DataFrame
- Strict vs non-strict clustering mode
- Validation (missing columns, bad weights)

```bash
pytest tests/doc_split/ -v
```
