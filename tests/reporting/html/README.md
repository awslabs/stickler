# HTML reporting tests

Tests for `src/stickler/reporting/html/`, whose own README documents the module.

| file | covers |
|---|---|
| `test_data_extractors.py` | `DataExtractor`, the single place the renderers read results from |
| `test_section_generator.py` | each `SectionGenerator` section's HTML, with `DataExtractor` patched |
| `test_html_reporter.py` | `HTMLReporter` orchestration and file output |
| `test_visualization_engine.py` | the chart and gauge fragments |
| `test_report_config.py` | `ReportConfig` validation |
| `test_output_validation.py` | well-formedness and escaping of the emitted HTML |
| `test_html_non_match_records.py` | the non-match section driven end to end, no mocks |

## Mocking boundary, and why one file deliberately crosses it

`test_section_generator.py` patches `DataExtractor.extract_non_matches` and friends, which
keeps each section's HTML independent of how results are shaped. The cost is that a defect
living *at* that boundary is invisible: a patched return value replaces exactly the code under
suspicion.

That is how #411 survived. `extract_non_matches` returned `[]` both when records had been
collected and were empty and when they were never collected at all, and the renderer turned
the second case into `No non-matches found.` over a document with a field scoring `0.0`.
Every section test passed throughout, because each one supplied its own list.

So when the question is *what the renderer decides about real results*, build the results for
real. `test_html_non_match_records.py` does that: actual `compare_with()` dicts and an actual
`BulkStructuredModelEvaluator`, through `SectionGenerator` with no patching. Add to it rather
than to the mocked file whenever a change concerns which conclusion the report draws, as
opposed to which markup it emits.

## Record semantics the tests pin

Three states, and they must stay distinguishable:

- `None`: no records collected. `compare_with()` writes the `non_matches` key only under
  `document_non_matches=True`, with or without `evaluator_format=True`, and `ProcessEvaluation.non_matches` is `None` when the bulk
  evaluator was built with `document_non_matches=False`. Says nothing about whether fields
  matched, so the section reports the records as missing.
- `[]`: collected, none produced. The only state that may claim success.
- a populated list: rendered as the table, capped by `ReportConfig.max_non_matches_displayed`.

The terminal printers in `structured_object_evaluator/utils/pretty_print.py` carry the same
three-state rule. They are separate implementations; unifying them is tracked by #137.
