# tests/decisions

Tests for `stickler.decisions`.

- `test_metrics.py`: metric functions on `DecisionArrays`, against sklearn (`pytest.importorskip`) and hand-computed values, including the 4-ticket example and the bootstrap.
- `test_decisions_core.py`: fields, `from_json`, `compare_with`, the bulk evaluator, accept-or-escalate, confidence intervals, nested and list decision fields. Its 4-ticket expected values were computed independently with sklearn.
- `test_fixture_e2e.py`: the System One fixture in `examples/notebooks/fixtures/systemone/`, end to end. The committed responses are synthetic, so these check consistency, not model quality.
