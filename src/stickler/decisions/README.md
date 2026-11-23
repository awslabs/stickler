# stickler.decisions

Evaluates typed decision models: models that answer a System One question (Choice, Score, Noul) with a probability distribution. Spec: [#469](https://github.com/awslabs/stickler/issues/469). Experimental: names and result keys may change in a minor release.

## Flow

1. **Fields** (`primitives.py`). `Choice`, `Score` and `Noul` are `ComparableField`s with `ExactComparator` and threshold 1.0. Each carries a `DecisionSpec` (kind, labels, Python values), stored as `_decision_spec` on the field's `json_schema_extra` callable and exported by `to_json_schema()` under `x-aws-stickler-decision`, which `from_json_schema()` reads back (`JsonSchemaFieldImporter._decision_field`). The spec comes from the question, never the answer. A Score level named after a number other than its position is rejected.
2. **Answers** (`normalize.py`). `from_json()` calls `answers_to_rich_values()`, which turns every System One answer on a decision field, at any depth, into a rich value: `{"_value": predicted, "_confidence": p_top}`. The answer object stays in `__stickler_raw_json__` (`prediction_raw`). `normalize_answer()` is the one place answers are read:
   - only `probabilities` (Choice, Score) or `noul` is read; `choice`, `score`, `confidence`, `legend` are ignored;
   - predicted = argmax in question order; Noul is yes when P(yes) >= 0.5;
   - confidence = probability of the predicted answer;
   - Score keys are level numbers (ASCII digits only) or level names;
   - a wrong `type`, a missing or unknown key, an out-of-range probability, or a sum off 1 by more than `sum_tolerance(k)` raises `AnswerError`, unless the field has `on_invalid="missing"` (`DecisionSpec.on_invalid`): then the field becomes None, the bad answer stays in `prediction_raw`, and the accumulator records it as `rejected`.
3. **Scoring**. A decision field with gold `None` was not asked: `ComparisonEngine` and `StructuredModel.compare()` give it no weight. Extraction fields are unchanged.
4. **Accumulation** (`accumulator.py`). `BulkStructuredModelEvaluator` adds a `DecisionAccumulator` when `target_schema` has decision fields (`has_decision_fields`). Per document it reads `field_comparisons` (gold, predicted) and `prediction_raw` (distributions), and records `{"doc", "gold", "probs"}` per asked decision. A bare predicted value (no answer object) raises `AnswerError`, and a document's records are added all or none. `doc` comes from the `doc_id` the caller passed, via the `start_document` hook: a repeat raises unless `repeated_doc_ids="group"`, and a generated id is never an identity. Records are keyed by path with list indexes removed (`msgs[].intent`, see `decision_paths`). Whole-object rows (below-threshold pairs, unmatched items, a missing list) are read field by field. Results land in `ProcessEvaluation.decision_metrics`, found by accumulator type, not name, with `settings.n_bins`. Fewer than `MIN_CALIBRATION_N` answered decisions warns.
5. **Metrics** (`metrics.py`, numpy only). `correctness()` and `calibration()` take a `DecisionArrays` (`arrays.py`): `P` (n x k, NaN rows for missing answers), `y`, `groups`. Missing answers count as wrong for accuracy and are excluded from calibration; `coverage` reports both. Confidences are rounded to `CONF_DECIMALS` so a Noul "no" at 1 - c ties with a "yes" at c. Undefined precision or recall is None.
6. **After the run**. `accept_or_escalate()` (`escalate.py`) and `confidence_intervals()` (`intervals.py`) rebuild `DecisionArrays` from the stored `records`, so they need only the `ProcessEvaluation`. `escalate.risk_coverage` is the one risk-coverage curve (the HTML report uses it too); `aurc` counts missing answers as errors accepted last, `aurc_answered` is the answered-only one. Intervals resample documents, bound `RATE_METRICS` by default, and use the run's `n_bins`.

## Extending

- A new metric: add a pure function on `DecisionArrays` in `metrics.py`, call it from `correctness()` or `calibration()`, and add a reference test against sklearn or a hand-computed value in `tests/decisions/test_metrics.py`. Return plain floats.
- The System One wire format (`model_from_questions`, `from_systemone`) is in `stickler.integrations.systemone`.
