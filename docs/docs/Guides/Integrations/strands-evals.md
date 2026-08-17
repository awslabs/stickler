---
title: Strands Evals
---

# Strands Evals

`StructuredOutputEvaluator` scores an agent's structured output field by field
inside the [Strands Evals](https://github.com/strands-agents/evals) harness.

```bash
pip install "stickler-eval[strands-evals]"
```

```python
from strands_evals import Case, Experiment
from stickler.integrations.strands_evals import StructuredOutputEvaluator

evaluator = StructuredOutputEvaluator(Invoice)
report = Experiment(cases=cases, evaluators=[evaluator]).run_evaluations(task)

report.overall_score          # weighted mean across the dataset
report.detailed_results[0]    # per-field outputs for case 0
evaluator.metrics()           # per-field confusion matrix across the dataset
```

## Why it exists

Strands Evals' deterministic evaluator for structured output is `Equals`:
whole-object `==`, scoring 0.0 or 1.0. On real documents almost nothing matches
labels exactly, so `Equals` collapses to a near-constant near-zero. It cannot
rank two extractors, detect a regression, or say which field broke.

On a six-document sample where one prediction was perfect, one differed only in
case, one had a number slightly off, one missed a field, one hallucinated a line
item and one was wrong throughout:

| | distinct scores | overall |
|---|---|---|
| `Equals` | 2 | 0.167 |
| stickler | 4 | 0.754 |

`Equals` gave 0.0 to five of six, including the one that differed only in
capitalisation.

## Two levels of detail

### Per case

`evaluate()` returns **one `EvaluationOutput` per top-level field**. The harness
keeps all of them in `report.detailed_results`, so field detail reaches the
report without a side channel:

```python
for output in report.detailed_results[3]:
    print(output.label, output.score, output.reason)
# invoice_id    1.00  ExactComparator scored 1.00, met threshold 1.0
# vendor_name   1.00  LevenshteinComparator scored 1.00, met threshold 0.85
# invoice_date  0.00  DateComparator scored 0.00, below threshold 0.95
# total_amount  1.00  NumericComparator scored 1.00, met threshold 0.95
```

### Across the dataset

`metrics()` returns stickler's five-category confusion matrix per field path,
including nested paths. It runs **no extra comparisons**: each case is compared
once, and the raw result is kept for aggregation at read time.

```
field                       tp  fn  fa  fd   prec   rec    f1
total_amount                 4   0   0   2   0.67  1.00  0.80
invoice_date                 4   1   0   1   0.80  0.80  0.80
line_items                   5   0   1   1   0.71  1.00  0.83
line_items.sku               5   0   1   0   0.83  1.00  0.91
```

The five categories separate failure modes a single score cannot. **FN** is a
field the extractor missed, **FA** one it invented, **FD** one it got wrong.
Those need different fixes, and `Equals` reports all three as the same 0.0.

## Design choices

### Multiple outputs rather than a side channel

`EvaluationOutput` has four scalar fields (`score`, `test_pass`, `reason`,
`label`) with no metadata dict and no extra fields, and `Evaluator` has no
post-run hook. The obvious reading is that per-field detail cannot cross the
harness boundary, and an earlier draft of this integration concluded exactly
that, exposing the rollup through a separate `aggregate()` method that re-ran
every comparison.

That was wrong. `evaluate()` returns a **list**, and the harness stores the
whole list per case in `EvaluationReport.detailed_results`, typed
`list[list[EvaluationOutput]]`. So the field-level view is native to the report;
it just needs one output per field instead of one per case.

The cost of getting this wrong was doubled work. The old shape compared every
document twice, once for the harness and once for the rollup, with nothing
guaranteeing the two passes agreed.

### `label` carries the field path

This is a deliberate stretch. The framework's own example uses `label` as a
grouping tag (`label="compliant"`). We put a dotted field path in it, because it
is the only string slot that identifies which field an output describes.

If Strands Evals later adds a structured field for this, that is the better home.

### The case score is weight-aware

The framework's default aggregator takes an **unweighted** mean of the outputs.
That coincidentally equals stickler's `overall_score` when weights are uniform,
which is the default, but diverges as soon as they are not:

| | unweighted mean | stickler `overall_score` |
|---|---|---|
| `weight_hints=False` | 0.333333 | 0.333333 |
| `weight_hints=True` | 0.333333 | **0.428571** |

`EvaluationOutput` carries no weight, so the aggregator cannot recover it from
the outputs alone. The evaluator installs its own aggregator, a bound method,
which looks the weights up by field name from the compiled spec. A field's
`weight` therefore means what it says.

With several inferred schemas in play, the label set selects which weights
apply. If two schemas share field names but weight them differently, the
aggregator falls back to an unweighted mean rather than applying the wrong
weights.

### Aggregation appends, then aggregates once

This is the part worth explaining, because two more obvious designs are both
worse.

`Experiment.run_evaluations_async` defaults to `max_workers=10` and invokes
evaluators through `asyncio.to_thread`, so `evaluate()` runs on many threads.
Anything the evaluator accumulates across cases is therefore shared mutable
state. A naive read-modify-write counter loses roughly 18% of documents at that
concurrency, and loses them **silently**: the numbers simply come out low.

Three options:

| approach | correct | cost |
|---|---|---|
| lock around a shared accumulator | yes | serialises every comparison, discarding the harness's parallelism |
| one accumulator per thread, `merge_state()` at the end | yes | needs a thread registry, because `threading.local()` cannot be enumerated to merge |
| **append raw results, aggregate at read time** | yes | holds one result dict per document |

The third is what this uses. `evaluate()` appends a `(model_cls, raw_result)`
tuple to a list, which is a single atomic operation under the GIL, so no lock is
held on the hot path and nothing is lost. `metrics()` then groups and calls
`aggregate_from_comparisons` once, on one thread, after the run.

The tradeoff is memory: O(documents) rather than O(fields). For evaluation
suites that is the right trade, and `prediction_raw` is dropped from each stored
result because only the confidence accumulators consume it and they need
`field_comparisons` alongside it. Keeping it made `aggregate_from_comparisons`
warn on every call and was the bulkiest part of the payload.

If a suite is large enough that holding one dict per document matters,
`BulkStructuredModelEvaluator` accumulates incrementally and exposes
`get_state()` / `merge_state()` for the per-thread variant.

### The rollup partitions by schema

`metrics()` returns `{model_name: ProcessEvaluation}`, never one merged rollup.

This is not defensive tidiness. Feeding two schemas into a single accumulator is
accepted **silently**, and it unions their field paths. A field present in only
half the documents then reports its counts against the full document count and
reads as though the extractor missed it in the rest.

`model_cls` is therefore optional:

- **Pass it** and every case is coerced to that class, with anything that will
  not validate raising. Right for a single-schema suite.
- **Omit it** and the class is inferred per case and the rollup is partitioned.

`Experiment[InputT, OutputT]` is generic over a single output type, so
homogeneous is the intended contract, but `Experiment.__init__` accepts an
unparameterised `list[Case]`, so a mixed list type-checks and runs. Partitioning
means a mixed suite stays interpretable instead of quietly wrong.

## Reading nested rows

A nested path's counts only cover documents whose **parent pair** scored at or
above `match_threshold`. Below that, threshold gating treats the pair as atomic
and emits no field breakdown, so those documents appear as `fd` on the list
field and are absent from the child rows.

So in the table above, `line_items.sku` showing `tp=5` across 6 documents does
not mean the SKU was right five times out of six. It means five documents had a
line item close enough to look inside, and in those the SKU was right. The `fd`
on `line_items` is where the sixth went.

Two consequences:

- Nested leaves carry counts and precision/recall/F1 but **no mean score**,
  because stickler emits no per-leaf score for list children
  ([#249](https://github.com/awslabs/stickler/issues/249)). This is why the
  rollup leads with the confusion matrix, which is populated at every depth.
- The row **set** is data-dependent. A path can be absent entirely, so use
  `.get()` rather than indexing.

## Reuse and concurrency

The evaluator accumulates across cases and across experiments. Call `reset()`
before reusing an instance for a second run.

It is safe at any concurrency, as described above. That safety comes from the
design rather than from configuration, so there is no need to pin
`max_workers=1`.

## Notebooks

- [`Strands_Evals_Evaluator.ipynb`](https://github.com/awslabs/stickler/blob/dev/examples/notebooks/Strands_Evals_Evaluator.ipynb)
  is the reference. Offline and deterministic, no credentials, covers every
  feature above on six invoices broken six different ways.
- [`Strands_Evals_FCC_Two_Schemas.ipynb`](https://github.com/awslabs/stickler/blob/dev/examples/notebooks/Strands_Evals_FCC_Two_Schemas.ipynb)
  runs the same evaluator on five real FCC invoices extracted live by Claude
  Haiku, scored two ways: against a hand-written model, and against a model
  built from the dataset's own `json_schema` column with nothing hand-written.
  Requires AWS credentials with Bedrock access.

## Two gaps upstream

Both are things this integration works around rather than blockers.

**Multi-output and `aggregator` are undocumented.** `evaluate()` returning
`list[EvaluationOutput]` appears in the Strands Evals repo's `AGENTS.md`, but
neither the README nor the docs site covers returning more than one, and
`Evaluator.aggregator` is documented nowhere. This integration depends on both.
They exist and the framework uses them internally, so this is a docs gap rather
than a feature request.

**There is no post-run aggregation hook.** No mechanism lets an evaluator
contribute dataset-level metrics, which is why `metrics()` is a method the caller
invokes rather than something that appears in `EvaluationReport`. A hook called
once after all cases would let field-level rollups land in the report directly,
and would help any evaluator wanting corpus-level output, not just this one.
