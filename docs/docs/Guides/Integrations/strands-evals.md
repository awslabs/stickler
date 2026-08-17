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
from strands import Agent
from strands_evals import Case, Experiment, eval_task
from stickler.integrations.strands_evals import StructuredOutputEvaluator

@eval_task()
def extract(case):
    agent = Agent(system_prompt="You extract invoice data.", callback_handler=None)
    result = agent(case.input, structured_output_model=Invoice)
    # A dict passes through EvalTaskHandler untouched; anything else is str()'d,
    # which would flatten the structured output into text.
    return {"output": result.structured_output}

cases = [Case[str, Invoice](name="doc-1", input=ocr_text, expected_output=label)]
evaluator = StructuredOutputEvaluator(Invoice)
report = Experiment[str, Invoice](cases=cases, evaluators=[evaluator]).run_evaluations(extract)

report.overall_score          # weighted mean across the dataset
report.scores                 # one weighted score per case
evaluator.per_case()          # per-document field scores
evaluator.metrics()           # per-field confusion matrix across the dataset
report.display()              # rich table; run_display() is the interactive variant
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

`evaluate()` returns one `EvaluationOutput` carrying stickler's weighted
`overall_score`, and `reason` names the weakest fields. Field-level detail comes
from `per_case()` rather than from that type:

```python
for entry in evaluator.per_case():
    print(entry["case"], entry["overall_score"], entry["field_scores"])
# doc-4  0.80  {'invoice_id': 1.0, 'vendor_name': 1.0,
#               'invoice_date': 0.0, 'total_amount': 1.0}
```

`EvaluationOutput` has four scalar fields, so `report.detailed_results` can only
ever echo what `evaluate()` returned. Reading the retained comparison instead
gives the real per-field numbers with no second comparison pass, and leaves room
to expose thresholds and classifications later without contorting them into a
score-and-two-strings shape.

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

### Field detail is read from the evaluator, not squeezed through the harness

`EvaluationOutput` has four scalar fields (`score`, `test_pass`, `reason`,
`label`), no metadata dict and no extra fields, and `Evaluator` has no post-run
hook. Two earlier drafts both got this wrong in opposite directions.

The first concluded field detail could not cross the boundary at all and exposed
a separate `aggregate()` method that **re-ran every comparison**, so each
document was compared twice with nothing guaranteeing the two passes agreed.

The second noticed that `evaluate()` returns a *list*, and that the harness keeps
the whole list per case in `EvaluationReport.detailed_results`. So it emitted one
output per field. That works, but it costs more than it buys: `label` gets
repurposed from a grouping tag into a field path, the case score has to be
recombined from the parts by a custom aggregator, and it leans on two behaviours
that are undocumented in the framework's README and docs site.

What it bought was per-case field detail as structured data. But the evaluator
already retains every comparison in order to build the dataset rollup, so that
detail was already in hand. `per_case()` serves it from there, and can carry more
than four scalars per field.

So: one output per case, `per_case()` for per-document detail, `metrics()` for the
dataset rollup. No aggregator override, `label` stays the model name, and the
only framework contract relied on is `evaluate()` returning a list.

### The case score is stickler's, not a mean of the parts

`EvaluationOutput.score` is `overall_score` directly, which is weighted by each
field's `weight`.

This is worth stating because the alternative is subtly wrong. The framework's
default aggregator takes an **unweighted** mean of a case's outputs. With one
output per case that is a no-op, so the default is correct and is left in place.
With several it silently discards weights:

| | unweighted mean | stickler `overall_score` |
|---|---|---|
| `weight_hints=False` | 0.333333 | 0.333333 |
| `weight_hints=True` | 0.333333 | **0.428571** |

The uniform-weight case agrees, which is what makes the divergence easy to miss:
it only appears once someone turns weights on.

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

The third is what this uses. `evaluate()` appends one record per case to a list,
which is a single atomic operation under the GIL, so no lock is held on the hot
path and nothing is lost. `metrics()` and `per_case()` then read that list after
the run, on one thread, and `metrics()` calls `aggregate_from_comparisons` once.

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
- [`Strands_Evals_FCC_Live_Agent.ipynb`](https://github.com/awslabs/stickler/blob/dev/examples/notebooks/Strands_Evals_FCC_Live_Agent.ipynb)
  runs the same evaluator against a live agent: five real FCC invoices extracted
  by Claude Haiku through Bedrock, with the Bedrock call inside the `@eval_task()`
  function. Its last section shows the other supported path, binding a configured
  `StructuredModel` as the agent's `structured_output_model`. Requires AWS
  credentials with Bedrock access.

## One gap upstream

**There is no post-run aggregation hook.** Nothing lets an evaluator contribute
dataset-level metrics, which is why `metrics()` and `per_case()` are methods the
caller invokes rather than data that appears in `EvaluationReport`. A hook called
once after all cases would let field-level rollups land in the report directly,
and would help any evaluator wanting corpus-level output, not just this one.

That is the only thing this integration cannot do from the outside. Everything
else it needs is in the documented contract: `evaluate()` returning a
`list[EvaluationOutput]` is specified in the repo's `AGENTS.md`.

An earlier draft also needed `Evaluator.aggregator`, which is undocumented, and
multi-output `evaluate()`, which the README and docs site do not cover even
though `AGENTS.md` states the return type. Emitting one output per case removed
both dependencies, so those are no longer asks.
