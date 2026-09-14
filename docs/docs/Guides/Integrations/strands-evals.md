---
title: Strands Evals
---

# Strands Evals

[Strands Evals](https://github.com/strands-agents/evals) ships a `StructuredOutput`
evaluator that scores an agent's structured output field by field, with stickler as
its scoring engine. You write ordinary Strands Evals, and stickler supplies the
comparison: type-aware comparators, order-independent list matching, and a per-field
confusion matrix.

The evaluator lives in Strands Evals, not in stickler. stickler is the optional
`stickler` extra it depends on.

```bash
pip install "strands-agents-evals[stickler]"
```

```python
from strands import Agent
from strands_evals import Case, Experiment, eval_task
from strands_evals.evaluators import StructuredOutput

@eval_task()
def extract(case):
    agent = Agent(system_prompt="You extract invoice data.", callback_handler=None)
    result = agent(case.input, structured_output_model=Invoice)
    # A dict passes through EvalTaskHandler untouched; anything else is str()'d,
    # which would flatten the structured output into text.
    return {"output": result.structured_output}

cases = [Case[str, Invoice](name="doc-1", input=ocr_text, expected_output=label)]
evaluator = StructuredOutput(Invoice)
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
labels exactly, so `Equals` collapses to a near-constant near-zero. It cannot rank
two extractors, detect a regression, or say which field broke.

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

`evaluate()` returns one `EvaluationOutput` per case carrying stickler's weighted
`overall_score`, and `reason` names the weakest fields. Because that type has four
scalar fields, per-field detail is read from the evaluator instead:

- **`per_case()`** returns per-document field scores.
- **`metrics()`** returns stickler's five-category confusion matrix per field path,
  including nested paths, computed once from the retained comparisons with no second
  pass over the data.

```
field                       tp  fn  fa  fd   prec   rec    f1
total_amount                 4   0   0   2   0.67  1.00  0.80
invoice_date                 4   1   0   1   0.80  0.80  0.80
line_items                   5   0   1   1   0.71  1.00  0.83
line_items.sku               5   0   1   0   0.83  1.00  0.91
```

The five categories separate failure modes a single score cannot. **FN** is a field
the extractor missed, **FA** one it invented, **FD** one it got wrong. Those need
different fixes, and `Equals` reports all three as the same 0.0.

## A sharp edge worth knowing

`test_pass` is the weighted **document** score against `match_threshold`, not a
per-field verdict, and a field absent on both sides scores `1.0`. On a sparse
extraction schema those two facts combine: a document can pass with a wrong field,
and in the extreme a prediction that returned nothing can still pass, while `f1`
correctly reads `0.0`. Gate a suite on `f1`, `recall`, or named fields from
`per_case()` rather than on `test_pass` alone, and raise `weight` on the fields you
expect to be populated. See [Sparse Objects](../../Getting-Started/thresholds-and-metrics.md#sparse-objects).

## Reading nested rows

A nested path's counts only cover documents whose **parent pair** scored at or above
`match_threshold`. Below that, threshold gating treats the pair as atomic and emits no
field breakdown, so those documents appear as `fd` on the list field and are absent
from the child rows. So `line_items.sku` showing `tp=5` across 6 documents means five
documents had a line item close enough to look inside, not that the SKU was right five
times out of six.

Nested leaves carry counts and precision/recall/F1 but no mean score, because stickler
emits no per-leaf score for list children
([#249](https://github.com/awslabs/stickler/issues/249)). The row set is
data-dependent, so use `.get()` rather than indexing.

## Notebooks

- [`Strands_Evals_Evaluator.ipynb`](https://github.com/awslabs/stickler/blob/dev/examples/notebooks/Strands_Evals_Evaluator.ipynb)
  is the reference. Offline and deterministic, no credentials, covers every feature
  above on six invoices broken six different ways.
- [`Strands_Evals_FCC_Live_Agent.ipynb`](https://github.com/awslabs/stickler/blob/dev/examples/notebooks/Strands_Evals_FCC_Live_Agent.ipynb)
  runs the same evaluator against a live agent: five real FCC invoices extracted by
  Claude Haiku through Bedrock, with the Bedrock call inside the `@eval_task()`
  function. Set `AWS_PROFILE` before launching Jupyter; it needs Bedrock access.

## Status

The `StructuredOutput` evaluator is in review at
[sromoam/evals#1](https://github.com/sromoam/evals/pull/1) and is not yet on PyPI. Until
it releases, install it from the branch:

```bash
pip install "strands-agents-evals @ git+https://github.com/sromoam/evals@feat/structured-output-evaluator"
```

The evaluator's own design notes (why field detail is read from the evaluator, why
aggregation is append-only, why the rollup partitions by schema) live with the
evaluator in that PR.
