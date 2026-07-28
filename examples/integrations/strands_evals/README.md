# Stickler as a Strands Evals evaluator

Reference implementation of the integration requested in
[strands-agents/evals#310](https://github.com/strands-agents/evals/issues/310),
proposed in [stickler discussion #164](https://github.com/awslabs/stickler/discussions/164).

## What is here

| File | Purpose |
|---|---|
| `stickler_evaluator.py` | `StructuredOutputEvaluator`, a `strands_evals.evaluators.Evaluator` subclass that scores structured output field by field via stickler. Written in the shape it would take upstream as `src/strands_evals/evaluators/stickler.py`. |
| `../../notebooks/Strands_Evals_Integration.ipynb` | Runnable demo: imports this module, runs it in the stock `Case`/`Experiment` harness, contrasts it with `Equals`, and shows the audit trail. Offline (stub agent), no credentials needed. |

## The gap it fills

Strands Evals' deterministic evaluators compare structured output with
`Equals`: whole-object `==`, scoring 0.0 or 1.0. An extraction that gets nine
of ten fields right scores the same as one that gets none right, and a
reordered list counts as wrong.

Stickler compares field by field: dates as dates, amounts as numbers, free
text fuzzily, lists matched order-independently (Hungarian), with per-field
thresholds. Same determinism, no LLM judge, no credentials, no per-call cost,
but the score reflects how wrong the output actually is and names which fields
to look at.

## Try it

```bash
pip install "stickler-eval>=0.5.0" strands-agents-evals
```

```python
from strands_evals import Case, Experiment
from stickler_evaluator import StructuredOutputEvaluator

experiment = Experiment(
    cases=[Case(name="doc-1", input=document, expected_output=labeled_invoice)],
    evaluators=[StructuredOutputEvaluator(Invoice)],
)
report = experiment.run_evaluations(
    lambda case: agent(case.input, structured_output_model=Invoice).structured_output
)
print(report.overall_score, report.reasons)
```

## Proposed upstream shape

- Module lands at `src/strands_evals/evaluators/stickler.py`, gated behind an
  optional extra (`stickler = ["stickler-eval>=0.5.0"]`), matching the existing
  `langfuse` / `langchain` extras pattern. The guarded import and the
  actionable `ImportError` are already written for that.
- Deterministic-only, extending the direction of the deterministic-evaluators
  epic ([evals#109](https://github.com/strands-agents/evals/issues/109)).
- Stickler-side prerequisites for a frictionless install are already handled:
  the `strands-agents` upper pin is lifted, so `stickler-eval[llm]` and
  `strands-agents-evals` co-install. Stickler's `requires-python` is `>=3.12`
  against their `>=3.10`; the suite passes unmodified on 3.11, so lowering the
  floor to 3.11 is available if they want it (3.10 is blocked by
  `scikit-learn>=1.8`).
