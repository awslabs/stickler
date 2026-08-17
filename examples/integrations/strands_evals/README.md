# Stickler as a Strands Evals evaluator

The evaluator now ships as part of stickler. It used to live in this directory as
a loose module that notebooks imported through a `sys.path` hack.

**Code:** [`src/stickler/integrations/strands_evals.py`](../../../src/stickler/integrations/strands_evals.py)
**Docs:** [Guides > Integrations > Strands Evals](../../../docs/docs/Guides/Integrations/strands-evals.md)
**Tests:** [`tests/integrations/test_strands_evals.py`](../../../tests/integrations/test_strands_evals.py)

```bash
pip install "stickler-eval[strands-evals]"
```

```python
from stickler.integrations.strands_evals import StructuredOutputEvaluator
```

## Notebooks

| Notebook | Needs credentials |
|---|---|
| [`Strands_Evals_Evaluator.ipynb`](../../notebooks/Strands_Evals_Evaluator.ipynb) | No. Offline and deterministic; the reference example. |
| [`Strands_Evals_FCC_Two_Schemas.ipynb`](../../notebooks/Strands_Evals_FCC_Two_Schemas.ipynb) | Yes. Live Bedrock extraction of real FCC invoices, scored against a hand-written model and against one built from the dataset's own JSON Schema. |

## Why it moved

As an examples file it was not importable by users, not covered by CI, and not
versioned. That showed: its docstring cited a stale version, its `TODO` named an
import path that did not exist, and the per-field rollup it described re-ran
every comparison a second time.

As a package module it is exercised on every Python version the project claims,
breakage surfaces as a red build rather than a user report, and it can be
installed with `pip`.

## Upstream

The intended end state is still a PR into
[strands-agents/evals](https://github.com/strands-agents/evals), per
[evals#310](https://github.com/strands-agents/evals/issues/310) and
[stickler discussion #164](https://github.com/awslabs/stickler/discussions/164),
landing at `src/strands_evals/evaluators/stickler.py` behind a `stickler` extra
to match their existing `providers/` and `mappers/` pattern.

Two things are worth having upstream first, both covered in the
[design doc](../../../docs/docs/Guides/Integrations/strands-evals.md): the
multi-output `evaluate()` contract and `Evaluator.aggregator` need documenting,
and a post-run aggregation hook would let dataset-level metrics land in
`EvaluationReport` instead of being a method the caller invokes.
