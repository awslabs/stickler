---
title: Evaluating Decision Models
---

# Evaluating Decision Models

A decision model answers a typed question with a probability distribution instead of free text. Strands Decider, OpenJev and Jev speak the [System One](https://docs.typesafe.ai/primitives) format, built on three question types: **Choice** (one option out of a set), **Score** (a level on an ordered scale) and **Noul** (yes or no).

Stickler scores those answers against your gold labels: per-class and ordinal correctness, calibration, and what a confidence threshold buys. For a runnable walkthrough on 100 support tickets, see [`Evaluate_A_Decider_Model.ipynb`](https://github.com/awslabs/stickler/blob/dev/examples/notebooks/Evaluate_A_Decider_Model.ipynb).

## The questions are the schema

Pass the `questions` dict you send the model to `model_from_questions`. Each question id becomes a field; `type` sets the field kind and `criteria` the options (Choice) or levels, lowest first (Score).

```python
from stickler.integrations.systemone import model_from_questions, from_systemone

questions = {
    "queue": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": None, "sales": None, "retail": None}},
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "severity": {"type": "score", "instructions": "How severe is it?",
                 "criteria": ["low", "medium", "high"]},
}
Triage = model_from_questions(questions, weights={"queue": 2.0})

pred = Triage.from_json(from_systemone(response))   # response: the System One response dict
```

Questions may be wire dicts or `typesafe_sdk` objects; neither `typesafe_sdk` nor `strands-decider` is a dependency. `weights` is the only Stickler-specific input, and a weight for an unknown question raises `ValueError`. `from_systemone` returns the response's `answers` unchanged.

To declare the model by hand, use `Choice`, `Score` and `Noul` from `stickler.decisions`. They take the same `criteria` shapes, and this class scores identically to the one above:

```python
from typing import Optional
from stickler import StructuredModel
from stickler.decisions import Choice, Noul, Score

class Triage(StructuredModel):
    queue: Optional[str] = Choice(criteria={"billing": None, "sales": None, "retail": None}, weight=2.0)
    urgent: Optional[bool] = Noul()
    severity: Optional[int] = Score(criteria=["low", "medium", "high"])
```

Decision fields mix freely with ordinary `ComparableField`s.

## Gold labels

| Type | Gold value |
|---|---|
| Choice | an option key: `"billing"` |
| Score | a level number from 0: `1` for `"medium"` |
| Noul | `True` or `False` |
| any | `None`: the question was not asked for this document |

A value that is not one of the question's answers, gold or predicted, raises a pydantic `ValidationError` when the model is built:

```python
Triage(queue="retail2")
```

```
ValidationError: 1 validation error for Triage
queue
  Value error, 'retail2' is not an option of this Choice question; options are ['billing', 'sales', 'retail'] [type=value_error, input_value='retail2', input_type=str]
```

A Score of `5` on three levels raises the same way (`Level 5 is outside 0..2`).

## Answers

The predicted side of `Triage.from_json(...)` is a System One answer object per field:

```python
{"queue": {"type": "choice", "probabilities": {"billing": 0.77, "sales": 0.12, "retail": 0.11}},
 "urgent": {"type": "noul", "noul": 0.86},
 "severity": {"type": "score", "probabilities": {"0": 0.20, "1": 0.57, "2": 0.23}}}
```

- **Only `probabilities` (Choice, Score) or `noul` is read.** `choice`, `score`, `confidence` and `legend` may be present and are ignored. The predicted answer is the argmax, in question order on ties; a Noul is yes when `noul >= 0.5`.
- **Confidence is the probability of the predicted answer**: the top value in `probabilities`, or `max(p, 1 - p)` for a Noul. An answer's own `confidence` field measures how peaked the distribution is, not P(correct), so it is never used. The same value feeds `res.confidence_metrics`.
- **Score keys** may be level numbers (`"0"`) or level names (`"low"`).

`stickler.decisions.AnswerError` (a `ValueError`) is raised by `from_json` when an answer cannot be scored against its question:

```
AnswerError: field 'queue': question is 'choice' but the answer's type is 'noul'
AnswerError: field 'queue': probabilities are missing for ['retail']
AnswerError: field 'queue': probabilities sum to 0.900000, not 1
```

It also raises for an unknown option key, a probability outside 0 to 1, an unknown Score level, and the same answer given twice (`"0"` and `"low"`). Sums off 1 by rounding (up to 0.01, more for many options) are renormalised; sums within 1e-9 of 1 are used as given.

## Unasked vs missing

- **Unasked**: gold is `None`. The field is left out of every decision metric, the coverage denominator, and the document's `field_scores` and `overall_score`.
- **Missing**: gold exists but no answer came back (refused, errored, timed out). It counts as wrong for `accuracy`, is excluded from calibration and from `answered_accuracy`, and is counted in `coverage`.

```python
gold = Triage(queue="sales", urgent=True, severity=None)   # severity not asked
r = gold.compare_with(Triage.from_json({"queue": {"type": "choice", "probabilities": {"billing": 0.6, "sales": 0.3, "retail": 0.1}}}))
r["field_scores"]   # {'queue': 0.0, 'urgent': 0.0}
```

`queue` is wrong and `urgent` is missing; `severity` does not appear.

## Reading the results

The examples use four tickets; the fourth has no `queue` answer.

| ticket | gold `queue`, `urgent`, `severity` | `queue` P(billing, sales, retail) | `urgent` P(yes) | `severity` P(low, medium, high) |
|---|---|---|---|---|
| t1 | billing, True, 1 | 0.77, 0.12, 0.11 | 0.86 | 0.20, 0.57, 0.23 |
| t2 | sales, False, 0 | 0.60, 0.30, 0.10 | 0.30 | 0.70, 0.25, 0.05 |
| t3 | retail, True, 2 | 0.05, 0.05, 0.90 | 0.40 | 0.05, 0.55, 0.40 |
| t4 | billing, False, 1 | (missing) | 0.10 | 0.10, 0.30, 0.60 |

`BulkStructuredModelEvaluator` detects decision fields in `target_schema` and needs no configuration:

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import BulkStructuredModelEvaluator

ev = BulkStructuredModelEvaluator(target_schema=Triage)
for doc_id, gold, answers in tickets:
    ev.update(Triage(**gold), Triage.from_json(answers), doc_id=doc_id)
res = ev.compute()

q = res.decision_metrics["fields"]["queue"]
q["correctness"]["accuracy"]          # 0.5
q["correctness"]["answered_accuracy"] # 0.667
q["correctness"]["coverage"]          # {'ratio': 0.75, 'ok': 3, 'missing': 1}
q["correctness"]["confusion"]         # {'billing': {'billing': 1, '(no answer)': 1}, 'sales': {'billing': 1}, 'retail': {'retail': 1}}
q["calibration"]["nll"]               # 0.524
```

Each field holds `kind`, `labels`, `correctness`, `calibration` and the per-decision `records`:

| Type | `correctness` | `calibration` |
|---|---|---|
| every type | `accuracy`, `answered_accuracy`, `coverage` | `auroc` (confidence vs. correct), `ece_mass` (equal-mass bins, ties kept together; bins in `ece_mass_bins`), `coverage` |
| Choice | `per_class` (precision, recall, f1, support), `macro_f1`, `confusion` | `brier_multiclass`, `nll` |
| Noul | `precision`, `recall`, `f1` of yes at `threshold` 0.5 | `auroc_label` (P(yes) vs. gold), `brier`, `log_loss` |
| Score | `qwk`, `mae_argmax`, `mae_expected` (from the probability-weighted level) | `rps` |

A metric that is undefined on the data (one class only, no answered rows) is `None`. `macro_f1` averages over the options seen in gold or predictions, as sklearn does by default.

## Accept or escalate

`accept_or_escalate` measures accepting each decision whose confidence is at least `tau` and escalating the rest. Missing answers are always escalated. It reports per field by default, because confidence scales differ by type: a Noul's never goes below 0.5.

```python
from stickler.decisions import accept_or_escalate

ops = accept_or_escalate(res, taus=[0.6, 0.7, 0.8], target_accuracy=0.95)
print(ops.table())
```

```
queue  (n=4, AURC=0.04)
  tau   coverage  accuracy_accepted  errors_escalated
  0.60  0.75      0.67               1 / 2
  0.70  0.50      1.00               2 / 2
  0.80  0.25      1.00               2 / 2
  coverage at accuracy >= 0.95: 0.50 (tau = 0.770)

urgent  (n=4, AURC=0.03)
  ...
```

`ops.fields[name]` holds the same numbers (`rows`, `aurc`, `coverage_at_target`, `tau_at_target`); `ops.to_dict()` is JSON-serialisable. `pool=True` pools all fields into one entry, and `fields=[...]` selects some.

Stickler measures at the thresholds you give it; it does not choose one. A `tau_at_target` read off the run it is reported on is optimistic, so pick it on a dev run and report it on a held-out test run:

```python
tau = accept_or_escalate(res_dev, target_accuracy=0.95).fields["queue"]["tau_at_target"]
print(accept_or_escalate(res_test, taus=[tau], fields=["queue"]).table())
```

`tau_at_target` is `None` when no threshold reaches the target.

## Confidence intervals

`confidence_intervals` bootstraps every numeric decision metric of one run. It resamples documents, not decisions, because decisions about one document are correlated.

```python
from stickler.decisions import confidence_intervals

ci = confidence_intervals(res, n_boot=1000, seed=0)
ci["urgent"]["calibration"]["brier"]   # {'low': 0.0148, 'high': 0.2749}
ci["settings"]                         # {'n_boot': 1000, 'seed': 0, 'alpha': 0.05, 'unit': 'document'}
```

Every metric keeps its place in the nested result, as `{"low", "high"}`. The same seed gives the same intervals. It runs after `compute()`, on the stored records, so it needs only `res`.

## Nested and list fields

Decision fields work inside nested models and lists of models. Metrics are keyed by path, with list indexes removed so every item contributes to one set of metrics:

```python
from typing import List
from stickler import ComparableField

class Msg(StructuredModel):
    text: Optional[str] = ComparableField()
    intent: Optional[str] = Choice(criteria=["ask", "complain"])

class Thread(StructuredModel):
    msgs: Optional[List[Msg]] = None

# ... evaluate as above ...
list(res.decision_metrics["fields"])   # ['msgs[].intent']
```

A nested model's field is keyed with a dot (`customer.tier`). List items are paired by Stickler's usual list matching before their decisions are read; a gold item with no predicted partner counts as missing.

The same list field has a different key in each part of the result. Pass `fields=` to `accept_or_escalate` and `confidence_intervals` in the `decision_metrics` form:

| Where | Key |
|---|---|
| `res.decision_metrics["fields"]` | `msgs[].intent` |
| `res.confidence_metrics["fields"]` | `msgs[0].intent`, one per item |
| `res.field_metrics` | `msgs.intent` |

## Out of scope

- **Comparing two runs** (paired deltas, McNemar's test).
- **Choosing thresholds** or fitting calibrators: that is policy for the system acting on the decisions.
- **Robustness and metamorphic testing**: see [JevBench](https://github.com/JevBench/jevbench), which tests coherence on fixed public tasks. Stickler scores your decisions against your gold labels.

## Further reading

- [Confidence Evaluation](confidence-evaluation.md): confidence metrics on ordinary extraction fields
- [Bulk Evaluation](bulk-evaluation.md): the evaluator, including `get_state` / `merge_state` for distributed runs
