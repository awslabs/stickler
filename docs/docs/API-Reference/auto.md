# Auto (Zero-Config Evaluation)

::: stickler.auto
    options:
      heading_level: 1
      members: false

`evaluate`, `eval_for`, `EvalResult` and `EvalSpec` are re-exported at the top level, so
`stickler.evaluate` and `stickler.auto.evaluate` are the same function. `InferredSpec` and
`infer_field_config` are public but only under `stickler.auto`.

!!! tip "Which path is this?"
    This is the inference path: it takes a live Pydantic **class** and infers a comparator and
    threshold per field from the type *and* the field name. The JSON Schema path takes a schema
    **dict** and reads structure only, never names, with different thresholds. See
    [Choosing a Configuration Path](../Getting-Started/choosing-a-configuration-path.md).

::: stickler.auto.facade
    options:
      heading_level: 2
      members: false

::: stickler.auto.facade.evaluate
    options:
      heading_level: 3

::: stickler.auto.facade.eval_for
    options:
      heading_level: 3

::: stickler.auto.facade.EvalSpec
    options:
      heading_level: 3

::: stickler.auto.facade.EvalResult
    options:
      heading_level: 3

## EvalResult attributes

Set in `__init__` from the raw comparison dict, so they do not appear in the generated signature
above:

| Attribute | Type | Value |
| --- | --- | --- |
| `overall_score` | `float` | Weighted average of all field scores |
| `field_scores` | `dict[str, float]` | Per-field score, after threshold clipping |
| `precision` | `float` | `cm_precision` from the overall confusion matrix |
| `recall` | `float` | `cm_recall` |
| `f1` | `float` | `cm_f1` |
| `accuracy` | `float` | `cm_accuracy` |
| `matched` | `bool` | `overall_score >= match_threshold` (see [Meeting a Threshold](../Getting-Started/thresholds-and-metrics.md#meeting-a-threshold)). A convenience roll-up, not a per-field guarantee: individual fields can be below their thresholds while `matched` is `True`. Read `field_scores` or the confusion matrix for that. |
| `confusion_matrix` | `dict` | The full confusion-matrix subtree |
| `raw` | `dict` | The unmodified `compare_with()` result |

## Auditing the inferred config

`explain()` reports what was chosen and why, so an inferred evaluation is never a black box:

```python
result = stickler.evaluate(gt, pred)
result.explain()["invoice_id"]
```

```python
{'comparator': 'ExactComparator',
 'threshold': 1.0,
 'weight': 1.0,
 'clip_under_threshold': True,
 'source': 'name-token',
 'why': ['type:str -> LevenshteinComparator@0.7',
         'name-token:invoice_id -> ExactComparator@1.0'],
 'score': 0.0,
 'raw_similarity': 0.0}
```

`why` is the ordered trail: the type signal fired first, then the name token overrode it. Calling
`explain()` on the `EvalSpec` instead omits `score` and `raw_similarity`, since no pair has been
scored yet — use it to review the configuration before running a dataset.

## Excluded fields

A field marked `Field(exclude=True)` is not evaluated, at any depth: in nested models, list
elements and dict values.

```python
class Answer(BaseModel):
    value: str
    reasoning: str = Field(exclude=True)

result = stickler.evaluate(Answer(value="42", reasoning="..."),
                           Answer(value="42", reasoning="something else"))
result.field_scores   # {'value': 1.0}
```

`reasoning` does not appear in `field_scores`, `explain()`, `non_matches` or an HTML report, and
it carries no weight in `overall_score`. A required excluded field is not required by the
evaluation either. A nested model left with no field to score is left out the same way.

The reasoning: `exclude` marks a field as outside the model's output, which on a `response_model`
is usually a reasoning trace or internal notes. Scoring it would grade text nobody asked to be
graded, and reporting a mismatch would publish a value the author chose not to serialize.

- **`exclude_if` is decided per value, as Pydantic decides it.** A value it keeps is scored. A
  value it omits is absent, like `None`: never read or published, and missing against a kept
  value. A required field it can omit is not required.
- **A `StructuredModel` is different.** Its `ComparableField`s declare what is compared, and
  `exclude` there only affects serialization, so the field is scored by `compare_with()` and by
  `stickler.evaluate()` alike. A plain model's excluded field is not scored under one either.
- **To score an excluded field,** remove `exclude`, or evaluate an explicit `StructuredModel`
  that declares the field.

## Inference

The rules behind every decision above, plus the precedence between the type signal and the
name-token refinement, are documented in
[`src/stickler/auto/README.md`](https://github.com/awslabs/stickler/blob/main/src/stickler/auto/README.md).

::: stickler.auto.inference
    options:
      heading_level: 3
      members: false

::: stickler.auto.inference.infer_field_config
    options:
      heading_level: 3

::: stickler.auto.inference.InferredSpec
    options:
      heading_level: 3
