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
    **dict** and infers from type alone, with different thresholds. See
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
| `matched` | `bool` | `True` when every field met its threshold |
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
