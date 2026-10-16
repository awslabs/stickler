---
title: Classification Logic
---

# Classification Logic

Stickler classifies every field comparison into one of five confusion-matrix categories. These categories drive all derived metrics (precision, recall, F1) and aggregate reporting.

## Core Definitions

| Category | Abbr. | Definition |
|----------|-------|------------|
| True Positive | TP | GT and EST are both non-null and match above the threshold |
| False Alarm | FA | GT is null, EST is non-null |
| True Negative | TN | GT and EST are both null |
| False Negative | FN | GT is non-null, EST is null |
| False Discovery | FD | GT and EST are both non-null but match below the threshold |

False Alarm (FA) and False Discovery (FD) together make up the broader False Positive (FP) count:

**FP = FA + FD**

## Classification by Data Type

### Simple Values (Strings, Numbers, Booleans)

| Ground Truth | Prediction | Classification | Notes |
|--------------|------------|----------------|-------|
| `"value"` | `"value"` | TP | Exact match |
| `"value"` | `"similar"` | FD | Both non-null, below threshold |
| `"value"` | `null` | FN | Missing prediction |
| `null` | `"value"` | FA | Spurious prediction |
| `null` | `null` | TN | Correctly absent |
| `""` | `null` | TN | Empty string treated as null |
| `NaN` | `NaN` | TN | Float NaN treated as null |

### Lists

#### The list contract {: #list-contract }

This table is the single statement of what Stickler does with a list field. Other pages link here rather than restate it.

| Aspect | Rule |
|---|---|
| Pairing | Every list is unordered: `["a", "b"]` vs `["b", "a"]` scores `1.0`. Elements are paired one-to-one by the [Hungarian algorithm](hungarian-matching.md), maximising total similarity. `None` and `NaN` elements are removed first ([`None` elements in a list](#none-elements-in-a-list)); reports keep original indices. Pair similarity is the field's comparator, or `GT[i].compare(Pred[j])` for `List[StructuredModel]` (fields absent on both sides excluded). |
| Bar | Primitive, dict and plain-model lists: the field's `threshold`, else a threshold named on the comparator, else `0.5`. `List[StructuredModel]`: the element class's `match_threshold`. A field `threshold` on a `List[StructuredModel]` raises `ValueError` in Python; JSON Schema warns and ignores it. A pair whose similarity is at or above the bar is a TP. |
| Counts (the list node's `overall`) | Paired element: TP if it meets the bar, else FD. Unpaired GT element: FN. Unpaired prediction: FA. An empty, `None` or all-`None` list is null: both null is one TN (score `1.0`); one side null gives one FN or FA per present element (score `0.0`). |
| Recursion | Only TP pairs of `StructuredModel` items are descended into, and their sub-fields are classified by the same rules as at top level ([#406](https://github.com/awslabs/stickler/pull/406)). FD, FN and FA items are atomic. `aggregate` holds the leaves of TP items only, so FD, FN and FA items are left out of it when any sibling was accepted. With no accepted item it falls back to the list node's own counts ([why](aggregate-metrics.md#aggregate-counts-objects-for-an-all-rejected-list)). |
| Score | `sum(pair scores) / max(n_gt, n_pred)`. The pair score is the element similarity, zeroed below the bar when `clip_under_threshold` is on (default on for primitive lists, off for dict and plain-model lists). For `List[StructuredModel]` the pair score is `GT[i].compare_with(Pred[j])["overall_score"]`, which can differ from the pairing similarity, and `clip_under_threshold` currently has no effect ([#415](https://github.com/awslabs/stickler/issues/415)). The list's score is never clipped as a whole against the field threshold. [Worked example](hungarian-matching.md#step-4-list-score). |
| Reports | `non_matches` and `field_comparisons`: one row per primitive element or per leaf of a TP item, and one object row per FD, FN or FA item. `match` is true when the row's own classification is TP or TN: a value or object row met its bar; a list row had every element paired as a TP, with no FD, FA or FN at the element level. Failures deeper down get their own rows. |

**Where `dev` does not yet follow it.** Each gap is a code change, not a docs change, and links its tracking issue where one exists:

- **Reports for primitive, dict and plain-model lists** are built from a second pairing that compares elements by equality against a `0.7` bar, not the field's comparator and bar. Rows can disagree with the counts: with `threshold=0.6`, `["abcde"]` vs `["abcxy"]` (similarity `0.6`) counts a TP but is reported as an FD with `similarity_score` `0.0`. Fixed by [#364](https://github.com/awslabs/stickler/pull/364).
- **A list row inside a list item** (`outers[0].tags`) carries `match` as the string `"True"`/`"False"`, set by comparing the list's score to the field threshold, so `outers[0].inner` can read `true` with an FD element inside. Fixed by [#364](https://github.com/awslabs/stickler/pull/364).
- **A `None` list against a non-empty one** gets one `field_comparisons` row for the whole field (`tags`), not one per element; `non_matches` and `[]` give one per element ([#418](https://github.com/awslabs/stickler/issues/418)).
- **Rows inside list items** report a clipped `score` and give a nested list or object one row instead of expanding it ([#418](https://github.com/awslabs/stickler/issues/418)).
- **Sub-fields inside list items**: some value kinds (tuple, set, bytes, whitespace-only string, a list in an `Any` field) count differently than at top level ([#406](https://github.com/awslabs/stickler/pull/406)).
- **Threshold boundary**: the object-level gate uses a bare `>=`, so a pair at `0.7999999999999999` against `0.8` is an FD ([#417](https://github.com/awslabs/stickler/issues/417)).
- **`raw_similarity_score` on a list node** is the list's score, which already has per-element clipping applied ([#330](https://github.com/awslabs/stickler/issues/330)).

#### `None` elements in a list {: #none-elements-in-a-list }

A `None` element marks an absence, not an object or a value, so it is left out before matching. This holds for a list field of models (`List[Optional[Model]]`) or of primitives (`List[Optional[str]]`), at any depth, and for a float `NaN` element, which counts as `None`. An empty string element is still a value. A list inside a list (`List[List[str]]`) or a dict value is compared as one whole value, so its `None`s are part of that value.

| GT | EST | Result |
|---|---|---|
| `[A, None]` | `[A, None]` | 1 TP |
| `[A, B, C]` | `[A, None, C]` | 2 TP, 1 FN: `B` was missed |
| `[A]` | `[None]` | 1 FN, as `[A] vs []` |
| `[None]` | `[A]` | 1 FA, as `[] vs [A]` |
| `[None]` | `[None]` | TN, as `[] vs []` |

This follows the null rules above: an absent value against a present one is an FN or FA, and two absences are not evidence of a match. Counting a `None` as an element would credit two `None`s as a TP, and would count a missed row as an FD (or, in a list of models, a TP), so the miss would keep recall at `1.0`. Elements keep their original positions in reports: in `[A, None, C]`, `C` is reported as `[2]`.

- **Positional lists are not supported either way.** Matching ignores order, so a `None` cannot mean "slot 3 is empty": `[None, A]` and `[A, None]` already compare as identical. Put the position in the data instead, for example a `page_number` field, and matching pairs elements by it.
- **To score an absence, give it content.** If ground truth marks a row as unreadable and agreeing on that should earn credit, model it as a value, such as `Line(status="illegible")`. A `None` carries nothing to compare.

#### Example: Mixed Matching

```
GT  = ["red", "blue", "green"]
EST = ["red", "yellow", "orange", "blue"]
```

The assignment pairs as many elements as the shorter list allows, so it makes
three pairs here and leaves one prediction over:

| GT | EST | Score | Classification |
|----|-----|-------|----------------|
| `"red"` | `"red"` | 1.00 | TP |
| `"blue"` | `"blue"` | 1.00 | TP |
| `"green"` | `"orange"` | 0.17 | FD |
| | `"yellow"` | | FA |

Result: TP=2, FD=1, FA=1, FN=0, and so FP=2

Note that `"green"` is an FD and not an FN. It did get a partner, and a low
score is what the FD category is for. Only an element the assignment could not
pair at all is FN or FA, which is why FN is zero whenever GT is the shorter
list.

#### Example: Below-Threshold Matches

```
GT  = ["apple", "banana", "cherry"]
EST = ["appx", "bnn", "chry"]        (threshold = 0.7)
```

All pairs match below 0.7 -- each is FD.

Result: TP=0, FA=0, FN=0, FD=3

### Nested Objects

Nested objects are evaluated recursively, field by field.

| Condition | Classification |
|-----------|---------------|
| Both have the field, similarity >= threshold | TP |
| Both have the field, similarity < threshold | FD |
| Only GT has the field | FN |
| Only EST has the field | FA |

#### Example

```
GT  = {name: "John", age: 30, address: "123 Main St"}
EST = {name: "John", age: 31, phone: "555-1234"}
```

- `name`: exact match -- TP
- `age`: both present, mismatch -- FD
- `address`: only in GT -- FN
- `phone`: only in EST -- FA

Result: TP=1, FA=1, FN=1, FD=1

### Objects of different classes {: #objects-of-different-classes }

Two objects of different classes are **always an FD**, whatever their attributes
say. Identical field names and identical values do not make them a match:

```
Pet(name="rex")  vs  Cat(name="rex")   ->  FD
```

This is the one FD in the table above that is not a threshold decision. Every
other row compares a similarity against a cutoff; this one is settled before any
comparator runs, because the class is part of the value's identity rather than
metadata about it. A `Cat` is not a `Pet` that happens to score well.

The rule uses the EXACT class, so a subclass against its base is also an FD. A
subclass carries fields the base does not, so the two do not describe the same
shape, and scoring them on their shared fields would report a near-match for what
is really a schema difference.

Stickler warns once per field rather than raising. Which class arrives is a
property of the prediction, so raising would end a bulk run on document N after
succeeding on N-1.

A correctly annotated field never reaches this rule: pydantic refuses a `Dog` for
an `Optional[Cat]` field when the object is constructed. It applies where the
annotation permitted more than one class (`Union[Cat, Dog]`, `Any`, `object`) or
where a subclass arrived for its base, which `Optional[Pet]` accepts.

!!! warning "Enforced for plain `BaseModel` objects, not yet for `StructuredModel`"

    As of this release the rule is applied to plain `pydantic.BaseModel` objects
    and to the elements of a list of them. Two **`StructuredModel`** objects of
    different classes are still scored field by field and can report `TP`:

    ```
    Pet(name="rex") vs Cat(name="rex")     plain BaseModel    0.0   FD
                                           StructuredModel    1.0   TP
    ```

    That is the older behaviour rather than a deliberate exception, and closing it is
    tracked in [#327](https://github.com/awslabs/stickler/issues/327). Until then, treat a heterogeneous `StructuredModel` pair as
    unverified rather than as endorsed: annotate the field with a single model type
    if you need the guarantee today.

    Whether a refused object should still *pair* inside a list, or should instead
    be counted as a missed ground truth plus an invented prediction, is the open
    question in [#321](https://github.com/awslabs/stickler/issues/321).

## Derived Metrics

From the base counts:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Precision | TP / (TP + FP) | Fraction of predictions that are correct |
| Recall | TP / (TP + FN) | Fraction of ground-truth values found (FD in neither term; see below) |
| F1 Score | 2 * Precision * Recall / (Precision + Recall) | Harmonic mean of precision and recall |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |

**An FD is invisible to recall under that formula**, because it appears in
neither term. The mixed-matching example above scores recall `1.000` even though
`"green"` was never found, since it is an FD and not an FN. Set
`recall_with_fd=True` for `TP / (TP + FN + FD)`, which scores the same example
`0.667`. `HungarianMatcher.calculate_metrics` always uses that second formula.
See [FD and recall](hungarian-matching.md#fd-and-recall).

These are computed at every node of the result tree, from that node's own counts, and which counts those are depends on which node you read. `overall` classifies that node's direct children: for a list field, whether each pairing was genuine or spurious; at the root, its own fields, which means the two units can mix in a single count. `aggregate` gives leaf detail, and for a list of objects that means only the items that were comparable, since a list item below the element class's `match_threshold` is one FD and is not descended into; unmatched FN and FA items are left out too when a sibling was accepted ([Recursion](#list-contract)). A single nested `StructuredModel` field is not gated this way: its leaves are always reported. See [Which node answers which question](aggregate-metrics.md#which-node-answers-which-question).

## Edge Cases

**Null vs. empty equivalence** -- Empty strings (`""`), empty lists (`[]`), empty objects (`{}`), and float `NaN` are treated as null. Comparing any of these with `null` yields TN.

**TN in object matching** -- A TN is absence of evidence, not evidence of a match. Fields absent on both sides are excluded from the weighted object similarity used by Hungarian matching. If every field is absent, the similarity is defined as `1.0`.

**Threshold boundary** -- A similarity score exactly equal to the threshold counts as a match (TP). Scores are floats, so one that prints as the threshold can sit just below it: the Laptop pair in [How Below-Threshold Pairs Are Classified](threshold-gated-evaluation.md#worked-scenarios) is `0.7999999999999999` against `0.8` and counts FD. The element gate in a primitive list allows a `1e-10` tolerance; the object and field gates do not ([#417](https://github.com/awslabs/stickler/issues/417)).

**List order** -- Order does not matter. The Hungarian algorithm finds the optimal pairing regardless of element position.

**Nested lists** -- For `List[StructuredModel]`, the Hungarian algorithm pairs objects at the list level, then each TP pair is evaluated recursively ([Recursion](#list-contract)).

**Missing vs. null fields** -- A missing field and a field explicitly set to `null` are handled the same way: if the other side has a non-null value, the result is FN or FA accordingly.

## See Also

- [Understanding Results](../Guides/Evaluation/understanding-results.md) -- interpreting the full result dictionary
- [Hungarian Matching](hungarian-matching.md) -- details on the list-pairing algorithm
