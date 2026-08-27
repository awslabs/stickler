# `stickler.algorithms`: optimal assignment between two lists

One module, one public class. `HungarianMatcher` pairs the items of two lists so
that the total similarity is as high as it can be, which is what lets a list
field be scored without caring about order. It is shared by the traditional and
the ANLS Star evaluation paths.

```python
from stickler.algorithms import HungarianMatcher
from stickler.comparators import LevenshteinComparator

matcher = HungarianMatcher(LevenshteinComparator(), match_threshold=0.7)
matcher.calculate_metrics(["apple", "banana"], ["banana", "aple"])
```

## The two methods, and which one to call

| Method | Returns | Use it when |
|---|---|---|
| `match(list1, list2)` | `(matched_indices, similarity_matrix)` | you want the assignment and the raw scores, and you will classify them yourself |
| `calculate_metrics(list1, list2)` | nine keys, described below | you want the assignment already classified against `match_threshold` |

`__call__` and `binary_compare` are legacy wrappers that return only
`(tp, fp)`.

## What the counts mean

The assignment decides what is paired. `match_threshold` then splits the paired
items into two groups. It never puts a pair back into the unpaired counts. With
`m` and `n` as the two list lengths and `k` as the pair count, which is always
`min(m, n)`:

| Key | Meaning | Closed form |
|---|---|---|
| `tp` | pair scoring at or above the threshold | |
| `fd` | pair scoring below the threshold, a false discovery | `k` minus `tp` |
| `fn` | ground truth item with no partner | `m` minus `k` |
| `fa` | prediction item with no partner, a false alarm | `n` minus `k` |
| `fp` | the rollup, `fd` plus `fa` | `n` minus `tp` |

The four leaf counts cover `max(m, n)` items once each, so nothing is counted
twice and nothing is dropped. Only one of `fn` and `fa` can be non zero.
`tp + fp == n` and `tp + fn + fd == m`, which is why `precision` and `recall`
have a denominator in closed form and the method needs no special case per
input shape.

A paired item is never also reported as missing. That was the defect in
[#231](https://github.com/awslabs/stickler/issues/231), where `fn` was derived
as `m` minus `tp` and so counted every low score pair as a missing item while
the method was returning that same pair inside `matched_pairs`.

## Two traps worth knowing before reading `tp`

**`match_threshold=0.0` is a capture all sentinel, not a threshold.** The test
is a bare `>=`, so every score satisfies zero and `tp` counts pairs rather than
true positives. `ComparisonHelper.compare_unordered_lists` builds a matcher
this way on purpose and reads only `matched_pairs`. Nothing should read `tp`
from a matcher built with `0.0`.

**The threshold test here is exact.** `ThresholdHelper`, `ComparisonHelper` and
`ConfusionMatrixCalculator` accept a score up to `1e-10` under the threshold, so
a score inside that window is a TP there and an FD here.

## How the evaluator uses it

Both internal callers read **only** `matched_pairs` and classify the scores
themselves:

| Caller | What it does with the pairs |
|---|---|
| `HungarianHelper.match_lists` | derives the matched and unmatched index sets for object matching over `List[StructuredModel]` |
| `ComparisonHelper.compare_unordered_lists` | hands them to `unordered_list_metrics`, which counts against its own `classification_threshold` |

`unordered_list_metrics` uses the same derivation as `calculate_metrics`, so the
two agree on what a count means. The remaining difference is the tolerant
threshold test noted above.

Because neither caller reads the counts, a change to how `calculate_metrics`
classifies cannot move an evaluator score. It is a public API in its own right
though, so a change there is still breaking for a direct caller.

## Design notes

**One derivation, one return.** The method used to return from five places, one
per input shape, three of which carried their own hardcoded rate values. Those
copies are what let the single item shortcut drift away from the general path in
[#224](https://github.com/awslabs/stickler/issues/224). The shortcut now
produces the pair and stops, and the counting below it is shared.

**Why the shortcut exists.** One item on each side has only one possible
assignment, so calling the solver is pure overhead on the most common list
shape. It is a performance branch, not a semantic one, and
`test_hungarian_path_parity.py` exists to keep it that way.

**`normalize_values` defaults to `True` and should usually be `False`.** It
lowercases and `str()` coerces every item before the comparator sees it, which
overrides the comparator a field declared and makes `List[bbox]` unscoreable.
See the note in `HungarianMatcher.__init__` for which call sites opt out and
why the default cannot change yet.

## Tests

`tests/common/algorithms/`. See the README there for what each file pins.
