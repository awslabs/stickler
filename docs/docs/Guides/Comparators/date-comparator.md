---
title: Date Comparator
---

# DateComparator

Deterministic, non-LLM comparator for date fields. Parses both sides into `datetime` objects (or date ranges), then scores the comparison so that surface-form differences don't penalize the model, partial-information cases get partial credit, and genuinely-different dates score zero.

This page is the reference for what `DateComparator` does and how to configure it. For the broader catalog of comparators, see the [comparators index](README.md).

---

## TL;DR

```python
from stickler.comparators import DateComparator
from stickler import StructuredModel, ComparableField

class Invoice(StructuredModel):
    invoice_date: str = ComparableField(
        comparator=DateComparator(),
        threshold=1.0
    )
```

By default this comparator is **conservative**: it accepts surface-form differences (separators, padding, named months, ISO vs slash) and refuses to score anything genuinely ambiguous. Two opt-in flags (`allow_partial_year`, `range_mode`) loosen specific cases that show up in real document-extraction data.

---

## What it handles at a glance

| Ground truth | Prediction | Score | Handles |
|---|---|---|---|
| `2025-01-01` | `Jan 1, 2025` | `1.0` | [surface-form variation](#surface-form-variation) |
| `2/1/2016` | `02/01/16` | `1.0` | [zero-padding + year format](#surface-form-variation) |
| `Mon 10/24/16` | `10/24/16` | `1.0` | [weekday prefix](#surface-form-variation) |
| `Oct 24` | `10/24` | `1.0` | [both sides year-less](#missing-years) |
| `11/03` | `11/03/2012` | `0.7`*  | [year hallucination](#missing-years) |
| `10/28/16` | `10/24/16 to 10/30/16` | varies | [range-vs-single](#range-comparisons) |
| `10/24/16 to 10/30/16` | `10/24/2016 - 10/30/2016` | `1.0` | [range-vs-range](#range-comparisons) |
| `01/02/2025` | `Jan 2, 2025` | `1.0` | [ambiguous numeric dates](#ambiguous-numeric-dates) |
| `2025-01-01` | `2025-01-02` | `1.0`* | [tolerance windows](#tolerance) |
| `10/24/2016` | `10/45AM` | `0.0` | [time-only / corrupted input](#corner-cases) |
| `2025-01-01` | `''` or `None` | `0.0` | [empty input](#corner-cases) |

*starred rows depend on configuration. See the linked sections.

---

## Configuration Reference

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `tolerance` | `timedelta`, `int`, or `float` | `timedelta(0)` | Allowed difference for same-day comparisons. Numeric inputs are interpreted as days. Single dates only. |
| `dayfirst` | `Optional[bool]` | `None` | Interpretation of ambiguous numeric dates. `None` tries both and takes the better-matching score. |
| `allow_partial_year` | `bool` | `False` | If `True`, year-less ↔ year-bearing pairs with matching m/d score `0.7`. |
| `range_mode` | `"strict"` \| `"reject"` \| `"contains"` \| `"graded"` | `"graded"` | How range comparisons score. |

`threshold` is the standard `BaseComparator` parameter (forwarded unchanged) and isn't listed here — see [Comparators index](README.md) for how `threshold` interacts with `ComparableField`.

These are also accepted via `comparator_config` in JSON schemas:

```json
{
  "type": "string",
  "x-aws-stickler-comparator": "DateComparator",
  "x-aws-stickler-comparator-config": {
    "allow_partial_year": true,
    "range_mode": "contains",
    "tolerance": 1
  }
}
```

The config round-trips: `DateComparator(...).config` returns a JSON-serializable dict that can be passed back through `create_comparator("DateComparator", cfg)` to rebuild the same instance.

---

## Surface-form variation

The most common case: same calendar day, different rendering. These all score `1.0` with default settings.

| Ground truth | Prediction | Notes |
|---|---|---|
| `2025-01-01` | `Jan 1, 2025` | Different format, same date |
| `2025-01-01` | `January 1, 2025` | Full month name |
| `10/24/2016` | `10/24/16` | Two-digit vs four-digit year |
| `2/1/2016` | `02/01/2016` | Zero-padding |
| `2/1/2016` | `02/01/16` | Mixed pad + year format |
| `Mon 10/24/16` | `10/24/16` | Day-of-week prefix |
| `Monday October 24, 2016` | `10/24/16` | Long weekday + named month |
| `Oct. 24, 2016` | `October 24 2016` | Punctuation, abbreviation |
| `2016-10-24` | `10/24/16` | ISO vs slash |
| `24/10/2016` | `10/24/16` | EU vs US (day=24 disambiguates) |

Two-digit years use `python-dateutil`'s pivot: `00–68` → `2000–2068`, `69–99` → `1969–1999`. If your data needs a different cutoff, normalize upstream.

---

## Missing years

Real document-extraction data often disagrees on whether a year was present. The `allow_partial_year` flag controls how this is handled.

### Both sides lack a year

Always scores `1.0` if month and day match — neither side claimed a year, so there's no year disagreement to penalize. This is **not** controlled by `allow_partial_year`.

| Ground truth | Prediction | Score |
|---|---|---|
| `Oct 24` | `10/24` | `1.0` |
| `Nov 4` | `11/4` | `1.0` |
| `Oct 24` | `Oct 25` | `0.0` (m/d differ) |

### One side has a year, the other doesn't

This is the year-hallucination case: the model emitted a year that wasn't on the page (or vice versa).

| Ground truth | Prediction | `allow_partial_year=False` | `allow_partial_year=True` |
|---|---|---|---|
| `11/03` | `11/03/2012` | `0.0` | `0.7` |
| `Nov 4 2016` | `11/4` | `0.0` | `0.7` |
| `Oct 24` | `10/25/16` | `0.0` | `0.0` (m/d differ) |

Default is conservative: the model *did* introduce a year that may be wrong. Turn on `allow_partial_year=True` when:

- Your eval prefers fixable failures over false-zero matches.
- You're characterizing model behavior rather than gating releases.
- Your ground truth is known to be year-less (e.g., a date field that's just MM/DD).

---

## Range comparisons

A range is any string with `' to '`, `' through '`, or `' - '` (with spaces) splitting two parsable dates. The `range_mode` parameter controls scoring.

### Modes at a glance

| Case | `strict` | `reject` | `contains` | `graded` (default) |
|---|---|---|---|---|
| Same-day single vs single | `1.0` | `1.0` | `1.0` | `1.0` |
| Single inside range | `0.0` | `0.0` | `1.0` | `0.5` |
| Single outside range | `0.0` | `0.0` | `0.0` | `0.0` |
| Both ranges, endpoints exact | `1.0` | `0.0` | `1.0` | `1.0` |
| Both ranges, partial overlap | `0.0` | `0.0` | `0.0` | Jaccard |
| Both ranges, no overlap | `0.0` | `0.0` | `0.0` | `0.0` |
| `X to X` vs `X` (degenerate range) | `1.0` (collapses) | `0.0` (no collapse) | `1.0` (collapses) | `1.0` (collapses) |

### `strict`

Shape must match. Range-vs-single is a structural mismatch and scores `0.0`. Range-vs-range requires endpoints to match exactly.

| Ground truth | Prediction | Score |
|---|---|---|
| `10/28/16` | `10/24/16 to 10/30/16` | `0.0` |
| `10/24/16 to 10/30/16` | `10/24/2016 - 10/30/2016` | `1.0` |
| `10/24/16 to 10/30/16` | `10/24/16 to 10/31/16` | `0.0` |

Use this when ranges and single dates are semantically distinct in your domain and conflating them is wrong.

### `reject`

Any input that parses as a range scores `0.0`, regardless of the other side. Single-day ranges are not collapsed.

| Ground truth | Prediction | Score |
|---|---|---|
| `10/28/16` | `10/24/16 to 10/30/16` | `0.0` |
| `10/24/16 to 10/30/16` | `10/24/2016 - 10/30/2016` | `0.0` |
| `Oct 28 to Oct 28` | `Oct 28` | `0.0` |
| `Oct 28` | `Oct 28` | `1.0` (no range present) |

Use this when your schema requires single dates and any range output is a structural error you want to surface (rather than partially credit). The comparator scores `0.0` silently; if you want to *detect* range outputs, do that upstream.

### `contains`

Single-in-range scores `1.0`. Range-vs-range requires endpoints exact.

| Ground truth | Prediction | Score |
|---|---|---|
| `10/28/16` | `10/24/16 to 10/30/16` | `1.0` |
| `11/15/16` | `10/24/16 to 10/30/16` | `0.0` (gt outside) |
| `10/24/16 to 10/30/16` | `10/24/2016 - 10/30/2016` | `1.0` |
| `10/24/16 to 10/30/16` | `10/24/16 to 10/31/16` | `0.0` |

Use this when ground truth annotates a single date but the source document shows a range (and the truth date is inside it). The model isn't wrong — the annotator picked one date out of a span. This is a common artifact in real document-extraction evaluations.

### `graded` (default)

Single-in-range gets partial credit. Range-vs-range uses Jaccard overlap (`overlap_days / union_days`, day-level, inclusive).

| Ground truth | Prediction | Score |
|---|---|---|
| `10/28/16` | `10/24/16 to 10/30/16` | `0.5` |
| `11/15/16` | `10/24/16 to 10/30/16` | `0.0` |
| `10/24/16 to 10/30/16` | `10/24/2016 - 10/30/2016` | `1.0` |
| `10/24/16 to 10/30/16` | `10/24/16 to 10/31/16` | `~0.875` (7/8) |
| `10/01/2016 to 10/10/2016` | `10/06/2016 to 10/15/2016` | `~0.333` (5/15) |
| `10/24/16 to 10/30/16` | `12/01/16 to 12/05/16` | `0.0` |

Use this as your default when you want to reward "close but not exact" range predictions without giving full credit for shape mismatches.

### Composing with `allow_partial_year`

When year-presence differs between the two sides being compared (e.g., year-less single vs year-bearing range), `allow_partial_year` applies as a multiplier on the base range score:

- `allow_partial_year=False`: year mismatch zeros out the comparison.
- `allow_partial_year=True`: year mismatch multiplies the base range score by `0.7`.

| Ground truth | Prediction | range_mode | allow_partial_year=False | allow_partial_year=True |
|---|---|---|---|---|
| `Oct 28` | `10/24/16 to 10/30/16` | `contains` | `0.0` | `0.7` |
| `Oct 28` | `10/24/16 to 10/30/16` | `graded` | `0.0` | `0.35` |
| `Oct 28` | `10/24/16 to 10/30/16` | `strict` | `0.0` | `0.0` |
| `Oct 24 to Oct 30` | `10/24/16 to 10/30/16` | `graded` | `0.0` | `0.7` |

The principle: each guard contributes a confidence cap, and they multiply.

---

## Ambiguous numeric dates

A numeric date like `01/02/2025` can mean either Jan 2 or Feb 1. The `dayfirst` parameter controls how this is resolved.

### `dayfirst=None` (default)

The comparator parses each side under both interpretations and returns the better-matching score. Pairs where one side is unambiguous (ISO, named month, day > 12) effectively disambiguate the other. Pairs where no consistent interpretation matches return `0.0`.

| Ground truth | Prediction | Score |
|---|---|---|
| `01/02/2025` | `Jan 2, 2025` | `1.0` (month-first matches the named-month side) |
| `01/02/2025` | `Feb 1, 2025` | `1.0` (day-first matches the named-month side) |
| `01/02/2025` | `01/02/2025` | `1.0` (identical strings, any interpretation works) |
| `10/03/16` | `03/10/16` | `0.0` (no consistent interpretation matches) |

This is symmetric — the comparator doesn't favor ground truth or prediction.

### `dayfirst=True` or `dayfirst=False`

Forces a single interpretation for both sides. Use when your data is reliably one locale and you want to fail loudly on inputs that don't conform.

```python
us = DateComparator(dayfirst=False)
us.compare("01/02/2025", "Jan 2, 2025")    # 1.0 (Jan 2)
us.compare("01/02/2025", "Feb 1, 2025")    # 0.0

eu = DateComparator(dayfirst=True)
eu.compare("01/02/2025", "Feb 1, 2025")    # 1.0 (Feb 1)
eu.compare("01/02/2025", "Jan 2, 2025")    # 0.0
```

ISO-leading strings (`2025-01-01`) and named-month strings (`Jan 2, 2025`) always parse the same way regardless of `dayfirst`.

---

## Tolerance

`tolerance` allows two single dates to compare equal if they're within N days of each other. Accepts a `timedelta`, an `int` (days), or a `float` (days, fractional allowed).

```python
DateComparator(tolerance=timedelta(days=1))
DateComparator(tolerance=1)        # equivalent — 1 day
DateComparator(tolerance=1.5)      # 36 hours
```

```python
cmp = DateComparator(tolerance=1)
cmp.compare("2025-01-01", "2025-01-02")   # 1.0
cmp.compare("2025-01-01", "2025-01-03")   # 0.0
```

Tolerance only applies when **both sides are year-bearing single dates** (the same-calendar-day path). It does not apply to:

- Both year-less — m/d match is exact.
- One year-less — m/d match is exact (Tier covered by `allow_partial_year`).
- Range comparisons — ranges already encode uncertainty bounds.

If you want range-aware fuzziness, use `range_mode="graded"`.

---

## Corner Cases

These behaviors are deliberate. If any surprises you, that's a documentation bug — file an issue.

### `None` and empty strings

| `gt` | `pred` | Score |
|---|---|---|
| `None` | `None` | `1.0` |
| `None` | `2025-01-01` | `0.0` |
| `2025-01-01` | `None` | `0.0` |
| `''` | `2025-01-01` | `0.0` |
| `'   '` | `2025-01-01` | `0.0` (whitespace-only is empty) |

`None`/`None` matching is the standard pydantic-friendly "absence equals absence" convention used across stickler comparators.

### Time-only inputs

A string like `12:30 PM` or `10/45AM` is a time, not a date. The comparator detects this (the parsed result lacks any month/day specificity) and returns `0.0` regardless of the other side.

### Corrupted inputs

Strings like `'07/17/ 6'` (embedded whitespace inside a year) or `'11/0316'` (missing separator) typically fail to parse and return `0.0`. The comparator does not attempt to repair these — surfacing them as misses is more honest than silently accepting them.

### Range edge cases

| Case | Behavior |
|---|---|
| `'10/24/16 to '` | Empty right side → fails range parse, falls through to single → `0.0` |
| `'10/24/16 - 10/24/16'` | Endpoints equal → collapses to a single date internally (except under `range_mode="reject"`) |
| `'10/30/16 to 10/24/16'` | Endpoints reversed → fails range parse, falls through to single |
| `'2025-01-01'` | Contains a `-` but no spaces around it → not a range, parses as single ISO date |
| `'10/24/16-10/30/16'` (no spaces) | Not detected as a range. Parses as single, which dateutil rejects → `0.0` |

The "spaces required around `-`" rule keeps ISO dates from being misread as ranges. If your data has unspaced range delimiters, normalize upstream.

### Mixed date types

You can pass `datetime`, `date`, or strings on either side:

```python
from datetime import date
cmp = DateComparator()
cmp.compare(date(2025, 1, 1), "Jan 1, 2025")   # 1.0
```

Native `datetime` and `date` objects are always treated as year-bearing.

### Timezones

If both sides are tz-aware, they're normalized to UTC before comparison. If one side is naive and the other aware, the aware side's tzinfo is borrowed (matches numpy/pandas conventions).

---

## Extending the comparator

The comparator is designed to be subclassed when you need behavior that doesn't fit the existing options. Common extension points:

### Custom range delimiters

Override `_RANGE_DELIMS` (module constant) at construction time, or subclass and shadow it. Order matters — the parser tries delimiters in order and uses the first one that splits into two parsable dates.

```python
from stickler.comparators import date as _date_module
from stickler.comparators import DateComparator

class HyphenRangeDateComparator(DateComparator):
    """Treats unspaced '-' as a range delimiter (e.g. '10/24/16-10/30/16').

    Use only when ISO-format inputs are guaranteed not to appear in your
    data, since this rule will misinterpret '2025-01-01' as a range.
    """

    def _try_parse_range(self, s, dayfirst):
        result = super()._try_parse_range(s, dayfirst=dayfirst)
        if result is not None:
            return result
        # Fall back to bare-hyphen splitting.
        if s.count("-") != 1:
            return None
        left, _, right = s.partition("-")
        left_p = self._try_parse_single(left.strip(), dayfirst=dayfirst)
        right_p = self._try_parse_single(right.strip(), dayfirst=dayfirst)
        if left_p is None or right_p is None or left_p.dt > right_p.dt:
            return None
        return _date_module._ParsedRange(start=left_p, end=right_p)
```

### Custom score values

The `0.7` partial-year multiplier and `0.5` graded contains-score are constants in `date.py`. To change them per-instance, override the relevant tier method:

```python
class StricterPartialYearComparator(DateComparator):
    """Score year hallucinations at 0.5 instead of 0.7."""

    PARTIAL_YEAR_SCORE = 0.5

    def _compare_singles(self, a, b):
        score = super()._compare_singles(a, b)
        # Detect Tier 3 (one year-less) and rescale.
        if score == 0.7:
            return self.PARTIAL_YEAR_SCORE
        return score
```

### Custom range scoring

To add a new `range_mode` (e.g., asymmetric containment that only credits range-on-prediction-side), subclass and override `_compare_range_single` or `_compare_range_range`. Look at the existing implementations in `date.py` for the multiplier composition pattern.

### Custom non-date detection

The comparator rejects time-only strings via a heuristic in `_try_parse_single`. To extend with stricter validation (e.g., reject any string containing a colon), override `_try_parse_single` and return `None` early on the input shapes you want to refuse.

---

## Why these choices?

A few decisions worth calling out:

**Why is range-vs-range equality the same in `strict`, `contains`, and `graded`?** All three modes treat exact-endpoint match as a `1.0`. The mode only changes behavior for the *partial* cases (single-in-range, range-with-overlap). This keeps the modes layered: each mode is more permissive than the last on the partial cases, but the "obviously equal" case is invariant.

**Why is `"contains"` symmetric in implementation?** A common motivating case is "annotator gives a single date; document shows a range." But the comparator doesn't know which side is annotator vs. extraction — and shouldn't. A range-vs-single in `"contains"` mode scores `1.0` regardless of which side has the range. If your evaluation cares about direction (e.g., "predictions must be at least as specific as truth"), enforce that upstream.

**Why does `tolerance` only apply to single-vs-single?** Tolerance is a *single-date* notion — "within N days of the target." For range comparisons, the range itself already encodes uncertainty. Adding tolerance on top would double-count and produce confusing scores.

**Why no `partial_match_score` knob?** The partial-year score (`0.7`) and the contains/graded scores (`1.0`/`0.5`) are baked in. Tuning them per-comparator deviates from how the rest of stickler's comparators work — the standard pattern is a continuous score from the comparator and `threshold` + `clip_under_threshold` on the `ComparableField`. If you want stricter behavior, raise `threshold`. If you want something fundamentally different, see [Extending the comparator](#extending-the-comparator).

---

## See also

- [Comparators index](README.md)
- [Customizing Your Evaluation](../Evaluation/README.md) — how `threshold`, `weight`, and `clip_under_threshold` interact at the `ComparableField` layer.
