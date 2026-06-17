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
        threshold=1.0,   # field-level: how much similarity counts as a match
    )
```

`DateComparator` returns a raw similarity (0.0–1.0); the `threshold` on `ComparableField` decides what counts as a match for that field. They're separate: the comparator's own `threshold` argument is unused in this pattern — set the gating threshold on the field, as above. A quick standalone check, no model required:

```python
DateComparator().compare("2025-01-01", "Jan 1, 2025")   # 1.0
```

By default this comparator is **conservative**: it accepts surface-form differences (separators, padding, named months, ISO vs slash) and refuses to score anything genuinely ambiguous — including a reduced-precision date (`Jan 2024`) against a fuller one (`Jan 1, 2024`), which scores `0.0` by default rather than letting the parser fabricate the missing day. Three knobs (`allow_partial_year`, `range_mode`, `precision_mode`) relax specific cases that show up in real document-extraction data.

---

## What it handles at a glance

| Ground truth | Prediction | Score | Handles |
|---|---|---|---|
| `2025-01-01` | `Jan 1, 2025` | `1.0` | [surface-form variation](#surface-form-variation) |
| `2/1/2016` | `02/01/16` | `1.0` | [zero-padding + year format](#surface-form-variation) |
| `Mon 10/24/16` | `10/24/16` | `1.0` | [weekday prefix](#surface-form-variation) |
| `Oct 24` | `10/24` | `1.0` | [both sides year-less](#missing-years) |
| `11/03` | `11/03/2012` | `0.7`*  | [year hallucination](#missing-years) |
| `Jan 2024` | `Jan 1, 2024` | `0.0`* | [reduced precision](#reduced-precision) |
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
| `tolerance` | `timedelta`, `int`, or `float` | `None` (→ `timedelta(0)`, same calendar day) | Allowed difference for same-day comparisons. Numeric inputs are interpreted as days. Single dates only. |
| `dayfirst` | `Optional[bool]` | `None` | Interpretation of ambiguous numeric dates. `None` tries both and takes the better-matching score. |
| `allow_partial_year` | `bool` | `False` | If `True`, year-less ↔ year-bearing pairs with matching m/d score `0.7`. |
| `range_mode` | `"strict"` \| `"reject"` \| `"contains"` \| `"graded"` | `"graded"` | How range comparisons score. |
| `precision_mode` | `"exact"` \| `"gt_loose"` \| `"overlap"` | `"exact"` | How month/day **resolution** mismatches score (`Jan 2024` vs `Jan 1, 2024`). |

`threshold` is the standard `BaseComparator` parameter (forwarded unchanged) and isn't listed here — see [Customizing Your Evaluation](../Evaluation/README.md) for how `threshold`, `weight`, and `clip_under_threshold` interact at the `ComparableField` layer.

These are also accepted via `comparator_config` in JSON schemas:

```json
{
  "type": "string",
  "x-aws-stickler-comparator": "DateComparator",
  "x-aws-stickler-comparator-config": {
    "allow_partial_year": true,
    "range_mode": "contains",
    "precision_mode": "gt_loose",
    "tolerance": 1
  }
}
```

The config round-trips: `DateComparator(...).config` returns a JSON-serializable dict that can be passed back through `create_comparator("DateComparator", cfg)` to rebuild the same instance. Only non-default values are included, and an all-default instance returns `None` (matching the other comparators) so no redundant `x-aws-stickler-comparator-config` block is written into exported schemas.

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

Two-digit years follow `python-dateutil`'s pivot: a **sliding 50-year window centered on the current year**, so a two-digit input maps to whichever century puts it within ~50 years of today. This means the boundary moves as the calendar advances (it is not a fixed cutoff). If you need deterministic control, write the year out in full upstream.

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

!!! note "Year presence vs. resolution are different axes"
    `allow_partial_year` is about the **year** specifically — one side omits the year while still pinning month and day exactly (`11/03` vs `11/03/2012`). When a side is coarser in its **month or day** (`Jan 2024`, `2024`), that's a *resolution* mismatch, governed by [`precision_mode`](#reduced-precision). The two compose independently.

---

## Reduced precision

A date can be written at different *resolutions*: `2024` (year only), `Jan 2024` (month), `Jan 1, 2024` (day). `python-dateutil` fills any field a string omits with a default, so a naive parse of `Jan 2024` silently becomes `Jan 1, 2024` — and a year-only `2024` becomes `Jan 1, 2024`. Treating those as equal inflates scores in the direction that *hides* extraction bugs, so by default the comparator detects which of year/month/day each side actually specified and **refuses to fabricate the difference away**.

`precision_mode` controls what happens when the two sides differ in resolution. The **first argument to `compare` is treated as ground truth**, which is what lets `gt_loose` be directional.

### Modes at a glance

| Ground truth | Prediction | `exact` (default) | `gt_loose` | `overlap` |
|---|---|---|---|---|
| `Jan 2024` | `Jan 1, 2024` (pred finer) | `0.0` | `1.0` | `1.0` |
| `Jan 1, 2024` | `Jan 2024` (pred coarser) | `0.0` | `0.0` | `1.0` |
| `2024` | `2024-06-15` (pred finer) | `0.0` | `1.0` | `1.0` |
| `2024` | `2024` (same resolution) | `1.0` | `1.0` | `1.0` |
| `Jan 2024` | `Feb 2024` (value differs) | `0.0` | `0.0` | `0.0` |
| `Jan 2024` | `Feb 1, 2024` (finer, but disagrees) | `0.0` | `0.0` | `0.0` |

Same-resolution pairs are unaffected by `precision_mode` — the entire surface-form, year-less, and partial-year behavior above is untouched. The mode only decides cross-resolution pairs.

**Which mode?** Resolutions must match → `exact` (default). Ground truth is deliberately coarse and a more-specific prediction should still count → `gt_loose`. Neither side is authoritative on precision and you only care that they're consistent → `overlap`.

```python
from stickler.comparators import DateComparator

# Default: a fabricated day is a miss.
DateComparator().compare("Jan 2024", "Jan 1, 2024")                      # 0.0

# gt_loose: prediction may be finer than the (coarse) ground truth...
DateComparator(precision_mode="gt_loose").compare("Jan 2024", "Jan 1, 2024")  # 1.0
# ...but not coarser than it.
DateComparator(precision_mode="gt_loose").compare("Jan 1, 2024", "Jan 2024")  # 0.0

# overlap: either side may be coarser, as long as they're consistent.
DateComparator(precision_mode="overlap").compare("Jan 1, 2024", "Jan 2024")   # 1.0
DateComparator(precision_mode="overlap").compare("Jan 2024", "Feb 1, 2024")   # 0.0 (month differs)
```

### `exact` (default)

Both sides must specify the same fields. A prediction that adds a month/day the ground truth didn't have (or drops one it did) is a miss. This is the conservative default for release-gating evals: a fabricated component is exactly the kind of extraction error you don't want scored as a match.

### `gt_loose`

The ground truth sets the required resolution. A prediction may be **more** precise than the ground truth — the extra precision is ignored as long as it's consistent at the ground truth's grain (`Jan 2024` vs `Jan 1, 2024` → `1.0`) — but may **not** be less precise (`Jan 1, 2024` vs `Jan 2024` → `0.0`, the prediction under-specified the truth). Use this when ground truth is deliberately coarse (a month-level or year-level field) and any in-grain prediction should count.

### `overlap`

Symmetric: either side may be the coarser one, and they match as long as they agree on every field both sides specify. Use this when neither side is authoritative on resolution and you only care that the two are *consistent*, not that they're equally precise.

### Composing with `allow_partial_year`

The two axes are independent. `precision_mode` judges month/day resolution; `allow_partial_year` judges year presence (and carries the `0.7` partial-year credit). A pair can be gated by either: e.g. under `precision_mode="exact"`, `Jan 2024` vs `Jan 1, 2024` is `0.0` regardless of `allow_partial_year`, because the resolution gate fails first.

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
| `Dec 25` | `Dec 20, 2024 to Jan 5, 2025` | `contains` | `0.0` | `0.7` |

The principle: each guard contributes a confidence cap, and they multiply.

When year-presence differs, the comparison drops to `(month, day)` space (the year-less side has no real year to compare). This applies to every range shape:

- **Range-vs-single** containment is checked on m/d. Ranges that **wrap the year boundary** in that space — e.g. `Dec 20 → Jan 5`, common for fiscal and holiday spans — are handled as a wrap-around: both `Dec 25` and `Jan 2` count as inside.
- **Range-vs-range** endpoint equality (`strict`/`contains`) and overlap (`graded`) are likewise measured on m/d, so `Oct 24 to Oct 30` vs `10/24/16 to 10/30/16` scores `1.0 × 0.7` under `contains` and its m/d Jaccard `× 0.7` under `graded`.

(When both sides carry a year, no m/d projection is involved — the full dates are compared directly.)

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
| `2025-02-01` | `01/02/2025` | `1.0` (ISO side pins Feb 1; month-first reading of the prediction agrees) |
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

ISO / year-first strings (`2025-01-01`, `2025/01/01`) and named-month strings (`Jan 2, 2025`) always parse the same way regardless of `dayfirst`. A leading four-digit year fixes month-then-day order, so the comparator **pins year-first inputs to month-first parsing even when `dayfirst=True`** — without that, an unambiguous canonical date like `2025-02-01` would be misread as Jan 2 and a non-canonical prediction of the same date would score `0.0`.

---

## Tolerance

`tolerance` allows two single dates to compare equal if they're within the window of each other. Accepts a `timedelta`, an `int` (days), or a `float` (days, fractional allowed).

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

### Whole-day vs. sub-day windows

The comparison granularity follows the tolerance:

- **Whole-day tolerance** (`0`, `1`, `2`, …) floors both sides to their calendar day before measuring, so intra-day times are ignored and the window counts whole calendar days. This is the common case and the default (`tolerance=0` → same calendar day).
- **Sub-day tolerance** (any value with an hours/minutes component, e.g. `1.5` = 36h or `0.5` = 12h) compares the **actual timestamps** without flooring, so the window means real elapsed time.

```python
cmp = DateComparator(tolerance=1.5)   # 36 hours, real elapsed time
cmp.compare("2025-01-01 00:00", "2025-01-02 12:00")   # 1.0 (exactly 36h)
cmp.compare("2025-01-01 00:00", "2025-01-02 13:00")   # 0.0 (37h)

day = DateComparator(tolerance=1)     # 1 calendar day, time ignored
day.compare("2025-01-01 00:00", "2025-01-02 23:00")   # 1.0 (1 calendar day apart)
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

**Why a silent `0.0` rather than an explicit error or warning?** Issue #117 asked that unparseable/ambiguous input surface as a failure rather than a silent guess, and that's exactly what `0.0` is here: an unparseable value scores as a non-match, which a threshold of any value treats as a miss. An earlier draft carried a `warn_on_corrupted_input` flag that logged when an input matched a known data-quality pattern; it was dropped because a comparator's job is to return an honest similarity score, not to run a side-channel logger — that conflates scoring with data-quality reporting and diverges from how every other stickler comparator behaves. If you need to *detect* corrupted inputs (not just score them as misses), validate upstream of the evaluation, where you have the field context to act on it.

Inputs longer than 256 characters are rejected (scored `0.0`) before parsing — a real date string is far shorter, and skipping the parse keeps a pathologically long garbage value from costing a full scan.

### Range edge cases

| Case | Behavior |
|---|---|
| `'10/24/16 to '` | Empty right side → fails range parse; the string still carries a range-delimiter signal, so it's rejected as a malformed range → `0.0` (not silently re-read as a single date) |
| `' - 10/24/16'`, `'10/24/16 -'` | Dangling dash at an edge → treated as a truncated range and rejected → `0.0` (in every `range_mode`, including `reject`) |
| `'10/24/16 - 10/24/16'` | Endpoints equal → collapses to a single date internally (except under `range_mode="reject"`) |
| `'10/30/16 to 10/24/16'` | Endpoints reversed → fails range parse; carries a delimiter signal, so rejected → `0.0` |
| `'2025-01-01'` | Dash sits *between digits*, not at an edge → not a range signal, parses as a single ISO date |
| `'10/24/16-10/30/16'` (no spaces) | Internal dash, no delimiter signal → parsed as a single, which dateutil rejects → `0.0` |

Two rules keep legitimate dates from being misread: a bare `-` is only a range delimiter when surrounded by spaces (so ISO `2025-01-01` is safe), and a string that carries a delimiter signal but fails to parse as a valid range is rejected rather than silently re-read as a single date (so a dangling `'- 10/24/16'` can't score as a clean date). If your data has unspaced range delimiters, normalize upstream.

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

`_RANGE_DELIMS` is a module-level constant that the parser reads directly, so assigning it on an instance has no effect. To change which delimiters count as a range, subclass and override `_try_parse_range` (as the example below does). The parser tries delimiters in order and uses the first one that splits into two parsable dates, so order matters.

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

**Why is `precision_mode` a separate axis from `allow_partial_year`?** They answer different questions. `allow_partial_year` is about whether a *year* was hallucinated or dropped, with month and day still pinned exactly — a frequent, well-characterized failure mode that earns its own tuned `0.7` partial credit. `precision_mode` is about *month/day resolution* — whether a side is coarser than the other. Folding both into one knob would force a single policy on two failure modes that real data treats differently (a hallucinated year is "probably wrong"; a coarser month is "less specific but not wrong"). Keeping them orthogonal is what makes the configuration space logically complete: every (year-presence, resolution) combination has a defined score.

**Why is `precision_mode` binary (1.0/0.0) rather than graded?** Partial credit already lives on the two axes that have a natural magnitude — `allow_partial_year` (the `0.7` year-hallucination credit) and `range_mode="graded"` (Jaccard overlap for explicit ranges). A resolution mismatch is a yes/no question ("is the prediction allowed to be this coarse/fine?"), so a third graded scale would add tuning surface without a principled magnitude behind it. If you want a non-binary resolution score, see [Extending the comparator](#extending-the-comparator).

**Why is `precision_mode="gt_loose"` directional when `"contains"` isn't?** Resolution genuinely has a "more specific" ordering (day ⊃ month ⊃ year), and ground truth is a meaningful anchor for it — `compare(gt, pred)` passes ground truth first at every real call site. Containment of an explicit range has no comparable canonical direction, so `"contains"` stays symmetric and `"overlap"` is offered for callers who want symmetry on resolution too.

**Why no `partial_match_score` knob?** The partial-year score (`0.7`) and the contains/graded scores (`1.0`/`0.5`) are baked in. Tuning them per-comparator deviates from how the rest of stickler's comparators work — the standard pattern is a continuous score from the comparator and `threshold` + `clip_under_threshold` on the `ComparableField`. If you want stricter behavior, raise `threshold`. If you want something fundamentally different, see [Extending the comparator](#extending-the-comparator).

---

## See also

- [Comparators index](README.md)
- [Customizing Your Evaluation](../Evaluation/README.md) — how `threshold`, `weight`, and `clip_under_threshold` interact at the `ComparableField` layer.
