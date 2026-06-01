# DateComparator: Requirements

Implementation spec for the `DateComparator` in stickler. Written from the
perspective of an evaluator running real document-extraction outputs against
ground truth, so the priority is **fairness**: don't penalize the model for
surface-form differences, don't reward it for genuinely wrong answers, and
make ambiguous cases configurable rather than baked in.

The current implementation (as of writing) handles separator/format/locale
normalization and most year-format pairs, but has gaps around partial dates,
date ranges, and inconsistent zero-padding. This document specifies the full
behavior we want.

## Empirical grounding

The requirements below are grounded in a deep characterization of real
date-field failures across two evaluation runs (qwen3.5-2b and qwen3.6-27b
on FCC invoices, 6,630 date failures across 14,530 comparisons). Key findings:

- **GT contains ranges too** (424 of 6630 failures). Range-vs-range is the
  fifth most common failure mode (375 occurrences) — not an out-of-scope
  oddity. Single-date-vs-range is also real (~600 failures).
- **Year hallucination is the dominant fixable case** (~1500 failures).
  Predictions consistently add a year when GT has none (e.g.
  `gt='11/03'` vs `pred='11/03/2012'` recurring 179 times verbatim).
- **Pure surface-form differences** account for ~400 failures
  (zero-padding, separator, year format). These are unambiguously fixable.
- **GT data quality issues exist** (~14 cases): embedded whitespace
  (`'07/17/ 6'`), missing separators (`'11/0316'`). The comparator should
  surface but not auto-fix these.
- **Out-of-domain predictions** exist (`'10/45AM'`, `'9/5AM'`): the model
  emitted a time into a date field. The comparator must not pretend these
  are dates.

## Goals

1. Treat dates as semantic values, not strings. Parse both sides, compare
   the parsed dates, fall back to text comparison only when parsing fails.
2. Distinguish "different surface form, same date" (full match) from
   "compatible but partial information" (reduced score) from "genuinely
   different" (no match).
3. Make permissive behavior **opt-in via comparator config**, never
   silently. Default behavior should be conservative.
4. Return a deterministic score in `[0.0, 1.0]` so existing thresholding
   continues to work.

## Configurable behavior (via constructor / `comparator_config`)

| Option | Default | Description |
|---|---|---|
| `allow_partial_year` | `False` | If `True`, treat year-less sides as compatible with year-bearing sides when month/day match (Tier 3 below). Returns `partial_match_score` (default 0.7). Otherwise returns 0.0. |
| `partial_match_score` | `0.7` | Score to return for accepted partial matches. |
| `range_match_score` | `0.5` | Score to return when a single-date side is contained in a range on the other side. Set to `0.0` to disable. |
| `range_endpoints_score` | `1.0` | Score to return when both sides are ranges and their endpoints match (after normalization). Set to `0.0` to disable range-vs-range matching entirely. |
| `two_digit_year_pivot` | `50` | 2-digit years `< pivot` map to `2000+yy`, `>= pivot` map to `1900+yy`. So `pivot=50` gives `00..49 → 2000..2049`, `50..99 → 1950..1999`. |
| `dayfirst` | `None` (auto) | When the date is ambiguous (both fields ≤ 12), prefer day-first parsing. `None` means refuse to match ambiguous pairs. |
| `case_insensitive_textual` | `True` | `oct` ≡ `Oct` ≡ `OCT`. |
| `strip_weekday_prefix` | `True` | `'Mon 10/24/16'` parsed as `'10/24/16'`. |
| `warn_on_corrupted_input` | `False` | If `True`, log a warning (without altering the score) when an input matches a known data-quality pattern: embedded whitespace inside a year (`'07/17/ 6'`), missing separators (`'11/0316'`), or non-date tokens like `'10/45AM'`. Useful for surfacing GT bugs without changing scoring. |

`comparator_config` should accept these via JSON (so the StructuredModel
schema files can configure them). All other comparator base-class options
(`threshold`, `weight`, etc.) continue to work unchanged.

## Behavior tiers

### Tier 1 — must match (score = 1.0)

Both sides parse to the same calendar day; differences are purely surface.

1. **Whitespace** — leading/trailing space anywhere on either side.
   - `' 10/14/12'` ≡ `'10/14/12'`
2. **Separator** — `/`, `-`, `.`, and runs of whitespace are interchangeable
   when both sides parse to the same date.
   - `'11-03-16'` ≡ `'11/03/16'`
3. **Zero-padding** — applies independently to month and day on each side.
   - `'2/1/2016'` ≡ `'02/01/2016'`
   - `'2/1/2016'` ≡ `'02/01/16'` (combined with year-format normalization)
4. **2-digit vs 4-digit year** — using `two_digit_year_pivot`.
   - `'10/24/2016'` ≡ `'10/24/16'`
5. **Numeric vs textual month** — case-insensitive when
   `case_insensitive_textual=True`.
   - `'October 24, 2016'` ≡ `'10/24/16'`
   - `'Oct 24, 2016'` ≡ `'10/24/16'`
6. **Locale layout when unambiguous** — if at least one of month/day is `> 12`,
   layout is unambiguous and parsing must succeed regardless of order.
   - `'2016-10-24'` (ISO) ≡ `'10/24/16'` (US)
   - `'24/10/2016'` (EU, day=24) ≡ `'10/24/16'` (US)
7. **Trailing punctuation on textual** — commas, periods after month or day.
   - `'Oct 24, 2016'`, `'Oct 24 2016'`, `'Oct. 24, 2016'` all parse the same.
8. **Day-of-week prefix** — when `strip_weekday_prefix=True`.
   - `'Mon 10/24/16'`, `'Monday October 24, 2016'` ≡ `'10/24/16'`

### Tier 2 — must match (more semantic, still score = 1.0)

Same calendar day, but the two sides differ in what fields they include —
and what's present is consistent.

9. **Both sides year-less, same month/day** — score = 1.0 because no year
   was claimed on either side. Score does NOT depend on `allow_partial_year`.
   - `'Oct 24'` ≡ `'10/24'`
   - `'11/4'` ≡ `'Nov 4'`

### Tier 3 — partial match (score = `partial_match_score`, default 0.7)

Only when `allow_partial_year=True`. One side has a year, the other doesn't,
and month/day are consistent.

10. **Year-less on GT, year on pred (or vice versa)** — match at reduced
    score. The comparator cannot verify the year is right.
    - `'11/4'` vs `'Nov 4 2016'` — score = 0.7
    - `'Oct 24'` vs `'10/24/16'` — score = 0.7
    - `'11/04'` vs `'11/4/2016'` — score = 0.7

When `allow_partial_year=False` (default), these return 0.0.

> **Note on year hallucination.** In practice, this case shows up as
> "model adds a year when GT has none." E.g. GT `'11/03'` and pred
> `'11/03/2012'` recurs 179 times verbatim in our data. With
> `allow_partial_year=True` they score 0.7. If you want to be stricter
> and treat year addition as a real loss of information, leave the
> default (False) and these will score 0.

### Tier 4 — partial match (score = `range_match_score`, default 0.5)

Only when `range_match_score > 0`.

11. **Range on one side contains a single date on the other** — score =
    `range_match_score`. The model produced a span instead of a single date,
    but the ground-truth date is inside it (inclusive on both endpoints).
    - GT `'10/28/16'`, pred `'10/24/16 to 10/30/16'` → score = 0.5
    - GT `'10/31/16'`, pred `'10/31/16 to 11/06/16'` → score = 0.5
    - Range delimiters: `' to '`, `'-'` (when surrounded by spaces or two
      parsable date tokens), `' through '`.
    - If the GT date is *outside* the range → score = 0.0.

### Tier 4b — range vs range (score = `range_endpoints_score`, default 1.0)

Only when `range_endpoints_score > 0`. **Both sides are ranges.** Compare
endpoint-to-endpoint after parsing each endpoint per the rules above.

12. **Both sides parse as ranges; both start dates equal AND both end
    dates equal** — score = `range_endpoints_score`. Surface differences
    inside each endpoint follow Tier 1/2 rules.
    - `'10/24/16 to 10/30/16'` vs `'10/24/2016 - 10/30/2016'` → score = 1.0
    - `'09-12-16 to 09-15-16'` vs `'09-12-2016 to 09-15-2016'` → score = 1.0
13. **Both sides ranges, but endpoints disagree** — score = 0.0. Even if
    one endpoint matches and the other is off by a day, do not award
    partial credit. Range overlap is real but its semantics are
    ambiguous; out-of-scope here.
    - `'09-17-16'` vs `'09-12-2016'` → score = 0.0 (these were ambiguous
      single-date vs range or genuinely different ranges in our data;
      either way, treat as no match).

### Tier 5 — must not match (score = 0.0)

14. **Different calendar day** — `'10/09/12'` vs `'10/10/12'`.
15. **Different year**, otherwise identical — `'10/24/15'` vs `'10/24/16'`.
16. **Genuinely ambiguous month/day order** — both fields ≤ 12, no year-less
    side providing a tiebreak, and `dayfirst=None`.
    - `'10/03/16'` vs `'03/10/16'` → score = 0.0.
    - If `dayfirst=True` is explicitly set, parse both with day-first; if
      `dayfirst=False`, parse both with month-first. Then compare normally.
17. **Empty side** — empty string or `None` on either side → score = 0.0.
    Do not attempt to interpret either side as the other.
18. **Non-date tokens** — predictions like `'10/45AM'`, `'9/5AM'`,
    `'12:30 PM'` (a time, not a date) → score = 0.0. The minute portion
    `45` is a giveaway — months don't have 45 days. The presence of `AM`
    or `PM` together with what looks like a numeric date is also a giveaway.
    Do not attempt to interpret these as dates. Optionally warn (see
    `warn_on_corrupted_input`).
19. **Corrupted GT** — patterns like `'07/17/ 6'` (embedded whitespace
    inside year) or `'11/0316'` (missing separator). Two acceptable
    behaviors:
    - **Conservative (default):** parse fails → score = 0.0.
    - **Repair attempt:** strip embedded whitespace, infer separator from
      length, then proceed normally. **Off by default**, controlled by a
      future `tolerate_corrupted_inputs` flag if needed.

### Tier 6 — out of scope

These exist in real data but are NOT this comparator's responsibility.
Implementations may parse them gracefully and score 0.0, or raise a clear
diagnostic. Do not attempt to interpret them.

20. **Date-time** — `'10/24/16 10:30 AM'` where one side is a date+time and
    the other is a date-only. Parse only the date portion if uniquely
    identifiable, else 0.0.
21. **Quarters, fiscal periods, week numbers** — `'Q4 2016'`, `'Week 43 2016'`.
22. **Relative expressions** — `'today'`, `'next Monday'`, `'last week'`.

## Test cases

The harness lives at
`emnlp_confidence_paper_2026/stickler_based_eval/scripts/date_test_cases.py`.
After implementation, run:

```bash
uv run python scripts/date_test_cases.py
```

The cases are derived from real failures harvested from FCC invoice
predictions (see `scripts/date_failures.py`). Each case has an expected
score band; the harness will assert the comparator's score falls inside it.

The harness will be updated alongside this spec to include:

- one config (`default`) with default options — only Tier 1, 2, and 5–6
  must pass
- one config (`permissive`) with `allow_partial_year=True`,
  `range_match_score=0.5` — Tier 3 and 4 must also pass

## Implementation guidance

- Use `dateutil.parser.parse` for the heavy lifting; it already handles
  most of Tier 1 (separators, padding, textual months, ISO).
- Pre-process input with a small helper:
  - strip whitespace
  - strip a leading weekday token if `strip_weekday_prefix`
  - replace `.` separators with `/` before passing to `dateutil`
- Detect "year present" by attempting parse with `default=date(1900, 1, 1)`
  and `default=date(2099, 1, 1)`; if the parsed year differs between the
  two, the input had no year. Same trick for partial dates without
  month or day, but day-less / month-less inputs are out of scope.
- Detect ranges with a single regex over `(token)\s*(to|-|through)\s*(token)`
  where `token` is a date-like substring. Parse both endpoints and verify
  ordering.
- For ambiguous month/day, parse twice (`dayfirst=True` and `dayfirst=False`)
  and compare; if both succeed and disagree, the input is ambiguous —
  apply the `dayfirst` config rule.

## Backwards compatibility

The current `DateComparator` should keep its existing public API.
Add new options with sensible defaults so existing code paths keep
returning the same scores.

## Acceptance

The implementation is complete when, against the updated test harness:

- All Tier 1, 2, 5, 6 cases pass under default config.
- All Tier 3, 4, 4b cases pass under `permissive` config (with
  `allow_partial_year=True`, `range_match_score=0.5`,
  `range_endpoints_score=1.0`).
- Tier 5 cases (corrupted GT, non-date tokens) explicitly score 0.0
  under both configs.
- No regressions on the existing stickler test suite.

## Expected impact on real eval data

For the qwen3.6-27b run (1109 docs, 750 evaluated), the comparator
upgrade should reclaim:

- **~400 pure-surface failures** automatically (Tier 1/2).
- **~1500 partial-year failures** under `allow_partial_year=True` (Tier 3).
- **~375 range-vs-range failures** under `range_endpoints_score=1.0` (Tier 4b).
- **~1000 single-vs-range failures** under `range_match_score=0.5` (Tier 4).

Total: ~3,275 of 6,630 date failures (49%) become matches or partial
matches. The remaining ~3,355 are real misreads, hallucinated dates, or
GT-side data quality issues that no comparator can fairly fix.
