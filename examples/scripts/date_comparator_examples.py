#!/usr/bin/env python3
"""Generate a CSV of DateComparator examples for review.

Run::

    uv run python examples/scripts/date_comparator_examples.py

Writes ``date_comparator_examples.csv`` to the current directory. Each row
covers one (gt, pred, config) triple drawn from the comparator's
behavior tiers, so reviewers can scan the score column and verify it
lines up with intent.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from stickler.comparators import DateComparator


@dataclass
class Case:
    tier: str
    description: str
    gt: str
    pred: str
    # Comparator config; ``None`` means "use defaults"
    config: dict
    # Comment for the reviewer — what we expect and why
    expectation: str


def _label(config: dict) -> str:
    if not config:
        return "default"
    parts = [f"{k}={v!r}" for k, v in sorted(config.items())]
    return ", ".join(parts)


CASES: list[Case] = [
    # ------------------------------------------------------------------
    # Tier 1 — surface form
    # ------------------------------------------------------------------
    Case("1", "Identical ISO", "2025-01-01", "2025-01-01", {}, "exact match"),
    Case("1", "ISO vs named month", "2025-01-01", "Jan 1, 2025", {}, "1.0"),
    Case("1", "Two-digit vs four-digit year", "10/24/2016", "10/24/16", {}, "1.0"),
    Case("1", "Zero padding", "2/1/2016", "02/01/2016", {}, "1.0"),
    Case("1", "Mixed pad + year format", "2/1/2016", "02/01/16", {}, "1.0"),
    Case("1", "ISO vs slash with day>12", "2016-10-24", "10/24/16", {}, "1.0"),
    Case("1", "EU vs US with day>12", "24/10/2016", "10/24/16", {}, "1.0 (unambiguous)"),
    Case("1", "Weekday prefix short", "Mon 10/24/16", "10/24/16", {}, "1.0"),
    Case("1", "Weekday prefix long", "Monday October 24, 2016", "10/24/16", {}, "1.0"),
    Case("1", "Trailing punctuation", "Oct. 24, 2016", "October 24 2016", {}, "1.0"),
    # ------------------------------------------------------------------
    # Tier 2 — both year-less
    # ------------------------------------------------------------------
    Case("2", "Both year-less, named vs numeric", "Oct 24", "10/24", {}, "1.0"),
    Case("2", "Both year-less, abbreviated", "Nov 4", "11/4", {}, "1.0"),
    Case("2", "Year-less mismatch", "Oct 24", "Oct 25", {}, "0.0"),
    # ------------------------------------------------------------------
    # Tier 3 — partial year
    # ------------------------------------------------------------------
    Case("3", "Year hallucination, default strict", "11/03", "11/03/2012", {}, "0.0"),
    Case(
        "3",
        "Year hallucination, allow_partial_year",
        "11/03",
        "11/03/2012",
        {"allow_partial_year": True},
        "0.7",
    ),
    Case(
        "3",
        "Year missing on pred, allow_partial_year",
        "Nov 4 2016",
        "11/4",
        {"allow_partial_year": True},
        "0.7",
    ),
    Case(
        "3",
        "Year missing, m/d differ",
        "Oct 24",
        "10/25/16",
        {"allow_partial_year": True},
        "0.0 (m/d differ)",
    ),
    # ------------------------------------------------------------------
    # Tier 4 — single-in-range under each mode
    # ------------------------------------------------------------------
    Case("4-graded", "Inside, default", "10/28/16", "10/24/16 to 10/30/16", {}, "0.5"),
    Case(
        "4-graded",
        "Outside, default",
        "11/15/16",
        "10/24/16 to 10/30/16",
        {},
        "0.0",
    ),
    Case(
        "4-graded",
        "On endpoint",
        "10/24/16",
        "10/24/16 to 10/30/16",
        {},
        "0.5 (inclusive)",
    ),
    Case(
        "4-strict",
        "Inside, strict",
        "10/28/16",
        "10/24/16 to 10/30/16",
        {"range_mode": "strict"},
        "0.0 (shape mismatch)",
    ),
    Case(
        "4-reject",
        "Inside, reject",
        "10/28/16",
        "10/24/16 to 10/30/16",
        {"range_mode": "reject"},
        "0.0 (range disallowed)",
    ),
    Case(
        "4-contains",
        "Inside, contains",
        "10/28/16",
        "10/24/16 to 10/30/16",
        {"range_mode": "contains"},
        "1.0 (annotator-vs-source artifact)",
    ),
    Case(
        "4-contains",
        "Outside, contains",
        "11/15/16",
        "10/24/16 to 10/30/16",
        {"range_mode": "contains"},
        "0.0 (gt outside range)",
    ),
    Case(
        "4-contains",
        "Through delimiter",
        "10/28/16",
        "10/24/16 through 10/30/16",
        {"range_mode": "contains"},
        "1.0",
    ),
    Case(
        "4-contains",
        "Dash delimiter (with spaces)",
        "10/28/16",
        "10/24/16 - 10/30/16",
        {"range_mode": "contains"},
        "1.0",
    ),
    # ------------------------------------------------------------------
    # Tier 4b — range-vs-range under each mode
    # ------------------------------------------------------------------
    Case(
        "4b",
        "Endpoints exact, surface differs",
        "10/24/16 to 10/30/16",
        "10/24/2016 - 10/30/2016",
        {},
        "1.0",
    ),
    Case(
        "4b-graded",
        "Off-by-one endpoint (Jaccard)",
        "10/24/16 to 10/30/16",
        "10/24/16 to 10/31/16",
        {},
        "0.875 (7/8)",
    ),
    Case(
        "4b-graded",
        "Shifted overlap (Jaccard)",
        "10/01/2016 to 10/10/2016",
        "10/06/2016 to 10/15/2016",
        {"dayfirst": False},
        "≈0.333 (5/15)",
    ),
    Case(
        "4b-graded",
        "No overlap",
        "10/24/16 to 10/30/16",
        "12/01/16 to 12/05/16",
        {},
        "0.0",
    ),
    Case(
        "4b-strict",
        "Off-by-one under strict",
        "10/24/16 to 10/30/16",
        "10/24/16 to 10/31/16",
        {"range_mode": "strict"},
        "0.0",
    ),
    Case(
        "4b-reject",
        "Even matching ranges → 0",
        "10/24/16 to 10/30/16",
        "10/24/2016 - 10/30/2016",
        {"range_mode": "reject"},
        "0.0",
    ),
    # ------------------------------------------------------------------
    # Single-day range collapse
    # ------------------------------------------------------------------
    Case(
        "collapse",
        "X to X collapses to single (default)",
        "10/28/16 to 10/28/16",
        "10/28/16",
        {},
        "1.0 (collapses)",
    ),
    Case(
        "collapse",
        "X to X NOT collapsed under reject",
        "10/28/16 to 10/28/16",
        "10/28/16",
        {"range_mode": "reject"},
        "0.0 (no collapse)",
    ),
    # ------------------------------------------------------------------
    # Year-presence multiplier in ranges
    # ------------------------------------------------------------------
    Case(
        "multiplier",
        "Year-less single in year range, default",
        "Oct 28",
        "10/24/16 to 10/30/16",
        {},
        "0.0 (year mismatch zeros)",
    ),
    Case(
        "multiplier",
        "Year-less single in year range, partial+graded",
        "Oct 28",
        "10/24/16 to 10/30/16",
        {"allow_partial_year": True},
        "0.35 (0.5 × 0.7)",
    ),
    Case(
        "multiplier",
        "Year-less single in year range, partial+contains",
        "Oct 28",
        "10/24/16 to 10/30/16",
        {"allow_partial_year": True, "range_mode": "contains"},
        "0.7 (1.0 × 0.7)",
    ),
    # ------------------------------------------------------------------
    # Tier 5 — must-not-match
    # ------------------------------------------------------------------
    Case("5", "Off-by-one day", "10/09/12", "10/10/12", {}, "0.0"),
    Case("5", "Different year", "10/24/15", "10/24/16", {}, "0.0"),
    Case("5", "Both ambiguous, swapped", "10/03/16", "03/10/16", {}, "0.0"),
    Case("5", "Time-only string", "10/24/2016", "10/45AM", {}, "0.0"),
    Case("5", "Empty pred", "2025-01-01", "", {}, "0.0"),
    Case("5", "Corrupted (embedded space)", "2025-01-01", "07/17/ 6", {}, "0.0"),
    Case("5", "Corrupted (no separator)", "2025-01-01", "11/0316", {}, "0.0"),
    # ------------------------------------------------------------------
    # Tolerance
    # ------------------------------------------------------------------
    Case(
        "tolerance",
        "1-day tolerance, off by 1",
        "2025-01-01",
        "2025-01-02",
        {"tolerance": timedelta(days=1)},
        "1.0",
    ),
    Case(
        "tolerance",
        "1-day tolerance, off by 2",
        "2025-01-01",
        "2025-01-03",
        {"tolerance": timedelta(days=1)},
        "0.0",
    ),
    Case(
        "tolerance",
        "Tolerance does not apply to partial year",
        "Oct 24",
        "10/26/16",
        {"allow_partial_year": True, "tolerance": timedelta(days=2)},
        "0.0 (m/d strict)",
    ),
    # ------------------------------------------------------------------
    # dayfirst
    # ------------------------------------------------------------------
    Case(
        "dayfirst",
        "Ambiguous resolved by named month (default)",
        "01/02/2025",
        "Jan 2, 2025",
        {},
        "1.0 (max-of-both interpretations)",
    ),
    Case(
        "dayfirst",
        "Ambiguous resolved opposite way",
        "01/02/2025",
        "Feb 1, 2025",
        {},
        "1.0 (max-of-both)",
    ),
    Case(
        "dayfirst",
        "Forced US (month-first)",
        "01/02/2025",
        "Jan 2, 2025",
        {"dayfirst": False},
        "1.0",
    ),
    Case(
        "dayfirst",
        "Forced US, different interpretation",
        "01/02/2025",
        "Feb 1, 2025",
        {"dayfirst": False},
        "0.0",
    ),
]


def _config_for_csv(config: dict) -> str:
    """Render the config dict into a stable CSV string."""
    if not config:
        return ""
    parts = []
    for k, v in sorted(config.items()):
        if isinstance(v, timedelta):
            parts.append(f"{k}=timedelta(days={v.days})")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def main() -> None:
    out_path = Path("date_comparator_examples.csv")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tier",
                "description",
                "gt",
                "pred",
                "config",
                "score",
                "expectation",
            ]
        )
        for case in CASES:
            cmp = DateComparator(**case.config)
            score = cmp.compare(case.gt, case.pred)
            writer.writerow(
                [
                    case.tier,
                    case.description,
                    case.gt,
                    case.pred,
                    _config_for_csv(case.config),
                    f"{score:.4f}",
                    case.expectation,
                ]
            )

    print(f"Wrote {len(CASES)} cases to {out_path.resolve()}")


if __name__ == "__main__":
    main()
