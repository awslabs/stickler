"""Golden output tests for the pretty printer.

These pin what ``_print_matrix_visualization`` prints, byte for byte, and what
``_normalize_results_format`` returns, including key order. They exist so the
printer can be refactored without changing its output: they must pass on the
code before and after any such change.

The fixtures and helpers are importable so later printer changes can reuse
them rather than re-deriving the cases.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest
from pydantic import BaseModel

import stickler
from stickler.structured_object_evaluator import ComparableField, StructuredModel
from stickler.structured_object_evaluator.utils import pretty_print
from stickler.utils.process_evaluation import ProcessEvaluation

# ANSI sequences, spelled out rather than imported from ``Colors`` so the
# expected output does not move if the module's constants do.
RESET = "\x1b[0m"
HEADER_ON = "\x1b[1m\x1b[34m"
BG_TP = "\x1b[42m"  # green
BG_TN = "\x1b[44m"  # blue
BG_FP = "\x1b[41m"  # red
BG_FN = "\x1b[43m"  # yellow
BG_FD = "\x1b[45m"  # magenta

PLAIN_HEADER = "=== CONFUSION MATRIX VISUALIZATION ==="
COLOR_HEADER = f"{HEADER_ON}{PLAIN_HEADER}{RESET}"


def plain_legend(tp: str, tn: str, fp: str, fn: str, fd: str) -> str:
    """The uncoloured legend; each argument is ``"<count> (<pct>)"``."""
    return (
        "\nLegend:\n"
        f"  T True Positive (TP): {tp}\n"
        f"  N True Negative (TN): {tn}\n"
        f"  F False Positive (FP): {fp}\n"
        f"  M False Negative (FN): {fn}\n"
        f"  D False Discovery (FD): {fd}\n"
        "\n\n"
    )


def color_legend(tp: str, tn: str, fp: str, fn: str, fd: str) -> str:
    """The coloured legend; each argument is ``"<count> (<pct>)"``."""
    return (
        "\nLegend:\n"
        f"  {BG_TP}  {RESET} True Positive (TP): {tp}\n"
        f"  {BG_TN}  {RESET} True Negative (TN): {tn}\n"
        f"  {BG_FP}  {RESET} False Positive (FP): {fp}\n"
        f"  {BG_FN}  {RESET} False Negative (FN): {fn}\n"
        f"  {BG_FD}  {RESET} False Discovery (FD): {fd}\n"
        "\n\n"
    )


def color_cell(bg: str, width: int) -> str:
    return f"{bg}{' ' * width}{RESET}"


# Each case: (counts, expected plain output, expected coloured output).
MATRIX_CASES: Dict[str, Tuple[Dict[str, int], str, str]] = {
    # Nothing to draw: the function returns before printing the header.
    "all_zero": ({}, "", ""),
    "single_key": (
        {"fn": 4},
        f"{PLAIN_HEADER}\n\n"
        + "M" * 40
        + "\n"
        + plain_legend(
            "0 (0.00%)", "0 (0.00%)", "0 (0.00%)", "4 (100.00%)", "0 (0.00%)"
        ),
        f"{COLOR_HEADER}\n\n"
        + color_cell(BG_FN, 40)
        + "\n"
        + color_legend(
            "0 (0.00%)", "0 (0.00%)", "0 (0.00%)", "4 (100.00%)", "0 (0.00%)"
        ),
    ),
    # fp is 1/61 of 40 = 0.66, truncated to 0, then forced up to 1.
    "forced_width_one": (
        {"tn": 60, "fp": 1},
        f"{PLAIN_HEADER}\n\n"
        + "N" * 39
        + "F"
        + "\n"
        + plain_legend(
            "0 (0.00%)", "60 (98.36%)", "1 (1.64%)", "0 (0.00%)", "0 (0.00%)"
        ),
        f"{COLOR_HEADER}\n\n"
        + color_cell(BG_TN, 39)
        + color_cell(BG_FP, 1)
        + "\n"
        + color_legend(
            "0 (0.00%)", "60 (98.36%)", "1 (1.64%)", "0 (0.00%)", "0 (0.00%)"
        ),
    ),
    # Three cells of 13 sum to 39; the spare column goes to the first of the
    # tied cells in print order (fp), not to fn or fd.
    "tie_first_max": (
        {"fp": 1, "fn": 1, "fd": 1},
        f"{PLAIN_HEADER}\n\n"
        + "F" * 14
        + "M" * 13
        + "D" * 13
        + "\n"
        + plain_legend(
            "0 (0.00%)", "0 (0.00%)", "1 (33.33%)", "1 (33.33%)", "1 (33.33%)"
        ),
        f"{COLOR_HEADER}\n\n"
        + color_cell(BG_FP, 14)
        + color_cell(BG_FN, 13)
        + color_cell(BG_FD, 13)
        + "\n"
        + color_legend(
            "0 (0.00%)", "0 (0.00%)", "1 (33.33%)", "1 (33.33%)", "1 (33.33%)"
        ),
    ),
    # 26 + 13 = 39; the spare column goes to the largest cell (tp).
    "widen_largest": (
        {"tp": 2, "fd": 1},
        f"{PLAIN_HEADER}\n\n"
        + "T" * 27
        + "D" * 13
        + "\n"
        + plain_legend(
            "2 (66.67%)", "0 (0.00%)", "0 (0.00%)", "0 (0.00%)", "1 (33.33%)"
        ),
        f"{COLOR_HEADER}\n\n"
        + color_cell(BG_TP, 27)
        + color_cell(BG_FD, 13)
        + "\n"
        + color_legend(
            "2 (66.67%)", "0 (0.00%)", "0 (0.00%)", "0 (0.00%)", "1 (33.33%)"
        ),
    ),
    # 39 + four forced 1s = 43: the bar runs past 40 and nothing is trimmed.
    "forced_sum_past_40": (
        {"tp": 1000, "tn": 1, "fp": 1, "fn": 1, "fd": 1},
        f"{PLAIN_HEADER}\n\n"
        + "T" * 39
        + "NFMD"
        + "\n"
        + plain_legend(
            "1000 (99.60%)", "1 (0.10%)", "1 (0.10%)", "1 (0.10%)", "1 (0.10%)"
        ),
        f"{COLOR_HEADER}\n\n"
        + color_cell(BG_TP, 39)
        + color_cell(BG_TN, 1)
        + color_cell(BG_FP, 1)
        + color_cell(BG_FN, 1)
        + color_cell(BG_FD, 1)
        + "\n"
        + color_legend(
            "1000 (99.60%)", "1 (0.10%)", "1 (0.10%)", "1 (0.10%)", "1 (0.10%)"
        ),
    ),
}


def matrix_results(counts: Dict[str, int]) -> Dict[str, Any]:
    """The smallest results dict ``_print_matrix_visualization`` reads."""
    return {"confusion_matrix": {"overall": dict(counts)}}


def force_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the printer believe stdout is a colour terminal."""
    monkeypatch.setattr(pretty_print, "_supports_color", lambda: True)


def key_order(value: Any, path: str = "") -> List[Tuple[str, List[str]]]:
    """Every dict's keys, in order, keyed by path; ``==`` alone ignores order."""
    if not isinstance(value, dict):
        return []
    orders = [(path, list(value))]
    for key, child in value.items():
        orders.extend(key_order(child, f"{path}.{key}"))
    return orders


def leaf_types(value: Any, path: str = "") -> List[Tuple[str, type]]:
    """Every non-dict value's type, keyed by path; ``0 == 0.0`` hides int vs float."""
    if not isinstance(value, dict):
        return [(path, type(value))]
    return [
        leaf
        for key, child in value.items()
        for leaf in leaf_types(child, f"{path}.{key}")
    ]


@pytest.mark.parametrize("case", list(MATRIX_CASES))
def test_matrix_visualization_plain(case, capsys):
    counts, expected, _ = MATRIX_CASES[case]
    pretty_print._print_matrix_visualization(matrix_results(counts), use_color=False)
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize("case", list(MATRIX_CASES))
def test_matrix_visualization_color(case, capsys, monkeypatch):
    force_color(monkeypatch)
    counts, _, expected = MATRIX_CASES[case]
    pretty_print._print_matrix_visualization(matrix_results(counts), use_color=True)
    assert capsys.readouterr().out == expected


def test_normalize_process_evaluation():
    # Missing keys default to 0 / 0.0; keys the printer does not use are dropped.
    evaluation = ProcessEvaluation(
        metrics={"tp": 5, "fp": 1, "fn": 2, "cm_precision": 0.8, "cm_f1": 0.7},
        field_metrics={
            "name": {"tp": 3, "fd": 1, "cm_recall": 0.75, "anls": 0.9},
            "city": {},
        },
        errors=[{"doc": "a"}, {"doc": "b"}],
        total_time=2.5,
    )
    expected = {
        "confusion_matrix": {
            "overall": {
                "tp": 5,
                "fp": 1,
                "tn": 0,
                "fn": 2,
                "fd": 0,
                "fa": 0,
                "derived": {
                    "cm_precision": 0.8,
                    "cm_recall": 0.0,
                    "cm_f1": 0.7,
                    "cm_accuracy": 0.0,
                },
            },
            "fields": {
                "name": {
                    "tp": 3,
                    "fp": 0,
                    "tn": 0,
                    "fn": 0,
                    "fd": 1,
                    "fa": 0,
                    "derived": {
                        "cm_precision": 0.0,
                        "cm_recall": 0.75,
                        "cm_f1": 0.0,
                        "cm_accuracy": 0.0,
                    },
                },
                "city": {
                    "tp": 0,
                    "fp": 0,
                    "tn": 0,
                    "fn": 0,
                    "fd": 0,
                    "fa": 0,
                    "derived": {
                        "cm_precision": 0.0,
                        "cm_recall": 0.0,
                        "cm_f1": 0.0,
                        "cm_accuracy": 0.0,
                    },
                },
            },
        },
        "bulk_info": {"total_time": 2.5, "error_count": 2, "total_fields": 2},
    }
    normalized = pretty_print._normalize_results_format(evaluation)
    assert normalized == expected
    assert key_order(normalized) == key_order(expected)
    assert leaf_types(normalized) == leaf_types(expected)


class _Invoice(StructuredModel):
    number: str = ComparableField(threshold=1.0)
    total: Optional[float] = ComparableField(default=None)


def test_normalize_compare_with_dict_is_returned_unchanged():
    results = _Invoice(number="A1", total=10.0).compare_with(
        _Invoice(number="A2", total=10.0), include_confusion_matrix=True
    )
    snapshot = key_order(results)
    normalized = pretty_print._normalize_results_format(results)
    assert normalized is results
    assert key_order(normalized) == snapshot


def test_normalize_direct_metrics_dict():
    # The input's own keys come first, in their order, then ``derived``.
    results = {"cm_f1": 0.5, "tp": 3, "fp": 1, "note": "kept"}
    expected = {
        "confusion_matrix": {
            "overall": {
                "cm_f1": 0.5,
                "tp": 3,
                "fp": 1,
                "note": "kept",
                "derived": {
                    "cm_precision": 0.0,
                    "cm_recall": 0.0,
                    "cm_f1": 0.5,
                    "cm_accuracy": 0.0,
                },
            },
            "fields": {},
        }
    }
    normalized = pretty_print._normalize_results_format(results)
    assert normalized == expected
    assert key_order(normalized) == key_order(expected)
    assert leaf_types(normalized) == leaf_types(expected)


class _Person(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None


def test_normalize_eval_result_unwraps_raw():
    result = stickler.evaluate(
        _Person(name="Ada", city="London"), _Person(name="Grace", city="London")
    )
    snapshot = key_order(result.raw)
    normalized = pretty_print._normalize_results_format(result)
    assert normalized is result.raw
    assert key_order(normalized) == snapshot


@pytest.mark.parametrize(
    "results",
    [{"confusion_matrix": {}}, {"confusion_matrix": {"fields": {}}}, {"x": 1}, 42],
)
def test_normalize_unrecognised_returns_none(results):
    assert pretty_print._normalize_results_format(results) is None
