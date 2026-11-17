"""Printers restore stdout on failure and retain file/terminal output parity."""

import sys

import pytest

from stickler import StructuredModel
from stickler.structured_object_evaluator.utils import pretty_print


class Record(StructuredModel):
    name: str


@pytest.fixture
def result():
    return Record(name="Alice").compare_with(
        Record(name="Bob"), include_confusion_matrix=True, document_non_matches=True
    )


@pytest.mark.parametrize(
    "printer, helper",
    [
        ("print_non_matches", "_print_non_matches_content"),
        ("print_evaluation_results", "_print_evaluation_results_content"),
        ("print_confusion_matrix", "_print_overall_summary"),
        ("print_confusion_matrix_html", "_print_overall_summary"),
    ],
)
@pytest.mark.parametrize("error_type", [ValueError, OSError])
def test_render_failure_restores_stdout(
    printer, helper, error_type, result, tmp_path, monkeypatch, capsys
):
    def fail(*args, **kwargs):
        print("partial output")
        raise error_type("render failed")

    monkeypatch.setattr(pretty_print, helper, fail)
    original = sys.stdout
    kwargs = {} if printer.endswith("_html") else {"output_file": tmp_path / "out.txt"}
    try:
        with pytest.raises(error_type, match="render failed"):
            getattr(pretty_print, printer)(result, **kwargs)
        assert sys.stdout is original
        assert not sys.stdout.closed
    finally:
        # Keep a failing implementation from corrupting pytest's own capture.
        sys.stdout = original
    assert "Error opening output file" not in capsys.readouterr().out
    print("still writable")
    assert capsys.readouterr().out == "still writable\n"


@pytest.mark.parametrize("show_details", [False, True])
def test_bulk_file_output_matches_terminal(result, tmp_path, capsys, show_details):
    result["bulk_info"] = {"total_time": 12.5, "total_fields": 7, "error_count": 2}
    pretty_print.print_confusion_matrix(
        result, use_color=False, show_details=show_details
    )
    expected = capsys.readouterr().out
    assert "BULK EVALUATION SUMMARY" in expected
    output = tmp_path / "bulk.txt"
    pretty_print.print_confusion_matrix(
        result, output_file=output, show_details=show_details
    )
    assert output.read_text() == expected
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "printer",
    ["print_non_matches", "print_evaluation_results", "print_confusion_matrix"],
)
def test_open_failure_reports_to_original_stdout(printer, result, tmp_path, capsys):
    original = sys.stdout
    getattr(pretty_print, printer)(result, output_file=tmp_path / "missing" / "out.txt")
    assert sys.stdout is original
    assert capsys.readouterr().out.startswith("Error opening output file:")


@pytest.mark.parametrize(
    "printer", ["print_non_matches", "print_confusion_matrix_html"]
)
def test_invalid_regex_restores_stdout(printer, result, tmp_path):
    import re

    original = sys.stdout
    kwargs = {} if printer.endswith("_html") else {"output_file": tmp_path / "out.txt"}
    try:
        with pytest.raises(re.error):
            getattr(pretty_print, printer)(result, field_filter="(", **kwargs)
        assert sys.stdout is original
    finally:
        sys.stdout = original


@pytest.mark.parametrize("printer", ["print_non_matches", "print_evaluation_results"])
def test_file_output_matches_terminal(printer, result, tmp_path, capsys):
    render = getattr(pretty_print, printer)
    render(result, use_color=False)
    expected = capsys.readouterr().out
    output = tmp_path / "result.txt"
    render(result, output_file=output)
    assert output.read_text() == expected
    assert capsys.readouterr().out == ""
