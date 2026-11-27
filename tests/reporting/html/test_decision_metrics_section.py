"""The HTML report's Decision Metrics section (#469).

The section renders only when ``ProcessEvaluation.decision_metrics`` is
present: a headline table, then per field a reliability diagram and a
risk-coverage plot, all as inline SVG. Field names and labels are
user-controlled (question ids and option keys), so each test of escaping uses
a real evaluation with hostile names rather than a hand-built dict.
"""

import html
from html.parser import HTMLParser
from typing import Optional

import pytest
from pydantic import create_model

from stickler import ComparableField, StructuredModel
from stickler.decisions import Choice, Noul, Score, accept_or_escalate
from stickler.reporting.html import html_reporter
from stickler.reporting.html.html_reporter import EvaluationHTMLReporter
from stickler.reporting.html.report_config import ReportConfig
from stickler.reporting.html.section_generator import SectionGenerator
from stickler.reporting.html.utils import DataExtractor
from stickler.reporting.html.visualization_engine import VisualizationEngine
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

HOSTILE_NAME = 'q<script>alert(1)</script>"x'
HOSTILE_LABELS = ['<b>a"</b>', "b&c<img src=x onerror=alert(2)>"]
VOID = {"meta", "br", "img", "hr", "input", "link"}


class Triage(StructuredModel):
    queue: Optional[str] = Choice(criteria={"billing": None, "sales": None, "retail": None})
    urgent: Optional[bool] = Noul()
    severity: Optional[int] = Score(criteria=["low", "medium", "high"])


class Invoice(StructuredModel):
    vendor: str = ComparableField()


def _choice(b, s, r):
    return {"type": "choice", "probabilities": {"billing": b, "sales": s, "retail": r}}


TICKETS = [
    ({"queue": "billing", "urgent": True, "severity": 1},
     {"queue": _choice(0.77, 0.12, 0.11), "urgent": {"type": "noul", "noul": 0.86},
      "severity": {"type": "score", "probabilities": {"0": 0.2, "1": 0.57, "2": 0.23}}}),
    ({"queue": "sales", "urgent": False, "severity": 0},
     {"queue": _choice(0.60, 0.30, 0.10), "urgent": {"type": "noul", "noul": 0.30},
      "severity": {"type": "score", "probabilities": {"0": 0.7, "1": 0.25, "2": 0.05}}}),
    ({"queue": "retail", "urgent": True, "severity": 2},
     {"queue": _choice(0.05, 0.05, 0.90), "urgent": {"type": "noul", "noul": 0.40},
      "severity": {"type": "score", "probabilities": {"0": 0.05, "1": 0.55, "2": 0.4}}}),
    ({"queue": "billing", "urgent": False, "severity": 1},  # queue answer missing
     {"urgent": {"type": "noul", "noul": 0.10},
      "severity": {"type": "score", "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}}}),
]


def _triage_result():
    ev = BulkStructuredModelEvaluator(target_schema=Triage)
    for i, (gold, answers) in enumerate(TICKETS):
        ev.update(Triage(**gold), Triage.from_json(answers), doc_id=f"t{i}")
    return ev.compute()


def _hostile_result():
    model = create_model(
        "Hostile", __base__=StructuredModel,
        **{HOSTILE_NAME: (Optional[str], Choice(criteria=HOSTILE_LABELS))},
    )
    a, b = HOSTILE_LABELS
    ev = BulkStructuredModelEvaluator(target_schema=model)
    for i, (gold, pa) in enumerate([(a, 0.8), (b, 0.7), (b, 0.2)]):
        answer = {HOSTILE_NAME: {"type": "choice", "probabilities": {a: pa, b: 1 - pa}}}
        ev.update(model(**{HOSTILE_NAME: gold}), model.from_json(answer), doc_id=f"d{i}")
    return ev.compute()


def _plain_result():
    ev = BulkStructuredModelEvaluator(target_schema=Invoice)
    ev.update(Invoice(vendor="Acme"), Invoice(vendor="Acme Corp"), doc_id="d1")
    return ev.compute()


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    """The footer prints the generation time; pin it so pages compare byte for byte."""
    monkeypatch.setattr(html_reporter.time, "strftime", lambda *a, **k: "2026-01-01 00:00:00")


def _page(tmp_path, results, name="report.html", **kwargs):
    out = tmp_path / name
    result = EvaluationHTMLReporter().generate_report(results, str(out), title="T", **kwargs)
    assert result.success, result.errors
    return out.read_text(encoding="utf-8"), result


class _Tags(HTMLParser):
    """Record start tags, attribute values, text, and tag balance."""

    def __init__(self):
        super().__init__()
        self.tags, self.attrs, self.names, self.text, self.stack, self.errors = [], [], set(), [], [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.extend(v for _, v in attrs if v is not None)
        self.names.update(k for k, _ in attrs)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.extend(v for _, v in attrs if v is not None)
        self.names.update(k for k, _ in attrs)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with open {self.stack[-3:]}")
            return
        self.stack.pop()

    def handle_data(self, data):
        self.text.append(data)


def _parse(fragment):
    p = _Tags()
    p.feed(fragment)
    p.close()
    return p


def _section(results):
    return SectionGenerator(results, VisualizationEngine()).generate_decision_metrics()


class TestPresence:
    def test_section_appears_with_decision_metrics(self, tmp_path):
        res = _triage_result()
        page, result = _page(tmp_path, res)

        assert "<h2>Decision Metrics</h2>" in page
        assert "decision_metrics" in result.sections_included
        assert page.count('aria-label="Reliability diagram for ') == 3
        assert page.count('aria-label="Risk-coverage plot for ') == 3
        # Headline numbers for queue: 3 of 4 answered, accuracy 2/4, macro-F1 0.5.
        table = page.split('id="decision-metrics-table"', 1)[1]
        row = table.split("<td>queue</td>", 1)[1].split("</tr>", 1)[0]
        assert "<td>choice</td>" in row
        assert "<td>billing, sales, retail</td>" in row
        assert "<td>0.750 (3 of 4)</td>" in row
        assert "<td>0.500</td>" in row
        assert "<td>0.500 (macro-F1)</td>" in row
        aurc = accept_or_escalate(res, taus=()).fields["queue"]["aurc"]
        assert f"<td>{aurc:.3f}</td>" in row
        assert "1 of 4 answers missing" in page
        assert "(QWK)" in page and "(F1)" in page

    def test_section_is_absent_without_decision_metrics(self, tmp_path):
        page, result = _page(tmp_path, _plain_result())

        assert "Decision Metrics" not in page
        assert "<svg" not in page
        assert "decision_metrics" not in result.sections_included

    def test_the_section_can_be_turned_off(self, tmp_path):
        page, result = _page(tmp_path, _triage_result(), config=ReportConfig(include_decision_metrics=False))

        assert "Decision Metrics" not in page
        assert "decision_metrics" not in result.sections_included

    @pytest.mark.parametrize("value", [None, {}, {"fields": {}}, {"fields": None}, "x"])
    def test_empty_or_malformed_metrics_count_as_absent(self, value):
        res = _plain_result()
        res.decision_metrics = value
        assert DataExtractor.extract_decision_metrics(res) is None
        assert _section(res) == ""

    def test_dict_results_never_carry_the_section(self):
        assert DataExtractor.extract_decision_metrics({"decision_metrics": {"fields": {"a": {}}}}) is None


class TestUnchangedWithoutDecisionMetrics:
    def test_report_without_decision_metrics_never_calls_the_section(self, tmp_path, monkeypatch):
        before, _ = _page(tmp_path, _plain_result(), name="before.html")

        def boom(self):
            raise AssertionError("decision section rendered without decision metrics")

        monkeypatch.setattr(SectionGenerator, "generate_decision_metrics", boom)
        after, _ = _page(tmp_path, _plain_result(), name="after.html")

        assert after == before

    def test_the_section_is_purely_additive(self, tmp_path):
        """Removing the decision metrics gives the same bytes minus exactly the section."""
        res = _triage_result()
        with_dm, _ = _page(tmp_path, res, name="with.html")
        section = _section(res)
        res.decision_metrics = None
        without_dm, _ = _page(tmp_path, res, name="without.html")

        assert section and with_dm.count(section) == 1
        assert with_dm.replace(section, "", 1) == without_dm


class TestEscaping:
    def test_hostile_field_name_and_labels_are_escaped_everywhere(self, tmp_path):
        plain, _ = _page(tmp_path, _plain_result(), name="plain.html")
        page, _ = _page(tmp_path, _hostile_result(), name="hostile.html")

        assert HOSTILE_NAME not in page
        for label in HOSTILE_LABELS:
            assert label not in page
        escaped = html.escape(HOSTILE_NAME, quote=True)
        # Every place the name's text shows up is inside its escaped form:
        # table cell, heading, both charts' aria-label and <title>.
        assert page.count(escaped) >= 6
        assert page.count("alert(1)") == page.count(escaped)
        assert page.count("alert(2)") == page.count(html.escape(HOSTILE_LABELS[1], quote=True))

        tags, baseline = _parse(page), _parse(plain)
        assert tags.tags.count("script") == baseline.tags.count("script")
        assert "b" not in tags.tags and "img" not in tags.tags
        # The browser shows the real name and labels back.
        text = "".join(tags.text)
        assert HOSTILE_NAME in text and all(label in text for label in HOSTILE_LABELS)
        assert f"Reliability diagram for {HOSTILE_NAME}" in tags.attrs

    def test_no_attribute_or_tag_is_injected_by_a_quote(self):
        p = _parse(_section(_hostile_result()))
        # A quote that broke out of aria-label would add an attribute such as `x"`.
        assert p.names <= {
            "class", "id", "style", "xmlns", "width", "height", "viewbox", "role", "aria-label",
            "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "points", "fill", "stroke",
            "stroke-width", "stroke-dasharray", "stroke-linejoin", "text-anchor", "transform",
        }
        assert set(p.tags) <= {
            "div", "h2", "h3", "p", "span", "table", "thead", "tbody", "tr", "th", "td",
            "figure", "figcaption", "svg", "title", "line", "text", "polyline", "circle",
        }


class TestParses:
    @pytest.mark.parametrize("make", [_triage_result, _hostile_result])
    def test_section_is_well_formed(self, make):
        p = _parse(_section(make()))
        assert p.errors == [] and p.stack == []

    def test_full_report_parses(self, tmp_path):
        page, _ = _page(tmp_path, _triage_result())
        p = _parse(page)
        assert p.errors == [] and p.stack == []
        assert p.tags.count("svg") == 6


class TestCharts:
    def test_risk_coverage_points_match_accept_or_escalate(self):
        res = _triage_result()
        curves = DataExtractor.extract_risk_coverage(res.decision_metrics)
        for name, curve in curves.items():
            assert curve["points"]
            for cov, risk, tau, accepted in curve["points"]:
                row = accept_or_escalate(res, taus=[tau], fields=[name]).fields[name]["rows"][0]
                assert row["coverage"] == pytest.approx(cov)
                assert 1 - row["accuracy_accepted"] == pytest.approx(risk)
                assert row["accepted"] == accepted
        assert curves["queue"]["n_missing"] == 1
        assert curves["queue"]["points"][-1][0] == pytest.approx(0.75)

    def test_long_curves_are_thinned_keeping_both_ends(self):
        res = _triage_result()
        full = DataExtractor.extract_risk_coverage(res.decision_metrics)["severity"]["points"]
        thin = DataExtractor.extract_risk_coverage(res.decision_metrics, max_points=2)["severity"]["points"]
        assert thin == [full[0], full[-1]]

    def test_metrics_without_records_still_render_table_and_reliability(self):
        res = _triage_result()
        for f in res.decision_metrics["fields"].values():
            f.pop("records")
        section = _section(res)
        assert section.count("<svg") == 3
        assert section.count("No risk-coverage curve") == 3
        assert _parse(section).errors == []

    def test_non_finite_and_missing_numbers_render_as_dash(self):
        res = _triage_result()
        cal = res.decision_metrics["fields"]["queue"]["calibration"]
        cal["auroc"], cal["ece_mass"] = None, float("nan")
        row = _section(res).split("<td>queue</td>", 1)[1].split("</tr>", 1)[0]
        assert row.count("<td>-</td>") == 2
        assert "nan" not in row.lower()
