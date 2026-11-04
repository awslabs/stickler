"""Decision-field metrics: sklearn references, hand-computed values, bootstrap."""

import itertools
import json

import numpy as np
import pytest

from stickler.decisions.arrays import DecisionArrays, empty_arrays
from stickler.decisions.metrics import (
    auroc,
    binary_brier,
    bootstrap_ci,
    brier_multiclass,
    calibration,
    correctness,
    ece_mass,
    log_loss,
    quadratic_weighted_kappa,
    ranked_probability_score,
)
from stickler.decisions.primitives import spec_from_question

QUEUE = spec_from_question("choice", {"billing": None, "sales": None, "retail": None})
URGENT = spec_from_question("noul")
SEVERITY = spec_from_question("score", ["low", "medium", "high"])


def _random_arrays(spec, n=200, n_docs=40, missing=0.1, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(spec.k), size=n)
    P[rng.random(n) < missing] = np.nan
    y = rng.integers(0, spec.k, size=n)
    groups = rng.integers(0, n_docs, size=n)
    return DecisionArrays(spec, P, y, groups)


def _assert_plain(obj):
    """Every leaf is a plain Python type, so the result is JSON-serialisable."""
    json.dumps(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_plain(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_plain(v)
    else:
        assert obj is None or type(obj) in (int, float, str, bool), type(obj)


# Reference tests against sklearn


@pytest.fixture(scope="module")
def skm():
    return pytest.importorskip("sklearn.metrics")


def test_auroc_matches_sklearn(skm):
    rng = np.random.default_rng(1)
    labels = rng.random(300) < 0.4
    scores = np.round(rng.random(300), 2)  # rounding forces ties
    assert auroc(labels, scores) == pytest.approx(skm.roc_auc_score(labels, scores))


def test_binary_brier_and_log_loss_match_sklearn(skm):
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=300)
    p = rng.uniform(0.01, 0.99, size=300)
    assert binary_brier(p, y) == pytest.approx(skm.brier_score_loss(y, p))
    assert log_loss(p, y) == pytest.approx(skm.log_loss(y, p))


def test_nll_matches_sklearn_multiclass_log_loss(skm):
    a = _random_arrays(QUEUE, missing=0.0, seed=3)
    cal = calibration(a)
    assert cal["nll"] == pytest.approx(skm.log_loss(a.y, a.P, labels=[0, 1, 2]))


def test_qwk_matches_sklearn(skm):
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 5, size=300)
    y_pred = np.clip(y_true + rng.integers(-2, 3, size=300), 0, 4)
    expected = skm.cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(5)))
    assert quadratic_weighted_kappa(y_true, y_pred, 5) == pytest.approx(expected)


def test_score_qwk_uses_answered_rows(skm):
    a = _random_arrays(SEVERITY, seed=5)
    ok = a.answered
    expected = skm.cohen_kappa_score(a.y[ok], a.pred[ok], weights="quadratic", labels=[0, 1, 2])
    assert correctness(a)["qwk"] == pytest.approx(expected)


def test_choice_per_class_matches_sklearn_with_missing_as_false_negative(skm):
    a = _random_arrays(QUEUE, seed=6)
    assert a.n_missing > 0
    labels = [0, 1, 2]
    # pred is -1 where missing, a label outside `labels`, so it is a false negative.
    p, r, f, s = skm.precision_recall_fscore_support(a.y, a.pred, labels=labels, zero_division=0)
    got = correctness(a)
    for i, name in enumerate(QUEUE.labels):
        assert got["per_class"][name]["precision"] == pytest.approx(p[i])
        assert got["per_class"][name]["recall"] == pytest.approx(r[i])
        assert got["per_class"][name]["f1"] == pytest.approx(f[i])
        assert got["per_class"][name]["support"] == s[i]
    expected_macro = skm.f1_score(a.y, a.pred, labels=labels, average="macro", zero_division=0)
    assert got["macro_f1"] == pytest.approx(expected_macro)


def test_noul_metrics_match_sklearn(skm):
    a = _random_arrays(URGENT, seed=7)
    ok = a.answered
    p_yes = a.P[ok, 1]
    says_yes = np.zeros(a.n_total, dtype=int)
    says_yes[ok] = p_yes >= 0.5
    says_yes[~ok] = -1  # missing: never a predicted yes, a false negative if gold is yes
    p, r, f, _ = skm.precision_recall_fscore_support(a.y, says_yes, labels=[1], zero_division=0)
    got = correctness(a)
    assert (got["precision"], got["recall"], got["f1"]) == pytest.approx((p[0], r[0], f[0]))
    cal = calibration(a)
    assert cal["auroc_label"] == pytest.approx(skm.roc_auc_score(a.y[ok], p_yes))
    assert cal["auroc"] == pytest.approx(skm.roc_auc_score(a.correct[ok], a.conf[ok]))
    assert cal["brier"] == pytest.approx(skm.brier_score_loss(a.y[ok], p_yes))
    assert cal["log_loss"] == pytest.approx(skm.log_loss(a.y[ok], p_yes))


# Hand-computed tests


def test_rps_is_half_multiclass_brier_when_k_is_two():
    rng = np.random.default_rng(8)
    P = rng.dirichlet([1, 1], size=50)
    y = rng.integers(0, 2, size=50)
    assert ranked_probability_score(P, y) == pytest.approx(brier_multiclass(P, y) / 2)


def test_ece_mass_keeps_ties_in_one_bin_regardless_of_order():
    conf = [0.6, 0.6, 0.6, 0.8]
    correct = [1, 0, 0, 1]
    results = set()
    for perm in itertools.permutations(range(4)):
        r = ece_mass([conf[i] for i in perm], [correct[i] for i in perm], n_bins=2)
        assert [b["count"] for b in r["bins"]] == [3, 1]
        results.add(round(r["value"], 12))
    # bin 1: |1/3 - 0.6| * 3/4 = 0.2; bin 2: |1 - 0.8| * 1/4 = 0.05
    assert results == {0.25}


def test_ece_mass_without_ties_is_one_row_per_bin_when_n_below_bins():
    r = ece_mass([0.9, 0.7], [1, 0], n_bins=10)
    assert [b["count"] for b in r["bins"]] == [1, 1]
    assert r["value"] == pytest.approx((0.7 + 0.1) / 2)


def test_ece_mass_empty_and_invalid():
    assert ece_mass([], []) == {"value": None, "bins": []}
    with pytest.raises(ValueError):
        ece_mass([0.5], [1], n_bins=0)


# Regression: the 4-ticket example


def _tickets():
    queue = DecisionArrays.from_records(QUEUE, [
        {"doc": 1, "gold": 0, "probs": [0.77, 0.12, 0.11]},
        {"doc": 2, "gold": 1, "probs": [0.60, 0.30, 0.10]},
        {"doc": 3, "gold": 2, "probs": [0.05, 0.05, 0.90]},
        {"doc": 4, "gold": 0, "probs": None},
    ])
    urgent = DecisionArrays.from_records(URGENT, [
        {"doc": d, "gold": g, "probs": [1 - p, p]}
        for d, (p, g) in enumerate([(0.86, 1), (0.30, 0), (0.40, 1), (0.10, 0)], start=1)
    ])
    severity = DecisionArrays.from_records(SEVERITY, [
        {"doc": 1, "gold": 1, "probs": [0.20, 0.57, 0.23]},
        {"doc": 2, "gold": 0, "probs": [0.70, 0.25, 0.05]},
        {"doc": 3, "gold": 2, "probs": [0.05, 0.55, 0.40]},
        {"doc": 4, "gold": 1, "probs": [0.10, 0.30, 0.60]},
    ])
    return queue, urgent, severity


def test_four_tickets_queue():
    queue, _, _ = _tickets()
    c, cal = correctness(queue), calibration(queue)
    assert c["accuracy"] == pytest.approx(0.5)
    assert c["answered_accuracy"] == pytest.approx(2 / 3)
    assert c["coverage"] == {"ratio": 0.75, "ok": 3, "missing": 1}
    assert c["per_class"] == {
        "billing": {"precision": 0.5, "recall": 0.5, "f1": pytest.approx(0.5), "support": 2},
        "sales": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
        "retail": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
    }
    assert c["macro_f1"] == pytest.approx(0.5)
    assert c["confusion"] == {
        "billing": {"billing": 1, "(no answer)": 1},
        "sales": {"billing": 1},
        "retail": {"retail": 1},
    }
    assert cal["auroc"] == pytest.approx(1.0)
    assert cal["brier_multiclass"] == pytest.approx(0.31813, abs=1e-5)
    assert cal["nll"] == pytest.approx(0.52357, abs=1e-5)
    assert cal["ece_mass"] == pytest.approx(0.31)
    assert cal["coverage"]["missing"] == 1


def test_four_tickets_urgent():
    _, urgent, _ = _tickets()
    c, cal = correctness(urgent), calibration(urgent)
    assert c["accuracy"] == pytest.approx(0.75)
    assert c["precision"] == pytest.approx(1.0)
    assert c["recall"] == pytest.approx(0.5)
    assert c["f1"] == pytest.approx(2 / 3)
    assert c["threshold"] == 0.5
    assert cal["auroc"] == pytest.approx(1.0)
    assert cal["auroc_label"] == pytest.approx(1.0)
    assert cal["brier"] == pytest.approx(0.1199)
    assert cal["log_loss"] == pytest.approx(0.38229, abs=1e-5)
    assert cal["ece_mass"] == pytest.approx(0.285)


def test_four_tickets_severity():
    _, _, severity = _tickets()
    c, cal = correctness(severity), calibration(severity)
    assert c["accuracy"] == pytest.approx(0.5)
    assert c["qwk"] == pytest.approx(0.5)
    assert c["mae_argmax"] == pytest.approx(0.5)
    assert c["mae_expected"] == pytest.approx(0.3825)
    assert cal["rps"] == pytest.approx(0.11474, abs=1e-5)
    assert cal["auroc"] == pytest.approx(0.75)
    assert cal["ece_mass"] == pytest.approx(0.47)


def test_results_are_plain_python():
    for a in _tickets():
        _assert_plain(correctness(a))
        _assert_plain(calibration(a))
        _assert_plain(bootstrap_ci(a, correctness, n_boot=20))


# Edge cases


@pytest.mark.parametrize("spec", [QUEUE, URGENT, SEVERITY])
def test_zero_rows(spec):
    a = empty_arrays(spec)
    c, cal = correctness(a), calibration(a)
    assert c["accuracy"] is None and c["answered_accuracy"] is None
    assert c["coverage"] == {"ratio": None, "ok": 0, "missing": 0}
    assert cal["auroc"] is None and cal["ece_mass"] is None
    _assert_plain(c)
    _assert_plain(cal)
    assert bootstrap_ci(a, correctness, n_boot=5)["coverage"] == {
        "ok": {"low": None, "high": None},
        "missing": {"low": None, "high": None},
    }


@pytest.mark.parametrize("spec", [QUEUE, URGENT, SEVERITY])
def test_all_answers_missing(spec):
    a = DecisionArrays.from_records(spec, [{"doc": i, "gold": i % 2, "probs": None} for i in range(3)])
    c, cal = correctness(a), calibration(a)
    assert c["accuracy"] == 0.0
    assert c["answered_accuracy"] is None
    assert c["coverage"] == {"ratio": 0.0, "ok": 0, "missing": 3}
    for key in ("auroc", "ece_mass", "brier_multiclass", "nll", "auroc_label", "brier", "log_loss", "rps"):
        assert cal.get(key) is None
    _assert_plain(c)
    _assert_plain(cal)


def test_single_class_auroc_is_none():
    a = DecisionArrays.from_records(URGENT, [
        {"doc": 1, "gold": 1, "probs": [0.2, 0.8]},
        {"doc": 2, "gold": 1, "probs": [0.1, 0.9]},
    ])
    cal = calibration(a)
    assert cal["auroc"] is None  # every answer correct
    assert cal["auroc_label"] is None  # every gold yes


def test_choice_recall_with_no_support_is_zero():
    a = DecisionArrays.from_records(QUEUE, [{"doc": 1, "gold": 0, "probs": [0.1, 0.8, 0.1]}])
    assert correctness(a)["per_class"]["retail"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


# Bootstrap


def test_bootstrap_is_deterministic():
    a = _random_arrays(QUEUE, seed=9)
    first = bootstrap_ci(a, calibration, n_boot=200, seed=3)
    assert first == bootstrap_ci(a, calibration, n_boot=200, seed=3)
    assert first != bootstrap_ci(a, calibration, n_boot=200, seed=4)


def test_bootstrap_bounds_bracket_point_estimate():
    a = _random_arrays(SEVERITY, n=400, n_docs=100, seed=10)
    point = correctness(a)
    ci = bootstrap_ci(a, correctness, n_boot=300)
    for key in ("accuracy", "answered_accuracy", "qwk", "mae_argmax", "mae_expected"):
        assert ci[key]["low"] <= point[key] <= ci[key]["high"], key
        assert ci[key]["low"] < ci[key]["high"], key
    assert set(ci) == {"accuracy", "answered_accuracy", "coverage", "qwk", "mae_argmax", "mae_expected"}


def test_bootstrap_resamples_documents_not_rows():
    # Document 0 has three rows, document 1 has one.
    a = DecisionArrays.from_records(URGENT, [
        {"doc": 0, "gold": 1, "probs": [0.1, 0.9]},
        {"doc": 0, "gold": 1, "probs": [0.2, 0.8]},
        {"doc": 0, "gold": 0, "probs": [0.7, 0.3]},
        {"doc": 1, "gold": 1, "probs": [0.6, 0.4]},
    ])
    sizes = {0: 3, 1: 1}
    seen = []

    def metric(sub):
        seen.append(sub)
        return correctness(sub)

    bootstrap_ci(a, metric, n_boot=50)
    resamples = seen[1:]  # the first call is the point estimate
    assert len(resamples) == 50
    for sub in resamples:
        assert sub.n_total in (6, 4, 2)  # two draws of whole documents
        for doc, count in zip(*np.unique(sub.groups, return_counts=True)):
            assert count % sizes[int(doc)] == 0


def test_bootstrap_skips_none_and_non_numeric_leaves():
    a = DecisionArrays.from_records(QUEUE, [
        {"doc": 0, "gold": 0, "probs": [0.9, 0.05, 0.05]},
        {"doc": 1, "gold": 1, "probs": [0.8, 0.1, 0.1]},
    ])
    ci = bootstrap_ci(a, calibration, n_boot=50)
    assert "ece_mass_bins" not in ci
    # auroc is defined on the full run but None in resamples with one document.
    assert ci["auroc"]["low"] == ci["auroc"]["high"] == 1.0
