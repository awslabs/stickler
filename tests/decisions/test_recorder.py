"""record_systemone.py against a stubbed server: what it writes, and when it refuses to."""

import importlib.util
import io
import json
import pathlib
import urllib.error

import pytest

SCRIPT = pathlib.Path(__file__).parents[2] / "examples/notebooks/fixtures/systemone/record_systemone.py"


@pytest.fixture
def recorder():
    spec = importlib.util.spec_from_file_location("record_systemone", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(recorder, monkeypatch, tmp_path, reply):
    """Run the recorder with ``reply(body)`` standing in for the server."""
    monkeypatch.setattr(recorder, "post", lambda url, body, timeout, api_key: reply(body))
    code = recorder.main(["--out-dir", str(tmp_path), "--url", "https://secret-host.internal/v1?token=abc"])
    return code, sorted(p.name for p in tmp_path.iterdir())


def answer_all(body):
    return {"model": "m-1", "answers": {q: {"type": "noul", "noul": 0.7} for q in body["questions"]}}


def test_writes_model_and_version_but_not_the_endpoint(recorder, monkeypatch, tmp_path):
    code, files = run(recorder, monkeypatch, tmp_path, answer_all)
    assert (code, files) == (0, ["provenance.json", "responses.jsonl"])
    provenance = (tmp_path / "provenance.json").read_text()
    assert "secret-host" not in provenance and "token" not in provenance
    assert json.loads(provenance)["model_id"] == "m-1"


def test_a_refusal_is_recorded_as_missing(recorder, monkeypatch, tmp_path):
    calls = {"n": 0}

    def reply(body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 422, "Unprocessable", {}, io.BytesIO(b"cannot decide"))
        return answer_all(body)

    code, _ = run(recorder, monkeypatch, tmp_path, reply)
    assert code == 0
    first = json.loads((tmp_path / "responses.jsonl").read_text().splitlines()[0])
    assert first["response"] == {"answers": {}}
    assert first["refusal"]["http_status"] == 422
    assert json.loads((tmp_path / "provenance.json").read_text())["refused"] == 1


def test_an_empty_answer_fails_and_writes_nothing(recorder, monkeypatch, tmp_path):
    code, files = run(recorder, monkeypatch, tmp_path, lambda body: {"model": "m-1", "answers": {}})
    assert (code, files) == (3, [])


def test_a_run_with_no_answers_fails(recorder, monkeypatch, tmp_path):
    def refuse(body):
        raise urllib.error.HTTPError("u", 422, "Unprocessable", {}, io.BytesIO(b"no"))

    code, files = run(recorder, monkeypatch, tmp_path, refuse)
    assert (code, files) == (3, [])
