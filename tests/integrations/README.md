# Strands Evals integration contract

strands-agents-evals scores structured output with stickler:
`StructuredOutputSimilarity` and `StructuredOutputReport`, in
`strands_evals/evaluators/deterministic/structured_output.py` (1.6.0). Their
suite pins a released stickler, so only these tests fail on the commit that
breaks the integration (#365).

The evaluator reads `EvalResult.raw` with `.get()` defaults, so a renamed key
makes every case pass instead of raising. Tests assert that keys exist.

| File | Pins | Needs strands |
| --- | --- | --- |
| `conftest.py` | Fails every test here unless `stickler` is imported from this checkout | no |
| `strands_corpus.py` | Models, cases, and `strands_test_pass`, a verbatim copy of 1.6.0's `_passed` | no |
| `test_strands_evals_contract.py` | Import paths, `eval_for` kwargs, `evaluate`/`explain` shapes, every `EvalResult` attribute, `raw["confusion_matrix"]["overall"]` counts, `matched`, the guard, `aggregate_from_comparisons` | no |
| `test_strands_evals_snapshot.py` + `strands_evals_snapshot.json` | Exact scores, verdicts and aggregate metrics for typical shapes | no |
| `test_strands_evals_e2e.py` | The real evaluator through `Experiment`, `per_case()`, `metrics()`, `explain()`, JSON round trip | yes, skipped otherwise |

## Running

```bash
pytest tests/integrations                   # e2e skips without strands_evals

# e2e: install strands-agents-evals WITHOUT the [stickler] extra, which would
# pull a PyPI stickler-eval over this checkout. Every test fails if it did.
# Same command as CI: -c keeps uv.lock's versions; the cutoff lets the pin
# through the [tool.uv] cooldown.
uv export --frozen --no-hashes --no-emit-project -o /tmp/uv-lock.txt
uv pip install --python .venv -c /tmp/uv-lock.txt \
  --exclude-newer-package "strands-agents-evals=2026-10-15T00:00:00Z" \
  "strands-agents-evals==1.6.0"
uv run --no-sync pytest tests/integrations
```

CI runs the e2e in the `strands-evals` job of `.github/workflows/run_pytest.yaml`.

## Changing the snapshot

A snapshot change changes what Strands Evals users see. It needs a CHANGELOG
entry and an e2e run against strands-agents-evals before merging. Regenerate,
then review the JSON diff:

```bash
STICKLER_UPDATE_STRANDS_SNAPSHOT=1 pytest tests/integrations/test_strands_evals_snapshot.py
```

Keep shapes an open change is deliberately moving out of the snapshot with
`snapshot=False` in `strands_corpus.py`; they still run through the contract.

## Bumping strands-agents-evals

Re-read `structured_output.py` in the new wheel, update the line references and
`strands_test_pass` if `_passed` changed, then bump the pin in the workflow and
move its `--exclude-newer-package` cutoff to just after the new release's upload
time, here too. The cutoff replaces the cooldown for strands-agents-evals, so a
pin newer than the cutoff never resolves. The e2e checks the copied guard
against the real evaluator on every case.
