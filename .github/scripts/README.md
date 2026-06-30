# CI helper scripts

This directory holds small, stdlib-only Python helpers invoked by the
workflows under [`.github/workflows/`](../workflows/). Anything more
complex (multi-file packages, third-party deps) belongs in the project
proper, not here.

## Inventory

| Script | Used by | Purpose |
|---|---|---|
| [`examples_summary.py`](./examples_summary.py) | [`examples.yaml`](../workflows/examples.yaml) | Render a JUnit XML report (from `pytest --junitxml=…`) as a Markdown summary for `$GITHUB_STEP_SUMMARY`. |

## Conventions

- **Stdlib only** by default. The one current exception is
  `examples_summary.py`, which uses
  [`defusedxml`](https://pypi.org/project/defusedxml/) for XML parsing so
  the AWS Security Helper (ASH) bot stays quiet. `defusedxml` is declared
  in the project's `test` dependency group, which the `examples`
  workflow installs.
- **`main(argv)` returns int.** Scripts expose a `main(argv: list[str]) -> int`
  entry point and call `sys.exit(main(sys.argv))` at module level so
  they're trivially unit-testable.
- **Never mask the underlying CI failure.** Summary or post-processing
  scripts should exit 0 on their own internal errors and emit a clearly
  marked fallback to the summary stream, so the workflow's exit status
  reflects only the actual job (pytest, lint, etc.) and not a bug in the
  helper.
- **Unit tests live under `tests/`.** See
  [`tests/test_examples_summary.py`](../../tests/test_examples_summary.py)
  for the pattern: load the script with `importlib.util` (because it
  lives outside `src/`) and drive `main()` against hand-crafted JUnit
  fixtures captured from real CI artifacts.

## Adding a new helper

1. Drop a single-file, stdlib-only Python script in this directory.
2. Reference it from the workflow that needs it via
   `uv run python .github/scripts/<name>.py …`.
3. Add a row to the inventory table above.
4. Add a `tests/test_<name>.py` that loads the script via `importlib.util`
   and exercises `main()` against representative fixtures.
