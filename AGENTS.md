# General Guidance for Agentic Coding Assistants

## Documentation
- When writing documentation, always try to say more with less.
- Documentation should be minimal, and only say what needs to be said to communicate how to work with and extend the system.

## MKDocs Documentation Project
- An MKDocs documentation project exists at [docs/](./docs/). More information can be found in the [docs/README.md](./docs/README.md) file.

## Testing
- Test documentation and guidelines can be found in [tests/README.md](./tests/README.md).

## Measuring behaviour by hand
- **Print which stickler you imported before you trust a number.** CI is not the risk: it runs `uv sync --frozen`, so its `import stickler` is always the checkout. The risk is a shell, a notebook, or a review where it is not.

  ```python
  import stickler, inspect; print(inspect.getfile(stickler))
  ```

  Or pin it: `PYTHONPATH=/path/to/checkout/src python ...`, or `uv run --project /path/to/checkout`.
- **Check the path, not `__version__`.** `dev` and the published release both report `0.7.0` until the 1.0 bump lands, so the version string cannot tell the two apart. The file path can.
- `pip install stickler-eval` currently gives **0.7.0**, which is weeks behind `dev` and pre-1.0. This repo is also routinely checked out a dozen times at once, so an editable install can point at a different worktree than the one you are reading.
- Known 0.7.0 vs `dev` divergences: schema-path comparator defaults (`format`/`enum`/`const` are read on `dev`, ignored in 0.7.0), the built-in comparator count (13 vs 11), extension-typo rejection (raises on `dev`, dropped silently in 0.7.0), and `EvalResult.matched` (`overall_score >= match_threshold` on `dev`, `all_fields_matched` in 0.7.0). Not exhaustive, and it grows until 1.0 publishes.
- A review that cites measured numbers should say which tree produced them. Measuring the wrong build produces internally consistent, confidently wrong conclusions, and it has twice sent a PR author to delete correct work. See [#342](https://github.com/awslabs/stickler/issues/342).

## Dependencies and `uv.lock`
- `uv.lock` is committed deliberately. Do not gitignore or delete it: every CI workflow runs `uv sync --frozen`, which fails immediately if the lockfile is absent.
- It does not constrain consumers of the published package. Installers resolve the `dependencies` ranges in `pyproject.toml`, which is what the wheel metadata carries; the lockfile is uv-specific and governs only this repo's dev and CI environments.
- It also carries the dependency cooldown (`exclude-newer`, `exclude-newer-span`) that the semgrep supply-chain rules check for, so removing it regresses the security scan.
- uv applies the cooldown only when resolving, from `[tool.uv] exclude-newer = "7 days"`; the lock's `exclude-newer-span = "P7D"` records it, and its absolute `exclude-newer` is unused when a span is present, so newer uv's `0001-01-01` placeholder is harmless. The `cooldown` job in `security.yaml` fails if either setting is weakened or a root `uv.toml` overrides them. It also fails on a `[tool.uv.pip]` override. Dependabot regenerates `uv.lock` with uv, so its PRs follow the same cooldown. The job does not age-check individual lock entries, so review `uv.lock` changes in PRs from people.
- **Emergency CVE patch inside the window:** add a dated exemption in `[tool.uv]`, with the advisory link in a comment above it: `exclude-newer-package = { pkg = "2026-11-11T12:00:00Z" }`. Use a full `YYYY-MM-DDTHH:MM:SSZ` time at or after the fix's upload, never `false` or a span (the `cooldown` job rejects both). Run `uv lock --upgrade-package pkg`, then restore the markers (next bullet). After 7 days the job warns: remove both the pyproject entry and the lock's `[options.exclude-newer-package]` table, or `uv lock --check` fails.
- **Prefer hand-editing the lines you need over running `uv lock`.** In this repo `uv lock` rewrites `exclude-newer` to `0001-01-01` (harmless, above) and strips platform markers from the CUDA and nvidia entries. If you do run it, diff the result and restore the markers. Verify any edit with `uv lock --check`, which is the authoritative consistency check.
- The lockfile records this package's own `version`, so a release bump must update it alongside `pyproject.toml` and `src/stickler/__init__.py`. See [RELEASING.md](./RELEASING.md).

## README.md 
- In addition to MKDocs, The project strives to maintain developer documentation distributed in README.md files throughout the codebase. This documentation exists to help human and AI coding assistants when working with the codebase.
- When creating directories or working in a directory that does not have a README.md create a README.md file and document the final state of the code and logic that's in the directory. Do this in language that an AI coding assistant trying to understand the implementation and codebas would understand.
- When working in a directory, always look for a README.md and/or AGENTS.md file for important context about the directory/code contained within.