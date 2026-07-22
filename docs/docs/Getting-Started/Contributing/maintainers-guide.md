# Maintainer's Guide

Everything a maintainer needs that isn't in the
[contributor docs](README.md).
Contributors' docs cover how to write code for Stickler; this page covers how
to run the project.

## The job

- **Review and merge PRs** into `dev` (see review bar below)
- **Triage issues** — label, reproduce, close what's stale
- **Cut releases** — follow
  [RELEASING.md](https://github.com/awslabs/stickler/blob/main/RELEASING.md)
  step by step; it's the single source of truth for the release process
- **Keep CI honest** — a green build must actually mean something

## Review bar

Hold every PR (including your own) to this:

1. CI green — lint, tests, security scans. Don't merge on red.
2. Tests accompany behavior changes. A fix without a regression test will
   regress.
3. Public API changes need docs (docstrings + MkDocs page) and a
   `CHANGELOG.md` entry under `[Unreleased]`.
4. Conventional commit titles (`feat:`, `fix:`, `chore:`, ...) — release notes
   are grouped by them.
5. When in doubt about a design, ask for an issue/RFC first. Design discussion
   lives in GitHub issues, not PR threads.

## Branch model

`main` = released code. `dev` = integration branch and default target for all
PRs. Releases are `dev` → `main` squash merges.
[Issue #103](https://github.com/awslabs/stickler/issues/103) proposes retiring
`dev` for GitHub Flow — read it before making structural changes to branching.

## Architecture orientation

Read in this order:

1. [Architecture overview](architecture.md) — component map
2. `src/stickler/` README files — each package documents itself
3. [StructuredModel refactoring notes](https://github.com/awslabs/stickler/blob/main/docs/structured_model_REFACTORING.md)
   — the delegation pattern that decomposes `StructuredModel`, and how far
   that refactor got

## Known landmines

Things that will surprise you if nobody tells you:

- **`StructuredModel` is still ~1,500 lines.** The delegation refactor
  ([#133](https://github.com/awslabs/stickler/issues/133)) is unfinished;
  the refactoring notes above explain the target end state. Related cleanup
  issues: [#134](https://github.com/awslabs/stickler/issues/134),
  [#136](https://github.com/awslabs/stickler/issues/136),
  [#137](https://github.com/awslabs/stickler/issues/137).
- **Two parallel evaluation paths exist** — `trees/` (ANLSTree) and the
  `ComparisonEngine`. [#135](https://github.com/awslabs/stickler/issues/135)
  is the RFC to resolve this. Don't build new features on `trees/`.
- **Known performance TODO** in
  `src/stickler/structured_object_evaluator/models/field_comparator.py`:
  redundant nested-tree traversal on deeply nested models.
- **Install failures on older GCC** (RHEL/Amazon Linux): see
  [known issues](../known-issues.md).
- **PEP 604 unions (`X | None`) are inconsistently handled** in schema export
  and `from_json` — see [#160](https://github.com/awslabs/stickler/issues/160),
  [#161](https://github.com/awslabs/stickler/issues/161),
  [#162](https://github.com/awslabs/stickler/issues/162).
- **Publishing is not idempotent.** A published GitHub Release fires the PyPI
  publish workflow; a version can never be re-uploaded. Details and recovery
  steps in RELEASING.md.

## Infrastructure a maintainer must have access to

| What | Where | Notes |
|------|-------|-------|
| PyPI project `stickler-eval` | pypi.org + test.pypi.org | Published via OIDC trusted publishing tied to the `release`/`testpypi` GitHub environments — no API tokens to rotate |
| GitHub environments & rulesets | repo Settings | `release`, `testpypi` environments; branch protection on `main`/`dev` |
| Docs hosting | GitHub Pages (`gh-pages` branch) | Deployed automatically by `docs.yml` on push to `main` |
| Dependabot | `.github/dependabot.yml` | Weekly grouped updates |

## CI map

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `run_pytest.yaml` | push/PR | Test suite with `llm` + `bert` extras |
| `lint.yaml` | push/PR | Ruff |
| `security.yaml` + `security-pr-comment.yaml` | push/PR | Bandit + AWS ASH, posts summary comment |
| `docs.yml` | push to `main` | Builds and deploys MkDocs to GitHub Pages |
| `workflow.yml` | Release published | Builds and publishes to TestPyPI + PyPI |

## Where decisions live

- **Design discussions / RFCs**: GitHub issues (e.g.
  [#135](https://github.com/awslabs/stickler/issues/135))
- **Refactoring rationale**: the refactoring notes linked above
- **Release history**: `CHANGELOG.md` +
  [GitHub Releases](https://github.com/awslabs/stickler/releases)
- **AI-assistant conventions**: `AGENTS.md` files throughout the repo
