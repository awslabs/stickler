# GitHub Actions Workflows

CI/CD workflows for the Stickler project.

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `run_pytest.yaml` | PR, push to `dev`/`main` | Test suite with coverage (installs `--extra llm --extra bert`) |
| `lint.yaml` | PR, push to `dev`/`main` | Ruff lint (non-blocking via `continue-on-error`) |
| `security.yaml` | PR, push to `dev`/`main` | Bandit + ASH security scans; uploads `security-reports` artifact |
| `security-pr-comment.yaml` | `workflow_run` after Security Scan | Posts/updates the ASH summary as a PR comment |
| `docs.yml` | push to `main` (src/docs paths) | Deploys MkDocs site to GitHub Pages |
| `workflow.yml` | release published | Builds and publishes to PyPI and TestPyPI (trusted publishing via OIDC) |

## Conventions

- **SHA pinning**: all `uses:` references are pinned to full 40-character
  commit SHAs with a trailing `# vX.Y.Z` comment (supply-chain hardening;
  mutable tags can be repointed). Dependabot (`.github/dependabot.yml`)
  bumps the SHA and comment together.
- **Locked dependencies**: all `uv sync` / `uv run` invocations use
  `--frozen` so CI resolves exactly what `uv.lock` specifies.
- **Least privilege**: workflows default to `permissions: contents: read`
  and grant extra scopes per job only where needed.
- **One run per pull request**: the three PR-validating workflows limit `push`
  to `dev` and `main`, because a branch with an open PR fired both events and
  validated the same commit twice. `run_pytest.yaml` and `lint.yaml` also carry
  a `concurrency` group so a superseded PR head is cancelled; a push gets a
  group of its own keyed on `github.run_id`, since a group holds one running
  and one pending run and a third push would cancel the pending one whatever
  `cancel-in-progress` says. `security.yaml` is deliberately left uncancelled:
  `security-pr-comment.yaml` wakes on its completion including a cancellation,
  and fails rather than skips when the `security-reports` artifact is absent.
