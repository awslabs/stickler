# GitHub Actions Workflows

CI/CD workflows for the Stickler project.

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `run_pytest.yaml` | push to `dev`/`main`, PR | Test suite on Python 3.12–3.14 with coverage (installs `--extra llm --extra bert`) |
| `lint.yaml` | push to `dev`/`main`, PR | Ruff lint (blocking) |
| `security.yaml` | push to `dev`/`main`, PR | Bandit + ASH security scans; uploads `security-reports` artifact |
| `security-pr-comment.yaml` | `workflow_run` after Security Scan | Posts/updates the ASH summary as a PR comment |
| `docs.yml` | push to `main` (src/docs paths) | Deploys MkDocs site to GitHub Pages |
| `workflow.yml` | release published | Builds and publishes to PyPI and TestPyPI (trusted publishing via OIDC) |

## Conventions

- **SHA pinning**: all `uses:` references (and git installs like ASH) are
  pinned to full 40-character commit SHAs with a trailing `# vX.Y.Z` comment
  (supply-chain hardening; mutable tags can be repointed). Dependabot
  (`.github/dependabot.yml`) bumps the SHA and comment together.
- **Locked dependencies**: all `uv sync` / `uv run` invocations use
  `--frozen` so CI resolves exactly what `uv.lock` specifies.
- **Least privilege**: workflows default to `permissions: contents: read`
  and grant extra scopes per job only where needed. Checkouts that never
  push use `persist-credentials: false` (only `docs.yml` keeps credentials,
  for `mkdocs gh-deploy`).
- **Concurrency + timeouts**: every job sets `timeout-minutes`; PR/push
  workflows use a `concurrency` group with `cancel-in-progress: true` so
  superseded runs are cancelled (deploy/release workflows set it to `false`
  so in-flight deploys are never interrupted).
- **Trigger dedup**: `push` triggers are filtered to `dev`/`main` so
  same-repo PR branches don't run every workflow twice.
