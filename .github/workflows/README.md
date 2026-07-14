# GitHub Actions Workflows

CI/CD workflows for the Stickler project.

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `run_pytest.yaml` | push, PR | Test suite with coverage (installs `--extra llm --extra bert`) |
| `lint.yaml` | push, PR | Ruff lint (non-blocking via `continue-on-error`) |
| `security.yaml` | push, PR | Bandit + ASH security scans; uploads `security-reports` artifact |
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
