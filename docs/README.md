# Stickler Documentation Site

## Overview

The Stickler documentation site is built using [MkDocs](https://www.mkdocs.org/) with the Material theme. The site structure includes conceptual documentation, practical guides, and auto-generated API reference from Python docstrings using mkdocstrings.

Documentation source files are in `docs/` (Markdown), configuration is in `mkdocs.yml`, and the built site outputs to `site/`. Navigation is automatically generated from the directory structure via the awesome-nav plugin.

## Prerequisites

- Python 3.12 or higher

## Installation

Install documentation dependencies:

```bash
uv sync --group docs --frozen
```

> **Using pip + venv?** Run `pip install -e ".[dev]"` — docs dependencies are included. Then run `mkdocs` commands directly without the `uv run` prefix.

This installs MkDocs, Material theme, mkdocstrings for API docs, and the awesome-nav plugin for automatic navigation generation.

## Development

Start the local development server with live reload:

```bash
uv run mkdocs serve --livereload
```

The site will be available at `http://127.0.0.1:8000`. Changes to Markdown files or `mkdocs.yml` will automatically reload the browser.

### Adding Content

- Add new pages as Markdown files in `docs/docs/`
- Navigation is auto-generated from directory structure via the awesome-nav plugin
- Use `README.md` files for section index pages
- API documentation is generated from Python docstrings in `src/stickler/`

### Testing

Verify the site builds without errors:

```bash
uv run mkdocs build
```

This generates the static site in `site/` and reports link diagnostics. An ordinary
build does not fail on warnings; use the checker below to enforce them. Link
validation levels are shared in `mkdocs.yml`, so `mkdocs serve` shows them too.

To run the pull-request check from the repository root:

```bash
uv run --frozen python docs/check_links.py
uv run --frozen pytest tests/test_docs_build.py
```

The checker fails on logged MkDocs warnings and errors, including awesome-nav
failures, orphan pages, missing pages/images/anchors, broken API references, and
unrecognized or absolute links. It tolerates only the nine existing griffe/autorefs
warnings identified by logger, message and source location in `check_links.py`.
They remain visible; changed messages/locations and all errors fail. Python
warnings such as `DeprecationWarning` are not routed into the gate. External URLs
are not fetched. Run `make check` from this directory for the same gate.
The read-only `docs-check.yml` workflow
runs for documentation, source, dependency, or checker-workflow changes and never
deploys the site. Its `uv run --only-group docs` installs the docs group without
installing the project; the regular test workflow runs
the checker's tests.

## Deployment

The site is deployed to GitHub Pages automatically via GitHub Actions (`.github/workflows/docs.yml`) on pushes to `main` that modify files in `src/` or `docs/`. This ensures API reference docs stay current when Python source code changes. To deploy manually:

```bash
uv run mkdocs gh-deploy --force
```

This builds the site and pushes it to the `gh-pages` branch using MkDocs' built-in deployment command.

### GitHub Actions Automation

Automatic deployment is configured in `.github/workflows/docs.yml`. The workflow:
- Triggers on pushes to `main` that modify files in `src/`, `docs/`, or the workflow itself
- Installs the stickler package via `uv sync --group docs` so mkdocstrings can import modules and read docstrings
- Deploys to GitHub Pages using `uv run mkdocs gh-deploy --force`

---

> **Note**: A `Makefile` is provided for convenience with targets: `install`, `docs`, `build`, `check`, `deploy`, and `clean`.
