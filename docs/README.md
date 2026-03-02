# Stickler Documentation Site

## Overview

The Stickler documentation site is built using [MkDocs](https://www.mkdocs.org/) with the Material theme. The site structure includes conceptual documentation, practical guides, and auto-generated API reference from Python docstrings using mkdocstrings.

Documentation source files are in `docs/` (Markdown), configuration is in `mkdocs.yml`, and the built site outputs to `site/`. Navigation is automatically generated from the directory structure via the awesome-nav plugin.

## Prerequisites

- Python 3.12 or higher

## Installation

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

This installs MkDocs, Material theme, mkdocstrings for API docs, and the awesome-nav plugin for automatic navigation generation.

## Development

Start the local development server with live reload:

```bash
mkdocs serve --livereload
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
mkdocs build
```

This generates the static site in `site/` and validates all internal links and references.

## Deployment

The site is deployed to GitHub Pages automatically via GitHub Actions (`.github/workflows/docs.yml`) on pushes to `main` that modify files in `src/` or `docs/`. This ensures API reference docs stay current when Python source code changes. To deploy manually:

```bash
mkdocs gh-deploy --force
```

This builds the site and pushes it to the `gh-pages` branch using MkDocs' built-in deployment command.

### GitHub Actions Automation

Automatic deployment is configured in `.github/workflows/docs.yml`. The workflow:
- Triggers on pushes to `main` that modify files in `src/`, `docs/`, or the workflow itself
- Installs the stickler package (`pip install -e .`) so mkdocstrings can import modules and read docstrings
- Installs documentation dependencies from `docs/requirements.txt`
- Deploys to GitHub Pages using `mkdocs gh-deploy --force`

---

> **Note**: A `Makefile` is provided for convenience with targets: `install`, `docs`, `build`, `deploy`, and `clean`.