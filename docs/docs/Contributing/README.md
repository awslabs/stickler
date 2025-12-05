---
title: Overview
---

# Contributing to Stickler

Welcome to the Stickler contributor documentation. This section provides everything you need to contribute code, tests, and documentation to the project.

## Getting Started

Before contributing, set up your development environment:

1. **[Development Setup](development-setup.md)** - Configure your local environment
2. Review the [Testing Guide](testing-guide.md) to understand testing conventions
3. Read the [Code Style Guide](code-style.md) for coding standards

## Guides

| Guide | Description |
|-------|-------------|
| [Testing Guide](testing-guide.md) | Test patterns, coverage, and best practices |
| [Code Style](code-style.md) | Naming conventions, linting, type hints |
| [Pull Request Guide](pull-request-guide.md) | Branch workflow, commit messages, PR process |

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| Run all tests | `pytest tests/` |
| Run with coverage | `coverage run -m pytest tests/ && coverage report` |
| Run specific module tests | `pytest tests/structured_object_evaluator/` |
| Run single test | `pytest tests/path/test_file.py::test_name` |
| Lint check | `ruff check .` |
| Lint fix | `ruff check --fix .` |

### Development Workflow

1. Fork the repository
2. Create a branch from `dev` (not `main`)
3. Make your changes
4. Run tests and linting
5. Submit PR to `dev` branch

### Commit Message Format

Refer to [this guide](https://www.conventionalcommits.org/en/v1.0.0/#examples) for clear conventional commit guidelines

```
type: brief description
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples:**
```
feat: add NumericComparator tolerance parameter
fix: handle None values in LevenshteinComparator
docs: add testing best practices guide
test: add edge case tests for empty lists
```

## Project Structure

```
stickler/
├── src/stickler/
│   ├── structured_object_evaluator/   # Core evaluation engine
│   ├── comparators/                   # Comparison algorithms
│   ├── algorithms/                    # Matching algorithms
│   ├── utils/                         # Shared utilities
│   └── reporting/                     # Result formatting
├── tests/
│   ├── structured_object_evaluator/   # Core evaluation tests
│   ├── common/                        # Comparator and algorithm tests
│   └── reporting/                     # Report generation tests
├── docs/                              # Documentation (MkDocs)
└── examples/                          # Usage examples
```

## Getting Help

- **Issues:** Report bugs or suggest features via [GitHub Issues](https://github.com/awslabs/stickler/issues)

## Code of Conduct

This project follows the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
