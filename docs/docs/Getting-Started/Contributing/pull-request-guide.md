# Pull Request Guide

This guide covers the branch strategy, commit conventions, and pull request process for contributing to Stickler.

> **Using pip + venv?** All commands below use `uv run`. If you installed with `pip install -e ".[dev]"`, run tools directly (e.g., `pytest` instead of `uv run pytest`).

## Branch Strategy

### Main Branches

| Branch | Purpose | Direct Commits |
|--------|---------|----------------|
| `main` | Stable releases only | No - protected via PRs from `dev` only |
| `dev` | Integration branch | No - via PRs only |

### Feature Branches

Create feature branches from `dev` using these prefixes:

| Prefix | Use Case | Example |
|--------|----------|---------|
| `feature/` | New features | `feature/add-semantic-comparator` |
| `bugfix/` | Bug fixes | `bugfix/fix-null-handling` |
| `docs/` | Documentation | `docs/testing-guide` |
| `test/` | Test additions | `test/edge-cases-coverage` |
| `refactor/` | Code restructuring | `refactor/simplify-evaluator` |

### Branch Naming Guidelines

- Use lowercase with hyphens: `feature/add-new-comparator`
- Be descriptive but concise: `bugfix/fix-empty-list-crash`
- Reference issue numbers when applicable: `bugfix/123-null-pointer`

## Commit Messages

### Format

```
type: brief description
```

Keep the first line under 72 characters. Add a blank line and body for longer explanations.

### Commit Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat: add NumericComparator tolerance parameter` |
| `fix` | Bug fix | `fix: handle None values in LevenshteinComparator` |
| `docs` | Documentation | `docs: add testing best practices guide` |
| `style` | Formatting (no code change) | `style: fix indentation in comparators module` |
| `refactor` | Code restructuring | `refactor: simplify field comparison logic` |
| `test` | Adding/updating tests | `test: add edge case tests for empty lists` |
| `chore` | Maintenance tasks | `chore: update pytest dependency version` |

### Examples

**Good commit messages:**

```
feat: add weighted scoring to StructuredModel comparison

Add weight parameter to ComparableField that affects the overall
score calculation. Fields with higher weights have more influence
on the final score.

Closes #42
```

```
fix: handle None values in Levenshtein comparison

Previously, comparing None with a string would raise TypeError.
Now returns 0.0 similarity score when either value is None.
```

```
docs: add testing guide with pytest examples
```

```
test: add parameterized tests for numeric comparator
```

**Avoid these patterns:**

```
# Too vague
fixed bug
updated code
changes

# Not following format
Added new feature
FIX: bug in comparator

# Too long first line
feat: add a new parameter to the NumericComparator that allows users to specify tolerance
```

### Referencing Issues

Reference related issues in commit messages:

```
fix: resolve crash on empty input

Fixes #123
```

```
feat: add fuzzy string matching

Implements #45
Related to #42
```

## Pull Request Process

### Before Creating a PR

1. **Sync with dev branch:**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout your-feature-branch
   git rebase dev
   ```

2. **Run tests locally:**
   ```bash
   uv run pytest tests/ -v
   ```

3. **Run linting:**
   ```bash
   uv run ruff check .
   ```

4. **Check coverage (optional):**
   ```bash
   uv run coverage run -m pytest tests/ && uv run coverage report
   ```

### Creating the PR

1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open PR on GitHub targeting the `dev` branch

3. Fill out the PR template

4. Request review from maintainers

### PR Checklist

Before requesting review, ensure:

- [ ] Branch created from `dev`
- [ ] PR targets `dev` branch (not `main`)
- [ ] Code follows [style guide](code-style.md)
- [ ] Tests added for new functionality
- [ ] All tests pass locally
- [ ] Documentation updated if needed
- [ ] PR description explains changes
- [ ] Related issues referenced

## Code Review

### What Reviewers Look For

- **Correctness:** Does the code do what it's supposed to?
- **Tests:** Are there adequate tests for new functionality?
- **Style:** Does it follow project conventions?
- **Documentation:** Are public APIs documented?
- **Performance:** Any obvious performance issues?
- **Security:** Any security concerns?

### Responding to Feedback

- Address all comments before requesting re-review
- Use "Resolve conversation" when changes are made
- Ask for clarification if feedback is unclear
- Be open to suggestions - reviewers have context you may not

### Making Changes

After review feedback:

1. Make requested changes
2. Commit with clear message referencing the feedback
3. Push to the same branch (PR updates automatically)
4. Re-request review when ready

```bash
# After making changes
git add .
git commit -m "fix: address review feedback - improve error message"
git push origin feature/your-feature-name
```

## CI/CD Pipeline

### Automated Checks

Every PR triggers these GitHub Actions:

| Workflow | Purpose | Blocking |
|----------|---------|----------|
| `run_pytest.yaml` | Run tests with coverage | Yes |
| `lint.yaml` | Ruff linting | No (warnings shown) |
| `security.yaml` | Bandit + ASH security scan | Yes |

### Handling CI Failures

**Test failures:**
```bash
# Run tests locally to reproduce
uv run pytest tests/ -v

# Run specific failing test
uv run pytest tests/path/test_file.py::test_name -v
```

**Linting issues:**
```bash
# Check issues
uv run ruff check .

# Auto-fix where possible
uv run ruff check --fix .
```

**Security issues:**
- Review the security scan output
- Fix identified issues before merging
- Ask maintainers if unsure about findings

## After Merge

### Cleanup

Delete your feature branch after merge:

```bash
# Delete local branch
git branch -d feature/your-feature-name

# Delete remote branch (if not auto-deleted)
git push origin --delete feature/your-feature-name
```

### Sync Your Fork

Keep your fork up to date:

```bash
git checkout dev
git pull upstream dev
git push origin dev
```

## Best Practices

1. **Keep PRs focused** - One feature/fix per PR
2. **Keep PRs small** - Easier to review and less risk
3. **Write clear descriptions** - Help reviewers understand context
4. **Respond promptly** - Keep the review process moving
5. **Test thoroughly** - Don't rely solely on CI
6. **Update docs** - If your change affects user-facing behavior

## Getting Help

- **Stuck on implementation:** Ask in the PR comments
- **CI issues:** Check the workflow logs first, then ask
