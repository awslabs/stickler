# Testing Guide

This guide covers testing conventions, patterns, and best practices for contributing tests to the Stickler project.

## Overview

Stickler uses **pytest** as its testing framework with **coverage** for code coverage reporting. Tests run automatically on every push and pull request via GitHub Actions.

## Test Organization

### Directory Structure

```
tests/
├── structured_object_evaluator/   # Core evaluation tests (~80+ files)
│   ├── test_structured_model.py           # StructuredModel functionality
│   ├── test_comparators.py                # Comparator integration
│   ├── test_integration.py                # End-to-end flows
│   ├── test_edge_cases.py                 # Edge case handling
│   ├── test_hungarian_matching_*.py       # List matching tests
│   └── ...
├── common/
│   ├── comparators/               # Comparator unit tests
│   │   ├── test_exact.py
│   │   ├── test_numeric.py
│   │   ├── test_comparators.py
│   │   └── ...
│   └── algorithms/                # Algorithm tests
│       └── test_hungarian.py
└── reporting/
    └── html/                      # Report generation tests
        ├── test_html_reporter.py
        └── ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Test files | `test_<feature>.py` | `test_comparators.py` |
| Test classes | `Test<Feature>` | `TestLevenshteinComparator` |
| Test functions | `test_<specific_behavior>` | `test_exact_match_returns_one` |

## Test Patterns

### Pattern 1: Function-Based Tests (Simple Cases)

Use for simple, independent tests:

```python
from stickler.comparators.levenshtein import LevenshteinComparator

def test_exact_match():
    """Test exact string match returns 1.0."""
    comparator = LevenshteinComparator()
    assert comparator.compare("hello", "hello") == 1.0

def test_completely_different():
    """Test completely different strings return low score."""
    comparator = LevenshteinComparator()
    score = comparator.compare("hello", "world")
    assert 0.0 <= score < 0.5
```

### Pattern 2: Class-Based Tests (Related Test Groups)

Use when tests share setup or test a single component:

```python
import pytest
from stickler.comparators.levenshtein import LevenshteinComparator

class TestLevenshteinComparator:
    """Test cases for the LevenshteinComparator."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.comparator = LevenshteinComparator()

    def test_exact_match(self):
        """Test exact string match."""
        assert self.comparator.compare("hello", "hello") == 1.0

    def test_single_character_difference(self):
        """Test single character difference."""
        score = self.comparator.compare("hello", "hallo")
        assert 0.8 <= score < 1.0

    def test_empty_strings(self):
        """Test empty string comparison."""
        assert self.comparator.compare("", "") == 1.0
```

### Pattern 3: Custom Model Fixtures (StructuredModel Tests)

Define test models within test files for isolated testing:

```python
from typing import Optional
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator

# Define test model
class SimpleTestModel(StructuredModel):
    """Simple model for testing StructuredModel functionality."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=1.0,
    )

    age: Optional[int] = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=0.5,
    )


def test_structured_model_compare_with_exact_match():
    """Test compare_with with exact match."""
    model1 = SimpleTestModel(name="John Doe", age=30)
    model2 = SimpleTestModel(name="John Doe", age=30)

    result = model1.compare_with(model2)

    # Check result structure
    assert "field_scores" in result
    assert "overall_score" in result
    assert "all_fields_matched" in result

    # Check field scores
    assert result["field_scores"]["name"] == 1.0
    assert result["field_scores"]["age"] == 1.0
    assert result["overall_score"] == 1.0
    assert result["all_fields_matched"] is True
```

### Pattern 4: Parameterized Tests

Use `pytest.mark.parametrize` for testing multiple inputs:

```python
import pytest
from stickler.comparators.levenshtein import LevenshteinComparator

@pytest.mark.parametrize("str1,str2,expected_min,expected_max", [
    ("hello", "hello", 1.0, 1.0),       # Exact match
    ("hello", "helo", 0.7, 0.95),       # Minor typo
    ("hello", "world", 0.0, 0.4),       # Different words
    ("", "", 1.0, 1.0),                 # Empty strings
])
def test_levenshtein_score_ranges(str1, str2, expected_min, expected_max):
    """Test Levenshtein comparator returns scores in expected ranges."""
    comparator = LevenshteinComparator()
    score = comparator.compare(str1, str2)
    assert expected_min <= score <= expected_max
```

### Pattern 5: Weighted Score Validation

Test the weighted average calculation:

```python
def test_weighted_score_calculation():
    """Test that overall_score uses correct weighted average."""
    model1 = SimpleTestModel(name="John Doe", age=30)
    model2 = SimpleTestModel(name="John Smith", age=30)

    result = model1.compare_with(model2)

    # name has weight 1.0, age has weight 0.5
    name_score = result["field_scores"]["name"]
    age_score = result["field_scores"]["age"]
    expected_score = (name_score * 1.0 + age_score * 0.5) / 1.5

    assert abs(result["overall_score"] - expected_score) < 0.001
```

## Common Assertions

### Result Structure Validation

```python
def test_result_structure():
    """Test that compare_with returns expected structure."""
    result = model1.compare_with(model2)

    # Required keys
    assert "field_scores" in result
    assert "overall_score" in result
    assert "all_fields_matched" in result

    # Field scores should be present
    assert "name" in result["field_scores"]
    assert "age" in result["field_scores"]
```

### Score Range Validation

```python
# Exact match
assert result["overall_score"] == 1.0

# Partial match
assert 0.0 < result["overall_score"] < 1.0

# No match (exact comparator with different values)
assert result["field_scores"]["age"] == 0.0
```

### Floating Point Comparison

Use `pytest.approx` for floating-point comparisons:

```python
import pytest

assert result["overall_score"] == pytest.approx(0.85, abs=1e-4)
assert result["field_scores"]["name"] == pytest.approx(0.923, rel=1e-3)
```

### Exception Testing

```python
import pytest

def test_type_error_on_invalid_input():
    """Test that invalid types raise TypeError."""
    comparator = NumericComparator()
    with pytest.raises(TypeError):
        comparator.compare("not a number", 42)
```

### Skip Conditions

```python
import pytest

# Skip if optional dependency not available
@pytest.mark.skipif(
    not SEMANTIC_AVAILABLE,
    reason="SemanticComparator requires sentence-transformers"
)
def test_semantic_comparison():
    """Test semantic similarity comparison."""
    ...

# Skip on specific platforms
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Test not supported on Windows"
)
def test_unix_specific():
    ...
```

## Test Categories

### 1. Unit Tests

Test single components in isolation.

- **Location:** `tests/common/comparators/`, `tests/common/algorithms/`
- **Focus:** Individual comparator methods, algorithm correctness
- **Example:** Testing `LevenshteinComparator.compare()` with various inputs

### 2. Integration Tests

Test component interactions.

- **Location:** `tests/structured_object_evaluator/test_integration.py`
- **Focus:** End-to-end model comparison, evaluator workflows
- **Example:** Testing complete model comparison with nested objects

### 3. Edge Case Tests

Test boundary conditions and unusual inputs.

- **Location:** `tests/structured_object_evaluator/test_edge_cases.py`
- **Focus:** None values, empty strings, type mismatches, missing fields
- **Example:** Testing comparison when one field is None

### 4. Regression Tests

Prevent bug recurrence.

- **Naming:** Named descriptively referencing the issue
- **Documentation:** Include issue reference in docstring
- **Example:** `test_aggregate_contact_issue.py`

```python
def test_aggregate_contact_issue():
    """Test fix for aggregate contact issue.

    Regression test for GitHub issue #123.
    Ensures that aggregate calculations handle
    contact fields correctly.
    """
    ...
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with print statements shown
pytest tests/ -v -s

# Run specific module
pytest tests/structured_object_evaluator/

# Run specific test file
pytest tests/structured_object_evaluator/test_comparators.py

# Run specific test class
pytest tests/structured_object_evaluator/test_comparators.py::TestLevenshteinComparator

# Run specific test method
pytest tests/structured_object_evaluator/test_comparators.py::TestLevenshteinComparator::test_exact_match

# Run tests matching pattern
pytest tests/ -k "levenshtein"

# Run tests excluding pattern
pytest tests/ -k "not slow"
```

### Coverage Commands

```bash
# Run with coverage
coverage run -m pytest tests/

# Generate terminal report
coverage report -m

# Generate HTML report
coverage html
open htmlcov/index.html

# Combined command
coverage run -m pytest tests/ && coverage report -m
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Specify number of workers
pytest tests/ -n 4
```

### Debugging

```bash
# Stop on first failure
pytest tests/ -x

# Enter debugger on failure
pytest tests/ --pdb

# Show local variables in traceback
pytest tests/ -l

# More detailed traceback
pytest tests/ --tb=long
```

## CI Integration

Tests run automatically on every push and pull request via GitHub Actions.

### GitHub Actions Workflows

1. **run_pytest.yaml** - Runs pytest with coverage
   - Triggers on push and pull_request
   - Uses Python 3.12
   - Generates coverage report

2. **lint.yaml** - Runs Ruff linter
   - Checks code style
   - Non-blocking (continue-on-error)

3. **security.yaml** - Security scanning
   - Runs Bandit for Python security issues
   - Runs ASH (AWS Security Helper)

### CI Requirements

Before submitting a PR, ensure:

- [ ] All tests pass locally (`pytest tests/`)
- [ ] New code has corresponding tests
- [ ] Coverage report generated (`coverage run -m pytest tests/`)

## Best Practices

### Writing Good Tests

1. **Name tests descriptively** - Test names should describe the behavior being tested
   ```python
   # Good
   def test_levenshtein_returns_zero_for_completely_different_strings():

   # Bad
   def test_levenshtein_1():
   ```

2. **One assertion focus** - Each test should focus on one specific behavior
   ```python
   # Good - focused test
   def test_exact_match_returns_one():
       assert comparator.compare("hello", "hello") == 1.0

   # Avoid - testing too many things
   def test_comparator():
       assert comparator.compare("hello", "hello") == 1.0
       assert comparator.compare("hello", "world") < 0.5
       assert comparator.compare("", "") == 1.0
   ```

3. **Use fixtures** - Leverage `setup_method` and pytest fixtures for shared setup

4. **Document complex tests** - Add docstrings explaining test purpose and expected behavior

5. **Test edge cases** - Include tests for None, empty, boundary values

6. **Avoid test interdependence** - Each test should be independent and runnable in isolation

7. **Clean up resources** - Use fixtures with teardown when tests create files or resources

### Test Documentation

```python
def test_hungarian_with_structured_models():
    """Test Hungarian algorithm with StructuredModel instances.

    This test validates:
    - Optimal pairing of items across lists
    - Correct TP/FP/FN classification
    - Handling of different list orderings

    The test uses two invoices with reordered line items
    to verify order-independent matching.
    """
    ...
```

### What to Test

When adding new functionality, test:

1. **Happy path** - Normal expected usage
2. **Edge cases** - Empty inputs, None values, boundary conditions
3. **Error cases** - Invalid inputs that should raise exceptions
4. **Integration** - How the component works with others

### What Not to Test

- Third-party library internals
- Trivial getters/setters with no logic
- Framework behavior (Pydantic validation, etc.)

## Troubleshooting

### Common Issues

**Tests not discovered:**
```bash
# Ensure test files follow naming convention
ls tests/  # Should see test_*.py files

# Check pytest configuration in pyproject.toml
```

**Import errors:**
```bash
# Ensure package is installed in development mode
pip install -e ".[dev]"
```

**Coverage not including files:**
```bash
# Run from project root
cd /path/to/stickler
coverage run -m pytest tests/
```

### Getting Help

- Check existing tests for patterns
- Review [pytest documentation](https://docs.pytest.org/)
