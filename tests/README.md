# Stickler Test Suite

Quick reference for running and writing tests.

## Quick Commands

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

# Run specific test
pytest tests/structured_object_evaluator/test_comparators.py::test_exact_match

# Run tests matching pattern
pytest tests/ -k "levenshtein"

# Run with coverage
coverage run -m pytest tests/
coverage report -m

# Run in parallel (faster)
pytest tests/ -n auto
```

## Directory Structure

```
tests/
├── structured_object_evaluator/   # Core evaluation tests
│   ├── test_structured_model.py           # StructuredModel tests
│   ├── test_comparators.py                # Comparator integration
│   ├── test_integration.py                # End-to-end tests
│   ├── test_edge_cases.py                 # Edge cases
│   ├── test_hungarian_*.py                # List matching
│   └── ...
├── common/
│   ├── comparators/               # Comparator unit tests
│   │   ├── test_exact.py
│   │   ├── test_numeric.py
│   │   └── test_comparators.py
│   └── algorithms/                # Algorithm tests
│       └── test_hungarian.py
└── reporting/
    └── html/                      # Report generation tests
```

## Writing Tests

### Basic Test Pattern

```python
from stickler.comparators.levenshtein import LevenshteinComparator

def test_exact_match():
    """Test exact string match returns 1.0."""
    comparator = LevenshteinComparator()
    assert comparator.compare("hello", "hello") == 1.0
```

### StructuredModel Test Pattern

```python
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator

class TestModel(StructuredModel):
    name: str = ComparableField(comparator=LevenshteinComparator())

def test_model_comparison():
    model1 = TestModel(name="John")
    model2 = TestModel(name="John")
    result = model1.compare_with(model2)
    assert result["overall_score"] == 1.0
```

## Naming Conventions

- Test files: `test_<feature>.py`
- Test classes: `Test<Feature>`
- Test functions: `test_<specific_behavior>`

## Full Documentation

See the [Testing Guide](../docs/docs/Contributing/testing-guide.md) for comprehensive testing best practices.
