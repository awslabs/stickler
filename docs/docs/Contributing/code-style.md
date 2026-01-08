# Code Style Guide

This guide covers coding conventions and standards for the Stickler project.

## Overview

Stickler follows Python best practices with **Ruff** for linting. The project uses modern Python features (3.12+) including type hints and Pydantic for data validation.

## Linting

### Ruff

Ruff is the primary linter for the project. The project uses Ruff's default configuration (no custom `ruff.toml` or `[tool.ruff]` section in `pyproject.toml`).

```bash
# Install Ruff (if not already installed)
pip install ruff

# Check code for style issues
ruff check .

# Auto-fix issues where possible
ruff check --fix .

# Check specific file or directory
ruff check src/stickler/comparators/
```

### Automatic Formatting in VS Code

To enable automatic code formatting and linting in VS Code:

1. Copy the example settings file:
   ```bash
   cp .vscode/settings.json.example .vscode/settings.json
   ```

2. Install the Ruff extension for VS Code (if not already installed):
   - Open VS Code
   - Go to Extensions (Cmd+Shift+X on macOS, Ctrl+Shift+X on Windows/Linux)
   - Search for "Ruff" by Charlie Marsh
   - Click Install

The settings file configures VS Code to automatically format Python files on save and organize imports using Ruff.

### CI Integration

Linting runs automatically on every push and pull request via GitHub Actions. While currently non-blocking, you should address linting issues before submitting PRs.

## Naming Conventions

### Classes

Use `PascalCase` for class names:

```python
# Good
class StructuredModel:
    pass

class LevenshteinComparator:
    pass

class NumericComparator:
    pass

# Bad
class structured_model:  # snake_case
    pass

class levenshteincomparator:  # lowercase
    pass
```

### Functions and Methods

Use `snake_case` for functions and methods:

```python
# Good
def compare_with(self, other: "StructuredModel") -> dict:
    pass

def binary_compare(self, value1: str, value2: str) -> float:
    pass

def get_field_scores(self) -> dict:
    pass

# Bad
def compareWith(self):  # camelCase
    pass

def GetFieldScores(self):  # PascalCase
    pass
```

### Constants

Use `UPPER_SNAKE_CASE` for constants:

```python
# Good
DEFAULT_THRESHOLD = 0.5
MAX_RECURSION_DEPTH = 100
COMPARISON_TOLERANCE = 1e-6

# Bad
default_threshold = 0.5  # lowercase
DefaultThreshold = 0.5   # PascalCase
```

### Private Members

Use leading underscore for private/internal members:

```python
class StructuredModel:
    # Private attribute
    _comparison_cache: dict

    # Private method
    def _compare_fields(self, other: "StructuredModel") -> dict:
        pass

    # Public method
    def compare_with(self, other: "StructuredModel") -> dict:
        pass
```

### Variables

Use descriptive `snake_case` names:

```python
# Good
field_scores = {}
comparison_result = model1.compare_with(model2)
total_weight = sum(weights)

# Bad
fs = {}              # Too short, unclear
fieldScores = {}     # camelCase
result = ...         # Too generic
```

## Type Hints

All public APIs should have type hints. The project uses Python 3.12+ typing features.

### Basic Type Hints

```python
from typing import Optional, List, Dict, Any

def compare(self, str1: str, str2: str) -> float:
    """Compare two strings and return similarity score."""
    ...

def evaluate(
    self,
    ground_truth: "StructuredModel",
    prediction: "StructuredModel"
) -> Dict[str, Any]:
    """Evaluate prediction against ground truth."""
    ...
```

### Optional Parameters

```python
from typing import Optional

def compare_with(
    self,
    other: "StructuredModel",
    threshold: Optional[float] = None
) -> dict:
    """Compare this model with another.

    Args:
        other: Model to compare against.
        threshold: Optional threshold override.
    """
    ...
```

### Generic Types

```python
from typing import List, Dict, Union

def process_items(
    self,
    items: List["StructuredModel"]
) -> Dict[str, float]:
    ...

def get_value(self) -> Union[str, int, float]:
    ...
```

### Type Aliases

For complex types, use type aliases:

```python
from typing import Dict, List, Any

# Type aliases
FieldScores = Dict[str, float]
ComparisonResult = Dict[str, Any]
ModelList = List["StructuredModel"]

def compare_all(self, models: ModelList) -> List[ComparisonResult]:
    ...
```

## Docstrings

Use Google-style docstrings for all public modules, classes, functions, and methods.

### Module Docstring

```python
"""Levenshtein string comparator implementation.

This module provides the LevenshteinComparator class for comparing
strings based on edit distance. It's useful for handling typos and
minor variations in text fields.

Example:
    >>> from stickler.comparators.levenshtein import LevenshteinComparator
    >>> comparator = LevenshteinComparator()
    >>> comparator.compare("hello", "helo")
    0.8
"""
```

### Class Docstring

```python
class LevenshteinComparator(BaseComparator):
    """Comparator using Levenshtein edit distance.

    Calculates string similarity based on the minimum number of
    single-character edits (insertions, deletions, substitutions)
    required to transform one string into another.

    Attributes:
        case_sensitive: Whether to perform case-sensitive comparison.

    Example:
        >>> comparator = LevenshteinComparator()
        >>> comparator.compare("hello", "hello")
        1.0
        >>> comparator.compare("hello", "hallo")
        0.8
    """
```

### Function/Method Docstring

```python
def compare_with(self, other: "StructuredModel") -> dict:
    """Compare this model with another model.

    Performs field-by-field comparison using configured comparators
    and calculates weighted overall score.

    Args:
        other: The model to compare against. Must be same type.

    Returns:
        Dictionary containing:
            - overall_score: Weighted average of field scores (0.0-1.0)
            - field_scores: Dict mapping field names to scores
            - all_fields_matched: Boolean if all fields meet thresholds

    Raises:
        TypeError: If other is not a StructuredModel instance.
        ValueError: If models have incompatible schemas.

    Example:
        >>> model1 = Person(name="John Doe", age=30)
        >>> model2 = Person(name="John Doe", age=30)
        >>> result = model1.compare_with(model2)
        >>> result["overall_score"]
        1.0
    """
    ...
```

### Property Docstring

```python
@property
def field_names(self) -> List[str]:
    """List of comparable field names.

    Returns:
        List of field names that have ComparableField descriptors.
    """
    ...
```

## Import Organization

Organize imports in three groups, separated by blank lines:

1. Standard library imports
2. Third-party package imports
3. Local application imports

```python
# Standard library
import json
from typing import List, Optional, Dict, Any

# Third-party packages
import pytest
from pydantic import Field, BaseModel

# Local imports
from stickler.structured_object_evaluator import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
```

### Import Guidelines

- Use absolute imports for clarity
- Import specific items rather than entire modules when practical
- Avoid wildcard imports (`from module import *`)

```python
# Good
from stickler.comparators.levenshtein import LevenshteinComparator

# Acceptable for commonly used items
from typing import List, Optional, Dict

# Avoid
from stickler.comparators import *
```

## Code Organization

### File Structure

Typical module structure:

```python
"""Module docstring describing purpose."""

# Imports (organized as above)
import ...

# Constants
DEFAULT_THRESHOLD = 0.5

# Type aliases
FieldScores = Dict[str, float]

# Helper functions (private)
def _helper_function():
    ...

# Classes
class MainClass:
    ...

# Module-level functions (if any)
def main_function():
    ...
```

### Class Structure

Organize class members in this order:

```python
class MyClass:
    """Class docstring."""

    # Class attributes
    default_value: ClassVar[int] = 0

    # Instance attributes (for dataclasses/pydantic)
    name: str
    value: int

    # __init__ (if not using dataclass)
    def __init__(self, name: str, value: int):
        ...

    # Properties
    @property
    def computed_value(self) -> int:
        ...

    # Public methods
    def public_method(self) -> None:
        ...

    # Private methods
    def _private_helper(self) -> None:
        ...

    # Magic methods
    def __repr__(self) -> str:
        ...
```

## Error Handling

### Exception Types

Use appropriate exception types:

```python
# Type errors
if not isinstance(other, StructuredModel):
    raise TypeError(f"Expected StructuredModel, got {type(other)}")

# Value errors
if threshold < 0 or threshold > 1:
    raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

# Custom exceptions (when needed)
class ComparisonError(Exception):
    """Raised when comparison cannot be performed."""
    pass
```

### Exception Messages

Provide clear, actionable error messages:

```python
# Good
raise ValueError(
    f"Threshold must be between 0.0 and 1.0, got {threshold}. "
    f"Consider using a value like 0.8 for fuzzy matching."
)

# Bad
raise ValueError("Invalid threshold")
```

## Comments

### When to Comment

- Explain **why**, not **what** (code shows what)
- Document non-obvious business logic
- Note workarounds or TODOs

```python
# Good - explains why
# Hungarian algorithm requires O(n^3) time, so we limit list size
# to prevent performance issues with large datasets
if len(items) > MAX_LIST_SIZE:
    items = items[:MAX_LIST_SIZE]

# Bad - states the obvious
# Increment counter
counter += 1
```

### TODO Comments

Use consistent format:

```python
# TODO: Implement caching for expensive comparisons
# TODO(username): Fix edge case with empty strings
# FIXME: This workaround should be removed after v2.0
```

## Best Practices Summary

1. **Be consistent** - Follow existing patterns in the codebase
2. **Be explicit** - Use type hints and clear names
3. **Be concise** - Avoid unnecessary complexity
4. **Document public APIs** - Docstrings for all public interfaces
5. **Handle errors gracefully** - Clear error messages
6. **Test your code** - See the [Testing Guide](testing-guide.md)
