# Best Practices

## Calibration Test

The following test demonstrates how to calibrate the similarity threshold for string fields. The pairs below are chosen so that the Levenshtein similarity score is either above or below the 0.75 threshold as expected.

```python
import pytest
from stickler.comparators import LevenshteinComparator

@pytest.mark.parametrize("gt,pred,expected_match", [
    ("Acme Corporation", "Acme Corporaton", True),   # 0.94
    ("Acme Corporation", "Beta Industries", False),  # 0.13
    ("Jon Smith", "John Smith", True),               # 0.90
    ("123 Main St", "456 Oak Ave", False),           # 0.18
])

def test_calibration(gt, pred, expected_match):
    score = LevenshteinComparator().compare(gt, pred)
    assert (score >= 0.75) == expected_match
```

## Business‑Aligned Weights Example

```python
from typing import List
from stickler.models import StructuredModel, ComparableField
from stickler.comparators import (
    ExactComparator,
    NumericComparator,
    LevenshteinComparator,
    FuzzyComparator,
)

class LineItem(StructuredModel):
    description: str = ComparableField(weight=1.0)
    quantity: int = ComparableField(weight=1.0)
    unit_price: float = ComparableField(weight=1.0)

class Invoice(StructuredModel):
    invoice_id: str = ComparableField(comparator=ExactComparator(), weight=3.0)
    total_amount: float = ComparableField(
        comparator=NumericComparator(absolute_tolerance=0.01), weight=2.5
    )
    vendor_name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8, weight=1.5)
    line_items: List[LineItem] = ComparableField(weight=1.0)
    internal_notes: str = ComparableField(comparator=FuzzyComparator(), weight=0.2)

# A `ComparableField()` with no `comparator=` uses Levenshtein at 0.5 on any scalar type, so name the comparator for IDs and numbers.
```

# ... rest of the file unchanged ...