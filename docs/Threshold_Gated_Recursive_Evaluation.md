# Threshold-Gated Recursive Evaluation for List[StructuredModel] Comparison

## Overview

This document outlines the threshold-gated recursive evaluation approach for comparing `List[StructuredModel]` objects in the GenAIDP library. This approach optimizes evaluation by only performing detailed nested field analysis on well-matched object pairs, while treating poorly-matched and unmatched objects as atomic units.

## Core Principle

**Only recurse into nested field evaluation for object pairs that meet the similarity threshold.**

## Algorithm Flow

### 1. Hungarian Matching
Use Hungarian algorithm to find optimal pairings between ground truth and prediction lists based on overall object similarity.

### 2. Threshold Classification
For each matched pair, check if the similarity score meets the `StructuredModel.match_threshold`:

- **similarity ≥ threshold** → **TP** (True Positive) + **recurse into nested fields**
- **similarity < threshold** → **FD** (False Discovery) + **stop recursion**

### 3. Unmatched Items
Handle items that couldn't be matched:
- **GT extras** → **FN** (False Negative) + **stop recursion**
- **Pred extras** → **FA** (False Alarm) + **stop recursion**

## Code Example

```python
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator
from typing import List

class Product(StructuredModel):
    product_id: str = ComparableField(
        comparator=ExactComparator(), 
        threshold=1.0,
        weight=3.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=2.0
    )
    price: float = ComparableField(
        threshold=0.9,
        weight=1.0
    )
    
    # Key: This threshold gates recursive evaluation
    match_threshold = 0.8

class Order(StructuredModel):
    order_id: str = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=2.0
    )
    products: List[Product] = ComparableField(
        aggregate=True,
        threshold=0.6,
        weight=3.0
    )

# Example data
gt_order = Order(
    order_id="ORD-12345",
    products=[
        Product(product_id="PROD-001", name="Laptop", price=999.99),
        Product(product_id="PROD-002", name="Mouse", price=29.99),
        Product(product_id="PROD-003", name="Cable", price=14.99)
    ]
)

pred_order = Order(
    order_id="ORD-12345",
    products=[
        Product(product_id="PROD-001", name="Laptop Computer", price=999.99),  # Good match (≥0.8)
        Product(product_id="PROD-002", name="Different Product", price=99.99),  # Poor match (<0.8)
        Product(product_id="PROD-004", name="New Product", price=19.99)  # Unmatched
        # PROD-003 is missing → FN
    ]
)
```

## Expected Behavior

### Scenario 1: Good Match (similarity ≥ 0.8)
**GT:** `Product(product_id="PROD-001", name="Laptop", price=999.99)`
**Pred:** `Product(product_id="PROD-001", name="Laptop Computer", price=999.99)`

**Result:**
- **Classification:** TP (True Positive)
- **Nested Field Analysis:** ✅ Performed
  - `product_id`: TP (exact match)
  - `name`: TP (similarity ~0.9)
  - `price`: TP (exact match)

### Scenario 2: Poor Match (similarity < 0.8)
**GT:** `Product(product_id="PROD-002", name="Mouse", price=29.99)`
**Pred:** `Product(product_id="PROD-002", name="Different Product", price=99.99)`

**Result:**
- **Classification:** FD (False Discovery)
- **Nested Field Analysis:** ❌ Not performed
- **Rationale:** Objects are too different to warrant detailed comparison

### Scenario 3: Unmatched Items
**GT:** `Product(product_id="PROD-003", name="Cable", price=14.99)` → **FN**
**Pred:** `Product(product_id="PROD-004", name="New Product", price=19.99)` → **FA**

**Result:**
- **Classification:** FN and FA respectively
- **Nested Field Analysis:** ❌ Not performed
- **Rationale:** No matching counterpart

## Edge Cases and Handling

### 1. Empty Lists
Empty lists are handled as follows:
- GT=[], Pred=[] → TN
- GT=[], Pred=[items] → All items are FA
- GT=[items], Pred=[] → All items are FN

### 2. Threshold Boundary Conditions
**Decision:** Use `similarity ≥ threshold` → TP + recurse
- Values exactly at the threshold are considered matches and trigger recursion

### 3. Nested List Scenarios
**Decision:** Full recursion through `i.compare_with(j)` is supported
- When a StructuredModel contains another `List[StructuredModel]`, the same threshold-gating applies recursively at all levels
- Each nested list uses its parent StructuredModel's `match_threshold` for gating decisions

### 4. Multiple Match Thresholds
**Decision:** Different StructuredModel types can have any user-defined `match_threshold` attribute
```python
class Product(StructuredModel):
    match_threshold = 0.8  # Strict matching for products

class Address(StructuredModel):
    match_threshold = 0.6  # More lenient for addresses
```

### 5. Aggregate Field Behavior
**Decision:** Leave aggregate field behavior unchanged for now
- Aggregate fields continue to sum metrics from all child fields as before
- This interaction will be addressed in future iterations

### 6. Performance Considerations
**Decision:** Performance is not a primary concern at this time
- The main goal is cleaner, more accurate metrics rather than performance optimization
- Hungarian matching still requires calculating all similarity scores regardless

## Comparison to Current Implementation

### Current Behavior
- Hungarian matching finds optimal pairs
- **All matched pairs** get recursive nested field analysis
- Results in detailed metrics for obviously poor matches

### Proposed Behavior
- Hungarian matching finds optimal pairs
- **Only good matches** get recursive nested field analysis
- Poor matches and unmatched items are atomic

### Benefits
1. **Cleaner Metrics:** Avoids misleading nested field metrics from forced comparisons
2. **Conceptual Clarity:** Matches human intuition about when detailed comparison is useful
3. **Potential Performance:** Fewer recursive operations for poor matches

### Risks
1. **Information Loss:** May lose insight into why objects didn't match well
2. **Threshold Sensitivity:** Results highly dependent on threshold selection
3. **Debugging Difficulty:** Harder to understand why objects were classified as FD

## Result Structure Enhancement

The result structure for `List[StructuredModel]` fields will be enhanced to include detailed non-match information:

```python
{
  "products": {
    "overall": { 
      "tp": 1, "fd": 1, "fn": 1, "fa": 1,
      "derived": { "cm_precision": 0.5, "cm_recall": 0.5, "cm_f1": 0.5 }
    },
    "fields": { 
      # Only recursive analysis for threshold-passing matches (≥ 0.8)
      "product_id": { "tp": 1, "derived": {...} },
      "name": { "tp": 1, "derived": {...} },
      "price": { "tp": 1, "derived": {...} }
    },
    "non_matches": [
      { 
        "type": "FD", 
        "gt_object": "Product(product_id='PROD-002', name='Mouse', price=29.99)",
        "pred_object": "Product(product_id='PROD-002', name='Different Product', price=99.99)",
        "similarity": 0.3
      },
      { 
        "type": "FN", 
        "gt_object": "Product(product_id='PROD-003', name='Cable', price=14.99)",
        "pred_object": null
      },
      { 
        "type": "FA", 
        "gt_object": null, 
        "pred_object": "Product(product_id='PROD-004', name='New Product', price=19.99)"
      }
    ]
  }
}
```

### Non-Matches Structure
- **FD (False Discovery):** Both objects exist but similarity < threshold
- **FN (False Negative):** GT object with no matching prediction
- **FA (False Alarm):** Prediction object with no matching GT
- **similarity:** Only included for FD cases where objects were compared

## Implementation Approach

### API Changes
**Decision:** Modify existing behavior globally (no new parameters)
- This approach fixes an inconsistency and provides conceptually cleaner behavior
- No backward compatibility concerns as this is a logic improvement

### Threshold Source
**Decision:** Use each StructuredModel's `match_threshold` attribute
- Each model type can define its own threshold for recursive evaluation
- Falls back to default `match_threshold = 0.7` if not specified

### Recursive Consistency
**Decision:** Full recursion through `i.compare_with(j)` maintains consistency
- When nested StructuredModel fields are compared, they use the same threshold-gating logic
- Ensures consistent behavior at all levels of nesting

## Implementation Plan

### Phase 1: Core Logic Update
1. **Modify `_calculate_nested_field_metrics()`** to check similarity scores before recursion
2. **Only recurse for matched pairs** where `similarity ≥ StructuredModel.match_threshold`
3. **Count below-threshold matches** as FD at the object level
4. **Skip nested field metrics** for FD, FN, and FA objects

### Phase 2: Result Structure Enhancement
1. **Add `non_matches` key** to List[StructuredModel] field results
2. **Track specific instances** that weren't matched with detailed information
3. **Maintain existing structure** for recursive analysis of good matches

### Phase 3: Testing and Validation
1. **Update existing tests** to expect new behavior
2. **Add comprehensive test cases** for edge cases
3. **Validate nested list scenarios** work correctly
4. **Test with different threshold values** across model types
