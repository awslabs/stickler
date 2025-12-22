---
title: Hungarian Matching
---

# Hungarian Matching for `List[StructuredModel]` A Complete Guide

## Overview

This document provides a comprehensive guide to Hungarian matching for `List[StructuredModel]` fields in the stickler library, including concrete examples, expected behavior, and validation of the current implementation.

## Core Algorithm: Threshold-Gated Recursive Evaluation

The Hungarian matching algorithm for structured lists follows this principle:

**🔑 Only recurse into nested field evaluation for object pairs that meet the similarity threshold.**

### Algorithm Steps

1. **Hungarian Matching**: Find optimal pairings between GT and Pred lists based on overall object similarity
2. **Threshold Classification**: For each matched pair, check if similarity ≥ `StructuredModel.match_threshold`
   - **similarity ≥ threshold** → **TP** + **recurse into nested fields**  
   - **similarity < threshold** → **FD** + **stop recursion (atomic)**
3. **Unmatched Items**: Handle items that couldn't be matched
   - **GT extras** → **FN** + **stop recursion (atomic)**
   - **Pred extras** → **FA** + **stop recursion (atomic)**

## Concrete Example: Transaction Matching

### Model Definitions

```python
class Transaction(StructuredModel):
    transaction_id: str = ComparableField(
        comparator=ExactComparator(), 
        threshold=1.0,
        weight=3.0
    )
    
    description: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=2.0
    )
    
    amount: float = ComparableField(
        threshold=0.9,
        weight=1.0
    )
    
    # ⚠️ CRITICAL: This threshold controls Hungarian matching recursion
    match_threshold = 0.8

class Account(StructuredModel):
    account_id: str = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=2.0
    )
    
    # Notice: NO threshold on this field - should use Transaction.match_threshold
    transactions: List[Transaction] = ComparableField(weight=3.0)
```

### Test Data

**Ground Truth:**
```python
gt_account = Account(
    account_id="ACC-12345",
    transactions=[
        Transaction(transaction_id="TXN-001", description="Coffee shop payment", amount=4.95),
        Transaction(transaction_id="TXN-002", description="Grocery store", amount=127.43),
        Transaction(transaction_id="TXN-003", description="Gas station", amount=45.67)
    ]
)
```

**Prediction:**
```python
pred_account = Account(
    account_id="ACC-12345", 
    transactions=[
        Transaction(transaction_id="TXN-001", description="Coffee shop", amount=4.95),      # Good match (TP)
        Transaction(transaction_id="TXN-002", description="Online purchase", amount=89.99), # Poor match (FD)  
        Transaction(transaction_id="TXN-004", description="Restaurant", amount=23.45)       # Poor match (FD)
    ]
)
```

## Step-by-Step Hungarian Matching Analysis

### Step 1: Pairwise Similarity Calculation

| GT Index | Pred Index | GT Transaction | Pred Transaction | Similarity | Above Threshold? |
|----------|------------|----------------|------------------|------------|------------------|
| 0 | 0 | TXN-001: Coffee shop payment | TXN-001: Coffee shop | **0.860** | ✅ **Yes (≥0.8)** |
| 0 | 1 | TXN-001: Coffee shop payment | TXN-002: Online purchase | 0.137 | ❌ No |
| 0 | 2 | TXN-001: Coffee shop payment | TXN-004: Restaurant | 0.154 | ❌ No |
| 1 | 0 | TXN-002: Grocery store | TXN-001: Coffee shop | 0.130 | ❌ No |
| 1 | 1 | TXN-002: Grocery store | TXN-002: Online purchase | **0.572** | ❌ **No (<0.8)** |
| 1 | 2 | TXN-002: Grocery store | TXN-004: Restaurant | 0.135 | ❌ No |
| 2 | 0 | TXN-003: Gas station | TXN-001: Coffee shop | 0.097 | ❌ No |
| 2 | 1 | TXN-003: Gas station | TXN-002: Online purchase | 0.056 | ❌ No |
| 2 | 2 | TXN-003: Gas station | TXN-004: Restaurant | 0.124 | ❌ No |

### Step 2: Hungarian Algorithm Assignment

The Hungarian algorithm finds the optimal assignment that maximizes total similarity:

- **GT[0] → Pred[0]**: 0.860 ✅ **Good Match**
- **GT[1] → Pred[1]**: 0.572 ❌ **Poor Match**  
- **GT[2] → Pred[2]**: 0.124 ❌ **Poor Match**

### Step 3: Threshold-Gated Classification

| Pair | Similarity | Classification | Nested Field Analysis |
|------|------------|----------------|----------------------|
| GT[0] → Pred[0] | 0.860 ≥ 0.8 | **TP** | ✅ **Generate nested metrics** |
| GT[1] → Pred[1] | 0.572 < 0.8 | **FD** | ❌ **Skip (atomic treatment)** |
| GT[2] → Pred[2] | 0.124 < 0.8 | **FD** | ❌ **Skip (atomic treatment)** |

## Expected Result Structure

### Object-Level Metrics (List Level)
```python
"transactions": {
    "overall": {
        "tp": 1,  # GT[0] → Pred[0] good match
        "fd": 2,  # GT[1] → Pred[1] AND GT[2] → Pred[2] poor matches  
        "fa": 0,  # No unmatched preds (equal length lists)
        "fn": 0,  # No unmatched GTs (equal length lists)
        "fp": 2   # fd + fa = 2 + 0
    },
    "fields": {
        # Only nested metrics for the TP pair (GT[0] → Pred[0])
        "transaction_id": {"tp": 1, "fa": 0, "fd": 0, "fn": 0},
        "description": {"tp": 1, "fa": 0, "fd": 0, "fn": 0}, 
        "amount": {"tp": 1, "fa": 0, "fd": 0, "fn": 0}
    },
    "non_matches": [
        {
            "type": "FD",
            "gt_object": "Transaction(TXN-002, Grocery store, $127.43)",
            "pred_object": "Transaction(TXN-002, Online purchase, $89.99)",
            "similarity": 0.572
        },
        {
            "type": "FD",
            "gt_object": "Transaction(TXN-003, Gas station, $45.67)",
            "pred_object": "Transaction(TXN-004, Restaurant, $23.45)",
            "similarity": 0.124
        }
    ]
}
```

## ✅ Current Implementation Status

### Implementation Validation Results

**Actual Test Results (Equal Length Lists):**
```
TEST CASE 1: Equal Length Lists (3x3)
Actual Metrics: TP=1, FD=2, FN=0, FA=0
Expected Metrics: TP=1, FD=2, FN=0, FA=0
✅ PASS: Metrics match corrected expected behavior

TEST CASE 2: GT Longer (4x2)
Actual Metrics: TP=1, FD=1, FN=2, FA=0
Expected Metrics: TP=1, FD=1, FN=2, FA=0
✅ PASS

TEST CASE 3: Pred Longer (2x4)
Actual Metrics: TP=1, FD=1, FN=0, FA=2
Expected Metrics: TP=1, FD=1, FN=0, FA=2
✅ PASS

TEST CASE 4: All Above Threshold (3x3)
Actual Metrics: TP=3, FD=0, FN=0, FA=0
Expected Metrics: TP=3, FD=0, FN=0, FA=0
✅ PASS
```

### Key Findings

1. **✅ Equal Length Behavior**: When GT and Pred lists have equal length, Hungarian algorithm pairs everyone up, resulting in only TP/FD classifications (no FN/FA)

2. **✅ Below-Threshold Treatment**: Both similarity scores of 0.572 and 0.124 are correctly treated the same way as FD since both are below the 0.8 threshold

3. **✅ Length Mismatch Behavior**: Only when lists have uneven lengths do we see FN (unmatched GT items) or FA (unmatched Pred items)

4. **✅ Threshold Gating**: The implementation correctly uses the `match_threshold = 0.8` to determine TP vs FD classification

### Conclusion

The implementation is **already correct** and matches the expected behavior described in this document. The key insight was understanding that:

- **Equal length lists** → Only TP/FD possible (Hungarian pairs everyone)
- **Uneven length lists** → FN/FA occur for unpaired items
- **Below threshold** → All treated as FD regardless of specific similarity value

## Implementation Plan

### Phase 1: Create Extraction Baseline Tests ✅ 
- Document current behavior with concrete examples
- Create test cases that capture existing Hungarian matching patterns
- Establish bit-for-bit compatibility baseline

### Phase 2: Extract Hungarian Logic to Dedicated Class
- Create `StructuredListComparator` class
- Move ~500 lines from `StructuredModel` to dedicated class:
  - `_compare_struct_list_with_scores()`
  - `_calculate_nested_field_metrics()`  
  - `_calculate_object_level_metrics()`
  - `_calculate_struct_list_similarity()`
- Update `StructuredModel._dispatch_field_comparison()` to delegate

### Phase 3: Fix Threshold Logic
- Correct threshold source in extracted class
- Implement proper threshold-gated recursion
- Update object-level vs field-level metric separation

### Phase 4: Integration Testing
- Verify Phase 1 baseline tests still pass
- Test with different `match_threshold` values
- Validate nested list scenarios work correctly

### Phase 5: Cleanup and Documentation
- Remove old code from `StructuredModel`
- Update documentation with corrected examples
- Add integration tests for edge cases

## Key Architectural Principles

### 1. Threshold Source Hierarchy
```
List[StructuredModel] Threshold Resolution:
1. Use StructuredModel.match_threshold (class attribute)
2. Fall back to default 0.7 if not specified  
3. NEVER use ComparableField.threshold from parent list field
```

### 2. Metric Separation
```
Object-Level Metrics: Count objects (TP, FD, FA, FN)
Field-Level Metrics: Count field comparisons within good matches only
```

### 3. Recursion Gating
```
IF object_similarity >= match_threshold:
    classification = TP
    RECURSE into nested field analysis
ELSE:
    classification = FD  
    STOP recursion (treat as atomic)
```

## Testing Strategy

### Baseline Compatibility Tests
- Capture exact current behavior before refactoring
- Bit-for-bit comparison of confusion matrix output
- Edge cases: empty lists, single items, all poor matches

### Threshold Boundary Tests  
- Objects exactly at threshold (0.8000...)
- Objects slightly above/below threshold
- Different model classes with different thresholds

### Nested List Scenarios
- `List[List[StructuredModel]]` recursive structures
- Mixed threshold values across nesting levels
- Performance with large lists

This comprehensive documentation serves as both a specification for the correct behavior and a baseline for testing the upcoming refactor.
