# TODO: Hungarian Matching Refactor & Threshold Fix

## Overview

Fix threshold-gated recursive evaluation by refactoring Hungarian matching logic to a separate class and correcting threshold source usage.

## Current Problems

1. **Wrong Threshold Source**: `_calculate_nested_field_metrics()` uses `ComparableField.threshold` instead of `StructuredModel.match_threshold`
2. **Complex Embedded Logic**: ~500 lines of Hungarian matching code embedded in `StructuredModel` class
3. **Test Failures**: `test_threshold_gated_poor_match` and `test_threshold_gated_mixed_scenario` failing due to incorrect threshold usage

## Architecture Decision

**Refactor First, Then Fix**: Extract Hungarian matching to dedicated class, then fix threshold logic in cleaner codebase.

### Benefits:
- Cleaner separation of concerns
- Easier to unit test Hungarian matching in isolation  
- More maintainable threshold logic
- Better long-term architecture

## Implementation Plan

### Phase 1: Create Extraction Baseline Tests
**Goal**: Ensure we can extract without breaking functionality

**Files to Create:**
- `tests/key_information_evaluation/structured_object_evaluator/test_list_comparator_extraction.py`

**Tests:**
```python
def test_extraction_baseline():
    """Test current behavior before extraction"""
    # Test all current Hungarian matching scenarios
    # Capture exact current behavior as baseline

def test_post_extraction_equivalence(): 
    """Test that extracted logic produces identical results"""
    # Compare old vs new implementation results
    # Ensure bit-for-bit identical output
```

**Success Criteria**: Comprehensive baseline test suite capturing current behavior

### Phase 2: Extract Hungarian Logic to Dedicated Class
**Goal**: Move Hungarian matching to dedicated class

**Files to Create:**
- `src/stickler/structured_object_evaluator/models/structured_list_comparator.py`

**Proposed Class Structure:**
```python
class StructuredListComparator:
    """Handles comparison of List[StructuredModel] fields using Hungarian matching."""
    
    def __init__(self, model_class, field_name, parent_model):
        self.model_class = model_class
        self.field_name = field_name  
        self.parent_model = parent_model
        self.match_threshold = getattr(model_class, 'match_threshold', 0.7)
    
    def compare_lists(self, gt_list, pred_list):
        """Main entry point - replaces _compare_struct_list_with_scores"""
        
    def calculate_nested_metrics(self, gt_list, pred_list):
        """Replaces _calculate_nested_field_metrics with cleaner threshold logic"""
```

**Code to Extract from `StructuredModel`:**
- `_compare_struct_list_with_scores()` (~200 lines)
- `_calculate_nested_field_metrics()` (~250 lines)
- `_calculate_object_level_metrics()` 
- `_calculate_struct_list_similarity()`
- `_handle_struct_list_empty_cases()`

**Files to Modify:**
- `src/stickler/structured_object_evaluator/models/structured_model.py`
  - Update `_dispatch_field_comparison()` to delegate to `StructuredListComparator`
  - Remove extracted methods

**Testing**: Re-run Phase 1 tests to ensure identical behavior

**Success Criteria**: All baseline tests pass with extracted implementation

### Phase 3: Fix Threshold Logic in Clean Codebase  
**Goal**: Fix threshold source in the cleaner, dedicated class

**Files to Create:**
- `tests/key_information_evaluation/structured_object_evaluator/test_threshold_gated_behavior.py`

**Tests:**
```python
def test_threshold_source_consistency():
    """Verify object threshold is used, not field threshold"""
    # Create Product with match_threshold = 0.6
    # Create ComparableField with threshold = 0.9  
    # Verify Hungarian matching uses 0.6, not 0.9

def test_poor_match_no_recursion():
    """Verify poor matches get no nested field analysis"""
    # Create objects with similarity < match_threshold
    # Verify no nested field metrics generated

def test_good_match_gets_recursion():
    """Verify good matches get full recursive analysis"""
    # Create objects with similarity >= match_threshold
    # Verify nested field metrics are generated

def test_mixed_scenario_threshold_gating():
    """Test mixed good/poor matches with proper threshold gating"""
    # Create scenario from failing test_threshold_gated_mixed_scenario
    # Verify only good matches get recursive analysis

def test_empty_list_edge_cases():
    """Test threshold detection with various empty list scenarios"""
    # Test: gt_list=[], pred_list=[items]
    # Test: gt_list=[items], pred_list=[]
    # Test: gt_list=[], pred_list=[]
    # Verify no crashes and sensible behavior
```

**Code Changes:**
In `StructuredListComparator.__init__()`:
```python
# Use StructuredModel's match_threshold for threshold-gated recursion  
self.match_threshold = getattr(model_class, 'match_threshold', 0.7)
```

In `calculate_nested_metrics()` method:
```python
# Use self.match_threshold consistently for gating decisions
for gt_idx, pred_idx, similarity_score in matched_pairs_with_scores:
    is_above_threshold = similarity_score >= self.match_threshold
    
    if is_above_threshold:
        # Perform recursive analysis
    else:
        # Skip recursive analysis - treat as atomic FD
```

**Success Criteria**: All new threshold tests pass

### Phase 4: Integration Testing
**Goal**: Verify extracted and fixed logic works in full system context

**Tests to Run:**
```bash
# Run the original failing tests
python -m pytest tests/key_information_evaluation/structured_object_evaluator/test_threshold_gated_recursion.py -v

# Run broader regression tests  
python -m pytest tests/key_information_evaluation/structured_object_evaluator/ -k "not (classification_logic or confusion_matrix_definitions)" -n 8 -v
```

**Success Criteria**: 
- `test_threshold_gated_poor_match` passes
- `test_threshold_gated_mixed_scenario` passes
- No regressions in working tests

### Phase 5: Clean Up and Documentation
**Goal**: Remove old code, update documentation

**Actions:**
1. Remove old Hungarian matching methods from `StructuredModel`
2. Update comments that reference old threshold behavior
3. Verify `docs/Threshold_Gated_Recursive_Evaluation.md` matches implementation
4. Run final comprehensive regression test suite

**Files to Modify:**
- `src/pedantic/structured_object_evaluator/models/structured_model.py` - remove old methods
- Update any misleading comments about threshold usage

**Success Criteria**: Clean codebase with no dead code, accurate documentation

## Risk Assessment

**Low Risk**: 
- Extraction with comprehensive baseline tests
- Incremental changes with testing at each step

**Medium Risk**: 
- Large amount of code to move (~500 lines)
- Complex nested field metric calculation logic

**Mitigation Strategy**: 
- Comprehensive test coverage before extraction
- Bit-for-bit compatibility verification
- Incremental implementation with rollback capability

## Definition of Done

✅ All original failing tests pass  
✅ No regressions in existing test suite  
✅ Hungarian matching logic cleanly separated  
✅ Threshold source consistent across all operations  
✅ Code complexity reduced in `StructuredModel`  
✅ Documentation updated and accurate

## Next Steps

1. Review this plan for completeness
2. Begin Phase 1: Create comprehensive baseline tests
3. Execute phases incrementally with full testing at each step

---

**Note**: This is a significant refactor. We should proceed carefully with thorough testing at each phase to ensure we don't introduce regressions while fixing the threshold logic.
