# Migration Guide: Annotated Pattern for Stickler

## Executive Summary

This document outlines a migration path from the current **function-based `ComparableField`** to a new **class-based `ComparableField` with `Annotated` type hints** pattern. The new pattern offers significant benefits while maintaining backward compatibility during the transition.

**Status**: Proof of Concept Complete (see `src/stickler/structured_object_evaluator/models/structured_model_test.py`)

---

## Table of Contents

1. [Current vs Proposed Pattern](#current-vs-proposed-pattern)
2. [Benefits of Migration](#benefits-of-migration)
3. [Migration Strategy](#migration-strategy)
4. [Integration with Existing Systems](#integration-with-existing-systems)
5. [Step-by-Step Migration Plan](#step-by-step-migration-plan)
6. [Breaking Changes & Compatibility](#breaking-changes--compatibility)
7. [Timeline & Effort Estimation](#timeline--effort-estimation)
8. [Risk Assessment](#risk-assessment)

---

## Current vs Proposed Pattern

### Current Pattern (Function-Based)

```python
class Invoice(StructuredModel):
    """Current approach using function-based ComparableField."""
    
    invoice_number: str = ComparableField(
        comparator=ExactComparator(),
        threshold=0.9,
        weight=2.0
    )
    
    vendor: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=1.0
    )
```

**How it works:**
- `ComparableField()` is a **function** that returns a `pydantic.Field`
- Comparison metadata stored in `json_schema_extra` function attributes
- Hybrid approach with runtime data attached to function objects
- ~212 lines in `comparable_field.py`

### Proposed Pattern (Class-Based with Annotated)

```python
class Invoice(StructuredModel):
    """New approach using Annotated pattern."""
    
    invoice_number: Annotated[str, ComparableField(
        comparator=ExactComparator(),
        threshold=0.9,
        weight=2.0
    )]
    
    vendor: Annotated[str, ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,
        weight=1.0
    )]
```

**How it works:**
- `ComparableField` is a **Pydantic model class** (like `BaseModel`)
- Configuration lives in type hints (self-documenting)
- `StructuredModel` base class has `@model_validator(mode='before')` that auto-wraps raw values
- Smart serialization: clean by default, full metadata with `context={'comp_info': True}`
- ~150 lines total (class + base validator)

---

## Benefits of Migration

### 1. **Self-Documenting Code**
```python
# Type hints show configuration at a glance
invoice_number: Annotated[str, ComparableField(threshold=0.9, weight=2.0)]

# IDEs can extract and display this information
# JSON Schema generation includes full type information
```

### 2. **Cleaner API**
```python
# Old: Need to specify value= in some cases
field = ComparableField(value="INV-001", threshold=0.9)

# New: Value is auto-wrapped by validator
invoice = Invoice(invoice_number="INV-001")  # Automatically wrapped!
```

### 3. **Reduced Code Complexity**
- **Old**: ~212 lines in `comparable_field.py` + per-field validators
- **New**: ~150 lines total (class + base validator)
- **Reduction**: ~30% less code, 70% less helper code

### 4. **Better Type Safety**
```python
# Old: Type checkers see the field as just 'str'
invoice_number: str = ComparableField(...)

# New: Type checkers understand the full structure
invoice_number: Annotated[str, ComparableField(...)]
# Access: invoice.invoice_number.value, .threshold, .weight, .comparator
```

### 5. **Smart Serialization**
```python
# Clean serialization by default
invoice.model_dump()
# → {'invoice_number': 'INV-001', 'vendor': 'ACME Corp'}

# Full metadata when needed
invoice.model_dump(context={'comp_info': True})
# → {'invoice_number': {'value': 'INV-001', 'threshold': 0.9, ...}}
```

### 6. **JSON Schema Integration**
```python
# Can dynamically generate models from JSON Schema
schema = {
    "properties": {
        "invoice_number": {
            "type": "string",
            "x-aws-stickler-threshold": 0.9,
            "x-aws-stickler-weight": 2.0
        }
    }
}

Invoice = SticklerSchemaParser.parse_schema(schema)
# Automatically creates Annotated fields!
```

---

## Migration Strategy

### Phase 1: Dual Support (Backward Compatible)

**Goal**: Support both patterns simultaneously

**Implementation**:
1. Create new `ComparableField` class alongside existing function
2. Update `StructuredModel` base to handle both patterns
3. Add validator that detects pattern and wraps accordingly

**Code Example**:
```python
# In StructuredModel base class
@model_validator(mode="before")
@classmethod
def auto_wrap_comparable_fields(cls, data: Any) -> Any:
    """Handle both old function-based and new Annotated pattern."""
    for field_name, field_info in cls.model_fields.items():
        if field_name in data:
            raw_value = data[field_name]
            
            # Pattern 1: Check for Annotated[Type, ComparableField(...)]
            if hasattr(cls, '__annotations__') and field_name in cls.__annotations__:
                annotation = cls.__annotations__[field_name]
                if get_origin(annotation) is Annotated:
                    # NEW PATTERN - extract template config
                    args = get_args(annotation)
                    for arg in args[1:]:
                        if isinstance(arg, ComparableField):
                            # Wrap using template
                            data[field_name] = ComparableField(value=raw_value, **arg.dict())
                            break
            
            # Pattern 2: Old function-based (fallback)
            # Check if field has json_schema_extra with comparison metadata
            if hasattr(field_info, 'json_schema_extra'):
                # Extract metadata from function attributes
                # Wrap using old-style config
                pass
    
    return data
```

### Phase 2: Gradual Migration

**Goal**: Migrate codebase incrementally

**Priority Order**:
1. **Documentation & Examples** (low risk, high visibility)
2. **New Features** (use new pattern from day 1)
3. **Core Models** (high-traffic, well-tested)
4. **Test Suite** (parallel to code migration)
5. **Edge Cases** (last, most complex)

### Phase 3: Deprecation

**Goal**: Phase out old pattern

**Steps**:
1. Add deprecation warnings to function-based `ComparableField`
2. Update all first-party code to new pattern
3. Give users 2-3 minor versions notice
4. Remove old function-based implementation

---

## Integration with Existing Systems

### 1. Comparator System

**Current Integration**:
```python
# comparable_field.py stores comparator in function attributes
json_schema_extra_func._comparator_instance = actual_comparator
```

**New Integration**:
```python
# ComparableField class stores comparator as instance attribute
class ComparableField[FieldType](BaseModel):
    value: FieldType | None = None
    comparator: BaseComparator | None = None  # Direct storage!
    threshold: float = 0.5
    weight: float = 1.0
```

**Impact**: ✅ **Simpler** - No need for function attribute hacks

### 2. StructuredModel.compare_with()

**Current Flow**:
```python
# structured_model.py extracts comparison config
def compare_with(self, other):
    for field_name, field_info in self.model_fields.items():
        # Extract from json_schema_extra function attributes
        comparator = field_info.json_schema_extra._comparator_instance
        threshold = field_info.json_schema_extra._threshold
```

**New Flow**:
```python
# structured_model.py accesses ComparableField instance directly
def compare_with(self, other):
    for field_name, field_info in self.model_fields.items():
        field_value = getattr(self, field_name)
        if isinstance(field_value, ComparableField):
            # Direct access to all metadata!
            comparator = field_value.comparator
            threshold = field_value.threshold
            score = comparator.compare(field_value.value, other_value.value)
```

**Impact**: ✅ **Much Cleaner** - Direct attribute access vs function attributes

### 3. Evaluator (StructuredModelEvaluator)

**Current Usage**:
```python
# evaluator.py uses compare_with() output
evaluator = StructuredModelEvaluator(model_class=Invoice)
metrics = evaluator.evaluate(ground_truth_list, prediction_list)
```

**New Usage**:
```python
# NO CHANGES NEEDED!
# Evaluator uses compare_with() which is updated internally
evaluator = StructuredModelEvaluator(model_class=Invoice)
metrics = evaluator.evaluate(ground_truth_list, prediction_list)
```

**Impact**: ✅ **Zero Changes** - Evaluator API remains identical

### 4. Hungarian Matching (List Comparison)

**Current Integration**:
```python
# structured_model.py handles List[StructuredModel] fields
if is_list_field:
    matches = HungarianMatcher.match(gt_list, pred_list)
```

**New Integration**:
```python
# Same logic, but cleaner field detection
if is_list_field:
    # Field is already unwrapped to List[StructuredModel]
    matches = HungarianMatcher.match(field_value, other_value)
```

**Impact**: ✅ **Minor Simplification** - Field type detection is cleaner

### 5. JSON Schema Generation

**Current Approach**:
```python
# model_json_schema() includes x-comparison metadata
schema = Invoice.model_json_schema()
# → Has x-comparison in json_schema_extra
```

**New Approach**:
```python
# Can serialize with context to include full metadata
schema = Invoice.model_json_schema()
# OR dynamically generate from schema
Invoice = SticklerSchemaParser.parse_schema(json_schema)
```

**Impact**: ✅ **Enhanced** - Bidirectional JSON Schema ↔ Model

### 6. Serialization & Deserialization

**Current Behavior**:
```python
# model_dump() returns just field values
invoice.model_dump()
# → {'invoice_number': 'INV-001'}
```

**New Behavior**:
```python
# Smart serialization with @model_serializer
invoice.model_dump()  # Clean
# → {'invoice_number': 'INV-001'}

invoice.model_dump(context={'comp_info': True})  # Full metadata
# → {'invoice_number': {'value': 'INV-001', 'threshold': 0.9, ...}}
```

**Impact**: ✅ **Improved** - Smart serialization + backward compatible

---

## Step-by-Step Migration Plan

### Prerequisites
- [x] Proof of concept implemented (`structured_model_test.py`)
- [ ] Performance benchmarks (old vs new)
- [ ] Memory profiling (ensure no regression)
- [ ] Comprehensive test coverage for new pattern

### Step 1: Create New Classes (Week 1-2)

**Files to Create/Modify**:
```
src/stickler/structured_object_evaluator/models/
├── comparable_field_v2.py          # New ComparableField class
├── structured_model_base.py        # New StructuredModel with validator
└── schema_parser.py                # SticklerSchemaParser
```

**Tasks**:
- [ ] Implement `ComparableField` as Pydantic model
- [ ] Implement `@model_serializer` for smart serialization
- [ ] Implement `StructuredModel` base with auto-wrapping validator
- [ ] Implement `SticklerSchemaParser` for JSON Schema support
- [ ] Add comprehensive unit tests

### Step 2: Update Core Infrastructure (Week 3-4)

**Files to Modify**:
```
src/stickler/structured_object_evaluator/models/
├── structured_model.py             # Update compare_with() logic
└── configuration_helper.py         # Update metadata extraction
```

**Tasks**:
- [ ] Update `compare_with()` to handle ComparableField instances
- [ ] Update field metadata extraction to support both patterns
- [ ] Add backward compatibility layer
- [ ] Update helper methods

### Step 3: Migrate Examples & Documentation (Week 5)

**Files to Update**:
```
examples/
├── scripts/
│   ├── quick_start.py              # Show both patterns
│   ├── bulk_evaluation_demo.py
│   └── aggregate_metrics_demo.py
└── notebooks/
    └── Quick_start.ipynb           # Update with Annotated pattern
```

**Tasks**:
- [ ] Update all example scripts
- [ ] Update Quick Start notebook
- [ ] Create migration guide (this document!)
- [ ] Update README.md with new pattern

### Step 4: Migrate Test Suite (Week 6-8)

**Files to Update** (~60 test files):
```
tests/structured_object_evaluator/
├── test_quickstart_examples.py
├── test_structured_model.py
├── test_evaluator.py
└── ... (~57 more files)
```

**Migration Pattern**:
```python
# Before
class Invoice(StructuredModel):
    invoice_number: str = ComparableField(threshold=0.9, weight=2.0)

# After
class Invoice(StructuredModel):
    invoice_number: Annotated[str, ComparableField(threshold=0.9, weight=2.0)]
```

**Tasks**:
- [ ] Create automated migration script
- [ ] Run script on all test files
- [ ] Manual review of generated code
- [ ] Fix edge cases
- [ ] Ensure 100% test pass rate

### Step 5: Add Deprecation Warnings (Week 9)

**Files to Modify**:
```
src/stickler/structured_object_evaluator/models/
└── comparable_field.py             # Add deprecation to function
```

**Implementation**:
```python
def ComparableField(...):
    """DEPRECATED: Use Annotated[Type, ComparableField(...)] pattern instead."""
    warnings.warn(
        "Function-based ComparableField is deprecated. "
        "Use: field: Annotated[Type, ComparableField(...)] instead. "
        "See migration guide: docs/Migration_to_Annotated_Pattern.md",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing implementation
```

### Step 6: Monitor & Gather Feedback (Month 3)

**Activities**:
- [ ] Release as beta feature
- [ ] Gather user feedback
- [ ] Monitor error reports
- [ ] Performance monitoring
- [ ] Fix issues as they arise

### Step 7: Full Cutover (Month 4+)

**Tasks**:
- [ ] Remove old function-based implementation
- [ ] Remove backward compatibility layer
- [ ] Update all documentation
- [ ] Major version bump (2.0.0)

---

## Breaking Changes & Compatibility

### Breaking Changes

#### 1. Field Access Pattern

**Before**:
```python
invoice = Invoice(invoice_number="INV-001")
value = invoice.invoice_number  # Direct access to str
# Type: str
```

**After**:
```python
invoice = Invoice(invoice_number="INV-001")
value = invoice.invoice_number.value  # Access via .value
# Type: ComparableField[str]
```

**Mitigation**: 
- Keep `__getattribute__` override in StructuredModel for compatibility
- OR: Provide migration script to update all field accesses

#### 2. Serialization Context

**Before**:
```python
# Always returns clean dict
invoice.model_dump()
```

**After**:
```python
# Clean by default
invoice.model_dump()

# Full metadata requires context
invoice.model_dump(context={'comp_info': True})
```

**Mitigation**: ✅ **Backward Compatible** - Default behavior unchanged

#### 3. Type Annotations

**Before**:
```python
class Invoice(StructuredModel):
    invoice_number: str = ComparableField(...)
```

**After**:
```python
class Invoice(StructuredModel):
    invoice_number: Annotated[str, ComparableField(...)]
```

**Mitigation**: 
- Support both during transition
- Automated migration script

### Non-Breaking Changes

✅ **Evaluator API** - No changes needed  
✅ **compare_with() API** - No changes needed  
✅ **JSON Schema generation** - Enhanced, not changed  
✅ **Hungarian matching** - Works identically  
✅ **Comparator system** - Cleaner integration  

---

## Timeline & Effort Estimation

### Optimistic Timeline (3 months)

| Phase | Duration | Parallel? | Risk |
|-------|----------|-----------|------|
| 1. Core Implementation | 2 weeks | No | Low |
| 2. Infrastructure Updates | 2 weeks | No | Medium |
| 3. Examples & Docs | 1 week | Yes | Low |
| 4. Test Migration | 3 weeks | Yes | Medium |
| 5. Deprecation | 1 week | Yes | Low |
| 6. Beta & Feedback | 4 weeks | No | High |
| **Total** | **3 months** | | |

### Realistic Timeline (4-5 months)

Adding buffer for:
- Unexpected edge cases
- User feedback integration
- Performance optimization
- Documentation refinement

### Effort Breakdown

**Code Changes**:
- ~500 lines new code (ComparableField class, validator, parser)
- ~200 lines infrastructure updates
- ~60 test files to migrate (~668 references)
- ~10 example files to update

**Total Estimated LOC**: ~1500-2000 lines changed

**Team Effort**:
- 1 developer full-time: **3-4 months**
- 2 developers: **2-3 months**
- With heavy test automation: **2 months**

---

## Risk Assessment

### High Risk

#### 1. **Field Access Breaking Changes**
- **Risk**: Existing code expects `invoice.field` returns value directly
- **Impact**: 🔴 High - Affects all users
- **Mitigation**: 
  - Provide `__getattribute__` compatibility layer
  - Automated migration tooling
  - Clear migration guide with examples

#### 2. **Performance Regression**
- **Risk**: Auto-wrapping adds overhead
- **Impact**: 🟡 Medium - Could affect high-volume use cases
- **Mitigation**:
  - Benchmark before/after
  - Profile hot paths
  - Optimize validator logic

### Medium Risk

#### 3. **Test Suite Migration Complexity**
- **Risk**: ~668 ComparableField references to update
- **Impact**: 🟡 Medium - Time-consuming, error-prone
- **Mitigation**:
  - Automated migration script
  - Comprehensive testing
  - Gradual rollout

#### 4. **Edge Cases in Type Introspection**
- **Risk**: Complex type annotations (Union, Optional, etc.)
- **Impact**: 🟡 Medium - May not handle all cases
- **Mitigation**:
  - Comprehensive type testing
  - Fallback to old pattern if detection fails

### Low Risk

#### 5. **Documentation Gaps**
- **Risk**: Users confused about migration
- **Impact**: 🟢 Low - Can be fixed quickly
- **Mitigation**:
  - Detailed migration guide (this doc!)
  - Code examples
  - FAQ section

#### 6. **Third-Party Integration**
- **Risk**: External tools depend on old pattern
- **Impact**: 🟢 Low - We control the ecosystem
- **Mitigation**:
  - Maintain backward compatibility
  - Deprecation period

---

## Proof of Concept Results

### Implementation Status

✅ **Complete**: `src/stickler/structured_object_evaluator/models/structured_model_test.py`

**What Works**:
- ✅ ComparableField as Pydantic model
- ✅ StructuredModel with auto-wrapping validator
- ✅ Annotated pattern for field definitions
- ✅ Smart serialization (clean vs full metadata)
- ✅ JSON Schema → StructuredModel conversion
- ✅ Dynamic model creation
- ✅ Template-based configuration

**Test Output**:
```
=== 1. Simple Model (defaults) ===
Created: name=ComparableField(John Doe) age=ComparableField(30)
Serialized: {'name': 'John Doe', 'age': 30}

=== 2. Configured Model (custom config in Annotated) ===
invoice_number.threshold: 0.9 (from Annotated)
Serialized with comp context: {'invoice_number': {'value': 'INV-2025-001', ...}}

=== 3. Creating StructuredModel from JSON Schema ===
✓ Created model: DynamicInvoice
Generated field annotations with proper thresholds/weights
```

---

## Recommendations

### Immediate Actions (Next Sprint)

1. **✅ POC Complete** - Review and validate
2. **Benchmark Performance** - Measure overhead
3. **Create Migration Script** - Automate test updates
4. **Stakeholder Review** - Get buy-in

### Short Term (1-2 Months)

1. **Implement Dual Support** - Both patterns work
2. **Migrate Examples** - Show new pattern
3. **Update Documentation** - Migration guide
4. **Start Test Migration** - Low-risk tests first

### Long Term (3-6 Months)

1. **Full Migration** - All code uses new pattern
2. **Deprecate Old Pattern** - Warnings in place
3. **Major Version Release** - 2.0.0 with new pattern
4. **Remove Old Code** - Clean codebase

---

## Conclusion

The **Annotated pattern migration** represents a significant improvement to Stickler's API:

**Pros**:
- ✅ Self-documenting code
- ✅ Cleaner, simpler implementation
- ✅ Better type safety
- ✅ JSON Schema integration
- ✅ Smart serialization

**Cons**:
- ⚠️ Breaking changes (mitigatable)
- ⚠️ Migration effort (~3-4 months)
- ⚠️ Test suite updates needed

**Verdict**: **Recommended** - Benefits outweigh costs, especially for long-term maintainability.

---

## Appendix A: Code Comparison

### Current Implementation Size

```
comparable_field.py:              212 lines (function-based)
structured_model.py:              2000+ lines (includes validators)
configuration_helper.py:          ~300 lines
field_helper.py:                  ~200 lines
```

### New Implementation Size

```
comparable_field_v2.py:           ~80 lines (class)
structured_model_base.py:         ~150 lines (base with validator)
schema_parser.py:                 ~120 lines
TOTAL:                            ~350 lines
```

**Code Reduction**: ~40% less code for core functionality

---

## Appendix B: Migration Script Example

```python
#!/usr/bin/env python3
"""
Automated migration script for ComparableField pattern.

Usage:
    python migrate_to_annotated.py <file_or_directory>
"""

import re
import sys
from pathlib import Path

def migrate_file(file_path: Path):
    """Migrate a single Python file to Annotated pattern."""
    content = file_path.read_text()
    
    # Pattern: field_name: Type = ComparableField(...)
    # Replace: field_name: Annotated[Type, ComparableField(...)]
    pattern = r'(\w+):\s*(\w+)\s*=\s*ComparableField\((.*?)\)'
    
    def replacer(match):
        field_name, type_name, args = match.groups()
        return f'{field_name}: Annotated[{type_name}, ComparableField({args})]'
    
    new_content = re.sub(pattern, replacer, content)
    
    # Add Annotated import if not present
    if 'from typing import' in new_content and 'Annotated' not in new_content:
        new_content = new_content.replace(
            'from typing import',
            'from typing import Annotated,'
        )
    
    file_path.write_text(new_content)
    print(f"✓ Migrated: {file_path}")

if __name__ == "__main__":
    target = Path(sys.argv[1])
    
    if target.is_file():
        migrate_file(target)
    else:
        for py_file in target.rglob("*.py"):
            migrate_file(py_file)
```

---

## Appendix C: FAQ

**Q: Do I need to migrate immediately?**  
A: No. Both patterns will be supported during the transition period (2-3 releases).

**Q: Will my existing code break?**  
A: Not immediately. Deprecation warnings will appear, but functionality remains.

**Q: How do I access field values?**  
A: Use `.value` attribute: `invoice.invoice_number.value`

**Q: Does this affect performance?**  
A: Minimal impact. Validator runs once during initialization.

**Q: Can I mix both patterns?**  
A: Yes, during migration. But recommended to use one pattern per model.

**Q: What about JSON Schema?**  
A: Enhanced! Can now bidirectionally convert between JSON Schema and models.

---

**Document Version**: 1.0  
**Last Updated**: October 29, 2025  
**Author**: Stickler Core Team  
**Status**: Proposal / RFC
