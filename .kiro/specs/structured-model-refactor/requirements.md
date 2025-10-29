# Requirements Document

## Introduction

The structured_model.py file is the core component of the Stickler library, responsible for structured data comparison using Pydantic models. Currently, this file contains 2584 lines of code with multiple responsibilities, making it difficult to read, maintain, and extend. This refactoring aims to decompose the monolithic StructuredModel class into a more modular, maintainable architecture while preserving all existing functionality and test compatibility.

## Phased Approach

This refactoring will be executed in three phases to minimize risk and allow for incremental validation:

### Phase 1: Foundation (Priority: CRITICAL)
Extract helper logic that's already partially separated and establish the delegation pattern. This phase has minimal risk since much of this code is already in helper classes.
- Requirements: 1, 6, 7, 10

### Phase 2: Core Comparison (Priority: HIGH)
Refactor the core comparison dispatch and field comparison logic. This is the heart of the refactoring and requires careful testing.
- Requirements: 1, 2, 3, 5, 10

### Phase 3: Metrics & Polish (Priority: MEDIUM)
Clean up metrics calculation and finalize the StructuredModel interface.
- Requirements: 1, 4, 8, 9, 10

Each phase maintains backward compatibility and passes all tests before proceeding to the next phase.

## Critical Analysis & Recommendations

### What's Good About This Plan:
1. **Phased approach reduces risk** - Each phase can be validated independently
2. **Many helpers already exist** - NonMatchesHelper, MetricsHelper, StructuredListComparator, etc. are already extracted
3. **Tests provide safety net** - Extensive test suite ensures we don't break functionality
4. **Clear separation of concerns** - Each phase targets a specific area

### Potential Risks:
1. **Circular dependencies** - StructuredModel and helpers may have circular references that need careful handling
2. **Performance regression** - Adding delegation layers could slow down hot paths
3. **Test brittleness** - Some tests may depend on internal implementation details
4. **Scope creep** - Easy to over-engineer the solution

### Recommended Priorities:

**MUST DO (Phase 1):**
- Extract model_from_json to ModelFactory (~100 lines saved)
- Move _collect_enhanced_non_matches to NonMatchesHelper (~50 lines saved)
- Extract _calculate_list_confusion_matrix to a helper (~100 lines saved)
- **Expected reduction: ~250 lines, minimal risk**

**SHOULD DO (Phase 2):**
- Extract _dispatch_field_comparison to ComparisonDispatcher (~200 lines saved)
- Extract _compare_primitive_list_with_scores to PrimitiveListComparator (~100 lines saved)
- Extract primitive comparison methods to FieldComparator (~100 lines saved)
- **Expected reduction: ~400 lines, moderate risk**

**NICE TO HAVE (Phase 3):**
- Extract _calculate_aggregate_metrics to MetricsHelper (~150 lines saved)
- Extract _add_derived_metrics_to_result to MetricsHelper (~100 lines saved)
- Clean up remaining helper methods (~100 lines saved)
- **Expected reduction: ~350 lines, low risk**

**Total Expected Reduction: ~1000 lines (from 2584 to ~1584)**
**Stretch Goal: Get to ~500 lines by moving more logic to helpers**

### Alternative Approach:
Instead of three phases, we could do **incremental extraction** where we extract one method at a time, run tests, commit, repeat. This is safer but slower. Given the extensive test coverage, the phased approach is recommended.

## Glossary

- **StructuredModel**: The main Pydantic BaseModel subclass that provides structured comparison capabilities
- **Comparison Engine**: The core logic that performs field-by-field comparisons between model instances
- **Confusion Matrix**: Statistical classification metrics (TP, FP, FN, TN, FD, FA) used to evaluate comparison results
- **Hungarian Matching**: Algorithm used for optimal matching of unordered list elements
- **Field Dispatch**: The process of routing field comparisons to appropriate handlers based on field type
- **Recursive Comparison**: The process of comparing nested StructuredModel objects hierarchically
- **Non-Match Documentation**: System for tracking and reporting fields that don't match between compared instances
- **Aggregate Metrics**: Rolled-up confusion matrix metrics from child fields to parent containers
- **Threshold Gating**: Logic that determines whether to perform recursive analysis based on similarity thresholds

## Requirements

### Requirement 1: Maintain Complete Backward Compatibility (ALL PHASES)

**User Story:** As a library user, I want the refactored code to work identically to the current implementation, so that my existing code continues to function without changes.

**Phase:** All phases - This is a cross-cutting requirement that applies to every change

**Priority:** CRITICAL - Non-negotiable for every phase

#### Acceptance Criteria

1. WHEN the refactored code is deployed, THE System SHALL pass all existing unit tests without modification
2. WHEN comparing two StructuredModel instances, THE System SHALL produce identical results to the current implementation
3. WHEN using the public API methods (compare, compare_with, compare_field), THE System SHALL maintain identical method signatures and return types
4. WHEN creating dynamic models via model_from_json, THE System SHALL produce functionally equivalent model classes
5. WHERE JSON schema generation is used, THE System SHALL produce identical schema output with comparison metadata

**Critique:** This is the most important requirement. Every change must be validated against the full test suite.

### Requirement 2: Separate Comparison Logic from Model Definition (PHASE 2)

**User Story:** As a developer, I want the comparison logic separated from the model definition, so that I can understand and modify each concern independently.

**Phase:** Phase 2 - Core Comparison

**Priority:** HIGH - This is the main goal of the refactoring

#### Acceptance Criteria

1. THE System SHALL extract all comparison dispatch logic into a dedicated ComparisonDispatcher class
2. THE System SHALL extract all field comparison logic into dedicated field comparator classes
3. THE System SHALL maintain the StructuredModel class as a thin wrapper that delegates to comparison components
4. WHEN reading the StructuredModel class, THE System SHALL present a clear, concise interface without implementation details
5. THE System SHALL use composition over inheritance for comparison functionality

**Critique:** This is the core of the refactoring. The dispatch logic (_dispatch_field_comparison) is the most complex part and needs careful extraction.

### Requirement 3: Implement Clear Type-Based Dispatch (PHASE 2)

**User Story:** As a developer, I want to easily trace how different field types are compared, so that I can debug issues and add new comparison types.

**Phase:** Phase 2 - Core Comparison

**Priority:** HIGH - Essential for maintainability

#### Acceptance Criteria

1. THE System SHALL implement a switch-statement or match-based dispatch for field type routing
2. WHEN dispatching a field comparison, THE System SHALL route to exactly one handler based on field type
3. THE System SHALL provide clear separation between primitive, list, and structured field comparisons
4. THE System SHALL document the dispatch decision tree in code comments
5. WHEN adding a new field type, THE System SHALL require changes in only one dispatch location

**Critique:** The current _dispatch_field_comparison method already uses match statements, which is good. We need to preserve this clarity while extracting it.

### Requirement 4: Modularize Confusion Matrix Calculation (PHASE 3)

**User Story:** As a developer, I want confusion matrix logic separated into its own module, so that I can understand and modify metrics calculation independently.

**Phase:** Phase 3 - Metrics & Polish

**Priority:** MEDIUM - Important but can be done after core comparison is refactored

#### Acceptance Criteria

1. THE System SHALL extract all confusion matrix calculation logic into a dedicated ConfusionMatrixCalculator class
2. THE System SHALL extract aggregate metrics calculation into a separate method or class
3. THE System SHALL extract derived metrics calculation into a separate method or class
4. WHEN calculating confusion matrices, THE System SHALL use a single traversal of the comparison tree
5. THE System SHALL maintain the hierarchical structure of confusion matrix results

**Critique:** Much of this is already in MetricsHelper. We mainly need to extract _calculate_aggregate_metrics and _add_derived_metrics_to_result.

### Requirement 5: Separate List Comparison Logic (PHASE 2)

**User Story:** As a developer, I want list comparison logic (both primitive and structured) in dedicated modules, so that I can understand and modify list handling independently.

**Phase:** Phase 2 - Core Comparison

**Priority:** HIGH - List comparison is a major part of the complexity

#### Acceptance Criteria

1. THE System SHALL maintain the existing StructuredListComparator for List[StructuredModel] comparisons
2. THE System SHALL create a PrimitiveListComparator for List[primitive] comparisons
3. THE System SHALL extract Hungarian matching coordination logic into a dedicated class
4. WHEN comparing lists, THE System SHALL delegate to the appropriate list comparator based on element type
5. THE System SHALL maintain object-level and field-level metrics separation for structured lists

**Critique:** StructuredListComparator already exists. We need to extract _compare_primitive_list_with_scores into a similar pattern.

### Requirement 6: Extract Non-Match Documentation (PHASE 1)

**User Story:** As a developer, I want non-match documentation logic in its own module, so that I can understand and extend error reporting independently.

**Phase:** Phase 1 - Foundation

**Priority:** CRITICAL - Already mostly extracted, just needs cleanup

#### Acceptance Criteria

1. THE System SHALL maintain the existing NonMatchesHelper for non-match collection
2. THE System SHALL extract enhanced non-match collection into dedicated methods
3. THE System SHALL separate object-level and field-level non-match documentation
4. WHEN documenting non-matches, THE System SHALL provide clear field paths and non-match types
5. THE System SHALL maintain the NonMatchField and NonMatchType structures

**Critique:** NonMatchesHelper already exists. We mainly need to move _collect_enhanced_non_matches and _collect_non_matches to the helper.

### Requirement 7: Simplify Dynamic Model Creation (PHASE 1)

**User Story:** As a developer, I want the model_from_json logic separated from the core StructuredModel class, so that I can understand dynamic model creation independently.

**Phase:** Phase 1 - Foundation

**Priority:** CRITICAL - Low risk extraction that reduces StructuredModel size significantly

#### Acceptance Criteria

1. THE System SHALL extract model_from_json logic into a ModelFactory class
2. THE System SHALL extract field validation logic into the ModelFactory
3. THE System SHALL maintain the existing field_converter module for field definition conversion
4. WHEN creating dynamic models, THE System SHALL validate all field configurations before model creation
5. THE System SHALL maintain support for nested schema validation

**Critique:** This is a large method (~100 lines) that can be easily extracted with minimal risk. Good candidate for Phase 1.

### Requirement 8: Reduce StructuredModel Class Size (PHASE 3)

**User Story:** As a developer, I want the StructuredModel class to be under 500 lines, so that I can quickly understand its responsibilities and interface.

**Phase:** Phase 3 - Metrics & Polish

**Priority:** MEDIUM - This is the outcome of all other refactorings

#### Acceptance Criteria

1. THE System SHALL reduce the StructuredModel class to fewer than 500 lines of code
2. THE System SHALL maintain only public API methods and essential Pydantic overrides in StructuredModel
3. THE System SHALL delegate all implementation details to helper classes
4. WHEN reading the StructuredModel class, THE System SHALL present a clear public interface within 200 lines
5. THE System SHALL document the delegation pattern in class-level docstrings

**Critique:** This is a goal, not a requirement. It will naturally happen as we extract logic. Target: ~400 lines after all phases.

### Requirement 9: Improve Code Readability (PHASE 3)

**User Story:** As a developer, I want clear, well-documented code with single responsibilities, so that I can quickly understand and modify the comparison logic.

**Phase:** Phase 3 - Metrics & Polish

**Priority:** MEDIUM - Quality improvement that happens throughout but finalized in Phase 3

#### Acceptance Criteria

1. THE System SHALL ensure each class has a single, well-defined responsibility
2. THE System SHALL ensure each method is fewer than 50 lines of code
3. THE System SHALL provide clear docstrings for all public methods
4. THE System SHALL use descriptive variable names that indicate purpose
5. THE System SHALL minimize nested conditionals through early returns and guard clauses

**Critique:** This is a continuous improvement goal. Apply during all phases but don't let it block progress.

### Requirement 10: Maintain Performance Characteristics (ALL PHASES)

**User Story:** As a library user, I want the refactored code to maintain current performance levels, so that my applications don't slow down.

**Phase:** All phases - Performance must be validated at each phase

**Priority:** CRITICAL - Non-negotiable for every phase

#### Acceptance Criteria

1. THE System SHALL maintain single-traversal comparison for efficiency
2. THE System SHALL avoid redundant Hungarian matching calculations
3. THE System SHALL maintain lazy evaluation where currently implemented
4. WHEN comparing large nested structures, THE System SHALL complete in comparable time to the current implementation
5. THE System SHALL not introduce additional object allocations in hot paths

**Critique:** Performance is critical. Use delegation (passing self) rather than copying data. Avoid creating intermediate objects in loops.
