---
title: Classification Logic
---

# Classification Logic for Evaluation Metrics

This document defines the classification logic used in the stickler library for evaluating predictions against ground truth.

## Core Definitions

The confusion matrix metrics classify comparisons into five categories:

| Category | Abbreviation | Definition |
|----------|--------------|------------|
| True Positive (TP) | TP | GT != null, EST != null, GT == EST (match above threshold) |
| False Alarm (FA) | FA | GT == null, EST != null |
| True Negative (TN) | TN | GT == null, EST == null |
| False Negative (FN) | FN | GT != null, EST == null |
| False Discovery (FD) | FD | GT != null, EST != null, GT != EST (match below threshold) |

False alarm (FA) and false discovery (FD) are a subset of false positives (FP). 

FP = FA + FD

Where:
- GT = Ground Truth
- EST = Estimate (Prediction)

## Classification Logic by Data Type

### Simple Values (Strings, Numbers, etc.)

| Ground Truth | Prediction | Classification | Explanation |
|--------------|------------|----------------|-------------|
| "value" | "value" | TP | Exact match |
| "value" | "similar" | FD | Both non-null but don't match above threshold |
| "value" | null | FN | Missing prediction for existing ground truth |
| null | "value" | FP | Prediction exists but no ground truth (False Alarm) |
| null | null | TN | Correctly predicted absence |
| "" (empty) | null | Treated as TN | Empty strings are treated as null |
| null | "" (empty) | Treated as TN | Empty strings are treated as null |

### Lists

For lists, we use the Hungarian algorithm to find optimal matching between elements:

1. **Empty Lists**:
   - GT = [], EST = [] → TN (both empty)
   - GT = [], EST = ["item"] → FA (False Alarm for each prediction item)
   - GT = ["item"], EST = [] → FN (False Negative for each ground truth item)

2. **Element Matching**:
   - Each element in GT is matched with at most one element in EST
   - Each element in EST is matched with at most one element in GT
   - Matching maximizes overall similarity

3. **Classification of Matched Elements**:
   - If similarity ≥ threshold → TP
   - If similarity < threshold → FD (False Discovery)

4. **Classification of Unmatched Elements**:
   - Unmatched GT elements → FN
   - Unmatched EST elements → FA (False Alarm)

#### Example 1: Mixed Matching

GT = ["red", "blue", "green"]  
EST = ["red", "yellow", "orange", "blue"]

Matching:
- "red" matches "red" → TP
- "blue" matches "blue" → TP
- "green" has no match → FN
- "yellow" has no match in GT → FA (False Alarm)
- "orange" has no match in GT → FA (False Alarm)

Result: TP=2, FP=A, TN=0, FN=1, FD=0

#### Example 2: Similar But Not Exact

GT = ["apple", "banana", "cherry"]  
EST = ["aple", "bananna", "cheery"]

Matching (assuming threshold = 0.7):
- "apple" matches "aple" with similarity 0.8 → TP
- "banana" matches "bananna" with similarity 0.85 → TP
- "cherry" matches "cheery" with similarity 0.83 → TP

Result: TP=3, FA=0, TN=0, FN=0, FD=0

#### Example 3: Below Threshold

GT = ["apple", "banana", "cherry"]  
EST = ["appx", "bnn", "chry"]

Matching (assuming threshold = 0.7):
- "apple" matches "appx" with similarity 0.5 → FD
- "banana" matches "bnn" with similarity 0.6 → FD
- "cherry" matches "chry" with similarity 0.65 → FD

Result: TP=0, FA=0, TN=0, FN=0, FD=3

### Nested Objects/Dictionaries

For nested objects, we apply the classification logic recursively:

1. **Empty Objects**:
   - GT = {}, EST = {} → TN
   - GT = {}, EST = {key: value} → FA (False Alarm)
   - GT = {key: value}, EST = {} → FN

2. **Field Matching**:
   - Each field is evaluated independently
   - Fields present in both GT and EST are compared for similarity
   - Fields present in only one are classified as FA or FN

3. **Classification of Fields**:
   - If both have field and similarity ≥ threshold → TP
   - If both have field and similarity < threshold → FD (False Discovery)
   - If only GT has field → FN
   - If only EST has field → FA (False Alarm)

#### Example:

GT = {name: "John", age: 30, address: "123 Main St"}  
EST = {name: "John", age: 31, phone: "555-1234"}

Field-by-field:
- name: Both have it, exact match → TP
- age: Both have it, but different → FD
- address: Only in GT → FN
- phone: Only in EST → FA (False Alarm)

Result: TP=1, FA=1, TN=0, FN=1, FD=1

## Derived Metrics

From the base confusion matrix counts, we derive the following metrics:

1. **Precision**: TP / (TP + FP)
   - Measures how many of the predicted values are correct

2. **Recall**: TP / (TP + FN)
   - Measures how many of the ground truth values are correctly predicted

3. **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
   - Harmonic mean of precision and recall

4. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
   - Overall correctness of predictions

## Edge Cases and Clarifications

### 1. Null vs. Empty Equivalence

**Design Decision**: Empty collections and null values are treated as equivalent in all comparisons.

- Empty strings (""), empty lists ([]), and empty objects ({}) are treated as null values
- This means comparing null with an empty collection results in TN (True Negative)
- **Examples**:
  - GT = `null`, EST = `[]` → TN (equivalent states representing "no data")
  - GT = `[]`, EST = `null` → TN (equivalent states representing "no data")
  - GT = `""`, EST = `null` → TN (equivalent states representing "no data")
  - GT = `{}`, EST = `null` → TN (equivalent states representing "no data")

**Rationale**: Semantically, both null values and empty collections represent the absence of meaningful data. Distinguishing between these states would introduce unnecessary complexity and inconsistency in evaluation metrics. For practical evaluation purposes, "no data" should be treated uniformly regardless of its representation.

### 2. Threshold Boundary

- Values exactly at the threshold are considered matches (TP)
- For example, if threshold = 0.7 and similarity = 0.7, this is a TP

### 3. List Order

- List order doesn't matter for matching
- The Hungarian algorithm finds the optimal matching regardless of order

### 4. Partial Matches in Lists

- For lists, we don't have "partial credit" for individual elements
- Each element is classified as TP, FA, FN, or FD independently

### 5. Nested Lists

- For lists of objects, we apply the Hungarian algorithm at the list level
- Each matched pair of objects is then evaluated recursively

### 6. Missing Fields vs. Null Fields

- A missing field and a field with null value are treated differently:
  - Missing field in EST when GT has it → FN
  - Null field in EST when GT has non-null → FN
  - Missing field in GT when EST has it → FA (False Alarm)
  - Null field in GT when EST has non-null → FA (False Alarm)

## Summary of Key Points

1. **False Alarm (FA)** occurs when the prediction includes something that doesn't exist in the ground truth (GT is null, EST is not null)

2. **False Discovery (FD)** occurs when the prediction recognizes something that exists but gets it wrong (both GT and EST are not null, but they don't match)

3. **List matching** uses the Hungarian algorithm to find optimal pairings, with unmatched items classified as FP or FN

4. **Nested structures** are evaluated recursively, with each field or element classified independently

5. **Empty collections** are generally treated as null values for classification purposes
