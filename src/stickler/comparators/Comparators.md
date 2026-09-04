# Comparators Guide

## Overview
The comparator system provides a flexible framework for measuring similarity between values, with each comparator returning a score between 0.0 and 1.0. All comparators inherit from the BaseComparator class, which provides common functionality like thresholding and binary comparison capabilities.

`BaseComparator.compare()` is a template method: it applies the shared `None` policy (see Best Practices below) and then delegates to `_compare()`. Comparators implement `_compare()` and never receive `None`.

## Available Comparators

### 1. Basic Comparators

#### Exact Comparator
```python
comparator = ExactComparator(threshold=1.0, case_sensitive=True)
```
- Performs exact string matching without normalization
- Options for case-sensitive comparison
- Returns 1.0 for exact matches, 0.0 otherwise
- Useful for strict matching requirements

#### Normalized Comparator
```python
comparator = NormalizedComparator(
    case_sensitive=False,
    ignore_whitespace=True,
    ignore_punctuation=True,
)
```
- Performs equality after independently configured text transforms
- Removes Unicode whitespace and Unicode `P*` punctuation by default
- Preserves symbols such as `$`, `±`, and emoji, plus accents and combining marks
- Uses NFC normalization and Unicode case folding
- Useful when formatting differences are noise but typos should not match

#### Numeric Comparator
```python
comparator = NumericComparator()
```
- Specialized for comparing numeric values
- Configurable absolute and relative tolerance for floating-point comparisons
- Handles type conversion from strings
- Ideal for financial data or measurement comparisons

#### Date Comparator
```python
comparator = DateComparator()
```
- Deterministic, non-LLM comparison of date fields (parses with `python-dateutil`)
- Normalizes surface form: separators, padding, named months, ISO vs slash, two-digit years, weekday prefixes
- Configurable tolerance, locale (`dayfirst`), partial-year handling, date ranges (`range_mode`), and reduced-precision handling (`precision_mode`)
- See the [DateComparator guide](../../../docs/docs/Guides/Comparators/date-comparator.md) for the full behavior reference

### 2. String Distance Comparators

#### Levenshtein Comparator
```python
comparator = LevenshteinComparator(normalize=True, threshold=0.7)
```
- Uses Levenshtein edit distance algorithm
- Normalizes distance to a 0.0-1.0 similarity score
- Optional string normalization (whitespace, case)
- Effective for catching typos and minor variations

#### Fuzzy Comparator
```python
comparator = FuzzyComparator(method="ratio", normalize=True, threshold=0.7)
```
- Advanced string matching using thefuzz library
- Multiple matching methods:
  - `ratio`: Standard Levenshtein distance ratio
  - `partial_ratio`: Partial string matching
  - `token_sort_ratio`: Token-based matching with sorting
  - `token_set_ratio`: Token-based matching with set operations
- Handles word reordering and partial matches

### 3. Semantic Comparators

#### BERT Comparator
```python
comparator = BERTComparator(threshold=0.7)
```
- Uses BERTScore for semantic similarity
- Based on distilbert-base-uncased model
- Returns F1 score as similarity measure
- Handles semantic variations well
- Falls back to exact matching if BERT fails

#### Semantic Comparator
```python
comparator = SemanticComparator(
    model_id="amazon.titan-embed-text-v2:0",
    sim_function="cosine_similarity",
    threshold=0.7
)
```
- Uses embeddings from Bedrock models
- Default: Titan embedding model
- Cosine similarity between embeddings
- Support for custom embedding functions
- Requires BedrockClient or custom embedding function

#### LLM Comparator
```python
comparator = LLMComparator(
    prompt=prompt_template,
    model_id=model_id,
    temp=0.5,
    threshold=0.5
)
```
- Uses LLM to determine semantic equivalence
- Customizable prompt templates
- Configurable temperature
- Binary output (1.0 or 0.0)
- Note: Implementation in progress

### 4. Structured Data Comparators

#### Structured Model Comparator
```python
comparator = StructuredModelComparator()
```
- Specialized for comparing structured model objects
- Handles nested data structures
- Integrates with evaluation framework
- Supports custom field comparators

### 5. Spatial Comparators

#### BBox IoU Comparator
```python
comparator = BBoxIoUComparator(threshold=0.5)
```
- Compares two bounding boxes using Intersection over Union (IoU)
- Accepts two-point `[[x1, y1], [x2, y2]]` and flat `[x1, y1, x2, y2]` formats
- Normalizes coordinates; non-finite values (NaN/inf) score 0.0
- Returns IoU as the similarity score (0.0-1.0)
- For end-to-end mean Average Precision (mAP) scoring over fields, see the
  Bounding Box mAP Metrics documentation rather than using this comparator directly

## Best Practices

1. **Choosing the Right Comparator**
   - Use ExactComparator for strict matching requirements
   - Use NormalizedComparator for formatting-insensitive equality
   - Use FuzzyComparator for general string matching with tolerance for variations
   - Use semantic comparators (BERT/Semantic) for meaning-based comparison
   - Use LevenshteinComparator for simple edit distance-based matching
   - Use NumericComparator for number comparisons
   - Use BBoxIoUComparator for bounding-box / spatial localization comparisons

2. **Threshold Selection**
   - Higher thresholds (>0.9) for strict matching
   - Medium thresholds (0.7-0.8) for general use
   - Lower thresholds (<0.7) for more lenient matching
   - Consider your use case's precision/recall requirements

3. **Error Handling**
   - `None` is handled once, in `BaseComparator.compare()`, for every
     comparator: `(None, None)` scores 1.0, and `None` against a present
     value scores 0.0. `None` is a *missing* value and never equals `""`,
     which is a *present but empty* one
   - Most provide fallback behavior if primary comparison fails
   - Check return values for expected range (0.0-1.0)

4. **Performance Considerations**
   - Semantic comparators are more computationally intensive
   - Levenshtein and Fuzzy comparators are moderate
   - Exact and Numeric comparators are fastest
   - Cache embeddings for repeated semantic comparisons

## Examples

### Basic Comparison
```python
from stickler.comparators import ExactComparator

comparator = ExactComparator(case_sensitive=False)
score = comparator.compare("Hello, world!", "hello world")  # Returns 1.0
```

### Fuzzy Matching
```python
from stickler.comparators import FuzzyComparator

comparator = FuzzyComparator(method="token_sort_ratio")
score = comparator.compare("Python programming", "programming in python")  # High score
```

### Semantic Similarity
```python
from stickler.comparators import SemanticComparator

comparator = SemanticComparator()
score = comparator.compare(
    "The cat sat on the mat",
    "A feline was resting on the rug"
)  # High score due to semantic similarity
```

### Binary Comparison
```python
from stickler.comparators import LevenshteinComparator

comparator = LevenshteinComparator(threshold=0.8)
tp, fp = comparator.binary_compare("color", "colour")  # Returns (1, 0) if score >= 0.8
```
