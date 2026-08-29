# Comparators

Comparators are the algorithms that determine how similar two field values are. Each comparator is optimized for a different data type or comparison strategy, returning a similarity score between 0.0 (completely different) and 1.0 (identical). When you define a `StructuredModel`, you assign a comparator to each field so Stickler knows *how* to evaluate that field.

---

## Which Comparator Should I Use?

| Comparator | Best For | Speed | Needs AWS? | Score Type |
|---|---|---|---|---|
| [**ExactComparator**](#exactcomparator) | IDs, codes, booleans | Instant | No | Binary (0.0 or 1.0) |
| [**LevenshteinComparator**](#levenshteincomparator) | Names, addresses, text with typos | Instant | No | Continuous (0.0--1.0) |
| [**NumericComparator**](#numericcomparator) | Prices, quantities, measurements | Instant | No | Binary (0.0 or 1.0) |
| [**DateComparator**](date-comparator.md) | Date fields with mixed formats, partial dates, ranges | Instant | No | Continuous (0.0--1.0) |
| [**FuzzyComparator**](#fuzzycomparator) | Flexible text, descriptions, reordered tokens | Fast | No | Continuous (0.0--1.0) |
| [**BBoxIoUComparator**](#bboxioucomparator) | Bounding boxes, spatial localization | Instant | No | Continuous (0.0--1.0) |
| [**SemanticComparator**](#semanticcomparator) | Meaning-based text similarity | Moderate | Yes (Bedrock) | Continuous (0.0--1.0) |
| [**BERTComparator**](#bertcomparator) | Contextual semantic similarity | Moderate | No (runs locally) | Continuous (0.0--1.0) |
| [**LLMComparator**](#llmcomparator) | Complex semantic evaluation with reasoning | Slow | Yes (Bedrock) | Binary (0.0 or 1.0) |

---

## Comparator Details

### ExactComparator

Checks for exact string matching after normalizing whitespace, punctuation, and (by default) case. Returns 1.0 for exact matches and 0.0 otherwise.

**When to use:** Critical identifiers, status codes, booleans, or any field where partial matches are meaningless.

```python
from stickler import StructuredModel, ComparableField
from stickler import ExactComparator

class Order(StructuredModel):
    order_id: str = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=3.0
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `1.0` | Similarity threshold for binary classification |
| `case_sensitive` | `False` | Whether comparison is case-sensitive |

---

### LevenshteinComparator

Calculates the Levenshtein edit distance between two strings and returns a normalized similarity score: `1.0 - (edit_distance / max_length)`. Optionally normalizes input by stripping whitespace and lowercasing.

**When to use:** Names, addresses, free-form text where typos and minor variations are expected. This is the **default comparator** for string fields.

```python
from stickler import StructuredModel, ComparableField
from stickler import LevenshteinComparator

class Contact(StructuredModel):
    name: str = ComparableField(
        comparator=LevenshteinComparator(threshold=0.8),
        weight=1.5
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.7` | Similarity threshold for binary classification |
| `normalize` | `True` | Strip whitespace and lowercase before comparison |

---

### NumericComparator

Extracts numeric values from strings or numbers and compares them with configurable tolerance. Handles currency symbols, commas, and accounting notation (e.g., `(123)` for negative values). Returns 1.0 if the numbers match within tolerance, 0.0 otherwise.

**When to use:** Prices, quantities, measurements, or any numeric field where small differences are acceptable.

```python
from stickler import StructuredModel, ComparableField
from stickler import NumericComparator

class Invoice(StructuredModel):
    amount: float = ComparableField(
        comparator=NumericComparator(relative_tolerance=0.05),
        weight=2.0
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `1.0` | Similarity threshold for binary classification |
| `relative_tolerance` | `0.0` | Relative tolerance (e.g., `0.1` = 10%) |
| `absolute_tolerance` | `0.0` | Absolute tolerance (e.g., `0.01` for cents) |
| `tolerance` | `None` | Alias for `absolute_tolerance` (backward compatibility) |

---

### DateComparator

Parses both sides as dates (or date ranges) using `python-dateutil`, then scores the comparison on a tier system: same calendar day after surface-form normalization is `1.0`, partial information (year-less, range-vs-single) is configurable, anything else is `0.0`. Handles ISO/named-month/slash formats, two-digit years, day-of-week prefixes, and timezone-aware datetimes.

**When to use:** Any date field where surface form varies (different separators, formats, locales, two- vs four-digit years), or where ground truth and predictions may disagree on year-presence or use ranges.

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import DateComparator

class Invoice(StructuredModel):
    invoice_date: str = ComparableField(
        comparator=DateComparator(allow_partial_year=True),
        threshold=0.7
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `tolerance` | `timedelta(0)` | Allowed difference for same-day comparisons (single dates only) |
| `dayfirst` | `None` | Interpretation hint for ambiguous numeric dates |
| `allow_partial_year` | `False` | If `True`, year-less ↔ year-bearing pairs with matching m/d score `0.7` |
| `range_mode` | `"graded"` | How range comparisons are scored: `"strict"`, `"reject"`, `"contains"`, or `"graded"` |
| `precision_mode` | `"exact"` | How month/day resolution mismatches score (`Jan 2024` vs `Jan 1, 2024`): `"exact"`, `"gt_loose"`, or `"overlap"` |

For the full behavior reference, configuration matrix, and corner cases, see the [DateComparator page](date-comparator.md).

---

### FuzzyComparator

Uses the [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) library for advanced fuzzy string matching. Supports multiple matching methods including standard ratio, partial matching, and token-based matching that is order-independent.

**When to use:** Descriptions, product names, or text where word order may vary or partial matches are valuable.

```python
from stickler import StructuredModel, ComparableField
from stickler import FuzzyComparator

class Product(StructuredModel):
    description: str = ComparableField(
        comparator=FuzzyComparator(method="token_sort_ratio"),
        threshold=0.7
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.7` | Similarity threshold for binary classification |
| `method` | `"ratio"` | Matching method: `"ratio"`, `"partial_ratio"`, `"token_sort_ratio"`, or `"token_set_ratio"` |
| `normalize` | `True` | Strip whitespace and lowercase before comparison |

**Matching methods explained:**

- `ratio` -- Standard Levenshtein distance ratio (similar to `LevenshteinComparator` but using rapidfuzz's optimized implementation).
- `partial_ratio` -- Finds the best partial match within the longer string. Good when one value is a substring of the other.
- `token_sort_ratio` -- Splits strings into tokens, sorts them, then compares. Handles reordered words (e.g., "John Smith" vs "Smith John").
- `token_set_ratio` -- Splits into token sets, comparing the intersection and remainder. Handles extra or missing words.

!!! note "Dependency"
    FuzzyComparator requires the `rapidfuzz` package. Install it with: `pip install rapidfuzz`

---

### BBoxIoUComparator

Compares two bounding boxes using Intersection over Union (IoU) as the similarity score. Accepts both two-point (`[[x1, y1], [x2, y2]]`) and flat (`[x1, y1, x2, y2]`) formats. Coordinates are automatically normalized so that x1 <= x2 and y1 <= y2.

**When to use:** Evaluating spatial localization accuracy for document fields, signature detection, logo identification, or any use case where you need to measure how well a predicted bounding box overlaps with ground truth.

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import BBoxIoUComparator

class DocumentField(StructuredModel):
    bbox: list = ComparableField(
        comparator=BBoxIoUComparator(threshold=0.5),
        weight=1.0
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.5` | IoU threshold for binary match classification |

!!! tip "Rich Value Pattern"
    For end-to-end mAP evaluation with per-field breakdown, use the [Bounding Box mAP Metrics](../../Advanced/bbox-map-metrics.md) feature instead of using BBoxIoUComparator directly.

---

### SemanticComparator

Uses AWS Bedrock Titan embeddings to generate vector representations of text, then computes cosine similarity. Captures meaning rather than surface-level string similarity.

**When to use:** Text fields where meaning matters more than exact wording. See [LLM-as-a-Judge Comparators](llm-as-a-judge.md) for a detailed guide.

```python
from stickler import StructuredModel, ComparableField
from stickler import SemanticComparator

class Review(StructuredModel):
    summary: str = ComparableField(
        comparator=SemanticComparator(threshold=0.8),
        weight=1.0
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.7` | Similarity threshold for binary classification |
| `model_id` | `"amazon.titan-embed-text-v2:0"` | Bedrock embedding model ID |
| `sim_function` | `"cosine_similarity"` | Similarity function to use |
| `embedding_function` | `None` | Optional custom embedding function (bypasses Bedrock) |

---

### BERTComparator

Uses the BERTScore metric (via the `evaluate` library) to calculate contextual semantic similarity. Returns the F1 score component of BERTScore as the similarity measure. Runs entirely locally -- no API calls required.

**When to use:** Text fields where you need semantic understanding without cloud dependencies. See [LLM-as-a-Judge Comparators](llm-as-a-judge.md) for a detailed guide.

```python
from stickler import StructuredModel, ComparableField
from stickler import BERTComparator

class Document(StructuredModel):
    summary: str = ComparableField(
        comparator=BERTComparator(threshold=0.85),
        weight=1.0
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.7` | Similarity threshold for binary classification |

The default model is `distilbert-base-uncased`, loaded globally via the `evaluate` library.

---

### LLMComparator

Uses a Large Language Model (via AWS Bedrock and the `strands-agents` library) to perform intelligent semantic comparisons. The LLM receives both values and optional evaluation guidelines, then returns a binary equivalence judgment. This is the most flexible comparator but also the most expensive.

**When to use:** Complex comparisons that require reasoning, domain-specific logic, or understanding of abbreviations and conventions. See [LLM-as-a-Judge Comparators](llm-as-a-judge.md) for a detailed guide.

```python
from stickler import StructuredModel, ComparableField
from stickler import LLMComparator

class Address(StructuredModel):
    street: str = ComparableField(
        comparator=LLMComparator(
            model="us.amazon.nova-lite-v1:0",
            eval_guidelines="Consider street abbreviations equivalent (St=Street, Ave=Avenue)"
        ),
        threshold=0.8
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | *Required* | Bedrock model ID string or a `strands.models.Model` instance |
| `eval_guidelines` | `None` | Custom guidelines for the LLM to follow during comparison |

!!! note "Dependency"
    LLMComparator requires the `strands-agents` package. Install it with: `pip install stickler-eval[llm]`

---

## Default Comparators by Type

When you do not specify a comparator in `ComparableField`, Stickler assigns one based on the JSON schema type of the field:

| JSON Schema Type | Default Comparator | Default Threshold | Rationale |
|---|---|---|---|
| `string` | LevenshteinComparator | 0.5 | Handles typos and minor variations |
| `number` | NumericComparator | 0.5 | Tolerates small numeric differences |
| `integer` | NumericComparator | 0.5 | Tolerates small numeric differences |
| `boolean` | ExactComparator | 1.0 | Must be exactly true or false |
| `array` (primitives) | Based on item type | Based on item type | Inherits from element type |
| `array` (objects) | Hungarian matching | 0.7 | Optimal pairing of list elements |
| `object` | Recursive comparison | 0.7 | Field-by-field nested comparison |

---

## Custom Comparators

You can create your own comparator by extending `BaseComparator`. The only requirement is implementing the `_compare` method, which takes two values and returns a float between 0.0 and 1.0.

### The BaseComparator Interface

`compare` is a template method: it applies the shared `None` policy, then delegates to `_compare`. You implement `_compare`; callers call `compare`.

```python
from stickler import BaseComparator

class BaseComparator(ABC):
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def compare(self, str1: Any, str2: Any) -> float:
        """Apply the shared None policy, then delegate to _compare."""
        if str1 is None and str2 is None:
            return 1.0
        if str1 is None or str2 is None:
            return 0.0
        return self._compare(str1, str2)

    @abstractmethod
    def _compare(self, str1: Any, str2: Any) -> float:
        """Compare two present values. Neither argument is ever None.

        Args:
            str1: First value, never None
            str2: Second value, never None

        Returns:
            Similarity score between 0.0 and 1.0
        """
        pass
```

`BaseComparator` also provides:

- `__call__` -- makes the comparator callable directly (delegates to `compare`).
- `binary_compare` -- converts the continuous similarity score to a `(tp, fp)` tuple based on the threshold.

### None Handling

Every comparator treats `None` uniformly: two `None` values are an exact match (`1.0`); a `None` compared against a present value is a non-match (`0.0`). `None` is a *missing* value and never equals an empty string, which is a *present but empty* one.

The policy lives in `BaseComparator.compare` and nowhere else. You don't write any `None`-handling code in your comparator -- by the time `_compare` runs, both arguments are guaranteed not to be `None`.

!!! warning "Implement `_compare`, not `compare`"
    Overriding `compare` directly bypasses the `None` policy and reintroduces exactly the divergence this design prevents. Implement `_compare`.

    A comparator written against the older interface — implementing `compare` — still constructs and behaves exactly as written, and emits a `DeprecationWarning` naming the rename. It does **not** receive the `None` policy, since its `compare` shadows the template method. That shim is removed in 1.0 ([#215](https://github.com/awslabs/stickler/issues/215)), after which such a comparator raises `TypeError` at construction.

### Example: Custom RegexComparator

```python
import re
from typing import Any
from stickler import BaseComparator

class RegexComparator(BaseComparator):
    """Comparator that checks if a value matches a reference regex pattern."""

    def __init__(self, threshold: float = 1.0):
        super().__init__(threshold=threshold)

    def _compare(self, pattern: Any, value: Any) -> float:
        # pattern and value are never None here -- BaseComparator handles that.
        try:
            return 1.0 if re.fullmatch(str(pattern), str(value)) else 0.0
        except re.error:
            return 0.0
```

Use it like any built-in comparator:

```python
from stickler import StructuredModel, ComparableField

class PhoneRecord(StructuredModel):
    phone: str = ComparableField(
        comparator=RegexComparator(),
        threshold=1.0
    )
```

---

## Next Steps

For a deep dive into the three AI-powered comparators (SemanticComparator, BERTComparator, and LLMComparator), see [LLM-as-a-Judge Comparators](llm-as-a-judge.md).
