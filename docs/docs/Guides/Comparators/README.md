# Comparators

Comparators are the algorithms that determine how similar two field values are. Each comparator is optimized for a different data type or comparison strategy, returning a similarity score between 0.0 (completely different) and 1.0 (identical). When you define a `StructuredModel`, you assign a comparator to each field so Stickler knows *how* to evaluate that field.

---

## Which Comparator Should I Use?

| Comparator | Best For | Speed | Needs AWS? | Score Type |
|---|---|---|---|---|
| [**ExactComparator**](#exactcomparator) | IDs, codes, booleans | Instant | No | Binary (0.0 or 1.0) |
| [**NormalizedComparator**](#normalizedcomparator) | Text where formatting differences are noise | Instant | No | Binary (0.0 or 1.0) |
| [**LevenshteinComparator**](#levenshteincomparator) | Names, addresses, text with typos | Instant | No | Continuous (0.0--1.0) |
| [**NumericComparator**](#numericcomparator) | Prices, quantities, measurements | Instant | No | Binary (0.0 or 1.0) |
| [**DateComparator**](date-comparator.md) | Date fields with mixed formats, partial dates, ranges | Instant | No | Continuous (0.0--1.0) |
| [**FuzzyComparator**](#fuzzycomparator) | Flexible text, descriptions, reordered tokens | Fast | No | Continuous (0.0--1.0) |
| [**BBoxIoUComparator**](#bboxioucomparator) | Bounding boxes, spatial localization | Instant | No | Continuous (0.0--1.0) |
| [**ANLSStarComparator**](#anlsstarcomparator) | Dicts and nested structures whose keys you do not declare | Moderate | No | Continuous (0.0--1.0) |
| [**SemanticComparator**](#semanticcomparator) | Meaning-based text similarity | Moderate | Yes (Bedrock) | Continuous (0.0--1.0) |
| [**BERTComparator**](#bertcomparator) | Contextual semantic similarity | Moderate | No (runs locally) | Continuous (0.0--1.0) |
| [**LLMComparator**](#llmcomparator) | Complex semantic evaluation with reasoning | Slow | Yes (Bedrock) | Binary (0.0 or 1.0) |

---

## Comparator Details

### ExactComparator

Checks for exact string matching. Case, whitespace, and punctuation are significant by default. Returns 1.0 for exact matches and 0.0 otherwise.

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
| `case_sensitive` | `True` | Whether comparison is case-sensitive |

### NormalizedComparator

Checks equality after a declared set of text transforms. By default it ignores
case, Unicode whitespace, and Unicode punctuation. Punctuation means characters
in the Unicode `P*` categories, so em dashes, curly quotes, and full-width
punctuation are handled consistently. Symbols such as `$`, `±`, and emoji remain
significant, as do accents. NFC normalization makes composed and decomposed
spellings equivalent.

**When to use:** OCR or LLM output where formatting drift is noise but edit
distance is not meaningful, such as `"U.S.A."` versus `"USA"`.

```python
from stickler import ComparableField, NormalizedComparator, StructuredModel

class Contact(StructuredModel):
    name: str = ComparableField(comparator=NormalizedComparator())
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `1.0` | Similarity threshold for binary classification |
| `case_sensitive` | `False` | Preserve case differences when `True` |
| `ignore_whitespace` | `True` | Remove Unicode whitespace |
| `ignore_punctuation` | `True` | Remove Unicode `P*` punctuation; symbols remain |

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

### ANLSStarComparator

Scores a dict whose keys you did not declare, giving partial credit instead of a
simple pass or fail.

**When to use:** a field like `metadata: Dict[str, Any]` holding whatever the
extractor returned. Comparing such a field for equality tells you only "identical"
or "not", so a prediction that got two keys of three right is indistinguishable
from one that got nothing right. This comparator scores the difference.

**When not to use:** if you know the keys, declare a nested [`StructuredModel`](../../API-Reference/models.md#stickler.structured_object_evaluator.models.structured_model.StructuredModel){:target="_blank"} instead. You get
the same partial credit *plus* a score per key, and each key can have its own
comparator and threshold. Use `ANLSStarComparator` for the case where you could
not have declared the shape.

#### A worked example

```python
from typing import Any, Dict

from pydantic import BaseModel

import stickler


class Invoice(BaseModel):
    invoice_id: str
    metadata: Dict[str, Any] = {}


truth = Invoice(
    invoice_id="INV-1042",
    metadata={"vendor": "Acme Corporation", "terms": "Net 30", "po": "PO-88231"},
)
prediction = Invoice(
    invoice_id="INV-1042",
    metadata={"vendor": "Acme Corp", "terms": "Net 30", "po": "PO-88231"},
)

result = stickler.evaluate(truth, prediction)
print(result.field_scores["metadata"])   # 0.8542
print(result.overall_score)              # 0.9271
```

`stickler.evaluate` picks this comparator for a `dict` field automatically, so
there is nothing to configure.

Running the same ground truth against a range of predictions shows what the score
is actually measuring:

| prediction's `metadata` | score |
|---|---|
| identical, in any key order | 1.0000 |
| `vendor` abbreviated to `"Acme Corp"` | 0.8542 |
| an extra `currency` key | 0.7500 |
| `po` missing | 0.6667 |
| `vendor` renamed to `vendor_name` | 0.5000 |
| every value wrong, or empty | 0.0000 |

Two things to read out of that table:

- **A missing key and an extra key both cost**, because the score is averaged over
  the union of both key sets. That is why a *renamed* key (0.5000) scores worse
  than a simply missing one (0.6667): a rename is charged twice, once as absent
  from the prediction and once as unexpected in it.
- **Nesting works.** Dicts inside dicts, and lists of dicts, are walked
  recursively, and list elements are paired by best fit rather than by position,
  so reordering a list of items does not penalise you.

#### Declaring it explicitly

To set a parameter, name the comparator on the field:

```python
from stickler import ANLSStarComparator, ComparableField, StructuredModel


class Invoice(StructuredModel):
    metadata: dict = ComparableField(
        comparator=ANLSStarComparator(leaf_threshold=0.85)
    )
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.7` | Score at or above which the whole mapping counts as a match |
| `leaf_threshold` | `0.5` | Cutoff below which a single value's similarity is treated as noise |

`leaf_threshold` is applied to each value as the structure is walked, so it changes
the score itself rather than only the verdict. On the abbreviated-vendor example
above:

| `leaf_threshold` | score |
|---|---|
| `0.5` (default) | 0.8542 |
| `0.85` | 0.6667 |

At `0.85` the abbreviation stops counting at all, so that key contributes nothing.

**When to raise it.** Character similarity scales with length, so the same cutoff
is lenient for a short value and strict for a long one. At the default, a wholly
wrong short code still earns half credit:

| kind | ground truth | prediction | `0.5` | `0.7` | `0.85` |
|---|---|---|---|---|---|
| state code | `CA` | `CO` | 0.5000 | 0.0000 | 0.0000 |
| status | `PAID` | `PEND` | 0.5000 | 0.0000 | 0.0000 |
| vendor name | `Acme Corporation` | `Acme Corp` | 0.5625 | 0.0000 | 0.0000 |
| description | `blue widget, 3 inch` | `blue widget 3in` | 0.7895 | 0.7895 | 0.0000 |

Raise it toward `0.7` if the dict holds **short codes**, where one wrong character
means wrong rather than close. Keep the default if it holds **names or free text**,
where a genuine abbreviation should still count. Note the two columns disagree:
`0.7` correctly rejects the wrong status but also discards the correct vendor
abbreviation, so if one dict holds both kinds, no single value is right for both.
Declare the fields you care about instead.

Do not set it to `0.0`: with no cutoff, an unrelated value earns credit for
incidental character overlap.

!!! warning "Every value is compared as text, whatever its type"
    Numbers, dates and identifiers are compared character by character. Incidental
    overlap in a long value therefore scores high, and scores *higher* than a
    genuine near-miss in ordinary text:

    | key | ground truth | prediction | score |
    |---|---|---|---|
    | `account` | `DE89370400440532013000` | `DE8937040044053201300`**`1`** | 0.9545 |
    | `invoice_date` | `2024-01-15` | `2024-01-1`**`6`** | 0.9000 |
    | `amount` | `1000000` | **`2`**`000000` | 0.8571 |
    | `total` | `1234.56` | `1234.5`**`7`** | 0.8571 |
    | `vendor` | `Acme Corporation` | `Acme Corp` (truncated) | 0.5625 |

    A wrong account number, a date off by a day and a **2x-wrong amount** all score
    above the one row that is a genuine near-miss.

    It fails in the other direction too. Values that are *numerically equal* score
    below 1.0, or not at all, because their text differs:

    | ground truth | prediction | score |
    |---|---|---|
    | `5` | `5.0` | 0.0000 |
    | `1000` | `1000.0` | 0.6667 |
    | `Decimal("10.50")` | `Decimal("10.5")` | 0.8000 |

    This is easy to hit by accident: a ground truth loaded from a database as an
    integer, against a prediction parsed from JSON as a float, is a perfect
    extraction scored as a miss. Declaring the field with
    [`NumericComparator`](#numericcomparator) compares the numbers instead of their
    spelling.

    Lowering `leaf_threshold` will not separate these: a cutoff high enough to
    reject the account number also removes the partial credit you wanted. **If a
    value's correctness matters, declare that field** so it gets a comparator
    chosen for its type, such as [`NumericComparator`](#numericcomparator) or
    [`DateComparator`](date-comparator.md). If you know *some* of the keys, declare
    those in a nested `StructuredModel` and leave the rest to this comparator.

**Values with no JSON form score 0.0.** Anything with a JSON representation is
compared normally, including `date`, `datetime`, `time`, `Decimal`, `UUID`, `Enum`,
`set`, `tuple`, `bytes` and `Path`. An arbitrary Python object has none, so it
cannot be compared: it scores `0.0` whether or not the two sides are equal, warns
once per type, and leaves the other keys' credit intact. Convert it to a JSON type
before evaluating, or declare a nested [`StructuredModel`](../../API-Reference/models.md#stickler.structured_object_evaluator.models.structured_model.StructuredModel){:target="_blank"} if you know its shape.

**Performance:** walking a structure costs noticeably more than comparing two
strings, and the cost multiplies inside a list, where every candidate pair is
scored. If a large corpus feels slow, declare a nested [`StructuredModel`](../../API-Reference/models.md#stickler.structured_object_evaluator.models.structured_model.StructuredModel){:target="_blank"} for the keys you
actually score.

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
