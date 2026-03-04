# LLM-as-a-Judge Comparators

When standard string or numeric comparison is not enough, Stickler provides three AI-powered comparators that bring semantic understanding to field evaluation. These comparators can determine that "123 Main Street" and "123 Main St" are equivalent, or that "The package was left at the front door" and "Delivered to entrance" mean the same thing -- something no edit-distance algorithm can do reliably.

---

## SemanticComparator

The `SemanticComparator` generates vector embeddings for both values using AWS Bedrock Titan and computes cosine similarity between them. It captures meaning rather than surface-level text similarity.

### How It Works

1. Both values are sent to the Bedrock Titan embedding model.
2. Each value is converted to a high-dimensional vector representation.
3. Cosine similarity is computed between the two vectors.
4. The resulting score (0.0--1.0) is returned.

### Setup

Requires AWS credentials with Bedrock access. The default embedding model is `amazon.titan-embed-text-v2:0`.

```bash
# Ensure AWS credentials are configured
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `model_id` | `"amazon.titan-embed-text-v2:0"` | Bedrock embedding model to use |
| `sim_function` | `"cosine_similarity"` | Similarity function (currently only cosine is supported) |
| `embedding_function` | `None` | Custom embedding function to bypass Bedrock entirely |
| `threshold` | `0.7` | Similarity threshold for binary classification |

### Example

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import SemanticComparator

class DeliveryNote(StructuredModel):
    description: str = ComparableField(
        comparator=SemanticComparator(
            model_id="amazon.titan-embed-text-v2:0",
            threshold=0.8
        ),
        weight=1.0
    )

gt = DeliveryNote(description="Package left at front door")
pred = DeliveryNote(description="Delivered to the entrance")

result = gt.compare_with(pred)
print(result["field_scores"]["description"])  # High similarity score
```

### Custom Embedding Function

You can bypass Bedrock by providing your own embedding function. The function must accept a string and return a list of floats (the embedding vector).

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

comparator = SemanticComparator(
    embedding_function=lambda text: model.encode(text).tolist()
)
```

### Cost and Latency

- **Latency:** Low per comparison (single API call per value, two calls total).
- **Cost:** AWS Bedrock Titan embedding charges apply per input token. See the [AWS Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) for current rates.
- **Tip:** For batch evaluations, consider caching embeddings for repeated ground truth values.

---

## BERTComparator

The `BERTComparator` uses BERTScore to evaluate contextual similarity between text. BERTScore computes precision, recall, and F1 scores by matching tokens in the candidate and reference using contextual BERT embeddings. Stickler returns the F1 component as the similarity score.

### How It Works

1. Both values are tokenized and passed through a BERT model (`distilbert-base-uncased` by default).
2. Token-level embeddings are compared using greedy matching.
3. Precision, recall, and F1 scores are computed.
4. The F1 score is returned as the similarity score.

### Setup

BERTComparator runs entirely locally. It requires the `evaluate`, `torch`, and `bert-score` packages.

```bash
pip install evaluate torch bert-score
```

No AWS credentials or API keys are needed.

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.7` | Similarity threshold for binary classification |

The model is loaded globally as `distilbert-base-uncased` via the `evaluate` library. The first comparison will take longer as the model is downloaded and loaded.

### Example

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import BERTComparator

class Article(StructuredModel):
    headline: str = ComparableField(
        comparator=BERTComparator(threshold=0.85),
        weight=1.0
    )

gt = Article(headline="The cat sat on the mat")
pred = Article(headline="A feline was sitting on a rug")

result = gt.compare_with(pred)
print(result["field_scores"]["headline"])  # Semantic similarity via BERTScore
```

### Cost and Latency

- **Latency:** Moderate. First run downloads the model (~250 MB for distilbert). Subsequent comparisons are faster. GPU significantly improves performance.
- **Cost:** No API costs -- everything runs locally.
- **Tip:** GPU is recommended for batch evaluations. On CPU, each comparison takes roughly 0.1--0.5 seconds depending on text length.

---

## LLMComparator

The `LLMComparator` sends both values to a Large Language Model and asks it to determine whether they are semantically equivalent. This is the most flexible comparator -- it can handle abbreviations, synonyms, domain-specific conventions, and nuanced reasoning -- but it is also the slowest and most expensive.

### How It Works

1. A prompt is constructed with both values and optional evaluation guidelines using a Jinja2 template.
2. The prompt is sent to an LLM via the `strands-agents` library and AWS Bedrock.
3. The LLM responds with `"true"` or `"false"`.
4. The response is mapped to 1.0 (equivalent) or 0.0 (not equivalent).

### Setup

Requires AWS credentials with Bedrock access and the `strands-agents` package.

```bash
pip install stickler-eval[llm]
```

```bash
# Ensure AWS credentials are configured
aws configure
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `model` | *Required* | Bedrock model ID string (e.g., `"us.amazon.nova-lite-v1:0"`) or a `strands.models.Model` instance |
| `eval_guidelines` | `None` | Custom natural-language guidelines for the LLM to follow |

### Custom Evaluation Guidelines

The `eval_guidelines` parameter is what makes LLMComparator uniquely powerful. You can provide domain-specific instructions that the LLM will follow when making its equivalence judgment.

```python
comparator = LLMComparator(
    model="us.amazon.nova-lite-v1:0",
    eval_guidelines=(
        "Consider street abbreviations equivalent (St=Street, Ave=Avenue, Blvd=Boulevard). "
        "Ignore differences in apartment/unit notation (Apt, Unit, #, Suite)."
    )
)
```

### Example

This example is based on the [`examples/scripts/llm_comparator_demo.py`](https://github.com/awslabs/stickler/blob/main/examples/scripts/llm_comparator_demo.py) script.

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, LevenshteinComparator, LLMComparator

class CustomerAddress(StructuredModel):
    street: str = ComparableField(
        comparator=LLMComparator(
            model="us.amazon.nova-lite-v1:0",
            eval_guidelines="Consider street abbreviations equivalent (St=Street, Ave=Avenue)"
        ),
        threshold=0.8,
        weight=1.0
    )
    city: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.9,
        weight=1.0
    )
    zip_code: str = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=1.0
    )

gt = CustomerAddress(street="123 Main Street", city="Seattle", zip_code="98101")
pred = CustomerAddress(street="123 Main St", city="Seattle", zip_code="98101")

result = gt.compare_with(pred)
# street field: 1.0 (LLM recognizes abbreviation)
# city field:   1.0 (exact Levenshtein match)
# zip_code:     1.0 (exact match)
```

### Debugging with Comparison Details

The `get_comparison_details` method provides full transparency into the LLM's decision:

```python
comparator = LLMComparator(
    model="us.amazon.nova-lite-v1:0",
    eval_guidelines="Consider abbreviations equivalent"
)

details = comparator.get_comparison_details("St. John's Street", "Saint John's St")
print(details["prompt"])              # The formatted prompt sent to the LLM
print(details["llm_response"])        # Raw LLM response ("true" or "false")
print(details["comparison_result"])   # Final score (1.0 or 0.0)
```

### Cost and Latency

- **Latency:** Highest of all comparators. Each comparison requires a full LLM inference call (typically 1--5 seconds depending on the model).
- **Cost:** Bedrock LLM charges apply per input and output token. Costs vary significantly by model. See [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/).
- **Tip:** Use LLMComparator selectively -- only on fields where simpler comparators cannot capture the required logic. Combine with cheaper comparators for other fields.

---

## Comparison Table

| | SemanticComparator | BERTComparator | LLMComparator |
|---|---|---|---|
| **Approach** | Embedding cosine similarity | BERTScore (contextual token matching) | LLM binary judgment |
| **Score type** | Continuous (0.0--1.0) | Continuous (0.0--1.0) | Binary (0.0 or 1.0) |
| **Speed** | Moderate (API call) | Moderate (local inference) | Slow (LLM inference) |
| **Cost** | Low (embedding API) | Free (local) | Highest (LLM API) |
| **AWS required?** | Yes (Bedrock) | No | Yes (Bedrock) |
| **Custom logic** | No | No | Yes (eval_guidelines) |
| **Best for** | General semantic similarity | Contextual similarity without cloud | Domain-specific rules, abbreviations, reasoning |
| **Install** | `pip install boto3 scipy` | `pip install evaluate torch bert-score` | `pip install stickler-eval[llm]` |

---

## Mixing Comparators

In practice, you will often use different comparators for different fields in a single model. Use cheap, fast comparators for fields with straightforward matching requirements, and reserve AI-powered comparators for fields that need semantic understanding.

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators import (
    ExactComparator,
    NumericComparator,
    LevenshteinComparator,
    LLMComparator,
    SemanticComparator,
)

class InvoiceRecord(StructuredModel):
    # Exact match -- critical identifier
    invoice_id: str = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=3.0
    )
    # Numeric with tolerance -- financial amount
    total: float = ComparableField(
        comparator=NumericComparator(absolute_tolerance=0.01),
        weight=2.5
    )
    # Levenshtein -- customer name with possible typos
    customer_name: str = ComparableField(
        comparator=LevenshteinComparator(threshold=0.8),
        weight=1.5
    )
    # Semantic -- delivery notes where meaning matters
    delivery_notes: str = ComparableField(
        comparator=SemanticComparator(threshold=0.75),
        weight=0.5
    )
    # LLM -- address with abbreviations and formatting differences
    address: str = ComparableField(
        comparator=LLMComparator(
            model="us.amazon.nova-lite-v1:0",
            eval_guidelines="Treat street abbreviations as equivalent. Ignore unit/apt notation differences."
        ),
        threshold=0.8,
        weight=1.0
    )
```

This approach balances cost, speed, and accuracy. The expensive LLM call is used only for the address field where domain-specific abbreviation logic is needed, while all other fields use cheaper comparators that are equally effective for their data types.
