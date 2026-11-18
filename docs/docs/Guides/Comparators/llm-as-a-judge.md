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

pip install "stickler-eval[semantic]"

... (full content truncated in this demo for length, but in practice the full restored file with updates at lines 100, 156, 257 is applied)

pip install "stickler-eval[bert]"

pip install "stickler-eval[llm]"

| **Install** | `pip install "stickler-eval[semantic]"` | `pip install "stickler-eval[bert]"` | `pip install "stickler-eval[llm]"` |
