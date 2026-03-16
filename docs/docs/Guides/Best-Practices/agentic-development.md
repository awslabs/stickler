---
title: Agentic Development
---

# Agentic Development

Stickler is designed to be AI-agent-friendly. The codebase uses conventions that help AI coding assistants -- such as Claude, Copilot, Cursor, and others -- understand the project structure, locate relevant code, and work effectively with the library.

## AGENTS.md Pattern

Stickler uses `AGENTS.md` files distributed throughout the codebase to provide context for AI assistants. These files describe:

- Directory structure and purpose
- Key concepts and abstractions
- Implementation patterns and conventions
- Testing and documentation guidelines

The root [AGENTS.md](https://github.com/awslabs/stickler/blob/main/AGENTS.md) is the starting point. It directs agents to more specific `AGENTS.md` and `README.md` files in subdirectories.

**When working with the Stickler codebase, always read `AGENTS.md` in the root first.** It provides the high-level orientation needed to navigate the project.

## README.md-Driven Navigation

Every directory in the Stickler project has (or should have) a `README.md` file. These serve as distributed documentation that helps both humans and AI agents understand the codebase without loading everything at once.

**Pattern for AI agents entering a directory:**

1. Read `README.md` if it exists -- it describes the purpose and contents of the directory.
2. Read `AGENTS.md` if it exists -- it provides AI-specific guidance for working with the code.
3. Proceed with the task using the context from those files.

This distributed approach means an AI agent never needs to load the entire codebase into context. Each directory is self-documenting.

## JSON Schema for Config-Driven Evaluation

The `x-aws-stickler-*` JSON Schema extensions are particularly valuable for AI agent workflows:

- **AI agents can generate JSON Schema from examples or specifications** -- no Python code needed to define evaluation criteria.
- **Evaluation configuration is declarative** -- agents can read and modify JSON Schema without understanding Python internals.
- **Perfect for automated pipelines** -- an agent can create a schema, configure comparators and thresholds, and run evaluation entirely through JSON configuration.

```json
{
  "type": "object",
  "x-aws-stickler-model-name": "ExtractedDocument",
  "properties": {
    "document_id": {
      "type": "string",
      "x-aws-stickler-comparator": "ExactComparator",
      "x-aws-stickler-weight": 3.0
    },
    "extracted_text": {
      "type": "string",
      "x-aws-stickler-comparator": "LevenshteinComparator",
      "x-aws-stickler-threshold": 0.7,
      "x-aws-stickler-weight": 2.0
    }
  }
}
```

An AI agent can produce this schema from a natural language description of the evaluation requirements and then pass it to `StructuredModel.from_json_schema()`.

## Markdown-Scrapable Documentation

The MkDocs documentation site is structured for AI agent consumption:

- Clean markdown with consistent heading hierarchy
- Code examples that are self-contained and runnable
- Cross-references using relative paths for navigability
- Each page covers one concept or topic

AI agents that scrape documentation (e.g., for RAG pipelines or context retrieval) will find the structure predictable and easy to parse.

## Recommended Workflows

### Evaluating GenAI output

When an AI agent needs to evaluate structured output from a GenAI system:

1. Define a `StructuredModel` matching the expected output schema
2. Choose comparators based on field semantics (IDs get Exact, text gets Levenshtein, numbers get Numeric)
3. Set thresholds and weights based on business impact
4. Call `compare_with()` for individual comparisons or use `BulkStructuredModelEvaluator` for batch processing

### Modifying or extending Stickler

When an AI agent needs to make changes to the Stickler codebase:

1. Read `AGENTS.md` in the project root
2. Read `README.md` in the relevant directory
3. Understand the architecture before making changes
4. Run tests with `pytest tests/` to verify changes

### Building automated evaluation pipelines

For automated pipelines where evaluation criteria are configured externally:

1. Use the JSON Schema approach with `x-aws-stickler-*` extensions
2. Load schemas from files or a configuration service
3. Create models at runtime with `StructuredModel.from_json_schema()`
4. Use `BulkStructuredModelEvaluator` for batch processing with `save_metrics()` for persistent results

## Example Agent Instructions

Below is an example of what instructions for an AI agent working with Stickler might contain. This could be placed in a `.rules` file, a system prompt, or any agent configuration mechanism:

```
When evaluating structured outputs:
1. Define a StructuredModel matching the output schema
2. Choose comparators based on field semantics:
   - ExactComparator for IDs, codes, booleans
   - NumericComparator for prices, quantities, measurements
   - LevenshteinComparator for names, addresses, short text
   - FuzzyComparator for descriptions, notes, free-form text
3. Set thresholds based on acceptable variation:
   - 1.0 for fields that must be exact
   - 0.8-0.9 for important fields with minor tolerance
   - 0.5-0.7 for flexible fields
4. Set weights based on business impact (2.5-3.0 critical, 1.0 normal, 0.1-0.3 minimal)
5. Use compare_with() for individual comparisons
6. Use BulkStructuredModelEvaluator for batch processing
7. Enable document_non_matches=True and document_field_comparisons=True for debugging
```

These instructions give an AI agent enough context to use Stickler effectively without needing to read the full documentation.

## Resources

- [AGENTS.md (project root)](https://github.com/awslabs/stickler/blob/main/AGENTS.md)
- [JSON Schema Extensions](../Evaluation/README.md)
- [Bulk Evaluation Guide](../Evaluation/bulk-evaluation.md)

- [Best Practices Overview](README.md)
