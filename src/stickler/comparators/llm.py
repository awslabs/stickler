"""LLM-based semantic comparison comparator.

This module provides the LLMComparator class, which uses Large Language Models
to perform semantic comparisons between values. Unlike traditional string-based
comparators, the LLMComparator can understand context, synonyms, abbreviations,
and other semantic relationships to determine if two values are equivalent.

The comparator integrates with AWS Bedrock models through the strands-agents
library and supports customizable evaluation guidelines for domain-specific
comparison logic.

Note:
    This comparator requires the optional 'llm' dependencies. Install with:
    pip install "stickler-eval[llm]"

Example:
    Integration with StructuredModel:
        >>> from stickler import ComparableField
        >>>
        >>> class Address(StructuredModel):
        ...     street: str = ComparableField(
        ...         comparator=LLMComparator(
        ...             model="us.amazon.nova-lite-v1:0",
        ...             eval_guidelines="Consider street abbreviations equivalent"
        ...         )
        ...     )
