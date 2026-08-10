# Comparators

::: stickler.comparators
    options:
      heading_level: 1
      members: false

::: stickler.comparators.BaseComparator
    options:
      heading_level: 2

::: stickler.comparators.ExactComparator
    options:
      heading_level: 2

::: stickler.comparators.LevenshteinComparator
    options:
      heading_level: 2

::: stickler.comparators.NumericComparator
    options:
      heading_level: 2

::: stickler.comparators.NumericExactC
    options:
      heading_level: 2

::: stickler.comparators.DateComparator
    options:
      heading_level: 2

::: stickler.comparators.FuzzyComparator
    options:
      heading_level: 2

<!-- BERTComparator and LLMComparator are addressed by their defining module
     rather than through the package. Both are exposed lazily via the package
     __getattr__ so that importing stickler does not pull torch/transformers or
     strands/boto3, and griffe resolves identifiers statically, so it cannot
     see a name that only exists at attribute-access time. -->
::: stickler.comparators.bert.BERTComparator
    options:
      heading_level: 2

::: stickler.comparators.SemanticComparator
    options:
      heading_level: 2

::: stickler.comparators.llm.LLMComparator
    options:
      heading_level: 2

::: stickler.comparators.StructuredModelComparator
    options:
      heading_level: 2

::: stickler.comparators.BBoxIoUComparator
    options:
      heading_level: 2
