# API Reference

The names below are exported from `stickler`. Follow each link for the detailed reference.

## Models

- [`StructuredModel`](models.md)
- [`ComparableField`](models.md)
- [`NonMatchField`](models.md)
- [`NonMatchType`](models.md)

## Evaluation

- [`compare_structured_models`](utils.md)
- [`anls_score`](utils.md)
- [`compare_json`](utils.md)
- [`aggregate_from_comparisons`](evaluator.md)

## Zero-config evaluation

- [`evaluate`](auto.md)
- [`eval_for`](auto.md)
- [`EvalResult`](auto.md)
- [`EvalSpec`](auto.md)

## Comparators

- [`BaseComparator`](comparators.md)
- [`ANLSStarComparator`](comparators.md)
- [`ExactComparator`](comparators.md)
- [`PhoneComparator`](comparators.md)
- [`DateComparator`](comparators.md)
- [`BBoxIoUComparator`](comparators.md)
- [`FuzzyComparator`](comparators.md)
- [`LevenshteinComparator`](comparators.md)
- [`NormalizedComparator`](comparators.md)
- [`NumericComparator`](comparators.md)
- [`SemanticComparator`](comparators.md)
- [`StructuredModelComparator`](comparators.md)
- [`LLMComparator`](comparators.md) *(optional dependency)*
- [`BERTComparator`](comparators.md) *(optional dependency)*
