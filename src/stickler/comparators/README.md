# Comparators

Comparators implement `BaseComparator._compare()`; the base class handles `None`
and thresholding. See [Comparators.md](Comparators.md) for their configuration.

`LevenshteinComparator` normalizes and stringifies inputs, delegates integer edit
distance to RapidFuzz, and retains its own floating-point similarity formula.
`ANLSStarComparator` uses this distance for tree leaves and list matching.
