# Comparators

See [Comparators.md](Comparators.md) for the comparator interface and supported implementations.

`LevenshteinComparator` normalizes and stringifies inputs, delegates integer edit
distance to RapidFuzz, and retains its own floating-point similarity formula.
`ANLSStarComparator` uses this distance for tree leaves and list matching.
