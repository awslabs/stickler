# Comparators

See [Comparators.md](Comparators.md) for the comparator interface and supported implementations.

`LevenshteinComparator` normalizes and stringifies inputs, delegates integer edit
distance to RapidFuzz, and retains its own floating-point similarity formula.
`ANLSStarComparator` uses this distance for tree leaves and list matching.

A list field pairs its elements and scores each pair. `scores_as_one_value(value)`
is the public hook for a custom comparator whose input is one list value: return
True for a list it reads whole, as `BBoxIoUComparator` does for a list that
parses as one box. Both sides are then passed to `compare()` unsplit, unless the
field annotation names sequence elements (`List[List[float]]`).
