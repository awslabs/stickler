# Comparators

See [Comparators.md](Comparators.md) for the comparator interface and supported implementations.

`LevenshteinComparator` normalizes and stringifies inputs, delegates integer edit
distance to RapidFuzz, and retains its own floating-point similarity formula.
`ANLSStarComparator` uses this distance for tree leaves and list matching.

`PointDistanceComparator` requires `max_distance` and has no default for it, so its `config`
always carries `max_distance`. That lets an exported list-of-points field import again. A
single `Tuple` point field still exports as a string type and does not round-trip.
