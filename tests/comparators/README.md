# Comparator tests

Tests for individual comparators in `stickler.comparators`.

Most comparator tests live in `tests/common/comparators/`. This directory holds
tests for comparators whose behaviour spans more than the comparator itself, so
a unit test of the class alone would not catch the defect.

## `test_anls_star.py`

`ANLSStarComparator` scores mappings structurally. Its correctness depends on
three things that are not in the class:

1. **The dispatcher must route a dict to a comparator at all.** Before
   `ComparisonDispatcher` grew a mapping branch, a dict fell through to the
   mismatched-types case and scored 0.0 without consulting any comparator, so a
   comparator-only test would have passed while the library was broken.
2. **The default comparator for a mapping annotation must be able to score a
   mapping.** `ComparableField` resolves its comparator before the annotation
   exists, so the substitution happens in `StructuredModel.__init_subclass__`.
   The tests cover every way a mapping field can be declared, because each
   reaches that decision by a different route.
3. **Tau must reach the leaves.** The per-leaf cutoff threads through
   `ANLSTree.make_tree` into every nested node, so a test that only checks the
   top-level score would not notice it being dropped en route.

The ranking test at the bottom is the property the whole feature exists for: a
near miss, a dropped key, a hallucinated key and a wholly wrong extraction must
all be distinguishable, where whole-object equality scored them identically.
