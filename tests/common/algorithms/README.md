# algorithms Tests

Tests for `src/stickler/algorithms/`, which is `HungarianMatcher`. The README
there describes the contract these files hold it to.

Three files, three jobs. They overlap on purpose, because each one exists to
catch a different kind of regression.

## test_hungarian.py

The behaviour of the class as a whole. Construction, the comparator variants,
list preparation including a scalar and a JSON string of a list, the size
warning, and the legacy `__call__` and `binary_compare` wrappers. Start here
when adding a feature.

## test_hungarian_path_parity.py

That the single item shortcut classifies exactly like the general path
([#224](https://github.com/awslabs/stickler/issues/224)). The shortcut skips the
solver, so it is a second implementation of the same rule and it did drift once.
The property under test is length independence: a one item list must not
classify differently from a two item list in the same situation.

## test_hungarian_fd_contract.py

That a paired item is never also reported as missing
([#231](https://github.com/awslabs/stickler/issues/231)). This is the executable
form of the five category rule, written as invariants rather than as expected
values, so it holds for every shape instead of for the shapes someone thought
of.

Two classes in it are worth knowing about:

- `TestTheCountsPartitionTheInputs` is the cheapest full statement of the rule.
  If those five assertions hold, no item is counted twice and none is dropped.
- `TestTheValuesThatMustNotChange` is a guard, not a specification. It pins the
  values that #231 did **not** change. A failure there means the scope of a
  change grew beyond one key, so read it as a question about the change and not
  as a test to update.

`TestTheOldFnValueIsRecoverable` pins `old fn == fn + fd`, which is the
migration note for callers who read `fn` before #231. If that identity ever
stops holding, the changelog entry has gone stale.

## Fixtures

`_gt` and `_pred` build values with zero pairwise similarity, so a pair exists
without any score to reason about. They assert their own length rather than
slicing and hoping, because a fixture that quietly truncates leaves a test
passing while checking something weaker than its name claims.

```bash
uv run pytest tests/common/algorithms/ -v
```
