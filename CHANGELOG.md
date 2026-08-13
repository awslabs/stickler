# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release links to full notes on the
[GitHub Releases page](https://github.com/awslabs/stickler/releases).

## [Unreleased]

### Added

- `PhoneComparator`, which compares phone numbers by the number they dial rather
  than as strings. `"555-123-4567"`, `"(555) 123-4567"`, `"+1-555-123-4567"` and
  `"5551234567"` all compare equal; extensions are reconciled
  (`"+1 (555) 123-4567 ext. 89"` matches `"+15551234567x89"`); a one-digit
  difference does not match. Pass `region=` for numbers written without an
  international prefix (default `"US"`).

  Zero-config evaluation routes `phone`-shaped field names here automatically.

  No string comparator can do this. `ExactComparator` scores a reformatted
  number `0.0`, `NumericComparator` strips non-digits and also reports `0.0`,
  and edit distance ranks the cases backwards -- a *different* number
  (`555-123-4568`) scores `0.917` while the same number reformatted scores
  `0.786`, so no threshold separates them.

  Unparseable input scores `0.0`, including when both sides are equally
  unparseable: `"N/A"` on both sides is a field that was not extracted, not a
  phone number that matched. Genuinely absent values are unaffected, since the
  shared `None` policy resolves those before any comparator runs.

  This adds `phonenumberslite` to the core dependencies: the metadata-only build
  of the libphonenumber port, 450 KB, zero dependencies, Apache-2.0
  ([#242](https://github.com/awslabs/stickler/issues/242))

- A `UserWarning` when a field `threshold` or a model `match_threshold` is set
  to exactly `0.0`. The threshold test is `>=`, so `0.0` is satisfied by every
  score including `0.0` itself: every compared pair counts as a true positive,
  and a wholly incorrect prediction reports perfect precision. Nothing errors
  and the numbers look ideal, which makes it the hardest misconfiguration to
  notice.

  The warning names what is invariant -- no false discovery can be reported,
  since FD means "compared and scored below threshold" and nothing scores below
  `0.0` -- rather than claiming a metric outcome. Perfect precision and perfect
  recall are both false in reachable cases, symmetrically: an unmatched
  prediction is still an FA (2 ground-truth objects vs 3 predictions gives
  precision `0.667`) and an unmatched ground-truth item is still an FN (2 vs 1
  gives recall `0.5`), because unmatched items are not subject to any
  threshold. For a `match_threshold` the message is also conditional, since the
  value is only read when the model is compared as a `List[StructuredModel]`
  element and is inert otherwise.

  Only exactly `0.0` warns. `0.01` already classifies correctly, so this is a
  single misbehaving value rather than a "low thresholds are risky" heuristic.
  The warning names the site, suggests a small positive value, and links to the
  threshold documentation. It fires once per configured site, so a bulk run
  does not emit one per document.

  Covers hand-written classes and config-driven models, including
  `match_threshold` supplied through `model_from_json()`, which the factory
  assigns after class creation. Stickler's own internal use of
  `match_threshold=0.0` as a capture-all sentinel does not warn
  ([#234](https://github.com/awslabs/stickler/issues/234))

- Support for Python 3.10 and 3.11, and testing through 3.14. `requires-python`
  moves from `>=3.12` to `>=3.10`, with trove classifiers for 3.10-3.14 and a
  CI matrix covering every version claimed, so the floor is enforced rather
  than asserted ([#201](https://github.com/awslabs/stickler/issues/201))
- New extras that scope the peripheral modules: `semantic` (Bedrock
  embeddings), `docsplit` (document packet splitting), `reporting` (HTML report
  tables). `all` aggregates every extra except `bert`, whose ML stack is large
  enough that installing it unasked is a surprise

### Fixed

- Zero-config evaluation no longer scores formatting-only differences in
  `email`, `url` and `phone` fields as complete mismatches. The name-token
  inference rules route those fields to comparators chosen for them, and when
  `ExactComparator` became strict the rules silently inherited the change:
  `A.Buyer@Example.COM` vs `a.buyer@example.com`, and `555-123-4567` vs
  `(555) 123-4567`, went from `1.0` to `0.0` through plain
  `stickler.evaluate()` with no configuration involved.

  `email`, `url` and `uri` now pass `case_sensitive=False` explicitly, since
  both are case-insensitive by specification. They stay otherwise exact:
  `a@b.com` vs `a@c.com` is still `0.0`, where a similarity comparator reports
  `0.857`.

  Every rule that selects `ExactComparator` now states its case sensitivity
  rather than inheriting it, so a future change to a comparator default cannot
  silently redefine what inference means. Identifier tokens (`id`, `sku`,
  `code`, `ref`, `uuid`, `isbn`, `ssn`) remain deliberately case-sensitive.

  **Postal codes stay exact, deliberately.** `98101-1234` does not match
  `98101 1234`. A generic normalizer would be right for the US and wrong
  elsewhere -- a UK postcode's internal space is significant (`SW1A 1AA`), and
  Dutch codes mix letters and digits -- so applying US rules everywhere would
  produce failures that look like successes. A similarity comparator is not a
  fallback either: `98101-1234` scores `0.9` against both `98101 1234` (same
  code) and `98102-1234` (different code), so no threshold separates them. The
  new [Postal Codes and Addresses](https://awslabs.github.io/stickler/Guides/Comparators/postal-codes/)
  guide covers how to handle this for a specific country, including a worked
  US and UK comparator and an assessment of the available third-party libraries
  ([#242](https://github.com/awslabs/stickler/issues/242))

- The Hungarian single-item shortcut now classifies pairs the same way the
  general multi-item path does, so a confusion-matrix result no longer depends
  on how many items happen to be in a list. Previously a 1-vs-1 comparison at
  zero similarity was reported as FN + FA where a 2-vs-2 reported FD, and the
  shortcut gated on `score > 0` instead of `match_threshold`, so any non-zero
  similarity counted as a true positive however far below threshold it was.

  This makes the 1-vs-1 case follow the two documented rules it was the sole
  violator of: a pair the algorithm assigns is a match, and `match_threshold`
  splits matched pairs into TP and FD without un-matching them
  ([Hungarian matching](https://awslabs.github.io/stickler/Advanced/hungarian-matching/));
  and a below-threshold pair is treated as atomic, with no field-by-field
  breakdown
  ([Threshold-gated evaluation](https://awslabs.github.io/stickler/Advanced/threshold-gated-evaluation/)).

  **This moves metrics, and the default moves them upward.** A 1-vs-1 list
  that is completely wrong now reports FD where it previously reported
  FN + FA. `recall_with_fd` defaults to `False`, which excludes FD from the
  recall denominator, so reclassifying FN as FD *raises* reported recall for
  an unchanged prediction:

  | three single-item list fields, one wholly wrong | before | after |
  |---|---|---|
  | `tp` / `fa` / `fd` / `fn` | 2 / 1 / 0 / 1 | 2 / 0 / 1 / 0 |
  | `cm_recall` | 0.667 | **1.000** |
  | `cm_f1` | 0.667 | 0.800 |
  | `cm_accuracy` | 0.500 | 0.667 |

  Multi-item lists already behaved this way, so this is the 1-vs-1 case
  becoming consistent rather than a new policy. Set `recall_with_fd=True` to
  count false discoveries against recall.

  **Two result shapes change for a below-threshold 1-vs-1 list**, both to
  match what multi-item lists already produced:

  `confusion_matrix.fields.<list>.fields` is now empty, where it previously
  carried an entry per sub-field. Reading a sub-field key directly raises
  `KeyError`:

  ```python
  cm["fields"]["lines"]["fields"]["sku"]   # KeyError: 'sku'
  ```

  Use `.get()`, or read the object-level counts at
  `cm["fields"]["lines"]["overall"]`, which record the FD. This follows the
  documented threshold-gating rule: below the threshold the pairing is
  spurious, so no field-by-field breakdown is generated
  ([Threshold-gated evaluation](https://awslabs.github.io/stickler/Advanced/threshold-gated-evaluation/)).

  `non_matches` (from `document_non_matches=True`) changes in the opposite
  direction, from two object-level records to one field-level record per
  sub-field:

  | | before | after |
  |---|---|---|
  | `field_path` | `lines[0]`, `lines[0]` | `lines[0].sku`, `lines[0].desc` |
  | `non_match_type` | `false_negative`, `false_alarm` | `false_discovery` (both) |
  | `ground_truth_value` | the whole object dict | the scalar field value |
  | `similarity_score` | `None` | the pair's score |

  Code that groups non-matches by `non_match_type` to count misses, or that
  parses `field_path` expecting an object-level path for FN/FA records, needs
  updating: a single-line-item document that previously produced one FN and
  one FA now produces false discoveries at the field level
  ([#224](https://github.com/awslabs/stickler/issues/224))

- Handle `None` consistently across all comparators. `None` is a missing
  value and no longer compares equal to an empty string: `(None, "")` now
  scores `0.0` everywhere, and `(None, None)` scores `1.0` everywhere.
  Previously `LevenshteinComparator` coerced `None` to `""` and scored
  `(None, "")` as `1.0`, while `SemanticComparator` and `BERTComparator`
  scored `(None, None)` as `0.0`.

  **This changes results for code calling comparators directly** --
  `LevenshteinComparator().compare(None, "")` goes from `1.0` to `0.0`.
  `compare_with()` is largely unaffected for string fields:
  `NullHelper.is_effectively_null_for_primitives` treats both `None` and
  `""` as null and the dispatcher resolves that pair as a true negative
  before any comparator runs
  ([#200](https://github.com/awslabs/stickler/issues/200))

- Restore per-comparator method documentation on the published API reference.
  mkdocstrings hides single-underscore members by default, so moving the
  extension point to `_compare()` dropped every comparator's `Args`, `Returns`
  and `Raises` detail from the site, leaving only the base class. The
  comparator page now renders all 12 `_compare` entries again
  ([#228](https://github.com/awslabs/stickler/issues/228))

- Fix the documentation build, which failed on `mkdocs build` with
  `Could not collect 'stickler.comparators.BERTComparator'`. `BERTComparator`
  and `LLMComparator` are exposed lazily through the package `__getattr__` so
  that `import stickler` does not pull `torch`/`transformers` or
  `strands`/`boto3`; griffe resolves identifiers statically and cannot see a
  name that exists only at attribute-access time. The API reference now
  addresses both by their defining module. This broke the docs deploy for any
  push to `main` touching `src/` or `docs/`
  ([#228](https://github.com/awslabs/stickler/issues/228))

- Clear three `non-literal-import` findings from the ASH security scan, at the
  two package `__getattr__` hooks and `ComparatorRegistry._resolve`. All three
  are false positives: the imported path is a literal from a module-level
  allowlist and the caller's string is only ever a dict key, so an
  unrecognized name is rejected before any import is attempted. Annotated with
  `# nosemgrep` plus the reasoning, and pinned by
  `tests/test_lazy_import_allowlists.py`, which imports a canary module to
  assert nothing is imported for a rejected name, whatever spelling the call
  site uses. No behavior change

- `ComparatorRegistry` construction no longer raises when an optional
  dependency is present in `sys.modules` without a `__spec__`, which
  `importlib.util.find_spec` reports as `ValueError` rather than a missing
  module. A test that injects a mock for an optional extra could take registry
  construction down with it. The availability probe now consults `sys.modules`
  before the filesystem, mirroring the package-level `_dependency_available`
  helpers, so a mocked dependency counts as available in both places rather
  than having `stickler.LLMComparator` resolve while
  `registry.get("LLMComparator")` reports it missing

### Changed

- **Breaking:** `ExactComparator` is now truly exact. The default is
  `case_sensitive=True` (was `False`), and punctuation/whitespace stripping
  has been removed entirely. This fixes
  [#199](https://github.com/awslabs/stickler/issues/199), where
  `"SHP-2024-001"` incorrectly matched `"shp 2024 001"`.

  **This lowers scores** on existing evaluation sets, and it reaches you even if
  you never named `ExactComparator`. The zero-config path (`stickler.evaluate()`,
  `eval_for()`, `from_pydantic()`) infers `ExactComparator` for id-shaped,
  code-shaped, email, zip and boolean fields, so on a typical extraction model
  it covers roughly half the fields:

  ```python
  class Invoice(BaseModel):           # plain pydantic, no stickler config
      invoice_id: str                 # inferred -> ExactComparator
      sku: str                        # inferred -> ExactComparator
      email: str                      # inferred -> ExactComparator
      zip_code: str                   # inferred -> ExactComparator
      vendor_name: str                # inferred -> LevenshteinComparator
  ```

  With predictions differing only in case and punctuation
  (`"SHP-2024-001"` vs `"shp 2024 001"`, `"98101-1234"` vs `"98101 1234"`):

  | | before | after |
  |---|---|---|
  | the four inferred-Exact fields | 1.0 each | **0.0 each** |
  | `overall_score` | 1.0 | **0.20** |

  That is the intended correction -- those pairs are not equal, and reporting
  them as perfect matches is the bug -- but if you track a metric across
  releases, expect a step change at this version rather than a drift.

  | what changed | before | after |
  |---|---|---|
  | `ExactComparator().compare("Hello", "hello")` | 1.0 | **0.0** |
  | `ExactComparator().compare("ID-123", "ID 123")` | 1.0 | **0.0** |
  | `ExactComparator().compare("SHP-2024-001", "shp 2024 001")` | 1.0 | **0.0** |

  **Migration:** For case-insensitive matching, pass `case_sensitive=False`.

  For punctuation or whitespace differences, `ExactComparator` is the wrong
  tool. Use a similarity comparator with a threshold tuned to your data:

  ```python
  vendor: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
  ```

  Note that a threshold of `1.0` will **not** work here: `"ID-123"` vs
  `"ID 123"` scores `0.833`, so requiring a perfect score still rejects it. Pick
  a threshold below the score your real data produces, or write a comparator
  that normalizes the way your domain requires.

  The `case_sensitive=False` path now uses `str.casefold()` (Unicode case
  folding) instead of `str.lower()`, correctly handling cases like
  `"STRASSE"` vs `"straße"` ([#199](https://github.com/awslabs/stickler/issues/199))

- **Breaking:** the peripheral modules now require their extra. `pandas`,
  `scipy`, `scikit-learn`, and `jinja2` are no longer core dependencies, so
  `pip install stickler-eval` installs the comparison engine and nothing else.
  Code that used these without installing an extra now raises `ImportError`
  naming the extra to install:

  | what you were using | now needs |
  |---|---|
  | `stickler.doc_split` (raises at import) | `stickler-eval[docsplit]` |
  | `MarkdownUtil.table_df()` | `stickler-eval[reporting]` |
  | `LLMComparator(...)` | `stickler-eval[llm]` |
  | `SemanticComparator` cosine similarity | `stickler-eval[semantic]` |

  Confidence calibration metrics are **not** affected: `scikit-learn` stays in
  the core dependency set because calibration is core functionality, not an
  add-on. The `confidence` extra is now empty and kept only so existing pins
  keep resolving.

  `pip install "stickler-eval[all]"` restores everything except `bert`.

  Only `stickler.doc_split` fails at import time; the rest fail at first use.
  `SemanticComparator` still constructs on a core install and raises when the
  similarity function runs
  ([#201](https://github.com/awslabs/stickler/issues/201))

- Optional comparators are now imported lazily. `import stickler` no longer
  pulls a scientific-computing stack: 421 modules on a core install, down from
  1664, with none of pandas, scipy, scikit-learn, jinja2, torch, transformers,
  strands, or boto3 on the path. `scikit-learn` is a core dependency but its
  import stays inside `AUROCMetric.compute()`, so it costs nothing at import
  time. `BERTComparator` no longer loads its model at import time. Accessing an
  optional comparator whose extra is missing raises `AttributeError`, so
  `hasattr()` gating keeps working
  ([#187](https://github.com/awslabs/stickler/issues/187))

- Relaxed the `scikit-learn` floor from `>=1.8.0` to `>=1.7.2`. 1.8.0 requires
  Python 3.11+, so the old floor made the 3.10 support above unresolvable. The
  lock pins 1.7.2 below 3.11 and 1.8.0 above

- **Deprecated (custom comparators):** the extension point for comparators is
  now `_compare()` instead of `compare()`. `BaseComparator.compare()` is a
  template method that applies the shared `None` policy and then delegates
  to `_compare()`, so the policy is defined once and cannot drift between
  comparators. Callers are unaffected -- `compare()`, `__call__`, and
  `binary_compare()` are unchanged.

  Custom comparators must rename their `compare()` to `_compare()` and can
  delete any `None` handling it contains, since `_compare()` only ever
  receives present values.

  This is not a hard break. A deprecation shim keeps pre-rename comparators
  working: one that implements `compare()` still constructs and behaves
  exactly as written, and emits a `DeprecationWarning` naming the rename.
  That holds whether it extends `BaseComparator` directly, extends a
  concrete comparator, or inherits `compare()` from a mixin.

  An un-migrated comparator does **not** receive the `None` policy, because
  its `compare()` shadows the template method, so the rename is still
  required. The shim is removed in 0.8.0, after which such a comparator
  raises `TypeError` at construction
  ([#215](https://github.com/awslabs/stickler/issues/215)).

  Note that the pre-fix `(None, "") -> 1.0` result cannot be inherited: the
  coercion was removed from Levenshtein's algorithm rather than guarded, so
  an un-migrated subclass that delegates upward gets the corrected score.
  Only a subclass that reimplemented the coercion in its own `compare()`
  still returns the old value
  ([#200](https://github.com/awslabs/stickler/issues/200))

- `ComparableField(aggregate=...)` now emits a `DeprecationWarning` for *any*
  explicit use, where previously only `aggregate=True` warned and
  `aggregate=False` was silent. The parameter has no effect: aggregation is
  applied at the comparison layer, and every node in `compare_with()` output
  already carries an `aggregate` block summing the primitive field metrics
  below it.

  Callers passing `aggregate=False` had no signal the parameter was going away
  and would have met a bare `TypeError` on removal. Remove the argument; there
  is no replacement to adopt. Scheduled for removal in 0.8.0.

  Reading a config does **not** count as explicit use: `to_stickler_config()`
  writes the `aggregate` key for every field, so `model_from_json()` restores
  the value without warning. Otherwise every exported-config round trip would
  warn once per field, blaming stickler's own frame for a key the caller never
  wrote, and would fail outright under `-W error::DeprecationWarning`. The
  export format is unchanged and stays idempotent: `export -> import -> export`
  reproduces the original, including for `aggregate=True`.

  Also removed a dead branch in `ConfusionMatrixCalculator` that zeroed and
  re-summed a list field's confusion matrix when the flag was set. It was
  unreachable by construction -- guarded on the argument *not* being a list, in
  a method only ever called with one (instrumented the whole suite: 0 calls) --
  and left a live-looking code path keyed on a parameter that is going away
  ([#226](https://github.com/awslabs/stickler/issues/226))

## [0.6.0] - 2026-07-30

### Added

- Zero-config evaluation of vanilla Pydantic models: `stickler.evaluate()`,
  `stickler.eval_for()`, `EvalSpec`, `EvalResult`, and
  `StructuredModel.from_pydantic()`. Scores a plain `BaseModel` (for example a
  Strands agent `response_model`) with no `StructuredModel` subclass,
  comparators, thresholds, or schema, inferring a comparator per field from the
  live Python type plus field-name hints. `EvalResult.explain()` reports the
  comparator, threshold, weight, and provenance behind every field decision
  ([#176](https://github.com/awslabs/stickler/pull/176))

### Fixed

- Support JSON Schema list-form nullable types (`"type": ["string", "null"]`)
  ([#127](https://github.com/awslabs/stickler/pull/127))
- Guard `None` elements in nullable object list comparison paths, which
  previously raised `AttributeError` on `List[Optional[Model]]`
  ([#181](https://github.com/awslabs/stickler/pull/181))
- Register `BERTComparator` under the correct name in the comparator registry
  ([#157](https://github.com/awslabs/stickler/pull/157))
- Guard non-finite values in `NumericComparator` tolerance comparisons, which
  previously raised `decimal.InvalidOperation` on NaN and infinite inputs
  ([#176](https://github.com/awslabs/stickler/pull/176))
- Resolve ASH security scan findings: top-level workflow permissions, bandit
  exec suppression ([#156](https://github.com/awslabs/stickler/pull/156))

### Changed

- Pin GitHub Actions to commit SHAs and add a uv dependency cooldown
  ([#170](https://github.com/awslabs/stickler/pull/170))
- Raise the `[llm]` extra floor to `strands-agents>=1.14.0`: the documented
  `structured_output_model=` keyword landed in 1.14.0, and on 1.0 to 1.13 it is
  silently swallowed by `**kwargs`
  ([#176](https://github.com/awslabs/stickler/pull/176))
- Dependency bumps via dependabot
  ([#167](https://github.com/awslabs/stickler/pull/167),
  [#168](https://github.com/awslabs/stickler/pull/168))

### Documentation

- Add maintainer handoff artifacts: `MAINTAINERS`, `CODEOWNERS`, `RELEASING`,
  `CHANGELOG`, and a maintainer's guide
  ([#175](https://github.com/awslabs/stickler/pull/175))
- Add the Ultra Quick Start page and an "Evaluating a Strands Agent" guide for
  the zero-config path ([#176](https://github.com/awslabs/stickler/pull/176))
- Add `CLAUDE.md` importing `AGENTS.md` for Claude Code
  ([#169](https://github.com/awslabs/stickler/pull/169))

## [0.5.0] - 2026-06-26

### Added

- `DateComparator` with year/range awareness, configurable tolerance, and
  JSON-config instantiation ([#141](https://github.com/awslabs/stickler/pull/141))
- Mean Average Precision (mAP) scoring for bounding-box evaluation:
  `MAPCalculator`, `BBoxMAPAccumulator`
  ([#151](https://github.com/awslabs/stickler/pull/151))
- Core comparators importable from the top-level package
  (`from stickler import ExactComparator`)
  ([#121](https://github.com/awslabs/stickler/pull/121))

### Fixed

- HTML report rendering and semantic-comparator fallback fixes
- Script injection hardening in the security workflow
  ([#154](https://github.com/awslabs/stickler/pull/154))

## [0.4.0] - 2026-05-20

### Added

- Confidence evaluation tooling: calibration metrics (ECE, Brier score,
  AUROC), `ConfidenceCalculator`, rich-value pattern for carrying confidence
  through the pipeline ([#98](https://github.com/awslabs/stickler/pull/98))
- Weight-aware aggregation in `BulkStructuredModelEvaluator`
  ([#124](https://github.com/awslabs/stickler/pull/124))
- `SemanticComparator` accepts `model_id` as a plain string
  ([#116](https://github.com/awslabs/stickler/pull/116))

### Fixed

- Bulk evaluator aggregation fixes ([#124](https://github.com/awslabs/stickler/pull/124))
- Example notebook repairs ([#98](https://github.com/awslabs/stickler/pull/98))

## [0.3.0] - 2026-04-17

### Added

- DocSplit metrics for packet-splitting evaluation
  ([#82](https://github.com/awslabs/stickler/pull/82))
- `StructuredModel` export helpers
  ([#56](https://github.com/awslabs/stickler/pull/56),
  [#107](https://github.com/awslabs/stickler/pull/107))

### Fixed

- Single-page group ordering metric
  ([#95](https://github.com/awslabs/stickler/pull/95))
- Packaging and distribution fixes for PyPI installs
  ([#91](https://github.com/awslabs/stickler/pull/91),
  [#100](https://github.com/awslabs/stickler/pull/100),
  [#101](https://github.com/awslabs/stickler/pull/101))
- `ComparableField` None-default regression
  ([#84](https://github.com/awslabs/stickler/pull/84))

### Changed

- Migrated from conda to [uv](https://docs.astral.sh/uv/) for dependency
  management ([#97](https://github.com/awslabs/stickler/pull/97))

## [0.2.0] - 2026-03-16

### Added

- Document packet splitting metrics
  ([#82](https://github.com/awslabs/stickler/pull/82))
- JSON/model export and import
  ([#56](https://github.com/awslabs/stickler/pull/56))

### Fixed

- Numeric and structured list comparator fix
  ([#83](https://github.com/awslabs/stickler/pull/83))

### Changed

- Major documentation restructure
  ([#86](https://github.com/awslabs/stickler/pull/86))

## [0.1.5] - 2026-02-17

### Added

- Confidence-aware evaluation with AUROC metrics
  ([#54](https://github.com/awslabs/stickler/pull/54))
- LLM-powered semantic comparator via AWS Bedrock
  ([#15](https://github.com/awslabs/stickler/pull/15))
- Bulk evaluator aggregations with streaming support
  ([#74](https://github.com/awslabs/stickler/pull/74))

### Fixed

- Security fixes ([#80](https://github.com/awslabs/stickler/pull/80))

## [0.1.4] - 2025-12-23

### Added

- `field_comparisons` parameter for matches and non-matches

### Removed

- `StructuredModelEvaluator.evaluate()` API

## [0.1.3] - 2025-11-18

Initial public release: structured JSON comparison with configurable
comparators, Hungarian-algorithm list matching, confusion-matrix metrics, and
HTML reporting.

[Unreleased]: https://github.com/awslabs/stickler/compare/v0.6.0...dev
[0.6.0]: https://github.com/awslabs/stickler/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/awslabs/stickler/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/awslabs/stickler/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/awslabs/stickler/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/awslabs/stickler/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/awslabs/stickler/compare/v.0.1.4...v0.1.5
[0.1.4]: https://github.com/awslabs/stickler/compare/v.0.1.3...v.0.1.4
[0.1.3]: https://github.com/awslabs/stickler/releases/tag/v.0.1.3
