# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release links to full notes on the
[GitHub Releases page](https://github.com/awslabs/stickler/releases).

## [Unreleased]

### Added

- Support for Python 3.10 and 3.11, and testing through 3.14. `requires-python`
  moves from `>=3.12` to `>=3.10`, with trove classifiers for 3.10-3.14 and a
  CI matrix covering every version claimed, so the floor is enforced rather
  than asserted ([#201](https://github.com/awslabs/stickler/issues/201))
- New extras that scope the peripheral modules: `semantic` (Bedrock
  embeddings), `docsplit` (document packet splitting), `reporting` (HTML report
  tables). `all` aggregates every extra except `bert`, whose ML stack is large
  enough that installing it unasked is a surprise

### Fixed

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

### Changed

- `model_json_schema()` now describes the model's shape the way an equivalent
  plain `BaseModel` would, so a configured `StructuredModel` can drive a
  Strands agent's structured output without degrading the schema the LLM sees:
  `required` is derived from the annotation (`shipment_id: str` renders
  required even though `ComparableField` assigns a `None` default for
  construction tolerance), required fields no longer widen to
  `["type", "null"]` or carry a contradictory `default: null`, and comparison
  configuration (`x-comparison`) is no longer emitted. Verified through
  Strands' `convert_pydantic_to_tool_spec`: a configured `StructuredModel` and
  its plain-`BaseModel` twin now produce the same tool spec.

  Field-level `description`, `examples`, and `alias` still reach the rendered
  schema, and the deliberate export path `to_json_schema()` still carries the
  comparison configuration as `x-aws-stickler-*` extensions. Runtime behavior
  is unchanged: predictions that omit fields still construct and score.

  Code that read `x-comparison` out of `model_json_schema()` output should
  read the field's `json_schema_extra` (as the engine does) or use
  `to_json_schema()`
  ([#188](https://github.com/awslabs/stickler/issues/188))

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
