# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release links to full notes on the
[GitHub Releases page](https://github.com/awslabs/stickler/releases).

## [Unreleased]

## [0.7.0] - 2026-08-18

### Added

- `DateComparator` and `BBoxIoUComparator` are now exported from the top-level
  `stickler` namespace, alongside every other comparator. Both were reachable
  only as `from stickler.comparators import DateComparator`, so the obvious
  import failed with an `ImportError` that gave no hint the name existed one
  level down. Neither had ever been exported from the package root, dating from
  when each was added ([#141](https://github.com/awslabs/stickler/pull/141) and
  [#151](https://github.com/awslabs/stickler/pull/151)).

  Purely additive: both are core and were already imported eagerly by
  `stickler.comparators`, so `import stickler` is unchanged at 523 modules.
  `tests/test_top_level_exports.py` now derives the expected set from
  `stickler.comparators` rather than a hand-maintained list, which is what let
  the gap survive ([#252](https://github.com/awslabs/stickler/issues/252))

- `PhoneComparator`, which compares phone numbers by the number they dial rather
  than as strings. `"206-555-0100"`, `"(206) 555-0100"`, `"+1-206-555-0100"` and
  `"2065550100"` all compare equal; extensions are reconciled
  (`"+1 (206) 555-0100 ext. 89"` matches `"+12065550100x89"`); a one-digit
  difference does not match. Pass `region=` for numbers written without an
  international prefix (default `"US"`).

  Zero-config evaluation routes `phone`-shaped field names here automatically.

  No string comparator can do this. `ExactComparator` scores a reformatted
  number `0.0`, `NumericComparator` strips non-digits and also reports `0.0`,
  and edit distance ranks the cases backwards -- a *different* number
  (`206-555-0101`) scores `0.917` while the same number reformatted scores
  `0.786`, so no threshold separates them.

  When **neither** side is a usable number, the comparison falls back to exact
  string equality, so two identical values are never reported as maximally
  different. Not every correctly extracted phone number is dialable under the
  configured region: a UK national format read as `"US"`, an extension fragment
  such as `"ext 4021"`, and any 555-area-code documentation number are all
  unusable *as numbers* while being exactly what the document said. The fallback
  is equality, not leniency -- `"N/A"` against `"unknown"` scores `0.0`, and when
  exactly **one** side is a usable number the score is `0.0`, because one side
  found a number and the other did not.

  Validity is still checked with `is_valid_number` rather than parseability.
  libphonenumber parses `"0000000000"` and renders it as E164, so a parse-only
  check would report a placeholder pair as a *canonical phone match* and would
  match two different placeholders that canonicalize alike. Under the fallback
  such a pair matches itself as a string instead: the score is the same and the
  claim behind it is honest. Genuinely absent values are unaffected, since the
  shared `None` policy resolves those before any comparator runs.

  Fixtures still want a real area code with the `555` **exchange** rather than
  555 in the area-code position: `"206-555-0100"` is fictional by convention
  (555-01xx is set aside for fiction) while being structurally valid, so it
  compares as a number rather than as a string.

  Extensions are compared separately, because E164 omits them:
  `"+12065550100x89"` and `"+12065550100x90"` reach different people and do not
  match.

  An unrecognised `region` raises `ValueError` at construction. `region="UK"`
  (the ISO code is `"GB"`) would otherwise leave every national-format number
  comparing as a string rather than as a number, with no error -- quieter than a
  typo has any right to be.

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
  threshold.

  `match_threshold=0.0` warns unconditionally, because the value reaches the
  comparison two ways: as the object-matching threshold for a
  `List[StructuredModel]` element, *and* as the default field threshold for any
  field with no explicit config of its own. A plainly annotated `name: str`
  inherits it, so a standalone model at `0.0` reports precision and recall
  `1.0` for a wholly incorrect prediction with no list involved. (Declaring the
  field with `ComparableField()` takes an earlier branch and gets a hardcoded
  `0.5`, which is why the value can look inert when probed that way -- see
  [#237](https://github.com/awslabs/stickler/issues/237).)

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
  `to_json_schema()`.

  Note a side effect: dropping the internal `extra_fields` property means
  `from_json_schema(M.model_json_schema())` now parses for a model whose fields
  are all required, where it previously raised `ValueError`. It still does
  **not** round-trip -- the rebuilt model carries default thresholds, weights
  and comparators, because a shape-only schema does not describe them. A model
  with any `Optional` field still raises, on the nullable `anyOf` gap that
  [#198](https://github.com/awslabs/stickler/pull/198) addresses, so most real
  models are unaffected either way. `model_json_schema()` remains documented as
  not round-trip-capable; use `to_json_schema()` or `to_stickler_config()` to
  preserve configuration. Tracked in
  [#214](https://github.com/awslabs/stickler/issues/214)
  ([#188](https://github.com/awslabs/stickler/issues/188))

- Optional fields built from a JSON Schema are now annotated `Optional[T]`
  rather than `T`, so `to_json_schema()` exports them with a nullable type:

  | | before | after |
  |---|---|---|
  | `to_json_schema()` | `{"type": "string"}` | `{"type": ["string", "null"]}` |
  | `model_json_schema()` | `{"type": "string"}` | `{"anyOf": [{"type": "string"}, {"type": "null"}]}` |

  The two spellings differ because `model_json_schema()` is Pydantic's own
  rendering of `Optional[T]`, while `to_json_schema()` uses the list form.
  Meaning is preserved and re-import is idempotent, but the exported bytes
  differ from the input schema, which matters when feeding our output into a
  validator or a codegen tool. `required` membership is unchanged
  ([#159](https://github.com/awslabs/stickler/pull/159))

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

### Deprecated

- **Custom comparators:** the extension point for comparators is now
  `_compare()` instead of `compare()`. `BaseComparator.compare()` is a
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

### Fixed

- `PhoneComparator` no longer scores two identical values `0.0`. When **neither**
  side is a usable number the comparison falls back to exact string equality, so
  a correctly extracted number that is not dialable under the configured region
  is not reported as a total mismatch against a byte-identical copy of itself. A
  UK national format read as `"US"`, an extension fragment like `"ext 4021"`, and
  any 555-area-code documentation number were all affected.

  This was reachable with no configuration at all. The `phone` name token routes
  every `phone`-shaped `str` field to `PhoneComparator(region="US")` at threshold
  `1.0`, and `evaluate()`/`eval_for()` take no region override, so a non-US or
  extension-bearing corpus had its precision and recall deflated with no signal
  why -- while `explain()` reported the routing as a deliberate choice.

  The fallback is equality, not leniency. `"N/A"` against `"unknown"` scores
  `0.0`, and when exactly **one** side is a usable number the score is `0.0`,
  because one side found a number and the other did not. Validity is still
  checked rather than mere parseability, so a placeholder pair matches itself as
  a *string* rather than canonicalizing into a plausible E164 and being reported
  as a phone-number match.

  Scores move **up** for affected corpora; nothing that scored `1.0` before
  changes. A region or per-field comparator override on the zero-config entry
  points is deferred to 0.8.0: it would widen the public API inside a release
  candidate, and it would not have fixed these cases on its own -- `"ext 4021"`
  is invalid under every region, and no single region works for a mixed corpus
  ([#258](https://github.com/awslabs/stickler/issues/258))

- `import stickler` no longer fails in environments carrying a boto3 stand-in
  module. `comparators/utils.py` ran `importlib.util.find_spec("boto3")` at
  module scope, and that module is on the import path via
  `comparators/semantic.py`. `find_spec` raises `ValueError` -- rather than
  returning `None` -- for an installed module whose `__spec__` is `None`, which
  is what a hand-rolled shim looks like, so the package became unimportable
  outright. The probe's result had no readers anywhere in the codebase, so it
  bought nothing; boto3 is still imported inside
  `generate_bedrock_embedding`, which is what keeps it off the import path
  ([#257](https://github.com/awslabs/stickler/issues/257))

- `LLMComparator` now resolves the same way from both import paths. Without the
  `llm` extra, `stickler.comparators.LLMComparator` returned a class that raised
  at instantiation while `stickler.LLMComparator` raised `AttributeError`, and
  neither module's `__all__` advertised the name -- so which import worked was a
  matter of luck. Both now gate on the `strands` probe, matching what `__all__`
  already did in both modules. Without the extra, both raise `AttributeError` and
  `hasattr()` is `False`; with it installed, nothing changes
  ([#259](https://github.com/awslabs/stickler/issues/259))

- A failed import no longer deregisters a built-in comparator.
  `ComparatorRegistry` removed a pending entry *before* attempting its import, so
  a broken-but-installed extra was dropped permanently by the first failed
  lookup: `is_registered()` returned `True` before the first `get()` and `False`
  after, and `list_available()` shrank as a side effect of a *failed* call. More
  quietly, `register()` rejects a name only when it is already known, so a failed
  resolve freed the built-in's name and let a later registration silently shadow
  it. A broken extra is still reported as unavailable rather than raising
  `ImportError` at the caller; only the bookkeeping changed
  ([#260](https://github.com/awslabs/stickler/issues/260))

- HTML reports no longer drop nested thresholds for a field annotated
  `Addr | None`. Threshold extraction unwrapped `Optional[X]` and
  `Union[X, None]` but tested only for the `typing.Union` origin, and a PEP 604
  union's origin is `types.UnionType`, so that spelling was never unwrapped and
  the nested-model and list branches never fired. The same class of bug fixed for
  `Optional[X]` in 0.6.0, left half-done -- and far more reachable now that 0.7.0
  raises the floor to Python 3.10, where `X | None` is the idiomatic spelling. A
  genuine multi-arm union is still left alone
  ([#162](https://github.com/awslabs/stickler/issues/162))

- Zero-config evaluation no longer scores formatting-only differences in
  `email`, `url` and `phone` fields as complete mismatches. The name-token
  inference rules route those fields to comparators chosen for them, and when
  `ExactComparator` became strict the rules silently inherited the change:
  `A.Buyer@Example.COM` vs `a.buyer@example.com`, and `206-555-0100` vs
  `(206) 555-0100`, went from `1.0` to `0.0` through plain
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

- Accept an explicit `null` for an optional field built from a JSON Schema.
  `from_json({"note": None})` raised `ValidationError` for a `{"type":
  "string"}` property absent from `required`; it now constructs and scores.
  Required fields are unaffected and still reject `None`. Only reproduced for
  fields nested at least one level down, and only on the
  `process_rich_values=True` path, which is why plain `ModelClass(**data)`
  appeared to work ([#159](https://github.com/awslabs/stickler/pull/159))

- `ConfigurationHelper.is_structured_field_type()` now recognises
  `Optional[SomeStructuredModel]`, where it previously returned `False`. This
  affects hand-written `StructuredModel` classes as well as schema-built ones,
  so a nullable nested model is now dispatched as a structured field rather
  than a primitive ([#159](https://github.com/awslabs/stickler/pull/159))

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

- Import Pydantic v2 JSON Schemas whose `Optional` fields use nullable
  two-branch `anyOf`, and infer nested object schemas when `type: object` is
  omitted ([#198](https://github.com/awslabs/stickler/pull/198))

- The zero-threshold `UserWarning` now links to documentation instead of to an
  issue tracker. Its target was
  `github.com/awslabs/stickler/issues/234`, an interim placeholder chosen
  because no docs section explained the cliff; the warning now points at
  [The Zero-Threshold Trap](https://awslabs.github.io/stickler/Getting-Started/thresholds-and-metrics/#the-zero-threshold-trap),
  written for it.

  That section also records what is and is not invariant at `0.0`, since the
  distinction is easy to overstate: no false discovery can ever be reported,
  because FD means "compared and scored below threshold" and nothing scores
  below `0.0`, but precision and recall are *not* both `1.0` in general, because
  unmatched items are not subject to any threshold.

  Two further threshold gaps are closed. `recall_with_fd` now documents that
  **TP → FD and FN → FD move recall in opposite directions**, so raising
  `match_threshold` does not reliably lower reported recall (checked
  exhaustively: 0 cases raised, 6 equal, 34 fell). And
  `Advanced/threshold-gated-evaluation.md` is retitled "How Below-Threshold
  Pairs Are Classified", since it covers classification rather than only
  recursion, and now defers to the Getting-Started explainer instead of
  restating threshold semantics. The page URL is unchanged
  ([#235](https://github.com/awslabs/stickler/issues/235))

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

[Unreleased]: https://github.com/awslabs/stickler/compare/v0.7.0...dev
[0.7.0]: https://github.com/awslabs/stickler/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/awslabs/stickler/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/awslabs/stickler/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/awslabs/stickler/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/awslabs/stickler/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/awslabs/stickler/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/awslabs/stickler/compare/v.0.1.4...v0.1.5
[0.1.4]: https://github.com/awslabs/stickler/compare/v.0.1.3...v.0.1.4
[0.1.3]: https://github.com/awslabs/stickler/releases/tag/v.0.1.3
