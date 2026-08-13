"""Tests for the ``infer_unspecified_fields`` opt-in flag (issue #239).

The flag routes fields WITHOUT a ``comparator`` (in ``model_from_json`` config)
or WITHOUT ``x-aws-stickler-comparator`` (in JSON Schema) through
``stickler.auto.inference`` for a type + name-aware comparator pick. It also
enriches ``format: date`` / ``format: date-time`` / ``enum`` primitive
mapping to their richer Python types on the JSON Schema path.

Load-bearing invariants pinned here:

- Flag off ⇒ behavior is unchanged from before the flag.
- Flag on ⇒ un-annotated fields get a type + name-aware comparator;
  provenance is retrievable via ``json_schema_extra._provenance``.
- Explicit config always wins per-parameter (author's threshold overrides
  the inferred threshold; inferred comparator still fills in).
- ``format: date`` and ``enum`` produce ``DateComparator`` and
  ``ExactComparator`` on the JSON Schema path when the flag is on.

The weak-spot type categories from #239 — ``format: date``, ``enum``, and
numeric types — get direct coverage here since they were the three cases
Spencer specifically called out as differentiators from the flat type-only
fallback.
"""

import datetime
import enum

import pytest

from stickler.comparators.date import DateComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.fuzzy import FuzzyComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


def _field_comparator(model_cls, field_name):
    """Read the comparator instance attached to a field via json_schema_extra."""
    extra = model_cls.model_fields[field_name].json_schema_extra
    return getattr(extra, "_comparator_instance", None)


def _field_provenance(model_cls, field_name):
    """Read the provenance list attached to a field via json_schema_extra.

    Returns ``None`` when the field was operator-configured (no inference
    fired) or when the model predates the flag. Returns a list of ordered
    rule-application strings when inference fired for the field.
    """
    extra = model_cls.model_fields[field_name].json_schema_extra
    return getattr(extra, "_provenance", None)


def _field_threshold(model_cls, field_name):
    """Read the threshold attached to a field via json_schema_extra."""
    extra = model_cls.model_fields[field_name].json_schema_extra
    return getattr(extra, "_threshold", None)


def _field_weight(model_cls, field_name):
    """Read the weight attached to a field via json_schema_extra."""
    extra = model_cls.model_fields[field_name].json_schema_extra
    return getattr(extra, "_weight", None)


# ---------------------------------------------------------------------------
# model_from_json — flag off preserves the strict "requires comparator" gate
# ---------------------------------------------------------------------------


class TestModelFromJsonFlagOff:
    """Flag OFF must preserve the pre-flag behavior exactly.

    The strict "primitive fields require a comparator" gate at
    ``field_converter.validate_nested_field_schema`` was existing behavior;
    flipping the flag on relaxes it but flipping it off (or omitting it)
    must leave every existing consumer's error path intact.
    """

    def test_missing_comparator_still_raises(self):
        """Pinned by test_model_from_json.py:792-804 as well; asserted here
        against the specific field-shape the flag would relax so a future
        refactor that accidentally always-relaxes fails this test."""
        config = {
            "model_name": "M",
            "fields": {
                "invoice_id": {"type": "str"},
            },
        }
        with pytest.raises(ValueError, match=r"requires.*'comparator'"):
            StructuredModel.model_from_json(config)

    def test_missing_comparator_still_raises_with_flag_explicitly_off(self):
        """Explicit ``infer_unspecified_fields=False`` behaves identically
        to omitting the kwarg — no drift between the two flag-off paths."""
        config = {
            "model_name": "M",
            "fields": {
                "invoice_id": {"type": "str"},
            },
        }
        with pytest.raises(ValueError, match=r"requires.*'comparator'"):
            StructuredModel.model_from_json(config, infer_unspecified_fields=False)


# ---------------------------------------------------------------------------
# model_from_json — flag on infers via type + name, attaches provenance
# ---------------------------------------------------------------------------


class TestModelFromJsonFlagOn:
    """Flag ON routes un-annotated fields through auto-inference.

    Covers the three weak-spot type categories called out in issue #239:
    - Numeric types (``int`` / ``float``) via type-only rules.
    - Date types via ``datetime.date`` / ``datetime.datetime`` annotations.
    - Enum-shaped fields via ``enum.Enum`` subclasses (assessed indirectly
      here since ``model_from_json`` accepts type strings — enums arrive
      through the JSON Schema path in ``TestFromJsonSchemaFlagOn`` below).
    """

    def test_missing_comparator_no_longer_raises_with_flag(self):
        """The strict gate at validate_nested_field_schema is relaxed
        when the caller opts in."""
        config = {
            "model_name": "M",
            "fields": {
                "invoice_id": {"type": "str"},
            },
        }
        # Must not raise
        Model = StructuredModel.model_from_json(config, infer_unspecified_fields=True)
        assert "invoice_id" in Model.model_fields

    def test_inference_picks_exact_by_name_token_for_id_field(self):
        """`invoice_id` is a Stickler name-token rule (``*_id``) that
        overrides the ``str → Levenshtein`` type default. Provenance
        trace must record both rules — type first, then the name-token
        upgrade."""
        config = {
            "model_name": "M",
            "fields": {"invoice_id": {"type": "str"}},
        }
        Model = StructuredModel.model_from_json(config, infer_unspecified_fields=True)

        assert isinstance(_field_comparator(Model, "invoice_id"), ExactComparator)

        provenance = _field_provenance(Model, "invoice_id")
        assert provenance is not None
        assert any("type:str" in step for step in provenance), provenance
        assert any(
            "name-token:invoice_id" in step and "ExactComparator" in step
            for step in provenance
        ), provenance

    def test_inference_picks_numeric_for_int_and_float(self):
        """Numeric types were called out in issue #239 as a weak spot —
        the flat table maps them to ``NumericComparator`` today too, but
        the flag path also injects the type-appropriate tolerance via
        ``comparator_config``. Sanity-check both int and float produce
        NumericComparator with a non-None provenance."""
        config = {
            "model_name": "M",
            "fields": {
                "quantity": {"type": "int"},
                "total_amount": {"type": "float"},
            },
        }
        Model = StructuredModel.model_from_json(config, infer_unspecified_fields=True)

        assert isinstance(_field_comparator(Model, "quantity"), NumericComparator)
        assert isinstance(_field_comparator(Model, "total_amount"), NumericComparator)

        assert _field_provenance(Model, "quantity") is not None
        # total_amount matches the ``*amount`` name-token — provenance
        # should record both type and name-token rules.
        prov = _field_provenance(Model, "total_amount")
        assert prov is not None
        assert any("name-token:total_amount" in step for step in prov), prov

    # NOTE: Date-typed fields on the ``model_from_json`` path are not
    # exercised here because ``type_resolver.resolve_type_string`` — the
    # existing helper that maps ``config["type"]`` strings to Python
    # types — does not recognize ``"date"`` / ``"datetime"`` as type
    # names today. That gap is orthogonal to this flag; the JSON Schema
    # path covers date types via ``format: date`` (see
    # ``TestFromJsonSchemaFlagOn.test_format_date_picks_date_comparator``).
    # Extending ``type_resolver`` is a separate change — filing a
    # follow-up issue would be the right home for it.

    def test_configured_fields_are_untouched(self):
        """Fields WITH an explicit ``comparator`` must not carry a
        provenance list — provenance's absence is the "author-configured"
        signal downstream code (e.g., a Source column renderer) reads."""
        config = {
            "model_name": "M",
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "threshold": 0.9,
                },
                "invoice_id": {"type": "str"},  # inferred
            },
        }
        Model = StructuredModel.model_from_json(config, infer_unspecified_fields=True)

        # Author-declared field: comparator matches declaration, no
        # provenance attached.
        assert isinstance(_field_comparator(Model, "name"), LevenshteinComparator)
        assert _field_provenance(Model, "name") is None

        # Inferred field: provenance present.
        assert isinstance(_field_comparator(Model, "invoice_id"), ExactComparator)
        assert _field_provenance(Model, "invoice_id") is not None

    def test_per_parameter_merge_threshold_wins_over_inferred(self):
        """Decision 2 semantics: author's explicit ``threshold`` wins
        over the InferredSpec's threshold, but the comparator is still
        inferred because ``comparator`` was omitted. Merges at the
        parameter granularity, not the field granularity.
        """
        config = {
            "model_name": "M",
            "fields": {
                # No comparator, but threshold explicitly set to 0.9.
                # Under the flag, inference picks Fuzzy (name-token
                # ``notes``) at its own threshold; the merge should
                # honor 0.9 instead.
                "notes": {"type": "str", "threshold": 0.9},
            },
        }
        Model = StructuredModel.model_from_json(config, infer_unspecified_fields=True)

        assert isinstance(_field_comparator(Model, "notes"), FuzzyComparator)
        assert _field_threshold(Model, "notes") == pytest.approx(0.9)
        # Weight-hint rules only fire when the caller passes
        # ``weight_hints=True`` into ``infer_field_config``. The flag
        # itself is scoped to comparator inference (that's what the
        # motivating IDP use case wants), so weight stays at the
        # InferredSpec default of 1.0. When a future extension lets
        # callers opt into weight-hint enrichment, that path is what
        # would raise this value to 0.3 for a ``notes`` field.
        assert _field_weight(Model, "notes") == pytest.approx(1.0)

    def test_flag_propagates_through_nested_structured_model(self):
        """Mirror of ``test_flag_propagates_through_nested_object`` on the
        ``model_from_json`` path: un-annotated leaves inside a nested
        ``type: structured_model`` block must inherit the parent's flag
        setting so ``_convert_nested_model_field`` doesn't drop it on the
        recursive call."""
        config = {
            "model_name": "M",
            "fields": {
                "invoice_id": {"type": "str"},
                "address": {
                    "type": "structured_model",
                    "fields": {
                        "zip_code": {"type": "str"},
                        "state": {"type": "str"},
                    },
                },
            },
        }
        Model = StructuredModel.model_from_json(config, infer_unspecified_fields=True)

        # Top-level inferred.
        assert isinstance(_field_comparator(Model, "invoice_id"), ExactComparator)

        # Reach into the nested model class and check its fields too.
        AddressModel = Model.model_fields["address"].annotation
        from typing import get_args, get_origin

        if get_origin(AddressModel) is not None:
            args = [a for a in get_args(AddressModel) if a is not type(None)]
            if args:
                AddressModel = args[0]

        # Nested un-annotated fields also went through inference (i.e.,
        # the flag threaded through the ``_convert_nested_model_field``
        # recursive call at ``field_converter.py:225``).
        assert _field_provenance(AddressModel, "zip_code") is not None
        assert _field_provenance(AddressModel, "state") is not None

    def test_per_parameter_merge_weight_wins_over_inferred(self):
        """Same as threshold merge but on the weight parameter — makes
        sure the merge is uniform across all three parameters. The
        author's explicit weight overrides the InferredSpec default
        (which itself defaults to 1.0 without weight_hints — see
        ``test_per_parameter_merge_threshold_wins_over_inferred``)."""
        config = {
            "model_name": "M",
            "fields": {
                "invoice_id": {"type": "str", "weight": 5.0},
            },
        }
        Model = StructuredModel.model_from_json(config, infer_unspecified_fields=True)

        # Author's 5.0 wins.
        assert _field_weight(Model, "invoice_id") == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# from_json_schema — flag off preserves the flat type-only fallback
# ---------------------------------------------------------------------------


class TestFromJsonSchemaFlagOff:
    """Flag OFF must not change any pre-existing JSON Schema behavior.

    Pinned by four extension tests at
    ``test_from_json_schema.py:1109-1173`` that specify partial extensions
    (only weight, only threshold+clip, etc.) and rely on the flat
    Levenshtein/Numeric/Exact defaults for the unspecified comparator.
    """

    def test_no_extension_falls_back_to_flat_default(self):
        schema = {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "amount": {"type": "number"},
                "paid": {"type": "boolean"},
            },
        }
        Model = StructuredModel.from_json_schema(schema)

        # Flat defaults, ignoring name-token signals.
        assert isinstance(_field_comparator(Model, "invoice_id"), LevenshteinComparator)
        assert isinstance(_field_comparator(Model, "amount"), NumericComparator)
        assert isinstance(_field_comparator(Model, "paid"), ExactComparator)
        # No provenance attached — inference didn't fire.
        assert _field_provenance(Model, "invoice_id") is None

    def test_format_date_stays_str_when_flag_off(self):
        """The ``format: date`` enrichment is gated on the flag —
        without it, the field type stays ``str`` and gets Levenshtein
        (preserving pre-flag behavior for schemas that already declared
        ``format: date`` for documentation only)."""
        schema = {
            "type": "object",
            "properties": {
                "issued": {"type": "string", "format": "date"},
            },
        }
        Model = StructuredModel.from_json_schema(schema)

        # The pydantic field annotation should still be str, not date.
        # Check by exercising validation: a string that isn't a valid
        # date string must NOT raise (would if annotation were date).
        instance = Model(issued="not-a-date")
        assert instance.issued == "not-a-date"

    def test_enum_stays_str_when_flag_off(self):
        """Same gating for enum — without the flag, ``{"type": "string",
        "enum": [...]}`` stays plain ``str`` and coerces raw string values,
        preserving the behavior pinned by
        ``test_complex_nested_schema_with_enums``."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["DRAFT", "FINAL"]},
            },
        }
        Model = StructuredModel.from_json_schema(schema)

        instance = Model(status="DRAFT")
        # Value stays a raw string — the Enum enrichment did NOT fire.
        assert instance.status == "DRAFT"
        assert isinstance(instance.status, str)


# ---------------------------------------------------------------------------
# from_json_schema — flag on infers, enriches format/enum, attaches provenance
# ---------------------------------------------------------------------------


class TestFromJsonSchemaFlagOn:
    """Flag ON routes un-annotated fields through auto-inference and
    enriches ``format: date`` / ``format: date-time`` / ``enum``.
    """

    def test_no_extension_uses_inference_with_flag(self):
        """When the schema has no comparator extension AND the flag is
        on, ``stickler.auto.inference`` picks the comparator. On an
        ``invoice_id`` string field it should pick Exact via the
        name-token rule instead of the flat Levenshtein default."""
        schema = {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        assert isinstance(_field_comparator(Model, "invoice_id"), ExactComparator)
        prov = _field_provenance(Model, "invoice_id")
        assert prov is not None
        assert any("name-token:invoice_id" in step for step in prov)

    def test_format_date_picks_date_comparator(self):
        """The ``format: date`` enrichment lands only when the flag is
        on. Under the flag, ``{"type": "string", "format": "date"}``
        resolves to ``datetime.date`` and inference picks
        DateComparator."""
        schema = {
            "type": "object",
            "properties": {
                "issued": {"type": "string", "format": "date"},
            },
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        assert isinstance(_field_comparator(Model, "issued"), DateComparator)
        # Field annotation should be datetime.date, and pydantic should
        # coerce string date values into ``date`` objects.
        instance = Model(issued="2024-05-15")
        assert isinstance(instance.issued, datetime.date)
        assert instance.issued == datetime.date(2024, 5, 15)

        prov = _field_provenance(Model, "issued")
        assert prov is not None
        assert any("DateComparator" in step for step in prov), prov

    def test_format_date_time_picks_date_comparator(self):
        """Same enrichment as ``format: date`` but for ``date-time``:
        DateComparator handles both. Different Python type
        (``datetime.datetime``) means pydantic coerces the value more
        precisely."""
        schema = {
            "type": "object",
            "properties": {
                "created_at": {"type": "string", "format": "date-time"},
            },
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        assert isinstance(_field_comparator(Model, "created_at"), DateComparator)
        instance = Model(created_at="2024-05-15T10:30:00Z")
        assert isinstance(instance.created_at, datetime.datetime)

    def test_enum_picks_exact_comparator(self):
        """Enum-shaped strings become a synthesized Enum class under the
        flag; Stickler's inference layer then picks ExactComparator via
        the ``_is_enum`` rule."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["DRAFT", "PENDING", "FINAL"]},
            },
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        assert isinstance(_field_comparator(Model, "status"), ExactComparator)
        # Value becomes an Enum member (the whole point of the
        # enrichment — the pydantic field is a real Enum subclass now).
        instance = Model(status="DRAFT")
        assert isinstance(instance.status, enum.Enum)
        assert instance.status.value == "DRAFT"

        prov = _field_provenance(Model, "status")
        assert prov is not None

    def test_numeric_types_pick_numeric_comparator(self):
        """Numeric types under the flag get the same NumericComparator
        pick as the flat table, but with type-appropriate tolerance
        injected via ``comparator_config`` and provenance attached."""
        schema = {
            "type": "object",
            "properties": {
                "quantity": {"type": "integer"},
                "total": {"type": "number"},
            },
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        assert isinstance(_field_comparator(Model, "quantity"), NumericComparator)
        assert isinstance(_field_comparator(Model, "total"), NumericComparator)
        assert _field_provenance(Model, "quantity") is not None
        assert _field_provenance(Model, "total") is not None

    def test_configured_extension_still_wins(self):
        """A field with an explicit ``x-aws-stickler-comparator``
        extension must be honored verbatim — inference must not
        overwrite operator choice. Provenance stays absent for
        author-configured fields.
        """
        schema = {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                },
                "invoice_id": {"type": "string"},  # inferred
            },
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        # Author's Levenshtein wins over inferred Fuzzy for ``notes``.
        assert isinstance(_field_comparator(Model, "notes"), LevenshteinComparator)
        assert _field_provenance(Model, "notes") is None

        # Un-annotated field still gets Exact via inference.
        assert isinstance(_field_comparator(Model, "invoice_id"), ExactComparator)
        assert _field_provenance(Model, "invoice_id") is not None

    def test_per_parameter_merge_threshold_extension_wins(self):
        """Decision 2 for the JSON Schema path — author's explicit
        ``x-aws-stickler-threshold`` merges with the inferred
        comparator instead of the inferred threshold."""
        schema = {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "x-aws-stickler-threshold": 0.9,
                },
            },
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        assert isinstance(_field_comparator(Model, "notes"), FuzzyComparator)
        assert _field_threshold(Model, "notes") == pytest.approx(0.9)

    def test_list_of_format_date_enriches_element_type(self):
        """The enrichment gating in ``_handle_array_type`` must apply to
        primitive array items too: an array of ``format: date`` strings
        under the flag becomes ``List[datetime.date]`` bound to a
        ``DateComparator``, not ``List[str]`` with Levenshtein. Without
        this test the item-level enrichment branch would be untested."""
        from typing import get_args, get_origin

        schema = {
            "type": "object",
            "properties": {
                "invoice_dates": {
                    "type": "array",
                    "items": {"type": "string", "format": "date"},
                },
            },
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        annotation = Model.model_fields["invoice_dates"].annotation
        # Field is Optional[List[datetime.date]] (non-required default None).
        if get_origin(annotation) is not None:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if args:
                annotation = args[0]
        assert get_origin(annotation) is list
        (element_type,) = get_args(annotation)
        assert element_type is datetime.date

        assert isinstance(_field_comparator(Model, "invoice_dates"), DateComparator)

        # Pydantic actually coerces the string items into ``date`` objects.
        instance = Model(invoice_dates=["2024-05-15", "2024-06-01"])
        assert all(isinstance(d, datetime.date) for d in instance.invoice_dates)

    def test_format_date_with_explicit_comparator_keeps_type_enrichment(self):
        """Type enrichment happens BEFORE comparator selection, so an
        explicit ``x-aws-stickler-comparator`` doesn't undo the ``format:
        date`` → ``datetime.date`` type change. Pydantic still coerces
        the string into a ``date`` object; the operator's chosen
        comparator (here ``LevenshteinComparator``) runs against the
        stringified date at compare time. Pinning this so a future
        change doesn't accidentally couple enrichment to comparator
        absence."""
        schema = {
            "type": "object",
            "properties": {
                "issued": {
                    "type": "string",
                    "format": "date",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                },
            },
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        # Author's comparator wins.
        assert isinstance(_field_comparator(Model, "issued"), LevenshteinComparator)
        # No provenance — inference didn't fire.
        assert _field_provenance(Model, "issued") is None
        # Type enrichment still happened.
        instance = Model(issued="2024-05-15")
        assert isinstance(instance.issued, datetime.date)

    def test_flag_propagates_through_nested_object(self):
        """Un-annotated leaves inside a nested object must inherit the
        parent's flag setting — otherwise ``address.city`` would fall
        back to flat defaults even though the top-level model opted in."""
        schema = {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "zip_code": {"type": "string"},
                        "state": {"type": "string"},
                    },
                },
            },
        }
        Model = StructuredModel.from_json_schema(schema, infer_unspecified_fields=True)

        # Top-level inferred.
        assert isinstance(_field_comparator(Model, "invoice_id"), ExactComparator)

        # Reach into the nested model class and check its fields too.
        AddressModel = Model.model_fields["address"].annotation
        # Optional wrapping: unwrap if needed.
        from typing import get_args, get_origin

        if get_origin(AddressModel) is not None:
            args = [a for a in get_args(AddressModel) if a is not type(None)]
            if args:
                AddressModel = args[0]

        # Nested un-annotated fields also went through inference.
        assert _field_provenance(AddressModel, "zip_code") is not None
        assert _field_provenance(AddressModel, "state") is not None
