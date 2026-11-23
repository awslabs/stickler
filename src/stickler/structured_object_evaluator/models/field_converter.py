"""Field converter for dynamic model creation.

This module provides utilities for converting JSON field configurations to
Pydantic Field instances with ComparableField functionality.
"""

import sys
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Type, get_args, get_origin

from pydantic import Field

from stickler.utils.deprecation import warn_once

from .comparable_field import (
    ComparableField,
    _mark_unstated,
    _named_comparator_threshold,
)
from .comparator_registry import create_comparator, normalize_comparator_config
from .threshold_rule import (
    CONFIG,
    ComparatorSource,
    Spelling,
    _of_model,
    is_threshold_error,
    is_valid_threshold,
    threshold_error,
)
from .threshold_rule import (
    STICKLER_DIR as _STICKLER_DIR,
)
from .type_resolver import resolve_type_string

# Prepended to an inferred spec whose field is a LIST, because inference reads the
# ELEMENT type and the resulting comparator/threshold apply per element rather than
# to the list as a whole. Shared with the JSON Schema importer, which reaches the
# same inference with the element already unwrapped: spelling it in both places let
# `explain()` report an array field with a scalar-looking trail on one path only.
LIST_ELEMENT_PROVENANCE = "list: spec applies to each element"

#: Field-level opt-in: `"comparator": "auto"` asks for the same inference
#: `stickler.evaluate()` uses, for this field only. Chosen over a separate key so a
#: reader of the config sees the decision on the line that would otherwise name a
#: comparator, and so it cannot be set alongside a real comparator name.
AUTO_COMPARATOR = "auto"

#: Every key a field config may carry. The union of what the primitive and
#: structured_model branches of ``convert_field_config`` read, plus
#: ``model_name``: a nested field never reads it -- ``_build_nested_model``
#: synthesises the class name -- but ``to_stickler_config()`` exports it, so
#: rejecting it here would make the library's own export fail to round-trip.
#: Kept as one set rather than per-branch sets so that a key which is valid on
#: the other branch is never reported as a typo.
ACCEPTED_FIELD_CONFIG_KEYS = frozenset(
    {
        "type",
        "comparator",
        "comparator_config",
        "threshold",
        "weight",
        "clip_under_threshold",
        "default",
        "required",
        "description",
        "alias",
        "examples",
        "infer_unspecified_fields",
        "fields",
        "match_threshold",
        "model_name",
    }
)

#: Frames between ``warnings.warn`` and the user's ``model_from_json`` call, for
#: the unknown-key warning. ``warn_once``'s default of 3 assumes one wrapping
#: call; this one is reached through six, so at 3 the warning was attributed to
#: this module instead of to the line the reader can fix. Counted outward from
#: ``warnings.warn``: warn_once, validate_field_config, the converter's
#: validate_fields_config, its module-level wrapper, create_model_from_json,
#: model_from_json, caller.
_UNKNOWN_KEY_STACKLEVEL = 7

#: The same count for the ``match_threshold``-on-a-primitive warning, from
#: ``warnings.warn``: warn_once, convert_field_config, the converter's
#: convert_fields_config, its module-level wrapper, create_model_from_json,
#: model_from_json, caller. Exact for a root field; a nested field is reached
#: through more frames, and the message names its dotted path instead.
_MATCH_THRESHOLD_STACKLEVEL = 7


def _infer_spec(
    field_name: str,
    field_type: Any,
    match_threshold: Optional[float] = None,
    *,
    already_element: bool = False,
):
    """Infer a comparison spec for one config-driven field from its resolved type.

    ``stickler.auto`` infers from a live ``FieldInfo``, and a config-driven field
    has none, so one is synthesised around the resolved annotation. That is enough:
    ``infer_field_config`` reads only the annotation and the field name, so the
    name-token heuristics work here too.

    ``match_threshold`` is forwarded because inference uses it as the FIELD
    threshold for a mapping field: a dict declines to name its keys, so it is
    judged as an object and takes the object-level value rather than the scalar
    default. Omitting it left ``{"type": "dict"}`` at 0.7 while
    ``stickler.eval_for(cls, match_threshold=0.9)`` gave 0.9 for the same field.

    A parameterized list infers from its ELEMENT type, because that is what the
    comparator is applied to. Inferring from ``List[float]`` itself landed on the
    exotic-type branch (whole-list canonical-JSON equality) while both the JSON
    Schema path and ``stickler.evaluate()`` scored the same shape per element.

    ``already_element`` says the CALLER has already unwrapped one list level, so
    this function must not unwrap another. Only the JSON Schema path passes it,
    because its array branch hands over the element type rather than the list.

    Without it a NESTED array was unwrapped twice and inferred from the innermost
    scalar, so the field's comparator was handed a ``list`` it could not read and
    an IDENTICAL pair scored 0.0:

        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}

                                     comparator            [[1,2,3,4]] vs itself
        flag off                      Exact@1.0             1.0
        flag on, unwrapped twice      Numeric@0.95          0.0
        flag on, this fix             Exact@1.0             1.0

    ``stickler.eval_for`` on the same shape unwraps ONCE and lands on the
    whole-list canonical-JSON branch, which is the answer this now matches -- the
    parity the docs promise for this flag. The deliberate whole-list branch exists
    because edit distance over a JSON blob is not a defensible metric, so
    descending past it was losing that decision as well as the score.
    """
    from pydantic.fields import FieldInfo

    from stickler.auto.inference import infer_field_config, unwrap_optional

    inner, _ = unwrap_optional(field_type)
    # `list` origin only, matching ``auto.builder._field_kind``: a `Set[...]`
    # there is a primitive, not a per-element list, and diverging here would
    # trade one entry-point disagreement for another.
    if get_origin(inner) is list and not already_element:
        args = get_args(inner)
        # Unparameterized `List` has no element type; `Any` is what the builder
        # infers from in that position, so the two agree on Exact@1.0.
        element, _ = unwrap_optional(args[0]) if args else (Any, False)
        spec = infer_field_config(
            field_name,
            FieldInfo(annotation=element),
            match_threshold=match_threshold,
        )
        spec.provenance.insert(0, LIST_ELEMENT_PROVENANCE)
        return spec

    return infer_field_config(
        field_name,
        FieldInfo(annotation=field_type),
        match_threshold=match_threshold,
    )


def _nested_infer_flag(
    field_config: Dict[str, Any],
    infer_unspecified: bool,
    field_name: str,
) -> bool:
    """The inference flag governing a nested model's own subtree.

    A nested field may set ``infer_unspecified_fields`` to scope inference to its
    own subtree in either direction, matching what a nested object can do on the
    JSON Schema path (``json_schema_importer._build_nested_model``). Hardcoding the
    parent's value discarded the nested setting both ways: a nested `false` still
    inferred, and a nested `true` was answered with an error advising exactly what
    the author had already written.
    """
    declared = field_config.get("infer_unspecified_fields", infer_unspecified)
    if not isinstance(declared, bool):
        raise ValueError(
            "infer_unspecified_fields must be true or false on field "
            f"'{field_name}', got: {declared!r}"
        )
    return declared


def _stacklevel_outside_stickler(*, through_warn_once: bool = False) -> int:
    """The ``stacklevel`` that attributes a warning to the first caller outside stickler.

    A fixed count is right for one depth only. A nested model is built by
    re-entering the factory, so each level of nesting adds frames, and a
    constant tuned for the root attributed a warning two levels down to
    library code. Python's default filter shows a message once per text and
    line, so every model then shared one library line and one warning.

    Call it from the function that calls ``warnings.warn``. Pass
    ``through_warn_once`` when that call goes through ``warn_once``, which adds
    its own frame.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_STICKLER_DIR):
        frame = frame.f_back
        level += 1
    return level + 1 if through_warn_once else level


def _nested_model_comparator_threshold(
    comparator: Any, comparator_config: Mapping, *, reads_threshold: bool
) -> Tuple[Optional[float], List[str], Any]:
    """Split a nested-model field's comparator keys into what is read and what is not.

    Both importers resolve these as ``ComparableField`` does on the Python path.
    A nested model is scored field by field, so its comparator is never called:
    ``ExactComparator`` and ``LevenshteinComparator`` there score a half-matching
    child the same 0.5. A threshold named on that comparator IS read, though:
    ``ComparableField(comparator=ExactComparator(threshold=0.8))`` resolves 0.8
    and gates the child. That holds whether the threshold is written in the
    config or set by the comparator's own constructor, so a named comparator is
    always asked. A list of models reads neither (``reads_threshold`` False),
    because Hungarian matching pairs elements on the element class's
    ``match_threshold`` and Python warns that a comparator threshold there is not
    consulted.

    ``comparator`` is the instance built from a named comparator, or ``None`` when
    the field gives only a config. Returns the threshold to adopt or ``None``,
    the names of the config keys that have no effect, and a named threshold that
    breaks the threshold rule (``None`` if there is none). The caller
    raises on that one unless the field states its own threshold, which wins.
    """
    named = (
        _named_comparator_threshold(comparator)
        if comparator is not None
        else comparator_config.get("threshold")
    )
    adopted = named if reads_threshold and is_valid_threshold(named) else None
    rejected = (
        named
        if reads_threshold and named is not None and not is_valid_threshold(named)
        else None
    )
    ignored = sorted(
        str(key)
        for key in comparator_config
        if not (key == "threshold" and reads_threshold and named is not None)
    )
    return adopted, ignored, rejected


def _model_label(model_id: str) -> Optional[str]:
    """The model's name for an error, or ``None`` when the config gives none.

    ``model_identity`` appends ``(fields: ...)`` to an unnamed model so two of
    them dedupe apart, and ``DynamicModel`` is a name the author never wrote.
    A ``model_name`` is an identifier, so it cannot contain the parenthesis.
    """
    label = model_id.partition("(")[0]
    return None if label in ("", "DynamicModel") else label


def _raise_comparator_threshold(
    value: Any,
    config: Mapping,
    *,
    spelling: Spelling,
    comparator_name: str,
    comparator_key: str,
    config_key: str,
    named: bool,
    field: str,
    model: Optional[str],
    nested: bool = False,
    defined_at: Optional[str] = None,
) -> None:
    """Raise for a comparator threshold the field adopts and that breaks the rule.

    Adopted unchecked, 85 failed every pair, ``"0.9"`` raised on every
    ``compare_with`` and -1 passed every pair. ``comparator_key`` and
    ``config_key`` are the keys as this path spells them (``comparator``,
    ``comparator_config``). When the config states no threshold, the
    comparator's own constructor set it, and the message names the comparator
    instead. ``named`` says whether the author named the comparator or only gave
    a config, which decides what a nested model, which reads only the
    threshold, should remove: the key the value is in, which is the config
    whenever it states the threshold.
    """
    # `{}` is the value; the doubled braces are the config's own.
    written = (
        '"' + config_key + '": {{"threshold": {}}}' if "threshold" in config else None
    )
    carrier = f'"{comparator_key}"' if named else f'"{config_key}"'
    raise threshold_error(
        value,
        spelling=spelling,
        field=field,
        model=model,
        comparator=ComparatorSource(
            written=written,
            name=f'"{comparator_name}"',
            carrier=carrier,
            remove=f'"{config_key}"' if named and written else "it",
        ),
        nested=nested,
        defined_at=defined_at,
    )


def _config_keys_phrase(config_key: str, keys: List[str]) -> str:
    """``'comparator_config' key 'a'`` or ``... keys 'a', 'b'`` for a warning."""
    plural = "keys" if len(keys) > 1 else "key"
    return f"'{config_key}' {plural} {', '.join(repr(k) for k in keys)}"


def _nested_model_keys_warning(
    field_path: str,
    keys: List[str],
    *,
    shape: str,
    is_list: bool,
    members: str,
    threshold_key: str,
    model: Optional[str],
    spelling: Spelling,
    adopted: Optional[float] = None,
) -> str:
    """The warning for comparator keys a nested-model field does not read.

    One wording for both importers, which differ only in what they call the
    shape and the key to use instead. Warned, not raised: the Python path accepts
    a comparator on a nested model and ignores it, and these keys loaded on both
    importers before, so a raise would break configs that load today.

    ``adopted`` is a threshold the named comparator carries and the field reads.
    The warning then says so and gives the value, since removing the comparator
    as advised would otherwise drop the gate and move the verdict.

    ``model`` and ``spelling`` name the model as the threshold errors do: by its
    name, or as an unnamed model with how to name it, never by the
    ``(fields: ...)`` identity that keys warn-once. The name is in the text
    because Python's default filter shows a message once per text and line, so
    two named models with the same field would otherwise show one warning
    between them.
    """
    subject = " and ".join(keys)
    # A phrase from _config_keys_phrase can name several keys on its own.
    plural = len(keys) > 1 or any(" keys " in k for k in keys)
    verb, be = ("have", "are") if plural else ("has", "is")
    where = f"field '{field_path}'{_of_model(model, spelling)}"
    if is_list:
        return (
            f"{subject} {verb} no effect on {where} and "
            f"{be} ignored: it is {shape}, which matches its items using "
            f"{threshold_key} and never calls a comparator. Set comparators on "
            f"the {members} inside each element, and how items match with "
            f"{threshold_key}, a number from 0.0 to 1.0."
        )
    if adopted is not None:
        return (
            f"{subject} {verb} no effect on {where} and "
            f"{be} ignored, except for its threshold {adopted!r}, which this "
            f"field uses: it is {shape}, scored field by field, so its "
            f"comparator is never called. Set comparators on the {members} "
            f"inside it, and set {threshold_key} to {adopted!r} to keep that "
            "threshold without the comparator."
        )
    return (
        f"{subject} {verb} no effect on {where} and "
        f"{be} ignored: it is {shape}, scored "
        "field by field, so its comparator is never called. Set comparators on "
        f"the {members} inside it, and this field's threshold with "
        f"{threshold_key}."
    )


def _wants_inference(field_config: Dict[str, Any], infer_unspecified: bool) -> bool:
    """Whether this field's unspecified parameters should be inferred.

    A field-level setting always wins over the model-level flag, in both
    directions: ``"comparator": "auto"`` opts one field in when the model did not,
    and naming a real comparator pins one field when the model opted everything in.

    Naming a comparator pins the field's threshold too, rather than inferring it.
    A threshold is only meaningful beside the metric that produced the score --
    0.85 means one thing on edit distance and another on numeric tolerance -- so
    inference's threshold belongs to the comparator inference would have chosen,
    not to the one the caller named. Per-parameter filling applies to a field that
    let inference pick the comparator.
    """
    declared = field_config.get("comparator")
    if declared == AUTO_COMPARATOR:
        return True
    if declared is not None:
        return False
    return infer_unspecified


class FieldConverter:
    """Converter for JSON field configurations to Pydantic fields."""

    def convert_field_config(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        infer_unspecified: bool = False,
        match_threshold: Optional[float] = None,
        path_prefix: str = "",
        model_id: str = "",
    ) -> Tuple[Type, Any]:
        """Convert a JSON field configuration to a Pydantic field definition.

        Args:
            field_name: Name of the field
            field_config: JSON configuration for the field
            infer_unspecified: When True, parameters the config does not name are
                inferred from the field's type and name rather than defaulted.
                Off by default, because turning it on moves reported metrics for
                any field that named no comparator.
            match_threshold: The model's ``match_threshold``, forwarded to
                inference because a mapping field takes it as its field threshold.
            path_prefix: Dotted path of the enclosing field, ``""`` at the root.
            model_id: ``model_identity`` of the root model.

        Returns:
            Tuple of (field_type, pydantic_field)

        Raises:
            ValueError: If configuration is invalid
        """
        field_path = f"{path_prefix}{field_name}"

        # Extract field type
        type_string = field_config.get("type")
        if not type_string:
            raise ValueError(f"Field '{field_name}' missing required 'type' parameter")

        # Check if this is a structured model type
        if type_string in [
            "structured_model",
            "list_structured_model",
            "optional_structured_model",
        ]:
            return self._convert_nested_model_field(
                field_name,
                field_config,
                infer_unspecified=infer_unspecified,
                path_prefix=path_prefix,
                model_id=model_id,
            )

        # `match_threshold` is the object-level threshold, read by the branch
        # above and by nothing on a primitive. It stays in the accepted-key set
        # because a `structured_model` field carries it legitimately and
        # `to_stickler_config()` exports it there, so removing it would break the
        # round-trip -- but on a leaf it was accepted and dropped, which is the
        # `threshold`/`match_threshold` confusion the union set cannot catch. The
        # JSON Schema path already refuses the same misplacement
        # ("x-aws-stickler-match-threshold is not read on field 'amount'. It
        # belongs on the object").
        #
        # 1.0.1 warns and ignores the key, as 1.0.0 ignored it, because 1.0.1
        # ships no breaking change. 1.1.0 turns this into a `ValueError`, which
        # is what makes the two front doors agree.
        if "match_threshold" in field_config:
            # Names the model: Python's default filter shows a message once per
            # text and line, so two models built on one line with the same
            # field name would otherwise show one warning between them.
            of_model = f" of model {model_id!r}" if model_id else ""
            warn_once(
                "field-config-match-threshold-on-primitive",
                # Keyed on (root model, dotted path), as the unknown-key warning
                # is, so a second model with the same misplacement still warns.
                f"{model_id}|{field_path}",
                f"'match_threshold' is not read on primitive field "
                f"'{field_path}'{of_model}; "
                f"it is the object-level threshold, so it is "
                f"ignored and the field scores at its 'threshold'. Use "
                f"'threshold' for this field, or set 'match_threshold' on the "
                f"model or on a 'structured_model' field. This becomes a "
                f"ValueError in 1.1.0.",
                category=UserWarning,
                stacklevel=_MATCH_THRESHOLD_STACKLEVEL,
            )

        # `infer_unspecified_fields` scopes a SUBTREE, so it is meaningful only on
        # a nested model, which the branch above handles. On a primitive field
        # there is no subtree and nothing reads it, so it was accepted and dropped
        # in silence -- a config declaring it on a leaf looked configured and was
        # not. The JSON Schema path already refuses the same misplacement with a
        # precise message ("is not read on field 'f'. It belongs on the object"),
        # so refusing here is what makes the two front doors agree. See #210 and
        # #312 for why a dropped key is not an acceptable outcome.
        if "infer_unspecified_fields" in field_config:
            raise ValueError(
                f"'infer_unspecified_fields' is not read on primitive field "
                f"'{field_path}'; it scopes a nested model's own subtree and has "
                f"no effect here. Set it on the model, or on a "
                f"'structured_model' field to scope that subtree. To infer this "
                f'one field, set "comparator": "auto" on it.'
            )

        # Handle primitive fields (existing logic)
        # Resolve the type
        try:
            field_type = resolve_type_string(type_string)
        except ValueError as e:
            raise ValueError(f"Invalid type for field '{field_name}': {e}")

        # Resolve each parameter independently, so a partly-configured field keeps
        # what it wrote and infers only the rest. Treating any config as "fully
        # configured" left `{"type": "float", "threshold": 0.99}` on the type-blind
        # Levenshtein default -- a threshold tuned against edit distance over
        # "1000.00" and "1000.0".
        inferred = (
            _infer_spec(field_name, field_type, match_threshold)
            if _wants_inference(field_config, infer_unspecified)
            else None
        )
        provenance = list(inferred.provenance) if inferred else []

        declared_comparator = field_config.get("comparator")
        if declared_comparator == AUTO_COMPARATOR:
            declared_comparator = None
        if declared_comparator is None and inferred is not None:
            comparator_name = inferred.comparator_name
            # MERGED, not replaced. A `comparator_config` beside an inferred
            # comparator is a tweak to the comparator inference chose, so
            # replacing the whole dict silently dropped the rest of it:
            # NumericComparator's inferred `relative_tolerance` of 0.001 fell to
            # 0.0, turning a tolerance-based numeric comparison into an exact one
            # for anyone who set any other key.
            comparator_config = {
                **inferred.comparator_config,
                **normalize_comparator_config(
                    field_config.get("comparator_config"), f"field '{field_name}'"
                ),
            }
        else:
            comparator_name = declared_comparator or "LevenshteinComparator"
            comparator_config = normalize_comparator_config(
                field_config.get("comparator_config"), f"field '{field_name}'"
            )

        # Create comparator instance
        try:
            comparator = create_comparator(comparator_name, comparator_config)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid comparator for field '{field_name}': {e}")

        # `None` for what the config does not state, so `ComparableField` resolves
        # it exactly as it does on the Python path. Defaulting to 0.5 and True here
        # marked both as stated, which discarded a threshold named in
        # `comparator_config` (#334) and suppressed the object-grade clip default,
        # zeroing a dict field's partial credit (#348). Precedence is field, then
        # comparator, then inference, so a comparator threshold outranks an
        # inferred one: the author wrote it.
        #
        # A comparator threshold the field adopts is checked as the field's own
        # `threshold` is, and raises on load. A field `threshold` wins, so the
        # comparator's is then not read and not checked.
        threshold = field_config.get("threshold")
        named = _named_comparator_threshold(comparator)
        if threshold is None and named is not None:
            if not is_valid_threshold(named):
                written = field_config.get("comparator_config")
                _raise_comparator_threshold(
                    named,
                    written if isinstance(written, Mapping) else {},
                    spelling=CONFIG,
                    comparator_name=comparator_name,
                    comparator_key="comparator",
                    config_key="comparator_config",
                    named=declared_comparator is not None,
                    field=field_path,
                    model=_model_label(model_id),
                )
        adopted = threshold is None and named is not None
        threshold_defaulted = threshold is None and not adopted and inferred is not None
        if threshold_defaulted:
            threshold = inferred.threshold
        weight = field_config.get("weight", inferred.weight if inferred else 1.0)
        clip_under_threshold = field_config.get("clip_under_threshold")
        clip_inferred = clip_under_threshold is None and inferred is not None
        if clip_inferred:
            clip_under_threshold = inferred.clip_under_threshold

        # Name the parameters the author took back, so the trail cannot contradict
        # the row it sits under. Inference fills only what the config left unnamed,
        # so a PARTLY configured field reported the inferred numbers while
        # `explain()` showed the author's -- a field with `threshold: 0.99` carried
        # a trail saying `@0.95` twice and no record of the override at all. The
        # docs recommend `explain()` for spotting a misspelled key, which is exactly
        # the read that obscured.
        if inferred is not None:
            overridden = [
                f"{key}={field_config[key]!r}"
                for key in ("threshold", "weight", "clip_under_threshold")
                if key in field_config
            ]
            # A threshold adopted from `comparator_config` replaces the inferred
            # one too, so it is named the same way. Without it the row reported
            # 0.9 while the trail still quoted `@0.7`.
            if adopted and "threshold" in comparator_config:
                overridden.append(f"comparator_config.threshold={named!r}")
            if overridden:
                provenance.append(f"config: {', '.join(overridden)} (overrides above)")

        # Extract Pydantic field parameters
        default = field_config.get("default", ...)  # Use Ellipsis for required fields
        required = field_config.get("required", False)
        description = field_config.get("description")
        alias = field_config.get("alias")
        examples = field_config.get("examples")

        # Handle required vs default logic
        if required and default != ...:
            # If explicitly required, ignore default
            default = ...
        elif not required and default == ...:
            # If not required and no default specified, use None
            default = None

        # Widen the annotation to Optional[...] whenever the resolved default is
        # None, so the None default is a valid value. Mirrors the JSON-Schema path
        # (json_schema_field_converter.py) — issue #149: the rich-value path
        # round-trips through from_json(...).model_dump(), which materializes the
        # None default and re-validates it against the annotation. Without this,
        # a schema -> to_stickler_config -> model_from_json round-trip rebuilds a
        # broken model.
        if default is None:
            field_type = Optional[field_type]

        # Create ComparableField
        comparable_field = ComparableField(
            comparator=comparator,
            threshold=threshold,
            weight=weight,
            default=default,
            clip_under_threshold=clip_under_threshold,
            alias=alias,
            description=description,
            examples=examples,
        )
        _mark_unstated(
            comparable_field, threshold=threshold_defaulted, clip=clip_inferred
        )

        # Record what was inferred, so `explain()` reports the reason and a
        # `source` naming what drove the choice (`type`, `name-token`) rather than
        # claiming every config-driven field was explicitly configured. Written onto the closure `ComparableField` just
        # returned, which no caller has a reference to yet -- the shared-field
        # hazard `structured_model.py` warns about needs sharing.
        if provenance:
            extra_callable = comparable_field.json_schema_extra
            if callable(extra_callable):
                extra_callable._inferred_provenance = tuple(provenance)

        return field_type, comparable_field

    def _convert_nested_model_field(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        infer_unspecified: bool = False,
        path_prefix: str = "",
        model_id: str = "",
    ) -> Tuple[Type, Any]:
        """Convert a nested structured model field configuration.

        Args:
            field_name: Name of the field
            field_config: JSON configuration for the nested field
            infer_unspecified: The enclosing model's flag, used unless this field
                sets ``infer_unspecified_fields`` itself.
            path_prefix: Dotted path of the enclosing field, extended by this
                field's own name before the nested model is built.
            model_id: ``model_identity`` of the root model, passed through
                unchanged so the whole tree shares one warn-once namespace.

        Returns:
            Tuple of (field_type, pydantic_field)

        Raises:
            ValueError: If configuration is invalid
        """
        type_string = field_config["type"]
        nested_fields_config = field_config["fields"]

        # Recursively create the nested model class
        from .structured_model import StructuredModel

        # Create nested model configuration
        nested_config = {
            "model_name": f"{field_name.title()}Model",
            "fields": nested_fields_config,
            "match_threshold": field_config.get("match_threshold", 0.7),
            # Forwarded, because the nested model is built by re-entering
            # `model_from_json` with this synthesised config. Omitting it meant a
            # model-level flag stopped at the first nesting level, so a nested
            # field silently kept the type-blind default while its siblings one
            # level up were inferred. Read from this field first, so a nested
            # field can scope its own subtree the way a nested object does on the
            # JSON Schema path.
            "infer_unspecified_fields": _nested_infer_flag(
                field_config, infer_unspecified, field_name
            ),
        }

        # Create the nested model class. Built through ModelFactory rather than
        # through `StructuredModel.model_from_json`, which is the same call with
        # `base_class=StructuredModel`, so that the field path and the root
        # model's identity reach the nested model's own validation. Without
        # them the nested warning names `city` rather than `billing.city`, and
        # two fields called `city` in one config share a warn-once slot.
        from .model_factory import ModelFactory

        # `items[].sku` for a field inside a list's element, as the JSON Schema
        # importer writes it, so both importers name a nested field alike.
        element = "[]" if type_string == "list_structured_model" else ""
        NestedModelClass = ModelFactory.create_model_from_json(
            nested_config,
            base_class=StructuredModel,
            path_prefix=f"{path_prefix}{field_name}{element}.",
            model_id=model_id,
        )

        # Determine the field type based on the type string
        if type_string == "structured_model":
            field_type = NestedModelClass
        elif type_string == "list_structured_model":
            field_type = List[NestedModelClass]
        elif type_string == "optional_structured_model":
            field_type = Optional[NestedModelClass]
        else:
            raise ValueError(f"Unknown structured model type: {type_string}")

        # Extract Pydantic field parameters
        default = field_config.get("default", ...)
        required = field_config.get("required", False)
        description = field_config.get("description")
        alias = field_config.get("alias")
        examples = field_config.get("examples")

        # Handle required vs default logic
        if required and default != ...:
            default = ...
        elif not required and default == ...:
            default = None

        # Widen a list of models to Optional when its default is None, as the
        # primitive path above does (#149). Otherwise an exported
        # `Optional[List[Model]]` rebuilt as `List[Model]` and rejected the
        # None the original accepts, which scores the same as [].
        if type_string == "list_structured_model" and default is None:
            field_type = Optional[field_type]

        # CRITICAL FIX: Create ComparableField for nested models to enable proper comparison
        # Extract threshold and weight from field configuration
        weight = field_config.get("weight", 1.0)  # Default weight
        # `None` when unstated, for the reason given in `convert_field_config`.
        clip_under_threshold = field_config.get("clip_under_threshold")

        is_list = type_string == "list_structured_model"
        field_path = f"{path_prefix}{field_name}"

        # Comparator keys resolve as they do on the Python path; see
        # `_nested_model_comparator_threshold`. A named comparator is never
        # called for scoring, and is warned about, but it is installed as on the
        # Python path: with the placeholder in its place, `explain()` reported a
        # LevenshteinComparator the author never wrote as their explicit choice.
        # `"auto"` and a comparator on a list of models are refused in
        # `validate_nested_field_schema` before this runs.
        from stickler.comparators.levenshtein import LevenshteinComparator

        ignored = []
        named_comparator = None
        if field_config.get("comparator") is not None:
            ignored.append("'comparator'")
            try:
                named_comparator = create_comparator(
                    field_config["comparator"],
                    normalize_comparator_config(
                        field_config.get("comparator_config"),
                        f"field '{field_name}'",
                    ),
                )
            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid comparator for field '{field_name}': {e}")
        raw_config = field_config.get("comparator_config")
        comparator_threshold = rejected = None
        if raw_config is not None and not isinstance(raw_config, Mapping):
            ignored.append("'comparator_config'")
        # Asked whenever a comparator is named, with or without a config: a
        # comparator whose constructor sets its own threshold gates the field on
        # the Python path and on the schema path, so asking only when a config
        # was present gated this path at 0.7 instead.
        if named_comparator is not None or isinstance(raw_config, Mapping):
            comparator_threshold, ignored_keys, rejected = (
                _nested_model_comparator_threshold(
                    named_comparator,
                    raw_config if isinstance(raw_config, Mapping) else {},
                    reads_threshold=not is_list,
                )
            )
            if ignored_keys:
                ignored.append(_config_keys_phrase("comparator_config", ignored_keys))
        stated = "threshold" in field_config
        if rejected is not None and not stated:
            _raise_comparator_threshold(
                rejected,
                raw_config if isinstance(raw_config, Mapping) else {},
                spelling=CONFIG,
                comparator_name=field_config.get("comparator")
                or "LevenshteinComparator",
                comparator_key="comparator",
                config_key="comparator_config",
                named=named_comparator is not None,
                field=field_path,
                model=_model_label(model_id),
                nested=True,
            )
        if ignored:
            warn_once(
                "nested-model-comparator-keys",
                f"{model_id}|{field_path}|{'|'.join(ignored)}",
                _nested_model_keys_warning(
                    field_path,
                    ignored,
                    shape=f"a {type_string} field"
                    if type_string != "optional_structured_model"
                    else f"an {type_string} field",
                    is_list=is_list,
                    members="fields",
                    threshold_key="'match_threshold' on this field"
                    if is_list
                    else "'threshold'",
                    model=_model_label(model_id),
                    spelling=CONFIG,
                    adopted=comparator_threshold
                    if named_comparator is not None and not stated
                    else None,
                ),
                category=UserWarning,
                stacklevel=_stacklevel_outside_stickler(through_warn_once=True),
            )

        # Unused for scoring; ComparableField requires one. Carries an adopted
        # comparator threshold, so `ComparableField` resolves it exactly as for
        # `ComparableField(comparator=X(threshold=t))`: field threshold first.
        if named_comparator is not None:
            placeholder = named_comparator
        elif comparator_threshold is not None:
            placeholder = LevenshteinComparator(threshold=comparator_threshold)
        else:
            placeholder = LevenshteinComparator()
        if is_list:
            # `None`, not the legacy 0.5: passing a value marks the field as
            # having named a threshold, and `StructuredModel.__init_subclass__`
            # refuses one on a list-of-model field. Hungarian matching uses the
            # element model's `match_threshold`.
            threshold = None
        elif "threshold" in field_config:
            threshold = field_config["threshold"]
        elif comparator_threshold is not None:
            threshold = None
        else:
            threshold = 0.7  # this path's default for a single nested model

        comparable_field = ComparableField(
            comparator=placeholder,
            threshold=threshold,
            weight=weight,
            default=default,
            clip_under_threshold=clip_under_threshold,
            alias=alias,
            description=description,
            examples=examples,
        )
        # The 0.7 above is this path's default, not the author's.
        _mark_unstated(comparable_field, threshold="threshold" not in field_config)

        return field_type, comparable_field

    def convert_fields_config(
        self,
        fields_config: Dict[str, Dict[str, Any]],
        *,
        infer_unspecified: bool = False,
        match_threshold: Optional[float] = None,
        path_prefix: str = "",
        model_id: str = "",
    ) -> Dict[str, Tuple[Type, Field]]:
        """Convert multiple field configurations.

        Args:
            fields_config: Dictionary of field configurations
            infer_unspecified: Model-level inference flag.
            match_threshold: The model's ``match_threshold``, forwarded to
                inference for mapping fields.
            path_prefix: Dotted path of the enclosing field, forwarded so a
                nested model's own warnings name the full path.
            model_id: ``model_identity`` of the root model, carried unchanged
                through nesting.

        Returns:
            Dictionary mapping field names to (type, field) tuples

        Raises:
            ValueError: If any field configuration is invalid
        """
        field_definitions = {}

        for field_name, field_config in fields_config.items():
            try:
                field_type, pydantic_field = self.convert_field_config(
                    field_name,
                    field_config,
                    infer_unspecified=infer_unspecified,
                    match_threshold=match_threshold,
                    path_prefix=path_prefix,
                    model_id=model_id,
                )
                field_definitions[field_name] = (field_type, pydantic_field)
            except ValueError as e:
                if is_threshold_error(e):
                    # Already names the field by its full path, and the model.
                    raise
                raise ValueError(f"Error processing field '{field_name}': {e}")

        return field_definitions

    def validate_field_config(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        field_path: Optional[str] = None,
        model_id: str = "",
    ) -> None:
        """Validate a field configuration without converting it.

        Args:
            field_name: Name of the field
            field_config: JSON configuration for the field
            field_path: Dotted path to this field from the root model, so an
                unknown key on a nested field reports ``billing.city`` rather
                than ``city``. Defaults to ``field_name``.
            model_id: ``model_identity`` of the ROOT model, carried unchanged
                through nesting. Part of the warn-once key, so that two models
                sharing a field name and a typo each warn.

        Raises:
            ValueError: If configuration is invalid
        """
        path = field_path or field_name

        # Refused up front, the way ``ModelFactory.validate_config`` refuses a
        # non-dict config. Two reasons. ``set()`` of a str iterates its
        # characters, so ``{"name": "str"}`` reported the field as not accepting
        # 'r', 's', 't'. And the ``"type" not in field_config`` check below is a
        # containment test, which reaches an undocumented ``TypeError`` rather
        # than the ``ValueError`` this function documents: ``5`` and ``None``
        # give "argument of type 'int' is not iterable", and a list or set
        # containing the string "type" gets past the check and dies later on
        # subscripting.
        if not isinstance(field_config, Mapping):
            raise ValueError(
                f"Field '{path}' configuration must be a mapping, got "
                f"{type(field_config).__name__}"
            )

        # An unrecognized key is silently ignored by every reader below, so a
        # misspelled 'threshhold' builds at the fallback 0.5 and every score is
        # wrong with no signal. Report it the way an unknown comparator_config
        # key is already reported, and keep building: the key was never applied,
        # so refusing the whole model would be a new failure rather than a fix.
        unknown = set(field_config) - ACCEPTED_FIELD_CONFIG_KEYS
        if unknown:
            warn_once(
                "field-config-unknown-keys",
                # Keyed on (root model, dotted path, keys). Keyed on the leaf
                # name alone, the first model in a process consumed the slot
                # for every later one, so a second config carrying the same
                # typo scored at the default in silence -- the very defect
                # this warning exists to end, reproduced through its own fix.
                # ``key=str`` because a non-string key must not turn this into
                # a TypeError, against a documented ``Raises: ValueError``.
                f"{model_id}|{path}|"
                f"{','.join(str(k) for k in sorted(unknown, key=str))}",
                f"Field '{path}' does not accept "
                f"{', '.join(repr(k) for k in sorted(unknown, key=str))} in "
                f"its config; ignored. It accepts "
                f"{', '.join(sorted(ACCEPTED_FIELD_CONFIG_KEYS))}. "
                f"A misspelled key leaves the field at its default, so the "
                f"value you wrote is not the value being scored.",
                category=UserWarning,
                stacklevel=_UNKNOWN_KEY_STACKLEVEL,
            )

        # Check required parameters
        if "type" not in field_config:
            raise ValueError(f"Field '{field_name}' missing required 'type' parameter")

        # Validate type string
        type_string = field_config["type"]
        try:
            resolve_type_string(type_string)
        except ValueError as e:
            raise ValueError(f"Invalid type for field '{field_name}': {e}")

        # Validate comparator if specified. `"auto"` is a request to infer, not a
        # comparator name, so it is not resolved through the registry -- doing so
        # reported it as an unknown comparator and listed the built-ins.
        if (
            "comparator" in field_config
            and field_config["comparator"] != AUTO_COMPARATOR
        ):
            comparator_name = field_config["comparator"]
            # Normalised here too, not only on the `"auto"` path. `create_instance`
            # does `config.items()`, which raises `AttributeError` for a non-mapping
            # -- and `AttributeError` is in neither the `except` below nor the one in
            # `convert_fields_config`, so it escaped `model_from_json` raw, against a
            # docstring that documents `Raises: ValueError`. Only the inferred branch
            # was guarded, so the clean message went to the new syntax and the crash
            # to every pre-existing config that names a comparator.
            comparator_config = normalize_comparator_config(
                field_config.get("comparator_config"), f"field '{field_name}'"
            )
            try:
                create_comparator(comparator_name, comparator_config)
            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid comparator for field '{field_name}': {e}")

        # Validate numeric parameters
        if "threshold" in field_config and not is_valid_threshold(
            field_config["threshold"]
        ):
            raise threshold_error(
                field_config["threshold"],
                spelling=CONFIG,
                field=path,
                model=_model_label(model_id),
                list_element="" if type_string == "list_structured_model" else None,
            )
        if "weight" in field_config:
            value = field_config["weight"]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Field '{field_name}' parameter 'weight' must be numeric, got {type(value)}"
                )
            if value < 0:
                raise ValueError(
                    f"Field '{field_name}' weight must be non-negative, got {value}"
                )

        # Validate boolean parameters
        boolean_params = ["required", "clip_under_threshold"]
        for param in boolean_params:
            if param in field_config:
                value = field_config[param]
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Field '{field_name}' parameter '{param}' must be boolean, got {type(value)}"
                    )

    def validate_fields_config(
        self,
        fields_config: Dict[str, Dict[str, Any]],
        *,
        path_prefix: str = "",
        model_id: str = "",
    ) -> None:
        """Validate multiple field configurations.

        Args:
            fields_config: Dictionary of field configurations
            path_prefix: Dotted path of the enclosing field, ``""`` at the root
                and ``"billing."`` one level down.
            model_id: ``model_identity`` of the root model, carried unchanged
                through nesting.

        Raises:
            ValueError: If any field configuration is invalid
        """
        for field_name, field_config in fields_config.items():
            self.validate_field_config(
                field_name,
                field_config,
                field_path=f"{path_prefix}{field_name}",
                model_id=model_id,
            )

    def validate_nested_field_schema(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        *,
        infer_unspecified: bool = False,
    ) -> None:
        """Validate schema for nested structured model fields.

        Args:
            field_name: Name of the field
            field_config: JSON configuration for the field

        Raises:
            ValueError: If schema is invalid for nested models
        """
        field_type = field_config.get("type", "")

        # Check if this is a structured model type
        if field_type in [
            "structured_model",
            "list_structured_model",
            "optional_structured_model",
        ]:
            # `"auto"` asks for inference, and a nested model has no comparator
            # to infer. Refused as the JSON Schema path refuses it on an object.
            if field_config.get("comparator") == AUTO_COMPARATOR:
                raise ValueError(
                    f"'comparator': '{AUTO_COMPARATOR}' cannot be applied to field "
                    f"'{field_name}': it is a '{field_type}', scored field by "
                    "field, so there is no comparator to infer. Set "
                    "'infer_unspecified_fields': true on it to infer the fields "
                    "inside it instead."
                )

            # A named comparator on a single nested model is accepted and
            # ignored, as on the Python path; `_convert_nested_model_field`
            # warns. On a list of models the Python path refuses any comparator
            # but the default LevenshteinComparator, and this path has always
            # refused every one, so it still does.
            if field_type == "list_structured_model" and "comparator" in field_config:
                raise ValueError(
                    f"Field '{field_name}' with type '{field_type}' cannot have a 'comparator'. "
                    "Structured models use recursive comparison, not primitive comparators."
                )

            # Structured model fields must have 'fields'
            if "fields" not in field_config:
                raise ValueError(
                    f"Field '{field_name}' with type '{field_type}' requires a 'fields' configuration "
                    "defining the nested model structure."
                )

            # Recursively validate nested fields
            nested_fields = field_config["fields"]
            if not isinstance(nested_fields, dict):
                raise ValueError(
                    f"Field '{field_name}' 'fields' must be a dictionary, got {type(nested_fields)}"
                )

            # Validate each nested field under the flag that will actually govern
            # the subtree. Using the parent's value made a nested
            # `"infer_unspecified_fields": true` produce an error telling the
            # author to set exactly what they had just set.
            nested_infer = _nested_infer_flag(
                field_config, infer_unspecified, field_name
            )
            # `items[].sku` inside a list's element, as the conversion names it.
            element = "[]" if field_type == "list_structured_model" else ""
            for nested_field_name, nested_field_config in nested_fields.items():
                self.validate_nested_field_schema(
                    f"{field_name}{element}.{nested_field_name}",
                    nested_field_config,
                    infer_unspecified=nested_infer,
                )

        else:
            # Primitive fields cannot have 'fields'
            if "fields" in field_config:
                raise ValueError(
                    f"Field '{field_name}' with primitive type '{field_type}' cannot have 'fields'. "
                    "Only structured_model types can have nested fields."
                )

            # Primitive fields must have comparators, unless inference was asked
            # for -- at which point omission is the point, not an oversight.
            if "comparator" not in field_config and not infer_unspecified:
                raise ValueError(
                    f"Field '{field_name}' with primitive type '{field_type}' requires a 'comparator'. "
                    "Primitive fields need comparators to define how they should be compared. "
                    "Set 'infer_unspecified_fields': true on the model, or "
                    f"'comparator': '{AUTO_COMPARATOR}' on this field, to have it "
                    "chosen from the field's type and name."
                )


# Global converter instance
_global_converter = FieldConverter()


def get_global_converter() -> FieldConverter:
    """Get the global field converter instance.

    Returns:
        Global FieldConverter instance
    """
    return _global_converter


def convert_field_config(
    field_name: str,
    field_config: Dict[str, Any],
    *,
    infer_unspecified: bool = False,
    match_threshold: Optional[float] = None,
) -> Tuple[Type, Field]:
    """Convert a field configuration using the global converter.

    Args:
        field_name: Name of the field
        field_config: JSON configuration for the field
        infer_unspecified: Model-level inference flag.
        match_threshold: The model's ``match_threshold``, forwarded to inference.

    Returns:
        Tuple of (field_type, pydantic_field)
    """
    return _global_converter.convert_field_config(
        field_name,
        field_config,
        infer_unspecified=infer_unspecified,
        match_threshold=match_threshold,
    )


def convert_fields_config(
    fields_config: Dict[str, Dict[str, Any]],
    *,
    infer_unspecified: bool = False,
    match_threshold: Optional[float] = None,
    path_prefix: str = "",
    model_id: str = "",
) -> Dict[str, Tuple[Type, Field]]:
    """Convert multiple field configurations using the global converter.

    Args:
        fields_config: Dictionary of field configurations
        infer_unspecified: Model-level inference flag.
        match_threshold: The model's ``match_threshold``, forwarded to inference.
        path_prefix: Dotted path of the enclosing field, forwarded so a nested
            model's own unknown-key warnings name the full path.
        model_id: ``model_identity`` of the root model, carried unchanged
            through nesting.

    Returns:
        Dictionary mapping field names to (type, field) tuples
    """
    return _global_converter.convert_fields_config(
        fields_config,
        infer_unspecified=infer_unspecified,
        match_threshold=match_threshold,
        path_prefix=path_prefix,
        model_id=model_id,
    )


def validate_field_config(field_name: str, field_config: Dict[str, Any]) -> None:
    """Validate a field configuration using the global converter.

    Args:
        field_name: Name of the field
        field_config: JSON configuration for the field

    Raises:
        ValueError: If configuration is invalid
    """
    _global_converter.validate_field_config(field_name, field_config)


def validate_fields_config(
    fields_config: Dict[str, Dict[str, Any]],
    *,
    path_prefix: str = "",
    model_id: str = "",
) -> None:
    """Validate multiple field configurations using the global converter.

    Args:
        fields_config: Dictionary of field configurations
        path_prefix: Dotted path of the enclosing field, ``""`` at the root.
        model_id: ``model_identity`` of the root model.

    Raises:
        ValueError: If any field configuration is invalid
    """
    _global_converter.validate_fields_config(
        fields_config, path_prefix=path_prefix, model_id=model_id
    )
