"""Configuration helper for StructuredModel field and schema operations.

This module provides utilities for handling field configuration, type checking,
JSON processing, and schema generation for StructuredModel instances.
"""

import inspect
from collections.abc import Mapping as abc_Mapping
from typing import TYPE_CHECKING, Any, Dict, List, get_args, get_origin

from pydantic import BaseModel

from stickler.comparators.anls import ANLSStarComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.utils.deprecation import warn_once

from .comparable_field import _LEGACY_DEFAULT_THRESHOLD
from .optional_annotation import (
    is_union,
    union_args,
    unwrap_annotated,
    unwrap_optional,
)

if TYPE_CHECKING:
    from stickler.structured_object_evaluator.models.comparison_info import (
        ComparableFieldConfig,
    )
from stickler.comparators.structured import StructuredModelComparator

# Comparators that must not be handed a mapping, by class name so an out-of-tree
# comparator is never caught by it. Both entries are here on measured evidence:
# Levenshtein raises, and Fuzzy ranks a changed value above a reordering.
#
# Deliberately a denylist. An allowlist would silently zero any mapping-capable
# comparator written outside this repo, since it could not know to opt in.
_COMPARATORS_THAT_CANNOT_SCORE_MAPPINGS = frozenset(
    {"LevenshteinComparator", "FuzzyComparator"}
)


class ConfigurationHelper:
    """Helper class for StructuredModel configuration and schema operations."""

    @staticmethod
    def from_json(cls, json_data: Dict[str, Any]):
        """Create a StructuredModel instance from JSON data.

        This method handles missing fields gracefully and stores extra fields
        in the extra_fields attribute.

        Args:
            cls: StructuredModel class
            json_data: Dictionary containing the JSON data

        Returns:
            StructuredModel instance created from the JSON data
        """
        # Make a copy of the input data
        data_copy = json_data.copy()

        # Extract field names defined in the model
        model_fields = set(cls.model_fields.keys())

        # Remove 'extra_fields' from consideration if it exists in the model
        if "extra_fields" in model_fields:
            model_fields.remove("extra_fields")

        # Find extra fields (those in json_data but not in model_fields)
        extra_field_names = set(data_copy.keys()) - model_fields

        # Extract extra fields into a separate dictionary
        extra_fields = {k: data_copy[k] for k in extra_field_names}

        # Since ComparableField is now always a function, we don't need special handling
        # for missing fields - Pydantic will handle them with the field's default value
        pass

        # CRITICAL FIX: Recursively handle nested StructuredModel objects
        # For each field that exists in the data and is a StructuredModel, process it recursively
        for field_name in model_fields:
            if field_name in data_copy:
                field_info = cls.model_fields.get(field_name)
                if field_info:
                    # Check if this field is a StructuredModel type
                    annotation = field_info.annotation

                    # Handle direct StructuredModel annotations
                    if ConfigurationHelper._is_structured_model_class(annotation):
                        # Recursively process the nested object
                        nested_data = data_copy[field_name]
                        if isinstance(nested_data, dict):
                            data_copy[field_name] = (
                                ConfigurationHelper._process_nested_structured_data(
                                    annotation, nested_data
                                )
                            )

                    # Handle Optional[StructuredModel] annotations
                    elif ConfigurationHelper._is_optional_structured_model(annotation):
                        nested_data = data_copy[field_name]
                        if isinstance(nested_data, dict):
                            # Extract the StructuredModel class from Optional[StructuredModel]
                            structured_class = ConfigurationHelper._extract_structured_class_from_optional(
                                annotation
                            )
                            if structured_class:
                                data_copy[field_name] = (
                                    ConfigurationHelper._process_nested_structured_data(
                                        structured_class, nested_data
                                    )
                                )

                    # Handle List[StructuredModel] and Optional[List[StructuredModel]] annotations
                    elif ConfigurationHelper._is_list_structured_model(annotation):
                        nested_data = data_copy[field_name]
                        if isinstance(nested_data, list):
                            # Extract the StructuredModel class from the list type
                            structured_class = (
                                ConfigurationHelper._extract_structured_class_from_list(
                                    annotation
                                )
                            )
                            if structured_class:
                                # Process each item in the list
                                processed_items = []
                                for item_data in nested_data:
                                    if isinstance(item_data, dict):
                                        processed_item = ConfigurationHelper._process_nested_structured_data(
                                            structured_class, item_data
                                        )
                                        processed_items.append(processed_item)
                                    else:
                                        # Non-dict items are kept as-is
                                        processed_items.append(item_data)
                                data_copy[field_name] = processed_items

        # Create the model instance
        instance = cls.model_validate(data_copy)

        # Store extra fields
        instance.extra_fields = extra_fields

        return instance

    @staticmethod
    def can_score_object(
        model_cls, field_name: str, comparator, *, shape: str, warn: bool = True
    ) -> bool:
        """Whether ``comparator`` can score a whole object, warning once if not.

        ``warn=False`` returns the same verdict silently. A list element is asked
        this question once per cell of the Hungarian cost matrix, and the matcher
        discards all but one cell per row, so warning while probing described an
        outcome that did not happen -- and, because ``warn_once`` spends its one
        message per field for the life of the process, spent it on a pair nothing
        was decided from. ``_ClassGatedComparator`` probes silently and replays
        the gate on the pairs the matcher actually selected.

        Half of :meth:`can_compare_object_pair`, which is what callers use; see
        there for why the halves are composed in one place.

        ``shape`` is ``"mapping"`` or ``"model"``. The two differ only in the
        wording of the warning and in the advice it gives, because the remedy is
        different: a mapping wants a ``Dict[...]`` annotation, a plain pydantic
        model wants the model named as the annotation.

        Both arrive here for the same reason. A field annotated ``Any``,
        ``object``, or a multi-arm ``Union`` declares nothing that could install
        a structural comparator, so it keeps the primitive default of
        ``LevenshteinComparator``, which cannot score an object -- and for a
        pydantic model does something worse than raising. Edit distance over
        ``str(model)`` compares the field-name boilerplate that is identical on
        both sides, so it never scores low:

            LineItem(quantity=2, unit_price=10.5, currency='USD')
              vs LineItem(quantity=9, unit_price=99.9, currency='EUR')  ->  0.8293

        which clears the default threshold and reports every value being wrong
        as a TRUE POSITIVE. Refusing is strictly better than a number that
        confident and that wrong.

        A DENYLIST, not an allowlist. Only the comparators known to be wrong on
        an object are refused; everything else is trusted.

        An allowlist keyed on an opt-in attribute looked safer and was worse: the
        attribute is new, so no comparator outside this repo can carry it, and a
        user who wrote a mapping comparator and asked for it BY NAME had their
        score silently replaced with 0.0. An explicit ``comparator=`` is consent
        by definition, and the gate was overriding it -- reproducing the very
        symptom of #297 for a different population.

        The two refused comparators are refused on evidence, not by category:

            LevenshteinComparator  raises TypeError on a dict
            FuzzyComparator        scores a CHANGED value (0.944) above a mere
                                   key reordering (0.667), so its ordering is
                                   not defensible as a metric

        Callers report a false discovery rather than raising: the shape of a value
        can be data-dependent, so raising ends a corpus run on document N after
        succeeding on N-1, and no test would catch it. The warning carries the
        same information without stopping.
        """
        if comparator.__class__.__name__ not in _COMPARATORS_THAT_CANNOT_SCORE_MAPPINGS:
            return True
        if not warn:
            return False
        if shape == "mapping":
            noun, plural, advice = (
                "a mapping",
                "mappings",
                "Annotate the field as a mapping (Dict[...] or Mapping[...]) to "
                "get ANLSStarComparator automatically, declare "
                "ComparableField(comparator=ANLSStarComparator()) explicitly, or "
                "use a nested StructuredModel if you know the keys.",
            )
        else:
            noun, plural, advice = (
                "a pydantic model",
                "models",
                "Annotate the field with the model class instead of Any or object "
                "to get ANLSStarComparator automatically, declare "
                "ComparableField(comparator=ANLSStarComparator()) explicitly, or "
                "make it a nested StructuredModel for per-field detail.",
            )
        warn_once(
            "dict-value-uncomparable"
            if shape == "mapping"
            else "model-value-uncomparable",
            f"{getattr(model_cls, '__name__', model_cls)}.{field_name}",
            f"Field '{field_name}' holds {noun}, but its comparator "
            f"({comparator.__class__.__name__}) scores scalars, so the pair is "
            f"counted as a false discovery even if the two {plural} are "
            f"identical. {advice}",
            category=UserWarning,
        )
        return False

    @staticmethod
    def is_object_grade_annotation(field_info) -> bool:
        """Whether a field is judged as one object rather than as a scalar.

        True for a mapping, a plain pydantic model, and a list of either. All
        four get `ANLSStarComparator`, the class's `match_threshold`, and
        `clip_under_threshold=False`.

        Reads only the annotation, so it is safe to ask before a class finishes
        being defined. `StructuredModel._install_object_grade_comparators` calls
        THIS rather than the memoised wrapper below for exactly that reason: an
        annotation that has not resolved yet answers False, and caching that
        False at class-definition time would outlive the annotation becoming
        readable.
        """
        return (
            ConfigurationHelper.is_dict_field_type(field_info)
            or ConfigurationHelper._is_list_of_mappings(field_info)
            or ConfigurationHelper.is_plain_model_field_type(field_info)
            or ConfigurationHelper._is_list_of_plain_models(field_info)
        )

    @staticmethod
    def object_grade_clip(extra) -> bool:
        """The ``clip_under_threshold`` an object-grade field should carry.

        An object is a container: a partly-correct one keeps its partial score
        rather than being zeroed by the field threshold, the same policy nested
        objects and lists use. So the default is off -- but only as a DEFAULT. An
        explicit ``clip_under_threshold=True`` is a decision, and overwriting it
        discarded it in silence.

        One function because two paths set this value and they must not disagree:
        ``StructuredModel._install_object_grade_comparators`` writes it into the
        ``FieldInfo`` at class-definition time, where the exported schema can see
        it, and :meth:`get_comparison_info` applies it at read time for a field
        that path could not reach. When each spelled the rule itself, a declared
        ``True`` was honoured on a dict field and dropped on a plain-model one.
        """
        if getattr(extra, "_clip_explicit", False):
            return bool(getattr(extra, "_clip_under_threshold", True))
        return False

    @staticmethod
    def _wants_object_grade_comparison(cls, field_name: str, field_info) -> bool:
        """:meth:`is_object_grade_annotation`, memoised per (class, field).

        The annotation is fixed once the class is defined, and
        `get_comparison_info` runs once per field per pairwise comparison --
        every cell of a Hungarian cost matrix, so 60x60 objects of 20 fields is
        72,000 calls -- while each predicate destructures the annotation again.
        Evaluating all four per call measured 18% slower on that shape (2.285s ->
        2.699s), against the same 23% regression the comment in
        `ComparisonHelper.compare_field_raw` records for adding work to this path.

        ONLY the annotation is cached. The comparator, threshold and weight
        built around it are not, because `match_threshold` is a plain class
        attribute a caller can reassign and `evaluate(..., match_threshold=...)`
        overrides it per call; caching those would serve a stale number.

        The cache lives in `cls.__dict__`, read with `.get` rather than
        `getattr`, so a subclass does not inherit its parent's dict and then
        write its own fields into it. Hanging it off the class also means it is
        collected with the class, where a module-level dict keyed on the class
        would keep every dynamically created model alive.

        Each entry remembers the ANNOTATION it was computed from, and a miss on
        that is what makes the cache safe. An annotation is NOT fixed for the
        whole life of a class: a forward reference is a `ForwardRef` until
        pydantic resolves it, which it does on `model_rebuild()` -- explicit, or
        the implicit one it performs the first time an incomplete model is used.
        Anything that reads a field's configuration before then (`explain()` and
        `to_json_schema()` both do) computed False from the unresolved annotation,
        and remembering that by field name alone made it permanent: two
        structurally identical models scored 1.0/tp=1 or 0.0/fd=1 depending only
        on whether something had looked at the class first. Keying on the
        annotation object costs one identity comparison and removes the ordering
        dependency, because pydantic installs a NEW annotation on resolution.
        """
        cache = cls.__dict__.get("_stickler_object_grade_cache")
        if cache is None:
            cache = {}
            setattr(cls, "_stickler_object_grade_cache", cache)
        annotation = field_info.annotation
        remembered = cache.get(field_name)
        if remembered is not None and remembered[0] is annotation:
            return remembered[1]
        answer = ConfigurationHelper.is_object_grade_annotation(field_info)
        cache[field_name] = (annotation, answer)
        return answer

    @staticmethod
    def strip_annotation_wrappers(annotation):
        """``Optional[...]`` and ``Annotated[...]`` removed, in any nesting order.

        Pydantic strips ``Annotated`` when it wraps a WHOLE annotation but NOT
        when it sits inside a union, so ``Annotated[List[Leaf], Field(...)] |
        None`` -- the spelling any ``Field(description=...)`` on an optional field
        produces -- is stored as ``Optional[Annotated[List[Leaf], FieldInfo]]``.
        ``get_origin`` on that arm reports ``Annotated`` rather than ``list``, so
        an origin test reads the wrapper and every object-grade predicate answers
        False.

        That is not cosmetic here. The field keeps the scalar Levenshtein
        default, and because ``_holds_a_plain_model`` still finds plain models in
        the list, the element comparator is still wrapped in
        ``_ClassGatedComparator`` -- which then refuses every pair, because
        Levenshtein is on the object denylist. Two IDENTICAL elements became two
        false discoveries: ``Optional[Annotated[List[Leaf], Field(...)]]`` scored
        1.0 with ``tp=2`` on ``dev`` and 0.0 with ``fd=2`` here.

        ``_annotation_is_list`` in ``structured_model.py`` records the same trap
        for the same reason; this applies that lesson to the four object-grade
        predicates. Unwrapped on both sides of the union step so
        ``Annotated[Optional[T], ...]`` answers the same as
        ``Optional[Annotated[T, ...]]``.
        """
        annotation, _ = unwrap_optional(unwrap_annotated(annotation))
        return unwrap_annotated(annotation)

    @staticmethod
    def is_plain_model_annotation(annotation) -> bool:
        """Whether an annotation is a plain pydantic model, not a StructuredModel.

        A ``StructuredModel`` is excluded because it has its own comparison
        path: the engine recurses into its fields. A plain ``BaseModel`` has no
        per-field configuration to recurse into, so it is scored as one object,
        exactly as a mapping is. See ``ComparisonDispatcher`` CASE 5.

        ``Optional[...]`` and ``Annotated[...]`` are unwrapped; see
        :meth:`strip_annotation_wrappers`. A multi-arm union such as
        ``Union[str, Cat]`` returns False for the same reason
        ``is_mapping_annotation`` refuses one: the field is not always an
        object, so an object-only comparator is the wrong default for it.
        """
        from .structured_model import StructuredModel

        try:
            annotation = ConfigurationHelper.strip_annotation_wrappers(annotation)
            return (
                isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
                and not issubclass(annotation, StructuredModel)
            )
        except Exception:
            return False

    @staticmethod
    def is_plain_model_field_type(field_info) -> bool:
        """Whether a field's annotation is a plain model. See is_plain_model_annotation."""
        try:
            return ConfigurationHelper.is_plain_model_annotation(field_info.annotation)
        except Exception:
            return False

    @staticmethod
    def _is_list_of_plain_models(field_info) -> bool:
        """Whether an annotation is a list whose ELEMENT is a plain model.

        The list form has to reach the same configuration as the singular one.
        `List[LineItem]` is not itself a model, so `is_plain_model_field_type`
        is correctly False for it, but its elements are still scored as objects.
        """
        try:
            annotation = ConfigurationHelper.strip_annotation_wrappers(
                field_info.annotation
            )
            if get_origin(annotation) not in (list, List):
                return False
            args = get_args(annotation)
            return bool(args) and ConfigurationHelper.is_plain_model_annotation(args[0])
        except Exception:
            return False

    @staticmethod
    def values_are_same_model_class(
        model_cls, field_name: str, gt_val, pred_val, *, warn: bool = True
    ) -> bool:
        """Whether two pydantic models are the same class, warning once if not.

        ``warn=False`` returns the same verdict silently; see
        :meth:`can_score_object` for why a cost-matrix probe must not warn.

        Reached only through :meth:`can_compare_object_pair`, which is the single
        gate every reader consults; see there for why the two halves of the rule
        are not callable separately.

        Judges any pair where BOTH sides are pydantic models. Anything else
        returns True: this answers a question about two models, and a
        model-versus-scalar pair is a type mismatch the caller already handles.

        A ``StructuredModel`` against a plain ``BaseModel`` IS judged, and is a
        mismatch. An earlier version waved that pair through on the grounds that
        such pairs are "left to the caller's own type dispatch", which was
        false: ``ComparisonDispatcher`` CASE 5 is the caller, and it has no
        further dispatch for them because CASE 3 requires BOTH sides to be
        ``StructuredModel``. So the pair fell through to be scored as one
        object, and two different classes -- one not even the same KIND of model
        -- reported 1.0 and a true positive where ``dev`` reported a false
        discovery. Two ``StructuredModel`` instances of the same class are
        untouched: ``type(gt) is type(pred)`` is the next line. They reach here
        only as list elements, since CASE 3 and the ``StructuredModel`` branch of
        ``ComparisonHelper.compare_field_raw`` take the singular form first.

        EXACT class, not ``isinstance``, so a subclass against its base is also
        a mismatch. A subclass carries fields the base does not, so the two do
        not describe the same shape, and scoring them on their shared fields
        would report a near-match for a schema difference.

        Scored as a false discovery rather than raised, following
        ``can_score_object``: which class arrives is prediction data, so
        raising ends a corpus run on document N after succeeding on N-1. A
        correctly annotated field cannot reach here at all -- pydantic refuses
        a ``Dog`` for an ``Optional[Cat]`` field at construction -- so this only
        fires where the annotation permitted both (``Union[Cat, Dog]``, ``Any``,
        ``object``) or where a subclass was passed for its base.
        """
        if not (isinstance(gt_val, BaseModel) and isinstance(pred_val, BaseModel)):
            return True
        if type(gt_val) is type(pred_val):
            return True
        if not warn:
            return False
        # Keyed on `model_identity`, not `cls.__name__`. `warn_once` memoises on
        # (id, context) for the process lifetime, and every model built by
        # `create_model` or the JSON schema importer is named `DynamicModel`, so a
        # bare name silently swallowed the warning for the SECOND such model -- the
        # same collision `__init_subclass__` uses `model_identity` to avoid.
        from .threshold_helper import model_identity

        warn_once(
            "plain-model-class-mismatch",
            f"{model_identity(getattr(model_cls, '__name__', str(model_cls)), getattr(model_cls, '__annotations__', {}))}"
            f".{field_name}",
            f"Field '{field_name}' compared a {type(gt_val).__name__} against a "
            f"{type(pred_val).__name__}. Different classes are scored as a false "
            "discovery (0.0), even where field names and values match. Annotate "
            "the field with a single model type, or use a nested StructuredModel, "
            "if you did not intend to permit both.",
            category=UserWarning,
        )
        return False

    @staticmethod
    def can_compare_object_pair(
        model_cls, field_name: str, comparator, gt_val, pred_val, *, warn: bool = True
    ) -> bool:
        """The whole gate on scoring a pair of values as ONE object.

        Two independent rules have to hold, and this is the only place either is
        asked, because every reader has to apply BOTH:

        1. the two values describe the same shape -- see
           :meth:`values_are_same_model_class`;
        2. the field's comparator can score an object at all -- see
           :meth:`can_score_object`.

        FIVE readers reach the question by different routes:
        ``ComparisonDispatcher`` CASE 4 and CASE 5 for ``compare_with``,
        ``StructuredModel.compare_field_raw`` and
        ``ComparisonHelper.compare_field_raw`` for ``compare``, and
        ``_ClassGatedComparator`` for a list element arriving through the
        Hungarian cost matrix. Each previously spelled its own subset, and each
        time a rule was added one of them was left behind: rule 1 landed in the
        dispatcher and not in ``compare_field_raw``, then rule 2 landed in the
        dispatcher and not in ``compare_field_raw``, and both times the result
        was ``compare()`` returning a non-zero score for a pair ``compare_with()``
        called a false discovery -- the disagreement #233 forbids, and the one
        that decides Hungarian pairings before ``compare_with`` overrules it.
        Composing the rules here rather than at five call sites is what makes
        that class of drift unspellable.

        The shape argument to :meth:`can_score_object` follows the VALUES, not
        the annotation, because that is what the comparator is about to be handed.

        Rule 2 reads both sides. Testing only ground truth meant a plain model
        arriving as the PREDICTION skipped the gate, so ``gt=[model] pred=[str]``
        scored 0.0 while the swap scored 1.0 -- a perfect match for a pair the
        code had just decided it could not score.

        Rule 2 asks about a PLAIN model only. A ``StructuredModel`` also passes
        ``isinstance(v, BaseModel)``, and refusing on that put a floor of 0.0
        under a shape the field has no say in: ``_ClassGatedComparator`` wraps the
        whole list's comparator as soon as ONE element anywhere in either list is
        a plain model, so ``[Cat(plain), Note(SM), Note(SM)]`` against an
        identical copy scored 0.0 with ``fd=3`` where ``dev`` scored 1.0 with
        ``tp=3``. The refusal exists because a plain model's annotation is what
        installs an object-grade comparator and an undeclared annotation cannot;
        a ``StructuredModel`` is scored by recursion instead and is not the
        comparator's problem. The ``Cat`` element is still refused, which is the
        declared treatment of a multi-arm union.

        ``warn=False`` gives the same verdict silently; see
        :meth:`can_score_object`.
        """
        if not ConfigurationHelper.values_are_same_model_class(
            model_cls, field_name, gt_val, pred_val, warn=warn
        ):
            return False

        from .structured_model import StructuredModel

        def is_plain_model(value) -> bool:
            return isinstance(value, BaseModel) and not isinstance(
                value, StructuredModel
            )

        gt_is_model, pred_is_model = is_plain_model(gt_val), is_plain_model(pred_val)
        if gt_is_model or pred_is_model:
            # A plain model against anything that is not one -- a bare dict, a
            # string -- is a type mismatch, and `compare_with` says so: CASE 5
            # needs BOTH sides to be a model, so the pair lands in the
            # type-mismatch branch and reports fd=1. Answering anything else here
            # is the #233 disagreement again, in the direction that matters:
            # `compare()` scored a model against a dict of the same content 1.0
            # while `compare_with()` called it a false discovery.
            if gt_is_model != pred_is_model:
                return False
            shape = "model"
        elif isinstance(gt_val, dict) or isinstance(pred_val, dict):
            shape = "mapping"
        else:
            # Neither side is an object the annotation could have configured, so
            # there is no object-grade comparator requirement to check. Scalars,
            # and StructuredModel list elements, are the comparator's own business.
            return True
        return ConfigurationHelper.can_score_object(
            model_cls, field_name, comparator, shape=shape, warn=warn
        )

    @staticmethod
    def is_mapping_annotation(annotation) -> bool:
        """Whether an annotation describes a mapping.

        Covers ``dict``, ``Dict[...]``, and the ``collections.abc.Mapping``
        family (``Mapping``, ``MutableMapping``, ``OrderedDict``,
        ``DefaultDict``, ``Counter``), plus ``Optional[...]`` around any of
        them. Recognising only ``dict``/``Dict[...]`` left ``Mapping[str, str]``
        with the type-blind Levenshtein default, which rejects mappings, so a
        field the user never configured raised at comparison time.

        A multi-arm union such as ``Union[str, Dict[str, str]]`` deliberately
        returns False: the field is not always a mapping, so a mapping-only
        comparator is the wrong default for it. Those land in the dispatcher's
        not-comparable branch instead.

        ``Annotated[...]`` is stripped on both sides of the union step, for the
        reason :meth:`strip_annotation_wrappers` records: pydantic leaves the
        wrapper on a union arm, so ``Annotated[Dict[str, str], Field(...)] |
        None`` read as a non-mapping and kept the scalar default.

        Args:
            annotation: A type annotation.

        Returns:
            True if values of this annotation are always mappings.
        """
        try:
            annotation = unwrap_annotated(annotation)
            if is_union(annotation):
                args = [a for a in union_args(annotation) if a is not type(None)]
                if len(args) != 1:
                    return False
                annotation = unwrap_annotated(args[0])
            if annotation is dict:
                return True
            origin = get_origin(annotation) or annotation
            if origin is dict:
                return True
            return isinstance(origin, type) and issubclass(origin, abc_Mapping)
        except Exception:
            return False

    @staticmethod
    def is_dict_field_type(field_info) -> bool:
        """Whether a field's annotation is a mapping. See is_mapping_annotation."""
        try:
            return ConfigurationHelper.is_mapping_annotation(field_info.annotation)
        except Exception:
            return False

    @staticmethod
    def _is_list_of_mappings(field_info) -> bool:
        """Whether an annotation is a list whose ELEMENT is a mapping.

        `List[Dict[str, str]]` is not itself a mapping, so `is_dict_field_type`
        is correctly False for it, but its elements still need a comparator that
        can score a mapping.
        """
        try:
            annotation = ConfigurationHelper.strip_annotation_wrappers(
                field_info.annotation
            )
            if get_origin(annotation) not in (list, List):
                return False
            args = get_args(annotation)
            return bool(args) and ConfigurationHelper.is_mapping_annotation(args[0])
        except Exception:
            return False

    @staticmethod
    def is_structured_field_type(field_info) -> bool:
        """Check if a field represents a structured type that needs special handling.

        Args:
            field_info: Pydantic field info object

        Returns:
            True if the field is a List[StructuredModel] or StructuredModel type
        """
        try:
            # Get the field annotation
            annotation = field_info.annotation

            # Import here to avoid circular import
            from .structured_model import StructuredModel

            # Handle List[SomeType] annotations
            if get_origin(annotation) is list:
                args = get_args(annotation)
                if args:
                    # Check if List element type is a StructuredModel subclass
                    element_type = args[0]
                    if inspect.isclass(element_type) and issubclass(
                        element_type, StructuredModel
                    ):
                        return True

            # Handle Optional[List[SomeType]] annotations, in every spelling.
            elif is_union(annotation):
                # Look for List[SomeType] in any arm.
                for union_arg in union_args(annotation):
                    if get_origin(union_arg) is list:
                        list_args = get_args(union_arg)
                        if list_args:
                            element_type = list_args[0]
                            if inspect.isclass(element_type) and issubclass(
                                element_type, StructuredModel
                            ):
                                return True

                # Handle Optional[StructuredModel] (Union[StructuredModel, NoneType]).
                # Non-required nested object fields are annotated this way (#149); without
                # this, optional nested objects inside list items are routed down the
                # non-hierarchical path and lose their nested metric breakdown.
                #
                # The spelling does not matter: `Inner | None` reaches here too.
                # It did not before, so a PEP 604-spelled optional nested object
                # silently lost exactly the breakdown #149 restored.
                if ConfigurationHelper._is_optional_structured_model(annotation):
                    return True

            # Handle direct StructuredModel annotations
            elif inspect.isclass(annotation):
                if issubclass(annotation, StructuredModel):
                    return True

        except (TypeError, AttributeError):
            # If we can't determine the type, assume it's not structured
            pass

        return False

    @staticmethod
    def get_comparison_info(cls, field_name: str) -> "ComparableFieldConfig":
        """Extract comparison info from a field.

        Args:
            cls: StructuredModel class
            field_name: Name of the field to get comparison info for

        Returns:
            ComparableFieldConfig object with comparison configuration
        """
        field_info = cls.model_fields[field_name]

        # NEW HYBRID APPROACH: Try function attribute access first (fixes custom comparators)
        if hasattr(field_info, "json_schema_extra") and callable(
            field_info.json_schema_extra
        ):
            json_func = field_info.json_schema_extra
            if hasattr(json_func, "_comparator_instance"):
                # Direct instance storage on function - this is the new, reliable approach
                comparator = getattr(json_func, "_comparator_instance")
                threshold = getattr(
                    json_func, "_threshold", _LEGACY_DEFAULT_THRESHOLD
                )
                weight = getattr(json_func, "_weight", 1.0)
                clip_under_threshold = getattr(json_func, "_clip_under_threshold", True)

                # A LAST-RESORT copy of the object-grade substitution, kept
                # because a read-time answer is better than a wrong one.
                #
                # `StructuredModel._install_object_grade_comparators` performs
                # this substitution at class-definition time for every field it
                # can see, and that is the path that matters: the metadata it
                # amends is the same object `json_schema_extra` writes as
                # `x-comparison`, so `to_json_schema()`, `explain()`, the HTML
                # reports and the engine all read one answer. Substituting only
                # here made `to_json_schema()` report LevenshteinComparator with
                # clipping ON for a field the engine scored with ANLS* and
                # clipping off.
                #
                # It still runs for a field that path could not reach -- a field
                # whose annotation was not yet resolvable when the class was
                # defined -- and it must agree with it exactly, which is why the
                # clip value comes from the shared `object_grade_clip` rather
                # than being spelled again here.
                #
                # `ComparableField()` with no comparator resolves to
                # LevenshteinComparator before the annotation is visible, and
                # Levenshtein REJECTS a mapping. Substitute the structural
                # comparator, but only when the caller named nothing: an
                # explicit choice is never overridden, so declaring Levenshtein
                # on a dict warns once and scores 0.0 rather than raising: raising
                # would end a corpus run on document N after succeeding on N-1.
                if not getattr(
                    json_func, "_comparator_explicit", True
                ) and ConfigurationHelper._wants_object_grade_comparison(
                    cls, field_name, field_info
                ):
                    comparator = ANLSStarComparator()
                    clip_under_threshold = ConfigurationHelper.object_grade_clip(
                        json_func
                    )

                from .comparison_info import ComparableFieldConfig

                return ComparableFieldConfig(
                    comparator=comparator,
                    threshold=threshold,
                    weight=weight,
                    clip_under_threshold=clip_under_threshold,
                )

        # FALLBACK: Legacy JSON schema approach for backward compatibility
        if hasattr(field_info, "json_schema_extra"):
            comparison_config = None

            if callable(field_info.json_schema_extra):
                # Handle callable json_schema_extra (from ComparableField function)
                schema = {}
                field_info.json_schema_extra(schema)
                comparison_config = schema.get("x-comparison")
            elif isinstance(field_info.json_schema_extra, dict):
                # Handle dict json_schema_extra
                comparison_config = field_info.json_schema_extra.get("x-comparison")

            if comparison_config:
                # Reconstruct from type name and config
                from .comparable_field import _reconstruct_comparator_from_type

                comparator_type = comparison_config.get(
                    "comparator_type", "LevenshteinComparator"
                )
                comparator_config_dict = comparison_config.get("comparator_config", {})
                comparator = _reconstruct_comparator_from_type(
                    comparator_type, comparator_config_dict
                )

                # Extract all configuration parameters
                threshold = comparison_config.get("threshold", 0.5)
                weight = comparison_config.get("weight", 1.0)
                clip_under_threshold = comparison_config.get(
                    "clip_under_threshold", True
                )

                from .comparison_info import ComparableFieldConfig

                return ComparableFieldConfig(
                    comparator=comparator,
                    threshold=threshold,
                    weight=weight,
                    clip_under_threshold=clip_under_threshold,
                )

        # Check if this is a structured field type that needs special handling
        if ConfigurationHelper.is_structured_field_type(field_info):
            # Use StructuredModelComparator with higher threshold for structured types
            from .comparison_info import ComparableFieldConfig

            return ComparableFieldConfig(
                comparator=StructuredModelComparator(),
                threshold=0.9,  # Higher threshold for structured object matching
                weight=1.0,
            )

        # A bare dict annotation declares no keys, so there is no per-key
        # comparison config to apply and the mapping is scored structurally by
        # ANLS*. The primitive fallback below would install
        # LevenshteinComparator, which REJECTS a dict outright: edit distance
        # over str(dict) makes key order significant, so two mappings with
        # identical content can score well below 1.0.
        #
        # clip_under_threshold=False for the same reason nested objects use it:
        # a mostly-correct mapping should keep its partial score rather than
        # being zeroed by the field threshold. See #276 and #277.
        # `List[Dict[...]]` too, keyed on the ELEMENT type. Without this the
        # element kept LevenshteinComparator, whose #281 fallback compares edit
        # distance over a canonical JSON blob: `[{"vendor": "Acme Corporation"}]`
        # against `[{"vendor": "Acme Corp"}]` scored 0.7667 and cleared a 0.7
        # threshold, while the auto path scored the same annotation 0.0. Two
        # answers for one annotation is the divergence this work removes.
        if ConfigurationHelper._wants_object_grade_comparison(
            cls, field_name, field_info
        ):
            from .comparison_info import ComparableFieldConfig

            # Same threshold source as the primitive fallback below: a dict
            # field must not be silently exempt from a match_threshold the
            # class declared. Falls back to 0.7 rather than the primitive
            # path's 0.5 because a mapping is judged as an object.
            return ComparableFieldConfig(
                comparator=ANLSStarComparator(),
                threshold=getattr(cls, "match_threshold", 0.7),
                weight=1.0,
                clip_under_threshold=False,
            )

        # Default fallback for primitive fields - use class-level threshold if available
        default_threshold = getattr(cls, "match_threshold", 0.5)
        from .comparison_info import ComparableFieldConfig

        return ComparableFieldConfig(
            comparator=LevenshteinComparator(), threshold=default_threshold, weight=1.0
        )

    @staticmethod
    def is_immediate_child(nested_path: str, field_name: str) -> bool:
        """
        Determines if nested_path is an immediate child of field_name.

        Args:
            nested_path (str): The nested path to check, e.g., 'owner.contact.phone'
            field_name (str): The potential parent path, e.g., 'owner.contact'

        Returns:
            bool: True if nested_path is an immediate child of field_name, False otherwise
        """
        # Check if field_name is a prefix of nested_path
        if not nested_path.startswith(field_name):
            return False

        # If field_name is a prefix, it should be followed by a dot
        if len(field_name) >= len(nested_path):
            return False

        if nested_path[len(field_name)] != ".":
            return False

        # The remaining part after field_name and the dot should not contain any more dots
        remaining = nested_path[len(field_name) + 1 :]
        return "." not in remaining

    @staticmethod
    def generate_model_json_schema(cls, **kwargs):
        """Override to add model-level comparison metadata.

        Extends the standard Pydantic JSON schema with comparison metadata
        at the field level.

        Args:
            cls: StructuredModel class
            **kwargs: Arguments to pass to the parent method

        Returns:
            JSON schema with added comparison metadata
        """
        schema = super(cls, cls).model_json_schema(**kwargs)

        # Add comparison metadata to each field in the schema
        for field_name, field_info in cls.model_fields.items():
            if field_name == "extra_fields":
                continue

            # Get the schema property for this field
            if field_name not in schema.get("properties", {}):
                continue

            field_props = schema["properties"][field_name]

            # Check for json_schema_extra function (ComparableField creates these)
            if hasattr(field_info, "json_schema_extra") and callable(
                field_info.json_schema_extra
            ):
                # Fallback: Check for json_schema_extra function
                temp_schema = {}
                field_info.json_schema_extra(temp_schema)

                if "x-comparison" in temp_schema:
                    # Copy the comparison metadata from the temp schema to the real schema
                    field_props["x-comparison"] = temp_schema["x-comparison"]

        return schema

    @staticmethod
    def _is_structured_model_class(annotation) -> bool:
        """Check if annotation is a direct StructuredModel class.

        Args:
            annotation: Type annotation to check

        Returns:
            True if annotation is a StructuredModel subclass
        """
        try:
            from .structured_model import StructuredModel

            return inspect.isclass(annotation) and issubclass(
                annotation, StructuredModel
            )
        except (TypeError, AttributeError):
            return False

    @staticmethod
    def _is_optional_structured_model(annotation) -> bool:
        """Check if annotation is Optional[StructuredModel].

        Args:
            annotation: Type annotation to check

        Returns:
            True if annotation is Optional[StructuredModel]
        """
        try:
            from .structured_model import StructuredModel

            # Handle Union types (like Optional[StructuredModel]), in every
            # spelling including `StructuredModel | None`. The union must
            # actually include None to be an "optional", but any arm may carry
            # the model, so a wider union still resolves.
            if is_union(annotation) and type(None) in get_args(annotation):
                for arg in union_args(annotation):
                    if inspect.isclass(arg) and issubclass(arg, StructuredModel):
                        return True
            return False
        except (TypeError, AttributeError):
            return False

    @staticmethod
    def _extract_structured_class_from_optional(annotation):
        """Extract the StructuredModel class from Optional[StructuredModel].

        Args:
            annotation: Type annotation (should be Optional[StructuredModel])

        Returns:
            The StructuredModel class, or None if not found
        """
        try:
            from .structured_model import StructuredModel

            for arg in union_args(annotation):
                if inspect.isclass(arg) and issubclass(arg, StructuredModel):
                    return arg
            return None
        except (TypeError, AttributeError):
            return None

    @staticmethod
    def _is_list_structured_model(annotation) -> bool:
        """Check if annotation is List[StructuredModel] or Optional[List[StructuredModel]].

        Args:
            annotation: Type annotation to check

        Returns:
            True if annotation is List[StructuredModel] or Optional[List[StructuredModel]]
        """
        try:
            from .structured_model import StructuredModel

            # Handle direct List[StructuredModel] annotations
            if get_origin(annotation) is list:
                args = get_args(annotation)
                if (
                    args
                    and inspect.isclass(args[0])
                    and issubclass(args[0], StructuredModel)
                ):
                    return True

            # Handle Optional[List[StructuredModel]], in every spelling.
            elif is_union(annotation):
                for arg in union_args(annotation):
                    if get_origin(arg) is list:
                        list_args = get_args(arg)
                        if (
                            list_args
                            and inspect.isclass(list_args[0])
                            and issubclass(list_args[0], StructuredModel)
                        ):
                            return True

            return False
        except (TypeError, AttributeError):
            return False

    @staticmethod
    def _extract_structured_class_from_list(annotation):
        """Extract the StructuredModel class from List[StructuredModel] or Optional[List[StructuredModel]].

        Args:
            annotation: Type annotation (should be List[StructuredModel] or Optional[List[StructuredModel]])

        Returns:
            The StructuredModel class, or None if not found
        """
        try:
            from .structured_model import StructuredModel

            # Handle direct List[StructuredModel]
            if get_origin(annotation) is list:
                args = get_args(annotation)
                if (
                    args
                    and inspect.isclass(args[0])
                    and issubclass(args[0], StructuredModel)
                ):
                    return args[0]

            # Handle Optional[List[StructuredModel]], in every spelling.
            elif is_union(annotation):
                for arg in union_args(annotation):
                    if get_origin(arg) is list:
                        list_args = get_args(arg)
                        if (
                            list_args
                            and inspect.isclass(list_args[0])
                            and issubclass(list_args[0], StructuredModel)
                        ):
                            return list_args[0]

            return None
        except (TypeError, AttributeError):
            return None

    @staticmethod
    def _process_nested_structured_data(structured_class, nested_data):
        """Process nested structured data recursively.

        Args:
            structured_class: The StructuredModel class to process with
            nested_data: Dictionary data for the nested object

        Returns:
            Dictionary with processed nested data
        """
        # Recursively call from_json to handle missing fields in nested object
        nested_instance = structured_class.from_json(
            nested_data, process_rich_values=False
        )
        # Return the model_dump to get properly processed data
        return nested_instance.model_dump()
