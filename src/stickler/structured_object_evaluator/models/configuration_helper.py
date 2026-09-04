"""Configuration helper for StructuredModel field and schema operations.

This module provides utilities for handling field configuration, type checking,
JSON processing, and schema generation for StructuredModel instances.
"""

import inspect
from collections.abc import Mapping as abc_Mapping
from typing import TYPE_CHECKING, Any, Dict, List, get_args, get_origin

from stickler.comparators.anls import ANLSStarComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.utils.deprecation import warn_once

from .comparable_field import _LEGACY_DEFAULT_THRESHOLD
from .optional_annotation import is_union, union_args, unwrap_optional

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
    def can_score_mapping(model_cls, field_name: str, comparator) -> bool:
        """Whether ``comparator`` can score a mapping, warning once if it cannot.

        Both public entry points reach a field's comparator by different routes
        (``compare_with`` through ComparisonDispatcher, ``compare`` through
        ``compare_field_raw``), so this check lives in one place rather than
        being written twice and drifting apart.

        A DENYLIST, not an allowlist. Only the comparators known to be wrong on a
        mapping are refused; everything else is trusted.

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
        warn_once(
            "dict-value-uncomparable",
            f"{getattr(model_cls, '__name__', model_cls)}.{field_name}",
            f"Field '{field_name}' holds a mapping, but its comparator "
            f"({comparator.__class__.__name__}) scores scalars, so the pair is "
            "counted as a false discovery even if the two mappings are "
            "identical. Annotate the field as a mapping (Dict[...] or "
            "Mapping[...]) to get ANLSStarComparator automatically, declare "
            "ComparableField(comparator=ANLSStarComparator()) explicitly, or "
            "use a nested StructuredModel if you know the keys.",
            category=UserWarning,
        )
        return False

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

        Args:
            annotation: A type annotation.

        Returns:
            True if values of this annotation are always mappings.
        """
        try:
            if is_union(annotation):
                args = [a for a in union_args(annotation) if a is not type(None)]
                if len(args) != 1:
                    return False
                annotation = args[0]
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
            annotation, _ = unwrap_optional(field_info.annotation)
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

                # `ComparableField()` with no comparator resolves to
                # LevenshteinComparator before the annotation is visible, and
                # Levenshtein REJECTS a mapping. Substitute the structural
                # comparator, but only when the caller named nothing: an
                # explicit choice is never overridden, so declaring Levenshtein
                # on a dict warns once and scores 0.0 rather than raising: raising
                # would end a corpus run on document N after succeeding on N-1.
                if not getattr(
                    json_func, "_comparator_explicit", True
                ) and (
                    ConfigurationHelper.is_dict_field_type(field_info)
                    # A list of mappings too. Testing only the field's own
                    # annotation left `List[Dict[str, str]] = ComparableField(...)`
                    # on Levenshtein, scored as edit distance over a canonical JSON
                    # blob at 0.7667 (a match), while the SAME annotation with no
                    # ComparableField got ANLS* at 0.5625. One annotation, two
                    # answers, which is the divergence this work removes.
                    or ConfigurationHelper._is_list_of_mappings(field_info)
                ):
                    comparator = ANLSStarComparator()
                    clip_under_threshold = False

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
        if ConfigurationHelper.is_dict_field_type(
            field_info
        ) or ConfigurationHelper._is_list_of_mappings(field_info):
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
