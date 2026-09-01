"""Structured model comparison using Pydantic models.

This module provides the StructuredModel class for defining structured data models
with comparison configuration and evaluation capabilities.
"""

from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field
from pydantic.json_schema import GenerateJsonSchema

from stickler.comparators.base import BaseComparator
from stickler.utils.deprecation import warn_once

from .comparable_field import ComparableField
from .comparison_helper import ComparisonHelper, _maybe_absent
from .configuration_helper import ConfigurationHelper
from .evaluator_format_helper import EvaluatorFormatHelper
from .hungarian_helper import HungarianHelper
from .metrics_helper import MetricsHelper
from .null_helper import NullHelper
from .optional_annotation import union_args, unwrap_annotated, unwrap_optional
from .rich_value_helper import RichValueHelper
from .threshold_helper import THRESHOLD_DOCS_URL
from .threshold_helper import model_identity as _model_identity
from .threshold_helper import warn_if_threshold_is_zero as _warn_if_threshold_is_zero

# Name of the internal field every StructuredModel carries to hold unmatched
# input keys. It is not part of a model's data contract and must not appear in
# an exported JSON Schema.
_EXTRA_FIELDS_KEY = "extra_fields"

# Core-schema wrappers that sit between a field's `default` wrapper and its
# actual type. Unwrapped when deciding whether an annotation is Optional.
_CORE_SCHEMA_WRAPPERS = frozenset(
    {"function-after", "function-before", "function-wrap", "function-plain"}
)


def _core_schema_is_nullable(schema: Dict[str, Any]) -> bool:
    """Whether a pydantic core schema describes an Optional annotation."""
    while schema.get("type") in _CORE_SCHEMA_WRAPPERS:
        schema = schema.get("schema", {})
    return schema.get("type") == "nullable"


class _AnnotationDrivenJsonSchema(GenerateJsonSchema):
    """Schema generator that derives `required` from the annotation.

    ``ComparableField`` gives every field ``default=None`` so that model
    construction tolerates partial predictions (the comparison engine builds
    instances from prediction JSON that may omit fields). Pydantic reads that
    default as "optional", so a rendered schema claims nothing is required and
    downstream consumers (e.g. Strands' ``convert_pydantic_to_tool_spec``)
    tell the LLM every field may be omitted or null.

    The annotation already says what the user meant: ``shipment_id: str`` is
    required, ``notes: Optional[str]`` is not. This generator restores that
    reading for schema purposes only -- a field is required when its
    annotation is non-Optional and it carries no real default -- without
    changing runtime construction. Running at generation time (rather than
    post-processing) means nested models rendered into ``$defs`` get the same
    treatment.
    """

    def field_is_required(self, field, total: bool) -> bool:
        if super().field_is_required(field, total):
            return True
        wrapped = field.get("schema", {})
        if wrapped.get("type") != "default":
            return False
        if "default_factory" in wrapped:
            # A real default (e.g. extra_fields' dict factory): optional.
            return False
        if wrapped.get("default") is not None:
            # An explicit, meaningful default: optional.
            return False
        # default=None on a non-Optional annotation is ComparableField's
        # construction-tolerance sentinel, not a statement that the field is
        # optional.
        return not _core_schema_is_nullable(wrapped.get("schema", {}))



def _compose_schema_generator(
    supplied: Optional[Type[GenerateJsonSchema]],
) -> Type[GenerateJsonSchema]:
    """Mix the annotation-driven requiredness rule into a caller's generator.

    Returns the mixin alone when nothing was supplied, the caller's class
    unchanged when it already carries the rule, and otherwise a synthesised
    subclass putting the mixin first so its ``field_is_required`` wins while
    every other customisation (``$ref`` templates, naming conventions) is
    preserved.
    """
    if supplied is None:
        return _AnnotationDrivenJsonSchema
    if issubclass(supplied, _AnnotationDrivenJsonSchema):
        return supplied
    return type(
        f"AnnotationDriven{supplied.__name__}",
        (_AnnotationDrivenJsonSchema, supplied),
        {},
    )


def _strip_x_comparison(node: Any) -> None:
    """Recursively remove ``x-comparison`` keys from a rendered schema.

    Comparison configuration (comparator, threshold, weight) is evaluation
    metadata, not part of the shape ``model_json_schema()`` describes. Leaving
    it in bloats tool specs sent to LLMs and shows the model the rubric it is
    about to be graded against. The deliberate export path,
    ``to_json_schema()``, still carries the configuration as
    ``x-aws-stickler-*`` extensions.
    """
    if isinstance(node, dict):
        node.pop("x-comparison", None)
        for value in node.values():
            _strip_x_comparison(value)
    elif isinstance(node, list):
        for item in node:
            _strip_x_comparison(item)


def _drop_null_defaults_for_required(schema_obj: Dict[str, Any]) -> None:
    """Remove ``default: null`` from properties listed in ``required``.

    Once a field is marked required, a leftover ``default: null`` is
    contradictory, and schema flatteners (Strands' among them) read it as
    permission to widen the type to nullable.
    """
    required = schema_obj.get("required")
    properties = schema_obj.get("properties")
    if not (isinstance(required, list) and isinstance(properties, dict)):
        return
    for name in required:
        prop = properties.get(name)
        if isinstance(prop, dict) and prop.get("default", ...) is None:
            del prop["default"]


def _strip_extra_fields_property(schema_obj: Dict[str, Any]) -> None:
    """Remove the internal ``extra_fields`` property from a schema object.

    Operates in place on a single JSON Schema object (a top-level schema or a
    ``$defs`` entry): drops it from ``properties`` and from ``required``. A
    no-op when ``extra_fields`` is absent.
    """
    properties = schema_obj.get("properties")
    if isinstance(properties, dict):
        properties.pop(_EXTRA_FIELDS_KEY, None)
    required = schema_obj.get("required")
    if isinstance(required, list) and _EXTRA_FIELDS_KEY in required:
        schema_obj["required"] = [r for r in required if r != _EXTRA_FIELDS_KEY]


def _annotation_is_list(annotation: Any) -> bool:
    """Check whether a type annotation denotes a list, parameterized or not.

    Every spelling of a list annotation has to answer the same way, because the
    answer decides whether a field gets list null semantics (``None`` and ``[]``
    both meaning "no items") or primitive ones. A spelling that reads as
    non-list records no TN when both sides are empty, so it silently drops
    classification evidence rather than failing loudly.

    Three traps make that easy to get wrong, and all three are handled here:

    - ``get_origin`` returns ``list`` only for a *parameterized* spelling, so
      bare ``list`` needs an identity test alongside it. Without one, ``list``
      and ``list | None`` read as non-list while ``list[str]`` and
      ``list[str] | None`` read as lists. ``typing.List`` needs no special case
      -- ``get_origin`` already normalizes it to ``list``.
    - A PEP 604 union (``list | None``) reports ``types.UnionType`` as its
      origin, not ``typing.Union``, so its arms are reached through
      :func:`optional_annotation.union_args` -- the package's single source for
      destructuring a union in every spelling -- rather than a hand-rolled
      origin check.
    - ``Annotated`` survives inside a union, where pydantic does not strip it,
      and ``get_origin`` reports ``Annotated`` rather than the type inside. So
      ``Optional[Annotated[List[str], ...]]`` -- the spelling a
      ``Field(description=...)`` produces, and what ``Annotated[List[str], ...]
      | None`` normalises to -- read as non-list. Arms are unwrapped through
      :func:`optional_annotation.unwrap_annotated` before being tested.

    Union members are recursed with the same rules rather than a bare ``is
    list``, so a bare and a parameterized member inside one union answer alike.
    Depth needs no special handling: ``typing`` flattens nested unions, so
    ``Optional[list[str] | None]`` is stored as ``Optional[list[str]]``.

    One spelling is still deliberately *not* covered: ``Any`` holding a list at
    runtime. Inferring list-ness from a value rather than an annotation is a
    different question from reading a spelling correctly, and every caller here
    has only the annotation.
    """
    # `Annotated` first: everything below asks about the type inside it, and a
    # wrapper reaching the `get_origin` test would answer for the wrapper.
    annotation = unwrap_annotated(annotation)

    if annotation is list:
        return True

    if get_origin(annotation) is list:
        return True

    # Any union arm being a list is enough, in every spelling. `union_args`
    # is the package's single source for a union's non-None arms and returns
    # `()` for a non-union, so this also bottoms out the recursion.
    return any(_annotation_is_list(arg) for arg in union_args(annotation))


class StructuredModel(BaseModel):
    """Base class for models with structured comparison capabilities.

    This class extends Pydantic's BaseModel with the ability to compare
    instances using configurable comparison metrics for each field.

    Architecture - Delegation Pattern:
    ----------------------------------
    StructuredModel uses a delegation pattern where comparison logic is
    distributed across specialized helper classes. This refactoring reduced
    the class from 2584 lines to ~1486 lines while maintaining all
    functionality. Several previously-monolithic concerns (recursive
    comparison, dispatch, list comparison, confusion-matrix metrics,
    non-match collection, evaluator formatting) now live in dedicated
    components, with thin delegating shims kept on the model for backward
    compatibility. See ``docs/structured_model_REFACTORING.md`` for the
    full component map.

    The delegation pattern works as follows:
    1. StructuredModel maintains the public API (compare, compare_with, compare_field_raw)
    2. All implementation details are delegated to specialized helper classes
    3. Each helper class has a single, well-defined responsibility
    4. Helpers receive the StructuredModel instance as a parameter (composition)
    5. This avoids circular dependencies and keeps the architecture clean

    Helper Classes and Their Responsibilities:
    ------------------------------------------

    **Model Creation:**
    - ModelFactory: Creates dynamic StructuredModel subclasses from JSON configuration
      - Validates configuration structure
      - Converts field definitions to Pydantic fields
      - Creates model classes using Pydantic's create_model()

    **Comparison Orchestration:**
    - ComparisonEngine: Main orchestrator for the comparison process
      - Coordinates between dispatcher, collectors, and calculators
      - Implements single-traversal optimization
      - Manages compare_recursive and compare_with methods

    **Field Comparison Routing:**
    - ComparisonDispatcher: Routes field comparisons to appropriate handlers
      - Uses match-statement based dispatch for clarity
      - Handles null cases and type mismatches
      - Delegates to specialized comparators based on field type

    **Field-Level Comparison:**
    - FieldComparator: Compares primitive and structured fields
      - Handles string, int, float comparisons
      - Handles nested StructuredModel comparisons
      - Applies threshold-based binary classification

    - PrimitiveListComparator: Compares lists of primitive values
      - Uses Hungarian matching for optimal pairing
      - Returns hierarchical structure for API consistency
      - Handles empty list cases

    - StructuredListComparator: Compares lists of StructuredModels
      - Uses Hungarian matching with object-level similarity
      - Performs threshold-gated recursive analysis
      - Calculates nested field metrics

    **Metrics Calculation:**
    - ConfusionMatrixCalculator: Calculates confusion matrix metrics
      - Computes TP, FP, TN, FN, FD, FA counts
      - Handles list-level and field-level metrics
      - Calculates nested field metrics for structured lists

    - AggregateMetricsCalculator: Rolls up child metrics to parent nodes
      - Performs recursive traversal of result tree
      - Sums child aggregate metrics to parent
      - Provides universal field-level granularity

    - DerivedMetricsCalculator: Calculates derived metrics
      - Computes precision, recall, F1, accuracy
      - Supports both traditional and FD-inclusive recall
      - Delegates to MetricsHelper for calculations

    - ConfusionMatrixBuilder: Orchestrates all metrics calculation
      - Coordinates between the three calculator classes
      - Ensures correct calculation order
      - Builds complete confusion matrices

    **Non-Match Documentation:**
    - NonMatchCollector: Documents non-matching fields
      - Collects object-level non-matches for lists
      - Collects field-level non-matches (legacy format)
      - Handles nested StructuredModel recursion

    **Existing Helpers (Pre-Refactoring):**
    - HungarianHelper: Hungarian algorithm for list matching
    - MetricsHelper: Derived metrics calculation formulas
    - ConfigurationHelper: Field configuration management
    - ComparisonHelper: Comparison utility methods
    - EvaluatorFormatHelper: Output formatting for evaluators
    - NonMatchesHelper: Non-match collection utilities
    - FieldHelper: Field type and null checking utilities

    Benefits of Delegation Pattern:
    --------------------------------
    1. **Maintainability**: Each class has a single responsibility
    2. **Testability**: Components can be tested in isolation
    3. **Extensibility**: Easy to add new field types or metrics
    4. **Readability**: Clear separation of concerns
    5. **Performance**: No overhead - delegation is just function calls

    Migration Notes:
    ----------------
    - All public APIs remain unchanged (complete backward compatibility)
    - All tests pass without modification (80+ test files)
    - Performance characteristics maintained (single-traversal optimization)
    - No breaking changes for existing users

    Features:
    ---------
    - Field-level comparison configuration via ComparableField
    - Nested model comparison with recursive evaluation
    - Integration with ANLS* comparators
    - JSON schema generation with comparison metadata
    - Unordered list comparison using Hungarian matching
    - Confusion matrix metrics (TP, FP, FN, TN, FA, FD)
    - Aggregate metrics rollup from nested fields
    - Retention of extra fields not defined in the model
    - Dynamic model creation from JSON configuration
    - Threshold-gated recursive analysis for performance

    Example Usage:
    --------------
    >>> from stickler import StructuredModel
    >>> from stickler import ComparableField
    >>> from stickler import LevenshteinComparator
    >>>
    >>> class Product(StructuredModel):
    ...     name: str = ComparableField(
    ...         comparator=LevenshteinComparator(),
    ...         threshold=0.8,
    ...         weight=2.0
    ...     )
    ...     price: float = ComparableField(
    ...         comparator=NumericComparator(),
    ...         threshold=0.9
    ...     )
    >>>
    >>> gt = Product(name="Widget", price=29.99)
    >>> pred = Product(name="Widgit", price=29.99)  # Typo in name
    >>>
    >>> # Simple comparison (returns overall similarity score)
    >>> score = gt.compare(pred)
    >>> print(f"Similarity: {score:.2f}")
    >>>
    >>> # Detailed comparison with confusion matrix
    >>> result = gt.compare_with(pred, include_confusion_matrix=True)
    >>> print(f"TP: {result['overall']['tp']}, FD: {result['overall']['fd']}")
    >>> print(f"F1: {result['aggregate']['derived']['cm_f1']:.2f}")
    """

    # Default match threshold - can be overridden in subclasses
    match_threshold: ClassVar[float] = 0.7

    extra_fields: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",  # Allow extra fields to be stored in extra_fields
    }

    def __init_subclass__(cls, **kwargs):
        """Validate field configurations when a StructuredModel subclass is defined."""
        super().__init_subclass__(**kwargs)

        # Validate field configurations using class annotations since model_fields isn't populated yet
        if hasattr(cls, "__annotations__"):
            for field_name, field_type in cls.__annotations__.items():
                if field_name == "extra_fields":
                    continue

                # Get the field default value if it exists
                field_default = getattr(cls, field_name, None)

                # Since ComparableField is now always a function that returns a Field,
                # we need to check if field_default has comparison metadata
                if hasattr(field_default, "json_schema_extra") and callable(
                    field_default.json_schema_extra
                ):
                    # Check for comparison metadata
                    temp_schema = {}
                    field_default.json_schema_extra(temp_schema)
                    if "x-comparison" in temp_schema:
                        # This field was created with ComparableField function - validate constraints
                        if cls._is_list_of_structured_model_type(field_type):
                            comparison_config = temp_schema["x-comparison"]

                            # Threshold validation - only flag if explicitly set to non-default value
                            threshold = comparison_config.get("threshold", 0.5)
                            if threshold != 0.5:  # Default threshold value
                                # Do not echo 0.0 back as advice: the threshold
                                # test is `>=`, so `match_threshold = 0.0` makes
                                # every paired object a true positive. Telling a
                                # user to set it would walk them straight into
                                # the misconfiguration warn_if_threshold_is_zero
                                # exists to flag.
                                remedy = (
                                    "Set a positive 'match_threshold' on the list element "
                                    "class (0.0 would classify every paired object as a "
                                    f"true positive). See {THRESHOLD_DOCS_URL}"
                                    if threshold == 0.0
                                    else f"Set 'match_threshold = {threshold}' on the list element class."
                                )
                                raise ValueError(
                                    f"Field '{field_name}' is a List[StructuredModel] and cannot have a "
                                    f"'threshold' parameter in ComparableField. Hungarian matching uses each "
                                    f"StructuredModel's 'match_threshold' class attribute instead. "
                                    f"{remedy}"
                                )

                            # Comparator validation - only flag if explicitly set to non-default type
                            comparator_type = comparison_config.get(
                                "comparator_type", "LevenshteinComparator"
                            )
                            if (
                                comparator_type != "LevenshteinComparator"
                            ):  # Default comparator type
                                raise ValueError(
                                    f"Field '{field_name}' is a List[StructuredModel] and cannot have a "
                                    f"'comparator' parameter in ComparableField. Object comparison uses each "
                                    f"StructuredModel's individual field comparators instead."
                                )
                    else:
                        continue

                    # Same identity scheme as the match_threshold check below:
                    # a dynamically built model is named "DynamicModel", so two
                    # anonymous configs that share a field name (amount, date,
                    # id -- these recur constantly across document schemas)
                    # would otherwise collide and the second would be silent.
                    _warn_if_threshold_is_zero(
                        temp_schema["x-comparison"].get("threshold"),
                        f"{_model_identity(cls.__name__, cls.__annotations__)}.{field_name}",
                        "threshold",
                    )

        # `match_threshold` is a plain class attribute rather than a field, so
        # it is not covered by the loop above.
        if "match_threshold" in cls.__dict__:
            _warn_if_threshold_is_zero(
                cls.__dict__["match_threshold"],
                _model_identity(cls.__name__, cls.__annotations__),
                "match_threshold",
            )

    def model_post_init(self, __context):
        """Initialize confidence storage after model creation."""
        # Use object.__setattr__ to bypass Pydantic field detection
        object.__setattr__(self, "__stickler_field_confidences__", {})

    @classmethod
    def _is_list_of_structured_model_type(cls, field_type) -> bool:
        """Check if a field type annotation represents List[StructuredModel].

        Args:
            field_type: The field type annotation

        Returns:
            True if the field is a List[StructuredModel] type
        """
        # Handle direct imports and typing constructs
        origin = get_origin(field_type)
        if origin is list or origin is List:
            args = get_args(field_type)
            if args:
                # Use consolidated method for element type check
                return cls._is_structured_model_type(args[0])

        # Handle Union types (like Optional[List[StructuredModel]]), in every
        # spelling -- `list[Model] | None` reaches here too. Searches every arm
        # rather than requiring a single one, so a wider union such as
        # `Optional[List[Model]] | Any` still resolves to a list of models.
        else:
            for arg in union_args(field_type):
                if cls._is_list_of_structured_model_type(arg):
                    return True

        return False

    def get_field_confidence(self, field_name: str) -> Optional[float]:
        """Get confidence for a field."""
        # Don't create the attribute - just check if it exists
        if not hasattr(self, "__stickler_field_confidences__"):
            return None
        return self.__stickler_field_confidences__.get(field_name)

    def get_all_confidences(self) -> Dict[str, float]:
        """Get all confidences."""
        # Don't create the attribute - return empty dict if no confidence data
        if not hasattr(self, "__stickler_field_confidences__"):
            return {}
        return self.__stickler_field_confidences__.copy()

    def get_field_extras(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get user-provided extras for a field (non-system metadata from rich values)."""
        if not hasattr(self, "__stickler_field_extras__"):
            return None
        return self.__stickler_field_extras__.get(field_name)

    def get_all_extras(self) -> Dict[str, Dict[str, Any]]:
        """Get all user-provided extras, keyed by field path."""
        if not hasattr(self, "__stickler_field_extras__"):
            return {}
        return self.__stickler_field_extras__.copy()

    # Names the library writes onto instances via object.__setattr__. User
    # JSON containing any of these at the top level would silently shadow
    # the library's own metadata under ``extra: "allow"``, so from_json
    # rejects them up front rather than letting confidence/extras get
    # overwritten by user data.
    _RESERVED_DUNDER_NAMES: ClassVar[frozenset] = frozenset(
        {
            "__stickler_raw_json__",
            "__stickler_field_confidences__",
            "__stickler_field_extras__",
        }
    )

    @classmethod
    def from_json(
        cls,
        json_data: Dict[str, Any],
        process_rich_values: Optional[bool] = None,
        process_confidence: Optional[bool] = None,
    ) -> "StructuredModel":
        """Create a StructuredModel instance from JSON data.

        This method handles missing fields gracefully and stores extra fields
        in the extra_fields attribute. When process_rich_values is True,
        rich value structures (e.g., {"_value": "Widget", "_confidence": 0.95})
        are automatically unwrapped, with metadata stored separately.

        Args:
            json_data: Dictionary containing the JSON data
            process_rich_values: Whether to unwrap rich values on this call.
                Set to False for recursive calls where the parent already handled it.
            process_confidence: Deprecated alias for ``process_rich_values``;
                emits a DeprecationWarning. Will be removed in 0.5.0.

        Returns:
            StructuredModel instance created from the JSON data

        Raises:
            ValueError: If ``json_data`` contains any reserved
                ``__stickler_*`` dunder name at the top level.
        """
        if isinstance(json_data, dict):
            reserved_in_payload = cls._RESERVED_DUNDER_NAMES.intersection(json_data)
            if reserved_in_payload:
                raise ValueError(
                    f"json_data contains reserved key(s): "
                    f"{sorted(reserved_in_payload)}. The "
                    f"'__stickler_*' namespace is reserved for library "
                    f"metadata and cannot appear in user payloads."
                )

        if process_confidence is not None:
            warn_once(
                "process_confidence_kwarg",
                "",
                "StructuredModel.from_json(process_confidence=...) is "
                "deprecated; use process_rich_values=... instead. Support "
                "for the legacy kwarg will be removed in 0.5.0.",
            )
            if process_rich_values is None:
                process_rich_values = process_confidence

        if process_rich_values is None:
            process_rich_values = True

        if process_rich_values:
            # Only process rich values on the top-level call
            processed_data, confidences, extras = (
                RichValueHelper.process_rich_values(json_data)
            )
            instance = ConfigurationHelper.from_json(cls, processed_data)
            if confidences:
                object.__setattr__(
                    instance, "__stickler_field_confidences__", confidences
                )
            if extras:
                object.__setattr__(instance, "__stickler_field_extras__", extras)
            # Unconditional so map/reduce aggregation works when confidence
            # scores are added later; matches the Rich Value Pattern doc.
            object.__setattr__(instance, "__stickler_raw_json__", json_data)
        else:
            # Skip rich value processing for recursive calls
            instance = ConfigurationHelper.from_json(cls, json_data)
        return instance

    @classmethod
    def model_from_json(cls, config: Dict[str, Any]) -> Type["StructuredModel"]:
        """Create a StructuredModel subclass from JSON configuration using Pydantic's create_model().

        This method leverages Pydantic's native dynamic model creation capabilities to ensure
        full compatibility with all Pydantic features while adding structured comparison
        functionality through inherited StructuredModel methods.

        The generated model inherits all StructuredModel capabilities:
        - compare_with() method for detailed comparisons
        - Field-level comparison configuration
        - Hungarian algorithm for list matching
        - Confusion matrix generation
        - JSON schema with comparison metadata

        Args:
            config: JSON configuration with fields, comparators, and model settings.
                   Required keys:
                   - fields: Dict mapping field names to field configurations
                   Optional keys:
                   - model_name: Name for the generated class (default: "DynamicModel")
                   - match_threshold: Overall matching threshold (default: 0.7)

                   Field configuration format:
                   {
                       "type": "str|int|float|bool|List[str]|etc.",  # Required
                       "comparator": "LevenshteinComparator|ExactComparator|etc.",  # Optional
                       "threshold": 0.8,  # Optional, default 0.5
                       "weight": 2.0,     # Optional, default 1.0
                       "required": true,  # Optional, default false
                       "default": "value", # Optional
                       "description": "Field description",  # Optional
                       "alias": "field_alias",  # Optional
                       "examples": ["example1", "example2"]  # Optional
                   }

        Returns:
            A fully functional StructuredModel subclass created with create_model()

        Raises:
            ValueError: If configuration is invalid or contains unsupported types/comparators
            KeyError: If required configuration keys are missing

        Examples:
            >>> config = {
            ...     "model_name": "Product",
            ...     "match_threshold": 0.8,
            ...     "fields": {
            ...         "name": {
            ...             "type": "str",
            ...             "comparator": "LevenshteinComparator",
            ...             "threshold": 0.8,
            ...             "weight": 2.0,
            ...             "required": True
            ...         },
            ...         "price": {
            ...             "type": "float",
            ...             "comparator": "NumericComparator",
            ...             "default": 0.0
            ...         }
            ...     }
            ... }
            >>> ProductClass = StructuredModel.model_from_json(config)
            >>> isinstance(ProductClass.model_fields, dict)  # Full Pydantic compatibility
            True
            >>> product = ProductClass(name="Widget", price=29.99)
            >>> product.name
            'Widget'
            >>> result = product.compare_with(ProductClass(name="Widget", price=29.99))
            >>> result["overall_score"]
            1.0
        """
        # Delegate to ModelFactory for dynamic model creation
        from .model_factory import ModelFactory

        return ModelFactory.create_model_from_json(config, base_class=cls)

    @classmethod
    def from_json_schema(cls, schema: Dict[str, Any]) -> Type["StructuredModel"]:
        """Create a StructuredModel subclass from a JSON Schema document.

        This method accepts standard JSON Schema documents and creates fully functional
        StructuredModel classes with comparison capabilities. Supports JSON Schema draft-07+.

        Comparison behavior can be customized using x-aws-stickler-* extension fields:

        Field-Level Extensions:
        -----------------------
        - x-aws-stickler-comparator: Comparator algorithm name (built-in or registered custom)
        - x-aws-stickler-threshold: Similarity threshold for match/no-match (0.0-1.0, default: 0.5)
        - x-aws-stickler-weight: Field importance in overall scoring (>0.0, default: 1.0)
        - x-aws-stickler-clip-under-threshold: Clip scores below threshold to 0.0 (bool, default: false)

        Model-Level Extensions:
        -----------------------
        - x-aws-stickler-model-name: Generated class name (default: "DynamicModel")
        - x-aws-stickler-match-threshold: Overall match threshold (default: 0.7)

        Supported Features:
        -------------------
        - Primitive types: string, number, integer, boolean
        - Nullable list-form types, e.g. {"type": ["string", "null"]}
        - Nullable two-branch anyOf types with one explicit null branch
        - oneOf alternatives are not interpreted or enforced
        - Object schemas inferred from properties when type is omitted
        - Nested objects and arrays (primitive/object items)
        - Required fields, defaults, descriptions
        - Schema references ($ref with #/definitions/ and #/$defs/)

        Default Type Mappings:
        ----------------------
        - string → LevenshteinComparator (threshold: 0.5)
        - number/integer → NumericComparator (threshold: 0.5)
        - boolean → ExactComparator (threshold: 1.0)
        - arrays → Hungarian matching with element-appropriate comparators
        - objects → Recursive field-by-field comparison

        Args:
            schema: JSON Schema document as a dictionary

        Returns:
            StructuredModel subclass created from the schema

        Raises:
            ValueError: If schema is invalid or contains unsupported features
            jsonschema.exceptions.SchemaError: If schema doesn't conform to JSON Schema spec

        Examples:
            Basic usage with standard JSON Schema:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "integer"},
            ...         "email": {"type": "string"}
            ...     },
            ...     "required": ["name", "email"]
            ... }
            >>> PersonModel = StructuredModel.from_json_schema(schema)
            >>> person1 = PersonModel(name="Alice", age=30, email="alice@example.com")
            >>> person2 = PersonModel(name="Alicia", age=30, email="alice@example.com")
            >>> result = person1.compare_with(person2)
            >>> # name field uses LevenshteinComparator, age uses NumericComparator

            Advanced usage with x-aws-stickler-* extensions:
            >>> schema = {
            ...     "type": "object",
            ...     "x-aws-stickler-model-name": "Product",
            ...     "x-aws-stickler-match-threshold": 0.8,
            ...     "properties": {
            ...         "name": {
            ...             "type": "string",
            ...             "x-aws-stickler-comparator": "LevenshteinComparator",
            ...             "x-aws-stickler-threshold": 0.9,
            ...             "x-aws-stickler-weight": 2.0,
            ...         },
            ...         "price": {
            ...             "type": "number",
            ...             "x-aws-stickler-comparator": "NumericComparator",
            ...             "x-aws-stickler-threshold": 0.95,
            ...             "x-aws-stickler-clip-under-threshold": true
            ...         }
            ...     },
            ...     "required": ["name"]
            ... }
            >>> ProductModel = StructuredModel.from_json_schema(schema)
            >>> result = product1.compare_with(product2)
            >>> # name field has weight=2.0, price field clips scores below 0.95
        """

        return cls._from_json_schema_internal(schema, field_path="")

    @classmethod
    def from_pydantic(
        cls,
        model_cls: Type,
        *,
        weight_hints: bool = False,
        match_threshold: float = 0.7,
    ) -> Type["StructuredModel"]:
        """Create a StructuredModel subclass from a vanilla pydantic model class.

        Walks the live ``model_cls.model_fields`` and infers a sensible
        comparator/threshold per field from the Python type and field name
        (see ``stickler.auto``): ``bool``/``Enum``/``Literal`` -> Exact,
        ``int``/``float`` -> Numeric, ``date``/``datetime`` -> Date,
        ``str`` -> Levenshtein, with name-token refinement (``*_id`` -> Exact,
        ``*amount`` -> Numeric, ...) gated on type compatibility. Nested
        ``BaseModel`` and ``List[BaseModel]`` fields recurse.

        The result is an ordinary StructuredModel subclass: construct
        instances from your pydantic instances via
        ``Model.from_json(instance.model_dump())``, compare with
        ``compare_with()``, feed pairs to ``BulkStructuredModelEvaluator``,
        or export with ``to_stickler_config()`` / ``to_json_schema()``, edit,
        and rebuild if you want different comparators.

        Args:
            model_cls: A ``pydantic.BaseModel`` subclass (e.g. a Strands
                agent ``response_model``). A StructuredModel subclass is
                returned unchanged (explicit configuration always wins).
            weight_hints: Apply name-token weight heuristics (default off, so
                weights stay uniform).
            match_threshold: Overall match threshold for the generated model.

        Returns:
            A StructuredModel subclass mirroring ``model_cls`` with inferred
            comparison configuration.

        Examples:
            >>> class Invoice(BaseModel):
            ...     invoice_id: str
            ...     total_amount: float
            >>> InvoiceEval = StructuredModel.from_pydantic(Invoice)
            >>> gt = InvoiceEval.from_json(gt_invoice.model_dump())
            >>> pred = InvoiceEval.from_json(pred_invoice.model_dump())
            >>> gt.compare_with(pred)["overall_score"]
        """
        from ...auto.builder import structured_model_for

        if isinstance(model_cls, type) and issubclass(model_cls, cls):
            return model_cls
        return structured_model_for(
            model_cls,
            weight_hints=weight_hints,
            match_threshold=match_threshold,
        )

    @classmethod
    def _from_json_schema_internal(
        cls, schema: Dict[str, Any], field_path: str
    ) -> Type["StructuredModel"]:
        """Internal method for creating StructuredModel from JSON Schema with field path tracking.

        This is used internally for recursive calls to track field paths for error messages.
        External callers should use from_json_schema() instead.

        Args:
            schema: JSON Schema document as a dictionary
            field_path: Current field path for error messages (e.g., "address.street")

        Returns:
            StructuredModel subclass created from the schema
        """
        # Import dependencies
        from ..utils.json_schema_validator import validate_json_schema
        from .json_schema_field_converter import JsonSchemaFieldConverter
        from .model_factory import ModelFactory

        # Subtask 4.2: Validate JSON Schema
        try:
            validate_json_schema(schema)
        except Exception as e:
            raise ValueError(
                f"Invalid JSON Schema: {e}. "
                f"Please ensure the schema conforms to JSON Schema draft-07 specification."
            )

        # Subtask 4.3: Extract model-level configuration
        model_name = schema.get("x-aws-stickler-model-name", "DynamicModel")
        match_threshold = schema.get("x-aws-stickler-match-threshold", 0.7)

        # Validate model name
        if not isinstance(model_name, str) or not model_name.isidentifier():
            raise ValueError(
                f"x-aws-stickler-model-name must be a valid Python identifier, "
                f"got: {model_name}"
            )

        # Validate match threshold
        if not isinstance(match_threshold, (int, float)):
            raise ValueError(
                f"x-aws-stickler-match-threshold must be a number, "
                f"got: {type(match_threshold).__name__}"
            )

        if not (0.0 <= match_threshold <= 1.0):
            raise ValueError(
                f"x-aws-stickler-match-threshold must be between 0.0 and 1.0, "
                f"got: {match_threshold}"
            )

        # Subtask 4.4: Convert fields and create model
        # Ensure schema has properties
        if "properties" not in schema:
            raise ValueError(
                "JSON Schema must contain 'properties' key for object type"
            )

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Create converter and convert properties to field definitions
        converter = JsonSchemaFieldConverter(schema, field_path=field_path)
        field_definitions = converter.convert_properties_to_fields(properties, required)

        # Create the model using ModelFactory
        return ModelFactory.create_model_from_fields(
            model_name=model_name,
            field_definitions=field_definitions,
            match_threshold=match_threshold,
            base_class=cls,
        )

    @classmethod
    def _is_structured_field_type(cls, field_info) -> bool:
        """Check if a field represents a structured type that needs special handling.

        Args:
            field_info: Pydantic field info object

        Returns:
            True if the field is a List[StructuredModel] or StructuredModel type
        """
        return ConfigurationHelper.is_structured_field_type(field_info)

    @classmethod
    def _get_comparison_info(cls, field_name: str) -> ComparableField:
        """Extract comparison info from a field.

        Args:
            field_name: Name of the field to get comparison info for

        Returns:
            ComparableField object with comparison configuration
        """
        return ConfigurationHelper.get_comparison_info(cls, field_name)


    def _should_use_hierarchical_structure(self, val: Any, field_name: str) -> bool:
        """Check if a list value should maintain hierarchical structure.

        For lists, we need to check if they should maintain hierarchical structure
        based on their field type configuration.

        Args:
            val: Value to check (typically a list)
            field_name: Name of the field being evaluated

        Returns:
            True if the value should use hierarchical structure, False otherwise
        """
        if isinstance(val, list):
            # Check if this field is configured as List[StructuredModel]
            field_info = self.__class__.model_fields.get(field_name)
            if field_info and self._is_structured_field_type(field_info):
                return True
        return False

    def _is_list_field(self, field_name: str) -> bool:
        """Check if a field is ANY list type.

        Args:
            field_name: Name of the field to check

        Returns:
            True if the field is a list type (List[str], List[StructuredModel], etc.)
        """
        field_info = self.__class__.model_fields.get(field_name)
        if not field_info:
            return False

        field_type = field_info.annotation
        return _annotation_is_list(field_type)

    def _handle_list_field_dispatch(
        self, gt_val: Any, pred_val: Any, weight: float
    ) -> dict:
        """Handle list field comparison using match statements.

        DEPRECATED: This method now delegates to ComparisonDispatcher.
        Kept for backward compatibility with any external callers.

        Args:
            gt_val: Ground truth list value
            pred_val: Predicted list value
            weight: Field weight for scoring

        Returns:
            Comparison result dictionary
        """
        from .comparison_dispatcher import ComparisonDispatcher

        dispatcher = ComparisonDispatcher(self)
        return dispatcher.handle_list_field_dispatch(gt_val, pred_val, weight)

    def _calculate_object_level_metrics(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        match_threshold: float,
    ) -> tuple:
        """Calculate object-level metrics using Hungarian matching.

        Args:
            gt_list: Ground truth list
            pred_list: Predicted list
            match_threshold: Threshold for considering objects as matches

        Returns:
            Tuple of (object_metrics_dict, matched_pairs, matched_gt_indices, matched_pred_indices)
        """
        # Use Hungarian matching for OBJECT-LEVEL counts - OPTIMIZED: Single call gets all info
        hungarian_helper = HungarianHelper()
        hungarian_info = hungarian_helper.get_complete_matching_info(gt_list, pred_list)
        matched_pairs = hungarian_info["matched_pairs"]

        # Count OBJECTS, not individual fields
        tp_objects = 0  # Objects with similarity >= match_threshold
        fd_objects = 0  # Objects with similarity < match_threshold
        for gt_idx, pred_idx, similarity in matched_pairs:
            if similarity >= match_threshold:
                tp_objects += 1
            else:
                fd_objects += 1

        # Count unmatched objects
        matched_gt_indices = {idx for idx, _, _ in matched_pairs}
        matched_pred_indices = {idx for _, idx, _ in matched_pairs}
        fn_objects = len(gt_list) - len(matched_gt_indices)  # Unmatched GT objects
        fa_objects = len(pred_list) - len(
            matched_pred_indices
        )  # Unmatched pred objects

        # Build list-level metrics counting OBJECTS (not fields)
        object_level_metrics = {
            "tp": tp_objects,
            "fa": fa_objects,
            "fd": fd_objects,
            "fp": fa_objects + fd_objects,  # Total false positives
            "tn": 0,  # No true negatives at object level for non-empty lists
            "fn": fn_objects,
        }

        return (
            object_level_metrics,
            matched_pairs,
            matched_gt_indices,
            matched_pred_indices,
        )

    def _compare_unordered_lists(
        self,
        gt_list: List[Any],
        pred_list: List[Any],
        comparator: BaseComparator,
        threshold: float,
    ) -> Dict[str, Any]:
        """Compare two lists as unordered collections using Hungarian matching.

        Args:
            list1: First list
            list2: Second list
            comparator: Comparator to use for item comparison
            threshold: Minimum score to consider a match

        Returns:
            Dictionary with confusion matrix metrics including:
            - tp: True positives (matches >= threshold)
            - fd: False discoveries (matches < threshold)
            - fa: False alarms (unmatched prediction items)
            - fn: False negatives (unmatched ground truth items)
            - fp: Total false positives (fd + fa)
            - overall_score: Similarity score for backward compatibility
        """
        return ComparisonHelper.compare_unordered_lists(
            gt_list, pred_list, comparator, threshold
        )

    def compare_field_raw(self, field_name: str, other_value: Any) -> float:
        """Compare a single field with a value WITHOUT applying thresholds.

        This version is used by the compare method to get raw similarity scores.

        Args:
            field_name: Name of the field to compare
            other_value: Value to compare with

        Returns:
            Raw similarity score between 0.0 and 1.0 without threshold filtering
        """
        # Get our field value
        my_value = getattr(self, field_name)

        # If both values are StructuredModel instances, use recursive compare_with
        if isinstance(my_value, StructuredModel) and isinstance(
            other_value, StructuredModel
        ):
            # Use compare_with for rich comparison, but extract the raw score
            comparison_result = my_value.compare_with(
                other_value,
                include_confusion_matrix=False,
                document_non_matches=False,
                evaluator_format=False,
                recall_with_fd=False,
            )
            return comparison_result["overall_score"]

        # For non-StructuredModel fields, use existing logic
        return ComparisonHelper.compare_field_raw(self, field_name, other_value)

    def compare_recursive(self, other: "StructuredModel") -> dict:
        """The ONE clean recursive function that handles everything.

        Enhanced to capture BOTH confusion matrix metrics AND similarity scores
        in a single traversal to eliminate double traversal inefficiency.

        PHASE 2: Delegates to ComparisonEngine while maintaining identical behavior.

        Args:
            other: Another instance of the same model to compare with

        Returns:
            Dictionary with clean hierarchical structure:
            - overall: TP, FP, TN, FN, FD, FA counts + similarity_score + all_fields_matched
            - fields: Recursive structure for each field with scores
            - non_matches: List of non-matching items
        """
        from .comparison_engine import ComparisonEngine

        engine = ComparisonEngine(self)
        return engine.compare_recursive(other)

    def _dispatch_field_comparison(
        self, field_name: str, gt_val: Any, pred_val: Any
    ) -> dict:
        """Enhanced case-based dispatch using match statements for clean logic flow.

        DEPRECATED: This method now delegates to ComparisonDispatcher.
        Kept for backward compatibility with any external callers.
        """
        from .comparison_dispatcher import ComparisonDispatcher

        dispatcher = ComparisonDispatcher(self)
        return dispatcher.dispatch_field_comparison(field_name, gt_val, pred_val)

    def _add_derived_metrics_to_result(
        self, result: dict, recall_with_fd: bool = False
    ) -> dict:
        """Walk through result and add 'derived' fields with F1, precision, recall, accuracy.

        This method delegates to DerivedMetricsCalculator for the actual implementation.

        Args:
            result: Result from compare_recursive with basic TP, FP, FN, etc. metrics
            recall_with_fd: If True, include FD in recall denominator (TP/(TP+FN+FD))
                           If False, use traditional recall (TP/(TP+FN))

        Returns:
            Modified result with 'derived' fields added at each level
        """
        from .derived_metrics_calculator import DerivedMetricsCalculator

        calculator = DerivedMetricsCalculator()
        return calculator.add_derived_metrics_to_result(result, recall_with_fd)

    def _has_basic_metrics(self, metrics_dict: dict) -> bool:
        """Check if a dictionary has basic confusion matrix metrics.

        Args:
            metrics_dict: Dictionary to check

        Returns:
            True if it has the basic metrics (tp, fp, fn, etc.)
        """
        basic_metrics = ["tp", "fp", "fn", "tn", "fa", "fd"]
        return all(metric in metrics_dict for metric in basic_metrics)

    def _classify_field_for_confusion_matrix(
        self, field_name: str, other_value: Any, threshold: float = None
    ) -> Dict[str, Any]:
        """Classify a field comparison according to the confusion matrix rules.

        This method delegates to ConfusionMatrixCalculator for the actual implementation.

        Args:
            field_name: Name of the field being compared
            other_value: Value to compare with
            threshold: Threshold for matching (uses field's threshold if None)

        Returns:
            Dictionary with TP, FP, TN, FN, FD counts and derived metrics
        """
        from .confusion_matrix_calculator import ConfusionMatrixCalculator

        calculator = ConfusionMatrixCalculator(self)
        return calculator.classify_field_for_confusion_matrix(
            field_name, other_value, threshold
        )

    def _calculate_list_confusion_matrix(
        self, field_name: str, other_list: List[Any]
    ) -> Dict[str, Any]:
        """Calculate confusion matrix for a list field, including nested field metrics.

        This method delegates to ConfusionMatrixCalculator for the actual implementation.

        Args:
            field_name: Name of the list field being compared
            other_list: Predicted list to compare with

        Returns:
            Dictionary with:
            - Top-level TP, FP, TN, FN, FD, FA counts and derived metrics for the list field
            - nested_fields: Dict with metrics for individual fields within list items (e.g., "transactions.date")
            - non_matches: List of individual object-level non-matches for detailed analysis
        """
        from .confusion_matrix_calculator import ConfusionMatrixCalculator

        calculator = ConfusionMatrixCalculator(self)
        return calculator.calculate_list_confusion_matrix(field_name, other_list)

    def _calculate_nested_field_metrics(
        self,
        list_field_name: str,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        threshold: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for individual fields within list items.

        This method delegates to ConfusionMatrixCalculator for the actual implementation.

        THRESHOLD-GATED RECURSION: Only perform recursive field analysis for object pairs
        with similarity >= StructuredModel.match_threshold. Poor matches and unmatched
        items are treated as atomic units.

        Args:
            list_field_name: Name of the parent list field (e.g., "transactions")
            gt_list: Ground truth list of StructuredModel objects
            pred_list: Predicted list of StructuredModel objects
            threshold: Matching threshold (not used for threshold-gating)

        Returns:
            Dictionary mapping nested field paths to their confusion matrix metrics
            E.g., {"transactions.date": {...}, "transactions.description": {...}}
        """
        from .confusion_matrix_calculator import ConfusionMatrixCalculator

        calculator = ConfusionMatrixCalculator(self)
        return calculator.calculate_nested_field_metrics(
            list_field_name, gt_list, pred_list, threshold
        )

    def _calculate_single_nested_field_metrics(
        self,
        parent_field_name: str,
        gt_nested: "StructuredModel",
        pred_nested: "StructuredModel",
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for fields within a single nested StructuredModel.

        This method delegates to ConfusionMatrixCalculator for the actual implementation.

        Args:
            parent_field_name: Name of the parent field (e.g., "address")
            gt_nested: Ground truth nested StructuredModel
            pred_nested: Predicted nested StructuredModel

        Returns:
            Dictionary mapping nested field paths to their confusion matrix metrics
            E.g., {"address.street": {...}, "address.city": {...}}
        """
        from .confusion_matrix_calculator import ConfusionMatrixCalculator

        calculator = ConfusionMatrixCalculator(self)
        return calculator.calculate_single_nested_field_metrics(
            parent_field_name, gt_nested, pred_nested
        )

    def _collect_enhanced_non_matches(
        self, recursive_result: dict, other: "StructuredModel"
    ) -> List[Dict[str, Any]]:
        """Collect enhanced non-matches with object-level granularity.

        This method delegates to NonMatchCollector for the actual implementation.

        Args:
            recursive_result: Result from compare_recursive containing field comparison details
            other: The predicted StructuredModel instance

        Returns:
            List of non-match dictionaries with enhanced object-level information
        """
        from .non_match_collector import NonMatchCollector

        collector = NonMatchCollector(self)
        return collector.collect_enhanced_non_matches(recursive_result, other)

    def compare(self, other: "StructuredModel") -> float:
        """Compare this model with another and return a scalar similarity score.

        Returns the overall weighted average score regardless of sufficient/necessary field matching.
        This provides a more nuanced score for use in comparators.

        Args:
            other: Another instance of the same model to compare with

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # We'll calculate the overall weighted score directly instead of using compare_with
        # This ensures that sufficient/necessary field rules don't cause a zero score
        # when at least some fields match

        total_score = 0.0
        total_weight = 0.0

        for field_name in self.__class__.model_fields:
            # Skip the extra_fields attribute in comparison
            if field_name == "extra_fields":
                continue
            if hasattr(other, field_name):
                self_value = getattr(self, field_name)
                other_value = getattr(other, field_name)

                # A true negative is absence of evidence, not evidence that two
                # objects match. Omit absent-on-both fields from the weighted
                # average that Hungarian matching uses. The cheap guard preserves
                # the populated-value fast path in pairwise cost matrices.
                if _maybe_absent(self_value) and _maybe_absent(other_value):
                    is_absent = (
                        NullHelper.is_effectively_null_for_lists
                        if self._is_list_field(field_name)
                        else NullHelper.is_effectively_null_for_primitives
                    )
                    if is_absent(self_value) and is_absent(other_value):
                        continue

                # Get field configuration
                info = self.__class__._get_comparison_info(field_name)
                # Use weight from ComparableField object
                weight = info.weight

                # Compare field values WITHOUT applying thresholds
                field_score = self.compare_field_raw(field_name, other_value)

                # Update total score
                total_score += field_score * weight
                total_weight += weight

        # Calculate overall score
        if total_weight > 0:
            return total_score / total_weight

        # Every compared field was absent on both sides. Nothing disagreed, so
        # identical empty objects remain a perfect match (#233).
        return 1.0

    def compare_with(
        self,
        other: "StructuredModel",
        include_confusion_matrix: bool = False,
        document_non_matches: bool = False,
        evaluator_format: bool = False,
        recall_with_fd: bool = False,
        add_derived_metrics: bool = True,
        document_field_comparisons: bool = False,
        add_confidence_metrics: bool = False,
        confidence_metrics: Optional[List[Any]] = None,
        add_bbox_metrics: bool = False,
        bbox_iou_thresholds: Optional[Union[float, Iterable[float]]] = None,
    ) -> Dict[str, Any]:
        """Compare this model with another instance using SINGLE TRAVERSAL optimization.

        PHASE 2: Delegates to ComparisonEngine while maintaining identical behavior.

        Args:
            other: Another instance of the same model to compare with
            include_confusion_matrix: Whether to include confusion matrix calculations
            document_non_matches: Whether to document non-matches for analysis
            evaluator_format: Whether to format results for the evaluator
            recall_with_fd: If True, include FD in recall denominator (TP/(TP+FN+FD))
                            If False, use traditional recall (TP/(TP+FN))
            add_derived_metrics: Whether to add derived metrics to confusion matrix
            document_field_comparisons: Whether to document all matches and non matches made in the comparison
            add_confidence_metrics: Whether to add confidence calibration metrics.
                Emits a UserWarning recommending BulkStructuredModelEvaluator for
                statistically meaningful results.
            confidence_metrics: Optional list of ConfidenceMetric instances to compute.
                Defaults to [AUROCMetric()] if not provided. Only used when
                add_confidence_metrics=True. For bulk evaluation, pass the metric
                list to BulkStructuredModelEvaluator instead.
            add_bbox_metrics: Whether to add bounding-box mAP metrics (single-doc
                sanity check). Emits a UserWarning recommending
                BulkStructuredModelEvaluator with BBoxMAPAccumulator for
                statistically meaningful results.
            bbox_iou_thresholds: A single IoU threshold or an iterable of them
                for mAP. Defaults to the COCO range (0.50, 0.55, ..., 0.95).
                Only used when add_bbox_metrics=True.

        Returns:
            Dictionary with comparison results including:
            - field_scores: Scores for each field
            - overall_score: Weighted average score
            - all_fields_matched: Whether all fields matched
            - confusion_matrix: (optional) Confusion matrix data if requested
            - non_matches: (optional) Non-match documentation if requested
            - field_comparisons: (optional) Field level comparison information if requested
            - confidence_metrics: (optional) Confidence calibration metrics if requested
        """
        from .comparison_engine import ComparisonEngine

        engine = ComparisonEngine(self)
        return engine.compare_with(
            other,
            include_confusion_matrix=include_confusion_matrix,
            document_non_matches=document_non_matches,
            evaluator_format=evaluator_format,
            recall_with_fd=recall_with_fd,
            add_derived_metrics=add_derived_metrics,
            document_field_comparisons=document_field_comparisons,
            add_confidence_metrics=add_confidence_metrics,
            confidence_metrics=confidence_metrics,
            add_bbox_metrics=add_bbox_metrics,
            bbox_iou_thresholds=bbox_iou_thresholds,
        )

    def _convert_score_to_binary_metrics(
        self, score: float, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Convert similarity score to binary classification metrics using MetricsHelper.

        Args:
            score: Similarity score [0-1]
            threshold: Threshold for considering a match

        Returns:
            Dictionary with TP, FP, FN, TN counts converted to metrics
        """
        metrics_helper = MetricsHelper()
        return metrics_helper.convert_score_to_binary_metrics(score, threshold)

    def _format_for_evaluator(
        self,
        result: Dict[str, Any],
        other: "StructuredModel",
        recall_with_fd: bool = False,
    ) -> Dict[str, Any]:
        """Format comparison results for evaluator compatibility.

        Args:
            result: Standard comparison result from compare_with
            other: The other model being compared
            recall_with_fd: Whether to include FD in recall denominator

        Returns:
            Dictionary in evaluator format with overall, fields, confusion_matrix
        """
        return EvaluatorFormatHelper.format_for_evaluator(
            self, result, other, recall_with_fd
        )

    def _calculate_list_item_metrics(
        self,
        field_name: str,
        gt_list: List[Any],
        pred_list: List[Any],
        recall_with_fd: bool = False,
    ) -> List[Dict[str, Any]]:
        """Calculate metrics for individual items in a list field.

        Args:
            field_name: Name of the list field
            gt_list: Ground truth list
            pred_list: Prediction list
            recall_with_fd: Whether to include FD in recall denominator

        Returns:
            List of metrics dictionaries for each matched item pair
        """
        return EvaluatorFormatHelper.calculate_list_item_metrics(
            field_name, gt_list, pred_list, recall_with_fd
        )

    @classmethod
    def model_json_schema(cls, **kwargs):
        """Render the model's shape for external consumers.

        This is Pydantic's contract for "describe this shape", and it is what
        schema consumers such as Strands' ``convert_pydantic_to_tool_spec``
        call. Three corrections are applied to the standard rendering so a
        configured ``StructuredModel`` describes the same shape as the plain
        ``BaseModel`` a developer would otherwise write (issue #188):

        - ``required`` is derived from the annotation, so ``shipment_id: str``
          renders required even though ``ComparableField`` assigns
          ``default=None`` for construction tolerance, and required fields do
          not carry a contradictory ``default: null``.
        - Comparison configuration (``x-comparison``) is not emitted.
          Evaluation config is not part of the shape; the deliberate export
          path ``to_json_schema()`` still carries it as ``x-aws-stickler-*``
          extensions.
        - The internal ``extra_fields`` property is not emitted (top level or
          nested ``$defs``); it holds unmatched input keys and is not part of
          the data contract. This also lets the output round-trip through
          ``from_json_schema()`` (issue #214).

        Field-level ``description``, ``examples``, and ``alias`` pass through
        untouched, since those are genuinely useful to a schema consumer.

        Args:
            **kwargs: Arguments to pass to the parent method

        Returns:
            JSON schema describing the model's shape
        """
        # Compose with a caller-supplied generator rather than deferring to it.
        # `schema_generator` is a documented public parameter, and
        # `setdefault` would leave a caller's class in place -- silently
        # dropping the requiredness derivation and rendering `required` as
        # absent again, which is the bug this method exists to fix. The mixin
        # only overrides `field_is_required`, so it composes with anything.
        kwargs["schema_generator"] = _compose_schema_generator(
            kwargs.get("schema_generator")
        )
        schema = super().model_json_schema(**kwargs)

        # `json_schema_extra` attaches `x-comparison` during generation, so the
        # strip below removes it rather than declining to add it. Comparison
        # config is stickler's own bookkeeping and has no meaning to a schema
        # consumer; `to_json_schema()` is the export that deliberately carries
        # it, as `x-aws-stickler-*`.
        for schema_obj in (schema, *schema.get("$defs", {}).values()):
            _strip_extra_fields_property(schema_obj)
            _drop_null_defaults_for_required(schema_obj)
        _strip_x_comparison(schema)

        return schema

    @classmethod
    def to_json_schema(cls) -> Dict[str, Any]:
        """Export model as JSON Schema with x-aws-stickler-* extensions.

        Creates a JSON Schema document compatible with from_json_schema() for
        round-trip serialization. Extracts comparison metadata from fields and
        formats them as x-aws-stickler-* extensions.

        Returns:
            JSON Schema dict with x-aws-stickler-* extensions

        Example:
            >>> class Product(StructuredModel):
            ...     name: str = ComparableField(threshold=0.8, weight=2.0)
            ...     price: float = ComparableField(threshold=0.95)
            >>> schema = Product.to_json_schema()
            >>> ReconstructedProduct = StructuredModel.from_json_schema(schema)
            >>> # ReconstructedProduct has identical comparison behavior
        """
        from .json_schema_field_converter import (
            PYTHON_TYPE_TO_JSON_TYPE,
            JsonSchemaFieldConverter,
        )

        # schema/field_path unused for export operations - only needed for import
        converter = JsonSchemaFieldConverter(schema={}, field_path="")

        schema = {
            "type": "object",
            "x-aws-stickler-model-name": cls.__name__,
            "properties": {},
            "required": [],
        }

        # Add match_threshold if available (check both attribute names for compatibility)
        threshold = getattr(cls, "match_threshold", None)
        if threshold is None:
            threshold = getattr(cls, "_match_threshold", None)
        if threshold is not None:
            schema["x-aws-stickler-match-threshold"] = threshold

        for field_name, field_info in cls.model_fields.items():
            # Skip extra_fields to avoid circular serialization issues
            if field_name == "extra_fields":
                continue

            field_type = field_info.annotation

            # Validate field has type annotation
            if field_type is None:
                # Defensive: unreachable through normal Pydantic model construction
                raise ValueError(f"Field '{field_name}' has no type annotation")

            # Unwrap Optional before type checking
            field_type, _ = cls._unwrap_optional(field_type)

            # Check if nested StructuredModel - recursively export to maintain full configuration
            if cls._is_structured_model_type(field_type):
                property_schema = field_type.to_json_schema()
                metadata = converter._extract_field_metadata(field_info)
                metadata.pop("comparator", None)
                extensions = converter._build_comparison_extensions(metadata, output_format="json_schema")
                property_schema.update(extensions)
            elif get_origin(field_type) is list:
                # Handle List[StructuredModel] or List[primitive]
                args = get_args(field_type)
                if not args:
                    # Defensive: unreachable through normal Pydantic model construction
                    raise ValueError(
                        f"Field '{field_name}' has unparameterized list type. "
                        f"Use List[str], List[int], etc."
                    )
                # Unwrap an optional element before dispatching on it.
                # `_is_structured_model_type` unwraps internally, so without this
                # `List[Optional[Model]]` passed the check and then called
                # `to_json_schema()` on the `Optional[...]` wrapper, which has no
                # such attribute -- an AttributeError instead of a schema. The
                # primitive branch needs it too: `Optional[int]` is not a key in
                # PYTHON_TYPE_TO_JSON_TYPE, so it fell through to "string".
                element_type, element_is_nullable = cls._unwrap_optional(args[0])

                if cls._is_structured_model_type(element_type):
                    # List of StructuredModels - recursively export element schema
                    items_schema = element_type.to_json_schema()
                    if element_is_nullable:
                        items_schema = {"anyOf": [items_schema, {"type": "null"}]}
                    property_schema = {
                        "type": "array",
                        "items": items_schema,
                    }
                    metadata = converter._extract_field_metadata(field_info)
                    metadata.pop("comparator", None)
                    extensions = converter._build_comparison_extensions(metadata, output_format="json_schema")
                    property_schema.update(extensions)
                else:
                    # Primitive list - build array schema manually
                    json_element_type = PYTHON_TYPE_TO_JSON_TYPE.get(
                        element_type, "string"
                    )
                    property_schema = {
                        "type": "array",
                        "items": {
                            "type": [json_element_type, "null"]
                            if element_is_nullable
                            else json_element_type
                        },
                    }
                    # Extract and add stickler extensions from field metadata
                    metadata = converter._extract_field_metadata(field_info)
                    extensions = converter._build_comparison_extensions(
                        metadata, output_format="json_schema"
                    )
                    property_schema.update(extensions)
            else:
                # Primitive type - use converter for consistent formatting.
                # field_type is already unwrapped above, so pass whether the
                # original annotation was Optional so nullability round-trips.
                _, field_is_nullable = cls._unwrap_optional(field_info.annotation)
                property_schema = converter.field_to_property(
                    field_type, field_info, is_nullable=field_is_nullable
                )

            schema["properties"][field_name] = property_schema

            # Add to required if field is required (Pydantic uses is_required())
            if field_info.is_required():
                schema["required"].append(field_name)

        return schema

    @staticmethod
    def _unwrap_optional(field_type: Type) -> tuple:
        """Unwrap Optional[T] to (T, True) or return (T, False) if not Optional.

        Recognises every spelling, including ``T | None``. This is load-bearing
        for ``to_json_schema()``: the nested-model branch, the list branch and
        the nullability of a primitive property all key off it, so a spelling it
        fails to recognise falls through to the scalar path and exports as
        ``{"type": "string"}`` -- silently replacing a nested model, or a whole
        array of models, with a string.

        Args:
            field_type: Type annotation to unwrap

        Returns:
            Tuple of (unwrapped_type, is_optional)
        """
        return unwrap_optional(field_type)

    @staticmethod
    def _is_structured_model_type(field_type: Type) -> bool:
        """Check if type is a StructuredModel subclass.

        Handles Union/Optional types by unwrapping them first.

        Args:
            field_type: Type annotation to check

        Returns:
            True if field_type is a StructuredModel subclass
        """
        # Unwrap Optional/Union types
        unwrapped_type, _ = StructuredModel._unwrap_optional(field_type)

        try:
            return isinstance(unwrapped_type, type) and issubclass(
                unwrapped_type, StructuredModel
            )
        except TypeError:
            return False

    @classmethod
    def to_stickler_config(cls) -> Dict[str, Any]:
        """Export model as custom Stickler JSON configuration.

        Creates a configuration dict compatible with model_from_json() for
        round-trip serialization. Extracts comparison metadata and formats
        them in the custom Stickler configuration format.

        Returns:
            Stickler config dict with model_name and fields

        Example:
            >>> class Product(StructuredModel):
            ...     name: str = ComparableField(threshold=0.8, weight=2.0)
            ...     price: float = ComparableField(threshold=0.95)
            >>> config = Product.to_stickler_config()
            >>> ReconstructedProduct = StructuredModel.model_from_json(config)
            >>> # ReconstructedProduct has identical comparison behavior
        """
        from .json_schema_field_converter import JsonSchemaFieldConverter

        # schema/field_path unused for export operations - only needed for import
        converter = JsonSchemaFieldConverter(schema={}, field_path="")

        config = {"model_name": cls.__name__, "fields": {}}

        # Add match_threshold if available (check both attribute names for compatibility)
        threshold = getattr(cls, "match_threshold", None)
        if threshold is None:
            threshold = getattr(cls, "_match_threshold", None)
        if threshold is not None:
            config["match_threshold"] = threshold

        for field_name, field_info in cls.model_fields.items():
            # Skip extra_fields to avoid circular serialization issues
            if field_name == "extra_fields":
                continue

            field_type = field_info.annotation

            # Validate field has type annotation
            if field_type is None:
                # Defensive: unreachable through normal Pydantic model construction
                raise ValueError(f"Field '{field_name}' has no type annotation")

            # Unwrap Optional before type checking
            field_type, _ = cls._unwrap_optional(field_type)

            # Check if nested StructuredModel - use "structured_model" type
            if cls._is_structured_model_type(field_type):
                nested_config = field_type.to_stickler_config()
                field_config = {"type": "structured_model", "fields": nested_config["fields"]}
                if nested_config.get("model_name"):
                    field_config["model_name"] = nested_config["model_name"]
                if nested_config.get("match_threshold") is not None:
                    field_config["match_threshold"] = nested_config["match_threshold"]
                metadata = converter._extract_field_metadata(field_info)
                metadata.pop("comparator", None)
                extensions = converter._build_comparison_extensions(metadata, output_format="stickler_config")
                field_config.update(extensions)
            elif get_origin(field_type) is list:
                # Handle List[StructuredModel] or List[primitive]
                args = get_args(field_type)
                if not args:
                    # Defensive: unreachable through normal Pydantic model construction
                    raise ValueError(
                        f"Field '{field_name}' has unparameterized list type. "
                        f"Use List[str], List[int], etc."
                    )
                # Unwrap an optional element for the same reason as
                # to_json_schema()'s list branch: the predicate below unwraps, so
                # `List[Optional[Model]]` reached `to_stickler_config()` on the
                # wrapper. The primitive branch below also reads
                # `element_type.__name__`, which a union does not have.
                element_type, _ = cls._unwrap_optional(args[0])

                if cls._is_structured_model_type(element_type):
                    nested_config = element_type.to_stickler_config()
                    field_config = {"type": "list_structured_model", "fields": nested_config["fields"]}
                    if nested_config.get("model_name"):
                        field_config["model_name"] = nested_config["model_name"]
                    if nested_config.get("match_threshold") is not None:
                        field_config["match_threshold"] = nested_config["match_threshold"]
                    metadata = converter._extract_field_metadata(field_info)
                    metadata.pop("comparator", None)
                    extensions = converter._build_comparison_extensions(metadata, output_format="stickler_config")
                    field_config.update(extensions)
                else:
                    # Primitive list - pass element type, then fix up type string
                    field_config = converter.field_to_stickler_config(
                        element_type, field_info
                    )
                    field_config["type"] = f"List[{element_type.__name__}]"
            else:
                # Primitive type - use converter for consistent formatting
                field_config = converter.field_to_stickler_config(
                    field_type, field_info
                )

            config["fields"][field_name] = field_config

        return config
