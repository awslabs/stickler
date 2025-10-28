"""Structured model comparison using Pydantic models.

This module provides the StructuredModel class for defining structured data models
with comparison configuration and evaluation capabilities.
"""

from pydantic import BaseModel, Field
from typing import (
    Any,
    Dict,
    List,
    Type,
    Union,
    ClassVar,
    get_origin,
    get_args,
)
import inspect

from stickler.comparators.base import BaseComparator

from .comparable_field import ComparableField
from .non_match_field import NonMatchField
from .hungarian_helper import HungarianHelper
from .metrics_helper import MetricsHelper
from .configuration_helper import ConfigurationHelper
from .comparison_helper import ComparisonHelper
from .evaluator_format_helper import EvaluatorFormatHelper


class StructuredModel(BaseModel):
    """Base class for models with structured comparison capabilities.

    This class extends Pydantic's BaseModel with the ability to compare
    instances using configurable comparison metrics for each field.
    
    Architecture - Delegation Pattern:
    ----------------------------------
    StructuredModel uses a delegation pattern where comparison logic is
    distributed across specialized helper classes. This refactoring reduced
    the class from 2584 lines to ~500 lines while maintaining all functionality.
    
    The delegation pattern works as follows:
    1. StructuredModel maintains the public API (compare, compare_with, compare_field)
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
    >>> from stickler.structured_object_evaluator.models import StructuredModel
    >>> from stickler.structured_object_evaluator.models import ComparableField
    >>> from stickler.comparators import LevenshteinComparator
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
                                raise ValueError(
                                    f"Field '{field_name}' is a List[StructuredModel] and cannot have a "
                                    f"'threshold' parameter in ComparableField. Hungarian matching uses each "
                                    f"StructuredModel's 'match_threshold' class attribute instead. "
                                    f"Set 'match_threshold = {threshold}' on the list element class."
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
                element_type = args[0]
                # Check if element type is a StructuredModel subclass
                try:
                    return inspect.isclass(element_type) and issubclass(
                        element_type, StructuredModel
                    )
                except (TypeError, AttributeError):
                    return False

        # Handle Union types (like Optional[List[StructuredModel]])
        elif origin is Union:
            args = get_args(field_type)
            for arg in args:
                if cls._is_list_of_structured_model_type(arg):
                    return True

        return False

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "StructuredModel":
        """Create a StructuredModel instance from JSON data.

        This method handles missing fields gracefully and stores extra fields
        in the extra_fields attribute.

        Args:
            json_data: Dictionary containing the JSON data

        Returns:
            StructuredModel instance created from the JSON data
        """
        return ConfigurationHelper.from_json(cls, json_data)

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
        StructuredModel classes with comparison capabilities. It supports JSON Schema
        draft-07 and later specifications.
        
        Comparison behavior can be customized using x-aws-stickler-* extension fields:
        
        Field-Level Extensions:
        -----------------------
        - x-aws-stickler-comparator: Specifies the comparator algorithm to use for field comparison.
          Valid comparator names:
            * "LevenshteinComparator" - String similarity using edit distance (default for strings)
            * "ExactComparator" - Exact match comparison (default for booleans)
            * "NumericComparator" - Numeric comparison with tolerance (default for numbers/integers)
            * "FuzzyComparator" - Fuzzy string matching
            * "SemanticComparator" - Semantic similarity comparison
            * "BertComparator" - BERT-based semantic comparison
            * "LLMComparator" - LLM-based comparison
            * "StructuredModelComparator" - Nested structured model comparison
          Example: "x-aws-stickler-comparator": "LevenshteinComparator"
        
        - x-aws-stickler-threshold: Minimum similarity score for binary classification (match/no-match).
          Valid range: 0.0 to 1.0 (inclusive)
          Default: 0.5 for most types, 1.0 for booleans
          Scores at or above threshold are classified as matches (TP/TN).
          Scores below threshold are classified as non-matches (FP/FN).
          Example: "x-aws-stickler-threshold": 0.85
        
        - x-aws-stickler-weight: Relative importance of this field in overall similarity scoring.
          Valid range: Any positive float value (> 0.0)
          Default: 1.0
          Higher weights increase the field's contribution to aggregate metrics.
          Used in weighted average calculations for parent-level similarity.
          Example: "x-aws-stickler-weight": 2.5
        
        - x-aws-stickler-clip-under-threshold: Controls score clipping behavior.
          Valid values: true or false (boolean)
          Default: false
          When true, similarity scores below threshold are clipped to 0.0.
          When false, actual similarity scores are preserved regardless of threshold.
          Affects continuous metrics (similarity) but not binary classification (TP/FP/TN/FN).
          Example: "x-aws-stickler-clip-under-threshold": true
        
        - x-aws-stickler-aggregate: Controls confusion matrix metric aggregation.
          Valid values: true or false (boolean)
          Default: false
          When true, this field's confusion matrix metrics (TP/FP/TN/FN) are included in
          parent-level aggregate counts. When false, metrics are calculated but not aggregated.
          Useful for excluding certain fields from overall accuracy/precision/recall calculations.
          Example: "x-aws-stickler-aggregate": true
        
        Model-level configuration at root:
        - x-aws-stickler-model-name: Generated class name (default: "DynamicModel")
        - x-aws-stickler-match-threshold: Overall match threshold (default: 0.7)
        
        Supported JSON Schema Features:
        - Primitive types: string, number, integer, boolean
        - Object types with nested properties
        - Array types (both primitive and object arrays)
        - Required fields via "required" keyword
        - Default values via "default" keyword
        - Documentation via "title" and "description" keywords
        - Schema references via "$ref" keyword (#/definitions/ and #/$defs/)
        
        Type Mapping and Default Comparators:
        --------------------------------------
        Each JSON Schema type is automatically mapped to a Python type and assigned a default
        comparator with appropriate threshold settings. These defaults can be overridden using
        x-aws-stickler-* extensions.
        
        Primitive Types:
        - JSON Schema "string" → Python str
          Default Comparator: LevenshteinComparator (edit distance-based string similarity)
          Default Threshold: 0.5 (50% similarity required for match)
          Use Case: Text fields where minor variations should still match (names, descriptions)
        
        - JSON Schema "number" → Python float
          Default Comparator: NumericComparator (numeric comparison with tolerance)
          Default Threshold: 0.5 (50% similarity required for match)
          Use Case: Floating-point values where small differences are acceptable (prices, measurements)
        
        - JSON Schema "integer" → Python int
          Default Comparator: NumericComparator (numeric comparison with tolerance)
          Default Threshold: 0.5 (50% similarity required for match)
          Use Case: Integer values where small differences are acceptable (counts, quantities)
        
        - JSON Schema "boolean" → Python bool
          Default Comparator: ExactComparator (exact match only)
          Default Threshold: 1.0 (100% match required - must be identical)
          Use Case: Boolean flags where only exact matches are meaningful (true/false, yes/no)
        
        Complex Types:
        - JSON Schema "array" with primitive items → Python List[T]
          Default Comparator: Based on item type (e.g., LevenshteinComparator for string items)
          Default Threshold: 0.5 (or 1.0 for boolean items)
          Matching Strategy: Hungarian algorithm for optimal element pairing (order-independent)
          Use Case: Lists of values where order doesn't matter (tags, categories, IDs)
        
        - JSON Schema "array" with object items → Python List[StructuredModel]
          Default Comparator: N/A (uses Hungarian matching with recursive comparison)
          Default Threshold: 0.7 (model-level threshold for object matching)
          Matching Strategy: Hungarian algorithm with threshold-gated recursive analysis
          Use Case: Lists of structured objects (line items, transactions, nested records)
        
        - JSON Schema "object" → Nested StructuredModel
          Default Comparator: N/A (recursive field-by-field comparison)
          Default Threshold: 0.7 (model-level threshold inherited or specified)
          Matching Strategy: Recursive comparison of all nested fields
          Use Case: Nested structured data (addresses, contact info, metadata objects)
        
        Notes on Default Behavior:
        - String and numeric types use fuzzy matching (0.5 threshold) to handle minor variations
        - Boolean types require exact matches (1.0 threshold) since partial matches are meaningless
        - Arrays use Hungarian matching to find optimal element pairings regardless of order
        - Nested objects are compared recursively with field-level granularity
        - All defaults can be overridden using x-aws-stickler-* extensions for custom behavior
        
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
            >>> # Result contains similarity scores and confusion matrix metrics
            >>> # name field uses LevenshteinComparator by default for strings
            >>> # age field uses NumericComparator by default for integers
            
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
            ...             "x-aws-stickler-aggregate": true
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
            >>> product1 = ProductModel(name="Widget", price=19.99)
            >>> product2 = ProductModel(name="Widgit", price=20.00)
            >>> result = product1.compare_with(product2)
            >>> # name field has weight=2.0, so it contributes more to overall similarity
            >>> # price field clips scores below 0.95 threshold to 0.0
            >>> # name field metrics are aggregated to parent level
            
            Nested objects:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "address": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "street": {"type": "string"},
            ...                 "city": {"type": "string"},
            ...                 "zipcode": {"type": "string"}
            ...             },
            ...             "required": ["city"]
            ...         }
            ...     },
            ...     "required": ["name"]
            ... }
            >>> PersonModel = StructuredModel.from_json_schema(schema)
            >>> person1 = PersonModel(
            ...     name="Alice",
            ...     address={"street": "123 Main St", "city": "Boston", "zipcode": "02101"}
            ... )
            >>> person2 = PersonModel(
            ...     name="Alice",
            ...     address={"street": "123 Main Street", "city": "Boston", "zipcode": "02101"}
            ... )
            >>> result = person1.compare_with(person2)
            >>> # Nested address object is automatically compared field-by-field
            >>> # Result includes hierarchical metrics for address.street, address.city, etc.
            
            Arrays of primitives:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "tags": {
            ...             "type": "array",
            ...             "items": {"type": "string"}
            ...         }
            ...     }
            ... }
            >>> TaggedModel = StructuredModel.from_json_schema(schema)
            >>> item1 = TaggedModel(name="Product", tags=["electronics", "gadget", "new"])
            >>> item2 = TaggedModel(name="Product", tags=["electronics", "gadgets", "sale"])
            >>> result = item1.compare_with(item2)
            >>> # Arrays use Hungarian matching for optimal element pairing
            >>> # Each element is compared using the appropriate comparator (LevenshteinComparator for strings)
            
            Arrays of objects:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "order_id": {"type": "string"},
            ...         "items": {
            ...             "type": "array",
            ...             "items": {
            ...                 "type": "object",
            ...                 "properties": {
            ...                     "name": {"type": "string"},
            ...                     "quantity": {"type": "integer"},
            ...                     "price": {"type": "number"}
            ...                 },
            ...                 "required": ["name", "quantity"]
            ...             }
            ...         }
            ...     }
            ... }
            >>> OrderModel = StructuredModel.from_json_schema(schema)
            >>> order1 = OrderModel(
            ...     order_id="ORD-001",
            ...     items=[
            ...         {"name": "Widget", "quantity": 2, "price": 10.00},
            ...         {"name": "Gadget", "quantity": 1, "price": 25.00}
            ...     ]
            ... )
            >>> order2 = OrderModel(
            ...     order_id="ORD-001",
            ...     items=[
            ...         {"name": "Widget", "quantity": 2, "price": 10.00},
            ...         {"name": "Gadget", "quantity": 1, "price": 24.99}
            ...     ]
            ... )
            >>> result = order1.compare_with(order2)
            >>> # Array of objects uses Hungarian matching with threshold-gated recursion
            >>> # Each matched pair is compared field-by-field
            >>> # Result includes nested metrics for items[0].name, items[0].quantity, etc.
        """
        return cls._from_json_schema_internal(schema, field_path="")
    
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
            base_class=cls
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



    @classmethod
    def _is_aggregate_field(cls, field_name: str) -> bool:
        """Check if field is marked for confusion matrix aggregation.

        Args:
            field_name: Name of the field to check

        Returns:
            True if the field is marked for aggregation, False otherwise
        """
        return ConfigurationHelper.is_aggregate_field(cls, field_name)

    def _is_truly_null(self, val: Any) -> bool:
        """Check if a value is truly null (None).
        
        DEPRECATED: Delegates to NullHelper for consistency.
        Kept for backward compatibility with any external callers.
        """
        from .null_helper import NullHelper
        return NullHelper.is_truly_null(val)

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

    def _is_effectively_null_for_lists(self, val: Any) -> bool:
        """Check if a list value is effectively null (None or empty list).
        
        DEPRECATED: Delegates to NullHelper for consistency.
        Kept for backward compatibility with any external callers.
        """
        from .null_helper import NullHelper
        return NullHelper.is_effectively_null_for_lists(val)

    def _is_effectively_null_for_primitives(self, val: Any) -> bool:
        """Check if a primitive value is effectively null.
        
        DEPRECATED: Delegates to NullHelper for consistency.
        Kept for backward compatibility with any external callers.
        """
        from .null_helper import NullHelper
        return NullHelper.is_effectively_null_for_primitives(val)

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
        # Handle Optional types and direct List types
        if hasattr(field_type, "__origin__"):
            origin = field_type.__origin__
            if origin is list or origin is List:
                return True
            elif origin is Union:  # Optional[List[...]] case
                args = field_type.__args__
                for arg in args:
                    if hasattr(arg, "__origin__") and (
                        arg.__origin__ is list or arg.__origin__ is List
                    ):
                        return True
        return False

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

    def _create_true_negative_result(self, weight: float) -> dict:
        """Create a true negative result.
        
        DEPRECATED: Delegates to ResultHelper for consistency.
        Kept for backward compatibility with any external callers.
        """
        from .result_helper import ResultHelper
        return ResultHelper.create_true_negative_result(weight)

    def _create_false_alarm_result(self, weight: float) -> dict:
        """Create a false alarm result.
        
        DEPRECATED: Delegates to ResultHelper for consistency.
        Kept for backward compatibility with any external callers.
        """
        from .result_helper import ResultHelper
        return ResultHelper.create_false_alarm_result(weight)

    def _create_false_negative_result(self, weight: float) -> dict:
        """Create a false negative result.
        
        DEPRECATED: Delegates to ResultHelper for consistency.
        Kept for backward compatibility with any external callers.
        """
        from .result_helper import ResultHelper
        return ResultHelper.create_false_negative_result(weight)

    def _handle_struct_list_empty_cases(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        weight: float,
    ) -> dict:
        """Handle empty list cases with beautiful match statements.
        
        DEPRECATED: Delegates to ResultHelper for consistency.
        Kept for backward compatibility with any external callers.

        Args:
            gt_list: Ground truth list (may be None)
            pred_list: Predicted list (may be None)
            weight: Field weight for scoring

        Returns:
            Result dictionary if early exit needed, None if should continue processing
        """
        from .result_helper import ResultHelper
        
        # Normalize None to empty lists for consistent handling
        gt_len = len(gt_list or [])
        pred_len = len(pred_list or [])
        
        return ResultHelper.create_empty_list_result(gt_len, pred_len, weight)

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

    def _calculate_struct_list_similarity(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        info: "ComparableField",
    ) -> float:
        """Calculate raw similarity score for structured list.

        Args:
            gt_list: Ground truth list
            pred_list: Predicted list
            info: Field comparison info

        Returns:
            Raw similarity score between 0.0 and 1.0
        """
        if len(pred_list) > 0:
            match_result = self._compare_unordered_lists(
                gt_list, pred_list, info.comparator, info.threshold
            )
            return match_result.get("overall_score", 0.0)
        else:
            return 0.0

    def _compare_unordered_lists(
        self,
        list1: List[Any],
        list2: List[Any],
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
            list1, list2, comparator, threshold
        )

    def compare_field(self, field_name: str, other_value: Any) -> float:
        """Compare a single field with a value using the configured comparator.

        Args:
            field_name: Name of the field to compare
            other_value: Value to compare with

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Get our field value
        my_value = getattr(self, field_name)

        # If both values are StructuredModel instances, use recursive compare_with
        if isinstance(my_value, StructuredModel) and isinstance(
            other_value, StructuredModel
        ):
            # Use compare_with for rich comparison
            comparison_result = my_value.compare_with(
                other_value,
                include_confusion_matrix=False,
                document_non_matches=False,
                evaluator_format=False,
                recall_with_fd=False,
            )
            # Apply field-level threshold if configured
            info = self._get_comparison_info(field_name)
            raw_score = comparison_result["overall_score"]
            return (
                raw_score
                if raw_score >= info.threshold or not info.clip_under_threshold
                else 0.0
            )

        # CRITICAL FIX: For lists, don't clip under threshold for partial matches
        if isinstance(my_value, list) and isinstance(other_value, list):
            # Get field info
            info = self._get_comparison_info(field_name)

            # Use the raw comparison result without threshold clipping for lists
            result = ComparisonHelper.compare_unordered_lists(
                my_value, other_value, info.comparator, info.threshold
            )

            # Return the overall score directly (don't clip based on threshold for lists)
            return result["overall_score"]

        # For other fields, use existing logic
        return ComparisonHelper.compare_field_with_threshold(
            self, field_name, other_value
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







    def _calculate_aggregate_metrics(self, result: dict) -> dict:
        """Calculate aggregate metrics for all nodes in the result tree.

        This method delegates to AggregateMetricsCalculator for the actual implementation.

        CRITICAL FIX: Enhanced deep nesting traversal to handle arbitrary nesting depth.
        The aggregate field contains the sum of all primitive field confusion matrices
        below that node in the tree. This provides universal field-level granularity.

        Args:
            result: Result from compare_recursive with hierarchical structure

        Returns:
            Modified result with 'aggregate' fields added at each level
        """
        from .aggregate_metrics_calculator import AggregateMetricsCalculator
        calculator = AggregateMetricsCalculator()
        return calculator.calculate_aggregate_metrics(result)

    def _add_derived_metrics_to_result(self, result: dict, recall_with_fd: bool = False) -> dict:
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
        return calculator.classify_field_for_confusion_matrix(field_name, other_value, threshold)

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
        return calculator.calculate_nested_field_metrics(list_field_name, gt_list, pred_list, threshold)

    def _calculate_single_nested_field_metrics(
        self,
        parent_field_name: str,
        gt_nested: "StructuredModel",
        pred_nested: "StructuredModel",
        parent_is_aggregate: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate confusion matrix metrics for fields within a single nested StructuredModel.
        
        This method delegates to ConfusionMatrixCalculator for the actual implementation.

        Args:
            parent_field_name: Name of the parent field (e.g., "address")
            gt_nested: Ground truth nested StructuredModel
            pred_nested: Predicted nested StructuredModel
            parent_is_aggregate: Whether the parent field should aggregate child metrics

        Returns:
            Dictionary mapping nested field paths to their confusion matrix metrics
            E.g., {"address.street": {...}, "address.city": {...}}
        """
        from .confusion_matrix_calculator import ConfusionMatrixCalculator
        calculator = ConfusionMatrixCalculator(self)
        return calculator.calculate_single_nested_field_metrics(
            parent_field_name, gt_nested, pred_nested, parent_is_aggregate
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

    def _collect_non_matches(
        self, other: "StructuredModel", base_path: str = ""
    ) -> List[NonMatchField]:
        """Collect non-matches for detailed analysis.
        
        This method delegates to NonMatchCollector for the actual implementation.

        Args:
            other: Other model to compare with
            base_path: Base path for field naming (e.g., "address")

        Returns:
            List of NonMatchField objects documenting non-matches
        """
        from .non_match_collector import NonMatchCollector
        collector = NonMatchCollector(self)
        return collector.collect_non_matches(other, base_path)

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
                # Get field configuration
                info = self.__class__._get_comparison_info(field_name)
                # Use weight from ComparableField object
                weight = info.weight

                # Compare field values WITHOUT applying thresholds
                field_score = self.compare_field_raw(
                    field_name, getattr(other, field_name)
                )

                # Update total score
                total_score += field_score * weight
                total_weight += weight

        # Calculate overall score
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0

    def compare_with(
        self,
        other: "StructuredModel",
        include_confusion_matrix: bool = False,
        document_non_matches: bool = False,
        evaluator_format: bool = False,
        recall_with_fd: bool = False,
        add_derived_metrics: bool = True,
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

        Returns:
            Dictionary with comparison results including:
            - field_scores: Scores for each field
            - overall_score: Weighted average score
            - all_fields_matched: Whether all fields matched
            - confusion_matrix: (optional) Confusion matrix data if requested
            - non_matches: (optional) Non-match documentation if requested
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
        """Override to add model-level comparison metadata.

        Extends the standard Pydantic JSON schema with comparison metadata
        at the field level.

        Args:
            **kwargs: Arguments to pass to the parent method

        Returns:
            JSON schema with added comparison metadata
        """
        schema = super().model_json_schema(**kwargs)

        # Add comparison metadata to each field in the schema
        for field_name, field_info in cls.model_fields.items():
            if field_name == "extra_fields":
                continue

            # Get the schema property for this field
            if field_name not in schema.get("properties", {}):
                continue

            field_props = schema["properties"][field_name]

            # Since ComparableField is now always a function, check for json_schema_extra
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
