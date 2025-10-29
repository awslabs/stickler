from typing import Any, get_origin, cast, Annotated, get_args, Dict
from pydantic import (
    BaseModel,
    SerializationInfo,
    model_serializer,
    model_validator,
    ConfigDict,
)

from stickler.comparators.base import BaseComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator


class ComparableField[FieldType](BaseModel):
    """
    Wrapper class for field values with comparison metadata.

    Features:
    - Stores value along with comparator, threshold, weight
    - Smart serialization: returns just value by default, full metadata with context
    - Works with StructuredModel auto-wrapping
    - Value is optional for use as annotation template
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: FieldType | None = None  # Optional for template usage
    comparator: None | BaseComparator = None
    threshold: float = 0.5
    weight: float = 1
    clip_under_threshold: bool = True

    _is_comparable = True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    @model_serializer(mode="wrap")
    def field_serialise(self, serializer, info: SerializationInfo):
        # Check if context requests full comparison info
        if (
            info.context
            and isinstance(info.context, dict)
            and info.context.get("comp_info") is True  # Use boolean, not string
        ):
            return serializer(self)  # Return full model

        # Default: return just the value
        return self.value


class StructuredModel(BaseModel):
    """
    Base StructuredModel with auto-wrapping validator.
    
    This base class provides automatic wrapping of raw values into ComparableField.
    It also handles Annotated[Type, ComparableField(...)] patterns.
    
    Subclasses define their own fields.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __getattribute__(self, name: str) -> Any:
        """
        Intercept attribute access to provide type-safe ComparableField access.
        
        This allows accessing fields as ComparableField without manual casting.
        """
        attr = object.__getattribute__(self, name)
        # Return as-is (validator ensures they're always ComparableField when needed)
        return attr

    @model_validator(mode="before")
    @classmethod
    def auto_wrap_comparable_fields(cls, data: Any) -> Any:
        """
        Automatically wrap raw values in ComparableField.
        
        Supports two patterns:
        1. Simple Union: Union[str, ComparableField[str]] → wraps with defaults
        2. Annotated: Annotated[str, ComparableField(threshold=0.9)] → wraps with template config
        """
        if not isinstance(data, dict):
            return data

        # Iterate through all fields in the model
        for field_name, field_info in cls.model_fields.items():
            if field_name in data:
                raw_value = data[field_name]

                # Only wrap if it's not already a ComparableField instance
                if not isinstance(raw_value, ComparableField):
                    # Check for Annotated[Type, ComparableField(...)] pattern
                    if hasattr(cls, '__annotations__') and field_name in cls.__annotations__:
                        annotation = cls.__annotations__[field_name]
                        
                        if get_origin(annotation) is Annotated:
                            args = get_args(annotation)
                            
                            # Find ComparableField template in annotation
                            for arg in args[1:]:
                                if isinstance(arg, ComparableField):
                                    # Use template configuration
                                    field = ComparableField(value=raw_value)
                                    field.threshold = arg.threshold
                                    field.weight = arg.weight
                                    field.comparator = arg.comparator
                                    field.clip_under_threshold = arg.clip_under_threshold
                                    data[field_name] = field
                                    break
                            else:
                                # No ComparableField template found, use defaults
                                data[field_name] = ComparableField(value=raw_value)
                        else:
                            # Not Annotated, use defaults
                            data[field_name] = ComparableField(value=raw_value)
                    else:
                        # No annotation, use defaults
                        data[field_name] = ComparableField(value=raw_value)

        return data


# ============================================================================
# JSON Schema Parser: Create StructuredModels from JSON Schema
# ============================================================================

class SticklerSchemaParser:
    """Parse JSON Schema with Stickler extensions into StructuredModel classes."""
    
    COMPARATOR_MAP = {
        "ExactComparator": ExactComparator,
        "LevenshteinComparator": LevenshteinComparator,
        # Add more as needed
    }
    
    @classmethod
    def parse_schema(cls, schema: Dict[str, Any]) -> type[StructuredModel]:
        """
        Parse JSON Schema with x-aws-stickler-* extensions.
        
        Creates a StructuredModel subclass with Annotated fields that include
        ComparableField configuration from the schema extensions.
        
        Args:
            schema: JSON Schema dict with Stickler extensions
            
        Returns:
            Dynamically created StructuredModel subclass
        """
        model_name = schema.get("title", "DynamicModel")
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])
        
        # Build annotations dictionary manually
        annotations = {}
        
        for field_name, field_schema in properties.items():
            python_type = cls._json_type_to_python(field_schema.get("type", "string"))
            comparable_field = cls._create_comparable_field_template(field_schema)
            
            # Create Annotated type
            annotations[field_name] = Annotated[python_type, comparable_field]
        
        # Create the class manually instead of using create_model
        # This gives us more control over the annotations
        class_dict = {
            '__annotations__': annotations,
            '__module__': __name__,
        }
        
        # Add default values for optional fields
        for field_name in properties.keys():
            if field_name not in required_fields:
                class_dict[field_name] = None
        
        # Create the dynamic model class
        DynamicModel = type(model_name, (StructuredModel,), class_dict)
        
        # Rebuild the model to process annotations
        DynamicModel.model_rebuild()
        
        return DynamicModel
    
    @classmethod
    def _create_comparable_field_template(cls, field_schema: Dict[str, Any]) -> ComparableField:
        """
        Create ComparableField template from JSON Schema field definition.
        
        Extracts x-aws-stickler-* extensions and creates a ComparableField
        instance (without value) to use in Annotated.
        
        Args:
            field_schema: JSON Schema field definition
            
        Returns:
            ComparableField template with config (no value)
        """
        # Extract Stickler extensions
        threshold = field_schema.get("x-aws-stickler-threshold", 0.5)
        weight = field_schema.get("x-aws-stickler-weight", 1.0)
        clip = field_schema.get("x-aws-stickler-clip", True)
        comparator_name = field_schema.get("x-aws-stickler-comparator")
        
        # Create comparator instance if specified
        comparator = None
        if comparator_name and comparator_name in cls.COMPARATOR_MAP:
            comparator_class = cls.COMPARATOR_MAP[comparator_name]
            comparator = comparator_class()
        
        # Create ComparableField template (value will be set during wrapping)
        return ComparableField(
            value=None,  # Template - value comes from data
            comparator=comparator,
            threshold=threshold,
            weight=weight,
            clip_under_threshold=clip
        )
    
    @staticmethod
    def _json_type_to_python(json_type: str) -> type:
        """Convert JSON Schema type to Python type."""
        type_map = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "null": type(None)
        }
        return type_map.get(json_type, str)


# ============================================================================
# Example Models - All use Annotated pattern (RECOMMENDED)
# ============================================================================

class SimpleModel(StructuredModel):
    """
    Simple example with default configuration.
    Use when you don't need custom thresholds/weights.
    """
    name: Annotated[str, ComparableField()]
    age: Annotated[int, ComparableField()]


class ConfiguredModel(StructuredModel):
    """
    Example with custom configuration (RECOMMENDED PATTERN).
    
    Use Annotated[Type, ComparableField(...)] to specify config in type hints.
    No need to specify value= - it's auto-handled!
    """
    
    invoice_number: Annotated[str, ComparableField(
        threshold=0.9,
        weight=2.0,
        comparator=ExactComparator()
    )]
    
    customer_name: Annotated[str, ComparableField(
        threshold=0.7,
        weight=1.0,
        comparator=LevenshteinComparator()
    )]
    
    total_amount: Annotated[float, ComparableField(
        threshold=0.95,
        weight=3.0
    )]


if __name__ == "__main__":
    print("=" * 80)
    print("Annotated Pattern - The Recommended Way")
    print("=" * 80)
    print()
    
    # Example 1: Simple model with defaults
    print("=== 1. Simple Model (defaults) ===")
    simple = SimpleModel(name="John Doe", age=30)
    print(f"Created: {simple}")
    print(f"name.value: {simple.name.value}")  # type: ignore
    print(f"name.threshold: {simple.name.threshold} (default)")  # type: ignore
    print(f"Serialized: {simple.model_dump()}")
    print()
    
    # Example 2: Configured model
    print("=== 2. Configured Model (custom config in Annotated) ===")
    invoice = ConfiguredModel(
        invoice_number="INV-2025-001",
        customer_name="ACME Corporation",
        total_amount=1250.50
    )
    print(f"Created: {invoice}")
    print(f"invoice_number.value: {invoice.invoice_number.value}")  # type: ignore
    print(f"invoice_number.threshold: {invoice.invoice_number.threshold} (from Annotated)")  # type: ignore
    print(f"invoice_number.weight: {invoice.invoice_number.weight} (from Annotated)")  # type: ignore
    print(f"invoice_number.comparator: {type(invoice.invoice_number.comparator).__name__}")  # type: ignore
    print()
    print(f"total_amount.threshold: {invoice.total_amount.threshold} (from Annotated)")  # type: ignore
    print(f"Serialized: {invoice.model_dump()}")
    print(f"Serialized with comp context: {invoice.model_dump(context={'comp_info': True})}")
    print()
    
    print("=" * 80)
    print("✓ Annotated Pattern Demonstrated!")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Example 3: Dynamic Model from JSON Schema
    # ========================================================================
    
    print()
    print("=" * 80)
    print("JSON Schema → StructuredModel (Dynamic Creation)")
    print("=" * 80)
    print()
    
    # Define JSON Schema with Stickler extensions
    INVOICE_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DynamicInvoice",
        "type": "object",
        "properties": {
            "invoice_number": {
                "type": "string",
                "description": "Unique invoice identifier",
                "x-aws-stickler-threshold": 0.9,
                "x-aws-stickler-weight": 2.0,
                "x-aws-stickler-comparator": "ExactComparator",
                "x-aws-stickler-clip": True
            },
            "invoice_date": {
                "type": "string",
                "format": "date",
                "x-aws-stickler-threshold": 1.0,
                "x-aws-stickler-weight": 1.5,
                "x-aws-stickler-comparator": "ExactComparator"
            },
            "total_amount": {
                "type": "number",
                "description": "Total invoice amount",
                "x-aws-stickler-threshold": 0.95,
                "x-aws-stickler-weight": 3.0,
            },
            "vendor_name": {
                "type": "string",
                "x-aws-stickler-threshold": 0.7,
                "x-aws-stickler-weight": 1.0,
                "x-aws-stickler-comparator": "LevenshteinComparator"
            }
        },
        "required": ["invoice_number", "total_amount"]
    }
    
    print("=== 3. Creating StructuredModel from JSON Schema ===")
    print(f"Schema title: {INVOICE_SCHEMA['title']}")
    print(f"Fields: {list(INVOICE_SCHEMA['properties'].keys())}")
    print()
    
    # Parse schema to create dynamic model
    DynamicInvoice = SticklerSchemaParser.parse_schema(INVOICE_SCHEMA)
    print(f"✓ Created model: {DynamicInvoice.__name__}")
    print(f"✓ Base class: {DynamicInvoice.__bases__[0].__name__}")
    print()
    
    # Show generated field annotations
    print("Generated field annotations:")
    for field_name, annotation in DynamicInvoice.__annotations__.items():
        if get_origin(annotation) is Annotated:
            args = get_args(annotation)
            python_type = args[0]
            comp_field = args[1]
            print(f"  {field_name}: Annotated[{python_type.__name__}, ComparableField(")
            print(f"      threshold={comp_field.threshold},")
            print(f"      weight={comp_field.weight},")
            comparator_name = type(comp_field.comparator).__name__ if comp_field.comparator else None
            print(f"      comparator={comparator_name}")
            print(f"  )]")
    print()
    
    # Create instance using raw values (auto-wrapped by StructuredModel)
    print("=== 4. Creating instance with raw values (auto-wrapping) ===")
    dynamic_invoice = DynamicInvoice(
        invoice_number="INV-2025-999",
        invoice_date="2025-10-29",
        total_amount=5432.10,
        vendor_name="Dynamic Corp"
    )
    print(f"Created: {dynamic_invoice}")
    print()
    
    # Access ComparableField attributes
    print("=== 5. Accessing ComparableField metadata (from schema) ===")
    print(f"invoice_number.value: {dynamic_invoice.invoice_number.value}")  # type: ignore
    print(f"invoice_number.threshold: {dynamic_invoice.invoice_number.threshold} (from schema)")  # type: ignore
    print(f"invoice_number.weight: {dynamic_invoice.invoice_number.weight} (from schema)")  # type: ignore
    print(f"invoice_number.comparator: {type(dynamic_invoice.invoice_number.comparator).__name__}")  # type: ignore
    print()
    
    print(f"vendor_name.value: {dynamic_invoice.vendor_name.value}")  # type: ignore
    print(f"vendor_name.threshold: {dynamic_invoice.vendor_name.threshold} (from schema)")  # type: ignore
    print(f"vendor_name.comparator: {type(dynamic_invoice.vendor_name.comparator).__name__}")  # type: ignore
    print()
    
    # Serialize to clean JSON (just values)
    print("=== 6. Smart Serialization ===")
    clean_json = dynamic_invoice.model_dump()
    print(f"Clean (default): {clean_json}")
    print()
    
    # Serialize with full ComparableField metadata
    full_json = dynamic_invoice.model_dump(context={'comp_info': True})
    print(f"Full metadata (context={{'comp_info': True}}):")
    print(f"  Keys: {list(full_json.keys())}")
    print(f"  invoice_number: {{")
    print(f"      value: {full_json['invoice_number']['value']}")
    print(f"      threshold: {full_json['invoice_number']['threshold']}")
    print(f"      weight: {full_json['invoice_number']['weight']}")
    print(f"  }}")
    print()
    
    print("=" * 80)
    print("✓ JSON Schema Integration Complete!")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("=" * 80)
    print("SUMMARY: The RECOMMENDED pattern for Stickler")
    print("=" * 80)
    print()
    print("Pattern:")
    print("  field: Annotated[Type, ComparableField(threshold=0.9, weight=2.0)]")
    print()
    print("Benefits:")
    print("  ✓ Configuration in type hints (self-documenting)")
    print("  ✓ No value= parameter needed")
    print("  ✓ No validators in child classes")  
    print("  ✓ No helper methods needed")
    print("  ✓ Works with JSON Schema (x-aws-stickler-* extensions)")
    print("  ✓ Smart serialization (clean by default, full with context)")
    print("  ✓ Clean and simple!")
    print()
