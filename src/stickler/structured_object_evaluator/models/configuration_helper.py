"""Configuration helper for StructuredModel field and schema operations.

This module provides utilities for handling field configuration, type checking,
JSON processing, and schema generation for StructuredModel instances.
"""

from typing import Any, Dict, Union, get_origin, get_args
import inspect

from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.structured import StructuredModelComparator

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
            
            # Handle Optional[List[SomeType]] annotations (Union[List[SomeType], NoneType])
            elif get_origin(annotation) is Union:
                union_args = get_args(annotation)
                # Look for List[SomeType] within the Union
                for union_arg in union_args:
                    if get_origin(union_arg) is list:
                        list_args = get_args(union_arg)
                        if list_args:
                            element_type = list_args[0]
                            if inspect.isclass(element_type) and issubclass(
                                element_type, StructuredModel
                            ):
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
                threshold = getattr(json_func, "_threshold", 0.5)
                weight = getattr(json_func, "_weight", 1.0)
                clip_under_threshold = getattr(json_func, "_clip_under_threshold", True)
                aggregate = getattr(json_func, "_aggregate", False)
                
                from .comparison_info import ComparableFieldConfig

                return ComparableFieldConfig(
                    comparator=comparator,
                    threshold=threshold,
                    weight=weight,
                    clip_under_threshold=clip_under_threshold,
                    aggregate=aggregate,
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
                aggregate = comparison_config.get("aggregate", False)
                
                from .comparison_info import ComparableFieldConfig

                return ComparableFieldConfig(
                    comparator=comparator,
                    threshold=threshold,
                    weight=weight,
                    clip_under_threshold=clip_under_threshold,
                    aggregate=aggregate,
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
            
            # Handle Union types (like Optional[StructuredModel])
            if get_origin(annotation) is Union:
                union_args = get_args(annotation)
                # Check if it's a Union with NoneType (Optional pattern)
                none_type = type(None)
                if none_type in union_args:
                    # Look for StructuredModel in the remaining args
                    for arg in union_args:
                        if (
                            arg != none_type
                            and inspect.isclass(arg)
                            and issubclass(arg, StructuredModel)
                        ):
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
            
            if get_origin(annotation) is Union:
                union_args = get_args(annotation)
                none_type = type(None)
                for arg in union_args:
                    if (
                        arg != none_type
                        and inspect.isclass(arg)
                        and issubclass(arg, StructuredModel)
                    ):
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
            
            # Handle Optional[List[StructuredModel]] annotations (Union[List[StructuredModel], NoneType])
            elif get_origin(annotation) is Union:
                union_args = get_args(annotation)
                none_type = type(None)
                # Look for List[StructuredModel] within the Union
                for arg in union_args:
                    if arg != none_type and get_origin(arg) is list:
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
            
            # Handle Optional[List[StructuredModel]]
            elif get_origin(annotation) is Union:
                union_args = get_args(annotation)
                none_type = type(None)
                for arg in union_args:
                    if arg != none_type and get_origin(arg) is list:
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
        nested_instance = structured_class.from_json(nested_data)
        # Return the model_dump to get properly processed data
        return nested_instance.model_dump()
