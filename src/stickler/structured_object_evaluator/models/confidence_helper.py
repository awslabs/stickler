    
from pydantic import BaseModel
from typing import Any, Dict

class ConfidenceHelper(BaseModel):
    @staticmethod
    def _is_confidence_structure(data: Dict) -> bool:
        """Check if the data structure is a valid confidence structure.

        Args:
            data: The data to check

        Return:
            bool: If the data has confidence values
        """
        return (isinstance(data, dict) and "value" in data and "confidence" in data and len(data) == 2)
    
    @classmethod
    def process_confidence_structures(cls, data: Any, field_path: str = "") -> tuple:
        """Process confidence structures recursively.

        Args:
            data: The data to process
            field_path: The current field path for error messages

        Returns:
            tuple: value dictionary, confidence dictionary if the confidence values exists 
        """
        if isinstance(data, dict):
            # Check if this is a confidence structure
            if cls._is_confidence_structure(data):
                return data["value"], {field_path: data["confidence"]}
            else:
                # Process nested dictionary
                processed = {}
                all_confidences = {}
                for key, value in data.items():
                    new_path = f"{field_path}.{key}" if field_path else key
                    processed_value, confidences = cls.process_confidence_structures(value, new_path)
                    processed[key] = processed_value
                    all_confidences.update(confidences)
                return processed, all_confidences
        elif isinstance(data, list):
            # Process list items
            processed_list = []
            all_confidences = {}
            for i, item in enumerate(data):
                item_path = f"{field_path}[{i}]"
                processed_item, confidences = cls.process_confidence_structures(item, item_path)
                processed_list.append(processed_item)
                all_confidences.update(confidences)
            return processed_list, all_confidences
        else:
            # Primitive value - no confidence
            return data, {}