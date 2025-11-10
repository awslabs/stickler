"""
Data extraction utilities for HTML reporting.
Centralizes data access patterns that were previously duplicated across multiple modules.
"""
import logging
from typing import Dict, Any, Union, List, Optional
from stickler.utils.process_evaluation import ProcessEvaluation

logger = logging.getLogger(__name__)

class DataExtractor:
    """Centralized data extraction utilities for consistent data access patterns."""
    
    @staticmethod
    def extract_field_metrics(results: Union[Dict[str, Any], ProcessEvaluation]) -> Dict[str, Any]:
        """
        Standardized field metrics extraction.
        
        Args:
            results: Evaluation results (individual or bulk)
            
        Returns:
            Dictionary of field metrics
        """
        if isinstance(results, ProcessEvaluation):
            return results.field_metrics or {}
        else:
            return results.get('fields', {})
    
    @staticmethod
    def extract_overall_metrics(results: Union[Dict[str, Any], ProcessEvaluation]) -> Dict[str, Any]:
        """
        Standardized overall metrics extraction.
        
        Args:
            results: Evaluation results (individual or bulk)
            
        Returns:
            Dictionary of overall metrics
        """
        if isinstance(results, ProcessEvaluation):
            return results.metrics or {}
        else:
            return results.get('overall', {})
    
    @staticmethod
    def extract_confusion_matrix(results: Union[Dict[str, Any], ProcessEvaluation]) -> Dict[str, Any]:
        """
        Standardized confusion matrix extraction.
        
        Args:
            results: Evaluation results (individual or bulk)
            
        Returns:
            Dictionary of confusion matrix data
        """
        if isinstance(results, ProcessEvaluation):
            return results.metrics or {}
        else:
            return results.get('confusion_matrix', {}).get('overall', {})
    
    @staticmethod
    def extract_non_matches(results: Union[Dict[str, Any], ProcessEvaluation]) -> List[Dict[str, Any]]:
        """
        Standardized non-matches extraction.
        
        Args:
            results: Evaluation results (individual or bulk)
            
        Returns:
            List of non-match dictionaries
        """
        if isinstance(results, ProcessEvaluation):
            return results.non_matches or []
        else:
            return results.get('non_matches', [])
    
    @staticmethod
    def get_document_count(results: Union[Dict[str, Any], ProcessEvaluation]) -> int:
        """
        Get document count from results.
        
        Args:
            results: Evaluation results (individual or bulk)
            
        Returns:
            Number of documents processed
        """
        if isinstance(results, ProcessEvaluation):
            return getattr(results, 'document_count', 1)
        else:
            # For individual results, count is always 1
            return 1
    
    @staticmethod
    def extract_similarity_score(results: Union[Dict[str, Any], ProcessEvaluation], field_name: Optional[str] = None) -> float:
        """
        Extract similarity score for overall or specific field.
        
        Args:
            results: Evaluation results
            field_name: Optional field name for field-specific score
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if field_name:
            field_metrics = DataExtractor.extract_field_metrics(results)
            field_data = field_metrics.get(field_name, {})
            return field_data.get('raw_similarity_score', field_data.get('similarity_score', 0.0))
        else:
            # Overall similarity
            if isinstance(results, ProcessEvaluation):
                overall_metrics = results.metrics or {}
                return overall_metrics.get('similarity_score', 0.0)
            else:
                return results.get('similarity_score', results.get('overall', {}).get('similarity_score', 0.0))
    
    
    @staticmethod
    def extract_field_threshold(model_schema: type, field_name: str) -> Optional[float]:
        """
        Extract field threshold from model schema.
        
        Args:
            model_schema: StructuredModel class
            field_name: Name of the field
            
        Returns:
            Threshold value or None if not found
        """
        if not model_schema or not hasattr(model_schema, '__fields__'):
            return None
        
        try:
            field_info = model_schema.__fields__.get(field_name)
            if not field_info:
                return None
            
            # Check json_schema_extra function
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if callable(json_schema_extra):
                # Check for threshold attribute
                if hasattr(json_schema_extra, '_threshold'):
                    threshold = getattr(json_schema_extra, '_threshold')
                    if isinstance(threshold, (int, float)):
                        return float(threshold)
                
                # Check for comparison metadata
                if hasattr(json_schema_extra, '_comparison_metadata'):
                    metadata = getattr(json_schema_extra, '_comparison_metadata', {})
                    if 'threshold' in metadata:
                        threshold = metadata['threshold']
                        if isinstance(threshold, (int, float)):
                            return float(threshold)
            
            # Check JSON schema directly
            elif isinstance(json_schema_extra, dict) and 'x-comparison' in json_schema_extra:
                comparison_info = json_schema_extra['x-comparison']
                if 'threshold' in comparison_info:
                    threshold = comparison_info['threshold']
                    if isinstance(threshold, (int, float)):
                        return float(threshold)
        
        except Exception as e:
            logging.warning(f"Error extracting threshold for field {field_name}: {e}")
            return None
        
        return None
    
    @staticmethod
    def extract_all_field_thresholds(model_schema: type) -> Dict[str, float]:
        """
        Extract all field thresholds from model schema.
        
        Args:
            model_schema: StructuredModel class
            
        Returns:
            Dictionary mapping field names to their thresholds
        """
        field_thresholds = {}
        
        if not model_schema or not hasattr(model_schema, '__fields__'):
            return field_thresholds
        
        try:
            for field_name in model_schema.__fields__:
                threshold = DataExtractor.extract_field_threshold(model_schema, field_name)
                if threshold is not None:
                    field_thresholds[field_name] = threshold
                
                # Handle nested fields
                field_info = model_schema.__fields__[field_name]
                field_type = getattr(field_info, 'annotation', None)
                
                # Check for nested StructuredModel
                if field_type and hasattr(field_type, '__fields__'):
                    nested_thresholds = DataExtractor.extract_all_field_thresholds(field_type)
                    for nested_field, nested_threshold in nested_thresholds.items():
                        field_thresholds[f"{field_name}.{nested_field}"] = nested_threshold
                
                # Check for List[StructuredModel]
                elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                    args = getattr(field_type, '__args__', ())
                    if args and hasattr(args[0], '__fields__'):
                        nested_thresholds = DataExtractor.extract_all_field_thresholds(args[0])
                        for nested_field, nested_threshold in nested_thresholds.items():
                            field_thresholds[f"{field_name}.{nested_field}"] = nested_threshold
        
        except Exception as e:
            logging.warning(f"Error extracting thresholds from model schema: {e}")
            return field_thresholds
        
        return field_thresholds
    
    @staticmethod
    def extract_document_ids(results: Union[Dict[str, Any], ProcessEvaluation]) -> List[str]:
        """
        Extract list of document IDs from results.
        
        Args:
            results: Evaluation results
            
        Returns:
            List of document IDs
        """
        if isinstance(results, ProcessEvaluation):
            # For bulk results, we'd need to extract from non_matches or other sources
            non_matches = results.non_matches or []
            doc_ids = set()
            for nm in non_matches:
                doc_id = nm.get('doc_id')
                if doc_id:
                    doc_ids.add(doc_id)
            return list(doc_ids)
        else:
            # For individual results, check if doc_id is present
            doc_id = results.get('doc_id', 'Unknown')
            return [doc_id]
