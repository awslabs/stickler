"""
Data extraction utilities for HTML reporting.
Centralizes data access patterns that were previously duplicated across multiple modules.
"""
import logging
import types
from typing import Any, Dict, List, Mapping, Optional, Union, get_args, get_origin

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
    def extract_non_matches(
        results: Union[Dict[str, Any], ProcessEvaluation],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Standardized non-matches extraction.

        Args:
            results: Evaluation results (individual or bulk)

        Returns:
            List of non-match dictionaries, or None when the results carry no
            records at all. `compare_with()` only writes the `non_matches` key
            when called with `document_non_matches=True`, and
            `ProcessEvaluation.non_matches` is None when the bulk evaluator was
            built with `document_non_matches=False`, so an empty list and no
            list mean different things and only the first one means the
            document was clean.
        """
        if isinstance(results, ProcessEvaluation):
            return results.non_matches
        else:
            return results.get('non_matches')
    
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

                # Unwrap Optional[X] / Union[X, None] / X | None so nested-model
                # and List[...] detection below also fires for non-required
                # fields, which are annotated Optional[T] (issue #149). Without
                # this, nested thresholds vanish from HTML reports for models
                # whose nested object / list fields are optional.
                #
                # A PEP 604 union (`X | None`) has origin types.UnionType rather
                # than typing.Union, so checking Union alone silently skipped
                # that spelling -- the same bug, left half-fixed, and more
                # reachable since 0.7.0 raised the floor to Python 3.10 where
                # `X | None` is idiomatic (issue #162). Python 3.14 unifies the
                # two, making the second arm redundant there but still required
                # for 3.10-3.13.
                #
                # stickler.auto.inference.unwrap_optional already handles both
                # forms and is the reference implementation. It is deliberately
                # not imported here: that would make stickler.reporting depend
                # on the inference machinery and pull pydantic model building
                # into the report path. The four lines are mirrored instead.
                #
                # A third copy lives at
                # stickler.structured_object_evaluator.models.optional_annotation,
                # added once ten sites in that package were found with this same
                # bug. Keep all three in sync; unifying them is a 1.0 item.
                if get_origin(field_type) in (Union, types.UnionType):
                    non_none_args = [
                        arg for arg in get_args(field_type) if arg is not type(None)
                    ]
                    if len(non_none_args) == 1:
                        field_type = non_none_args[0]

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

    @staticmethod
    def extract_decision_metrics(results: Any) -> Optional[Dict[str, Any]]:
        """Return ``results.decision_metrics`` if it holds decision fields, else None.

        Only a ``ProcessEvaluation`` carries decision metrics. Anything else,
        including an empty or malformed value, gives None so the report renders
        exactly as it does without them.
        """
        if not isinstance(results, ProcessEvaluation):
            return None
        dm = getattr(results, "decision_metrics", None)
        if not isinstance(dm, Mapping):
            return None
        fields = dm.get("fields")
        if not isinstance(fields, Mapping) or not fields:
            return None
        return dm

    @staticmethod
    def extract_risk_coverage(
        decision_metrics: Mapping[str, Any], max_points: int = 400
    ) -> Dict[str, Dict[str, Any]]:
        """Per-field risk-coverage curve over every distinct confidence threshold.

        A threshold accepts the decisions whose confidence is at least it.
        Coverage is accepted / asked, so missing answers (always escalated)
        keep the curve from reaching 1.0. Risk is 1 - accuracy of the accepted.
        AURC comes from ``accept_or_escalate`` so the report and the library
        agree. Curves longer than ``max_points`` are thinned evenly, keeping
        both ends.

        Returns:
            ``{field: {"points": [(coverage, risk, tau, accepted), ...],
            "aurc": float | None, "n_total": int, "n_missing": int}}``
        """
        import numpy as np

        from stickler.decisions.escalate import accept_or_escalate, arrays_from_result

        arrays = arrays_from_result(decision_metrics)
        report = accept_or_escalate(decision_metrics, taus=())
        out: Dict[str, Dict[str, Any]] = {}
        for name, a in arrays.items():
            answered = a.answered
            conf = a.conf[answered]
            correct = a.correct[answered].astype(float)
            n_total = a.n_total
            points = []
            if conf.size and n_total:
                order = np.argsort(-conf, kind="stable")
                c, ok = conf[order], correct[order]
                ends = np.flatnonzero(np.r_[c[1:] != c[:-1], True]) + 1
                acc = np.cumsum(ok)[ends - 1] / ends
                for end, accuracy in zip(ends, acc):
                    points.append((
                        float(end) / n_total,
                        1.0 - float(accuracy),
                        float(c[end - 1]),
                        int(end),
                    ))
                if len(points) > max_points:
                    idx = np.unique(np.linspace(0, len(points) - 1, max_points).round().astype(int))
                    points = [points[i] for i in idx]
            out[name] = {
                "points": points,
                "aurc": report.fields[name]["aurc"],
                "n_total": int(n_total),
                "n_missing": int(a.n_missing),
            }
        return out
