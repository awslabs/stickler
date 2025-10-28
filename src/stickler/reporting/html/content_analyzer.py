"""
Simple content analyzer for evaluation results - v0.
"""

from typing import Dict, Any, Union, Optional
from stickler.utils.process_evaluation import ProcessEvaluation
from stickler.reporting.html.utils import DataExtractor

class ContentAnalyzer:
    """
    Simple content analyzer for generating insights from evaluation results.
    """
    
    def __init__(self):
        """Initialize the content analyzer."""
        pass
    
    def analyze_results(self, results: Union[Dict[str, Any], ProcessEvaluation], is_bulk: bool = False, model_schema: Optional[type] = None) -> Dict[str, Any]:
        """
        Analyze evaluation results and generate insights.
        
        Args:
            results: Evaluation results (individual or bulk)
            is_bulk: Whether this is bulk results
            model_schema: StructuredModel class to extract field thresholds from
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract field thresholds from model schema or results
        field_thresholds = DataExtractor.extract_all_field_thresholds(model_schema) if model_schema else None
        
        analysis = {
            'executive_summary': self._generate_executive_summary(results, is_bulk, field_thresholds),
            'field_thresholds': field_thresholds
        }
        
        return analysis
    
    def _generate_executive_summary(self, results: Union[Dict, ProcessEvaluation]) -> Dict[str, Any]:
        """Generate executive summary data."""
        metrics = DataExtractor.extract_overall_metrics(results)
        if isinstance(results, ProcessEvaluation):
           doc_count = getattr(results, 'document_count', 1)
        
        key_metrics = {}
        metric_keys = ['cm_precision', 'cm_recall', 'cm_f1', 'cm_accuracy']
        for key in metric_keys:
            if key in metrics:
                key_metrics[key] = metrics[key]
        
        # Calculate overall performance score
        f1_score = metrics.get('cm_f1', 0)
        if isinstance(f1_score, (int, float)):
            overall_performance = f1_score
        else:
            overall_performance = 0.0
        
        
        return {
            'key_metrics': key_metrics,
            'overall_performance': overall_performance,
            'document_count': doc_count,
        }