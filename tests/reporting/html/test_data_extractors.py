"""
Tests for DataExtractor utility class.
"""

import pytest
from unittest.mock import Mock, MagicMock
from stickler.reporting.html.utils.data_extractors import DataExtractor
from stickler.utils.process_evaluation import ProcessEvaluation


class TestDataExtractor:
    """Test cases for DataExtractor class."""
    
    def test_extract_field_metrics_bulk_results(self):
        """Test field metrics extraction from ProcessEvaluation."""
        # Create mock ProcessEvaluation
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.field_metrics = {
            "name": {"cm_f1": 0.85, "cm_precision": 0.90, "cm_recall": 0.80},
            "price": {"cm_f1": 0.95, "cm_precision": 0.98, "cm_recall": 0.92}
        }
        
        result = DataExtractor.extract_field_metrics(mock_process_eval)
        
        assert result == mock_process_eval.field_metrics
        assert "name" in result
        assert "price" in result
        assert result["name"]["cm_f1"] == 0.85
    
    def test_extract_field_metrics_bulk_results_none(self):
        """Test field metrics extraction when field_metrics is None."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.field_metrics = None
        
        result = DataExtractor.extract_field_metrics(mock_process_eval)
        
        assert result == {}
    
    def test_extract_field_metrics_individual_results(self):
        """Test field metrics extraction from individual results dict."""
        individual_results = {
            "fields": {
                "name": {"f1": 0.75, "precision": 0.80, "recall": 0.70},
                "category": {"f1": 0.88, "precision": 0.85, "recall": 0.91}
            }
        }
        
        result = DataExtractor.extract_field_metrics(individual_results)
        
        assert result == individual_results["fields"]
        assert "name" in result
        assert "category" in result
    
    def test_extract_field_metrics_individual_results_missing(self):
        """Test field metrics extraction when fields key is missing."""
        individual_results = {"overall": {"f1": 0.85}}
        
        result = DataExtractor.extract_field_metrics(individual_results)
        
        assert result == {}
    
    def test_extract_overall_metrics_bulk_results(self):
        """Test overall metrics extraction from ProcessEvaluation."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.metrics = {
            "cm_f1": 0.87,
            "cm_precision": 0.89,
            "cm_recall": 0.85,
            "cm_accuracy": 0.92
        }
        
        result = DataExtractor.extract_overall_metrics(mock_process_eval)
        
        assert result == mock_process_eval.metrics
        assert result["cm_f1"] == 0.87
    
    def test_extract_overall_metrics_bulk_results_none(self):
        """Test overall metrics extraction when metrics is None."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.metrics = None
        
        result = DataExtractor.extract_overall_metrics(mock_process_eval)
        
        assert result == {}
    
    def test_extract_overall_metrics_individual_results(self):
        """Test overall metrics extraction from individual results dict."""
        individual_results = {
            "overall": {
                "f1": 0.82,
                "precision": 0.85,
                "recall": 0.79
            }
        }
        
        result = DataExtractor.extract_overall_metrics(individual_results)
        
        assert result == individual_results["overall"]
        assert result["f1"] == 0.82
    
    def test_extract_confusion_matrix_bulk_results(self):
        """Test confusion matrix extraction from ProcessEvaluation."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.metrics = {
            "tp": 45,
            "tn": 30,
            "fp": 5,
            "fn": 10,
            "fd": 3,
            "fa": 2
        }
        
        result = DataExtractor.extract_confusion_matrix(mock_process_eval)
        
        assert result == mock_process_eval.metrics
        assert result["tp"] == 45
        assert result["tn"] == 30
    
    def test_extract_confusion_matrix_individual_results(self):
        """Test confusion matrix extraction from individual results dict."""
        individual_results = {
            "confusion_matrix": {
                "overall": {
                    "tp": 20,
                    "tn": 15,
                    "fp": 3,
                    "fn": 7
                }
            }
        }
        
        result = DataExtractor.extract_confusion_matrix(individual_results)
        
        assert result == individual_results["confusion_matrix"]["overall"]
        assert result["tp"] == 20
    
    def test_extract_confusion_matrix_missing_data(self):
        """Test confusion matrix extraction with missing data."""
        individual_results = {"overall": {"f1": 0.85}}
        
        result = DataExtractor.extract_confusion_matrix(individual_results)
        
        assert result == {}
    
    def test_extract_non_matches_bulk_results(self):
        """Test non-matches extraction from ProcessEvaluation."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.non_matches = [
            {
                "doc_id": "doc1",
                "field_path": "name",
                "non_match_type": "MISMATCH",
                "ground_truth_value": "John",
                "prediction_value": "Jon"
            },
            {
                "doc_id": "doc2",
                "field_path": "price",
                "non_match_type": "MISSING",
                "ground_truth_value": "25.99",
                "prediction_value": None
            }
        ]
        
        result = DataExtractor.extract_non_matches(mock_process_eval)
        
        assert result == mock_process_eval.non_matches
        assert len(result) == 2
        assert result[0]["doc_id"] == "doc1"
    
    def test_extract_non_matches_bulk_results_none(self):
        """Test non-matches extraction when non_matches is None."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.non_matches = None
        
        result = DataExtractor.extract_non_matches(mock_process_eval)
        
        assert result == []
    
    def test_extract_non_matches_individual_results(self):
        """Test non-matches extraction from individual results dict."""
        individual_results = {
            "non_matches": [
                {
                    "field_path": "category",
                    "non_match_type": "EXTRA",
                    "ground_truth_value": None,
                    "prediction_value": "Electronics"
                }
            ]
        }
        
        result = DataExtractor.extract_non_matches(individual_results)
        
        assert result == individual_results["non_matches"]
        assert len(result) == 1
        assert result[0]["field_path"] == "category"
    
    def test_get_document_count_bulk_results(self):
        """Test document count extraction from ProcessEvaluation."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.document_count = 25
        
        result = DataExtractor.get_document_count(mock_process_eval)
        
        assert result == 25
    
    def test_get_document_count_bulk_results_missing_attr(self):
        """Test document count extraction when attribute is missing."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        # Don't set document_count attribute
        
        result = DataExtractor.get_document_count(mock_process_eval)
        
        assert result == 1  # Default value
    
    def test_get_document_count_individual_results(self):
        """Test document count extraction from individual results dict."""
        individual_results = {"overall": {"f1": 0.85}}
        
        result = DataExtractor.get_document_count(individual_results)
        
        assert result == 1
    
    def test_extract_similarity_score_overall_bulk(self):
        """Test overall similarity score extraction from ProcessEvaluation."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.metrics = {"similarity_score": 0.78}
        
        result = DataExtractor.extract_similarity_score(mock_process_eval)
        
        assert result == 0.78
    
    def test_extract_similarity_score_field_specific(self):
        """Test field-specific similarity score extraction."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.field_metrics = {
            "name": {"raw_similarity_score": 0.85, "similarity_score": 0.80}
        }
        
        result = DataExtractor.extract_similarity_score(mock_process_eval, "name")
        
        assert result == 0.85  # Should prefer raw_similarity_score
    
    def test_extract_similarity_score_field_fallback(self):
        """Test field-specific similarity score with fallback."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.field_metrics = {
            "name": {"similarity_score": 0.80}  # No raw_similarity_score
        }
        
        result = DataExtractor.extract_similarity_score(mock_process_eval, "name")
        
        assert result == 0.80
    
    def test_extract_similarity_score_missing_field(self):
        """Test similarity score extraction for non-existent field."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.field_metrics = {"name": {"similarity_score": 0.80}}
        
        result = DataExtractor.extract_similarity_score(mock_process_eval, "missing_field")
        
        assert result == 0.0
    
    def test_extract_document_ids_bulk_results(self):
        """Test document IDs extraction from ProcessEvaluation."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.non_matches = [
            {"doc_id": "doc1", "field_path": "name"},
            {"doc_id": "doc2", "field_path": "price"},
            {"doc_id": "doc1", "field_path": "category"},  # Duplicate
            {"field_path": "other"}  # Missing doc_id
        ]
        
        result = DataExtractor.extract_document_ids(mock_process_eval)
        
        assert set(result) == {"doc1", "doc2"}  # Should deduplicate
        assert len(result) == 2
    
    def test_extract_document_ids_individual_results(self):
        """Test document IDs extraction from individual results dict."""
        individual_results = {"doc_id": "single_doc"}
        
        result = DataExtractor.extract_document_ids(individual_results)
        
        assert result == ["single_doc"]
    
    def test_extract_document_ids_individual_results_missing(self):
        """Test document IDs extraction when doc_id is missing."""
        individual_results = {"overall": {"f1": 0.85}}
        
        result = DataExtractor.extract_document_ids(individual_results)
        
        assert result == ["Unknown"]
