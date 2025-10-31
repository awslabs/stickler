"""
Tests for VisualizationEngine class.
"""

import pytest
from unittest.mock import Mock, patch
from stickler.reporting.html.visualization_engine import VisualizationEngine


class TestVisualizationEngine:
    """Test cases for VisualizationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.viz_engine = VisualizationEngine()
    
    def test_initialization(self):
        """Test VisualizationEngine initialization."""
        engine = VisualizationEngine()
        assert engine is not None
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_performance_gauge(self, mock_color_utils):
        """Test performance gauge generation."""
        mock_color_utils.return_value = "#28a745"
        
        result = self.viz_engine.generate_performance_gauge(0.85)
        
        assert '<div class="performance-gauge">' in result
        assert '<div class="gauge-circle"' in result
        assert '85%' in result
        assert 'Overall' in result
        assert '#28a745' in result
        mock_color_utils.assert_called_once_with(0.85)
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_performance_gauge_zero_score(self, mock_color_utils):
        """Test performance gauge with zero score."""
        mock_color_utils.return_value = "#dc3545"
        
        result = self.viz_engine.generate_performance_gauge(0.0)
        
        assert '0%' in result
        assert '#dc3545' in result
        mock_color_utils.assert_called_once_with(0.0)
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_performance_gauge_perfect_score(self, mock_color_utils):
        """Test performance gauge with perfect score."""
        mock_color_utils.return_value = "#28a745"
        
        result = self.viz_engine.generate_performance_gauge(1.0)
        
        assert '100%' in result
        assert '#28a745' in result
        mock_color_utils.assert_called_once_with(1.0)
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_field_performance_chart(self, mock_color_utils):
        """Test field performance chart generation."""
        mock_color_utils.return_value = "#ffc107"
        
        field_metrics = {
            "name": {"cm_f1": 0.85, "cm_precision": 0.90, "cm_recall": 0.80},
            "price": {"cm_f1": 0.95, "cm_precision": 0.98, "cm_recall": 0.92}
        }
        
        result = self.viz_engine.generate_field_performance_chart(field_metrics)
        
        assert '<div class="field-chart">' in result
        assert '<h4 style="margin-bottom: 15px; color: #495057; font-size: 1.1em;">F1 Score</h4>' in result
        assert 'name' in result
        assert 'price' in result
        assert '0.850' in result
        assert '0.950' in result
        assert 'width: 85%' in result
        assert 'width: 95%' in result
        assert mock_color_utils.call_count == 2
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_field_performance_chart_with_f1_fallback(self, mock_color_utils):
        """Test field performance chart with f1 fallback key."""
        mock_color_utils.return_value = "#17a2b8"
        
        field_metrics = {
            "category": {"f1": 0.75, "precision": 0.80, "recall": 0.70}  # Uses 'f1' instead of 'cm_f1'
        }
        
        result = self.viz_engine.generate_field_performance_chart(field_metrics)
        
        assert 'category' in result
        assert '0.750' in result
        assert 'width: 75%' in result
        mock_color_utils.assert_called_once_with(0.75)
    
    def test_generate_field_performance_chart_empty_metrics(self):
        """Test field performance chart with empty metrics."""
        field_metrics = {}
        
        result = self.viz_engine.generate_field_performance_chart(field_metrics)
        
        assert '<div class="field-chart">' in result
        assert '<h4 style="margin-bottom: 15px; color: #495057; font-size: 1.1em;">F1 Score</h4>' in result
        assert '</div>' in result
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_performance_color')
    def test_generate_field_performance_table(self, mock_color_utils):
        """Test field performance table generation."""
        mock_color_utils.return_value = "#28a745"
        
        field_metrics = {
            "name": {
                "cm_precision": 0.90,
                "cm_recall": 0.80,
                "cm_f1": 0.85,
                "tp": 45,
                "fd": 3,
                "fa": 2,
                "fn": 5
            },
            "price": {
                "cm_precision": 0.98,
                "cm_recall": 0.92,
                "cm_f1": 0.95,
                "tp": 50,
                "fd": 1,
                "fa": 1,
                "fn": 3
            }
        }
        
        result = self.viz_engine.generate_field_performance_table(field_metrics)
        
        assert '<table class="data-table data-table-numeric" id="performance-table">' in result
        assert '<th>Field</th>' in result
        assert '<th>Precision</th>' in result
        assert '<th>Recall</th>' in result
        assert '<th>F1 Score</th>' in result
        assert '<th>TP</th>' in result
        assert '<th>FD</th>' in result
        assert '<th>FA</th>' in result
        assert '<th>FN</th>' in result
        
        # Check data rows
        assert 'name' in result
        assert 'price' in result
        assert '0.900' in result  # precision
        assert '0.800' in result  # recall
        assert '0.850' in result  # f1
        assert '45' in result     # tp
        assert 'background-color: #28a745' in result
        assert 'color: white' in result
        assert 'font-weight: bold' in result
        
        assert mock_color_utils.call_count == 2
    
    def test_generate_field_performance_table_with_fallback_keys(self):
        """Test field performance table with fallback metric keys."""
        field_metrics = {
            "category": {
                "precision": 0.85,  # No 'cm_' prefix
                "recall": 0.78,
                "f1": 0.81,
                "tp": 20,
                "fd": 2,
                "fa": 3,
                "fn": 4
            }
        }
        
        result = self.viz_engine.generate_field_performance_table(field_metrics)
        
        assert 'category' in result
        assert '0.850' in result
        assert '0.780' in result
        assert '0.810' in result
        assert '20' in result
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_confusion_matrix_colors')
    def test_generate_confusion_matrix_heatmap(self, mock_color_utils):
        """Test confusion matrix heatmap generation."""
        mock_color_utils.return_value = {
            'tp': '#28a745',
            'tn': '#17a2b8',
            'fd': '#ffc107',
            'fa': '#fd7e14',
            'fn': '#dc3545'
        }
        
        cm_data = {
            'tp': 45,
            'tn': 30,
            'fd': 5,
            'fa': 3,
            'fn': 7
        }
        
        result = self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        assert '<div class="cm-grid">' in result
        assert 'TP' in result
        assert 'TN' in result
        assert 'FD' in result
        assert 'FA' in result
        assert 'FN' in result
        assert '45' in result
        assert '30' in result
        assert '50.0%' in result  # 45/90 * 100
        assert '33.3%' in result  # 30/90 * 100
        assert 'border-left-color: #28a745' in result
        mock_color_utils.assert_called_once()
    
    def test_generate_confusion_matrix_heatmap_empty_data(self):
        """Test confusion matrix heatmap with empty data."""
        cm_data = {}
        
        result = self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        assert '<p>No confusion matrix data to visualize.</p>' in result
    
    def test_generate_confusion_matrix_heatmap_zero_total(self):
        """Test confusion matrix heatmap with all zero values."""
        cm_data = {
            'tp': 0,
            'tn': 0,
            'fd': 0,
            'fa': 0,
            'fn': 0
        }
        
        result = self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        assert '<p>No confusion matrix data to visualize.</p>' in result
    
    @patch('stickler.reporting.html.utils.ColorUtils.get_confusion_matrix_colors')
    def test_generate_confusion_matrix_heatmap_missing_metrics(self, mock_color_utils):
        """Test confusion matrix heatmap with missing metrics."""
        mock_color_utils.return_value = {
            'tp': '#28a745',
            'tn': '#17a2b8',
            'fd': '#ffc107',
            'fa': '#fd7e14',
            'fn': '#dc3545'
        }
        
        cm_data = {
            'tp': 20,
            'tn': 15
            # Missing fd, fa, fn
        }
        
        result = self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        assert '<div class="cm-grid">' in result
        assert 'TP' in result
        assert 'TN' in result
        assert 'FD' in result  # Should still appear with 0 value
        assert '20' in result
        assert '15' in result
        assert '0' in result  # Missing values should default to 0
        assert '57.1%' in result  # 20/35 * 100
        assert '42.9%' in result  # 15/35 * 100
        assert '0.0%' in result   # 0/35 * 100
