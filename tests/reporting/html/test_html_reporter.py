"""
Tests for EvaluationHTMLReporter class with comprehensive error handling.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path

from stickler.reporting.html.html_reporter import EvaluationHTMLReporter
from stickler.reporting.html.report_config import ReportConfig, ReportResult
from stickler.utils.process_evaluation import ProcessEvaluation


class TestEvaluationHTMLReporter:
    """Test cases for EvaluationHTMLReporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reporter = EvaluationHTMLReporter()
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_report.html")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test EvaluationHTMLReporter initialization."""
        reporter = EvaluationHTMLReporter()
        assert reporter is not None
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_generate_report_success_individual_results(self, mock_makedirs, mock_file):
        """Test successful report generation with individual results."""
  
        individual_results = {
            'overall': {'cm_f1': 0.85, 'cm_precision': 0.90, 'cm_recall': 0.80},
            'fields': {'name': {'cm_f1': 0.90}},
            'non_matches': []
        }
        
        config = ReportConfig()
        
        with patch.object(self.reporter, '_generate_html_content') as mock_generate_html:
            mock_generate_html.return_value = '<html>Test Report</html>'
            
            result = self.reporter.generate_report(
                evaluation_results=individual_results,
                output_path=self.output_path,
                config=config,
                title="Test Report"
            )
        
        assert result.success is True
        assert result.output_path == self.output_path
        assert len(result.errors) == 0
        assert result.metadata["is_bulk"] is False
        assert result.metadata["document_count"] == 1
        
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once_with(self.output_path, 'w', encoding='utf-8')
        mock_generate_html.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_generate_report_success_bulk_results(self, mock_makedirs, mock_file):
        """Test successful report generation with ProcessEvaluation results."""
        
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.document_count = 5
        
        with patch.object(self.reporter, '_generate_html_content') as mock_generate_html:
            mock_generate_html.return_value = '<html>Bulk Report</html>'
            
            result = self.reporter.generate_report(
                evaluation_results=mock_process_eval,
                output_path=self.output_path
            )
        
        assert result.success is True
        assert result.metadata["is_bulk"] is True
        assert result.metadata["document_count"] == 5
    
    def test_generate_report_file_write_error(self):
        """Test report generation with file write permission error."""
        
        individual_results = {'overall': {'cm_f1': 0.85}}
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            result = self.reporter.generate_report(
                evaluation_results=individual_results,
                output_path="/invalid/path/report.html"
            )
        
        assert result.success is False
        assert len(result.errors) == 1
        assert "Permission denied" in result.errors[0]
    
    
    def test_generate_report_makedirs_error(self):
        """Test report generation with directory creation error."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        
        with patch('os.makedirs', side_effect=OSError("Cannot create directory")):
            result = self.reporter.generate_report(
                evaluation_results=individual_results,
                output_path="/invalid/deep/path/report.html"
            )
        
        assert result.success is False
        assert len(result.errors) == 1
        assert "Cannot create directory" in result.errors[0]
    
    def test_generate_report_invalid_evaluation_results(self):
        """Test report generation with invalid evaluation results."""
        result = self.reporter.generate_report(
            evaluation_results=None,
            output_path=self.output_path
        )
        
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_generate_report_empty_output_path(self):
        """Test report generation with empty output path."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        
        result = self.reporter.generate_report(
            evaluation_results=individual_results,
            output_path=""
        )
        
        assert result.success is False
        assert len(result.errors) > 0
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_generate_report_with_document_files(self, mock_makedirs, mock_file):
        """Test report generation with document files."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        document_files = {
            'doc1': '/path/to/doc1.pdf',
            'doc2': '/path/to/doc2.jpg'
        }
        
        with patch.object(self.reporter, '_copy_files_to_report_dir') as mock_copy_files:
            mock_copy_files.return_value = {
                'doc1': 'images/doc1.pdf',
                'doc2': 'images/doc2.jpg'
            }
            
            with patch.object(self.reporter, '_generate_html_content') as mock_generate_html:
                mock_generate_html.return_value = '<html>Report with docs</html>'
                
                result = self.reporter.generate_report(
                    evaluation_results=individual_results,
                    output_path=self.output_path,
                    document_files=document_files
                )
        
        assert result.success is True
        mock_copy_files.assert_called_once_with(document_files, self.output_path)
    
    def test_copy_files_to_report_dir_success(self):
        """Test successful file copying to report directory."""
        # Create temporary source files
        source_dir = tempfile.mkdtemp()
        source_file1 = os.path.join(source_dir, "doc1.pdf")
        source_file2 = os.path.join(source_dir, "doc2.jpg")
        
        with open(source_file1, 'w') as f:
            f.write("PDF content")
        with open(source_file2, 'w') as f:
            f.write("Image content")
        
        document_files = {
            'doc1': source_file1,
            'doc2': source_file2
        }
        
        try:
            result = self.reporter._copy_files_to_report_dir(document_files, self.output_path)
            
            assert 'doc1' in result
            assert 'doc2' in result
            assert result['doc1'] == 'images/doc1.pdf'
            assert result['doc2'] == 'images/doc2.jpg'
            
            # Verify files were actually copied
            images_dir = os.path.join(os.path.dirname(self.output_path), "images")
            assert os.path.exists(os.path.join(images_dir, "doc1.pdf"))
            assert os.path.exists(os.path.join(images_dir, "doc2.jpg"))
            
        finally:
            shutil.rmtree(source_dir)
    
    def test_copy_files_to_report_dir_missing_source(self):
        """Test file copying with missing source files."""
        document_files = {
            'doc1': '/nonexistent/path/doc1.pdf',
            'doc2': '/another/missing/doc2.jpg'
        }
        
        result = self.reporter._copy_files_to_report_dir(document_files, self.output_path)
        
        # Should return original paths as fallback
        assert result['doc1'] == '/nonexistent/path/doc1.pdf'
        assert result['doc2'] == '/another/missing/doc2.jpg'
    
    @patch('shutil.copy2', side_effect=OSError("Copy failed"))
    def test_copy_files_to_report_dir_copy_error(self, mock_copy):
        """Test file copying with copy operation error."""
        # Create temporary source file
        source_dir = tempfile.mkdtemp()
        source_file = os.path.join(source_dir, "doc1.pdf")
        
        with open(source_file, 'w') as f:
            f.write("PDF content")
        
        document_files = {'doc1': source_file}
        
        try:
            result = self.reporter._copy_files_to_report_dir(document_files, self.output_path)
            
            # Should return original path as fallback
            assert result['doc1'] == source_file
            
        finally:
            shutil.rmtree(source_dir)
    
    def test_get_sections_included(self):
        """Test sections included determination."""
        config = ReportConfig(
            include_executive_summary=True,
            include_field_analysis=False,
            include_confusion_matrix=True,
            include_non_matches=False
        )
        
        sections = self.reporter._get_sections_included(config)
        
        assert "executive_summary" in sections
        assert "confusion_matrix" in sections
        assert "field_analysis" not in sections
        assert "non_matches" not in sections
    
    def test_get_document_count_process_evaluation(self):
        """Test document count extraction from ProcessEvaluation."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.document_count = 42
        
        count = self.reporter._get_document_count(mock_process_eval)
        assert count == 42
    
    def test_get_document_count_individual_results(self):
        """Test document count extraction from individual results."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        
        count = self.reporter._get_document_count(individual_results)
        assert count == 1
    
    def test_generate_title_bulk_results(self):
        """Test title generation for bulk results."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.document_count = 15
        
        with patch.object(self.reporter, '_get_document_count', return_value=15):
            title = self.reporter._generate_title(mock_process_eval, True)
        
        assert title == "Evaluation Report - 15 Documents"
    
    def test_generate_title_individual_results(self):
        """Test title generation for individual results."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        
        title = self.reporter._generate_title(individual_results, False)
        assert title == "Evaluation Report"
    
    @patch('builtins.open', new_callable=mock_open, read_data="body { color: blue; }")
    def test_get_basic_css_success(self, mock_file):
        """Test successful CSS loading."""
        css_content = self.reporter._get_basic_css()
        
        assert css_content == "body { color: blue; }"
        mock_file.assert_called_once()
    
    @patch('builtins.open', side_effect=FileNotFoundError("CSS file not found"))
    def test_get_basic_css_file_not_found(self, mock_file):
        """Test CSS loading with missing file."""
        css_content = self.reporter._get_basic_css()
        
        assert css_content is None
    
    @patch('builtins.open', new_callable=mock_open, read_data="console.log('test');")
    def test_load_javascript_file_success(self, mock_file):
        """Test successful JavaScript loading."""
        js_content = self.reporter._load_javascript_file()
        
        assert js_content == "console.log('test');"
        mock_file.assert_called_once()
    
    @patch('builtins.open', side_effect=FileNotFoundError("JS file not found"))
    def test_load_javascript_file_not_found(self, mock_file):
        """Test JavaScript loading with missing file."""
        js_content = self.reporter._load_javascript_file()
        
        assert js_content == "// JavaScript file not found"
    
    def test_build_html_document_basic(self):
        """Test basic HTML document building."""
        sections = ['<div>Section 1</div>', '<div>Section 2</div>']
        title = "Test Report"
        
        with patch.object(self.reporter, '_get_basic_css', return_value="body { margin: 0; }"):
            html = self.reporter._build_html_document(sections, title)
        
        assert '<!DOCTYPE html>' in html
        assert '<title>Test Report</title>' in html
        assert '<h1>Test Report</h1>' in html
        assert '<div>Section 1</div>' in html
        assert '<div>Section 2</div>' in html
        assert 'body { margin: 0; }' in html
        assert 'Generated by Stickler' in html
    
    def test_build_html_document_with_individual_docs(self):
        """Test HTML document building with individual documents."""
        sections = ['<div>Section 1</div>']
        title = "Test Report"
        individual_docs = [{'doc_id': 'doc1', 'field': 'value'}]
        mock_schema = Mock()
        
        with patch.object(self.reporter, '_get_basic_css', return_value=""):
            with patch.object(self.reporter, '_get_javascript', return_value="<script>test();</script>"):
                html = self.reporter._build_html_document(sections, title, individual_docs, mock_schema)
        
        assert '<script>test();</script>' in html
    
    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_all_field_thresholds')
    def test_get_javascript_with_data(self, mock_extract_thresholds):
        """Test JavaScript generation with individual documents and schema."""
        individual_docs = [{'doc_id': 'doc1', 'metrics': {'f1': 0.85}}]
        mock_schema = Mock()
        mock_extract_thresholds.return_value = {'field1': 0.8, 'field2': 0.9}
        
        with patch.object(self.reporter, '_load_javascript_file', return_value="function test() {}"):
            js_content = self.reporter._get_javascript(individual_docs, mock_schema)
        
        assert 'function test() {}' in js_content
        assert 'initializeDocumentData(' in js_content
        assert '"doc_id": "doc1"' in js_content
        assert '"field1": 0.8' in js_content
        mock_extract_thresholds.assert_called_once_with(mock_schema)


class TestEvaluationHTMLReporterIntegration:
    """Integration tests for EvaluationHTMLReporter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reporter = EvaluationHTMLReporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_report_generation(self):
        """Test complete end-to-end report generation."""
        individual_results = {
            'overall': {
                'cm_f1': 0.85,
                'cm_precision': 0.90,
                'cm_recall': 0.80,
                'cm_accuracy': 0.92
            },
            'fields': {
                'name': {'cm_f1': 0.90, 'cm_precision': 0.95, 'cm_recall': 0.85},
                'price': {'cm_f1': 0.80, 'cm_precision': 0.85, 'cm_recall': 0.75}
            },
            'confusion_matrix': {
                'overall': {'tp': 20, 'tn': 15, 'fp': 3, 'fn': 7}
            },
            'non_matches': [
                {
                    'field_path': 'category',
                    'non_match_type': 'EXTRA',
                    'ground_truth_value': None,
                    'prediction_value': 'Electronics'
                }
            ]
        }
        
        output_path = os.path.join(self.temp_dir, "integration_test_report.html")
        config = ReportConfig(
            include_executive_summary=True,
            include_field_analysis=True,
            include_confusion_matrix=True,
            include_non_matches=True
        )
        
        result = self.reporter.generate_report(
            evaluation_results=individual_results,
            output_path=output_path,
            config=config,
            title="Integration Test Report"
        )
        
        # Verify result
        assert result.success is True
        assert result.output_path == output_path
        assert len(result.errors) == 0
        assert "executive_summary" in result.sections_included
        assert "field_analysis" in result.sections_included
        assert "confusion_matrix" in result.sections_included
        assert "non_matches" in result.sections_included
        
        # Verify file was created and contains expected content
        assert os.path.exists(output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert '<!DOCTYPE html>' in html_content
        assert 'Integration Test Report' in html_content
        assert 'Executive Summary' in html_content
        assert 'Field Performance Analysis' in html_content
        assert 'Confusion Matrix' in html_content
        assert 'Non-Matches Analysis' in html_content
        assert '0.850' in html_content  # F1 score
        assert 'name' in html_content
        assert 'price' in html_content
        assert 'category' in html_content
