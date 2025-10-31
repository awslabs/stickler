"""
Output validation tests for HTML reporting module.
Tests HTML structure, content accuracy, and file system validation.
"""

import pytest
import os
import tempfile
import shutil
from bs4 import BeautifulSoup
from unittest.mock import Mock

from stickler.reporting.html.html_reporter import EvaluationHTMLReporter
from stickler.reporting.html.report_config import ReportConfig
from stickler.utils.process_evaluation import ProcessEvaluation


class TestHTMLOutputValidation:
    """Test cases for validating HTML output structure and content."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reporter = EvaluationHTMLReporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_html_structure_validation(self):
        """Test that generated HTML has valid structure."""
        individual_results = {
            'overall': {'cm_f1': 0.85, 'cm_precision': 0.90, 'cm_recall': 0.80},
            'fields': {'name': {'cm_f1': 0.90}},
            'non_matches': []
        }
        
        output_path = os.path.join(self.temp_dir, "structure_test.html")
        
        result = self.reporter.generate_report(
            evaluation_results=individual_results,
            output_path=output_path,
            title="Structure Test Report"
        )
        
        assert result.success is True
        assert os.path.exists(output_path)
        
        # Parse HTML and validate structure
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Validate basic HTML structure
        assert soup.find('html') is not None
        assert soup.find('head') is not None
        assert soup.find('body') is not None
        assert soup.find('title') is not None
        assert soup.find('title').text == "Structure Test Report"
        
        # Validate required sections
        assert soup.find('header') is not None
        assert soup.find('main') is not None
        assert soup.find('footer') is not None
        
        # Validate container structure
        container = soup.find('div', class_='container')
        assert container is not None
        
        # Validate h1 title
        h1 = soup.find('h1')
        assert h1 is not None
        assert h1.text == "Structure Test Report"
    
    def test_section_content_validation(self):
        """Test that all configured sections are present with correct content."""
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
        
        output_path = os.path.join(self.temp_dir, "content_test.html")
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
            title="Content Test Report"
        )
        
        assert result.success is True
        
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Validate Executive Summary section
        exec_summary = soup.find('h2', string='Executive Summary')
        assert exec_summary is not None
        
        # Check for metrics in executive summary
        assert '0.850' in html_content  # F1 score
        assert '0.900' in html_content  # Precision
        assert '0.800' in html_content  # Recall
        assert '0.920' in html_content  # Accuracy
        
        # Validate Field Analysis section
        field_analysis = soup.find('h2', string='Field Performance Analysis')
        assert field_analysis is not None
        
        # Check for field names
        assert 'name' in html_content
        assert 'price' in html_content
        
        # Validate Confusion Matrix section
        confusion_matrix = soup.find('h2', string='Confusion Matrix')
        assert confusion_matrix is not None
        
        # Check for confusion matrix values
        assert '20' in html_content  # TP
        assert '15' in html_content  # TN
        assert '3' in html_content   # FP
        assert '7' in html_content   # FN
        
        # Validate Non-Matches section
        non_matches = soup.find('h2', string='Non-Matches Analysis')
        assert non_matches is not None
        
        # Check for non-match content
        assert 'category' in html_content
        assert 'EXTRA' in html_content
        assert 'Electronics' in html_content
    
    def test_css_javascript_integration(self):
        """Test that CSS and JavaScript are properly integrated."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        output_path = os.path.join(self.temp_dir, "integration_test.html")
        
        result = self.reporter.generate_report(
            evaluation_results=individual_results,
            output_path=output_path
        )
        
        assert result.success is True
        
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Validate CSS is included
        style_tag = soup.find('style')
        assert style_tag is not None
        
        # Validate PDF.js CDN script is included
        pdf_script = soup.find('script', src=lambda x: x and 'pdf.min.js' in x)
        assert pdf_script is not None
        
        # Validate worker configuration
        assert 'pdf.worker.min.js' in html_content
        assert 'GlobalWorkerOptions.workerSrc' in html_content
    
    def test_metadata_accuracy(self):
        """Test that report metadata is accurate."""
        mock_process_eval = Mock(spec=ProcessEvaluation)
        mock_process_eval.document_count = 25
        mock_process_eval.metrics = {'cm_f1': 0.87}
        mock_process_eval.field_metrics = {'name': {'cm_f1': 0.90}}
        mock_process_eval.non_matches = []
        
        output_path = os.path.join(self.temp_dir, "metadata_test.html")
        config = ReportConfig(
            include_executive_summary=True,
            include_field_analysis=False,
            include_confusion_matrix=False,
            include_non_matches=True
        )
        
        result = self.reporter.generate_report(
            evaluation_results=mock_process_eval,
            output_path=output_path,
            config=config,
            title="Metadata Test Report"
        )
        
        # Validate result metadata
        assert result.success is True
        assert result.metadata["is_bulk"] is True
        assert result.metadata["document_count"] == 25
        assert "executive_summary" in result.sections_included
        assert "non_matches" in result.sections_included
        assert "field_analysis" not in result.sections_included
        assert "confusion_matrix" not in result.sections_included
        
        # Validate HTML content reflects metadata
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert '25' in html_content  # Document count should be displayed
        assert 'Metadata Test Report' in html_content
    
    def test_document_gallery_validation(self):
        """Test document gallery HTML structure and content."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        
        # Create temporary document files
        doc1_path = os.path.join(self.temp_dir, "doc1.jpg")
        doc2_path = os.path.join(self.temp_dir, "doc2.pdf")
        
        with open(doc1_path, 'w') as f:
            f.write("fake image content")
        with open(doc2_path, 'w') as f:
            f.write("fake pdf content")
        
        document_files = {
            'doc1': doc1_path,
            'doc2': doc2_path
        }
        
        output_path = os.path.join(self.temp_dir, "gallery_test.html")
        config = ReportConfig(document_file_type='image')
        
        result = self.reporter.generate_report(
            evaluation_results=individual_results,
            output_path=output_path,
            config=config,
            document_files=document_files
        )
        
        assert result.success is True
        
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Validate document gallery section
        gallery_section = soup.find('h2', string='Document Gallery')
        assert gallery_section is not None
        
        # Validate gallery container
        gallery_div = soup.find('div', class_='document-gallery')
        assert gallery_div is not None
        
        # Validate image items
        image_items = soup.find_all('div', class_='image-item')
        assert len(image_items) == 2
        
        # Validate img tags
        img_tags = soup.find_all('img')
        assert len(img_tags) == 2
        
        # Check that images have correct src attributes
        img_srcs = [img.get('src') for img in img_tags]
        assert any('doc1.jpg' in src for src in img_srcs)
        assert any('doc2.pdf' in src for src in img_srcs)
    
    def test_pdf_gallery_validation(self):
        """Test PDF gallery HTML structure for PDF mode."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        
        # Create temporary PDF file
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        with open(pdf_path, 'w') as f:
            f.write("fake pdf content")
        
        document_files = {'test_doc': pdf_path}
        
        output_path = os.path.join(self.temp_dir, "pdf_gallery_test.html")
        config = ReportConfig(document_file_type='pdf')
        
        result = self.reporter.generate_report(
            evaluation_results=individual_results,
            output_path=output_path,
            config=config,
            document_files=document_files
        )
        
        assert result.success is True
        
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Validate PDF gallery section
        pdf_gallery = soup.find('h2', string='PDF Gallery')
        assert pdf_gallery is not None
        
        # Validate PDF item structure
        pdf_item = soup.find('div', class_='pdf-item')
        assert pdf_item is not None
        assert pdf_item.get('data-doc-id') == 'test_doc'
        assert 'test.pdf' in pdf_item.get('data-pdf-path')
        
        # Validate canvas for PDF rendering
        canvas = soup.find('canvas', class_='pdf-canvas')
        assert canvas is not None
        assert canvas.get('id') == 'pdf-canvas-test_doc'
        
        # Validate loading and error divs
        loading_div = soup.find('div', class_='pdf-loading')
        assert loading_div is not None
        assert loading_div.get('id') == 'pdf-loading-test_doc'
        
        error_div = soup.find('div', class_='pdf-error')
        assert error_div is not None
        assert error_div.get('id') == 'pdf-error-test_doc'


class TestFileSystemValidation:
    """Test cases for file system operations and validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reporter = EvaluationHTMLReporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_output_file_creation(self):
        """Test that output file is created with correct permissions."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        output_path = os.path.join(self.temp_dir, "file_creation_test.html")
        
        result = self.reporter.generate_report(
            evaluation_results=individual_results,
            output_path=output_path
        )
        
        assert result.success is True
        assert os.path.exists(output_path)
        assert os.path.isfile(output_path)
        
        # Check file is readable
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 0
            assert '<!DOCTYPE html>' in content
    
    def test_directory_creation(self):
        """Test that nested directories are created correctly."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        nested_path = os.path.join(self.temp_dir, "reports", "html", "test_report.html")
        
        result = self.reporter.generate_report(
            evaluation_results=individual_results,
            output_path=nested_path
        )
        
        assert result.success is True
        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))
    
    def test_document_file_copying(self):
        """Test that document files are copied to correct locations."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        
        # Create source document files
        source_dir = tempfile.mkdtemp()
        source_file1 = os.path.join(source_dir, "document1.pdf")
        source_file2 = os.path.join(source_dir, "document2.jpg")
        
        with open(source_file1, 'w') as f:
            f.write("PDF content")
        with open(source_file2, 'w') as f:
            f.write("Image content")
        
        document_files = {
            'doc1': source_file1,
            'doc2': source_file2
        }
        
        output_path = os.path.join(self.temp_dir, "copy_test.html")
        
        try:
            result = self.reporter.generate_report(
                evaluation_results=individual_results,
                output_path=output_path,
                document_files=document_files
            )
            
            assert result.success is True
            
            # Verify images directory was created
            images_dir = os.path.join(self.temp_dir, "images")
            assert os.path.exists(images_dir)
            assert os.path.isdir(images_dir)
            
            # Verify files were copied
            copied_file1 = os.path.join(images_dir, "document1.pdf")
            copied_file2 = os.path.join(images_dir, "document2.jpg")
            
            assert os.path.exists(copied_file1)
            assert os.path.exists(copied_file2)
            
            # Verify file contents
            with open(copied_file1, 'r') as f:
                assert f.read() == "PDF content"
            with open(copied_file2, 'r') as f:
                assert f.read() == "Image content"
                
        finally:
            shutil.rmtree(source_dir)
    
    def test_file_path_handling(self):
        """Test handling of various file path formats."""
        individual_results = {'overall': {'cm_f1': 0.85}}
        
        # Test with different path formats
        test_paths = [
            os.path.join(self.temp_dir, "simple.html"),
            os.path.join(self.temp_dir, "with spaces.html"),
            os.path.join(self.temp_dir, "with-dashes.html"),
            os.path.join(self.temp_dir, "with_underscores.html"),
        ]
        
        for test_path in test_paths:
            result = self.reporter.generate_report(
                evaluation_results=individual_results,
                output_path=test_path
            )
            
            assert result.success is True, f"Failed for path: {test_path}"
            assert os.path.exists(test_path), f"File not created: {test_path}"
            assert result.output_path == test_path
    
    def test_file_encoding_validation(self):
        """Test that files are created with correct UTF-8 encoding."""
        individual_results = {
            'overall': {'cm_f1': 0.85},
            'non_matches': [
                {
                    'field_path': 'description',
                    'non_match_type': 'MISMATCH',
                    'ground_truth_value': 'Café résumé naïve',  # Unicode characters
                    'prediction_value': 'Cafe resume naive'
                }
            ]
        }
        
        output_path = os.path.join(self.temp_dir, "encoding_test.html")
        
        result = self.reporter.generate_report(
            evaluation_results=individual_results,
            output_path=output_path,
            config=ReportConfig(include_non_matches=True)
        )
        
        assert result.success is True
        
        # Read file with explicit UTF-8 encoding
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Verify Unicode characters are preserved
        assert 'Café résumé naïve' in content
        assert 'charset="UTF-8"' in content
