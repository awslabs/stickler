"""
Tests for SectionGenerator class.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from stickler.reporting.html.report_config import ReportConfig
from stickler.reporting.html.section_generator import SectionGenerator
from stickler.reporting.html.visualization_engine import VisualizationEngine
from stickler.utils.process_evaluation import ProcessEvaluation


class TestSectionGenerator:
    """Test cases for SectionGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_viz_engine = Mock(spec=VisualizationEngine)
        self.mock_results = Mock(spec=ProcessEvaluation)
        self.section_generator = SectionGenerator(self.mock_results, self.mock_viz_engine)
    
    def test_initialization(self):
        """Test SectionGenerator initialization."""
        assert self.section_generator.results == self.mock_results
        assert self.section_generator.viz_engine == self.mock_viz_engine
    
    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_overall_metrics')
    def test_generate_executive_summary(self, mock_extract_metrics):
        """Test executive summary generation."""
        # Mock data
        mock_extract_metrics.return_value = {
            'cm_f1': 0.85,
            'cm_precision': 0.90,
            'cm_recall': 0.80,
            'cm_accuracy': 0.92
        }
        self.mock_results.document_count = 5
        self.mock_viz_engine.generate_performance_gauge.return_value = '<div class="gauge">85%</div>'
        
        config = ReportConfig()
        result = self.section_generator.generate_executive_summary(config)
        
        assert '<div class="section">' in result
        assert '<h2>Executive Summary</h2>' in result
        assert '<div class="summary-grid">' in result
        assert '<div class="metric-value">5</div>' in result
        assert '<div class="metric-label">Documents</div>' in result
        assert '0.850' in result  # F1 score
        assert '0.900' in result  # Precision
        assert '0.800' in result  # Recall
        assert '0.920' in result  # Accuracy
        assert '<div class="gauge">85%</div>' in result
        
        mock_extract_metrics.assert_called_once_with(self.mock_results)
        self.mock_viz_engine.generate_performance_gauge.assert_called_once_with(0.85, config)
    
    
    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_overall_metrics')
    def test_generate_executive_summary_missing_document_count(self, mock_extract_metrics):
        """Test executive summary generation with missing document count."""
        mock_extract_metrics.return_value = {'cm_f1': 0.75}
        # Don't set document_count attribute
        
        config = ReportConfig()
        result = self.section_generator.generate_executive_summary(config)
        
        assert '<div class="metric-value">1</div>' in result  # Default value

    def test_executive_summary_escapes_the_document_count(self):
        """A document_count assigned after construction is not validated, so it is escaped."""
        results = ProcessEvaluation(document_count=3, metrics={})
        results.document_count = '<img src=x onerror=alert(1)>'
        generator = SectionGenerator(results, self.mock_viz_engine)

        result = generator.generate_executive_summary(ReportConfig())

        assert '<img src=x onerror=alert(1)>' not in result
        assert (
            '<div class="metric-value">&lt;img src=x onerror=alert(1)&gt;</div>'
            in result
        )

    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_field_metrics')
    def test_generate_field_analysis(self, mock_extract_field_metrics):
        """Test field analysis generation."""
        mock_field_metrics = {
            "name": {"cm_f1": 0.85, "cm_precision": 0.90, "cm_recall": 0.80},
            "price": {"cm_f1": 0.95, "cm_precision": 0.98, "cm_recall": 0.92}
        }
        mock_extract_field_metrics.return_value = mock_field_metrics
        
        self.mock_viz_engine.generate_field_performance_chart.return_value = '<div class="chart">Chart</div>'
        self.mock_viz_engine.generate_field_performance_table.return_value = '<table>Table</table>'
        
        config = ReportConfig()
        result = self.section_generator.generate_field_analysis(config)
        
        assert '<div class="section"><h2>Field Performance Analysis</h2>' in result
        assert '<div class="chart">Chart</div>' in result
        assert '<table>Table</table>' in result
        
        mock_extract_field_metrics.assert_called_once_with(self.mock_results)
        self.mock_viz_engine.generate_field_performance_chart.assert_called_once_with(mock_field_metrics, config)
        self.mock_viz_engine.generate_field_performance_table.assert_called_once_with(mock_field_metrics, config)
    
    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_field_metrics')
    def test_generate_field_analysis_no_data(self, mock_extract_field_metrics):
        """Test field analysis generation with no field data."""
        mock_extract_field_metrics.return_value = {}
        
        config = ReportConfig()
        result = self.section_generator.generate_field_analysis(config)
        
        assert '<div class="section"><h2>Field Performance Analysis</h2>' in result
        assert '<p>No field data available.</p></div>' in result
        
        # Should not call visualization methods
        self.mock_viz_engine.generate_field_performance_chart.assert_not_called()
        self.mock_viz_engine.generate_field_performance_table.assert_not_called()
    
    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_confusion_matrix')
    def test_generate_confusion_matrix(self, mock_extract_cm):
        """Test confusion matrix generation."""
        mock_cm_data = {
            'tp': 45,
            'tn': 30,
            'fp': 5,
            'fn': 10
        }
        mock_extract_cm.return_value = mock_cm_data
        
        self.mock_viz_engine.generate_confusion_matrix_heatmap.return_value = '<div class="heatmap">Heatmap</div>'
        
        result = self.section_generator.generate_confusion_matrix()
        
        assert '<div class="section"><h2>Confusion Matrix</h2>' in result
        assert '<div class="heatmap">Heatmap</div>' in result
        assert '</div>' in result
        
        mock_extract_cm.assert_called_once_with(self.mock_results)
        self.mock_viz_engine.generate_confusion_matrix_heatmap.assert_called_once_with(mock_cm_data, {})
    
    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_confusion_matrix')
    def test_generate_confusion_matrix_no_data(self, mock_extract_cm):
        """Test confusion matrix generation with no data."""
        mock_extract_cm.return_value = {}
        
        result = self.section_generator.generate_confusion_matrix()
        
        assert '<div class="section"><h2>Confusion Matrix</h2>' in result
        assert '<p>No confusion matrix data available.</p></div>' in result
        
        # Should not call visualization method
        self.mock_viz_engine.generate_confusion_matrix_heatmap.assert_not_called()
    
    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_non_matches')
    def test_generate_non_matches(self, mock_extract_non_matches):
        """Test non-matches generation."""
        mock_non_matches = [
            {
                'doc_id': 'doc1',
                'field_path': 'name',
                'non_match_type': 'MISMATCH',
                'ground_truth_value': 'John',
                'prediction_value': 'Jon'
            },
            {
                'doc_id': 'doc2',
                'field_path': 'price',
                'non_match_type': 'MISSING',
                'ground_truth_value': '25.99',
                'prediction_value': None
            }
        ]
        mock_extract_non_matches.return_value = mock_non_matches
        
        config = ReportConfig(max_non_matches_displayed=100)
        
        result = self.section_generator.generate_non_matches(config)
        
        assert '<div class="section"><h2>Non-Matches Analysis</h2>' in result
        assert 'Found 2 non-matches.' in result
        assert '<table class="data-table" id="non-matches-table">' in result
        assert '<th>Document</th>' in result
        assert '<th>Field</th>' in result
        assert '<th>Type</th>' in result
        assert '<th>Ground Truth</th>' in result
        assert '<th>Prediction</th>' in result
        assert 'doc1' in result
        assert 'name' in result
        assert 'MISMATCH' in result
        assert 'John' in result
        assert 'Jon' in result
        assert 'doc2' in result
        assert 'price' in result
        assert 'MISSING' in result
        assert '25.99' in result
        
        mock_extract_non_matches.assert_called_once_with(self.mock_results)
    
    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_non_matches')
    def test_generate_non_matches_no_data(self, mock_extract_non_matches):
        """Records were collected and there were none: that is a success."""
        mock_extract_non_matches.return_value = []

        config = ReportConfig()

        result = self.section_generator.generate_non_matches(config)

        assert '<div class="section"><h2>Non-Matches Analysis</h2>' in result
        assert '<p>No non-matches found.</p></div>' in result

    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_non_matches')
    def test_generate_non_matches_without_records_is_not_a_success(
        self, mock_extract_non_matches
    ):
        """No records collected says nothing about whether the fields matched (#411)."""
        mock_extract_non_matches.return_value = None

        config = ReportConfig()

        result = self.section_generator.generate_non_matches(config)

        assert 'No non-matches found' not in result
        assert 'document_non_matches=True' in result

    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_non_matches')
    def test_generate_non_matches_limit_displayed(self, mock_extract_non_matches):
        """Test non-matches generation with display limit."""
        # Create more non-matches than the limit
        mock_non_matches = []
        for i in range(150):
            mock_non_matches.append({
                'doc_id': f'doc{i}',
                'field_path': 'field',
                'non_match_type': 'MISMATCH',
                'ground_truth_value': f'value{i}',
                'prediction_value': f'pred{i}'
            })
        
        mock_extract_non_matches.return_value = mock_non_matches
        
        config = ReportConfig(max_non_matches_displayed=50)
        
        result = self.section_generator.generate_non_matches(config)
        
        assert 'Found 150 non-matches.' in result
        assert 'Showing 50 of 150 non-matches.' in result
        # Should only show first 50
        assert 'doc0' in result
        assert 'doc49' in result
        assert 'doc50' not in result
    
    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_non_matches')
    def test_generate_non_matches_truncate_long_values(self, mock_extract_non_matches):
        """Test non-matches generation with long values that get truncated."""
        long_value = "x" * 150  # Longer than 100 characters
        mock_non_matches = [
            {
                'doc_id': 'doc1',
                'field_path': 'description',
                'non_match_type': 'MISMATCH',
                'ground_truth_value': long_value,
                'prediction_value': 'short'
            }
        ]
        mock_extract_non_matches.return_value = mock_non_matches
        
        config = ReportConfig()
        
        result = self.section_generator.generate_non_matches(config)
        
        # Should truncate to 100 characters
        truncated_value = long_value[:100]
        assert truncated_value in result
        assert long_value not in result  # Full value should not be present

    @patch('stickler.reporting.html.utils.data_extractors.DataExtractor.extract_non_matches')
    def test_generate_non_matches_truncation_never_splits_an_entity(self, mock_extract_non_matches):
        """Truncate first, then escape, so the cut cannot land inside ``&amp;``.

        The fixture is 98 raw characters because ``section_generator`` truncates
        at 100: under the limit raw, over it once ``&`` becomes ``&amp;``. That
        is the only reason this test can tell the two orders apart. Raise the
        truncation limit without raising the fixture and escape-then-truncate
        also fits inside it, so the test keeps passing on the wrong order
        instead of failing -- measured at ``[:200]``.
        """
        mock_extract_non_matches.return_value = [
            {
                'doc_id': 'doc1',
                'field_path': 'description',
                'non_match_type': 'MISMATCH',
                'ground_truth_value': "x" * 97 + "&",
                'prediction_value': "y" * 97 + "&"
            }
        ]

        result = self.section_generator.generate_non_matches(ReportConfig())

        assert "x" * 97 + "&amp;" in result
        assert "y" * 97 + "&amp;" in result
        assert "&am<" not in result

    def test_generate_document_gallery_image_mode(self):
        """Test document gallery generation in image mode."""
        document_images = {
            'doc1': 'images/doc1.jpg',
            'doc2': 'images/doc2.png'
        }
        config = ReportConfig(document_file_type='image')
        
        result = SectionGenerator.generate_document_gallery(document_images, config)
        
        assert '<div class="section"><h2>Document Gallery</h2>' in result
        assert '<div class="document-gallery">' in result
        assert '<div class="image-item">' in result
        assert '<img src="images/doc1.jpg" alt="doc1">' in result
        assert '<img src="images/doc2.png" alt="doc2">' in result
        assert '<p><strong>doc1</strong></p>' in result
        assert '<p><strong>doc2</strong></p>' in result
    
    def test_generate_document_gallery_pdf_mode(self):
        """Test document gallery generation in PDF mode."""
        document_pdfs = {
            'doc1': 'pdfs/doc1.pdf',
            'doc2': 'pdfs/doc2.pdf'
        }
        config = ReportConfig(document_file_type='pdf')
        
        result = SectionGenerator.generate_document_gallery(document_pdfs, config)
        
        assert '<div class="section"><h2>PDF Gallery</h2>' in result
        assert '<div class="document-gallery">' in result
        assert '<div class="pdf-item" data-doc-id="doc1" data-pdf-path="pdfs/doc1.pdf">' in result
        assert '<canvas id="pdf-canvas-doc1" class="pdf-canvas"></canvas>' in result
        assert '<div class="pdf-loading" id="pdf-loading-doc1">Loading PDF...</div>' in result
        assert '<div class="pdf-error" id="pdf-error-doc1" style="display: none;">Error loading PDF</div>' in result
        assert '<p><strong>doc1</strong></p>' in result
        assert '<p><strong>doc2</strong></p>' in result
    
    def test_generate_document_gallery_empty_documents(self):
        """Test document gallery generation with empty document list."""
        document_images = {}
        config = ReportConfig(document_file_type='image')
        
        result = SectionGenerator.generate_document_gallery(document_images, config)
        
        assert '<div class="section"><h2>Document Gallery</h2>' in result
        assert '<div class="document-gallery">' in result
        assert '</div></div>' in result
        # Should not contain any image items
        assert '<div class="image-item">' not in result

    def test_generate_document_gallery_image_mode_coerces_int_key(self):
        """Image gallery accepts an integer document_files key (issue #454)."""
        document_images = {1001: 'images/scan.png'}
        config = ReportConfig(document_file_type='image')

        result = SectionGenerator.generate_document_gallery(document_images, config)

        assert '<img src="images/scan.png" alt="1001">' in result
        assert '<p><strong>1001</strong></p>' in result

    def test_generate_document_gallery_image_mode_coerces_int_path(self):
        """Image gallery accepts a non-str document_files path (issue #454)."""
        document_images = {'doc-1': 1001}
        config = ReportConfig(document_file_type='image')

        result = SectionGenerator.generate_document_gallery(document_images, config)

        assert '<img src="1001" alt="doc-1">' in result
        assert '<p><strong>doc-1</strong></p>' in result

    def test_generate_document_gallery_pdf_mode_coerces_int_key(self):
        """PDF gallery accepts an integer document_files key (issue #454)."""
        document_pdfs = {1001: 'pdfs/scan.pdf'}
        config = ReportConfig(document_file_type='pdf')

        result = SectionGenerator.generate_document_gallery(document_pdfs, config)

        assert 'data-doc-id="1001"' in result
        assert 'data-pdf-path="pdfs/scan.pdf"' in result
        assert 'id="pdf-canvas-1001"' in result
        assert '<p><strong>1001</strong></p>' in result

    def test_generate_document_gallery_pdf_mode_coerces_int_path(self):
        """PDF gallery accepts a non-str document_files path (issue #454)."""
        document_pdfs = {'doc-1': 1001}
        config = ReportConfig(document_file_type='pdf')

        result = SectionGenerator.generate_document_gallery(document_pdfs, config)

        assert 'data-doc-id="doc-1"' in result
        assert 'data-pdf-path="1001"' in result
        assert '<p><strong>doc-1</strong></p>' in result

    def test_document_gallery_coerces_numpy_key_and_path_in_both_modes(self):
        """The gallery accepts the DataFrame-shaped key and Path value from #454."""
        document_files = {np.int64(1001): Path('assets/scan.png')}

        image_result = SectionGenerator.generate_document_gallery(
            document_files, ReportConfig(document_file_type='image')
        )
        pdf_result = SectionGenerator.generate_document_gallery(
            document_files, ReportConfig(document_file_type='pdf')
        )

        assert '<img src="assets/scan.png" alt="1001">' in image_result
        assert 'data-doc-id="1001"' in pdf_result
        assert 'data-pdf-path="assets/scan.png"' in pdf_result

    def test_document_gallery_preserves_entries_with_colliding_string_keys(self):
        """Conversion preserves both ordered input entries when their keys collide."""
        document_files = {1001: 'images/first.png', '1001': 'images/second.png'}

        image_result = SectionGenerator.generate_document_gallery(
            document_files, ReportConfig(document_file_type='image')
        )
        pdf_result = SectionGenerator.generate_document_gallery(
            document_files, ReportConfig(document_file_type='pdf')
        )

        assert image_result.count('<div class="image-item">') == 2
        assert image_result.index('images/first.png') < image_result.index('images/second.png')
        assert image_result.count('alt="1001"') == 2
        assert pdf_result.count('data-doc-id="1001"') == 2
        assert pdf_result.index('images/first.png') < pdf_result.index('images/second.png')


class TestSectionGeneratorIntegration:
    """Integration tests for SectionGenerator with real data structures."""
    
    def test_generate_sections_with_individual_results(self):
        """Test section generation with individual results format."""
        individual_results = {
            'overall': {
                'f1': 0.82,
                'precision': 0.85,
                'recall': 0.79
            },
            'fields': {
                'name': {'f1': 0.90, 'precision': 0.95, 'recall': 0.85},
                'price': {'f1': 0.75, 'precision': 0.80, 'recall': 0.70}
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
        
        viz_engine = VisualizationEngine()
        section_generator = SectionGenerator(individual_results, viz_engine)
        
        # Test that sections can be generated without errors
        config = ReportConfig()
        executive_summary = section_generator.generate_executive_summary(config)
        field_analysis = section_generator.generate_field_analysis(config)
        confusion_matrix = section_generator.generate_confusion_matrix()
        non_matches = section_generator.generate_non_matches(config)
        
        assert '<div class="section">' in executive_summary
        assert '<div class="section">' in field_analysis
        assert '<div class="section">' in confusion_matrix
        assert '<div class="section">' in non_matches
        
        # Verify content is present
        assert '0.820' in executive_summary
        assert 'name' in field_analysis
        assert 'price' in field_analysis
        assert 'TP' in confusion_matrix
        assert 'category' in non_matches


class TestNonStrDocId:
    """A non-`str` doc_id must not take the whole report down.

    `BulkStructuredModelEvaluator.update` annotates `doc_id` as `Optional[str]`, but
    Python does not enforce that at runtime and `evaluate_dataframe` reads the value
    straight off a DataFrame column (`doc_id = row.get("doc_id", ...)`). An integer
    id — the ordinary case for one read from CSV, parquet or a database — used to
    reach `html.escape` unconverted and raise, so `generate_report` returned
    `success=False` and wrote no page at all.
    """

    def test_non_matches_table_renders_a_non_str_doc_id(self):
        viz = Mock(spec=VisualizationEngine)
        results = Mock(spec=ProcessEvaluation)
        results.non_matches = [
            {
                "doc_id": 1001,
                "field_path": 7,
                "non_match_type": "NonMatchType.FALSE_DISCOVERY",
                "ground_truth_value": "Acme",
                "prediction_value": "Globex",
            }
        ]
        html_out = SectionGenerator(results, viz).generate_non_matches(ReportConfig())
        assert "1001" in html_out
        assert ">7<" in html_out

    def test_a_dataframe_with_an_integer_doc_id_still_produces_a_report(self, tmp_path):
        import json

        import pandas as pd

        from stickler import StructuredModel
        from stickler.reporting.html.html_reporter import EvaluationHTMLReporter
        from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
            BulkStructuredModelEvaluator,
        )

        class Invoice(StructuredModel):
            vendor: str

        df = pd.DataFrame(
            [
                {
                    "doc_id": 1001,
                    "expected": json.dumps({"vendor": "Acme"}),
                    "predicted": json.dumps({"vendor": "Globex"}),
                }
            ]
        )
        results = BulkStructuredModelEvaluator(Invoice).evaluate_dataframe(df)
        assert {type(nm.get("doc_id")) for nm in results.non_matches} == {int}

        out = tmp_path / "report.html"
        report = EvaluationHTMLReporter().generate_report(results, str(out))
        assert report.success, report.errors
        assert out.exists()
