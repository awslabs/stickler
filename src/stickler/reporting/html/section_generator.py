from typing import Dict, Any, Union
from stickler.utils.process_evaluation import ProcessEvaluation
from stickler.reporting.html.visualization_engine import VisualizationEngine
from stickler.reporting.html.report_config import ReportConfig
from stickler.reporting.html.utils import ColorUtils, DataExtractor
import html

class SectionGenerator:
    def __init__(self, results, viz_engine):
        self.results: Union[Dict, ProcessEvaluation] = results
        self.viz_engine: VisualizationEngine = viz_engine

    def generate_executive_summary(self, config: ReportConfig) -> str:
        """Generate executive summary section."""
        metrics = DataExtractor.extract_overall_metrics(self.results)
        doc_count =  getattr(self.results, 'document_count', 1)

        html_string = f"""
        <div class="section">
            <h2>Executive Summary</h2>
        """
        
        # Add performance gauge for overall F1 score
        f1_score = metrics.get('cm_f1', metrics.get('f1', 0))
        if isinstance(f1_score, (int, float)) and f1_score > 0:
            html_string += f'<div class="performance-section">{self.viz_engine.generate_performance_gauge(f1_score, config)}</div>'
        
        html_string += f"""
            <div class="summary-grid">
                <div class="metric-card">
                    <div class="metric-value">{doc_count}</div>
                    <div class="metric-label">Documents</div>
                </div>
        """
        
        # Add key metrics with color coding
        key_metrics = ['cm_precision', 'cm_recall', 'cm_f1', 'cm_accuracy', 'f1', 'precision', 'recall', 'accuracy']
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    display_value = f"{value:.3f}"
                    color = ColorUtils.get_performance_color(value, config.color_thresholds) if hasattr(self.viz_engine, '_get_performance_color') else '#007bff'
                else:
                    display_value = str(value)
                    color = "#010101"
                    
                html_string += f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {color};">{html.escape(display_value)}</div>
                    <div class="metric-label">{metric.replace('cm_', '').replace('_', ' ').title()}</div>
                </div>
                """
        
        html_string += "</div></div>"
        return html_string
    
    def generate_field_analysis(self, config: ReportConfig) -> str:
        html_string = '<div class="section"><h2>Field Performance Analysis</h2>'
        
        field_metrics = DataExtractor.extract_field_metrics(self.results)
        
        if not field_metrics:
            html_string += "<p>No field data available.</p></div>"
            return html_string
        
    
        # Add color-coded field performance chart with thresholds
        html_string += self.viz_engine.generate_field_performance_chart(field_metrics, config)
        
        # Add color coded performance chart.
        html_string += self.viz_engine.generate_field_performance_table(field_metrics, config)

        return html_string
    
    def generate_confusion_matrix(self) -> str:
        html = '<div class="section"><h2>Confusion Matrix</h2>'
        
        cm_data = DataExtractor.extract_confusion_matrix(self.results)
        
        if not cm_data:
            html += "<p>No confusion matrix data available.</p></div>"
            return html
        
        # Add color-coded confusion matrix heatmap
        html += self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        html += '</div>'
        return html
    
    def generate_non_matches(self, config: ReportConfig) -> str:
        """Generate non-matches section."""
        html_string = '<div class="section"><h2>Non-Matches Analysis</h2>'
        
        non_matches = DataExtractor.extract_non_matches(self.results)
        
        if not non_matches:
            html_string += "<p>No non-matches found.</p></div>"
            return html_string
        
        # Summary
        total_non_matches = len(non_matches)
        html_string += f'<p>Found {total_non_matches} non-matches.</p>'
        
        # Limit displayed non-matches
        displayed = non_matches[:config.max_non_matches_displayed]
        
        # Non-matches table
        html_string += '<table class="data-table" id="non-matches-table">'
        html_string += '''
        <thead>
            <tr>
                <th>Document</th>
                <th>Field</th>
                <th>Type</th>
                <th>Ground Truth</th>
                <th>Prediction</th>
            </tr>
        </thead>
        <tbody>
        '''
        
        for nm in displayed:
            doc_id = html.escape(nm.get('doc_id', 'N/A'))
            field_path = html.escape(nm.get('field_path', 'N/A'))
            non_match_type = html.escape(str(nm.get('non_match_type', 'N/A')).replace('NonMatchType.',''))
            ground_truth_value = html.escape(str(nm.get('ground_truth_value', 'None')))[:100]  # Truncate long values
            prediction_value = html.escape(str(nm.get('prediction_value', 'None')))[:100]
            
            html_string += f'''
            <tr>
                <td>{doc_id}</td>
                <td>{field_path}</td>
                <td>{non_match_type}</td>
                <td>{ground_truth_value}</td>
                <td>{prediction_value}</td>
            </tr>
            '''
        
        html_string += '</tbody></table>'
        
        if total_non_matches > config.max_non_matches_displayed:
            html_string += f'<p><em>Showing {config.max_non_matches_displayed} of {total_non_matches} non-matches.</em></p>'
        
        html_string += '</div>'
        return html_string
    
    @staticmethod
    def generate_document_gallery(document_images: Dict[str, str], config: ReportConfig) -> str:
        """Generate document gallery section."""
        document_file_type = config.document_file_type
        html_string = ''

        if document_file_type == 'image':
            html_string += '<div class="section"><h2>Document Gallery</h2>'
            html_string += '<div class="document-gallery">'
            
            for doc_id, image_path in document_images.items():
                html_string += f'''
                    <div class="image-item">
                        <img src="{html.escape(image_path)}" alt="{html.escape(doc_id)}">
                        <p><strong>{html.escape(doc_id)}</strong></p>
                    </div>
                    '''
            
            html_string += '</div></div>'
        elif document_file_type == 'pdf':
            html_string = '<div class="section"><h2>PDF Gallery</h2>'
            html_string += '<div class="document-gallery">'

            for doc_id, pdf_path in document_images.items():
                html_string += f'''
                    <div class="pdf-item" data-doc-id="{html.escape(doc_id)}" data-pdf-path="{html.escape(pdf_path)}">
                        <div class="pdf-container">
                            <canvas id="pdf-canvas-{html.escape(doc_id)}" class="pdf-canvas"></canvas>
                            <div class="pdf-loading" id="pdf-loading-{html.escape(doc_id)}">Loading PDF...</div>
                            <div class="pdf-error" id="pdf-error-{html.escape(doc_id)}" style="display: none;">Error loading PDF</div>
                        </div>
                        <p><strong>{html.escape(doc_id)}</strong></p>
                    </div>
                    '''

            html_string += '</div></div>'
        return html_string
