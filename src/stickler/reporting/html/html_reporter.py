"""
Simple HTML reporter for evaluation results - v0.
"""

import os
import time
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from stickler.reporting.html.report_config import ReportConfig, ReportResult
from stickler.reporting.html.content_analyzer import ContentAnalyzer
from stickler.reporting.html.visualization_engine import VisualizationEngine
from stickler.reporting.html.utils import ColorUtils, DataExtractor
from stickler.utils.process_evaluation import ProcessEvaluation
from stickler.reporting.html.section_generator import SectionGenerator

class EvaluationHTMLReporter:
    """
    Simple HTML report generator for evaluation results.
    Supports both individual and bulk evaluation formats.
    """
    
    def __init__(self, theme: str = "professional"):
        """
        Initialize the HTML reporter.
        
        Args:
            theme: Visual theme (professional, dark, light)
        """
        self.theme = theme
        
    def generate_report(
        self,
        evaluation_results: Union[Dict[str, Any], ProcessEvaluation],
        output_path: str,
        config: Optional[ReportConfig] = None,
        document_images: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        model_schema: Optional[type] = None,
        individual_results_jsonl_path: Optional[str] = None
    ) -> ReportResult:
        """
        Generate HTML report from evaluation results.
        
        Args:
            evaluation_results: Results from evaluator (individual or bulk)
            output_path: Path where HTML report will be saved
            config: Report configuration options
            document_images: Dictionary mapping document IDs to image paths
            title: Custom report title
            model_schema: StructuredModel class to extract field thresholds from
            
        Returns:
            ReportResult with generation metadata
        """
        start_time = time.time()
        config = config or ReportConfig()
        
        try:
            # Determine if this is bulk or individual results
            is_bulk = isinstance(evaluation_results, ProcessEvaluation)
            
            # Generate HTML content
            html_content = self._generate_html_content(
                evaluation_results, config, document_images, title, is_bulk, model_schema, individual_results_jsonl_path
            )
            
            # Write to file
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Calculate file size and timing
            file_size = os.path.getsize(output_path)
            generation_time = time.time() - start_time
            
            return ReportResult(
                output_path=output_path,
                success=True,
                file_size_bytes=file_size,
                generation_time_seconds=generation_time,
                sections_included=self._get_sections_included(config),
                metadata={
                    "theme": self.theme,
                    "is_bulk": is_bulk,
                    "document_count": self._get_document_count(evaluation_results),
                }
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            return ReportResult(
                output_path=output_path,
                success=False,
                file_size_bytes=0,
                generation_time_seconds=generation_time,
                sections_included=[],
                errors=[str(e)],
                metadata={"theme": self.theme}
            )
    
    def _generate_html_content(
        self,
        results: Union[Dict[str, Any], ProcessEvaluation],
        config: ReportConfig,
        document_images: Optional[Dict[str, str]],
        title: Optional[str],
        is_bulk: bool,
        model_schema: Optional[type] = None,
        individual_results_jsonl_path: Optional[str] = None
    ) -> str:
        """Generate the complete HTML content."""
        
        # Initialize content analyzer and visualization engine
        content_analyzer = ContentAnalyzer()
        analysis = content_analyzer.analyze_results(results, is_bulk, model_schema)
        viz_engine = VisualizationEngine(theme=self.theme)
        
        # Generate sections
        sections = []
        section_generator = SectionGenerator(results, viz_engine)
        
        if config.include_executive_summary:
            sections.append(section_generator.generate_executive_summary(analysis))
        
        if config.include_field_analysis:
            sections.append(section_generator.generate_field_analysis())
        
        if config.include_confusion_matrix:
            sections.append(section_generator.generate_confusion_matrix())
        
        if config.include_non_matches:
            sections.append(section_generator.generate_non_matches(config))
        
        if document_images:
            sections.append(section_generator.generate_document_gallery(document_images, config))
        
        # Add individual document details section if JSONL path provided
        if individual_results_jsonl_path and os.path.exists(individual_results_jsonl_path):
            individual_docs = self._load_individual_results(individual_results_jsonl_path)
       
        # Build complete HTML
        html_content = self._build_html_document(
            sections=sections,
            title=title or self._generate_title(results, is_bulk),
            individual_docs=individual_docs if individual_results_jsonl_path and os.path.exists(individual_results_jsonl_path) else None,
            analysis=analysis
        )
        
        return html_content
    
    def _load_individual_results(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load individual document results from JSONL file."""
        individual_docs = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        doc_data = json.loads(line)
                        individual_docs.append(doc_data)
        except Exception as e:
            print(f"Warning: Failed to load individual results from {jsonl_path}: {e}")
        return individual_docs
    
    def _generate_individual_documents_section(self, individual_docs: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Generate interactive individual documents section."""
        field_thresholds = analysis.get('field_thresholds', {})
        
        html = '''
        <div class="section" id="individual-documents">
            <h2>Individual Document Analysis</h2>
            <p>Click on any document ID to view detailed field-by-field analysis with similarity scores and threshold compliance.</p>
            
            <div class="document-controls">
                <input type="text" id="doc-search" placeholder="Search documents..." class="doc-search-input">
                <select id="threshold-filter" class="doc-filter-select">
                    <option value="all">All Documents</option>
                    <option value="pass">Threshold Pass</option>
                    <option value="fail">Threshold Fail</option>
                </select>
            </div>
            
            <table class="document-table">
                <thead>
                    <tr>
                        <th>Document ID</th>
                        <th>Overall Similarity</th>
                        <th>Threshold Status</th>
                        <th>Field Issues</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for doc_data in individual_docs:
            doc_id = doc_data.get('doc_id', 'Unknown')
            comparison_result = doc_data.get('comparison_result', {})
            
            # Extract overall similarity score
            overall_similarity = comparison_result.get('similarity_score', 0.0)
            if not isinstance(overall_similarity, (int, float)):
                overall_similarity = comparison_result.get('overall', {}).get('similarity_score', 0.0)
            
            # Calculate threshold compliance
            field_issues = self._analyze_document_field_compliance(comparison_result, field_thresholds)
            threshold_status = "Pass" if field_issues['failing_count'] == 0 else "Fail"
            status_class = "threshold-pass" if threshold_status == "Pass" else "threshold-fail"
            
            html += f'''
                <tr class="doc-row {status_class}" data-doc-id="{doc_id}">
                    <td><a href="#" class="doc-link" data-doc-id="{doc_id}">{doc_id}</a></td>
                    <td>{overall_similarity:.3f}</td>
                    <td><span class="status-badge {status_class}">{threshold_status}</span></td>
                    <td>{field_issues['failing_count']}/{field_issues['total_count']} fields</td>
                    <td><button class="view-details-btn" data-doc-id="{doc_id}">View Details</button></td>
                </tr>
            '''
        
        html += '''
                </tbody>
            </table>
            
            <!-- Individual document detail view (initially hidden) -->
            <div id="document-detail" class="document-detail" style="display: none;">
                <div class="detail-header">
                    <button id="back-to-list" class="back-btn">← Back to Document List</button>
                    <h3 id="detail-title">Document Details</h3>
                    <div class="detail-navigation">
                        <button id="prev-doc" class="nav-btn">← Previous</button>
                        <button id="next-doc" class="nav-btn">Next →</button>
                    </div>
                </div>
                <div id="detail-content">
                    <!-- Detail content will be populated by JavaScript -->
                </div>
            </div>
        </div>
        '''
        
        return html
    
    def _analyze_document_field_compliance(self, comparison_result: Dict[str, Any], field_thresholds: Dict[str, float]) -> Dict[str, int]:
        """Analyze field compliance for a single document."""
        failing_count = 0
        total_count = 0
        
        # Access fields from the correct nested structure
        confusion_matrix = comparison_result.get('confusion_matrix', {})
        fields_data = confusion_matrix.get('fields', {})
        
        for field_name, field_result in fields_data.items():
            if field_name in field_thresholds:
                total_count += 1
                # Use raw_similarity_score as that's where the actual similarity is stored
                similarity_score = field_result.get('raw_similarity_score', 0.0)
                if similarity_score < field_thresholds[field_name]:
                    failing_count += 1
        
        return {
            'failing_count': failing_count,
            'total_count': total_count
        }
    
    def _build_html_document(self, sections: List[str], title: str, individual_docs: Optional[List[Dict]] = None, analysis: Optional[Dict] = None) -> str:
        """Build the complete HTML document."""
        css = self._get_basic_css()
        javascript = self._get_javascript(individual_docs, analysis) if individual_docs else ""
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <main>
            {"".join(sections)}
        </main>
        
        <footer>
            <p>Evaluation Report - Generated by Stickler</p>
        </footer>
    </div>
    {javascript}
</body>
</html>'''
        
        return html
    
    def _get_basic_css(self) -> str:
        """Load CSS from external file."""
        
        # Get the directory where this module is located
        module_dir = Path(__file__).parent
        css_dir = module_dir / "styling"
        
        css_path = css_dir / 'style.css'
        
        try:
            # Read CSS file
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            return css_content
            
        except FileNotFoundError:
            print(f"Warning: CSS file {css_path} not found.")
        
    
    def _get_sections_included(self, config: ReportConfig) -> List[str]:
        """Get list of sections included in the report."""
        sections = []
        if config.include_executive_summary:
            sections.append("executive_summary")
        if config.include_field_analysis:
            sections.append("field_analysis")
        if config.include_confusion_matrix:
            sections.append("confusion_matrix")
        if config.include_non_matches:
            sections.append("non_matches")
        return sections
    
    def _get_document_count(self, results: Union[Dict, ProcessEvaluation]) -> int:
        """Get document count by counting unique doc_ids from available data."""
        if isinstance(results, ProcessEvaluation):
           return getattr(results, 'document_count', 1)
        return 1
    
    def _generate_title(self, results: Union[Dict, ProcessEvaluation], is_bulk: bool) -> str:
        """Generate report title."""
        if is_bulk:
            doc_count = self._get_document_count(results)
            return f"Evaluation Report - {doc_count} Documents"
        return "Evaluation Report"
    
    def _get_javascript(self, individual_docs: List[Dict], analysis: Dict) -> str:
        """Generate JavaScript section with external file reference and data initialization."""
        field_thresholds = analysis.get('field_thresholds', {})
        
        # Convert documents and analysis to JSON for JavaScript
        docs_json = json.dumps(individual_docs)
        thresholds_json = json.dumps(field_thresholds)
        
        # Load external JavaScript file
        js_file_content = self._load_javascript_file()
        
        return f'''
            <script>
            {js_file_content}
            </script>
            <script>
            // Initialize document data
            initializeDocumentData({docs_json}, {thresholds_json});
            </script>
        '''
    
    def _load_javascript_file(self) -> str:
        """Load JavaScript from external file."""
        module_dir = Path(__file__).parent
        js_dir = module_dir / "styling"
        js_path = js_dir / 'interactive.js'
        
        try:
            with open(js_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: JavaScript file {js_path} not found.")
            return "// JavaScript file not found"
