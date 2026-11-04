import html
from typing import Dict, Union

from stickler.reporting.html.report_config import ReportConfig
from stickler.reporting.html.utils import ColorUtils, DataExtractor
from stickler.reporting.html.visualization_engine import (
    VisualizationEngine,
    _esc,
    _finite,
    _fmt,
)
from stickler.utils.process_evaluation import ProcessEvaluation


class SectionGenerator:
    def __init__(self, results, viz_engine):
        self.results: Union[Dict, ProcessEvaluation] = results
        self.viz_engine: VisualizationEngine = viz_engine

    def generate_executive_summary(self, config: ReportConfig) -> str:
        """Generate executive summary section."""
        metrics = DataExtractor.extract_overall_metrics(self.results)
        doc_count =  getattr(self.results, 'document_count', 1)

        html_string = """
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
                    <div class="metric-value">{html.escape(str(doc_count))}</div>
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
        html_string = '<div class="section"><h2>Confusion Matrix</h2>'
        
        cm_data = DataExtractor.extract_confusion_matrix(self.results)
        
        if not cm_data:
            html_string += "<p>No confusion matrix data available.</p></div>"
            return html_string
        
        # Add color-coded confusion matrix heatmap
        html_string += self.viz_engine.generate_confusion_matrix_heatmap(cm_data, {})
        
        html_string += '</div>'
        return html_string
    
    def generate_non_matches(self, config: ReportConfig) -> str:
        """Generate non-matches section."""
        html_string = '<div class="section"><h2>Non-Matches Analysis</h2>'
        
        non_matches = DataExtractor.extract_non_matches(self.results)

        if non_matches is None:
            # No records were collected, which says nothing about whether the
            # fields matched. Reporting success here is how a document with a
            # wrong field was rendered as clean (#411).
            html_string += (
                "<p>No non-match records in these results. Call "
                "<code>compare_with()</code> or the bulk evaluator with "
                "<code>document_non_matches=True</code> to collect them.</p></div>"
            )
            return html_string

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
            doc_id = html.escape(str(nm.get('doc_id', 'N/A')))
            field_path = html.escape(str(nm.get('field_path', 'N/A')))
            non_match_type = html.escape(str(nm.get('non_match_type', 'N/A')).replace('NonMatchType.',''))
            ground_truth_value = html.escape(str(nm.get('ground_truth_value', 'None'))[:100])  # Truncate long values
            prediction_value = html.escape(str(nm.get('prediction_value', 'None'))[:100])
            
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

    def generate_decision_metrics(self) -> str:
        """Decision metrics section (#469): a headline table, then per field a
        reliability diagram and a risk-coverage plot.

        Returns "" when the results carry no decision metrics, so a report
        without them is unchanged. Field names, labels and kinds are
        user-controlled and every one is escaped.
        """
        dm = DataExtractor.extract_decision_metrics(self.results)
        if dm is None:
            return ""
        fields = dm["fields"]
        try:
            curves = DataExtractor.extract_risk_coverage(dm)
        except (KeyError, TypeError, ValueError, IndexError):
            # No usable per-decision records (e.g. hand-built metrics): the
            # table and reliability diagrams still render.
            curves = {}

        out = '<div class="section" id="decision-metrics"><h2>Decision Metrics</h2>'
        out += (
            "<p>Choice, Score and Noul fields. Confidence is the probability of the "
            "predicted answer. Accuracy counts missing answers as wrong; calibration "
            "uses answered decisions only.</p>"
        )
        out += '<table class="data-table" id="decision-metrics-table"><thead><tr>'
        for head in ("Field", "Type", "Options", "Coverage", "Accuracy", "F1 / QWK", "AUROC", "ECE (mass)", "AURC"):
            out += f"<th>{head}</th>"
        out += "</tr></thead><tbody>"
        for name, f in fields.items():
            f = f if isinstance(f, dict) else {}
            corr = f.get("correctness") if isinstance(f.get("correctness"), dict) else {}
            cal = f.get("calibration") if isinstance(f.get("calibration"), dict) else {}
            cov = corr.get("coverage") if isinstance(corr.get("coverage"), dict) else {}
            kind = str(f.get("kind", ""))
            metric_key, metric_name = _HEADLINE.get(kind, (None, ""))
            headline = _fmt(corr.get(metric_key)) if metric_key else "-"
            if metric_name and headline != "-":
                headline += f" ({metric_name})"
            ok, missing = _finite(cov.get("ok")), _finite(cov.get("missing"))
            coverage = _fmt(cov.get("ratio"))
            if ok is not None and missing is not None:
                coverage += f" ({int(ok)} of {int(ok + missing)})"
            aurc = (curves.get(name) or {}).get("aurc")
            out += (
                "<tr>"
                f"<td>{_esc(name)}</td>"
                f"<td>{_esc(kind)}</td>"
                f"<td>{_labels(f.get('labels'))}</td>"
                f"<td>{_esc(coverage)}</td>"
                f"<td>{_esc(_fmt(corr.get('accuracy')))}</td>"
                f"<td>{_esc(headline)}</td>"
                f"<td>{_esc(_fmt(cal.get('auroc')))}</td>"
                f"<td>{_esc(_fmt(cal.get('ece_mass')))}</td>"
                f"<td>{_esc(_fmt(aurc))}</td>"
                "</tr>"
            )
        out += "</tbody></table>"

        for name, f in fields.items():
            f = f if isinstance(f, dict) else {}
            cal = f.get("calibration") if isinstance(f.get("calibration"), dict) else {}
            out += (
                '<div class="decision-field" style="margin-top: 24px;">'
                f'<h3>{_esc(name)} <span style="color: #6c757d; font-weight: normal;">({_esc(f.get("kind", ""))})</span></h3>'
                '<div style="display: flex; flex-wrap: wrap; gap: 24px;">'
                f'{self.viz_engine.generate_reliability_diagram(str(name), cal.get("ece_mass_bins"))}'
                f'{self.viz_engine.generate_risk_coverage_plot(str(name), curves.get(name))}'
                "</div></div>"
            )
        out += "</div>"
        return out


# kind -> (correctness key, display name) for the table's F1 / QWK column.
_HEADLINE = {"choice": ("macro_f1", "macro-F1"), "noul": ("f1", "F1"), "score": ("qwk", "QWK")}
_MAX_LABELS = 8


def _labels(labels) -> str:
    """Escaped, comma-separated option labels, truncated after ``_MAX_LABELS``."""
    if not isinstance(labels, (list, tuple)):
        return "-"
    shown = ", ".join(_esc(lab) for lab in labels[:_MAX_LABELS])
    if len(labels) > _MAX_LABELS:
        shown += f" (+{len(labels) - _MAX_LABELS} more)"
    return shown
