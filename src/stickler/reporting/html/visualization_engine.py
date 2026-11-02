"""
Simple visualization engine for HTML reports - v0.

Numeric cells that use a format spec (``:.3f`` / ``:.1f``) are rendered through
``html.escape(f"{{value:.Nf}}")`` so a caller-supplied numeric subclass whose
``__format__`` returns markup cannot land in the page (#453).

``generate_performance_gauge`` is excluded on purpose: ``int(score * 100)``
forces a plain ``int``, so ``__format__`` / ``__str__`` / ``__mul__`` on a
subclass cannot reach the template.
"""
import html
from typing import Any, Dict

from stickler.reporting.html.report_config import ReportConfig
from stickler.reporting.html.utils import ColorUtils


class VisualizationEngine:
    """
    Simple visualization engine for generating charts and graphs.
    """
    
    def generate_performance_gauge(self, score: float, config: ReportConfig) -> str:
        """
        Generate a simple performance gauge.
        
        Args:
            score: Performance score (0-1)
            
        Returns:
            HTML for performance gauge
        """
        percentage = int(score * 100)
        color = ColorUtils.get_performance_color(score, config.color_thresholds)
        
        return f'''
        <div class="performance-gauge">
            <div class="gauge-circle" style="background: conic-gradient({color} {percentage}%, #e9ecef {percentage}%);">
                <div class="gauge-inner">
                    <span class="gauge-value">{percentage}%</span>
                    <span class="gauge-label">Overall</span>
                </div>
            </div>
        </div>
        '''
    
    def generate_field_performance_chart(self, field_metrics: Dict[str, Any], config: ReportConfig) -> str:
        """
        Generate a simple field performance chart.
        
        Args:
            field_metrics: Dictionary of field metrics
            config: Report configuration
            field_thresholds: Dictionary of field-specific thresholds
            
        Returns:
            HTML for field performance chart
        """
        # create a simple horizontal chart
        html_string = '<div class="field-chart">'
        html_string += '<h4 style="margin-bottom: 15px; color: #495057; font-size: 1.1em;">F1 Score</h4>'
        
        for field_name, metrics in field_metrics.items():
            if isinstance(metrics, dict):
                f1_score = metrics.get('cm_f1', metrics.get('f1', 0))
                if isinstance(f1_score, (int, float)):
                    percentage = int(f1_score * 100)
                    
                    color = ColorUtils.get_performance_color(f1_score, config.color_thresholds)
                    
                    html_string += f'''
                    <div class="field-bar">
                        <div class="field-label">{html.escape(str(field_name))}</div>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: {percentage}%; background-color: {color};"></div>
                            <span class="bar-value">{html.escape(f"{f1_score:.3f}")}</span>
                        </div>
                    </div>
                    '''
        
        html_string += '</div>'
        return html_string
    
    def generate_field_performance_table(self, field_metrics: Dict[str, Any], config: ReportConfig) -> str:
        """
        Generate a simple field performance table visualization.
        
        Args:
            field_metrics: Field level data
            
        Returns:
            HTML for field performance scale.
        """
        html_string = ''

        # Generate detailed field performance table
        html_string += '<table class="data-table data-table-numeric" id="performance-table">'
        html_string += '''
        <thead>
            <tr>
                <th>Field</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
        '''
        
        html_string += '''
                <th>TP</th>
                <th>FD</th>
                <th>FA</th>
                <th>FN</th>
            </tr>
        </thead>
        <tbody>
        '''
        
        for field_name, metrics in field_metrics.items():
            if isinstance(metrics, dict):
                precision = metrics.get('cm_precision', metrics.get('precision', 0))
                recall = metrics.get('cm_recall', metrics.get('recall', 0))
                f1 = metrics.get('cm_f1', metrics.get('f1', 0))
                tp = metrics.get('tp', 0)
                fd = metrics.get('fd', 0)
                fa = metrics.get('fa', 0)
                fn = metrics.get('fn', 0)
                
            
                f1_color = ColorUtils.get_performance_color(f1, config.color_thresholds)
                
                html_string += f'''
                <tr>
                    <td>{html.escape(str(field_name))}</td>
                    <td>{html.escape(f"{precision:.3f}")}</td>
                    <td>{html.escape(f"{recall:.3f}")}</td>
                    <td style="background-color: {f1_color}; color: white; font-weight: bold;">{html.escape(f"{f1:.3f}")}</td>
                '''

                html_string += f'''
                    <td>{html.escape(str(tp))}</td>
                    <td>{html.escape(str(fd))}</td>
                    <td>{html.escape(str(fa))}</td>
                    <td>{html.escape(str(fn))}</td>
                </tr>
                '''
        
        html_string += '</tbody></table></div>'
        return html_string
    
    def generate_confusion_matrix_heatmap(self, cm_data: Dict[str, Any], config: Any) -> str:
        """
        Generate a simple confusion matrix visualization.
        
        Args:
            cm_data: Confusion matrix data
            config: Report configuration
            
        Returns:
            HTML for confusion matrix heatmap
        """
        metrics = ['tp', 'tn', 'fd', 'fa', 'fn',]
        total = sum(cm_data.get(m, 0) for m in metrics)
        
        if total == 0:
            return '<p>No confusion matrix data to visualize.</p>'
        
        html_string = '<div class="cm-grid">'
        metric_colors = ColorUtils.get_confusion_matrix_colors()
        
        for metric in metrics:
            value = cm_data.get(metric, 0)
            percentage = (value / total) * 100 if total > 0 else 0
            
            html_string += f'''
            <div class="cm-cell" style="border-left-color: {metric_colors[metric]}">
                <div class="cm-label">{metric.upper()}</div>
                <div class="cm-value">{html.escape(str(value))}</div>
                <div class="cm-percentage">{html.escape(f"{percentage:.1f}")}%</div>
            </div>
            '''
        
        html_string += '</div>'
        return html_string