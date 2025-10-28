"""
Simple visualization engine for HTML reports - v0.
"""

from typing import Dict, Any, Optional
from .utils import ColorUtils, DataExtractor


class VisualizationEngine:
    """
    Simple visualization engine for generating charts and graphs.
    """
    
    def __init__(self, theme: str = "professional", include_plotly: bool = False, branding: Optional[Dict] = None):
        """
        Initialize the visualization engine.
        
        Args:
            theme: Visual theme
            include_plotly: Whether to include Plotly for interactive charts
            branding: Branding configuration (unused in v0)
        """
        self.theme = theme
        self.include_plotly = include_plotly
        self.branding = branding or {}
    
    def generate_performance_gauge(self, score: float) -> str:
        """
        Generate a simple performance gauge.
        
        Args:
            score: Performance score (0-1)
            
        Returns:
            HTML for performance gauge
        """
        percentage = int(score * 100)
        color = ColorUtils.get_performance_color(score)
        
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
    
    def generate_field_performance_chart(self, field_metrics: Dict[str, Any]) -> str:
        """
        Generate a simple field performance chart.
        
        Args:
            field_metrics: Dictionary of field metrics
            config: Report configuration
            field_thresholds: Dictionary of field-specific thresholds
            
        Returns:
            HTML for field performance chart
        """
        # For v0, create a simple horizontal bar chart using CSS
        html = '<div class="field-chart">'
        
        for field_name, metrics in field_metrics.items():
            if isinstance(metrics, dict):
                f1_score = metrics.get('cm_f1', metrics.get('f1', 0))
                if isinstance(f1_score, (int, float)):
                    percentage = int(f1_score * 100)
                    
                    color = ColorUtils.get_performance_color(f1_score)
                    threshold_line = ''
                    
                    html += f'''
                    <div class="field-bar">
                        <div class="field-label">{field_name}</div>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: {percentage}%; background-color: {color};"></div>
                            {threshold_line}
                            <span class="bar-value">{f1_score:.3f}</span>
                        </div>
                    </div>
                    '''
        
        html += '</div>'
        return html
    
    def generate_field_performance_table(self, field_metrics: Dict[str, Any]) -> str:
        html = ''

        # Generate detailed field performance table
        html += '<table class="performance-table">'
        html += '''
        <thead>
            <tr>
                <th>Field</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
        '''
        
        html += '''
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
                
            
                f1_color = ColorUtils.get_performance_color(f1)
                
                html += f'''
                <tr>
                    <td>{field_name}</td>
                    <td>{precision:.3f}</td>
                    <td>{recall:.3f}</td>
                    <td style="background-color: {f1_color}; color: white; font-weight: bold;">{f1:.3f}</td>
                '''

                html += f'''
                    <td>{tp}</td>
                    <td>{fd}</td>
                    <td>{fa}</td>
                    <td>{fn}</td>
                </tr>
                '''
        
        html += '</tbody></table></div>'
        return html
    
    def generate_confusion_matrix_heatmap(self, cm_data: Dict[str, Any], config: Any) -> str:
        """
        Generate a simple confusion matrix visualization.
        
        Args:
            cm_data: Confusion matrix data
            config: Report configuration
            
        Returns:
            HTML for confusion matrix heatmap
        """
        # Simple grid visualization for v0
        metrics = ['tp', 'tn', 'fd', 'fa', 'fn',]
        total = sum(cm_data.get(m, 0) for m in metrics)
        
        if total == 0:
            return '<p>No confusion matrix data to visualize.</p>'
        
        html = '<div class="cm-grid">'
        metric_colors = ColorUtils.get_confusion_matrix_colors()
        
        for metric in metrics:
            value = cm_data.get(metric, 0)
            percentage = (value / total) * 100 if total > 0 else 0
            
            html += f'''
            <div class="cm-cell" style="border-left-color: {metric_colors[metric]}">
                <div class="cm-label">{metric.upper()}</div>
                <div class="cm-value">{value}</div>
                <div class="cm-percentage">{percentage:.1f}%</div>
            </div>
            '''
        
        html += '</div>'
        return html