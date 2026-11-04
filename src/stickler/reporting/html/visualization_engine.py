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
import math
import numbers
from typing import Any, Dict, Optional

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
    # Decision-metric charts (#469): inline SVG, no script and no external
    # assets. Every coordinate is a plain ``float`` formatted here, and every
    # string (field names, labels, tooltips) goes through ``_esc``.

    def generate_reliability_diagram(self, field_name: str, bins: Any) -> str:
        """Equal-mass reliability diagram: mean confidence vs accuracy per bin.

        Args:
            field_name: Decision field name (user-controlled; escaped).
            bins: ``calibration["ece_mass_bins"]``; each has ``count``,
                ``mean_confidence``, ``accuracy`` and ``range``.

        Returns:
            HTML ``<figure>`` with an inline SVG, or a note when there is nothing to plot.
        """
        pts = []
        for b in bins if isinstance(bins, list) else []:
            if not isinstance(b, dict):
                continue
            x, y = _finite(b.get("mean_confidence")), _finite(b.get("accuracy"))
            n = _finite(b.get("count"))
            if x is None or y is None:
                continue
            rng = b.get("range") if isinstance(b.get("range"), (list, tuple)) else (None, None)
            lo, hi = (rng[0], rng[1]) if len(rng) == 2 else (None, None)
            pts.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0), int(n or 0), lo, hi))
        if not pts:
            return '<p class="decision-chart-empty">No answered decisions to plot.</p>'
        pts.sort(key=lambda p: p[0])
        max_n = max(p[2] for p in pts) or 1

        svg = _svg_open(f"Reliability diagram for {field_name}")
        svg += _svg_axes(1.0, "Mean confidence", "Accuracy")
        svg += (
            f'<line x1="{_sx(0):.1f}" y1="{_sy(0, 1):.1f}" x2="{_sx(1):.1f}" y2="{_sy(1, 1):.1f}" '
            'stroke="#adb5bd" stroke-width="1.5" stroke-dasharray="4 4">'
            '<title>Perfect calibration</title></line>'
        )
        svg += _svg_polyline([(p[0], p[1]) for p in pts], 1.0)
        for x, y, n, lo, hi in pts:
            r = 4.0 + 4.0 * math.sqrt(n / max_n)
            tip = (
                f"confidence {_fmt(x)}, accuracy {_fmt(y)}, n = {n}, "
                f"bin {_fmt(lo)} to {_fmt(hi)}"
            )
            svg += (
                f'<circle cx="{_sx(x):.1f}" cy="{_sy(y, 1):.1f}" r="{r:.1f}" fill="#007bff" '
                f'stroke="#ffffff" stroke-width="2"><title>{_esc(tip)}</title></circle>'
            )
        svg += "</svg>"
        caption = "Equal-mass bins; marker area grows with the bin's count. Points on the dashed diagonal are calibrated."
        return _figure("Reliability", svg, caption)

    def generate_risk_coverage_plot(self, field_name: str, curve: Optional[Dict[str, Any]]) -> str:
        """Risk-coverage curve: error rate of accepted decisions vs share accepted.

        Args:
            field_name: Decision field name (user-controlled; escaped).
            curve: One entry of ``DataExtractor.extract_risk_coverage``, or None.

        Returns:
            HTML ``<figure>`` with an inline SVG, or a note when there is nothing to plot.
        """
        points = (curve or {}).get("points") or []
        if not points:
            return '<p class="decision-chart-empty">No risk-coverage curve: no answered decisions or no per-decision records.</p>'
        xy = [(float(c), float(r)) for c, r, _, _ in points]
        y_max = _nice_max(max(r for _, r in xy))

        svg = _svg_open(f"Risk-coverage plot for {field_name}")
        svg += _svg_axes(y_max, "Coverage (share accepted)", "Risk (1 - accuracy)")
        svg += _svg_polyline(xy, y_max)
        if len(points) <= 50:
            for cov, risk, tau, accepted in points:
                tip = f"accept confidence >= {_fmt(tau)}: coverage {_fmt(cov)}, risk {_fmt(risk)}, {int(accepted)} accepted"
                svg += (
                    f'<circle cx="{_sx(float(cov)):.1f}" cy="{_sy(float(risk), y_max):.1f}" r="4" '
                    f'fill="#007bff" stroke="#ffffff" stroke-width="2"><title>{_esc(tip)}</title></circle>'
                )
        svg += "</svg>"
        caption = f"AURC {_fmt(curve.get('aurc'))} (lower is better)."
        if int(curve.get("n_missing") or 0):
            caption += (
                f" {int(curve['n_missing'])} of {int(curve.get('n_total') or 0)} answers missing;"
                " they are always escalated, so coverage stops short of 1."
            )
        return _figure("Risk-coverage", svg, caption)


# Plot geometry shared by the decision charts.
_W, _H = 320, 280
_ML, _MR, _MT, _MB = 52, 12, 14, 44
_PW, _PH = _W - _ML - _MR, _H - _MT - _MB
_INK, _MUTED, _GRID = "#495057", "#6c757d", "#e9ecef"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _finite(value: Any) -> Optional[float]:
    """``value`` as a plain float, or None if it is not a finite real number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a metric for display; ``-`` when absent or not finite."""
    v = _finite(value)
    return "-" if v is None else f"{v:.{digits}f}"


def _nice_max(value: float) -> float:
    """Smallest of 0.1, 0.2, 0.5, 1 that holds ``value``; each splits into five clean ticks."""
    for top in (0.1, 0.2, 0.5):
        if value <= top:
            return top
    return 1.0


def _tick(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _sx(x: float) -> float:
    return _ML + x * _PW


def _sy(y: float, y_max: float) -> float:
    return _MT + _PH - (y / y_max) * _PH


def _svg_open(label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}" role="img" aria-label="{_esc(label)}" '
        'style="font-family: inherit; font-size: 11px;">'
        f'<title>{_esc(label)}</title>'
    )


def _svg_axes(y_max: float, x_label: str, y_label: str) -> str:
    out = ""
    for i in range(6):
        fx, fy = i / 5, y_max * i / 5
        out += (
            f'<line x1="{_sx(fx):.1f}" y1="{_MT}" x2="{_sx(fx):.1f}" y2="{_MT + _PH}" stroke="{_GRID}"/>'
            f'<line x1="{_ML}" y1="{_sy(fy, y_max):.1f}" x2="{_ML + _PW}" y2="{_sy(fy, y_max):.1f}" stroke="{_GRID}"/>'
            f'<text x="{_sx(fx):.1f}" y="{_MT + _PH + 16}" text-anchor="middle" fill="{_MUTED}">{_tick(fx)}</text>'
            f'<text x="{_ML - 6}" y="{_sy(fy, y_max) + 4:.1f}" text-anchor="end" fill="{_MUTED}">{_tick(fy)}</text>'
        )
    out += (
        f'<line x1="{_ML}" y1="{_MT + _PH}" x2="{_ML + _PW}" y2="{_MT + _PH}" stroke="{_MUTED}"/>'
        f'<line x1="{_ML}" y1="{_MT}" x2="{_ML}" y2="{_MT + _PH}" stroke="{_MUTED}"/>'
        f'<text x="{_ML + _PW / 2:.1f}" y="{_H - 6}" text-anchor="middle" fill="{_INK}">{_esc(x_label)}</text>'
        f'<text x="12" y="{_MT + _PH / 2:.1f}" text-anchor="middle" fill="{_INK}" '
        f'transform="rotate(-90 12 {_MT + _PH / 2:.1f})">{_esc(y_label)}</text>'
    )
    return out


def _svg_polyline(xy, y_max: float) -> str:
    pts = " ".join(f"{_sx(float(x)):.1f},{_sy(float(y), y_max):.1f}" for x, y in xy)
    return f'<polyline points="{pts}" fill="none" stroke="#007bff" stroke-width="2" stroke-linejoin="round"/>'


def _figure(title: str, svg: str, caption: str) -> str:
    return (
        '<figure style="margin: 0; max-width: 340px;">'
        f'<div style="font-weight: 600; color: {_INK}; margin-bottom: 4px;">{_esc(title)}</div>'
        f'{svg}<figcaption style="color: {_MUTED}; font-size: 0.85em;">{_esc(caption)}</figcaption>'
        '</figure>'
    )
