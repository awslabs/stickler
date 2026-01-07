"""
HTML reporting module for structured object evaluation results.
"""

from .html_reporter import EvaluationHTMLReporter
from .report_config import ReportConfig
from .section_generator import SectionGenerator
from .visualization_engine import VisualizationEngine

__all__ = [
    "EvaluationHTMLReporter",
    "ReportConfig",
    "SectionGenerator",
    "VisualizationEngine"
]
