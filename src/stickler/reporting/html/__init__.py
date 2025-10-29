"""
HTML reporting module for structured object evaluation results.
"""

from .html_reporter import EvaluationHTMLReporter
from .report_config import ReportConfig
from .visualization_engine import VisualizationEngine
from .section_generator import SectionGenerator

__all__ = [
    "EvaluationHTMLReporter",
]
