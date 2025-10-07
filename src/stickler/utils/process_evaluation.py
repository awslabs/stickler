from stickler.utils.markdown_util import MarkdownUtil
from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class ProcessEvaluation(BaseModel):
    """
    A Pydantic model for evaluation results that doesn't require all fields to be initialized.
    """

    metrics: Optional[Dict[str, Any]] = None
    field_metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    matrix: Optional[Any] = None
    total_time: Optional[float] = None
    non_matches: Optional[List[Dict[str, Any]]] = None

    def to_md(self) -> str:
        """
        Converts the evaluation results to a markdown representation using MarkdownUtil.

        Returns:
            A string containing the markdown representation of the evaluation results.
        """

        sections = []

        # Add overall metrics section if available
        if self.metrics:
            sections.append("## Overall Metrics")
            sections.append(MarkdownUtil.table_dict(self.metrics))
            sections.append("")

        # Add field metrics section if available
        if self.field_metrics:
            sections.append("## Field Metrics")
            sections.append(MarkdownUtil.table_dict(self.field_metrics))
            sections.append("")

        # Add errors section if available
        if self.errors and len(self.errors) > 0:
            sections.append("## Errors")
            sections.append(MarkdownUtil.table_list(self.errors))
            sections.append("")

        # Add total time section if available
        if self.total_time is not None:
            sections.append("## Processing Time")
            sections.append(f"Total processing time: {self.total_time:.2f} seconds")
            sections.append("")

        # Add matrix section if available
        # Note: Matrix handling depends on its structure, this is a simple example
        if self.matrix is not None and not self.matrix.empty:
            sections.append("## Confusion Matrix")
            sections.append(MarkdownUtil.table_df(df=self.matrix))

        return "\n".join(sections).strip()
