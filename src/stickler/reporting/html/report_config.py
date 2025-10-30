"""
Configuration classes for HTML report generation.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ReportConfig(BaseModel):
    """Comprehensive configuration for report generation"""

    include_executive_summary: bool = True
    include_field_analysis: bool = True
    include_non_matches: bool = True
    include_confusion_matrix: bool = True
    max_non_matches_displayed: int = 1000
    document_file_type: str = "image"
    image_thumbnail_size: int = 200

    
    @field_validator('max_non_matches_displayed')
    @classmethod
    def validate_max_displayed(cls, v):
        if v < 0:
            raise ValueError('must be non-negative')
        return v

    @field_validator('image_thumbnail_size')
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError('must be positive')
        return v



class ReportResult(BaseModel):
    """Result of report generation with metadata"""
    
    output_path: str
    success: bool
    generation_time_seconds: float
    sections_included: List[str]
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    
    def summary(self) -> str:
        """Human-readable summary of the report generation"""
        status = "✓ Success" if self.success else "✗ Failed"
        time_str = f"{self.generation_time_seconds:.2f}s"
        
        summary = f"{status} |  {time_str} | {len(self.sections_included)} sections"

        if self.errors:
            summary += f" | {len(self.errors)} errors"
            
        return summary
