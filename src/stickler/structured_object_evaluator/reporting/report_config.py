"""
Configuration classes for HTML report generation.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator


class ReportConfig(BaseModel):
    """Comprehensive configuration for report generation"""
    
    # Content Control
    include_executive_summary: bool = True
    include_field_analysis: bool = True
    include_non_matches: bool = True
    include_confusion_matrix: bool = True
    include_recommendations: bool = True
    
    # Visualization Options
    chart_types: List[str] = Field(default_factory=lambda: ["bar", "heatmap", "scatter"])
    interactive_charts: bool = True
    chart_height: int = 400
    chart_width: Optional[int] = None  # Auto-width if None
    
    # Detail Levels
    max_non_matches_displayed: int = 1000
    max_errors_displayed: int = 100
    field_detail_threshold: float = 0.0  # Only show fields with activity above this
    
    # Table Options
    paginate_large_tables: bool = True
    table_page_size: int = 50
    sortable_tables: bool = True
    searchable_tables: bool = True
    
    # Image Options
    lazy_load_images: bool = True
    image_thumbnail_size: int = 200
    enable_image_zoom: bool = True
    
    # Export Options
    enable_pdf_export: bool = False  # Requires additional dependencies
    enable_json_export: bool = True
    enable_csv_export: bool = True
    
    # Theme and Styling
    theme: str = "professional"  # professional, dark, light, medical, academic
    custom_css: Optional[str] = None
    logo_url: Optional[str] = None
    branding: Optional[Dict[str, str]] = None
    
    # Performance Options
    minify_html: bool = True
    inline_styles: bool = True
    inline_scripts: bool = True
    
    @field_validator('max_non_matches_displayed', 'max_errors_displayed')
    @classmethod
    def validate_max_displayed(cls, v):
        if v < 0:
            raise ValueError('must be non-negative')
        return v
    
    @field_validator('field_detail_threshold')
    @classmethod
    def validate_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError('must be between 0 and 1')
        return v
    
    @field_validator('chart_height', 'table_page_size', 'image_thumbnail_size')
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError('must be positive')
        return v
    
    @field_validator('chart_width')
    @classmethod
    def validate_chart_width(cls, v):
        if v is not None and v <= 0:
            raise ValueError('must be positive when specified')
        return v
    
    @field_validator('theme')
    @classmethod
    def validate_theme(cls, v):
        valid_themes = ["professional", "dark", "light", "medical", "academic"]
        if v not in valid_themes:
            raise ValueError(f'must be one of {valid_themes}')
        return v
    
    @field_validator('chart_types')
    @classmethod
    def validate_chart_types(cls, v):
        valid_chart_types = ["bar", "heatmap", "scatter", "line", "pie", "treemap"]
        invalid_charts = [ct for ct in v if ct not in valid_chart_types]
        if invalid_charts:
            raise ValueError(f'invalid chart types: {invalid_charts}. Valid types: {valid_chart_types}')
        return v


class ReportResult(BaseModel):
    """Result of report generation with metadata"""
    
    output_path: str
    success: bool
    file_size_bytes: int
    generation_time_seconds: float
    sections_included: List[str]
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def file_size_mb(self) -> float:
        """File size in megabytes"""
        return self.file_size_bytes / (1024 * 1024)
    
    def summary(self) -> str:
        """Human-readable summary of the report generation"""
        status = "✓ Success" if self.success else "✗ Failed"
        size_str = f"{self.file_size_mb:.2f} MB" if self.file_size_bytes > 0 else "N/A"
        time_str = f"{self.generation_time_seconds:.2f}s"
        
        summary = f"{status} | {size_str} | {time_str} | {len(self.sections_included)} sections"
        
        if self.warnings:
            summary += f" | {len(self.warnings)} warnings"
        if self.errors:
            summary += f" | {len(self.errors)} errors"
            
        return summary
