"""
Tests for ReportConfig and ReportResult classes.
"""

import pytest
from pydantic import ValidationError
from stickler.reporting.html.report_config import ReportConfig, ReportResult


class TestReportConfig:
    """Test cases for ReportConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()
        
        assert config.include_executive_summary is True
        assert config.include_field_analysis is True
        assert config.include_non_matches is True
        assert config.include_confusion_matrix is True
        assert config.max_non_matches_displayed == 1000
        assert config.document_file_type == "image"
        assert config.image_thumbnail_size == 200
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReportConfig(
            include_executive_summary=False,
            include_field_analysis=False,
            max_non_matches_displayed=50,
            document_file_type="pdf",
            image_thumbnail_size=150
        )
        
        assert config.include_executive_summary is False
        assert config.include_field_analysis is False
        assert config.include_non_matches is True  # Default
        assert config.max_non_matches_displayed == 50
        assert config.document_file_type == "pdf"
        assert config.image_thumbnail_size == 150
    
    def test_negative_max_non_matches_validation(self):
        """Test validation of negative max_non_matches_displayed."""
        with pytest.raises(ValidationError) as exc_info:
            ReportConfig(max_non_matches_displayed=-1)
        
        assert "must be non-negative" in str(exc_info.value)
    
    def test_zero_max_non_matches_allowed(self):
        """Test that zero max_non_matches_displayed is allowed."""
        config = ReportConfig(max_non_matches_displayed=0)
        assert config.max_non_matches_displayed == 0
    
    def test_negative_thumbnail_size_validation(self):
        """Test validation of negative image_thumbnail_size."""
        with pytest.raises(ValidationError) as exc_info:
            ReportConfig(image_thumbnail_size=-1)
        
        assert "must be positive" in str(exc_info.value)
    
    def test_zero_thumbnail_size_validation(self):
        """Test validation of zero image_thumbnail_size."""
        with pytest.raises(ValidationError) as exc_info:
            ReportConfig(image_thumbnail_size=0)
        
        assert "must be positive" in str(exc_info.value)


class TestReportResult:
    """Test cases for ReportResult class."""
    
    def test_successful_result(self):
        """Test successful report result."""
        result = ReportResult(
            output_path="/path/to/report.html",
            success=True,
            generation_time_seconds=1.5,
            sections_included=["executive_summary", "field_analysis"],
            metadata={"document_count": 5}
        )
        
        assert result.output_path == "/path/to/report.html"
        assert result.success is True
        assert result.generation_time_seconds == 1.5
        assert result.sections_included == ["executive_summary", "field_analysis"]
        assert result.errors == []  # Default empty list
        assert result.metadata == {"document_count": 5}
    
    def test_failed_result(self):
        """Test failed report result."""
        result = ReportResult(
            output_path="/path/to/report.html",
            success=False,
            generation_time_seconds=0.5,
            sections_included=[],
            errors=["File permission denied", "Invalid data format"]
        )
        
        assert result.success is False
        assert len(result.errors) == 2
        assert "File permission denied" in result.errors
        assert "Invalid data format" in result.errors
    
    def test_summary_successful(self):
        """Test summary for successful result."""
        result = ReportResult(
            output_path="/path/to/report.html",
            success=True,
            generation_time_seconds=2.34,
            sections_included=["executive_summary", "field_analysis", "confusion_matrix"]
        )
        
        summary = result.summary()
        assert "✓ Success" in summary
        assert "2.34s" in summary
        assert "3 sections" in summary
        assert "errors" not in summary
    
    def test_summary_failed(self):
        """Test summary for failed result."""
        result = ReportResult(
            output_path="/path/to/report.html",
            success=False,
            generation_time_seconds=0.1,
            sections_included=[],
            errors=["Error 1", "Error 2"]
        )
        
        summary = result.summary()
        assert "✗ Failed" in summary
        assert "0.10s" in summary
        assert "0 sections" in summary
        assert "2 errors" in summary
    
    def test_default_values(self):
        """Test default values for optional fields."""
        result = ReportResult(
            output_path="/path/to/report.html",
            success=True,
            generation_time_seconds=1.0,
            sections_included=["executive_summary"]
        )
        
        assert result.errors == []
        assert result.metadata == {}
