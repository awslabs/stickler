# HTML Reporting Module

The HTML reporting module generates comprehensive, interactive HTML reports from evaluation results. It creates visual dashboards with performance metrics, field analysis, confusion matrices, and detailed breakdowns of evaluation outcomes.

## Features

- **Executive Summary** - Overall performance metrics with visual gauges
- **Field Analysis** - Per-field performance breakdown with color-coded charts
- **Confusion Matrix** - Visual representation of classification results
- **Non-Matches Analysis** - Detailed view of misclassified items
- **Interactive Elements** - Expandable sections and detailed document views

## Quick Start

```python
from stickler.reporting.html import EvaluationHTMLReporter
from stickler.reporting.html import ReportConfig

# Initialize the reporter
reporter = EvaluationHTMLReporter()

# Configure report options (optional)
config = ReportConfig(
    include_executive_summary=True,
    include_field_analysis=True,
    include_confusion_matrix=True,
    include_non_matches=True,
    max_non_matches_displayed=100
)

# Generate the report
result = reporter.generate_report(
    evaluation_results=your_evaluation_data,
    output_path="evaluation_report.html",
    config=config,
    title="My Evaluation Report"
)

# Check result
if result.success:
    print(f"Report generated successfully: {result.output_path}")
    print(f"Generation time: {result.generation_time_seconds:.2f}s")
else:
    print(f"Report generation failed: {result.errors}")
```

## Configuration Options

The `ReportConfig` class allows you to customize report generation:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_executive_summary` | bool | True | Include overall performance summary |
| `include_field_analysis` | bool | True | Include per-field performance breakdown |
| `include_confusion_matrix` | bool | True | Include confusion matrix visualization |
| `include_non_matches` | bool | True | Include detailed non-matches analysis |
| `max_non_matches_displayed` | int | 1000 | Maximum number of non-matches to show |
| `image_thumbnail_size` | int | 200 | Size of document image thumbnails (pixels) |

## Report Result

The `generate_report()` method returns a `ReportResult` object with:

```python
class ReportResult:
    output_path: str                    # Path to generated HTML file
    success: bool                       # Whether generation succeeded
    generation_time_seconds: float     # Time taken to generate report
    sections_included: List[str]        # List of sections included
    errors: List[str]                   # Any errors encountered
    metadata: Dict[str, Any]           # Additional metadata
```

## Input Data Formats

The reporter accepts two types of evaluation results:

1. **Individual Results**: Dictionary containing evaluation metrics for a single document
2. **Bulk Results**: `ProcessEvaluation` object containing aggregated metrics across multiple documents

## Generated Report Sections

- **Executive Summary**: Key performance indicators with visual gauges
- **Field Analysis**: Performance breakdown by field with charts and tables
- **Confusion Matrix**: Visual breakdown of true positives, false positives, etc.
- **Non-Matches**: Detailed table of classification errors with context

## Advanced Usage

```python
# Include document images in the report
document_images = {
    "doc_1": "/path/to/image1.jpg",
    "doc_2": "/path/to/image2.jpg"
}

# Include individual document details from JSONL file
result = reporter.generate_report(
    evaluation_results=bulk_results,
    output_path="detailed_report.html",
    config=config,
    document_images=document_images,
    individual_results_jsonl_path="individual_results.jsonl",
    model_schema=MyStructuredModel
)
```

The generated HTML report is self-contained and can be opened directly in any web browser for interactive viewing.
