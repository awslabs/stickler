"""
Pretty print utilities for StructuredModelEvaluator results.

This module provides functions for displaying confusion matrix metrics
in a more readable and visually appealing format.
"""

import os
import re
import sys
from typing import Dict, Any, Optional, List, Union


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def _supports_color() -> bool:
    """
    Check if the terminal supports color output.

    Returns:
        bool: True if color is supported, False otherwise
    """
    # Check if the output is a terminal
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check platform-specific terminal types that support color
    plat = sys.platform
    supported_platform = plat != "win32" or "ANSICON" in os.environ

    # Only support color if we're on a supported platform and not in a Jupyter notebook
    # (We'll handle Jupyter HTML output separately)
    is_jupyter = "ipykernel" in sys.modules

    return supported_platform and not is_jupyter


def _colorize(text: str, color: str, use_color: bool = True) -> str:
    """
    Colorize text if color support is enabled.

    Args:
        text: Text to colorize
        color: Color code from the Colors class
        use_color: Whether to use color (overrides auto-detection)

    Returns:
        str: Colorized text if supported, original text otherwise
    """
    if use_color and _supports_color():
        return f"{color}{text}{Colors.RESET}"
    return text


def _format_percentage(value: float) -> str:
    """
    Format a float as a percentage with 2 decimal places.

    Args:
        value: Float value between 0 and 1

    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.2f}%"


def _format_float(value: float, precision: int = 4) -> str:
    """
    Format a float value with specified precision.

    Args:
        value: Float value to format
        precision: Number of decimal places

    Returns:
        str: Formatted float string
    """
    format_str = f"{{:.{precision}f}}"
    return format_str.format(value)


def _create_bar(value: float, width: int = 20, use_color: bool = True) -> str:
    """
    Create a visual bar representing a value between 0 and 1.

    Args:
        value: Value between 0 and 1
        width: Width of the bar in characters
        use_color: Whether to use color

    Returns:
        str: Visual bar representation
    """
    if not (0 <= value <= 1):
        value = max(0, min(value, 1))  # Clamp to [0,1]

    filled_width = int(value * width)
    empty_width = width - filled_width

    # Select color based on value
    if value < 0.5:
        color = Colors.RED
    elif value < 0.8:
        color = Colors.YELLOW
    else:
        color = Colors.GREEN

    # Create the bar
    if use_color and _supports_color():
        bar = f"{color}{'█' * filled_width}{Colors.RESET}{'░' * empty_width}"
    else:
        bar = f"{'█' * filled_width}{'░' * empty_width}"

    return bar


def _create_header(text: str, char: str = "=", use_color: bool = True) -> str:
    """
    Create a section header with decorative characters.

    Args:
        text: Header text
        char: Character to use for decoration
        use_color: Whether to use color

    Returns:
        str: Formatted header
    """
    padding = char * 3
    header = f"{padding} {text} {padding}"

    if use_color and _supports_color():
        return f"{Colors.BOLD}{Colors.BLUE}{header}{Colors.RESET}"
    return header


def _format_field_name(name: str, use_color: bool = True) -> str:
    """
    Format a field name with appropriate styling.

    Args:
        name: Field name
        use_color: Whether to use color

    Returns:
        str: Formatted field name
    """
    if use_color and _supports_color():
        return f"{Colors.BOLD}{name}{Colors.RESET}"
    return name


def _format_nested_field_name(name: str, level: int = 0, use_color: bool = True) -> str:
    """
    Format a nested field name with appropriate indentation and styling.

    Args:
        name: Field name
        level: Nesting level
        use_color: Whether to use color

    Returns:
        str: Formatted nested field name
    """
    indent = "  " * level
    parts = name.split(".")
    last_part = parts[-1]

    if use_color and _supports_color():
        return f"{indent}{Colors.BOLD}{last_part}{Colors.RESET}"
    return f"{indent}{last_part}"


def _normalize_results_format(
    results: Union[Dict[str, Any], Any],
) -> Optional[Dict[str, Any]]:
    """
    Normalize different result formats to the expected confusion matrix structure.

    Args:
        results: Either single document results dict or ProcessEvaluation from bulk evaluator

    Returns:
        Normalized results dict or None if unrecognized format
    """
    # Handle ProcessEvaluation objects (from bulk evaluator)
    if hasattr(results, "metrics") and hasattr(results, "field_metrics"):
        # This is a ProcessEvaluation object from bulk evaluator
        overall_metrics = results.metrics
        field_metrics = results.field_metrics

        # Create derived metrics structure for overall metrics
        overall_derived = {
            "cm_precision": overall_metrics.get("cm_precision", 0.0),
            "cm_recall": overall_metrics.get("cm_recall", 0.0),
            "cm_f1": overall_metrics.get("cm_f1", 0.0),
            "cm_accuracy": overall_metrics.get("cm_accuracy", 0.0),
        }

        # Create normalized field structure
        normalized_fields = {}
        for field_name, field_data in field_metrics.items():
            # Extract base confusion matrix values and derived metrics
            normalized_fields[field_name] = {
                "tp": field_data.get("tp", 0),
                "fp": field_data.get("fp", 0),
                "tn": field_data.get("tn", 0),
                "fn": field_data.get("fn", 0),
                "fd": field_data.get("fd", 0),
                "fa": field_data.get("fa", 0),
                "derived": {
                    "cm_precision": field_data.get("cm_precision", 0.0),
                    "cm_recall": field_data.get("cm_recall", 0.0),
                    "cm_f1": field_data.get("cm_f1", 0.0),
                    "cm_accuracy": field_data.get("cm_accuracy", 0.0),
                },
            }

        return {
            "confusion_matrix": {
                "overall": {
                    "tp": overall_metrics.get("tp", 0),
                    "fp": overall_metrics.get("fp", 0),
                    "tn": overall_metrics.get("tn", 0),
                    "fn": overall_metrics.get("fn", 0),
                    "fd": overall_metrics.get("fd", 0),
                    "fa": overall_metrics.get("fa", 0),
                    "derived": overall_derived,
                },
                "fields": normalized_fields,
            },
            "bulk_info": {
                "total_time": getattr(results, "total_time", 0),
                "error_count": len(getattr(results, "errors", [])),
                "total_fields": len(field_metrics),
            },
        }

    # Handle regular single document results
    elif isinstance(results, dict) and "confusion_matrix" in results:
        return results

    # Handle direct metrics dict (fallback)
    elif isinstance(results, dict) and "tp" in results:
        # This might be a direct metrics dict, wrap it
        return {
            "confusion_matrix": {
                "overall": {
                    **results,
                    "derived": {
                        "cm_precision": results.get("cm_precision", 0.0),
                        "cm_recall": results.get("cm_recall", 0.0),
                        "cm_f1": results.get("cm_f1", 0.0),
                        "cm_accuracy": results.get("cm_accuracy", 0.0),
                    },
                },
                "fields": {},
            }
        }

    return None


def print_confusion_matrix(
    results: Union[Dict[str, Any], Any],
    field_filter: Optional[str] = None,
    sort_by: str = "name",
    show_details: bool = True,
    use_color: bool = True,
    output_file: Optional[str] = None,
    nested_detail: str = "standard",
) -> None:
    """
    Pretty print confusion matrix metrics in a readable, visually appealing format.

    Args:
        results: Results from StructuredModelEvaluator.evaluate() or ProcessEvaluation from bulk evaluator
        field_filter: Optional regex to filter fields to display
        sort_by: How to sort fields ('name', 'precision', 'recall', 'f1', etc.)
        show_details: Whether to show detailed metrics for each field
        use_color: Whether to use color in the output
        output_file: Optional file path to write the output to
        nested_detail: Level of detail for nested objects:
                       'minimal' - Show only top-level fields
                       'standard' - Show nested fields with basic metrics (default)
                       'detailed' - Show comprehensive metrics for nested fields and their items
    """
    # Normalize results format
    normalized_results = _normalize_results_format(results)
    if normalized_results is None:
        print("Error: Results do not contain recognizable confusion matrix metrics")
        return

    # Use normalized results for processing
    results = normalized_results

    # Direct output to file if specified
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                original_stdout = sys.stdout
                sys.stdout = f
                use_color = False  # Disable color for file output

                # Print overall summary
                _print_overall_summary(results, use_color)

                # Print field-level metrics if requested
                if show_details:
                    _print_field_details(
                        results, field_filter, sort_by, use_color, nested_detail
                    )

                    # Print matrix visualization
                    _print_matrix_visualization(results, use_color)

                # Restore stdout before context manager closes
                sys.stdout = original_stdout
        except Exception as e:
            print(f"Error opening output file: {e}")
            return
    else:
        # Print to stdout
        # Print overall summary
        _print_overall_summary(results, use_color)

        # Print bulk evaluation info if available
        if "bulk_info" in results:
            _print_bulk_info_summary(results, use_color)

        # Print field-level metrics if requested
        if show_details:
            _print_field_details(
                results, field_filter, sort_by, use_color, nested_detail
            )

            # Print matrix visualization
            _print_matrix_visualization(results, use_color)


def _print_overall_summary(results: Dict[str, Any], use_color: bool = True) -> None:
    """
    Print overall summary of confusion matrix metrics.

    Args:
        results: Results dictionary from StructuredModelEvaluator.evaluate()
        use_color: Whether to use color in the output
    """
    cm = results["confusion_matrix"]["overall"]

    print(_create_header("CONFUSION MATRIX SUMMARY", "=", use_color))
    print()

    # Raw counts
    print(_create_header("Raw Counts", "-", use_color))

    row_format = "{:<15} {:>8}"
    print(row_format.format("Metric", "Count"))
    print("-" * 25)

    metrics = {
        "True Positive": cm["tp"],
        "False Positive": cm["fp"],
        "True Negative": cm["tn"],
        "False Negative": cm["fn"],
        "False Discovery": cm["fd"],
    }

    for name, value in metrics.items():
        print(row_format.format(name, value))

    print()

    # Derived metrics
    derived = cm["derived"]
    print(_create_header("Derived Metrics", "-", use_color))

    row_format = "{:<15} {:>10} {:<22}"
    print(row_format.format("Metric", "Value", "Visual"))
    print("-" * 50)

    derived_metrics = {
        "Precision": derived["cm_precision"],
        "Recall": derived["cm_recall"],
        "F1 Score": derived["cm_f1"],
        "Accuracy": derived["cm_accuracy"],
    }

    for name, value in derived_metrics.items():
        bar = _create_bar(value, width=20, use_color=use_color)
        print(row_format.format(name, _format_percentage(value), bar))

    print("\n")


def _print_bulk_info_summary(results: Dict[str, Any], use_color: bool = True) -> None:
    """
    Print bulk evaluation summary information.

    Args:
        results: Normalized results dictionary containing bulk_info
        use_color: Whether to use color in the output
    """
    bulk_info = results.get("bulk_info", {})

    print(_create_header("BULK EVALUATION SUMMARY", "=", use_color))
    print()

    row_format = "{:<20} {:>12}"
    print(row_format.format("Metric", "Value"))
    print("-" * 35)

    # Total processing time
    total_time = bulk_info.get("total_time", 0)
    if total_time > 60:
        time_str = f"{total_time / 60:.1f} minutes"
    else:
        time_str = f"{total_time:.2f} seconds"
    print(row_format.format("Processing Time", time_str))

    # Number of fields evaluated
    total_fields = bulk_info.get("total_fields", 0)
    print(row_format.format("Total Fields", total_fields))

    # Error count
    error_count = bulk_info.get("error_count", 0)
    if error_count > 0:
        error_text = _colorize(str(error_count), Colors.YELLOW, use_color)
    else:
        error_text = _colorize("0", Colors.GREEN, use_color)
    print(row_format.format("Errors", error_text))

    print("\n")


def _print_field_details(
    results: Dict[str, Any],
    field_filter: Optional[str] = None,
    sort_by: str = "name",
    use_color: bool = True,
    nested_detail: str = "standard",
) -> None:
    """
    Print detailed field-level confusion matrix metrics.

    Args:
        results: Results dictionary from StructuredModelEvaluator.evaluate()
        field_filter: Optional regex to filter fields to display
        sort_by: How to sort fields
        use_color: Whether to use color in the output
        nested_detail: Level of detail for nested objects ('minimal', 'standard', or 'detailed')
    """
    cm_fields = results["confusion_matrix"]["fields"]

    print(_create_header("FIELD-LEVEL METRICS", "=", use_color))
    print()

    # Filter fields if requested
    if field_filter:
        pattern = re.compile(field_filter)
        cm_fields = {k: v for k, v in cm_fields.items() if pattern.search(k)}

    # Sort fields
    if sort_by == "name":
        sorted_fields = sorted(cm_fields.items())
    else:
        # For sorting by metrics
        metric_key = (
            f"cm_{sort_by}"
            if sort_by in ["precision", "recall", "f1", "accuracy"]
            else "cm_f1"
        )

        def get_metric(field_data):
            return field_data[1]["derived"].get(metric_key, 0)

        sorted_fields = sorted(cm_fields.items(), key=get_metric, reverse=True)

    # Prepare format strings for the table
    name_width = max(max(len(name) for name in cm_fields.keys()), 20)
    row_format = (
        "{:<"
        + str(name_width)
        + "} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:<22}"
    )

    # Print header
    print(
        row_format.format(
            "Field",
            "TP",
            "FP",
            "TN",
            "FN",
            "FD",
            "Prec",
            "Recall",
            "F1",
            "Acc",
            "Visual",
        )
    )
    print("-" * (name_width + 105))

    # Group fields by top-level name for hierarchical display
    grouped_fields = {}

    for name, field_data in sorted_fields:
        top_level = name.split(".")[0]
        if top_level not in grouped_fields:
            grouped_fields[top_level] = []
        grouped_fields[top_level].append((name, field_data))

    # Print each group
    for top_level, fields in grouped_fields.items():
        # Determine which fields are containers (have nested fields under them)
        field_children = {}
        for name, field_data in fields:
            parts = name.split(".")
            if len(parts) > 1:
                parent = ".".join(parts[:-1])
                if parent not in field_children:
                    field_children[parent] = []
                field_children[parent].append(name)

        # Print top-level field
        for name, field_data in fields:
            if name == top_level:  # This is the top-level field itself
                is_container = top_level in field_children
                _print_field_row(
                    name, field_data, row_format, 0, use_color, is_container
                )
                break

        # Handle nested fields based on the detail level
        if nested_detail != "minimal":
            # Print nested fields
            for name, field_data in fields:
                if name != top_level and "." in name:  # This is a nested field
                    parts = name.split(".")
                    level = len(parts) - 1
                    is_container = name in field_children
                    _print_field_row(
                        name, field_data, row_format, level, use_color, is_container
                    )

        # If detailed mode and this is a list field, show item-specific metrics
        if (
            nested_detail == "detailed"
            and "fields" in results
            and top_level in results["fields"]
        ):
            field_info = results["fields"][top_level]
            if isinstance(field_info, dict) and "items" in field_info:
                _print_list_field_details(top_level, field_info, use_color)

        print()  # Add space between groups

    print("\n")


def _print_field_row(
    name: str,
    field_data: Dict[str, Any],
    row_format: str,
    level: int = 0,
    use_color: bool = True,
    is_container: bool = False,
) -> None:
    """
    Print a single row for a field in the metrics table.

    Args:
        name: Field name
        field_data: Field metrics data
        row_format: Format string for the row
        level: Nesting level (for indentation)
        use_color: Whether to use color in the output
        is_container: Whether this field contains nested fields
    """
    # Format the field name based on nesting level
    if level > 0:
        base_name = _format_nested_field_name(name, level, use_color)
    else:
        base_name = _format_field_name(name, use_color)

    # Add container indicator for object fields
    if is_container:
        if use_color and _supports_color():
            obj_indicator = _colorize(" {obj}", Colors.CYAN, use_color)
        else:
            obj_indicator = " {obj}"
        display_name = base_name + obj_indicator
    else:
        display_name = base_name

    # Get metrics
    tp = field_data.get("tp", 0)
    fp = field_data.get("fp", 0)
    tn = field_data.get("tn", 0)
    fn = field_data.get("fn", 0)
    fd = field_data.get("fd", 0)

    # Get derived metrics
    derived = field_data.get("derived", {})
    precision = derived.get("cm_precision", 0)
    recall = derived.get("cm_recall", 0)
    f1 = derived.get("cm_f1", 0)
    accuracy = derived.get("cm_accuracy", 0)

    # Create visual bar for F1 score
    bar = _create_bar(f1, width=20, use_color=use_color)

    # Print the row
    print(
        row_format.format(
            display_name,
            tp,
            fp,
            tn,
            fn,
            fd,
            _format_percentage(precision),
            _format_percentage(recall),
            _format_percentage(f1),
            _format_percentage(accuracy),
            bar,
        )
    )


def _print_matrix_visualization(
    results: Dict[str, Any], use_color: bool = True
) -> None:
    """
    Print a visual representation of the confusion matrix.

    Args:
        results: Results dictionary from StructuredModelEvaluator.evaluate()
        use_color: Whether to use color in the output
    """
    cm = results["confusion_matrix"]["overall"]

    tp = cm.get("tp", 0)
    fp = cm.get("fp", 0)
    tn = cm.get("tn", 0)
    fn = cm.get("fn", 0)
    fd = cm.get("fd", 0)

    total = tp + fp + tn + fn + fd
    if total == 0:
        return  # Avoid division by zero

    print(_create_header("CONFUSION MATRIX VISUALIZATION", "=", use_color))
    print()

    matrix_width = 40

    # Calculate cell widths proportional to counts
    cell_tp = int((tp / total) * matrix_width) if tp > 0 else 0
    cell_tn = int((tn / total) * matrix_width) if tn > 0 else 0
    cell_fp = int((fp / total) * matrix_width) if fp > 0 else 0
    cell_fn = int((fn / total) * matrix_width) if fn > 0 else 0
    cell_fd = int((fd / total) * matrix_width) if fd > 0 else 0

    # Ensure at least 1 character for non-zero values
    # Create dictionaries to safely track cell widths and corresponding counts
    cell_data = {
        "cell_tp": cell_tp,
        "cell_tn": cell_tn,
        "cell_fp": cell_fp,
        "cell_fn": cell_fn,
        "cell_fd": cell_fd,
    }

    count_data = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "fd": fd}

    # Ensure at least 1 character for non-zero values
    for cell_key in cell_data:
        count_key = cell_key.replace("cell_", "")
        if count_data[count_key] > 0 and cell_data[cell_key] == 0:
            cell_data[cell_key] = 1

    # Update the original variables
    cell_tp = cell_data["cell_tp"]
    cell_tn = cell_data["cell_tn"]
    cell_fp = cell_data["cell_fp"]
    cell_fn = cell_data["cell_fn"]
    cell_fd = cell_data["cell_fd"]

    # Adjust to ensure total width is maintained
    current_width = cell_tp + cell_tn + cell_fp + cell_fn + cell_fd
    if current_width < matrix_width:
        # Add extra to largest cell
        max_cell = max([cell_tp, cell_tn, cell_fp, cell_fn, cell_fd])
        if max_cell == cell_tp:
            cell_tp += matrix_width - current_width
        elif max_cell == cell_tn:
            cell_tn += matrix_width - current_width
        elif max_cell == cell_fp:
            cell_fp += matrix_width - current_width
        elif max_cell == cell_fn:
            cell_fn += matrix_width - current_width
        else:
            cell_fd += matrix_width - current_width

    # Create the matrix visualization
    if use_color and _supports_color():
        tp_cell = (
            f"{Colors.BG_GREEN}{' ' * cell_tp}{Colors.RESET}" if cell_tp > 0 else ""
        )
        tn_cell = (
            f"{Colors.BG_BLUE}{' ' * cell_tn}{Colors.RESET}" if cell_tn > 0 else ""
        )
        fp_cell = f"{Colors.BG_RED}{' ' * cell_fp}{Colors.RESET}" if cell_fp > 0 else ""
        fn_cell = (
            f"{Colors.BG_YELLOW}{' ' * cell_fn}{Colors.RESET}" if cell_fn > 0 else ""
        )
        fd_cell = (
            f"{Colors.BG_MAGENTA}{' ' * cell_fd}{Colors.RESET}" if cell_fd > 0 else ""
        )
    else:
        tp_cell = "T" * cell_tp if cell_tp > 0 else ""
        tn_cell = "N" * cell_tn if cell_tn > 0 else ""
        fp_cell = "F" * cell_fp if cell_fp > 0 else ""
        fn_cell = "M" * cell_fn if cell_fn > 0 else ""  # M for "Miss" (False Negative)
        fd_cell = (
            "D" * cell_fd if cell_fd > 0 else ""
        )  # D for "Discovery" (False Discovery)

    # Print matrix
    print(f"{tp_cell}{tn_cell}{fp_cell}{fn_cell}{fd_cell}")

    # Print legend
    print("\nLegend:")
    if use_color and _supports_color():
        print(
            f"  {Colors.BG_GREEN}  {Colors.RESET} True Positive (TP): {tp} ({_format_percentage(tp / total if total else 0)})"
        )
        print(
            f"  {Colors.BG_BLUE}  {Colors.RESET} True Negative (TN): {tn} ({_format_percentage(tn / total if total else 0)})"
        )
        print(
            f"  {Colors.BG_RED}  {Colors.RESET} False Positive (FP): {fp} ({_format_percentage(fp / total if total else 0)})"
        )
        print(
            f"  {Colors.BG_YELLOW}  {Colors.RESET} False Negative (FN): {fn} ({_format_percentage(fn / total if total else 0)})"
        )
        print(
            f"  {Colors.BG_MAGENTA}  {Colors.RESET} False Discovery (FD): {fd} ({_format_percentage(fd / total if total else 0)})"
        )
    else:
        print(
            f"  T True Positive (TP): {tp} ({_format_percentage(tp / total if total else 0)})"
        )
        print(
            f"  N True Negative (TN): {tn} ({_format_percentage(tn / total if total else 0)})"
        )
        print(
            f"  F False Positive (FP): {fp} ({_format_percentage(fp / total if total else 0)})"
        )
        print(
            f"  M False Negative (FN): {fn} ({_format_percentage(fn / total if total else 0)})"
        )
        print(
            f"  D False Discovery (FD): {fd} ({_format_percentage(fd / total if total else 0)})"
        )

    print("\n")


def _print_list_field_details(
    field_name: str, field_info: Dict[str, Any], use_color: bool = True
) -> None:
    """
    Print holistic metrics for list field items instead of individual items.

    Args:
        field_name: Name of the list field
        field_info: Field metrics information including 'items' list
        use_color: Whether to use color in the output
    """
    if not field_info.get("items"):
        return

    items = field_info["items"]

    # Print list field header
    print(
        f"\n  {_colorize('List Field Summary for', Colors.CYAN, use_color)} "
        f"{_colorize(field_name, Colors.BOLD, use_color)} "
        f"({len(items)} items):"
    )

    # Calculate aggregated metrics across all items
    avg_precision = 0.0
    avg_recall = 0.0
    avg_f1 = 0.0
    avg_score = 0.0

    # Count items with metrics available
    valid_items = 0

    # Aggregate field-level metrics
    field_metrics = {}

    for item in items:
        if "overall" not in item:
            continue

        valid_items += 1
        item_metrics = item["overall"]

        # Aggregate overall metrics
        avg_precision += item_metrics.get("precision", 0.0)
        avg_recall += item_metrics.get("recall", 0.0)
        avg_f1 += item_metrics.get("f1", 0.0)
        avg_score += item_metrics.get("anls_score", 0.0)

        # Aggregate field-level metrics
        if "fields" in item:
            for field_name, field_metrics_data in item["fields"].items():
                if field_name not in field_metrics:
                    field_metrics[field_name] = {"score_sum": 0.0, "count": 0}

                field_score = field_metrics_data.get("anls_score", 0.0)
                field_metrics[field_name]["score_sum"] += field_score
                field_metrics[field_name]["count"] += 1

    # Calculate averages
    if valid_items > 0:
        avg_precision /= valid_items
        avg_recall /= valid_items
        avg_f1 /= valid_items
        avg_score /= valid_items

    # Print summary metrics
    print(f"  Items: {valid_items}")
    print(f"  Average ANLS Score: {avg_score:.4f}")
    print(f"  Average Precision: {_format_percentage(avg_precision)}")
    print(f"  Average Recall: {_format_percentage(avg_recall)}")

    # Show F1 score with visual bar
    f1_bar = _create_bar(avg_f1, width=20, use_color=use_color)
    print(f"  Average F1 Score: {_format_percentage(avg_f1)} {f1_bar}")

    # Print field-level average scores
    if field_metrics:
        print("\n  Field-level average scores:")
        for field_name, data in sorted(field_metrics.items()):
            avg_field_score = (
                data["score_sum"] / data["count"] if data["count"] > 0 else 0
            )
            field_color = (
                Colors.GREEN
                if avg_field_score >= 0.8
                else (Colors.YELLOW if avg_field_score >= 0.5 else Colors.RED)
            )

            print(
                f"    {field_name}: {_colorize(f'{avg_field_score:.4f}', field_color, use_color)}"
            )

    print()


def print_non_matches(
    results: Union[Dict[str, Any], Any],
    group_by: str = "type",
    field_filter: Optional[str] = None,
    show_summary: bool = True,
    use_color: bool = True,
    max_items: Optional[int] = None,
    output_file: Optional[str] = None,
) -> None:
    """
    Pretty print non-match information in a readable, visually appealing format.

    Args:
        results: Results from StructuredModelEvaluator.evaluate() or ProcessEvaluation from bulk evaluator
        group_by: How to group non-matches ('type', 'field', 'document')
        field_filter: Optional regex to filter fields to display
        show_summary: Whether to show summary statistics
        use_color: Whether to use color in the output
        max_items: Maximum number of non-matches to display per group
        output_file: Optional file path to write the output to
    """
    # Normalize results format and extract non-matches
    non_matches = _extract_non_matches(results)
    if not non_matches:
        if show_summary:
            print(
                _colorize(
                    "✅ No non-matches found - all fields matched successfully!",
                    Colors.GREEN,
                    use_color,
                )
            )
        return

    # Direct output to file if specified
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                original_stdout = sys.stdout
                sys.stdout = f
                use_color = False  # Disable color for file output

                _print_non_matches_content(
                    non_matches,
                    group_by,
                    field_filter,
                    show_summary,
                    use_color,
                    max_items,
                )

                # Restore stdout before context manager closes
                sys.stdout = original_stdout
        except Exception as e:
            print(f"Error opening output file: {e}")
            return
    else:
        _print_non_matches_content(
            non_matches, group_by, field_filter, show_summary, use_color, max_items
        )


def _extract_non_matches(results: Union[Dict[str, Any], Any]) -> List[Dict[str, Any]]:
    """
    Extract non-matches from various result formats.

    Args:
        results: Results from compare_with or bulk evaluator

    Returns:
        List of non-match dictionaries
    """
    # Handle ProcessEvaluation objects (from bulk evaluator)
    if hasattr(results, "non_matches") and results.non_matches:
        return results.non_matches

    # Handle regular single document results
    elif isinstance(results, dict):
        if "non_matches" in results and results["non_matches"]:
            return results["non_matches"]
        elif (
            "confusion_matrix" in results
            and "non_matches" in results["confusion_matrix"]
        ):
            return results["confusion_matrix"]["non_matches"]

    return []


def _print_non_matches_content(
    non_matches: List[Dict[str, Any]],
    group_by: str,
    field_filter: Optional[str],
    show_summary: bool,
    use_color: bool,
    max_items: Optional[int],
) -> None:
    """
    Print the actual non-matches content with formatting.
    """
    # Filter non-matches if requested
    if field_filter:
        pattern = re.compile(field_filter)
        non_matches = [
            nm for nm in non_matches if pattern.search(nm.get("field_path", ""))
        ]

    if show_summary:
        _print_non_matches_summary(non_matches, use_color)

    # Group non-matches
    grouped_non_matches = _group_non_matches(non_matches, group_by)

    # Print each group
    for group_name, group_items in grouped_non_matches.items():
        _print_non_match_group(group_name, group_items, group_by, use_color, max_items)


def _print_non_matches_summary(
    non_matches: List[Dict[str, Any]], use_color: bool = True
) -> None:
    """
    Print summary statistics for non-matches.
    """
    print(_create_header("NON-MATCH ANALYSIS SUMMARY", "=", use_color))
    print()

    total_count = len(non_matches)

    # Count by type
    type_counts = {}
    doc_counts = {}

    for nm in non_matches:
        nm_type = nm.get("non_match_type", "unknown")
        type_counts[nm_type] = type_counts.get(nm_type, 0) + 1

        doc_id = nm.get("doc_id")
        if doc_id:
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    # Print total and document info
    if doc_counts:
        doc_count = len(doc_counts)
        print(
            f"Total: {_colorize(str(total_count), Colors.BOLD, use_color)} non-matches across {_colorize(str(doc_count), Colors.BOLD, use_color)} documents"
        )
    else:
        print(
            f"Total: {_colorize(str(total_count), Colors.BOLD, use_color)} non-matches"
        )

    print()

    # Print type breakdown with visual indicators
    for nm_type, count in sorted(type_counts.items()):
        percentage = (count / total_count) * 100 if total_count > 0 else 0

        # Choose appropriate emoji and color for each type
        if nm_type == "false_discovery" or nm_type == "NonMatchType.FALSE_DISCOVERY":
            icon = "🔍"
            color = Colors.YELLOW
            description = "Below Threshold"
        elif nm_type == "false_negative" or nm_type == "NonMatchType.FALSE_NEGATIVE":
            icon = "❌"
            color = Colors.RED
            description = "Missing Predictions"
        elif nm_type == "false_alarm" or nm_type == "NonMatchType.FALSE_ALARM":
            icon = "⚠️"
            color = Colors.MAGENTA
            description = "Extra Predictions"
        else:
            icon = "❓"
            color = Colors.WHITE
            description = nm_type.replace("_", " ").title()

        type_display = _colorize(f"{description} ({nm_type})", color, use_color)
        count_display = _colorize(
            f"{count} items ({percentage:.1f}%)", Colors.BOLD, use_color
        )

        print(f"{icon} {type_display}: {count_display}")

    print("\n")


def _group_non_matches(
    non_matches: List[Dict[str, Any]], group_by: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group non-matches by the specified criteria.
    """
    groups = {}

    for nm in non_matches:
        if group_by == "type":
            key = nm.get("non_match_type", "unknown")
        elif group_by == "field":
            key = nm.get("field_path", "unknown")
        elif group_by == "document":
            key = nm.get("doc_id", "single_document")
        else:
            key = nm.get("non_match_type", "unknown")  # Default to type

        if key not in groups:
            groups[key] = []
        groups[key].append(nm)

    return groups


def _print_non_match_group(
    group_name: str,
    group_items: List[Dict[str, Any]],
    group_by: str,
    use_color: bool,
    max_items: Optional[int],
) -> None:
    """
    Print a group of non-matches with appropriate formatting.
    """
    # Determine group header and styling
    if group_by == "type":
        if (
            group_name == "false_discovery"
            or group_name == "NonMatchType.FALSE_DISCOVERY"
        ):
            header = (
                f"🔍 FALSE DISCOVERIES (Below Threshold) - {len(group_items)} items"
            )
            color = Colors.YELLOW
        elif (
            group_name == "false_negative"
            or group_name == "NonMatchType.FALSE_NEGATIVE"
        ):
            header = (
                f"❌ FALSE NEGATIVES (Missing Predictions) - {len(group_items)} items"
            )
            color = Colors.RED
        elif group_name == "false_alarm" or group_name == "NonMatchType.FALSE_ALARM":
            header = f"⚠️ FALSE ALARMS (Extra Predictions) - {len(group_items)} items"
            color = Colors.MAGENTA
        else:
            header = f"❓ {group_name.upper()} - {len(group_items)} items"
            color = Colors.WHITE
    elif group_by == "field":
        header = f"📍 Field: {group_name} - {len(group_items)} non-matches"
        color = Colors.CYAN
    elif group_by == "document":
        header = f"📄 Document: {group_name} - {len(group_items)} non-matches"
        color = Colors.BLUE
    else:
        header = f"{group_name} - {len(group_items)} items"
        color = Colors.WHITE

    print(_colorize(header, color, use_color))
    print(
        _colorize(
            "─"
            * len(
                header.replace("🔍 ", "")
                .replace("❌ ", "")
                .replace("⚠️ ", "")
                .replace("❓ ", "")
                .replace("📍 ", "")
                .replace("📄 ", "")
            ),
            color,
            use_color,
        )
    )

    # Limit items if requested
    items_to_show = group_items[:max_items] if max_items else group_items

    for nm in items_to_show:
        _print_single_non_match(nm, use_color)

    # Show truncation message if needed
    if max_items and len(group_items) > max_items:
        remaining = len(group_items) - max_items
        print(f"   ... and {remaining} more items (use max_items=None to show all)")

    print()


def _print_single_non_match(nm: Dict[str, Any], use_color: bool) -> None:
    """
    Print a single non-match with detailed information.
    """
    field_path = nm.get("field_path", "unknown")
    gt_value = nm.get("ground_truth_value")
    pred_value = nm.get("prediction_value")
    similarity_score = nm.get("similarity_score")
    details = nm.get("details", {})
    doc_id = nm.get("doc_id")

    # Format field path with indentation for nested fields
    if "." in field_path:
        parts = field_path.split(".")
        indent = "  " * (len(parts) - 1)
        display_path = f"{indent}└─ {parts[-1]}"
    else:
        display_path = field_path

    # Format the main comparison line
    if pred_value is None:
        comparison = f'"{gt_value}" → null'
    elif gt_value is None:
        comparison = f'null → "{pred_value}"'
    else:
        comparison = f'"{gt_value}" → "{pred_value}"'

    field_color = Colors.CYAN if use_color else ""
    print(f"📍 {_colorize(display_path, field_color, use_color)}: {comparison}")

    # Add similarity score and threshold info if available
    if similarity_score is not None or details:
        info_parts = []

        if similarity_score is not None:
            info_parts.append(f"Similarity: {similarity_score:.3f}")

        if details and "reason" in details:
            info_parts.append(f"Reason: {details['reason']}")

        if info_parts:
            info_text = " | ".join(info_parts)
            print(f"   └─ {_colorize(info_text, Colors.BRIGHT_BLACK, use_color)}")

    # Add document ID for bulk results
    if doc_id and doc_id != "single_document":
        print(
            f"   └─ {_colorize(f'Document: {doc_id}', Colors.BRIGHT_BLACK, use_color)}"
        )


def print_evaluation_results(
    results: Union[Dict[str, Any], Any],
    show_confusion_matrix: bool = True,
    show_non_matches: bool = True,
    show_aggregates: bool = True,
    use_color: bool = True,
    output_file: Optional[str] = None,
) -> None:
    """
    Print comprehensive evaluation results including confusion matrix and non-matches.

    Args:
        results: Results from StructuredModelEvaluator.evaluate() or ProcessEvaluation from bulk evaluator
        show_confusion_matrix: Whether to show confusion matrix metrics
        show_non_matches: Whether to show non-match analysis
        show_aggregates: Whether to highlight aggregate fields in confusion matrix
        use_color: Whether to use color in the output
        output_file: Optional file path to write the output to
    """
    # Direct output to file if specified
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                original_stdout = sys.stdout
                sys.stdout = f
                use_color = False  # Disable color for file output

                _print_evaluation_results_content(
                    results,
                    show_confusion_matrix,
                    show_non_matches,
                    show_aggregates,
                    use_color,
                )

                # Restore stdout before context manager closes
                sys.stdout = original_stdout
        except Exception as e:
            print(f"Error opening output file: {e}")
            return
    else:
        _print_evaluation_results_content(
            results, show_confusion_matrix, show_non_matches, show_aggregates, use_color
        )


def _print_evaluation_results_content(
    results: Union[Dict[str, Any], Any],
    show_confusion_matrix: bool,
    show_non_matches: bool,
    show_aggregates: bool,
    use_color: bool,
) -> None:
    """
    Print the actual evaluation results content.
    """
    print(_create_header("EVALUATION RESULTS", "=", use_color))
    print()

    # Show confusion matrix if requested
    if show_confusion_matrix:
        print_confusion_matrix(results, use_color=use_color, show_details=True)

    # Show non-matches if requested and available
    if show_non_matches:
        non_matches = _extract_non_matches(results)
        if non_matches:
            print_non_matches(results, use_color=use_color)
        elif (
            show_confusion_matrix
        ):  # Only show this message if we're showing other results
            print(
                _colorize(
                    "✅ No non-matches found - all fields matched successfully!",
                    Colors.GREEN,
                    use_color,
                )
            )
            print()


def print_confusion_matrix_html(
    results: Dict[str, Any],
    field_filter: Optional[str] = None,
    nested_detail: str = "standard",
) -> str:
    """
    Generate HTML representation of confusion matrix metrics for Jupyter notebooks.

    Args:
        results: Results dictionary from StructuredModelEvaluator.evaluate()
        field_filter: Optional regex to filter fields to display

    Returns:
        str: HTML string representation of the confusion matrix
    """
    # This is a placeholder for HTML output formatting
    # Implement this if you need to display results in Jupyter notebooks
    # with richer HTML formatting, tables, and visualizations

    # For now, use the same output as the terminal version
    from io import StringIO
    import sys

    # Capture output in a string
    old_stdout = sys.stdout
    mystdout = StringIO()
    sys.stdout = mystdout

    print_confusion_matrix(
        results, field_filter, use_color=False, nested_detail=nested_detail
    )

    sys.stdout = old_stdout

    # Return the captured output
    return f"<pre>{mystdout.getvalue()}</pre>"
