import argparse


def get_args(args=None):
    """
    Parse command line arguments for the KIE evaluation script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Key Information Extraction Evaluation"
    )

    # Required arguments
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file (local path or s3:// URI)",
    )

    # Flat evaluator arguments
    parser.add_argument(
        "--attributes_file",
        type=str,
        required=False,
        help="Path to attributes JSON file defining field evaluation methods",
    )

    # Structured object evaluator arguments
    parser.add_argument(
        "--model_class",
        type=str,
        required=False,
        help="Fully qualified class name of the model to evaluate",
    )

    # Optional processing arguments
    parser.add_argument(
        "--truncate",
        type=int,
        default=None,
        help="Maximum number of rows to process. Default: all rows",
    )
    parser.add_argument(
        "--true_column",
        type=str,
        default="expected",
        help="Column name for ground truth JSON. Default: 'groundtruth'",
    )
    parser.add_argument(
        "--predicted_column",
        type=str,
        default="predicted",
        help="Column name for predicted JSON. Default: 'predicted'",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results. Default: './eval_results'",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(args)
