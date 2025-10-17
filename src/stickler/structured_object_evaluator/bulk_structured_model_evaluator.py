#!/usr/bin/env python3

"""
Stateful Bulk Evaluator for StructuredModel objects.

This module provides a modern stateful bulk evaluator inspired by PyTorch Lightning's
stateful metrics and scikit-learn's incremental learning patterns. It supports
memory-efficient processing of large datasets through accumulation-based evaluation.
"""

import gc
import time
import csv
from tqdm import tqdm
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Type, Tuple, Union
import logging
import s3fs

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.utils.process_evaluation import ProcessEvaluation
from stickler.structured_object_evaluator.models.metrics_helper import MetricsHelper

logger = logging.getLogger(__name__)

# Choose appropriate file opening method
def open_file(path, mode, **kwargs):
    if path.startswith('s3://'):
        fs = s3fs.S3FileSystem()
        return fs.open(path, mode, **kwargs)
    else:
        return open(path, mode, **kwargs)

class BulkStructuredModelEvaluator:
    """
    Stateful bulk evaluator for StructuredModel objects.

    Inspired by PyTorch Lightning's stateful metrics and scikit-learn's incremental
    learning patterns. This evaluator accumulates evaluation state across multiple
    document processing calls, enabling memory-efficient evaluation of arbitrarily
    large datasets without loading everything into memory at once.

    Key Features:
    - Stateful accumulation (like PyTorch Lightning metrics)
    - Memory-efficient streaming processing (like scikit-learn partial_fit)
    - External control over data flow and error handling
    - Checkpointing and recovery capabilities
    - Distributed processing support via state merging
    - Uses StructuredModel.compare_with() method directly
    """

    def __init__(
        self,
        target_schema: Type[StructuredModel],
        verbose: bool = False,
        document_non_matches: bool = True,
        elide_errors: bool = False,
        individual_results_jsonl: Optional[str] = None,
        include_aggregates: bool = False,
        recall_with_fd: bool = False,
    ):
        """
        Initialize the stateful bulk evaluator.

        Args:
            target_schema: StructuredModel class for validation and processing
            verbose: Whether to print detailed progress information
            document_non_matches: Whether to document detailed non-match information
            elide_errors: If True, skip documents with errors; if False, accumulate error metrics
            individual_results_jsonl: Optional path to JSONL file for appending individual comparison results
            include_aggregates: Whether to enable aggregate metrics accumulation for performance control
        """
        self.target_schema = target_schema
        self.verbose = verbose
        self.document_non_matches = document_non_matches
        self.elide_errors = elide_errors
        self.individual_results_jsonl = individual_results_jsonl
        self.include_aggregates = include_aggregates
        self.recall_with_fd = recall_with_fd
        
        # Initialize state
        self.reset()

        if self.verbose:
            print(
                f"Initialized BulkStructuredModelEvaluator for {target_schema.__name__}"
            )
            if self.individual_results_jsonl:
                print(
                    f"Individual results will be appended to: {self.individual_results_jsonl}"
                )

    def reset(self) -> None:
        """
        Clear all accumulated state and start fresh evaluation.

        This method resets all internal counters, metrics, and error tracking
        to initial state, enabling reuse of the same evaluator instance for
        multiple evaluation runs.
        """
        # Accumulated confusion matrix state using nested defaultdicts
        self._confusion_matrix = {
            "overall": defaultdict(int),
            "fields": defaultdict(lambda: defaultdict(int))
        }
        
        # Accumulated aggregate metrics state (when include_aggregates=True)
        self._aggregate_metrics = {
            "overall": defaultdict(int),
            "fields": defaultdict(lambda: defaultdict(int))
        }

        # Non-match tracking (when document_non_matches=True)
        self._non_matches = []

        # Error tracking
        self._errors = []

        # Processing statistics
        self._processed_count = 0
        self._start_time = time.time()

        if self.verbose:
            print("Reset evaluator state")

    def update(
        self,
        gt_model: StructuredModel,
        pred_model: StructuredModel,
        doc_id: Optional[str] = None,
    ) -> None:
        """
        Process a single document pair and accumulate the results in internal state.

        This is the core method for stateful evaluation, inspired by PyTorch Lightning's
        training_step pattern. Each call processes one document pair and updates
        the internal confusion matrix counters.

        Args:
            gt_model: Ground truth StructuredModel instance
            pred_model: Predicted StructuredModel instance
            doc_id: Optional document identifier for error tracking
        """
        if doc_id is None:
            doc_id = f"doc_{self._processed_count}"

        try:
            # Use compare_with method directly on the StructuredModel
            # Pass document_non_matches to achieve parity with compare_with method
            # Pass include_aggregates to enable aggregate metrics when requested
            comparison_result = gt_model.compare_with(
                pred_model,
                include_confusion_matrix=True,
                document_non_matches=self.document_non_matches,
            )
            
            # Collect non-matches if enabled
            if self.document_non_matches and "non_matches" in comparison_result:
                # Add doc_id to each non-match for bulk tracking
                for non_match in comparison_result["non_matches"]:
                    non_match_with_doc = non_match.copy()
                    non_match_with_doc["doc_id"] = doc_id
                    self._non_matches.append(non_match_with_doc)

            # Simple JSONL append of raw comparison result (before any processing)
            if self.individual_results_jsonl:
                record = {
                    "doc_id": doc_id,
                    "comparison_result": comparison_result
                }
                with open_file(self.individual_results_jsonl, 'a', encoding = 'utf-8') as f:
                        f.write(json.dumps(record, default=str) + '\n')
            
            # Accumulate the results into our state (this flattens for aggregation)
            self._accumulate_confusion_matrix(comparison_result["confusion_matrix"])
            
            # Accumulate aggregate metrics if enabled and available
            if self.include_aggregates and "confusion_matrix" in comparison_result:
                self._accumulate_aggregate_metrics(comparison_result["confusion_matrix"])
            
            self._processed_count += 1

            if self.verbose and self._processed_count % 1000 == 0:
                elapsed = time.time() - self._start_time
                print(f"Processed {self._processed_count} documents ({elapsed:.2f}s)")

        except Exception as e:
            error_record = {
                "doc_id": doc_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            
            if not self.elide_errors:
                self._errors.append(error_record)

                # For errors, add a "failed" classification to overall metrics
                # This represents complete failure to process the document
                self._confusion_matrix["overall"]["fn"] += 1

            if self.verbose:
                print(f"Error processing document {doc_id}: {str(e)}")

    def update_batch(
        self, batch_data: List[Tuple[StructuredModel, StructuredModel, Optional[str]]]
    ) -> None:
        """
        Process multiple document pairs efficiently in a batch.
        
        This method provides efficient batch processing by calling update() 
        multiple times with optional garbage collection for memory management.
        
        Args:
            batch_data: List of tuples containing (gt_model, pred_model, doc_id)
        """
        batch_start = self._processed_count
        
        for gt_model, pred_model, doc_id in batch_data:
            self.update(gt_model, pred_model, doc_id)
        
        # Garbage collection for large batches
        if len(batch_data) >= 1000:
            gc.collect()
            
        if self.verbose:
            batch_size = self._processed_count - batch_start
            print(f"Processed batch of {batch_size} documents")

    def get_current_metrics(self) -> ProcessEvaluation:
        """
        Get current accumulated metrics without clearing state.
        
        This method allows monitoring evaluation progress by returning current
        metrics computed from accumulated state. Unlike compute(), this does
        not clear the internal state.
        
        Returns:
            ProcessEvaluation with current accumulated metrics
        """
        return self._build_process_evaluation()

    def compute(self) -> ProcessEvaluation:
        """
        Calculate final aggregated metrics from accumulated state.
        
        This method performs the final computation of all derived metrics from
        the accumulated confusion matrix state, similar to PyTorch Lightning's
        training_epoch_end pattern.
        
        Returns:
            ProcessEvaluation with final aggregated metrics
        """
        result = self._build_process_evaluation()
        
        if self.verbose:
            total_time = time.time() - self._start_time
            print(
                f"Final computation completed: {self._processed_count} documents in {total_time:.2f}s"
            )
            print(f"Overall accuracy: {result.metrics.get('cm_accuracy', 0.0):.3f}")
        
        return result

    def _accumulate_confusion_matrix(self, cm_result: Dict[str, Any]) -> None:
        """
        Accumulate confusion matrix results from a single document evaluation.
        
        This method handles the core accumulation logic, properly aggregating
        both overall metrics and field-level metrics while maintaining correct
        nested field paths.
        
        Args:
            cm_result: Confusion matrix result from compare_with method
        """
        # Accumulate overall metrics
        if "overall" in cm_result:
            for metric_name, value in cm_result["overall"].items():
                if isinstance(value, (int, float)) and metric_name in [
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                    "fd",
                    "fa",
                ]:
                    self._confusion_matrix["overall"][metric_name] += value

        # Accumulate field-level metrics with proper path handling
        if "fields" in cm_result:
            self._accumulate_field_metrics(cm_result["fields"], "")

    def _accumulate_field_metrics(
        self, fields_dict: Dict[str, Any], path_prefix: str
    ) -> None:
        """
        Recursively accumulate field-level metrics with proper nested path construction.
        
        This method fixes the nested field aggregation bugs from the original implementation
        by properly handling different field structure formats and maintaining correct
        dotted notation paths for nested fields.
        
        Args:
            fields_dict: Dictionary containing field metrics to accumulate
            path_prefix: Current path prefix for building nested field paths
        """
        for field_name, field_data in fields_dict.items():
            current_path = f"{path_prefix}.{field_name}" if path_prefix else field_name
            
            if not isinstance(field_data, dict):
                continue
            
            # Handle field with direct confusion matrix metrics (simple leaf field)
            direct_metrics = {
                k: v
                for k, v in field_data.items()
                if k in ["tp", "fp", "tn", "fn", "fd", "fa"]
                and isinstance(v, (int, float))
            }
            if direct_metrics:
                self._accumulate_single_field_metrics(current_path, direct_metrics)
            
            # Handle hierarchical field structure (object fields with overall + fields)
            if "overall" in field_data:
                # Accumulate the overall metrics for this field
                self._accumulate_single_field_metrics(
                    current_path, field_data["overall"]
                )
            
            # Handle nested fields - check if there's a "fields" structure
            if "fields" in field_data and isinstance(field_data["fields"], dict):
                # For each nested field, create the proper dotted path
                for nested_field_name, nested_field_data in field_data[
                    "fields"
                ].items():
                    nested_path = f"{current_path}.{nested_field_name}"
                    
                    if isinstance(nested_field_data, dict):
                        # If nested field has "overall", use those metrics
                        if "overall" in nested_field_data:
                            self._accumulate_single_field_metrics(
                                nested_path, nested_field_data["overall"]
                            )
                        else:
                            # Otherwise, look for direct metrics
                            nested_metrics = {
                                k: v
                                for k, v in nested_field_data.items()
                                if k in ["tp", "fp", "tn", "fn", "fd", "fa"]
                                and isinstance(v, (int, float))
                            }
                            if nested_metrics:
                                self._accumulate_single_field_metrics(
                                    nested_path, nested_metrics
                                )
                        
                        # Continue recursion if there are more nested fields
                        if "fields" in nested_field_data:
                            self._accumulate_field_metrics(
                                nested_field_data["fields"], nested_path
                            )
            
            # Handle list field structure with nested_fields
            elif "nested_fields" in field_data:
                # Accumulate list-level metrics
                list_metrics = {
                    k: v
                    for k, v in field_data.items()
                    if k in ["tp", "fp", "tn", "fn", "fd", "fa"]
                    and isinstance(v, (int, float))
                }
                if list_metrics:
                    self._accumulate_single_field_metrics(current_path, list_metrics)
                
                # Accumulate nested field metrics from the list items
                for nested_field_name, nested_metrics in field_data[
                    "nested_fields"
                ].items():
                    nested_path = f"{current_path}.{nested_field_name}"
                    self._accumulate_single_field_metrics(nested_path, nested_metrics)

    def _accumulate_single_field_metrics(
        self, field_path: str, metrics: Dict[str, Union[int, float]]
    ) -> None:
        """
        Accumulate metrics for a single field path.
        
        Args:
            field_path: Dotted path to the field (e.g., 'transactions.date')
            metrics: Dictionary of confusion matrix metrics to accumulate
        """
        for metric_name, value in metrics.items():
            if metric_name in ["tp", "fp", "tn", "fn", "fd", "fa"] and isinstance(
                value, (int, float)
            ):
                self._confusion_matrix["fields"][field_path][metric_name] += value

    def _accumulate_aggregate_metrics(self, cm_result: Dict[str, Any]) -> None:
        """
        Accumulate aggregate metrics from comparison results.
        
        This method handles accumulation of aggregate metrics that are calculated
        automatically by the universal aggregation feature. Aggregate metrics
        provide hierarchical confusion matrix counts at each level.
        
        Args:
            cm_result: Confusion matrix result containing aggregate metrics
        """
        # Accumulate root-level aggregate metrics
        if "aggregate" in cm_result and isinstance(cm_result["aggregate"], dict):
            for metric_name, value in cm_result["aggregate"].items():
                if isinstance(value, (int, float)) and metric_name in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                    self._aggregate_metrics["overall"][metric_name] += value

        # Accumulate field-level aggregate metrics with proper path handling
        if "fields" in cm_result:
            self._accumulate_field_aggregate_metrics(cm_result["fields"], "")

    def _accumulate_field_aggregate_metrics(
        self, 
        fields_dict: Dict[str, Any], 
        path_prefix: str
    ) -> None:
        """
        Recursively accumulate field-level aggregate metrics with proper nested path construction.
        
        Args:
            fields_dict: Dictionary containing field aggregate metrics to accumulate
            path_prefix: Current path prefix for building nested field paths
        """
        for field_name, field_data in fields_dict.items():
            current_path = f"{path_prefix}.{field_name}" if path_prefix else field_name
            
            if not isinstance(field_data, dict):
                continue
            
            # Handle field-level aggregate metrics
            if "aggregate" in field_data and isinstance(field_data["aggregate"], dict):
                for metric_name, value in field_data["aggregate"].items():
                    if isinstance(value, (int, float)) and metric_name in ["tp", "fp", "tn", "fn", "fd", "fa"]:
                        self._aggregate_metrics["fields"][current_path][metric_name] += value
            
            # Continue recursion if there are nested fields
            if "fields" in field_data and isinstance(field_data["fields"], dict):
                self._accumulate_field_aggregate_metrics(field_data["fields"], current_path)

    def _build_process_evaluation(self) -> ProcessEvaluation:
        """
        Build ProcessEvaluation from current accumulated state.
        
        Returns:
            ProcessEvaluation with computed metrics from accumulated state
        """
        metrics_helper = MetricsHelper()
        # Calculate derived metrics for overall results
        overall_cm = dict(self._confusion_matrix["overall"])
        overall_derived = metrics_helper.calculate_derived_metrics(overall_cm, recall_with_fd=self.recall_with_fd)
        overall_metrics = {**overall_cm, **overall_derived}
        
        # Add aggregate metrics to overall if enabled
        if self.include_aggregates:
            overall_aggregate = dict(self._aggregate_metrics["overall"])
            if any(overall_aggregate.values()):  # Only add if there are aggregate metrics
                overall_aggregate_derived = metrics_helper.calculate_derived_metrics(overall_aggregate, recall_with_fd=self.recall_with_fd)
                overall_metrics["aggregate"] = {**overall_aggregate, **overall_aggregate_derived}
        
        # Calculate derived metrics for each field
        field_metrics = {}
        for field_path, field_cm in self._confusion_matrix["fields"].items():
            field_cm_dict = dict(field_cm)
            field_derived = metrics_helper.calculate_derived_metrics(field_cm_dict, recall_with_fd=self.recall_with_fd)
            field_metrics[field_path] = {**field_cm_dict, **field_derived}
        
        # Add aggregate metrics to fields if enabled
        if self.include_aggregates:
            for field_path, field_aggregate in self._aggregate_metrics["fields"].items():
                field_aggregate_dict = dict(field_aggregate)
                if any(field_aggregate_dict.values()):  # Only add if there are aggregate metrics
                    field_aggregate_derived = metrics_helper.calculate_derived_metrics(field_aggregate_dict, recall_with_fd=self.recall_with_fd)
                    if field_path not in field_metrics:
                        field_metrics[field_path] = {}
                    field_metrics[field_path]["aggregate"] = {**field_aggregate_dict, **field_aggregate_derived}
        
        total_time = time.time() - self._start_time
        
        return ProcessEvaluation(
            metrics=overall_metrics,
            field_metrics=field_metrics,
            errors=list(self._errors),  # Copy to avoid external modification
            total_time=total_time,
            non_matches=list(self._non_matches) if self.document_non_matches else None,
        )

    def save_metrics(self, filepath: str) -> None:
        """
        Save current accumulated metrics to a JSON file.
        
        Args:
            filepath: Path where metrics will be saved as JSON
        """
        process_eval = self._build_process_evaluation()
        
        # Build comprehensive metrics dictionary
        metrics_data = {
            "overall_metrics": process_eval.metrics,
            "field_metrics": process_eval.field_metrics,
            "evaluation_summary": {
                "total_documents_processed": self._processed_count,
                "total_evaluation_time": process_eval.total_time,
                "documents_per_second": self._processed_count / process_eval.total_time
                if process_eval.total_time > 0
                else 0,
                "error_count": len(process_eval.errors),
                "error_rate": len(process_eval.errors) / self._processed_count
                if self._processed_count > 0
                else 0,
                "target_schema": self.target_schema.__name__,
            },
            "errors": process_eval.errors,
            "metadata": {
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "evaluator_config": {
                    "verbose": self.verbose,
                    "document_non_matches": self.document_non_matches,
                    "elide_errors": self.elide_errors,
                    "individual_results_jsonl": self.individual_results_jsonl,
                    "include_aggregates": self.include_aggregates
                }
            }
        }
        
        # Ensure directory exists
        import os

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Write to file
        with open_file(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        if self.verbose:
            print(f"Metrics saved to: {filepath}")

    def pretty_print_metrics(self) -> None:
        """
        Pretty print current accumulated metrics in a format similar to StructuredModel.
        
        Displays overall metrics, field-level metrics, and evaluation summary
        in a human-readable format.
        """
        process_eval = self._build_process_evaluation()
        
        # Header
        print("\n" + "=" * 80)
        print(f"BULK EVALUATION RESULTS - {self.target_schema.__name__}")
        print("=" * 80)
        
        # Overall metrics
        overall_metrics = process_eval.metrics
        print("\nOVERALL METRICS:")
        print("-" * 40)
        print(f"Documents Processed: {self._processed_count:,}")
        print(f"Evaluation Time: {process_eval.total_time:.2f}s")
        print(
            f"Processing Rate: {self._processed_count / process_eval.total_time:.1f} docs/sec"
            if process_eval.total_time > 0
            else "Processing Rate: N/A"
        )
        
        # Confusion matrix
        print(f"\nCONFUSION MATRIX:")
        print(f"  True Positives (TP):    {overall_metrics.get('tp', 0):,}")
        print(f"  False Positives (FP):   {overall_metrics.get('fp', 0):,}")  
        print(f"  True Negatives (TN):    {overall_metrics.get('tn', 0):,}")
        print(f"  False Negatives (FN):   {overall_metrics.get('fn', 0):,}")
        print(f"  False Discovery (FD):   {overall_metrics.get('fd', 0):,}")
        print(f"  False Alarm (FA):   {overall_metrics.get('fa', 0):,}")
        
        # Derived metrics
        print(f"\nDERIVED METRICS:")
        print(f"  Precision:     {overall_metrics.get('cm_precision', 0.0):.4f}")
        print(f"  Recall:        {overall_metrics.get('cm_recall', 0.0):.4f}")
        print(f"  F1 Score:      {overall_metrics.get('cm_f1', 0.0):.4f}")
        print(f"  Accuracy:      {overall_metrics.get('cm_accuracy', 0.0):.4f}")
        
        # Aggregate metrics (if enabled)
        if self.include_aggregates and "aggregate" in overall_metrics:
            aggregate = overall_metrics["aggregate"]
            print(f"\nAGGREGATE METRICS:")
            print("-" * 40)
            print(f"  True Positives (TP):    {aggregate.get('tp', 0):,}")
            print(f"  False Positives (FP):   {aggregate.get('fp', 0):,}")  
            print(f"  True Negatives (TN):    {aggregate.get('tn', 0):,}")
            print(f"  False Negatives (FN):   {aggregate.get('fn', 0):,}")
            print(f"  False Discovery (FD):   {aggregate.get('fd', 0):,}")
            print(f"  False Alarm (FA):       {aggregate.get('fa', 0):,}")
            print(f"  Precision:              {aggregate.get('cm_precision', 0.0):.4f}")
            print(f"  Recall:                 {aggregate.get('cm_recall', 0.0):.4f}")
            print(f"  F1 Score:               {aggregate.get('cm_f1', 0.0):.4f}")
            print(f"  Accuracy:               {aggregate.get('cm_accuracy', 0.0):.4f}")
        
        # Field-level metrics
        if process_eval.field_metrics:
            print(f"\nFIELD-LEVEL METRICS:")
            print("-" * 40)
            
            # Sort fields by F1 score descending for better readability
            sorted_fields = sorted(
                process_eval.field_metrics.items(),
                key=lambda x: x[1].get("cm_f1", 0.0),
                reverse=True,
            )
            
            for field_path, field_metrics in sorted_fields:
                tp = field_metrics.get("tp", 0)
                fp = field_metrics.get("fp", 0)
                fn = field_metrics.get("fn", 0)
                precision = field_metrics.get("cm_precision", 0.0)
                recall = field_metrics.get("cm_recall", 0.0)
                f1 = field_metrics.get("cm_f1", 0.0)
                
                # Only show fields with some activity
                if tp + fp + fn > 0:
                    print(
                        f"  {field_path:30} P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | TP: {tp:,} | FP: {fp:,} | FN: {fn:,}"
                    )
                    # Show aggregate metrics for this field if available
                    if self.include_aggregates and "aggregate" in field_metrics:
                        agg = field_metrics["aggregate"]
                        agg_tp = agg.get('tp', 0)
                        agg_fp = agg.get('fp', 0)
                        agg_fn = agg.get('fn', 0)
                        agg_precision = agg.get('cm_precision', 0.0)
                        agg_recall = agg.get('cm_recall', 0.0)
                        agg_f1 = agg.get('cm_f1', 0.0)
                        
                        if agg_tp + agg_fp + agg_fn > 0:
                            print(f"    └─ Aggregate:          P: {agg_precision:.3f} | R: {agg_recall:.3f} | F1: {agg_f1:.3f} | TP: {agg_tp:,} | FP: {agg_fp:,} | FN: {agg_fn:,}")
        
        # Error summary
        if process_eval.errors:
            print(f"\nERROR SUMMARY:")
            print("-" * 40)
            print(f"Total Errors: {len(process_eval.errors):,}")
            print(
                f"Error Rate: {len(process_eval.errors) / self._processed_count * 100:.2f}%"
                if self._processed_count > 0
                else "Error Rate: N/A"
            )
            
            # Group errors by type
            error_types = {}
            for error in process_eval.errors:
                error_type = error.get("error_type", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error_types:
                print("Error Types:")
                for error_type, count in sorted(
                    error_types.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"  {error_type}: {count:,}")
        
        # Configuration info
        print(f"\nCONFIGURATION:")
        print("-" * 40)
        print(f"Target Schema: {self.target_schema.__name__}")
        print(f"Document Non-matches: {'Yes' if self.document_non_matches else 'No'}")
        print(f"Elide Errors: {'Yes' if self.elide_errors else 'No'}")
        print(f"Include Aggregates: {'Yes' if self.include_aggregates else 'No'}")
        if self.individual_results_jsonl:
            print(f"Individual Results JSONL: {self.individual_results_jsonl}")
        
        print("=" * 80)

    def get_state(self) -> Dict[str, Any]:
        """
        Get serializable state for checkpointing and recovery.
        
        Returns a dictionary containing all internal state that can be serialized
        and later restored using load_state(). This enables checkpointing for
        long-running evaluation jobs.
        
        Returns:
            Dictionary containing serializable evaluator state
        """
        state = {
            "confusion_matrix": {
                "overall": dict(self._confusion_matrix["overall"]),
                "fields": {
                    path: dict(metrics)
                    for path, metrics in self._confusion_matrix["fields"].items()
                },
            },
            "errors": list(self._errors),
            "processed_count": self._processed_count,
            "start_time": self._start_time,
            # Configuration
            "target_schema": self.target_schema.__name__,
            "elide_errors": self.elide_errors,
            "include_aggregates": self.include_aggregates
        }
        
        # Include aggregate metrics if enabled
        if self.include_aggregates:
            state["aggregate_metrics"] = {
                "overall": dict(self._aggregate_metrics["overall"]),
                "fields": {path: dict(metrics) for path, metrics in self._aggregate_metrics["fields"].items()}
            }
        
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore evaluator state from serialized data.
        
        This method restores the internal state from data previously saved
        with get_state(), enabling recovery from checkpoints.
        
        Args:
            state: State dictionary from get_state()
        """
        # Validate state compatibility
        if state.get("target_schema") != self.target_schema.__name__:
            raise ValueError(
                f"State schema {state.get('target_schema')} doesn't match evaluator schema {self.target_schema.__name__}"
            )
        
        # Restore confusion matrix state
        cm_state = state["confusion_matrix"]
        self._confusion_matrix = {
            "overall": defaultdict(int, cm_state["overall"]),
            "fields": defaultdict(lambda: defaultdict(int)),
        }
        
        for field_path, field_metrics in cm_state["fields"].items():
            self._confusion_matrix["fields"][field_path] = defaultdict(
                int, field_metrics
            )
        
        # Restore aggregate metrics state if available and enabled
        if self.include_aggregates and "aggregate_metrics" in state:
            agg_state = state["aggregate_metrics"]
            self._aggregate_metrics = {
                "overall": defaultdict(int, agg_state["overall"]),
                "fields": defaultdict(lambda: defaultdict(int))
            }
            
            for field_path, field_metrics in agg_state["fields"].items():
                self._aggregate_metrics["fields"][field_path] = defaultdict(int, field_metrics)
        
        # Restore other state
        self._errors = list(state["errors"])
        self._processed_count = state["processed_count"]
        self._start_time = state["start_time"]
        
        if self.verbose:
            print(f"Loaded state: {self._processed_count} documents processed")

    def merge_state(self, other_state: Dict[str, Any]) -> None:
        """
        Merge results from another evaluator instance.
        
        This method enables distributed processing by merging confusion matrix
        counts from multiple evaluator instances that processed different
        portions of a dataset.
        
        Args:
            other_state: State dictionary from another evaluator instance
        """
        # Validate compatibility
        if other_state.get("target_schema") != self.target_schema.__name__:
            raise ValueError(
                f"Cannot merge incompatible schemas: {other_state.get('target_schema')} vs {self.target_schema.__name__}"
            )
        
        # Merge overall metrics
        other_cm = other_state["confusion_matrix"]
        for metric, value in other_cm["overall"].items():
            self._confusion_matrix["overall"][metric] += value
        
        # Merge field-level metrics
        for field_path, field_metrics in other_cm["fields"].items():
            for metric, value in field_metrics.items():
                self._confusion_matrix["fields"][field_path][metric] += value
        
        # Merge aggregate metrics if both evaluators have them enabled
        if self.include_aggregates and "aggregate_metrics" in other_state:
            other_agg = other_state["aggregate_metrics"]
            
            # Merge overall aggregate metrics
            for metric, value in other_agg["overall"].items():
                self._aggregate_metrics["overall"][metric] += value
            
            # Merge field-level aggregate metrics
            for field_path, field_metrics in other_agg["fields"].items():
                for metric, value in field_metrics.items():
                    self._aggregate_metrics["fields"][field_path][metric] += value
        
        # Merge errors and counts
        self._errors.extend(other_state["errors"])
        self._processed_count += other_state["processed_count"]
        
        if self.verbose:
            print(
                f"Merged state: now {self._processed_count} total documents processed"
            )

    def save_non_matches(self, file_path): 
        """Save non-matches to CSV file with enhanced field-level granularity.
        
        Args:
            file_path: Path to save the CSV file
        """
        with open_file(self.individual_results_jsonl, "r", encoding="utf-8") as infile, open_file(file_path, "w", newline='', encoding="utf-8") as outfile:            
            writer = csv.writer(outfile)
            # Enhanced header with level column to distinguish object vs field level
            writer.writerow(["doc_id", "non_match_field", "non_match_type", "ground_truth", "prediction"]) 

            for line in infile:
                data = json.loads(line)
                document_id = data.get("doc_id")
                non_match_fields = data.get("comparison_result", {}).get('non_matches', [])

                for field in non_match_fields:
                    field_path = field.get('field_path', '')
                    
                    writer.writerow([
                        document_id, 
                        field_path, 
                        field.get('non_match_type', ''), 
                        field.get('ground_truth_value', ''), 
                        field.get('prediction_value', '')
                    ])

    # Legacy compatibility methods

    def evaluate_dataframe(self, df) -> Tuple[ProcessEvaluation, int]:
        """
        Legacy compatibility method for DataFrame-based evaluation.

        This method provides backward compatibility with the original DataFrame-based
        API while leveraging the new stateful processing internally.

        Args:
            df: DataFrame with columns for ground truth and predictions

        Returns:
            ProcessEvaluation with aggregated results
        """
        # Reset state for clean evaluation
        self.reset()
        count_errors = 0
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc='Evaluation'):
            doc_id = row.get("doc_id", f"row_{idx}")

            try:
                gt_data = row['expected']
                if isinstance(gt_data, dict):
                    if 'error' in gt_data:
                        raise ValueError(f"Ground truth contains error: {gt_data['error']}")
                else:
                    gt_data = json.loads(row['expected'])

                pred_data = row['predicted']
                if isinstance(pred_data, dict):
                    if 'error' in pred_data:
                        raise ValueError(f"Prediction contains error: {pred_data['error']}")
                else:
                    pred_data = json.loads(row['predicted'])
                
                # Create StructuredModel instances
                gt_model = self.target_schema(**gt_data)
                pred_model = self.target_schema(**pred_data)

                # Process using stateful update
                self.update(gt_model, pred_model, doc_id)

            except Exception as e:
                count_errors += 1
                if self.verbose:
                    print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"There were a total of {count_errors} errors.")

        return self.compute(), count_errors