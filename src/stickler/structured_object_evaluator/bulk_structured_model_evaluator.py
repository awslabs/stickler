#!/usr/bin/env python3

"""
Stateful Bulk Evaluator for StructuredModel objects.

This module provides a modern stateful bulk evaluator inspired by PyTorch Lightning's
stateful metrics and scikit-learn's incremental learning patterns. It supports
memory-efficient processing of large datasets through accumulation-based evaluation.
"""

import gc
import json
import logging
import math
import time
from collections import Counter, defaultdict
from typing import IO, Any, Dict, List, Optional, Tuple, Type, Union

from stickler.structured_object_evaluator.models.aggregate import (
    AggregateConfusionMatrixAccumulator,
)
from stickler.structured_object_evaluator.models.confidence import (
    ConfidenceMetric,
)
from stickler.structured_object_evaluator.models.confidence.accumulator import (
    ConfidenceAccumulator,
)
from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.utils.process_evaluation import ProcessEvaluation

logger = logging.getLogger(__name__)


def _join_path(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _migrate_legacy_acc_states(state: Dict[str, Any]) -> Dict[str, Any]:
    """Lift pre-accumulator confidence keys into the new ``accumulators`` shape.

    Older states stored confidence pairs at the top level. New states nest
    them under ``state["accumulators"]["confidence_metrics"]``. Returns the
    new-shape acc_states dict — empty if neither form is present.

    Note: a state with a non-empty ``accumulators`` dict AND legacy
    top-level keys takes the new shape and silently drops the legacy
    keys. The supported save→load cycle never produces such mixed
    states; they only arise from manual editing.
    """
    acc_states = state.get("accumulators", {})
    if not acc_states and "keyed_confidence_pairs" in state:
        return {
            "confidence_metrics": {
                "keyed_confidence_pairs": state["keyed_confidence_pairs"],
                "confidence_fields_with": state.get("confidence_fields_with", 0),
                "confidence_fields_total": state.get("confidence_fields_total", 0),
            }
        }
    return acc_states


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
        target_schema: Optional[Type[StructuredModel]] = None,
        verbose: bool = False,
        document_non_matches: bool = True,
        elide_errors: bool = False,
        individual_results_jsonl: Optional[str] = None,
        confidence_metrics: Optional[List[ConfidenceMetric]] = None,
        accumulators: Optional[List[PostComparisonAccumulator]] = None,
    ):
        """
        Initialize the stateful bulk evaluator.

        Args:
            target_schema: Optional StructuredModel class for validation and processing.
                Required for update() and evaluate_dataframe(). Not required when using
                update_from_comparison_result() with pre-computed results.
            verbose: Whether to print detailed progress information
            document_non_matches: Whether to document detailed non-match information
            elide_errors: If True, skip documents with errors; if False, accumulate error metrics
            individual_results_jsonl: Optional path to JSONL file for appending individual comparison results
            confidence_metrics: Optional list of ConfidenceMetric instances.
                Defaults to [AUROCMetric()]. Mutually exclusive with
                ``accumulators`` — pass metrics through ``ConfidenceAccumulator``
                instead, e.g. ``ConfidenceAccumulator(metrics=[AUROCMetric()])``.
            accumulators: Optional list of PostComparisonAccumulator instances.
                Defaults to [ConfidenceAccumulator(), AggregateConfusionMatrixAccumulator()].

        Raises:
            ValueError: If both ``accumulators`` and ``confidence_metrics`` are set,
                or two accumulators share the same ``.name``.
        """
        if accumulators is not None and confidence_metrics is not None:
            raise ValueError(
                "Pass either `accumulators` or `confidence_metrics`, not both."
            )

        self.target_schema = target_schema
        self.verbose = verbose
        self.document_non_matches = document_non_matches
        self.elide_errors = elide_errors
        self.individual_results_jsonl = individual_results_jsonl

        # Lazy-initialized persistent JSONL handle so the per-doc write
        # path is one fwrite() rather than open()/write()/close() ×N.
        self._jsonl_handle: Optional[IO[str]] = None

        # Build accumulators list
        if accumulators is not None:
            self._accumulators = accumulators
        else:
            self._accumulators = [
                ConfidenceAccumulator(metrics=confidence_metrics),
                AggregateConfusionMatrixAccumulator(),
            ]

        # Names key accumulator_metrics; duplicates would silently overwrite.
        name_counts = Counter(acc.name for acc in self._accumulators)
        duplicates = sorted(name for name, count in name_counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate accumulator names: {duplicates}")

        # Initialize state
        self.reset()

        self._schema_name = target_schema.__name__ if target_schema else "unknown"

        if self.verbose:
            print(f"Initialized BulkStructuredModelEvaluator for {self._schema_name}")
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
            "fields": defaultdict(lambda: defaultdict(int)),
        }

        self._overall_score_sum: float = 0.0
        self._overall_score_count: int = 0
        self._field_score_sums: Dict[str, float] = defaultdict(float)
        self._field_score_counts: Dict[str, int] = defaultdict(int)

        # Non-match tracking (when document_non_matches=True)
        self._non_matches = []

        # Error tracking
        self._errors = []

        # Per-accumulator failure counts surfaced on
        # ProcessEvaluation.accumulator_errors so silently-failing
        # accumulators show up in compute() output, not just _errors.
        self._accumulator_errors: Dict[str, int] = defaultdict(int)

        # Processing statistics
        self._processed_count = 0
        self._start_time = time.time()

        # Reset all post-comparison accumulators
        for acc in self._accumulators:
            acc.reset()

        # Drop any open JSONL handle so a subsequent write reopens the
        # path (covers the "reset between independent runs" case).
        self.close()

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

        Runs compare_with() on the model pair, optionally writes the raw result
        to JSONL, then delegates accumulation to update_from_comparison_result().

        Args:
            gt_model: Ground truth StructuredModel instance
            pred_model: Predicted StructuredModel instance
            doc_id: Optional document identifier for error tracking
        """
        if doc_id is None:
            doc_id = f"doc_{self._processed_count}"

        try:
            comparison_result = gt_model.compare_with(
                pred_model,
                include_confusion_matrix=True,
                document_non_matches=self.document_non_matches,
                document_field_comparisons=True,
            )

            # Delegate to update_from_comparison_result which handles both
            # confusion matrix accumulation and confidence extraction
            # (via prediction_raw in the comparison result).
            self.update_from_comparison_result(comparison_result, doc_id)

            # JSONL append of raw comparison result after accumulation
            # succeeds, so the file reflects "successfully accumulated"
            # rather than "attempted".
            if self.individual_results_jsonl:
                if self._jsonl_handle is None:
                    self._jsonl_handle = open(
                        self.individual_results_jsonl, "a", encoding="utf-8"
                    )
                record = {"doc_id": doc_id, "comparison_result": comparison_result}
                # default=str for parity with save_metrics() — keeps
                # numpy scalars and similar non-JSON-native values
                # from crashing the writer.
                self._jsonl_handle.write(json.dumps(record, default=str) + "\n")
                # Flush per line preserves crash-resilience: a process
                # killed mid-run still leaves a complete-line JSONL.
                self._jsonl_handle.flush()

        except Exception as e:
            error_record = {
                "doc_id": doc_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }

            if not self.elide_errors:
                self._errors.append(error_record)
                self._confusion_matrix["overall"]["fn"] += 1

            if self.verbose:
                print(f"Error processing document {doc_id}: {str(e)}")

    def update_from_comparison_result(
        self,
        comparison_result: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> None:
        """Accumulate a pre-computed compare_with() result into internal state.

        Accepts the dict output of ``compare_with(include_confusion_matrix=True)``
        and accumulates its confusion matrix. When ``prediction_raw`` and
        ``field_comparisons`` are present, confidence pairs are extracted too —
        producing identical confidence metrics to the ``update()`` path.

        Args:
            comparison_result: Dict from ``compare_with(include_confusion_matrix=True)``.
                Must contain ``confusion_matrix``; ``prediction_raw`` and
                ``field_comparisons`` are optional (used for confidence).
            doc_id: Optional document identifier for error tracking.
        """
        if doc_id is None:
            doc_id = f"doc_{self._processed_count}"

        # Re-raise (don't fold into the per-doc fail path) so a malformed
        # input surfaces directly instead of silently bumping fn.
        if "confusion_matrix" not in comparison_result:
            raise ValueError("comparison_result missing 'confusion_matrix' key")

        try:
            # Collect non-matches if enabled and present
            if self.document_non_matches and "non_matches" in comparison_result:
                for non_match in comparison_result["non_matches"]:
                    non_match_with_doc = non_match.copy()
                    non_match_with_doc["doc_id"] = doc_id
                    self._non_matches.append(non_match_with_doc)

            cm_result = comparison_result["confusion_matrix"]
            self._accumulate_confusion_matrix(cm_result)

            if "overall_score" in comparison_result:
                self._accumulate_overall_score(comparison_result["overall_score"])

            # Isolate per-accumulator failures so one bad accumulator can't tank the cm.
            prediction_raw = comparison_result.get("prediction_raw")
            for acc in self._accumulators:
                try:
                    acc.accumulate(comparison_result, prediction_raw)
                except Exception as acc_err:
                    self._accumulator_errors[acc.name] += 1
                    acc_error_record = {
                        "doc_id": doc_id,
                        "error": str(acc_err),
                        "error_type": type(acc_err).__name__,
                        "accumulator": acc.name,
                    }
                    if not self.elide_errors:
                        self._errors.append(acc_error_record)
                    if self.verbose:
                        print(
                            f"Accumulator {acc.name!r} failed on {doc_id}: "
                            f"{acc_err}"
                        )

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

        # Flush and release the JSONL handle on the natural end-of-run
        # path so callers don't have to remember to close() explicitly.
        # Idempotent: safe to call again.
        self.close()

        if self.verbose:
            total_time = time.time() - self._start_time
            print(
                f"Final computation completed: {self._processed_count} documents in {total_time:.2f}s"
            )
            print(f"Overall accuracy: {result.metrics.get('cm_accuracy', 0.0):.3f}")

        return result

    def close(self) -> None:
        """Close the persistent JSONL handle if open. Idempotent."""
        handle = getattr(self, "_jsonl_handle", None)
        if handle is not None:
            self._jsonl_handle = None
            try:
                handle.close()
            except Exception as exc:
                # Closing should never crash the surrounding flow
                # (compute, reset, GC). Log and continue.
                logger.debug("Failed to close JSONL handle: %s", exc)

    def __del__(self) -> None:
        # GC fallback for the case where compute()/reset() were never
        # called before the evaluator went out of scope.
        try:
            self.close()
        except Exception as exc:
            logger.debug("close() raised during __del__: %s", exc)

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
        """Recursively accumulate field-level CM counts and threshold_applied_score.

        Walks both ``fields`` (object subtrees) and ``nested_fields``
        (list-of-StructuredModel) so per-field ``mean_score`` is recorded
        at every node compare_with emits, including leaves under list
        parents.
        """
        for field_name, field_data in fields_dict.items():
            if not isinstance(field_data, dict):
                continue
            current_path = _join_path(path_prefix, field_name)

            direct_metrics = {
                k: v
                for k, v in field_data.items()
                if k in ["tp", "fp", "tn", "fn", "fd", "fa"]
                and isinstance(v, (int, float))
            }
            if direct_metrics:
                self._accumulate_single_field_metrics(current_path, direct_metrics)

            if isinstance(field_data.get("overall"), dict):
                self._accumulate_single_field_metrics(
                    current_path, field_data["overall"]
                )

            if "threshold_applied_score" in field_data:
                score = field_data["threshold_applied_score"]
                if self._is_valid_score(score):
                    self._field_score_sums[current_path] += float(score)
                    self._field_score_counts[current_path] += 1
                else:
                    logger.debug(
                        "Skipping non-finite threshold_applied_score=%r at %s",
                        score,
                        current_path,
                    )

            if isinstance(field_data.get("fields"), dict):
                self._accumulate_field_metrics(field_data["fields"], current_path)
            if isinstance(field_data.get("nested_fields"), dict):
                self._accumulate_field_metrics(
                    field_data["nested_fields"], current_path
                )

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

    @staticmethod
    def _is_valid_score(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    def _accumulate_overall_score(self, overall_score: Any) -> None:
        if self._is_valid_score(overall_score):
            self._overall_score_sum += float(overall_score)
            self._overall_score_count += 1
        else:
            logger.debug(
                "Skipping non-finite overall_score=%r from weighted aggregate",
                overall_score,
            )

    def _calculate_derived_metrics(
        self, cm_dict: Dict[str, Union[int, float]]
    ) -> Dict[str, float]:
        """
        Calculate derived confusion matrix metrics (precision, recall, f1, accuracy).

        This method replicates the derivation logic for confusion matrix metrics.

        Args:
            cm_dict: Dictionary with basic confusion matrix counts

        Returns:
            Dictionary with derived metrics
        """
        tp = cm_dict.get("tp", 0)
        fp = cm_dict.get("fp", 0)
        tn = cm_dict.get("tn", 0)
        fn = cm_dict.get("fn", 0)

        # Calculate derived metrics with safe division
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            "cm_precision": precision,
            "cm_recall": recall,
            "cm_f1": f1,
            "cm_accuracy": accuracy,
        }

    def _build_process_evaluation(self) -> ProcessEvaluation:
        """
        Build ProcessEvaluation from current accumulated state.

        Returns:
            ProcessEvaluation with computed metrics from accumulated state
        """
        # Calculate derived metrics for overall results
        overall_cm = dict(self._confusion_matrix["overall"])
        overall_derived = self._calculate_derived_metrics(overall_cm)
        overall_metrics = {**overall_cm, **overall_derived}

        overall_metrics["weighted_overall_score"] = (
            self._overall_score_sum / self._overall_score_count
            if self._overall_score_count > 0
            else 0.0
        )

        field_metrics = {}
        field_paths = set(self._confusion_matrix["fields"].keys())
        field_paths.update(self._field_score_sums.keys())
        for field_path in field_paths:
            field_cm_dict = dict(self._confusion_matrix["fields"].get(field_path, {}))
            field_derived = self._calculate_derived_metrics(field_cm_dict)
            field_metrics[field_path] = {**field_cm_dict, **field_derived}

            # Omit mean_score (vs. 0.0) to preserve "no data" vs. "observed zero".
            count = self._field_score_counts.get(field_path, 0)
            if count > 0:
                field_metrics[field_path]["mean_score"] = (
                    self._field_score_sums.get(field_path, 0.0) / count
                )

        total_time = time.time() - self._start_time

        # Compute metrics from all post-comparison accumulators
        accumulator_metrics: Dict[str, Any] = {}
        for acc in self._accumulators:
            computed = acc.compute()
            if computed is not None:
                accumulator_metrics[acc.name] = computed

        # Extract confidence_metrics for backward compatibility
        confidence_metrics = accumulator_metrics.get("confidence_metrics")

        # Surface per-accumulator failure counts only when at least one
        # accumulator actually raised. Empty dict → None preserves the
        # "additive optional field" contract used elsewhere on
        # ProcessEvaluation.
        accumulator_errors = (
            dict(self._accumulator_errors) if self._accumulator_errors else None
        )

        return ProcessEvaluation(
            document_count=self._processed_count,
            metrics=overall_metrics,
            field_metrics=field_metrics,
            errors=list(self._errors),
            total_time=total_time,
            non_matches=list(self._non_matches) if self.document_non_matches else None,
            confidence_metrics=confidence_metrics,
            accumulator_metrics=accumulator_metrics or None,
            accumulator_errors=accumulator_errors,
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
            # Surface accumulator outputs (confidence_metrics today, future
            # bbox mAP etc.) plus non-match details alongside the confusion
            # matrix so `save_metrics()` is a complete snapshot of what
            # `compute()` would return, not a confusion-matrix-only dump.
            "confidence_metrics": process_eval.confidence_metrics,
            "accumulator_metrics": process_eval.accumulator_metrics,
            "non_matches": process_eval.non_matches,
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
                "target_schema": self._schema_name,
            },
            "errors": process_eval.errors,
            "metadata": {
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "evaluator_config": {
                    "verbose": self.verbose,
                    "document_non_matches": self.document_non_matches,
                    "elide_errors": self.elide_errors,
                    "individual_results_jsonl": self.individual_results_jsonl,
                },
            },
        }

        # Ensure directory exists
        import os

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
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
        print(f"BULK EVALUATION RESULTS - {self._schema_name}")
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
        print("\nCONFUSION MATRIX:")
        print(f"  True Positives (TP):    {overall_metrics.get('tp', 0):,}")
        print(f"  False Positives (FP):   {overall_metrics.get('fp', 0):,}")
        print(f"  True Negatives (TN):    {overall_metrics.get('tn', 0):,}")
        print(f"  False Negatives (FN):   {overall_metrics.get('fn', 0):,}")
        print(f"  False Discovery (FD):   {overall_metrics.get('fd', 0):,}")
        print(f"  False Alarm (FA):   {overall_metrics.get('fa', 0):,}")

        # Derived metrics
        print("\nDERIVED METRICS:")
        print(f"  Precision:     {overall_metrics.get('cm_precision', 0.0):.4f}")
        print(f"  Recall:        {overall_metrics.get('cm_recall', 0.0):.4f}")
        print(f"  F1 Score:      {overall_metrics.get('cm_f1', 0.0):.4f}")
        print(f"  Accuracy:      {overall_metrics.get('cm_accuracy', 0.0):.4f}")
        print(
            f"  Weighted Overall Score: "
            f"{overall_metrics.get('weighted_overall_score', 0.0):.4f}"
        )

        # Field-level metrics
        if process_eval.field_metrics:
            print("\nFIELD-LEVEL METRICS:")
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
                mean_score = field_metrics.get("mean_score")
                mean_cell = f"{mean_score:.3f}" if mean_score is not None else "  n/a"

                # Only show fields with some activity
                if tp + fp + fn > 0:
                    display_path = (
                        field_path if len(field_path) <= 30 else field_path[:27] + "..."
                    )
                    print(
                        f"  {display_path:30} Mean: {mean_cell} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | TP: {tp:,} | FP: {fp:,} | FN: {fn:,}"
                    )

        # Error summary
        if process_eval.errors:
            print("\nERROR SUMMARY:")
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

        # Corpus-level aggregate slice (universal aggregate accumulator).
        # Surfaces the field-level rollup that's separately accumulated from
        # the threshold-gated overall metrics — see docs/Advanced/aggregate-metrics.md.
        aggregate_metrics = (process_eval.accumulator_metrics or {}).get(
            "aggregate_metrics"
        )
        if aggregate_metrics:
            agg_overall = aggregate_metrics.get("overall") or {}
            agg_derived = agg_overall.get("derived") or {}
            print("\nAGGREGATE METRICS (corpus rollup, ungated by match_threshold):")
            print("-" * 40)
            print(f"  TP: {agg_overall.get('tp', 0):,}  "
                  f"FD: {agg_overall.get('fd', 0):,}  "
                  f"FA: {agg_overall.get('fa', 0):,}  "
                  f"FN: {agg_overall.get('fn', 0):,}  "
                  f"FP: {agg_overall.get('fp', 0):,}  "
                  f"TN: {agg_overall.get('tn', 0):,}")
            print(f"  Precision: {agg_derived.get('cm_precision', 0.0):.4f}  "
                  f"Recall: {agg_derived.get('cm_recall', 0.0):.4f}  "
                  f"F1: {agg_derived.get('cm_f1', 0.0):.4f}  "
                  f"Accuracy: {agg_derived.get('cm_accuracy', 0.0):.4f}")
            agg_fields = aggregate_metrics.get("fields") or {}
            if agg_fields:
                # Sort by F1 desc for readability, mirror the field-metrics block.
                sorted_agg = sorted(
                    agg_fields.items(),
                    key=lambda x: (x[1].get("derived") or {}).get("cm_f1", 0.0),
                    reverse=True,
                )
                for path, counts in sorted_agg:
                    derived = counts.get("derived") or {}
                    tp = counts.get("tp", 0)
                    fd = counts.get("fd", 0)
                    fa = counts.get("fa", 0)
                    fn = counts.get("fn", 0)
                    if tp + fd + fa + fn == 0:
                        continue
                    display_path = path if len(path) <= 30 else path[:27] + "..."
                    print(
                        f"  {display_path:30} "
                        f"P: {derived.get('cm_precision', 0.0):.3f} | "
                        f"R: {derived.get('cm_recall', 0.0):.3f} | "
                        f"F1: {derived.get('cm_f1', 0.0):.3f} | "
                        f"TP: {tp:,} | FD: {fd:,} | FA: {fa:,} | FN: {fn:,}"
                    )

        # Per-accumulator failure visibility — surfaced separately so a
        # silently-failing accumulator (whose errors don't affect the
        # confusion matrix) still shows up clearly.
        if process_eval.accumulator_errors:
            print("\nACCUMULATOR ERRORS:")
            print("-" * 40)
            for acc_name, count in sorted(
                process_eval.accumulator_errors.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {acc_name}: {count:,}")

        # Configuration info
        print("\nCONFIGURATION:")
        print("-" * 40)
        print(f"Target Schema: {self._schema_name}")
        print(f"Document Non-matches: {'Yes' if self.document_non_matches else 'No'}")
        print(f"Elide Errors: {'Yes' if self.elide_errors else 'No'}")
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
        return {
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
            "accumulator_errors": dict(self._accumulator_errors),
            # Post-comparison accumulator states
            "accumulators": {
                acc.name: acc.get_state() for acc in self._accumulators
            },
            "overall_score_sum": self._overall_score_sum,
            "overall_score_count": self._overall_score_count,
            "field_score_sums": dict(self._field_score_sums),
            "field_score_counts": dict(self._field_score_counts),
            # Configuration
            "target_schema": self._schema_name,
            "elide_errors": self.elide_errors,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore evaluator state from serialized data.

        This method restores the internal state from data previously saved
        with get_state(), enabling recovery from checkpoints.

        Args:
            state: State dictionary from get_state()
        """
        # Validate state compatibility
        if state.get("target_schema") != self._schema_name:
            raise ValueError(
                f"State schema {state.get('target_schema')} doesn't match evaluator schema {self._schema_name}"
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

        # Restore other state
        self._errors = list(state["errors"])
        self._processed_count = state["processed_count"]
        self._start_time = state["start_time"]
        # .get() keeps older state dicts (no key) loadable.
        self._accumulator_errors = defaultdict(
            int, state.get("accumulator_errors", {})
        )

        acc_states = _migrate_legacy_acc_states(state)
        for acc in self._accumulators:
            if acc.name in acc_states:
                acc.load_state(acc_states[acc.name])

        # .get() keeps older state dicts (no score keys) loadable.
        self._overall_score_sum = float(state.get("overall_score_sum", 0.0))
        self._overall_score_count = int(state.get("overall_score_count", 0))
        self._field_score_sums = defaultdict(float, state.get("field_score_sums", {}))
        self._field_score_counts = defaultdict(int, state.get("field_score_counts", {}))

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
        if other_state.get("target_schema") != self._schema_name:
            raise ValueError(
                f"Cannot merge incompatible schemas: {other_state.get('target_schema')} vs {self._schema_name}"
            )

        # Merge overall metrics
        other_cm = other_state["confusion_matrix"]
        for metric, value in other_cm["overall"].items():
            self._confusion_matrix["overall"][metric] += value

        # Merge field-level metrics
        for field_path, field_metrics in other_cm["fields"].items():
            for metric, value in field_metrics.items():
                self._confusion_matrix["fields"][field_path][metric] += value

        # Merge errors and counts
        self._errors.extend(other_state["errors"])
        self._processed_count += other_state["processed_count"]
        for name, count in other_state.get("accumulator_errors", {}).items():
            self._accumulator_errors[name] += int(count)

        acc_states = _migrate_legacy_acc_states(other_state)
        for acc in self._accumulators:
            if acc.name in acc_states:
                acc.merge_state(acc_states[acc.name])

        # .get() keeps older peer states (no score keys) mergeable.
        self._overall_score_sum += float(other_state.get("overall_score_sum", 0.0))
        self._overall_score_count += int(other_state.get("overall_score_count", 0))
        for path, s in other_state.get("field_score_sums", {}).items():
            self._field_score_sums[path] += float(s)
        for path, c in other_state.get("field_score_counts", {}).items():
            self._field_score_counts[path] += int(c)

        if self.verbose:
            print(
                f"Merged state: now {self._processed_count} total documents processed"
            )

    # Legacy compatibility methods

    def evaluate_dataframe(self, df) -> ProcessEvaluation:
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

        # Process each row
        for idx, row in df.iterrows():
            doc_id = row.get("doc_id", f"row_{idx}")

            try:
                # Parse JSON data
                gt_data = json.loads(row["expected"])
                pred_data = json.loads(row["predicted"])

                # Create StructuredModel instances
                gt_model = self.target_schema(**gt_data)
                pred_model = self.target_schema(**pred_data)

                # Process using stateful update
                self.update(gt_model, pred_model, doc_id)

            except Exception as e:
                if self.verbose:
                    print(f"Error processing row {idx}: {e}")
                continue

        return self.compute()


def aggregate_from_comparisons(
    comparison_results: List[Dict[str, Any]],
) -> ProcessEvaluation:
    """
    Aggregate a list of pre-computed compare_with() results into field-level metrics.

    This is a convenience function for aggregating stored comparison results
    without needing the original StructuredModel instances. It accepts the raw
    dictionary outputs of StructuredModel.compare_with(include_confusion_matrix=True).

    When comparison results include "prediction_raw" and "field_comparisons",
    confidence metrics are also aggregated automatically.

    Args:
        comparison_results: List of dictionaries, each returned by
            StructuredModel.compare_with(include_confusion_matrix=True).

    Returns:
        ProcessEvaluation with aggregated metrics including overall and
        per-field precision, recall, F1, accuracy, and confidence metrics
        (when prediction_raw is present in the comparison results).
    """
    evaluator = BulkStructuredModelEvaluator()
    for result in comparison_results:
        evaluator.update_from_comparison_result(result)
    return evaluator.compute()
