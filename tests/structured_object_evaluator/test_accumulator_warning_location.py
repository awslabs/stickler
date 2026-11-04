"""Pin where the "no field_comparisons" warning is attributed.

Both accumulators warn when the caller passed ``prediction_raw`` but forgot
``document_field_comparisons=True``. That warning is only useful if it points at
the evaluator's ``accumulate`` call — otherwise the reader gets sent into
whichever test happened to trigger it.

The refactor in #517 moves the warn call one frame deeper (into a shared base),
so its ``stacklevel`` has to go from 2 to 3. These tests pass before and after
the refactor; they are the fence that catches a wrong stacklevel.
"""

import linecache
import warnings

import pytest

from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.bbox import BBoxMAPAccumulator
from stickler.structured_object_evaluator.models.confidence.accumulator import (
    ConfidenceAccumulator,
)

ACCUMULATORS = [ConfidenceAccumulator, BBoxMAPAccumulator]


@pytest.mark.parametrize("factory", ACCUMULATORS, ids=lambda f: f.__name__)
def test_warning_points_at_the_accumulate_call(factory):
    evaluator = BulkStructuredModelEvaluator(accumulators=[factory()])
    comparison_result = {
        "confusion_matrix": {},
        "field_comparisons": [],
        "prediction_raw": {"vendor_name": "Acme Corp"},
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        evaluator.update_from_comparison_result(comparison_result)

    assert len(caught) == 1, [str(w.message) for w in caught]
    warning = caught[0]

    assert warning.filename.endswith("bulk_structured_model_evaluator.py"), (
        f"warning attributed to {warning.filename}, expected the evaluator"
    )
    source = linecache.getline(warning.filename, warning.lineno)
    assert ".accumulate(" in source, (
        f"warning points at {warning.filename}:{warning.lineno} -> {source!r}"
    )
