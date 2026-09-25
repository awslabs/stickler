"""
Regression test for print_confusion_matrix on an empty per-field matrix.

The field-name column width was computed with a max() that had no default,
so a results dict with no fields aborted the print helper with a ValueError
after the summary header had already been written.

See: https://github.com/awslabs/stickler/issues/307
"""

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.utils.pretty_print import (
    print_confusion_matrix,
)


class EmptyModel(StructuredModel):
    pass


def test_print_confusion_matrix_with_no_fields(capsys):
    """A model with no comparable fields should still print a table."""
    results = EmptyModel().compare_with(EmptyModel(), include_confusion_matrix=True)

    print_confusion_matrix(results, use_color=False)

    assert "FIELD-LEVEL METRICS" in capsys.readouterr().out
