# Comparator tests

Run `pytest tests/common/comparators/` from the repository root. Tests cover comparator scores, input handling, and model integration.

`test_levenshtein_oracle.py` compares RapidFuzz-backed distances with the original
Python dynamic program over a seeded Unicode and boundary-length corpus. It also
pins exact public similarity scores and the documented ANLS invoice example.
