# Comparator unit tests

These tests cover comparator scores, normalization, thresholds, and input handling.
Run them with `uv run pytest tests/common/comparators/`.

`test_levenshtein_oracle.py` compares RapidFuzz-backed distances with the original
Python dynamic program over a seeded Unicode and boundary-length corpus. It also
pins exact public similarity scores and the documented ANLS invoice example.
