# Evaluation utilities

`pretty_print.py` renders evaluation summaries, confusion matrices and non-matches.
File output disables terminal colors and uses the same rendering path as stdout.
Output-file open errors are reported to the caller's stdout; rendering errors
propagate after stdout is restored. See [printer usage](PRETTY_PRINT_README.md).

The remaining utilities calculate ANLS and key scores, compare JSON data and
validate JSON schemas.
