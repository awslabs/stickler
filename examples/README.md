# Examples Directory

This directory contains comprehensive examples demonstrating the core functionality of the stickler library for structured object evaluation and comparison.

## 🚀 Quick Start

**For beginners, start here:**
- [`scripts/quick_start.py`](scripts/quick_start.py) - Essential functionality demonstration
- [`notebooks/Quick_start.ipynb`](notebooks/Quick_start.ipynb) - Interactive Jupyter notebook

## 📚 Example Files

### 1. Basic Usage
- **`scripts/quick_start.py`** - Core functionality in 5 minutes
  - Individual object comparison
  - List comparison with Hungarian algorithm
  - Basic evaluation metrics

### 2. Advanced Features  
- **`scripts/non_match_analysis_demo.py`** - Debugging and error analysis
  - Detailed non-match reporting
  - Error classification (FD, FA, FN)
  - Actionable debugging insights
  
- **`scripts/bulk_evaluation_demo.py`** - Large-scale evaluation
  - Memory-efficient processing
  - Batch processing capabilities
  - Performance comparison

- **`scripts/print_results_demo.py`** - Beautiful results formatting
  - Colored terminal output with visual bars
  - Works with all evaluation result types
  - Field filtering and sorting options

- **`scripts/model_from_json_demo.py`** - Dynamic model creation
  - Configuration-driven model creation
  - A/B testing different field configurations
  - JSON/YAML configuration loading
  - Error handling and validation examples

- **`scripts/json_to_evaluation_demo.py`** - Complete JSON workflow (NEW!)
  - Zero Python object construction required
  - JSON configuration + JSON data → evaluation
  - Complex nested structures and list matching
  - Production-ready JSON-driven evaluation

- **`scripts/bert_comparator_demo.py`** - BERTComparator for document extraction eval
  - Semantic similarity via BERTScore (token-level embedding matching)
  - Threshold tuning for paraphrased LLM outputs
  - StructuredModel integration with mixed comparators
  - Requires `--extra bert` (torch, bert-score, evaluate)

### 3. Interactive Notebooks
- **`notebooks/Quick_start.ipynb`** - Interactive introduction
  - Step-by-step guided examples  
  - Individual and list comparison
  - Metrics interpretation

- **`notebooks/Complex_nested_structure.ipynb`** - Advanced structures
  - Deeply nested object evaluation
  - Optional field handling
  - Complex error analysis

- **`notebooks/Confidence_Estimation.ipynb`** - Single-document confidence evaluation
  - Rich value pattern in JSON (confidence, bbox, or value-only)
  - Per-field confidence metrics and coverage
  - Nested object path handling

- **`notebooks/Bulk_Confidence_AUROC.ipynb`** - Dataset-level confidence evaluation
  - Bulk AUROC, Brier Score, ECE with bin data
  - Per-field breakdowns and side-by-side model comparison
  - State merge for distributed evaluation

- **`notebooks/Map_Reduce_Evaluation.ipynb`** - Production map/reduce pattern
  - Compare individual docs (map), save to JSONL
  - Aggregate from JSONL (reduce) with full confidence metrics
  - Verify direct bulk vs JSONL replay produce identical results

## 🎯 What Each Example Demonstrates

| Example | Individual Objects | List Comparison | Nested Structures | Error Analysis | Large Scale | Pretty Print |
|---------|:------------------:|:---------------:|:-----------------:|:--------------:|:-----------:|:------------:|
| `quick_start.py` | ✅ | ✅ | ➖ | ➖ | ➖ | ➖ |
| `non_match_analysis_demo.py` | ✅ | ✅ | ✅ | ✅ | ➖ | ➖ |
| `bulk_evaluation_demo.py` | ✅ | ➖ | ➖ | ➖ | ✅ | ➖ |
| `print_results_demo.py` | ✅ | ➖ | ➖ | ➖ | ➖ | ✅ |
| `Quick_start.ipynb` | ✅ | ✅ | ➖ | ➖ | ➖ | ✅ |
| `Complex_nested_structure.ipynb` | ✅ | ✅ | ✅ | ✅ | ➖ | ✅ |

## 🏃‍♂️ Running the Examples

### Python Scripts
```bash
# Basic functionality
python examples/scripts/quick_start.py

# Error analysis and debugging
python examples/scripts/non_match_analysis_demo.py

# Large-scale evaluation  
python examples/scripts/bulk_evaluation_demo.py
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open notebooks in examples/notebooks/
# - Quick_start.ipynb
# - Complex_nested_structure.ipynb
```

## 🤖 Running examples in CI

Every notebook under `examples/notebooks/` and every script under `examples/scripts/` is executed end-to-end by the `examples` GitHub Actions workflow on pull requests and pushes targeting `dev` and `main`, plus on a daily schedule. The same `pytest`-based entry point reproduces this behavior on a contributor machine.

### Reproducing CI locally

```bash
uv sync --group test --extra llm --frozen
uv run pytest examples/ --nbmake
```

This is the same command CI runs. Local and CI invocations differ only in environment, not in test selection or runner logic — the only deltas are CI-side output formatting (JUnit XML, the GitHub Actions job summary, and the uploaded artifact).

Scripts are executed as subprocesses (`python <script>.py`), each with its **own directory** as the working directory, so any incidental file output (HTML reports, JSON dumps, model caches) lands next to the script rather than dirtying the repo root when you run `uv run pytest examples/` locally. Notebooks run via `nbmake` in the standard kernel cwd.

When you run the command without AWS credentials configured, the local invocation skips exactly the same set of credentialed examples that CI skips on the GitHub-hosted runner. Skipped paths are listed in pytest's standard summary section, e.g.:

```
SKIPPED [1] examples/scripts/llm_comparator_demo.py: credentialed example: requires AWS credentials
SKIPPED [1] examples/scripts/bert_comparator_demo.py: long-running example: scheduled runs only
```

### Run modes

The workflow sets the `EXAMPLES_RUN_MODE` environment variable to control which examples execute:

| Mode | Trigger | Long-running examples | Credentialed examples |
|---|---|---|---|
| `pr` | `pull_request` or `push` to `dev`/`main` | Skipped | Skipped |
| `scheduled` | Daily cron (07:00 UTC) or `workflow_dispatch` | Included | Included if AWS credentials are detected, otherwise skipped |
| `local` (default when env var is absent) | Contributor machine | Skipped | Included if AWS credentials are detected, otherwise skipped |

Long-running examples are skipped by default locally so contributors don't accidentally trigger multi-gigabyte model downloads. To exercise everything locally, run:

```bash
EXAMPLES_RUN_MODE=scheduled uv run pytest examples/ --nbmake
```

### Credentialed examples

A **credentialed example** requires AWS credentials (e.g., AWS Bedrock access for the `LLMComparator`) or other live external services to execute successfully. CI runs on the standard GitHub-hosted runner and on forks where Bedrock access is not available, so credentialed examples are declared up front and skipped cleanly when credentials are absent.

The credential-marker mechanism is a path-based registry in [`examples/conftest.py`](conftest.py):

```python
CREDENTIALED_EXAMPLES: frozenset[str] = frozenset({
    "examples/scripts/llm_comparator_demo.py",
})
```

A test is skipped when its repo-relative path is in `CREDENTIALED_EXAMPLES` and either the run mode is `pr` or no AWS credentials are detected (none of `AWS_ACCESS_KEY_ID`, `AWS_PROFILE`, `AWS_WEB_IDENTITY_TOKEN_FILE` are set).

Current entries:

| Path | Why credentialed |
|---|---|
| `examples/scripts/llm_comparator_demo.py` | Uses `LLMComparator` with AWS Bedrock (`us.amazon.nova-lite-v1:0`) via `strands-agents`. |

To add a new credentialed example:

1. Add the example's repo-relative POSIX path to `CREDENTIALED_EXAMPLES` in `examples/conftest.py`.
2. Append a row to the table above with a one-line rationale (which AWS service the example calls, or which other external dependency it has).
3. Verify locally without credentials: `uv run pytest examples/ --nbmake --collect-only -q` should show the new path as `SKIPPED`.

### Long-running examples

A **long-running example** has a typical wall-clock runtime on `ubuntu-latest` that exceeds 5 minutes. Long-running examples are excluded from PR runs (to keep PR feedback within the 10-minute budget) and included on the daily scheduled run.

The registry lives in [`examples/conftest.py`](conftest.py):

```python
LONG_RUNNING_EXAMPLES: frozenset[str] = frozenset({
    "examples/scripts/bert_comparator_demo.py",
})
```

A test is skipped when its repo-relative path is in `LONG_RUNNING_EXAMPLES` and the run mode is anything other than `scheduled`.

Current entries:

| Path | Why long-running |
|---|---|
| `examples/scripts/bert_comparator_demo.py` | First run downloads ~1.4 GB RoBERTa model from HuggingFace; CPU-only BERTScore inference is slow on `ubuntu-latest`. |

To add a new long-running example:

1. Measure typical wall-clock on `ubuntu-latest` (e.g., from a recent workflow run). If it consistently exceeds 5 minutes, the example qualifies.
2. Add the example's repo-relative POSIX path to `LONG_RUNNING_EXAMPLES` in `examples/conftest.py`.
3. Append a row to the table above describing the cost driver (model download, large input fixture, expensive computation).
4. If the example pulls in a heavy optional dependency (e.g., `torch`, `boto3`), add the corresponding extra to the **scheduled-mode install line** in `.github/workflows/examples.yaml` (`UV_SYNC_EXTRAS`). PR runs deliberately do not install these — the long-running example is skipped before it ever imports.
5. Verify locally: `uv run pytest examples/ --nbmake --collect-only -q` should show the new path as `SKIPPED`, and `EXAMPLES_RUN_MODE=scheduled uv run pytest examples/ --nbmake --collect-only -q` should include it.

### Runtime budget and scheduling

- **PR runtime budget**: 10 minutes wall-clock on the standard GitHub-hosted Ubuntu runner. Examples that don't fit are added to `LONG_RUNNING_EXAMPLES` and run on the daily schedule instead.
- **Daily scheduled run**: 07:00 UTC. The scheduled run executes every example, including those excluded from PR runs.
- **Dependency install differs by mode.** PR runs install only `--group test --extra llm` (the minimum needed to collect every example and execute every non-skipped one). Scheduled and `workflow_dispatch` runs additionally install `--extra bert` because the BERT demo executes in those modes; if you add a new long-running example that needs more extras, extend `UV_SYNC_EXTRAS` in `.github/workflows/examples.yaml`.

### Why we skip credentialed examples instead of mocking or wiring OIDC

The current behavior is to declare credentialed examples up front and skip them in CI rather than mock the external service or wire up GitHub OIDC + an IAM role. The trade-offs:

- **Security**: granting Bedrock access to PR runs on a public repository requires either long-lived AWS access keys (a credential-leak risk on every contributor's PR) or OIDC-trusted IAM roles, and OIDC + role provisioning is out of scope for this feature.
- **Fidelity**: mocking Bedrock inside the example would defeat its purpose. The `llm_comparator_demo` exists to show what real Bedrock integration looks like; a mocked version would teach the wrong API surface and would not catch breakage in the real strands-agents → Bedrock path.
- **Scope**: the requirement is that credentialed examples must not cause CI failures, not that they must execute somewhere right now. Skipping is the simplest correct behavior. The daily scheduled run is the natural place to add a real Bedrock execution later via OIDC, without re-architecting the workflow.

### Branch protection rollout

To make the example check **required** for merging into `dev` and `main`, a maintainer must add the `examples / run-examples` check to the branch protection rules for `dev` and `main`. This is a one-time step performed in the GitHub UI (Settings → Branches → Branch protection rules → Require status checks to pass before merging) or via `gh api`, and it should be done **after** the workflow has produced at least one successful run on the target branch so GitHub recognizes the check name.

This step is not part of the merged code because branch protection lives outside the repository.

## 🎯 Key Concepts Demonstrated

### 1. **Individual Object Comparison**
Compare two structured objects field-by-field:
- Configure comparison rules per field
- Weight fields by importance
- Get detailed similarity scores

### 2. **List Comparison (Main Strength!)**
Optimally match objects in lists using Hungarian algorithm:
- Handle different ordering
- Manage missing/extra objects  
- Classify matches vs. non-matches

### 3. **Complex Nested Structures**
Evaluate hierarchical data with multiple nesting levels:
- Nested objects within objects
- Lists of objects within objects
- Optional fields and missing data

### 4. **Error Analysis & Debugging**
Identify specific issues in your data:
- False Discoveries (wrong values)
- False Alarms (extra fields)
- False Negatives (missing fields)

### 5. **Scalable Evaluation**
Process large datasets efficiently:
- Memory-efficient streaming
- Batch processing capabilities
- Progress tracking and metrics

## 🚀 Perfect Use Cases

The examples demonstrate evaluation scenarios for:

- **Document Extraction** - Invoices, forms, receipts
- **OCR Quality Assessment** - Text extraction accuracy
- **Entity Extraction** - Named entity recognition
- **ML Model Evaluation** - Structured output validation
- **Data Quality Monitoring** - Production system assessment

## 💡 Next Steps

1. **Start with `quick_start.py`** to understand the basics
2. **Explore notebooks** for interactive learning
3. **Use `non_match_analysis_demo.py`** for debugging real data
4. **Scale up with `bulk_evaluation_demo.py`** for production use
5. **Adapt examples** to your specific data structures

## 📖 Documentation

For comprehensive documentation, see:
- [Main README](../README.md)
- [Comparators Guide](../src/stickler/comparators/Comparators.md)
- [StructuredModel README](../src/stickler/structured_object_evaluator/README.md)

## 🔧 Troubleshooting

If examples don't run:
1. Install dependencies: `uv sync` (if developing) or `pip install stickler-eval` (if using from PyPI)
2. Check Python version: Requires Python 3.12+
3. Verify installation: `python -c "import stickler; print('Success!')"`

## 🎉 Key Insight

The **Hungarian algorithm for list matching** is what makes this library special - it finds optimal pairings between objects even when they're in different orders or partially missing. This is demonstrated throughout the examples and is the core strength for real-world structured data evaluation.
