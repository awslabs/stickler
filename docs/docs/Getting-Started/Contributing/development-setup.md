# Development Environment Setup

This guide walks you through setting up a complete development environment for contributing to Stickler.

## Prerequisites

### Required

- **Python 3.12+** - Stickler requires Python 3.12 or higher
- **Git** - For version control

### Recommended

- [**conda**](https://github.com/conda/conda?tab=readme-ov-file#installation) - For environment management
- **VS Code** - IDE with Python support

### Verify Prerequisites

```bash
# Check Python version
python --version  # Should be 3.12.x or higher

# Check Git
git --version
```

## Quick Setup

### Option A: Conda (Recommended)

Conda provides isolated environments with easy Python version management.

```bash
# Create conda environment with Python 3.12
conda create -n stickler python=3.12 -y

# Activate the environment
conda activate stickler

# Clone the repository (if you haven't already)
git clone https://github.com/awslabs/stickler.git
cd stickler

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Option B: venv

Use Python's built-in virtual environment:

```bash
# Clone the repository
git clone https://github.com/awslabs/stickler.git
cd stickler

# Create virtual environment
python -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Verify Installation

After installation, verify everything is working:

```bash
# Run tests
pytest tests/ -v --tb=short

# Check a quick example
python examples/scripts/quick_start.py
```

You should see all tests passing.

## Development Dependencies

The `[dev]` extras install these additional packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=7.0.0 | Testing framework |
| `pytest-xdist` | >=3.0.0 | Parallel test execution |
| `coverage` | >=7.0.0 | Code coverage reporting |
| `beautifulsoup4` | >=4.14.2 | HTML report testing |

## IDE Configuration

### VS Code

#### Recommended Extensions

Install these extensions for the best experience:

- [**Python**](https://marketplace.visualstudio.com/items?itemName=ms-python.python) (Microsoft) - Python language support
- **Pylance** - Fast, feature-rich language support (comes with the above extension)
- [**Ruff**](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) - Linting integration


## Project Structure

Understanding the project structure helps navigate the codebase:

```
stickler/
├── src/stickler/                    # Source code
│   ├── __init__.py
│   ├── structured_object_evaluator/ # Core evaluation engine
│   │   ├── models/                  # StructuredModel, ComparableField
│   │   ├── evaluators/              # Evaluation logic
│   │   └── ...
│   ├── comparators/                 # Comparison algorithms
│   │   ├── base.py                  # BaseComparator
│   │   ├── levenshtein.py
│   │   ├── numeric.py
│   │   ├── exact.py
│   │   └── ...
│   ├── algorithms/                  # Matching algorithms
│   ├── utils/                       # Shared utilities
│   └── reporting/                   # Result formatting
├── tests/                           # Test suite
│   ├── structured_object_evaluator/ # Core tests
│   ├── common/                      # Comparator/algorithm tests
│   └── reporting/                   # Report tests
├── docs/                            # Documentation (MkDocs)
├── examples/                        # Usage examples
│   ├── scripts/                     # Python scripts
│   └── notebooks/                   # Jupyter notebooks
├── pyproject.toml                   # Project configuration
├── CONTRIBUTING.md                  # Contribution guidelines
└── README.md                        # Project overview
```

## Common Development Tasks

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific module
pytest tests/structured_object_evaluator/

# Run specific test file
pytest tests/structured_object_evaluator/test_comparators.py

# Run tests matching pattern
pytest tests/ -k "levenshtein"

# Run with coverage
coverage run -m pytest tests/
coverage report -m
```

### Linting

```bash
# Install Ruff (not included in dev dependencies)
pip install ruff

# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Building Documentation

The project uses MkDocs with Material theme for documentation.

```bash
# Navigate to docs directory
cd docs

# Install documentation dependencies
make install

# Serve documentation locally with live reload
make docs
# Opens at http://127.0.0.1:8000

# Build static documentation
make build

# Clean generated files
make clean
```

**Documentation Dependencies** (automatically installed via `make install`):

- mkdocs - Static site generator
- mkdocs-material - Material Design theme  
- mkdocs-awesome-nav - Automatic navigation from directory structure
- mkdocstrings-python - Auto-generate API docs from docstrings
- pymdown-extensions - Additional markdown extensions

**Available Makefile Targets:**

| Target | Command | Purpose |
|--------|---------|---------|
| `make install` | Install dependencies | Install all documentation requirements |
| `make docs` | Serve with live reload | Start local dev server at http://127.0.0.1:8000 |
| `make build` | Build static site | Generate production site in `site/` directory |
| `make deploy` | Deploy to GitHub Pages | Build and push to gh-pages branch |
| `make clean` | Remove build artifacts | Clean the `site/` directory |

### Running Examples

```bash
# Quick start example
python examples/scripts/quick_start.py

# JSON schema example
python examples/scripts/json_schema_demo.py
```

## Troubleshooting

### Common Issues

#### Python Version Mismatch

```
ERROR: Package requires Python >=3.12
```

**Solution:** Ensure you're using Python 3.12+:
```bash
python --version
# If wrong version, create new conda environment
conda create -n stickler python=3.12 -y
conda activate stickler
```

#### Import Errors

```
ModuleNotFoundError: No module named 'stickler'
```

**Solution:** Install in development mode:
```bash
pip install -e ".[dev]"
```

#### NumPy/SciPy Build Issues

On older systems, you may encounter compilation errors:

```
ERROR: Failed building wheel for numpy
```

**Solution:** Use conda which provides pre-built binaries:
```bash
conda install numpy scipy
pip install -e ".[dev]"
```

#### Permission Denied

```
PermissionError: [Errno 13] Permission denied
```

**Solution:** Don't use `sudo`. Use a virtual environment:
```bash
conda activate stickler
# or
source .venv/bin/activate
```

#### Tests Not Found

```
no tests ran
```

**Solution:** Run from project root:
```bash
cd /path/to/stickler
pytest tests/
```

## Getting Help

If you encounter issues not covered here:

1. Check [Known Issues](../known-issues.md)
2. Search existing [GitHub Issues](https://github.com/awslabs/stickler/issues)
3. Open a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

## Next Steps

Once your environment is set up:

1. Read the [Testing Guide](testing-guide.md) to understand testing conventions
2. Review the [Code Style Guide](code-style.md) for coding standards
3. Check the [Pull Request Guide](pull-request-guide.md) when ready to contribute
