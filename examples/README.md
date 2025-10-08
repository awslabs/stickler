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

### 3. Interactive Notebooks
- **`notebooks/Quick_start.ipynb`** - Interactive introduction
  - Step-by-step guided examples  
  - Individual and list comparison
  - Metrics interpretation

- **`notebooks/Complex_nested_structure.ipynb`** - Advanced structures
  - Deeply nested object evaluation
  - Optional field handling
  - Complex error analysis

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
1. Install dependencies: `pip install -r requirements.txt`
2. Check Python version: Requires Python 3.8+
3. Verify installation: `python -c "import stickler; print('Success!')"`

## 🎉 Key Insight

The **Hungarian algorithm for list matching** is what makes this library special - it finds optimal pairings between objects even when they're in different orders or partially missing. This is demonstrated throughout the examples and is the core strength for real-world structured data evaluation.
