# stickler

A Python library for structured object comparison and evaluation with configurable comparison strategies and detailed metrics.

## Installation
```bash
pip install stickler-eval
```



### Requirements
- Python 3.12+
- conda (recommended)

### Quick Install
```bash
# Create conda environment
conda create -n stickler python=3.12 -y
conda activate stickler

# Install the library
pip install -e .
```

### Development Install
```bash
# Install with testing dependencies
pip install -e ".[dev]"
```

## Quick Test

Run the example to verify installation:
```bash
python examples/scripts/quick_start.py
```

Run tests:
```bash
pytest tests/
```

## Basic Usage

### Static Model Definition

```python
from stickler import StructuredModel, ComparableField, StructuredModelEvaluator
from stickler.comparators.levenshtein import LevenshteinComparator

# Define your data structure
class Invoice(StructuredModel):
    invoice_number: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.9
    )
    total: float = ComparableField(threshold=0.95)

# Compare objects
evaluator = StructuredModelEvaluator()
result = evaluator.evaluate(ground_truth, prediction)

print(f"Overall Score: {result['overall']['anls_score']:.3f}")
```

### Dynamic Model Creation (New!)

Create models from JSON configuration for maximum flexibility:

```python
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

# Define model configuration
config = {
    "model_name": "Product",
    "match_threshold": 0.8,
    "fields": {
        "name": {
            "type": "str",
            "comparator": "LevenshteinComparator",
            "threshold": 0.8,
            "weight": 2.0
        },
        "price": {
            "type": "float",
            "comparator": "NumericComparator",
            "default": 0.0
        }
    }
}

# Create dynamic model class
Product = StructuredModel.model_from_json(config)

# Use like any Pydantic model
product1 = Product(name="Widget", price=29.99)
product2 = Product(name="Gadget", price=29.99)

# Full comparison capabilities
result = product1.compare_with(product2)
print(f"Similarity: {result['overall_score']:.2f}")
```

### Complete JSON-to-Evaluation Workflow (New!)

For maximum flexibility, load both configuration AND data from JSON:

```python
# Load model config from JSON
with open('model_config.json') as f:
    config = json.load(f)

# Load test data from JSON  
with open('test_data.json') as f:
    data = json.load(f)

# Create model and instances from JSON
Model = StructuredModel.model_from_json(config)
ground_truth = Model(**data['ground_truth'])
prediction = Model(**data['prediction'])

# Evaluate - no Python object construction needed!
result = ground_truth.compare_with(prediction)
```

**Benefits of JSON-Driven Approach:**
- Zero Python object construction required
- Configuration-driven model creation
- A/B testing different field configurations
- Runtime model generation from external schemas
- Production-ready JSON-based evaluation pipeline
- Full Pydantic compatibility with comparison capabilities

See [`examples/scripts/json_to_evaluation_demo.py`](examples/scripts/json_to_evaluation_demo.py) for a complete working example and [`docs/StructuredModel_Dynamic_Creation.md`](docs/StructuredModel_Dynamic_Creation.md) for comprehensive documentation.

## Examples

Check out the `examples/` directory for more detailed usage examples and notebooks.

