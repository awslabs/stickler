---
title: Quick Start
---

# Stickler: Structured Object Evaluation for GenAI

When in the course of human events, it becomes necessary to evaluate structured outputs from generative AI systems, we must acknowledge that traditional evaluation treats all fields equally. But **not all fields are created equal**.

**Stickler is a Python library for structured object comparison and evaluation** that lets you focus on the fields your customer actually cares about, to answer the question: "Is it doing a good job?" 

Stickler uses specialized comparators for different data types: exact matching for critical identifiers, numeric tolerance for currency amounts, semantic similarity for text fields, and fuzzy matching for names and addresses. You can build custom comparators for domain-specific logic. The Hungarian algorithm ensures optimal list matching regardless of order, while the recursive evaluation engine handles unlimited nesting depth. Business-weighted scoring reflects actual operational impact, not just technical accuracy.

Consider an invoice extraction agent that perfectly captures shipment numbers—which must be exact or packages get routed to the wrong warehouse—but sometimes garbles driver notes like "delivered to front door" vs "left at entrance." Those note variations don't affect logistics operations at all. Traditional evaluation treats both error types identically and reports your agent as "95% accurate" without telling you if that 5% error rate matters. Stickler tells you exactly where the errors are and whether they're actually problems.

Whether you're extracting data from documents, performing ETL transformations, evaluating ML model outputs, or simply trying to diff complex JSON structures, Stickler transforms evaluation from a technical afterthought into a business-aligned decision tool.

## Installation
```bash
pip install stickler-eval
```

## Get Started in 30 Seconds

```python
# pip install stickler-eval
from typing import List
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, NumericComparator, LevenshteinComparator

# Define your models
class LineItem(StructuredModel):
    product: str = ComparableField(comparator=LevenshteinComparator(), weight=1.0)
    quantity: int = ComparableField(weight=0.8)
    price: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=1.2)

class Invoice(StructuredModel):
    shipment_id: str = ComparableField(comparator=ExactComparator(), weight=3.0)  # Critical
    amount: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=2.0)
    line_items: List[LineItem] = ComparableField(weight=2.0)  # Hungarian matching!

# JSON from your systems (agent output, ground truth, etc.)
ground_truth_json = {
    "shipment_id": "SHP-2024-001",
    "amount": 1247.50,
    "line_items": [
        {"product": "Wireless Mouse", "quantity": 2, "price": 29.99},
        {"product": "USB Cable", "quantity": 5, "price": 12.99}
    ]
}

prediction_json = {
    "shipment_id": "SHP-2024-001",  # Perfect match
    "amount": 1247.48,  # Within tolerance
    "line_items": [
        {"product": "USB Cord", "quantity": 5, "price": 12.99},  # Name variation
        {"product": "Wireless Mouse", "quantity": 2, "price": 29.99}  # Reordered
    ]
}

# Construct from JSON and compare
ground_truth = Invoice(**ground_truth_json)
prediction = Invoice(**prediction_json)
result = ground_truth.compare_with(prediction)

print(f"Overall Score: {result['overall_score']:.3f}")  # 0.693
print(f"Shipment ID: {result['field_scores']['shipment_id']:.3f}")  # 1.000 - exact match
print(f"Line Items: {result['field_scores']['line_items']:.3f}")  # 0.926 - Hungarian optimal matching
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
