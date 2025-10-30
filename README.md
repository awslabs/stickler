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

## JSON Schema Extensions: `x-aws-stickler-*` Complete Reference

Stickler supports standard JSON Schema (Draft 7+) with custom `x-aws-stickler-*` extensions for controlling comparison behavior. These extensions let you configure exactly how each field is evaluated without writing Python code.

### Why Use JSON Schema Extensions?

- **Configuration-driven evaluation**: Define models and comparison logic in JSON
- **No Python code required**: Perfect for non-Python systems or runtime configuration
- **Version control friendly**: Track evaluation logic changes alongside your schemas
- **A/B testing**: Easily test different comparison strategies
- **Integration ready**: Works with existing JSON Schema tooling and validators

### Field-Level Extensions

Add these to any property in your JSON Schema to control comparison behavior:

#### `x-aws-stickler-comparator`

**Type:** `string`  
**Required:** No  
**Default:** Type-dependent (see table below)

Specifies the comparison algorithm for this field.

**Available Comparators:**

| Comparator Name | Best For | How It Works |
|-----------------|----------|--------------|
| `"LevenshteinComparator"` | Names, addresses, text with typos | Calculates edit distance between strings. Score = 1 - (edits / max_length) |
| `"ExactComparator"` | IDs, codes, booleans, exact matches | Returns 1.0 for exact match, 0.0 otherwise |
| `"NumericComparator"` | Prices, quantities, measurements | Compares numbers with configurable tolerance |
| `"FuzzyComparator"` | Flexible text, descriptions | Token-based fuzzy matching (order-independent) |
| `"SemanticComparator"` | Semantic similarity | Embedding-based comparison for meaning |
| `"BertComparator"` | Deep semantic understanding | BERT model for contextual similarity |
| `"LLMComparator"` | Complex semantic evaluation | LLM-powered comparison with reasoning |

**Default Comparators by JSON Schema Type:**

| JSON Schema Type | Default Comparator | Default Threshold | Rationale |
|------------------|-------------------|-------------------|-----------|
| `"string"` | `LevenshteinComparator` | `0.5` | Handles typos and minor variations |
| `"number"` | `NumericComparator` | `0.5` | Tolerates small numeric differences |
| `"integer"` | `NumericComparator` | `0.5` | Tolerates small numeric differences |
| `"boolean"` | `ExactComparator` | `1.0` | Must be exactly true or false |
| `"array"` (primitives) | Based on item type | Based on item type | Inherits from element type |
| `"array"` (objects) | Hungarian matching | `0.7` | Optimal pairing of list elements |
| `"object"` | Recursive comparison | `0.7` | Field-by-field nested comparison |

**Example:**

```json
{
  "type": "object",
  "properties": {
    "invoice_id": {
      "type": "string",
      "description": "Must match exactly - routing depends on this",
      "x-aws-stickler-comparator": "ExactComparator"
    },
    "customer_name": {
      "type": "string",
      "description": "Allow typos but catch major errors",
      "x-aws-stickler-comparator": "LevenshteinComparator"
    },
    "description": {
      "type": "string",
      "description": "Flexible matching for free-form text",
      "x-aws-stickler-comparator": "FuzzyComparator"
    }
  }
}
```

#### `x-aws-stickler-threshold`

**Type:** `number` (0.0 to 1.0, inclusive)  
**Required:** No  
**Default:** `0.5` (or `1.0` for booleans)

Minimum similarity score required for binary match classification.

**How Threshold Works:**
- Similarity score **≥ threshold** → Classified as **MATCH** (True Positive or True Negative)
- Similarity score **< threshold** → Classified as **NO MATCH** (False Positive or False Negative)

**Threshold Selection Guide:**

| Threshold Range | Use Case | Example Fields |
|-----------------|----------|----------------|
| **1.0** | Must be perfect | IDs, codes, critical identifiers |
| **0.9 - 0.95** | Very important, minimal tolerance | Amounts, dates, status codes |
| **0.8 - 0.85** | Important, some tolerance | Names, addresses, product codes |
| **0.7 - 0.75** | Moderate tolerance | Descriptions, categories |
| **0.5 - 0.6** | Flexible, variations OK | Notes, comments, metadata |

**Example:**

```json
{
  "type": "object",
  "properties": {
    "shipment_id": {
      "type": "string",
      "description": "CRITICAL: Wrong ID = wrong warehouse",
      "x-aws-stickler-comparator": "ExactComparator",
      "x-aws-stickler-threshold": 1.0
    },
    "total_amount": {
      "type": "number",
      "description": "Very important: affects billing",
      "x-aws-stickler-comparator": "NumericComparator",
      "x-aws-stickler-threshold": 0.95
    },
    "customer_name": {
      "type": "string",
      "description": "Important: allow minor typos",
      "x-aws-stickler-comparator": "LevenshteinComparator",
      "x-aws-stickler-threshold": 0.8
    },
    "driver_notes": {
      "type": "string",
      "description": "Flexible: variations don't affect operations",
      "x-aws-stickler-comparator": "FuzzyComparator",
      "x-aws-stickler-threshold": 0.6
    }
  }
}
```

#### `x-aws-stickler-weight`

**Type:** `number` (positive float, > 0.0)  
**Required:** No  
**Default:** `1.0`

Relative importance of this field in aggregate scoring and weighted averages.

**How Weights Work:**
- Fields with `weight > 1.0` have **greater influence** on overall scores
- Fields with `weight < 1.0` have **less influence** on overall scores
- Used in weighted average calculations: `overall_score = Σ(field_score × weight) / Σ(weight)`

**Weight Selection Guide:**

| Weight Range | Priority Level | Use Case |
|--------------|----------------|----------|
| **2.5 - 3.0** | Critical | Business-critical fields (IDs, amounts, status) |
| **1.5 - 2.0** | High | Important operational fields |
| **1.0** | Normal | Standard fields |
| **0.5 - 0.8** | Low | Nice-to-have fields |
| **0.1 - 0.3** | Minimal | Metadata, debug fields |

**Example:**

```json
{
  "type": "object",
  "properties": {
    "order_id": {
      "type": "string",
      "description": "CRITICAL: Wrong order = wrong customer",
      "x-aws-stickler-comparator": "ExactComparator",
      "x-aws-stickler-threshold": 1.0,
      "x-aws-stickler-weight": 3.0
    },
    "total_amount": {
      "type": "number",
      "description": "Very important: affects billing",
      "x-aws-stickler-comparator": "NumericComparator",
      "x-aws-stickler-threshold": 0.95,
      "x-aws-stickler-weight": 2.5
    },
    "customer_name": {
      "type": "string",
      "description": "Important but not critical",
      "x-aws-stickler-comparator": "LevenshteinComparator",
      "x-aws-stickler-threshold": 0.8,
      "x-aws-stickler-weight": 1.0
    },
    "internal_notes": {
      "type": "string",
      "description": "Low priority: internal use only",
      "x-aws-stickler-comparator": "FuzzyComparator",
      "x-aws-stickler-threshold": 0.6,
      "x-aws-stickler-weight": 0.3
    }
  }
}
```

#### `x-aws-stickler-clip-under-threshold`

**Type:** `boolean`  
**Required:** No  
**Default:** `false`

Controls whether similarity scores below threshold are clipped to 0.0.

**How Clipping Works:**
- `true`: Scores below threshold are **set to 0.0** (hard cutoff)
- `false`: Actual similarity scores are **preserved** (soft cutoff)

**Important:** This affects **continuous similarity metrics** but NOT **binary classification** (TP/FP/TN/FN). Binary classification always uses threshold as a hard cutoff.

**When to Use:**

| Setting | Use Case | Effect |
|---------|----------|--------|
| `true` | Critical fields where partial matches are meaningless | Score is either threshold or 0.0 |
| `false` | Fields where partial similarity has value | Preserves granular similarity information |

**Example:**

```json
{
  "type": "object",
  "properties": {
    "status_code": {
      "type": "string",
      "description": "Either correct or completely wrong",
      "x-aws-stickler-comparator": "ExactComparator",
      "x-aws-stickler-threshold": 1.0,
      "x-aws-stickler-clip-under-threshold": true
    },
    "description": {
      "type": "string",
      "description": "Partial matches still have value",
      "x-aws-stickler-comparator": "LevenshteinComparator",
      "x-aws-stickler-threshold": 0.8,
      "x-aws-stickler-clip-under-threshold": false
    }
  }
}
```

#### `x-aws-stickler-aggregate`

**Type:** `boolean`  
**Required:** No  
**Default:** `false`

Controls whether this field's confusion matrix metrics (TP/FP/TN/FN) are included in parent-level aggregate counts.

**How Aggregation Works:**
- `true`: Field's metrics are **included** in parent's aggregate counts
- `false`: Field's metrics are **calculated but not aggregated** to parent

**When to Use:**

| Setting | Use Case |
|---------|----------|
| `true` | Include field in overall accuracy/precision/recall calculations |
| `false` | Exclude field from aggregate metrics (debugging, metadata, experimental fields) |

**Example:**

```json
{
  "type": "object",
  "properties": {
    "invoice_id": {
      "type": "string",
      "description": "Include in accuracy metrics",
      "x-aws-stickler-comparator": "ExactComparator",
      "x-aws-stickler-threshold": 1.0,
      "x-aws-stickler-aggregate": true
    },
    "debug_field": {
      "type": "string",
      "description": "For debugging only - don't affect metrics",
      "x-aws-stickler-comparator": "ExactComparator",
      "x-aws-stickler-threshold": 1.0,
      "x-aws-stickler-aggregate": false
    }
  }
}
```

### Model-Level Extensions

Add these at the root level of your JSON Schema:

#### `x-aws-stickler-model-name`

**Type:** `string`  
**Required:** No  
**Default:** `"DynamicModel"`

Name of the generated Python class.

```json
{
  "type": "object",
  "x-aws-stickler-model-name": "Invoice",
  "properties": { ... }
}
```

#### `x-aws-stickler-match-threshold`

**Type:** `number` (0.0 to 1.0)  
**Required:** No  
**Default:** `0.7`

Overall matching threshold for the model. Used for Hungarian algorithm matching when comparing lists of objects.

```json
{
  "type": "object",
  "x-aws-stickler-model-name": "LineItem",
  "x-aws-stickler-match-threshold": 0.8,
  "properties": { ... }
}
```

### Complete Real-World Example

Here's a production-ready invoice schema with all extensions:

```json
{
  "type": "object",
  "x-aws-stickler-model-name": "Invoice",
  "x-aws-stickler-match-threshold": 0.75,
  "description": "Invoice extraction schema with business-aligned comparison",
  "properties": {
    "invoice_id": {
      "type": "string",
      "description": "Unique invoice identifier - must be exact",
      "examples": ["INV-2024-001", "INV-2024-002"],
      "x-aws-stickler-comparator": "ExactComparator",
      "x-aws-stickler-threshold": 1.0,
      "x-aws-stickler-weight": 3.0,
      "x-aws-stickler-clip-under-threshold": true,
      "x-aws-stickler-aggregate": true
    },
    "customer_name": {
      "type": "string",
      "description": "Customer's full name - allow minor typos",
      "examples": ["John Smith", "Acme Corporation"],
      "x-aws-stickler-comparator": "LevenshteinComparator",
      "x-aws-stickler-threshold": 0.8,
      "x-aws-stickler-weight": 1.5,
      "x-aws-stickler-clip-under-threshold": false,
      "x-aws-stickler-aggregate": true
    },
    "total_amount": {
      "type": "number",
      "description": "Total invoice amount in USD - critical for billing",
      "examples": [1234.56, 99.99],
      "x-aws-stickler-comparator": "NumericComparator",
      "x-aws-stickler-threshold": 0.95,
      "x-aws-stickler-weight": 2.5,
      "x-aws-stickler-clip-under-threshold": false,
      "x-aws-stickler-aggregate": true
    },
    "line_items": {
      "type": "array",
      "description": "Individual line items - uses Hungarian matching",
      "items": {
        "type": "object",
        "properties": {
          "description": {
            "type": "string",
            "description": "Item description",
            "x-aws-stickler-comparator": "FuzzyComparator",
            "x-aws-stickler-threshold": 0.7,
            "x-aws-stickler-weight": 1.0
          },
          "quantity": {
            "type": "integer",
            "description": "Item quantity",
            "x-aws-stickler-comparator": "NumericComparator",
            "x-aws-stickler-threshold": 1.0,
            "x-aws-stickler-weight": 1.2
          },
          "unit_price": {
            "type": "number",
            "description": "Price per unit",
            "x-aws-stickler-comparator": "NumericComparator",
            "x-aws-stickler-threshold": 0.95,
            "x-aws-stickler-weight": 1.5
          }
        },
        "required": ["description", "quantity", "unit_price"]
      }
    },
    "internal_notes": {
      "type": "string",
      "description": "Internal processing notes - low priority",
      "x-aws-stickler-comparator": "FuzzyComparator",
      "x-aws-stickler-threshold": 0.5,
      "x-aws-stickler-weight": 0.2,
      "x-aws-stickler-clip-under-threshold": false,
      "x-aws-stickler-aggregate": false
    }
  },
  "required": ["invoice_id", "customer_name", "total_amount", "line_items"]
}
```

### Using Your Schema

```python
from stickler import StructuredModel
import json

# Load schema
with open('invoice_schema.json') as f:
    schema = json.load(f)

# Create model class
Invoice = StructuredModel.from_json_schema(schema)

# Load test data
ground_truth = Invoice(**{
    "invoice_id": "INV-2024-001",
    "customer_name": "Acme Corporation",
    "total_amount": 1250.00,
    "line_items": [
        {"description": "Widget A", "quantity": 10, "unit_price": 50.00},
        {"description": "Widget B", "quantity": 5, "unit_price": 100.00}
    ],
    "internal_notes": "Processed by system A"
})

prediction = Invoice(**{
    "invoice_id": "INV-2024-001",
    "customer_name": "ACME Corp",  # Variation
    "total_amount": 1250.00,
    "line_items": [
        {"description": "Widget B", "quantity": 5, "unit_price": 100.00},  # Reordered
        {"description": "Widget A", "quantity": 10, "unit_price": 50.00}   # Reordered
    ],
    "internal_notes": "Processed by system B"  # Different
})

# Compare
result = ground_truth.compare_with(prediction)

print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Invoice ID: {result['field_scores']['invoice_id']:.3f}")  # 1.000 - exact
print(f"Customer: {result['field_scores']['customer_name']:.3f}")  # ~0.85 - close
print(f"Line Items: {result['field_scores']['line_items']:.3f}")  # ~1.0 - matched
```

### Quick Reference

| Extension | Type | Default | Purpose |
|-----------|------|---------|---------|
| `x-aws-stickler-comparator` | string | Type-dependent | Comparison algorithm |
| `x-aws-stickler-threshold` | number (0.0-1.0) | 0.5 or 1.0 | Match classification cutoff |
| `x-aws-stickler-weight` | number (> 0.0) | 1.0 | Field importance multiplier |
| `x-aws-stickler-clip-under-threshold` | boolean | false | Zero out low scores |
| `x-aws-stickler-aggregate` | boolean | false | Include in parent metrics |
| `x-aws-stickler-model-name` | string | "DynamicModel" | Generated class name |
| `x-aws-stickler-match-threshold` | number (0.0-1.0) | 0.7 | Model-level threshold |

### Additional Resources

- **Complete examples**: [`examples/scripts/json_schema_demo.py`](examples/scripts/json_schema_demo.py)
- **Dynamic model creation**: [`docs/StructuredModel_Dynamic_Creation.md`](docs/StructuredModel_Dynamic_Creation.md)
- **Comparator details**: [`src/stickler/comparators/Comparators.md`](src/stickler/comparators/Comparators.md)

## Examples

Check out the `examples/` directory for more detailed usage examples and notebooks.
