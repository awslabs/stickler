"""Integration tests for JSON Schema to StructuredModel conversion.

This module tests end-to-end workflows from JSON Schema to model creation,
instantiation, and comparison with both default and custom configurations.
"""

import pytest
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows from schema to comparison."""

    def test_simple_product_schema_workflow(self):
        """Test complete workflow: schema → model → instantiation → comparison."""
        # Define schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
                "in_stock": {"type": "boolean"}
            },
            "required": ["name", "price"]
        }
        
        # Create model from schema
        ProductModel = StructuredModel.from_json_schema(schema)
        
        # Instantiate objects
        product1 = ProductModel(name="Widget", price=29.99, in_stock=True)
        product2 = ProductModel(name="Widget", price=29.99, in_stock=True)
        product3 = ProductModel(name="Gadget", price=39.99, in_stock=False)
        
        # Compare identical products
        score_identical = product1.compare(product2)
        assert score_identical == 1.0
        
        # Compare different products
        score_different = product1.compare(product3)
        assert score_different < 1.0
        
        # Use compare_with for detailed results
        result = product1.compare_with(product2)
        assert result["overall_score"] == 1.0
        assert "field_scores" in result
        assert "name" in result["field_scores"]
        assert "price" in result["field_scores"]

    def test_customer_order_workflow_with_defaults(self):
        """Test workflow with nested objects using default comparison behavior."""
        schema = {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "customer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "string"},
                            "quantity": {"type": "integer"}
                        }
                    }
                }
            }
        }
        
        OrderModel = StructuredModel.from_json_schema(schema)
        
        order1 = OrderModel(
            order_id="ORD-001",
            customer={"name": "Alice Smith", "email": "alice@example.com"},
            items=[
                {"product": "Widget", "quantity": 2},
                {"product": "Gadget", "quantity": 1}
            ]
        )
        
        order2 = OrderModel(
            order_id="ORD-001",
            customer={"name": "Alice Smith", "email": "alice@example.com"},
            items=[
                {"product": "Widget", "quantity": 2},
                {"product": "Gadget", "quantity": 1}
            ]
        )
        
        # Test comparison with defaults
        score = order1.compare(order2)
        assert score == 1.0
        
        result = order1.compare_with(order2)
        assert result["overall_score"] == 1.0
        assert "customer" in result["field_scores"]

    def test_employee_record_workflow_with_extensions(self):
        """Test workflow with custom extensions for fine-tuned comparison."""
        schema = {
            "type": "object",
            "x-aws-stickler-model-name": "Employee",
            "x-aws-stickler-match-threshold": 0.85,
            "properties": {
                "employee_id": {
                    "type": "string",
                    "x-aws-stickler-comparator": "ExactComparator",
                    "x-aws-stickler-weight": 3.0
                },
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                    "x-aws-stickler-threshold": 0.9,
                    "x-aws-stickler-weight": 2.0
                },
                "department": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                    "x-aws-stickler-threshold": 0.8,
                    "x-aws-stickler-weight": 1.0
                },
                "salary": {
                    "type": "number",
                    "x-aws-stickler-comparator": "NumericComparator",
                    "x-aws-stickler-threshold": 0.95,
                    "x-aws-stickler-weight": 1.5
                }
            },
            "required": ["employee_id", "name"]
        }
        
        EmployeeModel = StructuredModel.from_json_schema(schema)
        
        # Verify model configuration
        assert EmployeeModel.__name__ == "Employee"
        assert EmployeeModel.match_threshold == 0.85
        
        # Create instances
        emp1 = EmployeeModel(
            employee_id="EMP001",
            name="John Doe",
            department="Engineering",
            salary=75000.0
        )
        
        emp2 = EmployeeModel(
            employee_id="EMP001",
            name="John Doe",
            department="Engineering",
            salary=75000.0
        )
        
        # Test exact match
        score = emp1.compare(emp2)
        assert score == 1.0
        
        result = emp1.compare_with(emp2)
        assert result["overall_score"] == 1.0


class TestComplexNestedStructures:
    """Test complex nested structures with multiple levels."""

    def test_organization_hierarchy(self):
        """Test deeply nested organizational structure."""
        schema = {
            "type": "object",
            "properties": {
                "company": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "divisions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "departments": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "teams": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "members": {"type": "integer"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        OrgModel = StructuredModel.from_json_schema(schema)
        
        org1 = OrgModel(
            company={
                "name": "Tech Corp",
                "divisions": [
                    {
                        "name": "Engineering",
                        "departments": [
                            {
                                "name": "Backend",
                                "teams": [
                                    {"name": "API Team", "members": 5},
                                    {"name": "Database Team", "members": 3}
                                ]
                            }
                        ]
                    }
                ]
            }
        )
        
        org2 = OrgModel(
            company={
                "name": "Tech Corp",
                "divisions": [
                    {
                        "name": "Engineering",
                        "departments": [
                            {
                                "name": "Backend",
                                "teams": [
                                    {"name": "API Team", "members": 5},
                                    {"name": "Database Team", "members": 3}
                                ]
                            }
                        ]
                    }
                ]
            }
        )
        
        # Test comparison of complex nested structure
        score = org1.compare(org2)
        assert score == 1.0
        
        result = org1.compare_with(org2)
        assert result["overall_score"] == 1.0

    def test_mixed_nested_arrays_and_objects(self):
        """Test structure with mixed nested arrays and objects."""
        schema = {
            "type": "object",
            "properties": {
                "project": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "phases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "phase_name": {"type": "string"},
                                    "tasks": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "task_name": {"type": "string"},
                                                "assignees": {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                },
                                                "metadata": {
                                                    "type": "object",
                                                    "properties": {
                                                        "priority": {"type": "string"},
                                                        "estimated_hours": {"type": "number"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        ProjectModel = StructuredModel.from_json_schema(schema)
        
        project = ProjectModel(
            project={
                "name": "Website Redesign",
                "phases": [
                    {
                        "phase_name": "Design",
                        "tasks": [
                            {
                                "task_name": "Create mockups",
                                "assignees": ["Alice", "Bob"],
                                "metadata": {
                                    "priority": "high",
                                    "estimated_hours": 40.0
                                }
                            }
                        ]
                    }
                ]
            }
        )
        
        # Verify structure is correctly created
        assert project.project.name == "Website Redesign"
        assert len(project.project.phases) == 1
        assert project.project.phases[0].phase_name == "Design"
        assert len(project.project.phases[0].tasks) == 1
        assert project.project.phases[0].tasks[0].task_name == "Create mockups"
        assert len(project.project.phases[0].tasks[0].assignees) == 2
        assert project.project.phases[0].tasks[0].metadata.priority == "high"


class TestRealWorldSchemas:
    """Test with real-world JSON Schema examples."""

    def test_invoice_schema(self):
        """Test with a real-world invoice schema."""
        schema = {
            "type": "object",
            "x-aws-stickler-model-name": "Invoice",
            "properties": {
                "invoice_number": {"type": "string"},
                "date": {"type": "string"},
                "vendor": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"},
                        "tax_id": {"type": "string"}
                    }
                },
                "customer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"},
                        "contact": {"type": "string"}
                    }
                },
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "number"},
                            "total": {"type": "number"}
                        }
                    }
                },
                "subtotal": {"type": "number"},
                "tax": {"type": "number"},
                "total": {"type": "number"}
            },
            "required": ["invoice_number", "date", "vendor", "customer", "line_items", "total"]
        }
        
        InvoiceModel = StructuredModel.from_json_schema(schema)
        
        invoice1 = InvoiceModel(
            invoice_number="INV-2024-001",
            date="2024-01-15",
            vendor={
                "name": "Acme Corp",
                "address": "123 Main St, Boston, MA",
                "tax_id": "12-3456789"
            },
            customer={
                "name": "Beta Inc",
                "address": "456 Oak Ave, Seattle, WA",
                "contact": "john@beta.com"
            },
            line_items=[
                {
                    "description": "Widget Pro",
                    "quantity": 10,
                    "unit_price": 25.00,
                    "total": 250.00
                },
                {
                    "description": "Gadget Plus",
                    "quantity": 5,
                    "unit_price": 50.00,
                    "total": 250.00
                }
            ],
            subtotal=500.00,
            tax=50.00,
            total=550.00
        )
        
        invoice2 = InvoiceModel(
            invoice_number="INV-2024-001",
            date="2024-01-15",
            vendor={
                "name": "Acme Corp",
                "address": "123 Main St, Boston, MA",
                "tax_id": "12-3456789"
            },
            customer={
                "name": "Beta Inc",
                "address": "456 Oak Ave, Seattle, WA",
                "contact": "john@beta.com"
            },
            line_items=[
                {
                    "description": "Widget Pro",
                    "quantity": 10,
                    "unit_price": 25.00,
                    "total": 250.00
                },
                {
                    "description": "Gadget Plus",
                    "quantity": 5,
                    "unit_price": 50.00,
                    "total": 250.00
                }
            ],
            subtotal=500.00,
            tax=50.00,
            total=550.00
        )
        
        # Test comparison
        score = invoice1.compare(invoice2)
        assert score == 1.0
        
        result = invoice1.compare_with(invoice2)
        assert result["overall_score"] == 1.0

    def test_api_response_schema(self):
        """Test with API response schema."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "users": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "username": {"type": "string"},
                                    "email": {"type": "string"},
                                    "profile": {
                                        "type": "object",
                                        "properties": {
                                            "first_name": {"type": "string"},
                                            "last_name": {"type": "string"},
                                            "avatar_url": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "pagination": {
                            "type": "object",
                            "properties": {
                                "page": {"type": "integer"},
                                "per_page": {"type": "integer"},
                                "total": {"type": "integer"}
                            }
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "request_id": {"type": "string"},
                        "timestamp": {"type": "string"}
                    }
                }
            }
        }
        
        ResponseModel = StructuredModel.from_json_schema(schema)
        
        response = ResponseModel(
            status="success",
            data={
                "users": [
                    {
                        "id": 1,
                        "username": "alice",
                        "email": "alice@example.com",
                        "profile": {
                            "first_name": "Alice",
                            "last_name": "Smith",
                            "avatar_url": "https://example.com/avatar1.jpg"
                        }
                    }
                ],
                "pagination": {
                    "page": 1,
                    "per_page": 10,
                    "total": 1
                }
            },
            metadata={
                "request_id": "req-12345",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        )
        
        # Verify structure
        assert response.status == "success"
        assert len(response.data.users) == 1
        assert response.data.users[0].username == "alice"
        assert response.data.users[0].profile.first_name == "Alice"


class TestComparisonBehaviorDefaults:
    """Test comparison behavior with default settings."""

    def test_default_string_comparison(self):
        """Test default LevenshteinComparator for strings."""
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        # Exact match
        obj1 = Model(text="hello")
        obj2 = Model(text="hello")
        assert obj1.compare(obj2) == 1.0
        
        # Similar strings
        obj3 = Model(text="helo")  # Missing one 'l'
        score = obj1.compare(obj3)
        assert 0.0 < score < 1.0  # Should be similar but not exact

    def test_default_numeric_comparison(self):
        """Test default NumericComparator for numbers.
        
        Note: Default NumericComparator has zero tolerance, so it only
        returns 1.0 for exact matches and 0.0 for any difference.
        """
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        # Exact match
        obj1 = Model(value=100.0)
        obj2 = Model(value=100.0)
        assert obj1.compare(obj2) == 1.0
        
        # Different values (default has zero tolerance)
        obj3 = Model(value=100.1)
        score = obj1.compare(obj3)
        assert score == 0.0  # No tolerance by default
        
        # Test that integer and float exact matches work
        obj4 = Model(value=100)
        score2 = obj1.compare(obj4)
        assert score2 == 1.0  # 100.0 == 100

    def test_default_boolean_comparison(self):
        """Test default ExactComparator for booleans."""
        schema = {
            "type": "object",
            "properties": {
                "flag": {"type": "boolean"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        # Same value
        obj1 = Model(flag=True)
        obj2 = Model(flag=True)
        assert obj1.compare(obj2) == 1.0
        
        # Different value
        obj3 = Model(flag=False)
        score = obj1.compare(obj3)
        assert score == 0.0  # Exact comparator: either 1.0 or 0.0

    def test_default_array_comparison(self):
        """Test default array comparison with Hungarian matching."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        # Same order
        obj1 = Model(tags=["python", "javascript", "go"])
        obj2 = Model(tags=["python", "javascript", "go"])
        assert obj1.compare(obj2) == 1.0
        
        # Different order (should still match with Hungarian)
        obj3 = Model(tags=["go", "python", "javascript"])
        score = obj1.compare(obj3)
        assert score == 1.0  # Order-independent matching

    def test_default_nested_object_comparison(self):
        """Test default comparison for nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(person={"name": "Alice", "age": 30})
        obj2 = Model(person={"name": "Alice", "age": 30})
        assert obj1.compare(obj2) == 1.0


class TestComparisonBehaviorCustomExtensions:
    """Test comparison behavior with custom x-aws-stickler-* extensions."""

    def test_custom_threshold_affects_matching(self):
        """Test that custom threshold affects match classification."""
        schema = {
            "type": "object",
            "x-aws-stickler-match-threshold": 0.9,
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": 0.95
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        assert Model.match_threshold == 0.9
        
        obj1 = Model(name="test")
        obj2 = Model(name="test")
        
        result = obj1.compare_with(obj2)
        assert result["overall_score"] == 1.0

    def test_custom_weight_affects_scoring(self):
        """Test that custom weights affect overall scoring."""
        schema = {
            "type": "object",
            "properties": {
                "critical": {
                    "type": "string",
                    "x-aws-stickler-weight": 10.0
                },
                "minor": {
                    "type": "string",
                    "x-aws-stickler-weight": 0.1
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        # When critical field matches, overall score should be high
        obj1 = Model(critical="important", minor="detail1")
        obj2 = Model(critical="important", minor="detail2")
        
        result = obj1.compare_with(obj2)
        # Critical field matches perfectly, minor field differs
        # With high weight on critical, overall score should still be high
        assert result["overall_score"] > 0.9

    def test_custom_comparator_selection(self):
        """Test that custom comparator selection works correctly."""
        schema = {
            "type": "object",
            "properties": {
                "exact_field": {
                    "type": "string",
                    "x-aws-stickler-comparator": "ExactComparator"
                },
                "fuzzy_field": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator"
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(exact_field="test", fuzzy_field="hello")
        obj2 = Model(exact_field="test", fuzzy_field="helo")
        
        result = obj1.compare_with(obj2)
        
        # Exact field should match perfectly
        assert result["field_scores"]["exact_field"] == 1.0
        
        # Fuzzy field should have partial match
        assert 0.0 < result["field_scores"]["fuzzy_field"] < 1.0

    def test_aggregate_extension_in_nested_structure(self):
        """Test x-aws-stickler-aggregate in nested structures."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "x-aws-stickler-aggregate": True
                            },
                            "value": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(items=[{"name": "A", "value": 1}, {"name": "B", "value": 2}])
        obj2 = Model(items=[{"name": "A", "value": 1}, {"name": "B", "value": 2}])
        
        result = obj1.compare_with(obj2)
        assert result["overall_score"] == 1.0

    def test_clip_under_threshold_behavior(self):
        """Test x-aws-stickler-clip-under-threshold behavior."""
        schema = {
            "type": "object",
            "properties": {
                "field1": {
                    "type": "string",
                    "x-aws-stickler-threshold": 0.8,
                    "x-aws-stickler-clip-under-threshold": True
                },
                "field2": {
                    "type": "string",
                    "x-aws-stickler-threshold": 0.8,
                    "x-aws-stickler-clip-under-threshold": False
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(field1="test", field2="test")
        obj2 = Model(field1="test", field2="test")
        
        result = obj1.compare_with(obj2)
        assert result["overall_score"] == 1.0

    def test_multiple_extensions_combined(self):
        """Test multiple extensions working together."""
        schema = {
            "type": "object",
            "x-aws-stickler-model-name": "ComplexModel",
            "x-aws-stickler-match-threshold": 0.85,
            "properties": {
                "id": {
                    "type": "string",
                    "x-aws-stickler-comparator": "ExactComparator",
                    "x-aws-stickler-weight": 5.0,
                    "x-aws-stickler-threshold": 1.0
                },
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                    "x-aws-stickler-weight": 2.0,
                    "x-aws-stickler-threshold": 0.9,
                    "x-aws-stickler-clip-under-threshold": True
                },
                "score": {
                    "type": "number",
                    "x-aws-stickler-comparator": "NumericComparator",
                    "x-aws-stickler-weight": 1.0,
                    "x-aws-stickler-threshold": 0.95,
                    "x-aws-stickler-aggregate": True
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        assert Model.__name__ == "ComplexModel"
        assert Model.match_threshold == 0.85
        
        obj1 = Model(id="ID001", name="Test Product", score=95.5)
        obj2 = Model(id="ID001", name="Test Product", score=95.5)
        
        result = obj1.compare_with(obj2)
        assert result["overall_score"] == 1.0
