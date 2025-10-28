"""Tests for StructuredModel.from_json_schema() method.

This module tests the JSON Schema to StructuredModel conversion functionality.
"""

import pytest
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class TestFromJsonSchemaBasic:
    """Test basic JSON Schema conversion functionality."""

    def test_basic_schema_conversion(self):
        """Test converting a basic JSON Schema to StructuredModel."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        assert Model.__name__ == "DynamicModel"
        assert "name" in Model.model_fields
        assert "age" in Model.model_fields
        
        instance = Model(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30

    def test_all_primitive_types(self):
        """Test JSON Schema with all primitive types."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "price": {"type": "number"},
                "active": {"type": "boolean"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(name="Test", count=5, price=9.99, active=True)
        assert instance.name == "Test"
        assert instance.count == 5
        assert instance.price == 9.99
        assert instance.active is True

    def test_with_default_values(self):
        """Test JSON Schema with default values."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "default": 0}
            },
            "required": ["name"]
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(name="Test")
        assert instance.name == "Test"
        assert instance.count == 0


class TestFromJsonSchemaModelConfiguration:
    """Test model-level configuration extraction."""

    def test_custom_model_name(self):
        """Test x-aws-stickler-model-name extension."""
        schema = {
            "type": "object",
            "x-aws-stickler-model-name": "Product",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        assert Model.__name__ == "Product"

    def test_custom_match_threshold(self):
        """Test x-aws-stickler-match-threshold extension."""
        schema = {
            "type": "object",
            "x-aws-stickler-match-threshold": 0.8,
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        assert Model.match_threshold == 0.8

    def test_default_model_name(self):
        """Test default model name when not specified."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        assert Model.__name__ == "DynamicModel"

    def test_default_match_threshold(self):
        """Test default match threshold when not specified."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        assert Model.match_threshold == 0.7


class TestFromJsonSchemaValidation:
    """Test JSON Schema validation."""

    def test_invalid_schema_raises_error(self):
        """Test that invalid JSON Schema raises ValueError."""
        invalid_schema = {
            "type": "invalid_type"
        }
        
        with pytest.raises(ValueError, match="Invalid JSON Schema"):
            StructuredModel.from_json_schema(invalid_schema)

    def test_missing_properties_raises_error(self):
        """Test that schema without properties raises error."""
        schema = {
            "type": "object"
        }
        
        with pytest.raises(ValueError, match="must contain 'properties'"):
            StructuredModel.from_json_schema(schema)

    def test_invalid_model_name_raises_error(self):
        """Test that invalid model name raises error."""
        schema = {
            "type": "object",
            "x-aws-stickler-model-name": "123Invalid",  # Can't start with number
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        with pytest.raises(ValueError, match="valid Python identifier"):
            StructuredModel.from_json_schema(schema)

    def test_invalid_match_threshold_type_raises_error(self):
        """Test that invalid match threshold type raises error."""
        schema = {
            "type": "object",
            "x-aws-stickler-match-threshold": "invalid",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        with pytest.raises(ValueError, match="must be a number"):
            StructuredModel.from_json_schema(schema)

    def test_match_threshold_out_of_range_raises_error(self):
        """Test that match threshold out of range raises error."""
        schema = {
            "type": "object",
            "x-aws-stickler-match-threshold": 1.5,
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            StructuredModel.from_json_schema(schema)


class TestFromJsonSchemaComparison:
    """Test comparison functionality with JSON Schema models."""

    def test_basic_comparison(self):
        """Test basic comparison with JSON Schema model."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(name="Alice", age=30)
        obj2 = Model(name="Alice", age=30)
        
        score = obj1.compare(obj2)
        assert score == 1.0

    def test_comparison_with_extensions(self):
        """Test comparison with x-aws-stickler-* extensions."""
        schema = {
            "type": "object",
            "x-aws-stickler-match-threshold": 0.8,
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                    "x-aws-stickler-threshold": 0.9,
                    "x-aws-stickler-weight": 2.0
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(name="Widget")
        obj2 = Model(name="Widget")
        
        result = obj1.compare_with(obj2)
        assert result["overall_score"] == 1.0

    def test_compare_with_method(self):
        """Test compare_with method with JSON Schema model."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(name="Product", price=29.99)
        obj2 = Model(name="Product", price=29.99)
        
        result = obj1.compare_with(obj2)
        assert "overall_score" in result
        assert "field_scores" in result
        assert result["overall_score"] == 1.0

    def test_complex_nested_schema_with_enums(self):
        """Test complex real-world schema with nested objects, arrays, and enums."""
        schema = {
            "type": "object",
            "required": ["header", "items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["sn", "type", "brand", "reference", "productSpecsJson", "unit", "quantity"],
                        "properties": {
                            "sn": {"type": "number"},
                            "type": {
                                "enum": ["LED Luminaire", "Power Supply", "Control System", "Mounting Bracket", 
                                        "LED Driver", "Cable Assembly", "Junction Box", "Sensor Module"],
                                "type": "string"
                            },
                            "unit": {
                                "enum": ["pcs", "set", "unit"],
                                "type": "string"
                            },
                            "brand": {
                                "enum": ["Philips", "Osram", "Schneider Electric", "Siemens", 
                                        "ABB", "Lutron", "Legrand", "Mean Well"],
                                "type": "string"
                            },
                            "quantity": {"type": "number"},
                            "reference": {"type": "string"},
                            "productSpecsJson": {
                                "type": "object",
                                "required": ["power", "physical", "ratings"],
                                "properties": {
                                    "power": {
                                        "type": "object",
                                        "required": ["watts", "inputVoltage"],
                                        "properties": {
                                            "watts": {"type": "number"},
                                            "inputVoltage": {
                                                "enum": ["100-240v", "100-277v", "220-240v"],
                                                "type": "string"
                                            }
                                        }
                                    },
                                    "ratings": {
                                        "type": "object",
                                        "required": ["protection"],
                                        "properties": {
                                            "protection": {
                                                "enum": ["ip65", "ip66", "ip67"],
                                                "type": "string"
                                            }
                                        }
                                    },
                                    "physical": {
                                        "type": "object",
                                        "required": ["material", "mountingType"],
                                        "properties": {
                                            "material": {
                                                "enum": ["aluminum", "steel", "plastic"],
                                                "type": "string"
                                            },
                                            "mountingType": {
                                                "enum": ["surface", "recessed", "track"],
                                                "type": "string"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "header": {
                    "type": "object",
                    "required": ["from", "to", "date", "projectName"],
                    "properties": {
                        "to": {
                            "type": "object",
                            "required": ["companyName", "reference", "location", "attention", "tel", "email", "vatNumber"],
                            "properties": {
                                "tel": {"type": "string"},
                                "email": {"type": "string"},
                                "location": {"type": "string"},
                                "attention": {"type": "string"},
                                "reference": {"type": "string"},
                                "vatNumber": {"type": "string"},
                                "companyName": {"type": "string"}
                            }
                        },
                        "date": {"type": "string"},
                        "from": {
                            "type": "object",
                            "required": ["companyName", "reference", "location", "tel", "email", "vatNumber"],
                            "properties": {
                                "tel": {"type": "string"},
                                "email": {"type": "string"},
                                "location": {"type": "string"},
                                "reference": {"type": "string"},
                                "vatNumber": {"type": "string"},
                                "companyName": {"type": "string"}
                            }
                        },
                        "projectName": {"type": "string"}
                    }
                }
            }
        }
        
        # Create model from complex schema
        Model = StructuredModel.from_json_schema(schema)
        
        # Verify model was created
        assert Model.__name__ == "DynamicModel"
        assert "header" in Model.model_fields
        assert "items" in Model.model_fields
        
        # Create instance with sample data
        instance = Model(
            header={
                "from": {
                    "companyName": "Acme Corp",
                    "reference": "REF001",
                    "location": "New York",
                    "tel": "555-1234",
                    "email": "contact@acme.com",
                    "vatNumber": "VAT123"
                },
                "to": {
                    "companyName": "Beta Inc",
                    "reference": "REF002",
                    "location": "Boston",
                    "attention": "John Doe",
                    "tel": "555-5678",
                    "email": "john@beta.com",
                    "vatNumber": "VAT456"
                },
                "date": "2024-01-15",
                "projectName": "LED Installation"
            },
            items=[
                {
                    "sn": 1,
                    "type": "LED Luminaire",
                    "brand": "Philips",
                    "reference": "LED-001",
                    "unit": "pcs",
                    "quantity": 10,
                    "productSpecsJson": {
                        "power": {
                            "watts": 50,
                            "inputVoltage": "100-240v"
                        },
                        "physical": {
                            "material": "aluminum",
                            "mountingType": "surface"
                        },
                        "ratings": {
                            "protection": "ip65"
                        }
                    }
                }
            ]
        )
        
        # Verify nested structure
        # Note: 'from' is a Python keyword, but Pydantic handles it as a regular field
        from_data = getattr(instance.header, 'from')
        assert from_data.companyName == "Acme Corp"
        assert instance.header.to.companyName == "Beta Inc"
        assert instance.header.projectName == "LED Installation"
        assert len(instance.items) == 1
        assert instance.items[0].sn == 1
        assert instance.items[0].type == "LED Luminaire"
        assert instance.items[0].productSpecsJson.power.watts == 50
        assert instance.items[0].productSpecsJson.physical.material == "aluminum"
        
        # Test comparison
        instance2 = Model(
            header={
                "from": {
                    "companyName": "Acme Corp",
                    "reference": "REF001",
                    "location": "New York",
                    "tel": "555-1234",
                    "email": "contact@acme.com",
                    "vatNumber": "VAT123"
                },
                "to": {
                    "companyName": "Beta Inc",
                    "reference": "REF002",
                    "location": "Boston",
                    "attention": "John Doe",
                    "tel": "555-5678",
                    "email": "john@beta.com",
                    "vatNumber": "VAT456"
                },
                "date": "2024-01-15",
                "projectName": "LED Installation"
            },
            items=[
                {
                    "sn": 1,
                    "type": "LED Luminaire",
                    "brand": "Philips",
                    "reference": "LED-001",
                    "unit": "pcs",
                    "quantity": 10,
                    "productSpecsJson": {
                        "power": {
                            "watts": 50,
                            "inputVoltage": "100-240v"
                        },
                        "physical": {
                            "material": "aluminum",
                            "mountingType": "surface"
                        },
                        "ratings": {
                            "protection": "ip65"
                        }
                    }
                }
            ]
        )
        
        # Compare identical instances
        score = instance.compare(instance2)
        assert score == 1.0
        
        result = instance.compare_with(instance2)
        assert result["overall_score"] == 1.0


class TestFromJsonSchemaNestedObjects:
    """Test deeply nested object support with JSON Schema."""

    def test_single_level_nesting(self):
        """Test schema with one level of nesting."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zipcode": {"type": "string"}
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            name="John Doe",
            address={"street": "123 Main St", "city": "Boston", "zipcode": "02101"}
        )
        
        assert instance.name == "John Doe"
        assert isinstance(instance.address, StructuredModel)
        assert instance.address.street == "123 Main St"
        assert instance.address.city == "Boston"

    def test_two_level_nesting(self):
        """Test schema with two levels of nesting."""
        schema = {
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "location": {
                    "type": "object",
                    "properties": {
                        "office": {
                            "type": "object",
                            "properties": {
                                "building": {"type": "string"},
                                "floor": {"type": "integer"}
                            }
                        },
                        "city": {"type": "string"}
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            company="Acme Corp",
            location={
                "office": {"building": "Tower A", "floor": 5},
                "city": "Seattle"
            }
        )
        
        assert instance.company == "Acme Corp"
        assert isinstance(instance.location, StructuredModel)
        assert isinstance(instance.location.office, StructuredModel)
        assert instance.location.office.building == "Tower A"
        assert instance.location.office.floor == 5

    def test_three_level_nesting(self):
        """Test schema with three levels of nesting."""
        schema = {
            "type": "object",
            "properties": {
                "organization": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "department": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "team": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "size": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            organization={
                "name": "Tech Corp",
                "department": {
                    "name": "Engineering",
                    "team": {
                        "name": "Backend",
                        "size": 8
                    }
                }
            }
        )
        
        assert isinstance(instance.organization, StructuredModel)
        assert isinstance(instance.organization.department, StructuredModel)
        assert isinstance(instance.organization.department.team, StructuredModel)
        assert instance.organization.department.team.name == "Backend"
        assert instance.organization.department.team.size == 8

    def test_nested_object_comparison(self):
        """Test comparison with nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "contact": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "phone": {"type": "string"}
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(
            name="Alice",
            contact={"email": "alice@example.com", "phone": "555-1234"}
        )
        obj2 = Model(
            name="Alice",
            contact={"email": "alice@example.com", "phone": "555-1234"}
        )
        
        score = obj1.compare(obj2)
        assert score == 1.0

    def test_nested_arrays_of_objects(self):
        """Test nested arrays containing objects."""
        schema = {
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "departments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "employees": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "role": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            company="Tech Inc",
            departments=[
                {
                    "name": "Engineering",
                    "employees": [
                        {"name": "Alice", "role": "Engineer"},
                        {"name": "Bob", "role": "Manager"}
                    ]
                }
            ]
        )
        
        assert instance.company == "Tech Inc"
        assert len(instance.departments) == 1
        assert isinstance(instance.departments[0], StructuredModel)
        assert len(instance.departments[0].employees) == 2
        assert isinstance(instance.departments[0].employees[0], StructuredModel)


class TestFromJsonSchemaArrays:
    """Test JSON Schema with array types."""

    def test_array_of_primitives_strings(self):
        """Test schema with array of string primitives."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(name="Product", tags=["electronics", "gadget", "new"])
        assert instance.name == "Product"
        assert isinstance(instance.tags, list)
        assert len(instance.tags) == 3
        assert instance.tags[0] == "electronics"

    def test_array_of_primitives_numbers(self):
        """Test schema with array of number primitives."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "scores": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(name="Test", scores=[95.5, 87.3, 92.1])
        assert instance.name == "Test"
        assert isinstance(instance.scores, list)
        assert len(instance.scores) == 3
        assert instance.scores[0] == 95.5

    def test_array_of_primitives_integers(self):
        """Test schema with array of integer primitives."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "quantities": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(name="Inventory", quantities=[10, 25, 5])
        assert instance.name == "Inventory"
        assert isinstance(instance.quantities, list)
        assert all(isinstance(q, int) for q in instance.quantities)

    def test_array_of_objects(self):
        """Test schema with array of objects."""
        schema = {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "price": {"type": "number"}
                        },
                        "required": ["name"]
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            order_id="ORD-001",
            items=[
                {"name": "Widget", "quantity": 2, "price": 10.00},
                {"name": "Gadget", "quantity": 1, "price": 25.00}
            ]
        )
        
        assert instance.order_id == "ORD-001"
        assert len(instance.items) == 2
        assert all(isinstance(item, StructuredModel) for item in instance.items)
        assert instance.items[0].name == "Widget"
        assert instance.items[0].quantity == 2
        assert instance.items[1].name == "Gadget"

    def test_array_comparison_primitives(self):
        """Test comparison with arrays of primitives."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(name="Product", tags=["electronics", "gadget"])
        obj2 = Model(name="Product", tags=["electronics", "gadget"])
        
        score = obj1.compare(obj2)
        assert score == 1.0

    def test_array_comparison_objects(self):
        """Test comparison with arrays of objects."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "integer"}
                        }
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        obj1 = Model(items=[{"name": "A", "value": 1}, {"name": "B", "value": 2}])
        obj2 = Model(items=[{"name": "A", "value": 1}, {"name": "B", "value": 2}])
        
        score = obj1.compare(obj2)
        assert score == 1.0


class TestFromJsonSchemaWithReferences:
    """Test JSON Schema with $ref and definitions."""

    def test_schema_with_definitions(self):
        """Test schema using definitions with $ref."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "#/definitions/Address"}
            },
            "definitions": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zipcode": {"type": "string"}
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            name="John Doe",
            address={"street": "123 Main St", "city": "Boston", "zipcode": "02101"}
        )
        
        assert instance.name == "John Doe"
        assert isinstance(instance.address, StructuredModel)
        assert instance.address.street == "123 Main St"

    def test_schema_with_defs(self):
        """Test schema using $defs (JSON Schema draft 2019-09+)."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "contact": {"$ref": "#/$defs/Contact"}
            },
            "$defs": {
                "Contact": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "phone": {"type": "string"}
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            name="Alice",
            contact={"email": "alice@example.com", "phone": "555-1234"}
        )
        
        assert instance.name == "Alice"
        assert isinstance(instance.contact, StructuredModel)
        assert instance.contact.email == "alice@example.com"

    def test_schema_with_multiple_refs(self):
        """Test schema with multiple $ref references."""
        schema = {
            "type": "object",
            "properties": {
                "person": {"$ref": "#/definitions/Person"},
                "company": {"$ref": "#/definitions/Company"}
            },
            "definitions": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                },
                "Company": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "industry": {"type": "string"}
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            person={"name": "Alice", "age": 30},
            company={"name": "Tech Corp", "industry": "Software"}
        )
        
        assert isinstance(instance.person, StructuredModel)
        assert isinstance(instance.company, StructuredModel)
        assert instance.person.name == "Alice"
        assert instance.company.name == "Tech Corp"

    def test_schema_with_nested_refs(self):
        """Test schema with nested $ref references."""
        schema = {
            "type": "object",
            "properties": {
                "employee": {"$ref": "#/definitions/Employee"}
            },
            "definitions": {
                "Employee": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"$ref": "#/definitions/Address"}
                    }
                },
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            employee={
                "name": "Bob",
                "address": {"street": "456 Oak Ave", "city": "Portland"}
            }
        )
        
        assert isinstance(instance.employee, StructuredModel)
        assert isinstance(instance.employee.address, StructuredModel)
        assert instance.employee.address.city == "Portland"

    def test_schema_with_array_of_refs(self):
        """Test schema with array items using $ref."""
        schema = {
            "type": "object",
            "properties": {
                "employees": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/Employee"}
                }
            },
            "definitions": {
                "Employee": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"}
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            employees=[
                {"name": "Alice", "role": "Engineer"},
                {"name": "Bob", "role": "Manager"}
            ]
        )
        
        assert len(instance.employees) == 2
        assert all(isinstance(emp, StructuredModel) for emp in instance.employees)
        assert instance.employees[0].name == "Alice"
        assert instance.employees[1].role == "Manager"

    def test_refs_with_extensions(self):
        """Test $ref with x-aws-stickler-* extensions."""
        schema = {
            "type": "object",
            "x-aws-stickler-model-name": "Company",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-weight": 2.0
                },
                "address": {"$ref": "#/definitions/Address"}
            },
            "definitions": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "x-aws-stickler-comparator": "LevenshteinComparator",
                            "x-aws-stickler-threshold": 0.9
                        }
                    }
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        assert Model.__name__ == "Company"
        instance = Model(name="Acme", address={"city": "Seattle"})
        assert isinstance(instance.address, StructuredModel)
        
        # Test comparison works with extensions
        instance2 = Model(name="Acme", address={"city": "Seattle"})
        score = instance.compare(instance2)
        assert score == 1.0


class TestFromJsonSchemaExtensions:
    """Test all x-aws-stickler-* extensions comprehensively."""

    def test_all_field_extensions(self):
        """Test all x-aws-stickler-* field extensions together."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                    "x-aws-stickler-threshold": 0.85,
                    "x-aws-stickler-weight": 2.5,
                    "x-aws-stickler-clip-under-threshold": True,
                    "x-aws-stickler-aggregate": True
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        instance = Model(name="Test")
        assert instance.name == "Test"

    def test_extension_comparator_numeric(self):
        """Test x-aws-stickler-comparator with NumericComparator."""
        schema = {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number",
                    "x-aws-stickler-comparator": "NumericComparator",
                    "x-aws-stickler-threshold": 0.95
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        obj1 = Model(price=100.0)
        obj2 = Model(price=100.0)
        
        score = obj1.compare(obj2)
        assert score == 1.0

    def test_extension_comparator_exact(self):
        """Test x-aws-stickler-comparator with ExactComparator."""
        schema = {
            "type": "object",
            "properties": {
                "active": {
                    "type": "boolean",
                    "x-aws-stickler-comparator": "ExactComparator"
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        obj1 = Model(active=True)
        obj2 = Model(active=True)
        
        score = obj1.compare(obj2)
        assert score == 1.0

    def test_extension_weight(self):
        """Test x-aws-stickler-weight affects comparison."""
        schema = {
            "type": "object",
            "properties": {
                "important": {
                    "type": "string",
                    "x-aws-stickler-weight": 5.0
                },
                "minor": {
                    "type": "string",
                    "x-aws-stickler-weight": 0.5
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        obj1 = Model(important="Critical", minor="Detail")
        obj2 = Model(important="Critical", minor="Detail")
        
        result = obj1.compare_with(obj2)
        assert result["overall_score"] == 1.0

    def test_extension_clip_under_threshold(self):
        """Test x-aws-stickler-clip-under-threshold behavior."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "x-aws-stickler-threshold": 0.9,
                    "x-aws-stickler-clip-under-threshold": True
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        obj1 = Model(name="Test")
        obj2 = Model(name="Test")
        
        score = obj1.compare(obj2)
        assert score == 1.0

    def test_extension_aggregate(self):
        """Test x-aws-stickler-aggregate for confusion matrix aggregation."""
        schema = {
            "type": "object",
            "properties": {
                "field1": {
                    "type": "string",
                    "x-aws-stickler-aggregate": True
                },
                "field2": {
                    "type": "string",
                    "x-aws-stickler-aggregate": False
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        obj1 = Model(field1="A", field2="B")
        obj2 = Model(field1="A", field2="B")
        
        result = obj1.compare_with(obj2)
        assert "overall_score" in result

    def test_multiple_comparator_types(self):
        """Test schema with multiple different comparator types."""
        schema = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator"
                },
                "number": {
                    "type": "number",
                    "x-aws-stickler-comparator": "NumericComparator"
                },
                "flag": {
                    "type": "boolean",
                    "x-aws-stickler-comparator": "ExactComparator"
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        obj1 = Model(text="Hello", number=42.0, flag=True)
        obj2 = Model(text="Hello", number=42.0, flag=True)
        
        score = obj1.compare(obj2)
        assert score == 1.0


class TestFromJsonSchemaEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_properties(self):
        """Test schema with empty properties object raises error."""
        schema = {
            "type": "object",
            "properties": {}
        }
        
        # Empty properties should raise an error since models need at least one field
        with pytest.raises(ValueError, match="at least one field"):
            StructuredModel.from_json_schema(schema)

    def test_optional_fields_only(self):
        """Test schema with all optional fields (no required)."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        instance = Model()
        assert instance.name is None
        assert instance.age is None

    def test_all_required_fields(self):
        """Test schema with all fields required."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        
        Model = StructuredModel.from_json_schema(schema)
        
        # Should raise validation error if required fields missing
        with pytest.raises(Exception):
            Model()

    def test_deeply_nested_structure(self):
        """Test deeply nested object structure (4+ levels)."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {
                                        "level4": {
                                            "type": "object",
                                            "properties": {
                                                "value": {"type": "string"}
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
        
        Model = StructuredModel.from_json_schema(schema)
        
        instance = Model(
            level1={
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        )
        
        assert instance.level1.level2.level3.level4.value == "deep"

    def test_empty_array(self):
        """Test schema with empty array."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        Model = StructuredModel.from_json_schema(schema)
        instance = Model(items=[])
        assert instance.items == []

    def test_mixed_required_optional_nested(self):
        """Test nested objects with mixed required/optional fields."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "email": {"type": "string"}
                    },
                    "required": ["name"]
                }
            },
            "required": ["person"]
        }
        
        Model = StructuredModel.from_json_schema(schema)
        instance = Model(person={"name": "Alice"})
        assert instance.person.name == "Alice"
        assert instance.person.age is None
