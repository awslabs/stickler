"""JSON-configuration tests for DateComparator.

Locks in the IDP-style flow: a JSON config dict is enough to build a
``DateComparator`` (directly via the registry, or indirectly via
``StructuredModel.model_from_json``), and the resulting comparator's
``config`` property round-trips back to an equivalent dict.

These tests guard against regressions in any of:

* ``ComparatorRegistry`` registering DateComparator under the canonical name
* DateComparator accepting JSON-friendly types in its constructor
  (string ``range_mode``, numeric-days ``tolerance``, ``None``/``True``/``False`` ``dayfirst``)
* DateComparator's ``config`` property exposing every non-default option
* ``model_from_json`` wiring ``comparator_config`` through to the comparator
* End-to-end scoring through a JSON-built model
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from stickler.comparators.date import DateComparator
from stickler.structured_object_evaluator.models.comparator_registry import (
    create_comparator,
    get_global_registry,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class TestRegistryRegistration:
    """The registry exposes DateComparator under its canonical name."""

    def test_registered_under_canonical_name(self):
        registry = get_global_registry()
        assert registry.is_registered("DateComparator")

    def test_registered_class_is_date_comparator(self):
        registry = get_global_registry()
        assert registry.get("DateComparator") is DateComparator

    def test_listed_among_available_comparators(self):
        assert "DateComparator" in get_global_registry().list_available()


class TestCreateFromJsonConfig:
    """``create_comparator`` accepts JSON-friendly config dicts."""

    def test_no_config_yields_defaults(self):
        cmp = create_comparator("DateComparator")
        assert isinstance(cmp, DateComparator)
        assert cmp.tolerance == timedelta(0)
        assert cmp.dayfirst is None
        assert cmp.allow_partial_year is False
        assert cmp.range_mode == "graded"
        assert cmp.threshold == 1.0

    def test_empty_config_yields_defaults(self):
        cmp = create_comparator("DateComparator", {})
        assert cmp.allow_partial_year is False
        assert cmp.range_mode == "graded"

    def test_full_config_round_trips_to_attributes(self):
        cmp = create_comparator(
            "DateComparator",
            {
                "threshold": 0.7,
                "tolerance": 7,
                "dayfirst": True,
                "allow_partial_year": True,
                "range_mode": "contains",
            },
        )
        assert cmp.threshold == 0.7
        assert cmp.tolerance == timedelta(days=7)
        assert cmp.dayfirst is True
        assert cmp.allow_partial_year is True
        assert cmp.range_mode == "contains"

    def test_tolerance_int_interpreted_as_days(self):
        cmp = create_comparator("DateComparator", {"tolerance": 3})
        assert cmp.tolerance == timedelta(days=3)
        assert cmp.compare("2025-01-01", "2025-01-04") == 1.0
        assert cmp.compare("2025-01-01", "2025-01-05") == 0.0

    def test_tolerance_float_interpreted_as_days(self):
        cmp = create_comparator("DateComparator", {"tolerance": 0.5})
        assert cmp.tolerance == timedelta(days=0.5)

    @pytest.mark.parametrize("mode", ["strict", "reject", "contains", "graded"])
    def test_every_range_mode_accepted(self, mode):
        cmp = create_comparator("DateComparator", {"range_mode": mode})
        assert cmp.range_mode == mode

    def test_invalid_range_mode_raises(self):
        with pytest.raises(ValueError, match="range_mode must be one of"):
            create_comparator("DateComparator", {"range_mode": "fuzzy"})

    def test_invalid_dayfirst_raises(self):
        with pytest.raises(ValueError, match="dayfirst must be"):
            create_comparator("DateComparator", {"dayfirst": "yes"})

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            create_comparator("DateComparator", {"tolerance": -1})


class TestConfigPropertyRoundTrip:
    """DateComparator.config -> create_comparator -> equivalent instance."""

    def test_default_config_round_trip(self):
        original = DateComparator()
        rebuilt = create_comparator("DateComparator", original.config)
        assert rebuilt.dayfirst == original.dayfirst
        assert rebuilt.allow_partial_year == original.allow_partial_year
        assert rebuilt.range_mode == original.range_mode
        assert rebuilt.tolerance == original.tolerance

    def test_full_config_round_trip(self):
        original = DateComparator(
            tolerance=5,
            dayfirst=False,
            allow_partial_year=True,
            range_mode="contains",
        )
        rebuilt = create_comparator("DateComparator", original.config)
        assert rebuilt.tolerance == original.tolerance
        assert rebuilt.dayfirst == original.dayfirst
        assert rebuilt.allow_partial_year == original.allow_partial_year
        assert rebuilt.range_mode == original.range_mode

    def test_tolerance_serializes_as_int_days_when_whole(self):
        cmp = DateComparator(tolerance=timedelta(days=7))
        assert cmp.config["tolerance"] == 7
        assert isinstance(cmp.config["tolerance"], int)

    def test_tolerance_serializes_as_float_when_subday(self):
        cmp = DateComparator(tolerance=timedelta(hours=12))
        assert cmp.config["tolerance"] == 0.5
        assert isinstance(cmp.config["tolerance"], float)

    def test_zero_tolerance_omitted_from_config(self):
        # A fully-default instance emits no config at all (matches the
        # NumericComparator sibling; keeps the schema export clean).
        assert DateComparator().config is None
        # A non-default instance still omits the unset tolerance key.
        cmp = DateComparator(allow_partial_year=True)
        assert "tolerance" not in cmp.config

    def test_config_is_json_serializable(self):
        cmp = DateComparator(
            tolerance=2,
            dayfirst=True,
            allow_partial_year=True,
            range_mode="reject",
        )
        # If anything in the config is a non-JSON type (timedelta, datetime,
        # tuple, etc.), this raises. Important for IDP UI persistence.
        encoded = json.dumps(cmp.config)
        decoded = json.loads(encoded)
        rebuilt = create_comparator("DateComparator", decoded)
        assert rebuilt.tolerance == cmp.tolerance
        assert rebuilt.dayfirst == cmp.dayfirst
        assert rebuilt.allow_partial_year == cmp.allow_partial_year
        assert rebuilt.range_mode == cmp.range_mode


class TestModelFromJsonIntegration:
    """``StructuredModel.model_from_json`` wires DateComparator end-to-end."""

    def test_minimal_model_with_date_field(self):
        config = {
            "model_name": "EventLog",
            "match_threshold": 0.8,
            "fields": {
                "event_date": {
                    "type": "str",
                    "comparator": "DateComparator",
                    "weight": 1.0,
                },
            },
        }
        Cls = StructuredModel.model_from_json(config)
        gt = Cls(event_date="10/24/2016")
        pred = Cls(event_date="Oct 24, 2016")
        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0
        assert result["field_scores"]["event_date"] == 1.0

    def test_comparator_config_propagates(self):
        """Tier 3 partial-year credit only fires when allow_partial_year=True."""
        config = {
            "model_name": "Form",
            "match_threshold": 0.5,
            "fields": {
                "due_date": {
                    "type": "str",
                    "comparator": "DateComparator",
                    "comparator_config": {"allow_partial_year": True},
                    "weight": 1.0,
                },
            },
        }
        Cls = StructuredModel.model_from_json(config)
        gt = Cls(due_date="11/03")
        pred = Cls(due_date="11/03/2012")
        result = gt.compare_with(pred)
        # Without allow_partial_year=True this would be 0.0.
        assert result["field_scores"]["due_date"] == pytest.approx(0.7)

    def test_range_mode_propagates(self):
        """range_mode='contains' should make a single-in-range pair score 1.0."""
        config = {
            "model_name": "Period",
            "match_threshold": 0.5,
            "fields": {
                "service_period": {
                    "type": "str",
                    "comparator": "DateComparator",
                    "comparator_config": {"range_mode": "contains"},
                    "weight": 1.0,
                },
            },
        }
        Cls = StructuredModel.model_from_json(config)
        gt = Cls(service_period="10/01/2016 to 10/31/2016")
        pred = Cls(service_period="10/15/2016")
        result = gt.compare_with(pred)
        assert result["field_scores"]["service_period"] == 1.0

    def test_tolerance_propagates(self):
        config = {
            "model_name": "Shipment",
            "match_threshold": 0.5,
            "fields": {
                "ship_date": {
                    "type": "str",
                    "comparator": "DateComparator",
                    "comparator_config": {"tolerance": 1},
                    "weight": 1.0,
                },
            },
        }
        Cls = StructuredModel.model_from_json(config)
        gt = Cls(ship_date="2025-01-01")
        pred = Cls(ship_date="2025-01-02")
        # Default DateComparator would score this 0.0. With tolerance=1 it's 1.0.
        assert gt.compare_with(pred)["field_scores"]["ship_date"] == 1.0

    def test_idp_invoice_scenario_end_to_end(self):
        """The realistic IDP shape: multiple date fields each tuned differently."""
        config = {
            "model_name": "InvoiceFromJson",
            "match_threshold": 0.8,
            "fields": {
                "invoice_id": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "weight": 1.0,
                },
                "invoice_date": {
                    "type": "str",
                    "comparator": "DateComparator",
                    "comparator_config": {"dayfirst": False},
                    "weight": 2.0,
                },
                "due_date": {
                    "type": "str",
                    "comparator": "DateComparator",
                    "comparator_config": {"allow_partial_year": True},
                    "weight": 1.0,
                },
                "service_period": {
                    "type": "str",
                    "comparator": "DateComparator",
                    "comparator_config": {"range_mode": "contains"},
                    "weight": 1.0,
                },
            },
        }
        Cls = StructuredModel.model_from_json(config)
        gt = Cls(
            invoice_id="INV-001",
            invoice_date="10/24/2016",
            due_date="11/24",
            service_period="10/01/2016 to 10/31/2016",
        )
        pred = Cls(
            invoice_id="INV-001",
            invoice_date="Oct 24, 2016",
            due_date="11/24/2016",
            service_period="10/15/2016",
        )
        result = gt.compare_with(pred)
        assert result["field_scores"]["invoice_id"] == 1.0
        assert result["field_scores"]["invoice_date"] == 1.0
        assert result["field_scores"]["due_date"] == pytest.approx(0.7)
        assert result["field_scores"]["service_period"] == 1.0

    def test_model_from_json_invalid_config_surfaces_error(self):
        config = {
            "model_name": "Bad",
            "match_threshold": 0.5,
            "fields": {
                "d": {
                    "type": "str",
                    "comparator": "DateComparator",
                    "comparator_config": {"range_mode": "not_a_mode"},
                },
            },
        }
        with pytest.raises((ValueError, TypeError)):
            StructuredModel.model_from_json(config)


class TestFromJsonSchemaIntegration:
    """``StructuredModel.from_json_schema`` is the IDP-accelerator entry point.

    These tests guard the ``x-aws-stickler-comparator`` and
    ``x-aws-stickler-comparator-config`` extensions specifically — that's
    the surface the accelerator uses to declare per-field date semantics
    in a standard JSON Schema document.
    """

    def test_minimal_x_aws_extension_works(self):
        schema = {
            "type": "object",
            "properties": {
                "event_date": {
                    "type": "string",
                    "x-aws-stickler-comparator": "DateComparator",
                },
            },
        }
        Cls = StructuredModel.from_json_schema(schema)
        gt = Cls(event_date="10/24/2016")
        pred = Cls(event_date="October 24, 2016")
        assert gt.compare_with(pred)["field_scores"]["event_date"] == 1.0

    def test_comparator_config_extension_propagates(self):
        """`x-aws-stickler-comparator-config` reaches the comparator's ctor."""
        schema = {
            "type": "object",
            "properties": {
                "due_date": {
                    "type": "string",
                    "x-aws-stickler-comparator": "DateComparator",
                    "x-aws-stickler-comparator-config": {
                        "allow_partial_year": True,
                    },
                },
            },
        }
        Cls = StructuredModel.from_json_schema(schema)
        gt = Cls(due_date="11/03")
        pred = Cls(due_date="11/03/2012")
        # 0.7 only fires when allow_partial_year=True propagated.
        assert gt.compare_with(pred)["field_scores"]["due_date"] == pytest.approx(0.7)

    def test_range_mode_extension_propagates(self):
        schema = {
            "type": "object",
            "properties": {
                "service_period": {
                    "type": "string",
                    "x-aws-stickler-comparator": "DateComparator",
                    "x-aws-stickler-comparator-config": {"range_mode": "contains"},
                },
            },
        }
        Cls = StructuredModel.from_json_schema(schema)
        gt = Cls(service_period="10/01/2016 to 10/31/2016")
        pred = Cls(service_period="10/15/2016")
        assert gt.compare_with(pred)["field_scores"]["service_period"] == 1.0

    def test_tolerance_extension_propagates(self):
        schema = {
            "type": "object",
            "properties": {
                "ship_date": {
                    "type": "string",
                    "x-aws-stickler-comparator": "DateComparator",
                    "x-aws-stickler-comparator-config": {"tolerance": 1},
                },
            },
        }
        Cls = StructuredModel.from_json_schema(schema)
        assert (
            Cls(ship_date="2025-01-01")
            .compare_with(Cls(ship_date="2025-01-02"))
            ["field_scores"]["ship_date"]
            == 1.0
        )

    def test_all_options_together_via_x_aws_extensions(self):
        schema = {
            "type": "object",
            "properties": {
                "due_date": {
                    "type": "string",
                    "x-aws-stickler-comparator": "DateComparator",
                    "x-aws-stickler-comparator-config": {
                        "tolerance": 0,
                        "dayfirst": False,
                        "allow_partial_year": True,
                        "range_mode": "graded",
                    },
                    "x-aws-stickler-threshold": 0.65,
                    "x-aws-stickler-weight": 3.0,
                },
            },
        }
        Cls = StructuredModel.from_json_schema(schema)
        # The 0.7 partial-year score crosses 0.65 threshold but not 0.7,
        # so this is also a clean threshold-propagation check.
        gt = Cls(due_date="11/03")
        pred = Cls(due_date="11/03/2012")
        assert gt.compare_with(pred)["field_scores"]["due_date"] == pytest.approx(0.7)

    def test_idp_invoice_scenario_from_json_schema(self):
        """End-to-end: a multi-field IDP-style schema produces correct scores."""
        schema = {
            "type": "object",
            "x-aws-stickler-model-name": "Invoice",
            "x-aws-stickler-match-threshold": 0.8,
            "properties": {
                "invoice_id": {"type": "string"},
                "invoice_date": {
                    "type": "string",
                    "x-aws-stickler-comparator": "DateComparator",
                    "x-aws-stickler-comparator-config": {"dayfirst": False},
                    "x-aws-stickler-weight": 2.0,
                },
                "due_date": {
                    "type": "string",
                    "x-aws-stickler-comparator": "DateComparator",
                    "x-aws-stickler-comparator-config": {"allow_partial_year": True},
                    "x-aws-stickler-weight": 1.0,
                },
                "service_period": {
                    "type": "string",
                    "x-aws-stickler-comparator": "DateComparator",
                    "x-aws-stickler-comparator-config": {"range_mode": "contains"},
                },
            },
            "required": ["invoice_id"],
        }
        Cls = StructuredModel.from_json_schema(schema)
        assert Cls.__name__ == "Invoice"
        gt = Cls(
            invoice_id="INV-001",
            invoice_date="10/24/2016",
            due_date="11/24",
            service_period="10/01/2016 to 10/31/2016",
        )
        pred = Cls(
            invoice_id="INV-001",
            invoice_date="Oct 24, 2016",
            due_date="11/24/2016",
            service_period="10/15/2016",
        )
        result = gt.compare_with(pred)
        assert result["field_scores"]["invoice_id"] == 1.0
        assert result["field_scores"]["invoice_date"] == 1.0
        assert result["field_scores"]["due_date"] == pytest.approx(0.7)
        assert result["field_scores"]["service_period"] == 1.0

    def test_invalid_x_aws_config_raises_with_field_path(self):
        """A bad comparator config should fail loudly at schema-parse time."""
        schema = {
            "type": "object",
            "properties": {
                "d": {
                    "type": "string",
                    "x-aws-stickler-comparator": "DateComparator",
                    "x-aws-stickler-comparator-config": {"range_mode": "nope"},
                },
            },
        }
        with pytest.raises(ValueError) as exc_info:
            StructuredModel.from_json_schema(schema)
        msg = str(exc_info.value)
        assert "DateComparator" in msg
        assert "range_mode" in msg
        # The error should help the user pinpoint which field failed.
        assert "'d'" in msg

    def test_json_schema_round_trip_via_string(self):
        """A schema authored as a JSON string (the realistic IDP shape) works."""
        schema_json = json.dumps(
            {
                "type": "object",
                "properties": {
                    "due_date": {
                        "type": "string",
                        "x-aws-stickler-comparator": "DateComparator",
                        "x-aws-stickler-comparator-config": {
                            "allow_partial_year": True,
                            "range_mode": "graded",
                            "tolerance": 0,
                        },
                    },
                },
            }
        )
        schema = json.loads(schema_json)
        Cls = StructuredModel.from_json_schema(schema)
        gt = Cls(due_date="11/03")
        pred = Cls(due_date="11/03/2012")
        assert gt.compare_with(pred)["field_scores"]["due_date"] == pytest.approx(0.7)
