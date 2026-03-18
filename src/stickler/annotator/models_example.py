"""Example StructuredModel subclasses for testing the Pydantic import path.

These models demonstrate how to define annotation schemas as Python classes
rather than JSON Schema files. Import path for the annotator tool:

    stickler.annotator.models_example.FccInvoiceModel
"""

from __future__ import annotations

from typing import List, Optional

from stickler.structured_object_evaluator import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator


class LineItemModel(StructuredModel):
    """A single airtime spot line item on an FCC political advertising invoice."""

    air_date: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, description="Date the spot aired (MM/DD/YYYY)"
    )
    program: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, description="Program or daypart name"
    )
    spot_length: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.95, description="Spot length (e.g. :30)"
    )
    gross_rate: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, description="Gross rate for this spot"
    )
    net_rate: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, description="Net rate after commissions"
    )


class FccInvoiceModel(StructuredModel):
    """FCC Political Advertising Invoice — Pydantic import path example.

    Use in the annotator tool as:
        stickler.annotator.models_example.FccInvoiceModel
    """

    station_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, description="Broadcasting station name"
    )
    invoice_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.95, description="Invoice ID number"
    )
    invoice_date: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, description="Invoice date (MM-DD-YYYY)"
    )
    contract_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.95, description="Contract ID number"
    )
    client_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.95, description="Client ID number"
    )
    advertiser_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, description="Political advertiser or candidate name"
    )
    bill_cycle: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, description="Billing cycle (e.g. 02/16)"
    )
    gross_advertising_fee: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, description="Gross advertising fee amount"
    )
    net_advertising_fee: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, description="Net advertising fee after commissions"
    )
    total_this_invoice: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, description="Total amount due on this invoice"
    )
    # Optional fields
    campaign_id: Optional[str] = ComparableField(
        default=None, comparator=LevenshteinComparator(), threshold=0.95, description="Campaign ID number"
    )
    estimate_id: Optional[str] = ComparableField(
        default=None, comparator=LevenshteinComparator(), threshold=0.95, description="Estimate ID number"
    )
    po_number: Optional[str] = ComparableField(
        default=None, comparator=LevenshteinComparator(), threshold=0.95, description="Purchase order number"
    )
    agency_commission: Optional[str] = ComparableField(
        default=None, comparator=LevenshteinComparator(), threshold=0.9, description="Agency commission amount"
    )
    rep_firm_commission: Optional[str] = ComparableField(
        default=None, comparator=LevenshteinComparator(), threshold=0.9, description="Rep firm commission amount"
    )
    line_items: List[LineItemModel] = ComparableField(
        default=None, comparator=LevenshteinComparator(), threshold=0.8, description="Individual spot/airtime line items"
    )
