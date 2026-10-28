"""Shared corpus for the Strands Evals contract, snapshot and end-to-end tests.

Models are the shapes users pass to a Strands agent as ``structured_output_model``,
and cases are the (expected, actual) pairs an eval suite feeds them. They live at
module level so ``StructuredOutputSimilarity.to_dict`` can name them by dotted path.

Every case is in the golden snapshot. The two ``None``-in-a-list shapes were
held out of it while the change that moves them (#399) was still open; that
change is merged, so the snapshot now records their current values and any
further movement of them turns the snapshot test red. ``CorpusCase.snapshot``
is kept for the next time a shape has to be held out; no case sets it to
``False`` today, so ``SNAPSHOT_CORPUS`` is currently every case.
"""

from __future__ import annotations

import datetime
import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field

#: The `match_threshold` StructuredOutputSimilarity defaults to in 1.6.0.
STRANDS_DEFAULT_THRESHOLD = 0.7


# --- models -----------------------------------------------------------------


class Status(str, enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class Contact(BaseModel):
    """Flat scalars of every common type, a description, an alias and defaults."""

    name: str = Field(description="Full legal name as printed on the document")
    age: int
    height_m: float
    is_active: bool
    birth_date: datetime.date
    updated_at: datetime.datetime
    status: Status
    tier: Literal["gold", "silver", "bronze"]
    nickname: Optional[str] = None
    email: str = Field(alias="emailAddress", description="Primary email")
    country: str = "US"


class Address(BaseModel):
    street: str
    city: str
    postal_code: str


class LineItem(BaseModel):
    sku: str
    description: str
    quantity: int
    unit_price: float


class Invoice(BaseModel):
    """Nested model, Optional nested, List[str], List[Model], Optional[List[Model]]."""

    invoice_number: str
    issue_date: datetime.date
    total: float
    currency: Literal["USD", "EUR"] = "USD"
    vendor: Address
    ship_to: Optional[Address] = None
    tags: List[str] = Field(default_factory=list)
    line_items: List[LineItem] = Field(default_factory=list)
    credits: Optional[List[LineItem]] = None


class Lead(BaseModel):
    """A sparse schema: every field optional, most documents populate a few."""

    company: Optional[str] = None
    contact_name: Optional[str] = None
    employees: Optional[int] = None
    annual_revenue: Optional[float] = None
    founded: Optional[datetime.date] = None
    is_public: Optional[bool] = None
    status: Optional[Status] = None
    keywords: Optional[List[str]] = None
    hq: Optional[Address] = None
    notes: Optional[str] = Field(default=None, description="Free text")


class Shipment(BaseModel):
    tracking_id: str
    carrier: str
    items: List[LineItem] = Field(default_factory=list)


class Order(BaseModel):
    """A list of models whose items carry their own list of models."""

    order_id: str
    placed_on: datetime.date
    shipments: List[Shipment] = Field(default_factory=list)


class Tagged(BaseModel):
    """Edge shape: a None element inside a list (#386, #399 changes this)."""

    label: str
    values: List[Optional[str]] = Field(default_factory=list)


MODELS: List[Type[BaseModel]] = [Contact, Invoice, Lead, Order, Tagged]


# --- documents --------------------------------------------------------------


def contact(**overrides: Any) -> Contact:
    base: Dict[str, Any] = dict(
        name="Ada Lovelace",
        age=36,
        height_m=1.65,
        is_active=True,
        birth_date=datetime.date(1815, 12, 10),
        updated_at=datetime.datetime(2024, 3, 1, 9, 30, 0),
        status=Status.ACTIVE,
        tier="gold",
        nickname="Countess",
        emailAddress="ada@example.org",
        country="UK",
    )
    if "email" in overrides:
        overrides["emailAddress"] = overrides.pop("email")
    base.update(overrides)
    return Contact(**base)


def _lines() -> List[LineItem]:
    return [
        LineItem(sku="A-100", description="Steel bolts", quantity=40, unit_price=0.25),
        LineItem(sku="B-200", description="Hex nuts", quantity=40, unit_price=0.10),
        LineItem(sku="C-300", description="Washers", quantity=80, unit_price=0.05),
    ]


def invoice(**overrides: Any) -> Invoice:
    base: Dict[str, Any] = dict(
        invoice_number="INV-2024-0042",
        issue_date=datetime.date(2024, 5, 17),
        total=18.0,
        currency="USD",
        vendor=Address(street="1 Foundry Way", city="Pittsburgh", postal_code="15201"),
        ship_to=Address(street="9 Dock Rd", city="Erie", postal_code="16501"),
        tags=["hardware", "net30"],
        line_items=_lines(),
        credits=[
            LineItem(sku="R-1", description="Return", quantity=1, unit_price=-2.0)
        ],
    )
    base.update(overrides)
    return Invoice(**base)


def lead(**overrides: Any) -> Lead:
    """A sparse document: two of ten fields populated."""
    base: Dict[str, Any] = dict(company="Globex", employees=250)
    base.update(overrides)
    return Lead(**base)


def full_lead() -> Lead:
    return Lead(
        company="Initech",
        contact_name="Bill Lumbergh",
        employees=1200,
        annual_revenue=4.5e7,
        founded=datetime.date(1989, 4, 1),
        is_public=False,
        status=Status.ACTIVE,
        keywords=["software", "consulting"],
        hq=Address(street="4120 Freidrich Ln", city="Austin", postal_code="78744"),
        notes="Prefers email follow-ups",
    )


def order(**overrides: Any) -> Order:
    base: Dict[str, Any] = dict(
        order_id="ORD-7",
        placed_on=datetime.date(2024, 6, 2),
        shipments=[
            Shipment(tracking_id="1Z999", carrier="UPS", items=_lines()[:2]),
            Shipment(tracking_id="9400111", carrier="USPS", items=_lines()[2:]),
        ],
    )
    base.update(overrides)
    return Order(**base)


# --- cases ------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusCase:
    id: str
    model: Type[BaseModel]
    expected: BaseModel
    actual: BaseModel
    snapshot: bool = True


def _cases() -> List[CorpusCase]:
    lines = _lines()
    wrong_line = LineItem(sku="Z-999", description="Rivets", quantity=7, unit_price=9.0)
    return [
        # Contact: flat scalars.
        CorpusCase("contact/identical", Contact, contact(), contact()),
        CorpusCase(
            "contact/one_wrong_str",
            Contact,
            contact(),
            contact(email="ada@wrong.example"),
        ),
        CorpusCase(
            "contact/wrong_int_and_enum",
            Contact,
            contact(),
            contact(age=63, status=Status.CLOSED),
        ),
        CorpusCase(
            "contact/wrong_date_bool_literal",
            Contact,
            contact(),
            contact(
                birth_date=datetime.date(1816, 1, 1), is_active=False, tier="bronze"
            ),
        ),
        CorpusCase(
            "contact/near_float_and_typo",
            Contact,
            contact(),
            contact(height_m=1.651, name="Ada Lovelase"),
        ),
        CorpusCase(
            "contact/optional_missing",
            Contact,
            contact(),
            contact(nickname=None),
        ),
        CorpusCase(
            "contact/optional_invented",
            Contact,
            contact(nickname=None),
            contact(nickname="Enchantress"),
        ),
        CorpusCase(
            "contact/default_left",
            Contact,
            contact(),
            contact(country="US"),
        ),
        CorpusCase(
            "contact/all_wrong",
            Contact,
            contact(),
            Contact(
                name="Grace Hopper",
                age=85,
                height_m=1.55,
                is_active=False,
                birth_date=datetime.date(1906, 12, 9),
                updated_at=datetime.datetime(1992, 1, 1, 0, 0, 0),
                status=Status.CLOSED,
                tier="bronze",
                nickname="Amazing Grace",
                emailAddress="grace@navy.example",
                country="USA",
            ),
        ),
        # Invoice: nested, Optional nested, lists.
        CorpusCase("invoice/identical", Invoice, invoice(), invoice()),
        CorpusCase(
            "invoice/wrong_total",
            Invoice,
            invoice(),
            invoice(total=81.0),
        ),
        CorpusCase(
            "invoice/nested_field_wrong",
            Invoice,
            invoice(),
            invoice(
                vendor=Address(
                    street="1 Foundry Way", city="Pittsburgh", postal_code="15999"
                )
            ),
        ),
        CorpusCase(
            "invoice/optional_nested_missing",
            Invoice,
            invoice(),
            invoice(ship_to=None),
        ),
        CorpusCase(
            "invoice/optional_nested_invented",
            Invoice,
            invoice(ship_to=None),
            invoice(),
        ),
        CorpusCase(
            "invoice/line_items_reordered",
            Invoice,
            invoice(),
            invoice(line_items=list(reversed(lines)), tags=["net30", "hardware"]),
        ),
        CorpusCase(
            "invoice/line_item_missing",
            Invoice,
            invoice(),
            invoice(line_items=lines[:2]),
        ),
        CorpusCase(
            "invoice/line_item_extra",
            Invoice,
            invoice(),
            invoice(line_items=lines + [wrong_line]),
        ),
        CorpusCase(
            "invoice/line_item_replaced",
            Invoice,
            invoice(),
            invoice(line_items=lines[:2] + [wrong_line]),
        ),
        CorpusCase(
            "invoice/line_item_sub_field_wrong",
            Invoice,
            invoice(),
            invoice(
                line_items=[
                    lines[0],
                    lines[1].model_copy(update={"quantity": 4}),
                    lines[2],
                ]
            ),
        ),
        CorpusCase(
            "invoice/tags_partial",
            Invoice,
            invoice(),
            invoice(tags=["hardware"]),
        ),
        CorpusCase(
            "invoice/tags_extra",
            Invoice,
            invoice(),
            invoice(tags=["hardware", "net30", "rush"]),
        ),
        CorpusCase(
            "invoice/optional_list_missing",
            Invoice,
            invoice(),
            invoice(credits=None),
        ),
        CorpusCase(
            "invoice/optional_list_invented",
            Invoice,
            invoice(credits=None),
            invoice(),
        ),
        CorpusCase(
            "invoice/lists_emptied",
            Invoice,
            invoice(),
            invoice(tags=[], line_items=[], credits=None, ship_to=None),
        ),
        CorpusCase(
            "invoice/all_wrong",
            Invoice,
            invoice(),
            Invoice(
                invoice_number="X-1",
                issue_date=datetime.date(1999, 1, 1),
                total=1.0,
                currency="EUR",
                vendor=Address(street="Elm", city="Oslo", postal_code="0150"),
                ship_to=Address(street="Oak", city="Bergen", postal_code="5003"),
                tags=["misc"],
                line_items=[wrong_line],
                credits=[wrong_line],
            ),
        ),
        # Lead: sparse schema, where the nothing-to-find guard matters.
        CorpusCase("lead/sparse_identical", Lead, lead(), lead()),
        CorpusCase("lead/sparse_empty_prediction", Lead, lead(), Lead()),
        CorpusCase(
            "lead/sparse_half_found",
            Lead,
            lead(),
            Lead(company="Globex"),
        ),
        CorpusCase(
            "lead/sparse_wrong_value",
            Lead,
            lead(),
            lead(employees=25),
        ),
        CorpusCase(
            "lead/sparse_extra_invented",
            Lead,
            lead(),
            lead(notes="Called twice"),
        ),
        CorpusCase("lead/nothing_to_find_empty", Lead, Lead(), Lead()),
        CorpusCase(
            "lead/nothing_to_find_invented",
            Lead,
            Lead(),
            Lead(company="Hallucinated Inc"),
        ),
        CorpusCase("lead/full_identical", Lead, full_lead(), full_lead()),
        CorpusCase("lead/full_empty_prediction", Lead, full_lead(), Lead()),
        CorpusCase(
            "lead/full_keywords_reordered",
            Lead,
            full_lead(),
            full_lead().model_copy(update={"keywords": ["consulting", "software"]}),
        ),
        # A float NaN reads as absent since #381, so these move in both
        # directions against v1.0.0: a NaN where a number is expected is now a
        # miss (fn, so recall drops), and NaN against NaN or None now matches.
        CorpusCase(
            "lead/nan_for_value",
            Lead,
            lead(annual_revenue=4.5e7),
            lead(annual_revenue=float("nan")),
        ),
        CorpusCase(
            "lead/nan_both",
            Lead,
            lead(annual_revenue=float("nan")),
            lead(annual_revenue=float("nan")),
        ),
        CorpusCase(
            "lead/nan_vs_none",
            Lead,
            lead(annual_revenue=float("nan")),
            lead(),
        ),
        # Order: List[Model] inside List[Model].
        CorpusCase("order/identical", Order, order(), order()),
        CorpusCase(
            "order/shipments_reordered",
            Order,
            order(),
            order(shipments=list(reversed(order().shipments))),
        ),
        CorpusCase(
            "order/inner_item_missing",
            Order,
            order(),
            order(
                shipments=[
                    Shipment(tracking_id="1Z999", carrier="UPS", items=lines[:1]),
                    Shipment(tracking_id="9400111", carrier="USPS", items=lines[2:]),
                ]
            ),
        ),
        CorpusCase(
            "order/shipment_missing",
            Order,
            order(),
            order(shipments=order().shipments[:1]),
        ),
        CorpusCase(
            "order/no_shipments",
            Order,
            order(),
            order(shipments=[]),
        ),
        # The two shapes the merged #399 moves. In the snapshot on purpose: that is
        # what makes a further move fail CI instead of passing unnoticed.
        CorpusCase(
            "tagged/none_element",
            Tagged,
            Tagged(label="x", values=["a", None, "c"]),
            Tagged(label="x", values=["a", "b", "c"]),
        ),
        CorpusCase(
            "tagged/leading_none",
            Tagged,
            Tagged(label="x", values=[None, "b"]),
            Tagged(label="x", values=["b"]),
        ),
    ]


CORPUS: List[CorpusCase] = _cases()
SNAPSHOT_CORPUS: List[CorpusCase] = [case for case in CORPUS if case.snapshot]


# --- the 1.6.0 verdict, re-implemented --------------------------------------


def strands_test_pass(result: Any, match_threshold: float) -> bool:
    """`test_pass` as strands-agents-evals 1.6.0 computes it, copied verbatim.

    Source: strands_evals/evaluators/deterministic/structured_output.py, lines
    128-147 (`_passed`), in the strands-agents-evals 1.6.0 wheel. Keep this a
    faithful copy: it is what tells us whether a stickler change moves a Strands
    verdict. The e2e test checks it against the real evaluator on every case.
    """
    if not bool(result.matched):
        return False
    overall = (result.raw or {}).get("confusion_matrix", {}).get("overall", {})
    findable = overall.get("tp", 0) + overall.get("fn", 0) + overall.get("fd", 0)
    if findable == 0:
        return overall.get("fa", 0) == 0
    return result.recall is not None and result.recall >= match_threshold


def stored_comparison(result: Any) -> Dict[str, Any]:
    """What 1.6.0 stores per case as `metadata["comparison"]` (line 448)."""
    return {k: v for k, v in result.raw.items() if k != "prediction_raw"}
