"""from_json must not force the first model arm of Union[A, B, None] (#505)."""

from typing import Annotated, Optional, Union

from pydantic import BaseModel

from stickler import ComparableField, StructuredModel


class Card(StructuredModel):
    number: Optional[str] = None


class Bank(StructuredModel):
    iban: Optional[str] = None


class Pay(StructuredModel):
    method: Union[Card, Bank, None] = ComparableField(default=None)


def test_from_json_keeps_bank_arm_for_bank_data():
    assert isinstance(Pay.from_json({"method": {"iban": "DE89"}}).method, Bank)


def test_from_json_matches_model_validate_for_union_arms():
    for data in ({"method": {"iban": "DE89"}}, {"method": {"number": "4111"}}):
        assert type(Pay.from_json(data).method) is type(Pay.model_validate(data).method)


def test_different_bank_ibans_do_not_score_one():
    gold = Pay.from_json({"method": {"iban": "DE89"}})
    pred = Pay.from_json({"method": {"iban": "XXXX"}})
    assert gold.compare_with(pred)["overall_score"] == 0.0


def test_equal_bank_ibans_score_one():
    gold = Pay.from_json({"method": {"iban": "DE89"}})
    pred = Pay.from_json({"method": {"iban": "DE89"}})
    assert gold.compare_with(pred)["overall_score"] == 1.0


class PlainBank(BaseModel):
    iban: Optional[str] = None


class PayPlain(StructuredModel):
    method: Union[Card, PlainBank, None] = ComparableField(default=None)


class PayAnnotated(StructuredModel):
    method: Union[Card, Annotated[Bank, "m"], None] = ComparableField(default=None)


class PayPep604(StructuredModel):
    method: Card | Bank | None = ComparableField(default=None)


def test_from_json_matches_model_validate_for_other_union_spellings():
    data = {"method": {"iban": "DE89"}}
    for cls in (PayPlain, PayAnnotated, PayPep604):
        assert type(cls.from_json(data).method) is type(cls.model_validate(data).method)
        assert type(cls.from_json(data).method) is not Card


def test_wrong_iban_scores_zero_for_other_union_spellings():
    for cls in (PayPlain, PayAnnotated, PayPep604):
        gold = cls.from_json({"method": {"iban": "DE89"}})
        pred = cls.from_json({"method": {"iban": "XXXX"}})
        assert gold.compare_with(pred)["overall_score"] == 0.0
