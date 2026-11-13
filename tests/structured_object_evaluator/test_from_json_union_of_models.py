"""from_json must not force the first model arm of Union[A, B, None] (#505)."""

from typing import Optional, Union

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
