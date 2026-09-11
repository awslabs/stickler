"""A ``SecretStr`` / ``SecretBytes`` leaf is refused, never scored (#320).

Pydantic masks a secret to the constant ``'**********'`` the moment it is
converted to JSON form, which is the first thing every comparison path does.
By the time any comparator sees the value, both sides are byte-identical and
the real difference is gone -- so two DIFFERENT secrets scored a perfect 1.0,
an invented match on a credential mismatch.

The fix routes a secret leaf to the same refusal ANLS* already uses for values
it cannot represent (see ``_UNSCOREABLE`` in ``stickler.comparators.anls``): it
scores 0.0 and warns, rather than comparing the mask. Refusal does not consult
equality -- an unscoreable value is not scored whether or not the two happen to
be equal -- so identical secrets are refused too. That is the honest signal:
"we could not score this," not "these matched."

The issue makes two constraints non-negotiable, both pinned here: the fix must
cover ``SecretBytes`` as well as ``SecretStr``, and it must apply to a bare
``Dict[str, SecretStr]`` as well as a plain-model field.
"""

from typing import Dict, Optional

import pytest
from pydantic import BaseModel, Field, Secret, SecretBytes, SecretStr

import stickler
from stickler import ANLSStarComparator, StructuredModel


class Creds(BaseModel):
    api_key: Optional[SecretStr] = None


class CredsBytes(BaseModel):
    token: Optional[SecretBytes] = None


class DocPlainModel(StructuredModel):
    creds: Optional[Creds] = None


class DocPlainModelBytes(StructuredModel):
    creds: Optional[CredsBytes] = None


class DocSecretDict(StructuredModel):
    creds: Optional[Dict[str, SecretStr]] = None


class TestTheComparatorRefusesSecretLeaves:
    """Directly on ANLS*, the smallest unit that owns the coercion."""

    def test_two_different_secrets_are_not_a_match(self):
        c = ANLSStarComparator()
        assert c.compare(
            {"k": SecretStr("s3cret")}, {"k": SecretStr("hunter2")}
        ) == pytest.approx(0.0)

    def test_identical_secrets_are_refused_not_matched(self):
        """Refusal does not consult equality: an unscoreable value scores 0.0
        whether or not the two sides are equal."""
        c = ANLSStarComparator()
        assert c.compare(
            {"k": SecretStr("s3cret")}, {"k": SecretStr("s3cret")}
        ) == pytest.approx(0.0)

    def test_secret_bytes_are_refused_too(self):
        c = ANLSStarComparator()
        assert c.compare(
            {"k": SecretBytes(b"aaa")}, {"k": SecretBytes(b"bbb")}
        ) == pytest.approx(0.0)

    def test_a_secret_nested_in_a_plain_model_is_refused(self):
        c = ANLSStarComparator()
        assert c.compare(
            Creds(api_key=SecretStr("s3cret")),
            Creds(api_key=SecretStr("hunter2")),
        ) == pytest.approx(0.0)


class TestTheRefusalIsAnnounced:
    def test_a_secret_leaf_warns(self):
        """A silent 0.0 is indistinguishable from a wrong extraction, so the
        caller is told the value was refused because it is masked."""
        with pytest.warns(UserWarning, match="masks it to a constant"):
            ANLSStarComparator().compare(
                {"k": SecretStr("s3cret")}, {"k": SecretStr("hunter2")}
            )

    def test_secret_bytes_also_warns(self):
        """warn_once keys on the type name, so SecretBytes is tracked separately
        from SecretStr and needs its own coverage."""
        with pytest.warns(UserWarning, match="masks it to a constant"):
            ANLSStarComparator().compare(
                {"k": SecretBytes(b"aaa")}, {"k": SecretBytes(b"bbb")}
            )


class TestTheModelLevelReproduction:
    """The exact shapes from issue #320, via ``compare_with``."""

    def test_plain_model_field_no_longer_invents_a_match(self):
        r = DocPlainModel(creds=Creds(api_key=SecretStr("s3cret"))).compare_with(
            DocPlainModel(creds=Creds(api_key=SecretStr("hunter2"))),
            include_confusion_matrix=True,
        )
        assert r["field_scores"]["creds"] == pytest.approx(0.0)
        assert r["confusion_matrix"]["overall"]["tp"] == 0

    def test_bare_secret_dict_no_longer_invents_a_match(self):
        r = DocSecretDict(creds={"k": SecretStr("aaa")}).compare_with(
            DocSecretDict(creds={"k": SecretStr("bbb")})
        )
        assert r["field_scores"]["creds"] == pytest.approx(0.0)

    def test_secret_bytes_field_no_longer_invents_a_match(self):
        r = DocPlainModelBytes(
            creds=CredsBytes(token=SecretBytes(b"aaa"))
        ).compare_with(
            DocPlainModelBytes(creds=CredsBytes(token=SecretBytes(b"bbb"))),
            include_confusion_matrix=True,
        )
        assert r["field_scores"]["creds"] == pytest.approx(0.0)
        assert r["confusion_matrix"]["overall"]["tp"] == 0

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Separate masking site, out of #320's scope. The zero-config "
            "`evaluate` (auto) facade normalizes the whole instance with "
            "`model_dump(mode='json')` before comparison, which masks every "
            "secret up front -- earlier than, and independent of, the ANLS* "
            "coercion #320 fixes. Refusing it needs `auto/inference.py` to route "
            "SecretStr to a refusing comparator, which lands in the builder/"
            "inference files #255/#250/#263 are rewriting. Tracked separately; "
            "when that fix lands this flips to XPASS and should be un-xfailed."
        ),
    )
    def test_zero_config_evaluate_path(self):
        """The route a user hits without any Stickler-specific code."""
        result = stickler.evaluate(
            Creds(api_key=SecretStr("s3cret")),
            Creds(api_key=SecretStr("hunter2")),
        )
        assert result.field_scores["api_key"] == pytest.approx(0.0)


class TestTheRefusalDoesNotBreakWhatItPassesThrough:
    """`_refuse_secrets` runs in front of EVERY `to_jsonable_python` call in
    `_compare`, so it sees values with no secret in them at all. Whatever it
    does to those has to be a no-op. Two ways it was not.
    """

    def test_a_set_of_tuples_still_scores(self):
        """Rebuilding a set from the recursive results raised.

        The `(list, tuple)` branch returns a LIST, and lists are unhashable, so
        `{_refuse_secrets(item) for item in value}` raised
        `TypeError: unhashable type: 'list'` on any set or frozenset holding a
        tuple. No secret involved: this broke a comparison that worked before.
        """
        c = ANLSStarComparator()
        assert c.compare({(1, 2), (3, 4)}, {(1, 2), (3, 4)}) == pytest.approx(1.0)
        assert c.compare(frozenset({(1, 2)}), frozenset({(1, 2)})) == pytest.approx(1.0)

    def test_an_aliased_model_keeps_its_alias_keys(self):
        """`model_dump()` keys by FIELD NAME; `to_jsonable_python` keys by ALIAS.

        Dumping without `by_alias=True` silently changed the key convention for
        every aliased model, including models with no secret anywhere in them, so
        a model compared against its own `model_dump(by_alias=True)` lost a field
        per alias. It is a straight swap, so both directions are pinned.
        """

        class Aliased(BaseModel):
            model_config = {"populate_by_name": True}
            tok: str = Field(alias="token")
            a: int = 1

        m = Aliased(token="plain")
        c = ANLSStarComparator()
        assert c.compare(m, m.model_dump(by_alias=True)) == pytest.approx(1.0)
        assert c.compare(m, m.model_dump()) < 1.0


class TestEverySecretFamilyIsRefused:
    def test_secret_of_t_is_refused_too(self):
        """`Secret[T]` is a SIBLING of `SecretStr`, not a parent or a child.

        `isinstance(v, Secret)` does not match a `SecretStr`, and naming the two
        concrete classes misses every `Secret[T]`, so `Secret[int](1)` against
        `Secret[int](2)` kept scoring 1.0 -- #320's exact symptom. The public
        duck-type `get_secret_value` is what covers all three families.
        """
        c = ANLSStarComparator()
        with pytest.warns(UserWarning, match="refused"):
            assert c.compare(Secret[int](1), Secret[int](2)) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "make",
        [
            lambda v: SecretStr(str(v)),
            lambda v: SecretBytes(str(v).encode()),
            lambda v: Secret[int](v),
        ],
        ids=["SecretStr", "SecretBytes", "Secret[int]"],
    )
    def test_identical_secrets_are_refused_not_matched(self, make):
        """Refusal does not consult equality, so identical is refused as well."""
        c = ANLSStarComparator()
        assert c.compare(make(1), make(1)) == pytest.approx(0.0)
