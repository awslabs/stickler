"""Unit tests for the shared optional-annotation helper.

Every spelling of "may be None" must be recognised, and a genuine multi-arm
union must not be mistaken for one.
"""

from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import Field

from stickler.structured_object_evaluator.models.optional_annotation import (
    is_optional_union,
    is_union,
    union_args,
    unwrap_annotated,
    unwrap_optional,
)


class _A:
    pass


class _B:
    pass


class TestIsOptionalUnion:
    def test_all_three_spellings_of_an_optional_agree(self):
        assert is_optional_union(Optional[str]) is True
        assert is_optional_union(Union[str, None]) is True
        assert is_optional_union(str | None) is True

    def test_optional_of_a_class(self):
        assert is_optional_union(Optional[_A]) is True
        assert is_optional_union(_A | None) is True

    def test_optional_of_a_generic(self):
        assert is_optional_union(Optional[List[_A]]) is True
        assert is_optional_union(list[_A] | None) is True
        assert is_optional_union(Optional[Dict[str, int]]) is True
        assert is_optional_union(dict[str, int] | None) is True

    def test_multi_arm_union_with_none_is_not_an_optional(self):
        """No single inner type to descend into, so it must be left alone."""
        assert is_optional_union(Union[_A, _B, None]) is False
        assert is_optional_union(_A | _B | None) is False

    def test_union_without_none_is_not_an_optional(self):
        assert is_optional_union(Union[_A, _B]) is False
        assert is_optional_union(_A | _B) is False

    def test_bare_types_are_not_optional(self):
        assert is_optional_union(str) is False
        assert is_optional_union(_A) is False
        assert is_optional_union(List[str]) is False
        assert is_optional_union(list[str]) is False

    def test_none_itself_is_not_an_optional_wrapper(self):
        assert is_optional_union(None) is False
        assert is_optional_union(type(None)) is False

    def test_nested_optional_collapses_to_a_single_optional(self):
        """Python flattens ``Optional[Optional[T]]``; it stays an optional."""
        assert is_optional_union(Optional[Optional[str]]) is True


class TestIsUnionAndUnionArgs:
    """The permissive pair, for sites that search a union's arms.

    Eight of the ten converted sites ask "does *any* arm look like X", not "is
    there a single inner type". Narrowing them to the latter broke issue #33's
    models, which are annotated ``Optional[List[str]] | Any``.
    """

    def test_is_union_covers_any_arity_and_spelling(self):
        assert is_union(Optional[str]) is True
        assert is_union(Union[str, None]) is True
        assert is_union(str | None) is True
        assert is_union(Union[_A, _B]) is True
        assert is_union(_A | _B | None) is True

    def test_is_union_is_false_for_non_unions(self):
        assert is_union(str) is False
        assert is_union(List[str]) is False
        assert is_union(None) is False

    def test_union_args_drops_none_and_keeps_the_rest(self):
        assert union_args(Optional[str]) == (str,)
        assert union_args(str | None) == (str,)
        assert set(union_args(Union[_A, _B, None])) == {_A, _B}
        assert set(union_args(_A | _B | None)) == {_A, _B}

    def test_union_args_is_empty_for_a_non_union(self):
        """So callers can loop unconditionally."""
        assert union_args(str) == ()
        assert union_args(List[str]) == ()

    def test_union_args_finds_a_list_arm_in_a_wide_union(self):
        """The issue #33 shape: three arms, one of which is a list."""
        annotation = Optional[List[str]] | Any
        assert is_optional_union(annotation) is False  # not a single inner type
        assert any(
            getattr(arg, "__origin__", None) is list for arg in union_args(annotation)
        )

    def test_the_two_spellings_of_a_wide_union_agree(self):
        """Same arms, differing only in how the *union* is written."""
        assert set(union_args(Union[list[str], None, int])) == set(
            union_args(list[str] | None | int)
        )


class TestUnwrapOptional:
    def test_unwraps_every_spelling_to_the_same_inner_type(self):
        assert unwrap_optional(Optional[str]) == (str, True)
        assert unwrap_optional(Union[str, None]) == (str, True)
        assert unwrap_optional(str | None) == (str, True)

    def test_unwraps_a_generic_inner_type(self):
        assert unwrap_optional(Optional[List[_A]]) == (List[_A], True)
        assert unwrap_optional(list[_A] | None) == (list[_A], True)

    def test_returns_a_bare_type_unchanged(self):
        assert unwrap_optional(str) == (str, False)
        assert unwrap_optional(List[str]) == (List[str], False)

    def test_does_not_unwrap_a_multi_arm_union(self):
        """Must not pick an arbitrary arm."""
        assert unwrap_optional(Union[_A, _B, None]) == (Union[_A, _B, None], False)
        multi = _A | _B | None
        assert unwrap_optional(multi) == (multi, False)

    def test_the_two_spellings_unwrap_identically(self):
        assert unwrap_optional(Optional[_A]) == unwrap_optional(_A | None)
        assert unwrap_optional(Optional[List[_A]])[1] == unwrap_optional(list[_A] | None)[1]


class TestUnwrapAnnotated:
    def test_strips_the_wrapper_down_to_the_wrapped_type(self):
        assert unwrap_annotated(Annotated[str, "m"]) is str
        assert unwrap_annotated(Annotated[List[str], "m"]) == List[str]
        assert unwrap_annotated(Annotated[list, "m"]) is list

    def test_keeps_multiple_metadata_entries_out_of_the_result(self):
        assert unwrap_annotated(Annotated[int, "a", "b", "c"]) is int

    def test_unwraps_a_pydantic_field_as_metadata(self):
        """The spelling a `Field(description=...)` produces."""
        assert unwrap_annotated(Annotated[List[str], Field(description="d")]) == List[str]

    def test_returns_anything_else_unchanged(self):
        assert unwrap_annotated(str) is str
        assert unwrap_annotated(List[str]) == List[str]
        assert unwrap_annotated(Optional[str]) == Optional[str]
        assert unwrap_annotated(list) is list

    def test_does_not_reach_inside_a_union(self):
        """A union is not `Annotated`; destructuring it is `union_args`' job.

        `Optional[Annotated[T, ...]]` therefore comes back untouched -- the
        wrapper is on the *arm*, so a caller has to unwrap after descending.
        """
        wrapped = Optional[Annotated[List[str], "m"]]
        assert unwrap_annotated(wrapped) == wrapped

    def test_composes_with_union_args_to_reach_the_inner_type(self):
        """How the readers actually use it: descend, then unwrap."""
        arms = union_args(Optional[Annotated[List[str], "m"]])
        assert [unwrap_annotated(arm) for arm in arms] == [List[str]]

    def test_nested_annotated_is_flattened_by_typing_so_one_pass_suffices(self):
        assert unwrap_annotated(Annotated[Annotated[str, "a"], "b"]) is str

    def test_pep604_annotated_optional_normalises_to_the_subscript_form(self):
        """`Annotated[T, ...] | None` and `Optional[Annotated[T, ...]]` are one type."""
        assert (Annotated[List[str], "m"] | None) == Optional[Annotated[List[str], "m"]]
