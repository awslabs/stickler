"""ANLS* comparison for structured values whose shape is not declared.

ANLS* generalizes ANLS (Average Normalized Levenshtein Similarity, a string
metric from the scene-text VQA literature) to hierarchical values: dicts,
lists, and nesting of both. Leaves are compared as strings, container scores
are averaged over the union of both sides, and list elements are paired
optimally rather than positionally.

Attribution
-----------
The ANLS* metric, and the approach of scoring structured output whose shape is
not declared by walking it as a tree, are taken from the ``anls_star`` project
(https://pypi.org/project/anls_star/), Apache-2.0. See the repository NOTICE.
Stickler's implementation lives in
``stickler.structured_object_evaluator.trees``.

One divergence: the per-leaf cutoff (tau) is a parameter here, ``leaf_threshold``,
rather than a fixed constant, because whether a near match at a leaf should earn
partial credit depends on the data being evaluated. It is deliberately not called
``threshold``: that name means "the score at which this counts as a match" on
every comparator, and a cutoff applied inside the recursion is a different thing.
"""

from typing import Any, Dict, Optional

from pydantic_core import to_jsonable_python

from stickler.comparators.base import BaseComparator

# Tau applied at each leaf during the recursion: the standard ANLS value.
#
# This is an algorithm parameter, NOT a verdict about the field. It changes the
# score the comparator returns, by discarding leaf similarities below it as
# noise. `threshold` is the verdict, and means here exactly what it means on
# every other comparator.
#
# Not raised above this. Tau is what separates "close enough to count" from
# noise, and a stricter cutoff collapses distinct outcomes into the same score.
# At 0.85, an abbreviated value and a missing key both score 0.6667 on a
# three-key mapping -- indistinguishable, which is the exact ranking failure
# this comparator exists to remove. At 0.5 they are 0.8542 and 0.6667.
DEFAULT_LEAF_THRESHOLD = 0.5

# Verdict threshold: at what overall ANLS* score does a mapping count as a match.
#
# 0.7 rather than BaseComparator's 0.5 because a mapping is judged as an object,
# and it preserves the value the dict branch of `ConfigurationHelper` already
# supplied before the comparator carried its own.
DEFAULT_VERDICT_THRESHOLD = 0.7


class ANLSStarComparator(BaseComparator):
    """Score two structured values (dicts, lists, nesting) by ANLS*.

    For values whose keys are not known when the model is written, so no
    per-key comparison config can be declared. Where the keys *are* known,
    a nested ``StructuredModel`` is the better tool: it lets each field carry
    its own comparator and threshold, and it reports per-field results.

    Unlike whole-object equality, this gives partial credit and therefore
    ranks two extractors. On a three-key dict with one abbreviated value:

        ``{"vendor": "Acme Corporation", ...}`` vs ``{"vendor": "Acme Corp", ...}``

        whole-object ``==``                   -> 0.0
        ANLS* at leaf_threshold 0.5 (default) -> 0.8542
        ANLS* at leaf_threshold 0.85          -> 0.6667  (abbreviation rejected)

    Key-set differences are charged on both sides: a renamed key counts once as
    missing from the prediction and once as unexpected in it, because the score
    is normalized over the union of both key sets. A dict compared against
    itself is 1.0, and key order never matters.

    Example:
        ```python
        comparator = ANLSStarComparator()
        comparator.compare({"a": "x", "b": "y"}, {"b": "y", "a": "x"})   # 1.0
        comparator.compare({"a": "x"}, {"a": "x", "b": "y"})             # 0.5

        # Demand a closer match before a mapping counts as one
        strict = ANLSStarComparator(threshold=0.9)

        # Reject looser leaf matches when scoring
        picky = ANLSStarComparator(leaf_threshold=0.85)
        ```

    Args:
        threshold: The verdict threshold, meaning what it means on every other
            comparator: the score at or above which a field using this comparator
            counts as a match, unless the field states its own via
            ``ComparableField(threshold=...)``. Default
            ``DEFAULT_VERDICT_THRESHOLD``, 0.7, rather than ``BaseComparator``'s
            0.5, because a mapping is judged as an object.
        leaf_threshold: Tau, the per-leaf cutoff below which a leaf's string
            similarity is discarded as noise (default
            ``DEFAULT_LEAF_THRESHOLD``, 0.5).

            This is an algorithm parameter, not a verdict. It changes the score
            this comparator returns, by deciding which leaf similarities are
            signal, and it is applied at every leaf of the recursion. Keeping it
            under its own name is deliberate: ``threshold`` means one thing
            across every comparator, and an internal cutoff is a different thing.

            It is not safe to set to 0.0. Without a cutoff, an unrelated string
            earns credit for incidental character overlap, so a wholly wrong
            value scores above zero.

    Leaves are text:
        Every leaf is compared as a string, whatever its Python type. That is
        canonical ANLS*: the metric comes from scene-text VQA, where every answer
        is text. What ANLS* generalises is the STRUCTURE -- aligning dict keys,
        normalising over the union of both key sets, and pairing list elements by
        Hungarian assignment.

        The cost is real and is documented rather than patched. Incidental
        character overlap on a long numeric or identifier value scores high, and
        scores ABOVE a genuine text near-miss::

            wrong IBAN, one character        0.9545
            date one day off                 0.9000
            amount 2x wrong                  0.8571
            "Acme Corporation"/"Acme Corp"   0.5625

        So ``leaf_threshold`` does not separate them: a cutoff high enough to
        reject the IBAN also deletes the partial credit this comparator exists to
        award. **Declare a field whose values you care about**, where it gets a
        comparator chosen for its type. This comparator is for values whose shape
        you could not declare.

        A future ``infer_types=True`` would make leaves type-aware. It must
        delegate to ``stickler.auto.inference`` rather than define its own rules:
        a dict key is a field name, so ``{"amount": 1000}`` would route through
        ``infer_field_config("amount", ...)`` and get the same
        ``NumericComparator@0.95`` with ``relative_tolerance`` that a declared
        ``amount: float`` gets. Hand-rolled leaf rules would be a second
        inference table contradicting the first, which is the divergence
        https://github.com/awslabs/stickler/issues/239 exists to remove.

        The open question blocking it: inference needs a type, and the two sides
        can disagree about theirs (ground truth ``Decimal("10.50")`` against a
        prediction's string ``"10.50"``), so something must decide which side is
        authoritative. That is coercion, unresolved in
        https://github.com/awslabs/stickler/issues/49.

    Cost:
        Scoring a mapping structurally is not free, and it is dramatically more
        expensive than the whole-object equality it replaces. Measured on a
        200-key flat dict: **3.25 ms** per comparison, against 0.0001 ms for
        ``ExactComparator`` over a canonical JSON string.

        The case to watch is a dict field inside a ``List[Model]``, where
        Hungarian matching builds an n x m matrix and every cell pays the cost:

            20 line items, 30-key dict  ->  220 ms for ONE document

        which is roughly 37 minutes over a 10,000-document corpus for that one
        field. If that bites, the remedies are to declare a nested
        ``StructuredModel`` for the keys you actually score, or to follow
        https://github.com/awslabs/stickler/issues/204, which proposes a batch
        comparator hook and a prefilter for exactly this n x m blow-up.

    .. versionadded:: 1.0
    """

    def __init__(
        self,
        threshold: float = DEFAULT_VERDICT_THRESHOLD,
        leaf_threshold: float = DEFAULT_LEAF_THRESHOLD,
    ):
        """Initialize the comparator.

        Args:
            threshold: The verdict threshold, meaning the same thing it means on
                every other comparator: the score at or above which a field
                using this comparator counts as a match, unless the field states
                its own. Default ``DEFAULT_VERDICT_THRESHOLD`` (0.7).
            leaf_threshold: Tau, the per-leaf cutoff applied during the
                recursion. An algorithm parameter, not a verdict: it changes the
                score returned. Default ``DEFAULT_LEAF_THRESHOLD`` (0.5).
        """
        super().__init__(threshold=threshold)
        self.leaf_threshold = leaf_threshold

    @property
    def name(self) -> str:
        """Comparator name for registry and export."""
        return "ANLSStarComparator"

    def _compare(self, str1: Any, str2: Any) -> float:
        """Score two structured values. Neither is None.

        Args:
            str1: Ground truth value (dict, list, or primitive).
            str2: Predicted value.

        Returns:
            ANLS* score in [0.0, 1.0].
        """
        # Imported here rather than at module scope: the trees package imports
        # comparators, so a top-level import would be circular.
        from stickler.structured_object_evaluator.trees.base import ANLSTree

        # Normalize to JSON form FIRST rather than catching the tree's
        # ValueError afterwards. `make_tree` raises for any type it has no node
        # for, and separately for a tuple on the prediction side (tuples are
        # reserved for 1-of-n ground truths). Both are reachable through
        # `Dict[str, Any]`, which can hold arbitrary objects.
        #
        # Catching ValueError around the construction would work, but it would
        # also convert a genuine bug inside tree building into a silent 0.0 --
        # the failure mode rejected in #281 for the equivalent TypeError catch.
        # Normalizing up front means the tree only ever sees types it handles,
        # so no catch is needed and a real error still surfaces.
        #
        # BOTH sides normalize identically, which means a tuple becomes a list
        # on both. ANLS*'s 1-of-n-alternatives contract (a tuple in the ground
        # truth meaning "any one of these is correct") deliberately does NOT
        # apply here: a tuple reached through a pydantic field is a tuple, not a
        # set of alternatives. Preserving it on the ground-truth side only made
        # the normalization asymmetric, so `Dict[str, Tuple[int, int]]` scored
        # 0.0 against an identical copy and 1.0 against a truncated prediction.
        # The 1-of-n contract lives on `anls_score`, which normalizes nothing.
        gt_value = to_jsonable_python(str1, fallback=str)
        pred_value = to_jsonable_python(str2, fallback=str)

        gt_tree = ANLSTree.make_tree(
            gt_value, is_gt=True, threshold=self.leaf_threshold
        )
        pred_tree = ANLSTree.make_tree(
            pred_value, is_gt=False, threshold=self.leaf_threshold
        )

        score, _closest_gt, _key_scores = gt_tree.anls(pred_tree)
        return float(score)

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Round-trippable config for JSON-schema export.

        Only non-default values are emitted, and an all-default instance returns
        ``None``, matching ``NumericComparator.config`` and ``DateComparator``
        so an unremarkable instance adds no
        ``x-aws-stickler-comparator-config`` block to every exported schema.
        """
        config: Dict[str, Any] = {}
        if self.threshold != DEFAULT_VERDICT_THRESHOLD:
            config["threshold"] = self.threshold
        if self.leaf_threshold != DEFAULT_LEAF_THRESHOLD:
            config["leaf_threshold"] = self.leaf_threshold
        return config or None

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ANLSStarComparator(threshold={self.threshold}, "
            f"leaf_threshold={self.leaf_threshold})"
        )
