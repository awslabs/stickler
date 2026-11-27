"""Tests for SemanticComparator."""

import logging
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from stickler.comparators import SemanticComparator
from stickler.comparators.semantic import _cosine_distance

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestSemanticComparator:
    """Test semantic comparator fallback behavior."""

    def test_model_id_remains_a_string_when_using_default_embedding(self):
        """Default embedding setup stores model_id as a string."""
        comparator = SemanticComparator(model_id="test-model")

        assert comparator.model_id == "test-model"
        assert isinstance(comparator.model_id, str)

    def test_compare_logs_embedding_failures_before_fallback(self, caplog):
        """Embedding failures are logged before returning unequal fallback."""

        def raise_embedding_failure(_value):
            raise RuntimeError("simulated embedding outage")

        comparator = SemanticComparator(
            embedding_function=raise_embedding_failure, model_id="custom-model"
        )

        with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
            score = comparator.compare("cat on mat", "feline on rug")

        record = caplog.records[0]
        assert score == 0.0
        assert "Semantic embedding comparison failed" in caplog.text
        assert "simulated embedding outage" in caplog.text
        assert record.embedding_function == "raise_embedding_failure"
        assert record.model_id == "custom-model"
        assert record.input_1_length == len("cat on mat")
        assert record.input_2_length == len("feline on rug")
        assert record.exception_type == "RuntimeError"

    def test_compare_keeps_exact_match_fallback_when_embedding_fails(self, caplog):
        """Embedding failures keep exact-match fallback semantics."""

        def raise_embedding_failure(_value):
            raise RuntimeError("simulated embedding outage")

        comparator = SemanticComparator(embedding_function=raise_embedding_failure)

        with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
            score = comparator.compare("same", "same")

        assert score == 1.0
        assert "Semantic embedding comparison failed" in caplog.text

    def test_compare_logs_non_string_input_lengths_before_fallback(self, caplog):
        """Fallback logging does not raise for primitive non-string inputs."""

        def raise_embedding_failure(_value):
            raise RuntimeError("simulated embedding outage")

        comparator = SemanticComparator(embedding_function=raise_embedding_failure)

        with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
            score = comparator.compare(123, 456)

        record = caplog.records[0]
        assert score == 0.0
        assert record.input_1_length == len("123")
        assert record.input_2_length == len("456")

    def test_default_bedrock_fallback_log_includes_function_and_model(
        self, caplog, monkeypatch
    ):
        """Default Bedrock partial logs the wrapped function name and model ID."""

        def raise_bedrock_failure(_value, model_id):
            raise RuntimeError(f"simulated {model_id} outage")

        monkeypatch.setattr(
            "stickler.comparators.semantic.generate_bedrock_embedding",
            raise_bedrock_failure,
        )
        comparator = SemanticComparator(model_id="amazon.test-model")

        with caplog.at_level(logging.ERROR, logger="stickler.comparators.semantic"):
            score = comparator.compare("cat", "dog")

        record = caplog.records[0]
        assert score == 0.0
        assert record.embedding_function == "raise_bedrock_failure"
        assert record.model_id == "amazon.test-model"
        assert record.exception_type == "RuntimeError"


# --- numpy cosine parity with scipy (#605) ----------------------------------
#
# SemanticComparator used scipy.spatial.distance.cosine; it now computes the
# same distance in numpy so the `semantic` extra need not install scipy. scipy
# stays in the dev/test environment, where it is the oracle for these tests.

_RNG = np.random.default_rng(605)

_PARITY_CASES = {
    # Titan v2's default output size, plus a short and a single-element vector.
    "random_1024": (_RNG.standard_normal(1024), _RNG.standard_normal(1024)),
    "random_8": (_RNG.standard_normal(8), _RNG.standard_normal(8)),
    "random_positive": (_RNG.random(256), _RNG.random(256)),
    "single_element": (np.array([3.0]), np.array([-2.0])),
    "identical": (np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])),
    "scaled": (np.array([1.0, 2.0, 3.0]), np.array([100.0, 200.0, 300.0])),
    "orthogonal": (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
    # Distance 2.0, the top of scipy's clip range.
    "opposite": (np.array([1.0, -2.0, 3.0]), np.array([-1.0, 2.0, -3.0])),
    "partly_aligned": (np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    # Python lists and ints, as a custom embedding_function may return.
    "python_int_lists": ([1, 2, 3], [4, 5, 6]),
    "float32": (
        _RNG.standard_normal(64).astype(np.float32),
        _RNG.standard_normal(64).astype(np.float32),
    ),
}

# What scipy returns for a zero vector: NaN (0/0), not 0.0 or 1.0.
_ZERO_CASES = {
    "both_zero": (np.zeros(4), np.zeros(4)),
    "first_zero": (np.zeros(3), np.array([1.0, 2.0, 3.0])),
    "second_zero": (np.array([1.0, 2.0, 3.0]), np.zeros(3)),
}


def _scipy_cosine():
    distance = pytest.importorskip("scipy.spatial.distance")
    return distance.cosine


@pytest.mark.parametrize("case", sorted(_PARITY_CASES))
def test_numpy_cosine_matches_scipy(case):
    x, y = _PARITY_CASES[case]

    expected = _scipy_cosine()(x, y)

    assert _cosine_distance(x, y) == pytest.approx(expected, abs=1e-12, rel=0)


def test_numpy_cosine_matches_scipy_on_many_random_vectors():
    cosine = _scipy_cosine()
    rng = np.random.default_rng(1605)
    for _ in range(200):
        n = int(rng.integers(1, 2048))
        x, y = rng.standard_normal(n), rng.standard_normal(n)
        assert _cosine_distance(x, y) == pytest.approx(cosine(x, y), abs=1e-12, rel=0)


@pytest.mark.parametrize("case", sorted(_ZERO_CASES))
def test_zero_vector_matches_scipy_nan(case):
    """Pinned: a zero vector gives NaN, as scipy did, with a RuntimeWarning."""
    x, y = _ZERO_CASES[case]

    with pytest.warns(RuntimeWarning):
        expected = _scipy_cosine()(x, y)
    with pytest.warns(RuntimeWarning):
        actual = _cosine_distance(x, y)

    assert math.isnan(expected)
    assert math.isnan(actual)


def test_zero_vector_similarity_is_nan_through_the_comparator():
    """The NaN reaches compare(); this pins the pre-#605 behaviour."""
    comparator = SemanticComparator(embedding_function=lambda _s: [0.0, 0.0, 0.0])

    with pytest.warns(RuntimeWarning):
        score = comparator.compare("a", "b")

    assert math.isnan(score)


def test_known_values_without_scipy():
    """Hand-computed values, so the formula is checked even where scipy is absent."""
    assert _cosine_distance([1, 0, 0], [0, 1, 0]) == 1.0
    assert _cosine_distance([100, 0, 0], [0, 1, 0]) == 1.0
    assert _cosine_distance([1, 1, 0], [0, 1, 0]) == pytest.approx(
        1 - 1 / math.sqrt(2), abs=1e-15
    )
    assert _cosine_distance([1, 2], [-1, -2]) == 2.0
    assert _cosine_distance([0.3, 0.4], [0.3, 0.4]) == 0.0


@pytest.mark.parametrize(
    "x,scale,expected",
    [
        # Unclipped, these round to -2.2e-16 and 2.0000000000000004. scipy
        # clips to [0, 2]; a similarity above 1.0 or below -1.0 would leak out.
        ([-2.3250307746388343, -0.21879166393254573, -1.2459109472530652], 0.1, 0.0),
        ([1.1153177306963604, 1.2485357836511943, 0.710303668529703], -0.1, 2.0),
    ],
)
def test_rounding_past_the_range_is_clipped_like_scipy(x, scale, expected):
    x = np.array(x)
    y = x * scale

    assert _cosine_distance(x, y) == expected
    assert _scipy_cosine()(x, y) == expected


def test_similarity_is_one_minus_distance():
    comparator = SemanticComparator(
        embedding_function=lambda s: [1.0, 1.0, 0.0] if s == "a" else [0.0, 1.0, 0.0]
    )

    assert comparator.compare("a", "b") == pytest.approx(1 / math.sqrt(2), abs=1e-15)


@pytest.mark.parametrize(
    "x,y",
    [
        (np.ones((2, 2)), np.ones((2, 2))),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
    ],
)
def test_bad_shapes_raise_like_scipy(x, y):
    with pytest.raises(ValueError):
        _cosine_distance(x, y)
    cosine = _scipy_cosine()
    with pytest.raises(ValueError):
        cosine(x, y)


def test_complex_input_raises_like_scipy():
    with pytest.raises(TypeError):
        _cosine_distance([1 + 1j, 2.0], [1.0, 2.0])
    cosine = _scipy_cosine()
    with pytest.raises(TypeError):
        cosine([1 + 1j, 2.0], [1.0, 2.0])


def test_semantic_comparator_scores_without_scipy():
    """With scipy unimportable, a real embedding still scores, not exact-match."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["scipy"] = None  # any `import scipy...` now raises ImportError
        from stickler.comparators import SemanticComparator
        vectors = {"cat": [1.0, 1.0, 0.0], "feline": [0.0, 1.0, 0.0]}
        c = SemanticComparator(embedding_function=vectors.__getitem__)
        print(repr(c.compare("cat", "feline")))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert float(result.stdout) == pytest.approx(1 / math.sqrt(2), abs=1e-15)
    assert "Semantic embedding comparison failed" not in result.stderr


def test_semantic_extra_does_not_list_scipy():
    text = (REPO_ROOT / "pyproject.toml").read_text()
    block = re.search(r"^semantic = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    assert block, "could not find the semantic extra in pyproject.toml"
    names = re.findall(r'^\s*"([A-Za-z0-9._-]+)', block.group(1), re.MULTILINE)

    assert names == ["boto3"]
