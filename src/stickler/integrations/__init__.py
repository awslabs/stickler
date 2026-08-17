"""Adapters that expose stickler through another framework's interfaces.

Each integration lives behind its own extra so a core install pulls none of
them. Import the submodule directly; nothing is re-exported here, because
importing this package must not require any third-party framework:

    from stickler.integrations.strands_evals import StructuredOutputEvaluator

| module | extra | provides |
|---|---|---|
| ``strands_evals`` | ``stickler-eval[strands-evals]`` | ``StructuredOutputEvaluator``, a ``strands_evals.evaluators.Evaluator`` |
"""
