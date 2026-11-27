# stickler.integrations

Adapters between Stickler and other systems' wire formats. Each subpackage is thin: it converts shapes and builds models, and leaves scoring to the core. None adds a runtime dependency on the system it adapts.

## systemone

For System One decision models (Strands Decider, OpenJev, Jev).

- `model_from_questions(questions, weights=None)` builds a `StructuredModel` with one `Choice`, `Score` or `Noul` field per question id (see `stickler.decisions`). Every field is `Optional`, so an unasked question is `None`. Questions may be wire dicts or SDK objects; the type is read from `type`, or from the class name.
- `from_systemone(response)` returns `response["answers"]` unchanged, from a dict or an object with `.answers`.

Contract tests: `tests/integrations/test_systemone.py`, pinned to Decider's published answer shapes. Fixture: `examples/notebooks/fixtures/systemone/`.
