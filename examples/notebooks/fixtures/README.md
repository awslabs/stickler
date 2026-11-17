# Notebook fixtures

Recorded agent responses, so notebooks can exercise a real SDK code path without credentials or
network access.

## `strands_offline_agent.json`

One agent turn, consumed by `../Strands_Evals_Offline_Agent.ipynb`.

```
provenance.model_id       which model's wire format this follows
provenance.hand_authored  true if the values were written by hand rather than captured
provenance.recorded       ISO timestamp, or null when hand_authored
provenance.note           how it was obtained — keep this honest
turns[].toolUse.name          tool the model called (equals the Pydantic class name)
turns[].toolUse.toolUseId     correlation id echoed in the tool result
turns[].toolUse.inputChunks   the tool input as streamed, split mid-token
turns[].toolUse.input         the same JSON already assembled, for readability
turns[].toolUse.usage         token counts
```

**The committed fixture is hand-authored, not recorded.** It imitates the Bedrock wire shape and
carries three deliberate extraction errors so the notebook has something to score. `model_id` names
the model whose format it follows and whose output it would replace — no model produced these values.
Do not cite the notebook's scores as a measurement of any model.

`inputChunks` is the field that matters. The notebook's `ReplayModel` yields those fragments as
`contentBlockDelta` events, so Strands buffers and parses exactly as it does against a live provider.
`input` is a convenience copy; nothing replays from it.

`turns` is a list because an agent may take several model turns. The replay model raises when the agent
asks for a turn that is not recorded, rather than repeating the last one — a replay that loops looks like
a passing test while proving nothing.

## `record_agent_exchange.py`

Regenerates the fixture from a live Bedrock call. Not imported by any notebook.

```bash
AWS_PROFILE=<profile> python record_agent_exchange.py
```

`RecordingModel` subclasses the SDK's `Model` and delegates to a real `BedrockModel`, copying stream
events through, so the recorded fragment boundaries and token counts are the ones Bedrock emitted.
Subclassing is not optional: `Agent.__init__` reads `model.stateful`, a concrete property on the ABC,
so a duck-typed wrapper raises `AttributeError` before the call is made.

It commits the turn only after the stream ends, because a provider may place `metadata` either side of
`messageStop` and a message may carry text blocks alongside the tool use — so neither event is a safe
commit point. Verified against all three orderings.

Exit codes, all of which leave the fixture untouched:

| code | meaning |
|---|---|
| 1 | `strands-agents` is not installed |
| 2 | the agent call failed — credentials, model access, region, malformed payload, anything else |
| 3 | the call succeeded but the model never invoked the structured-output tool |

Absent credentials are not a separate code: boto3 raises them from inside `invoke_async`, so they share
exit 2. The printed exception type is what distinguishes them — with expired credentials it reads
`ClientError` / `ExpiredTokenException`.

Two shapes are rejected rather than silently mangled: a message carrying more than one tool-use block
(this format records one per turn), and tool input that will not parse as JSON. Both raise inside the
stream and surface as exit 2.

Review the printed payload before committing a new recording. Changing the fixture changes every score
the notebook prints, so re-execute the notebook in the same commit.
