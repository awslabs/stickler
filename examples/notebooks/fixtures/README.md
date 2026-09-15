# Notebook fixtures

Recorded agent responses, so notebooks can exercise a real SDK code path without credentials or
network access.

## `strands_offline_agent.json`

One agent turn, consumed by `../Strands_Evals_Offline_Agent.ipynb`.

```
provenance.model_id   which model produced it
provenance.recorded   ISO timestamp, or null if hand-authored
provenance.note       how it was obtained — keep this honest
turns[].toolUse.name          tool the model called (equals the Pydantic class name)
turns[].toolUse.toolUseId     correlation id echoed in the tool result
turns[].toolUse.inputChunks   the tool input as streamed, split mid-token
turns[].toolUse.input         the same JSON already assembled, for readability
turns[].toolUse.usage         token counts
```

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

It wraps a real `BedrockModel` and copies the stream events through, so the recorded fragment boundaries
and token counts are the ones Bedrock emitted. A missing install, a failed call and a model that never
invoked the tool exit `1`, `2` and `3` respectively and leave the fixture untouched, so a failure is never
written in as though it were an agent response. Review the printed payload before committing a new
recording.

Changing the fixture changes every score the notebook prints. Re-execute the notebook in the same commit.
