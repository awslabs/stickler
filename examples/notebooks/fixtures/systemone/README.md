# System One fixtures

Support tickets, the System One questions asked about them, and one response per ticket, so the
decision-model notebook and `tests/integrations/test_systemone.py` run offline. Our own data,
Apache-2.0 like the rest of the repo.

**The committed responses are synthetic, not recorded.** `make_synthetic_responses.py` draws them
from the gold labels with a seeded generator. No model produced them. Do not cite any score computed
from them as a measurement of any model. Replace them with `record_systemone.py`.

## Files

| File | What it is |
|---|---|
| `questions.json` | The System One `questions` dict, exactly as sent: `queue`, `priority`, `ticket_type` (Choice), `frustration` (Score, 4 levels, calm to furious), `needs_refund`, `urgent` (Noul). It is also the eval schema: `model_from_questions(questions)`. |
| `tickets.jsonl` | 100 hand-written tickets, one per line: `{"id", "state", "asked", "gold"}`. |
| `responses.jsonl` | One System One response per ticket: `{"id", "response": {"model", "answers", "usage", "latency_ms"}}`. |
| `provenance.json` | Where the responses came from. |
| `make_synthetic_responses.py` | Writes the synthetic `responses.jsonl` and `provenance.json`. Stdlib only, seeded. |
| `record_systemone.py` | Replaces them with answers from a live System One server. Stdlib only. |

### `tickets.jsonl`

```
state   the ticket text, sent as the System One state
asked   the question ids sent for this ticket
gold    question id -> label: an option key (Choice), a level number from 0 (Score), true/false (Noul)
```

Ten tickets leave one or two questions unasked. Those ids are missing from `asked` and their gold is
`null`, which keeps them out of every decision metric. About thirty tickets are deliberately
ambiguous (routing, type or tone could go either way), so a model has something to get wrong.

### `responses.jsonl`

Each response answers only the questions in its ticket's `asked`, in Decider's wire shape: Choice has
`choice`, `probabilities` over every option and `confidence`; Score has `score`, `legend`,
`probabilities` keyed `"0"` and up, and `confidence`; Noul has `noul`. An asked question with no answer
is a refusal, scored as missing. The synthetic set has six.

### `provenance.json`

```
model_id                 the model whose output the responses are (or stand in for)
response_model           the "model" field the responses carry ("synthetic-placeholder" until recorded)
hand_authored, synthetic true until recorded
recorded                 ISO timestamp, or null when synthetic
strands_decider_version  the strands-decider version that served the answers, or null
note                     how the responses were obtained; keep this honest
```

## Recording real responses

```bash
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000
python record_systemone.py
```

On Apple silicon add `--device mlx` (or `--device mps`) to `serve`; the
[strands-decider README](https://github.com/strands-labs/strands-decider#how-to-start-using-strands-decider)
lists the extras each device needs. Keep the server on localhost.

`record_systemone.py` POSTs `{"state", "questions"}` per ticket, with only the asked questions, to
`--url` (default `http://localhost:8000/v1/systemone`). It works with any System One server
(strands-decider, OpenJev, Jev); for hosted Jev pass `--url` and `--model` and set `TYPESAFE_API_KEY`.
It writes `responses.jsonl` and `provenance.json` (model name from the responses, timestamp,
`hand_authored: false`) only after every ticket has a response. The strands-decider version is read
from the local install, or pass `--decider-version` when the server runs elsewhere.

| exit | meaning (fixture untouched) |
|---|---|
| 1 | a request failed: server down, wrong URL, HTTP error, timeout |
| 2 | a response was not a System One response, or answered a question it was not asked |

Review the diff before committing a recording. It changes every number the notebook prints, so
re-execute the notebook in the same commit.
