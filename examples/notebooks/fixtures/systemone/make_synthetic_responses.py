"""Write synthetic placeholder System One responses into responses.jsonl.

These are NOT model outputs. They stand in until a real run is recorded with
record_systemone.py, so the fixture has the right shape and the notebook and
tests run offline. The answers are drawn from a seeded generator: mostly right,
with errors on plausible confusions, more errors on the tickets written to be
ambiguous, mild overconfidence, and a few refusals (asked, not answered).

    python make_synthetic_responses.py          # rewrites responses.jsonl and provenance.json

Stdlib only. The output is deterministic for a given SEED and input files.
"""

import json
import math
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
SEED = 469
MODEL = "synthetic-placeholder"
REFUSALS = 6  # asked questions left unanswered, across the whole set
REPLACES = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"

# P(argmax is the gold label), before the penalty for ambiguous tickets.
ACCURACY = {
    "queue": 0.92,
    "priority": 0.78,
    "ticket_type": 0.85,
    "frustration": 0.74,
    "needs_refund": 0.95,
    "urgent": 0.88,
}
HARD_PENALTY = 0.25

# Tickets written to be ambiguous (routing, type or tone could go either way).
HARD = {
    "t007",
    "t010",
    "t013",
    "t023",
    "t025",
    "t027",
    "t028",
    "t040",
    "t043",
    "t045",
    "t046",
    "t047",
    "t049",
    "t053",
    "t059",
    "t064",
    "t068",
    "t070",
    "t071",
    "t075",
    "t077",
    "t078",
    "t079",
    "t082",
    "t085",
    "t089",
    "t092",
    "t095",
    "t096",
}

# For a wrong Choice answer: the labels a gold label is most often confused with, likeliest first.
CONFUSIONS = {
    "queue": {
        "billing": ["sales", "account"],
        "technical": ["account", "billing"],
        "account": ["technical", "billing"],
        "shipping": ["billing", "technical"],
        "sales": ["billing", "technical"],
    },
    "priority": {"low": ["normal"], "normal": ["low", "high"], "high": ["normal"]},
    "ticket_type": {
        "question": ["bug", "complaint", "feature_request"],
        "bug": ["complaint", "question"],
        "refund_request": ["complaint"],
        "complaint": ["refund_request", "bug", "question"],
        "feature_request": ["question"],
    },
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wrong_label(rng, qid, gold, labels):
    likely = CONFUSIONS.get(qid, {}).get(gold, [])
    others = [x for x in labels if x != gold]
    weights = [3.0 / (likely.index(x) + 1) if x in likely else 0.3 for x in others]
    return rng.choices(others, weights)[0]


def peak(rng, correct, k):
    """Probability of the predicted answer. Right answers are peakier than wrong ones."""
    floor = 1.0 / k + 0.05
    u = rng.betavariate(5.0, 1.3) if correct else rng.betavariate(2.2, 2.2)
    return floor + (0.995 - floor) * u


def spread(rng, k, top, p_top, gold, closeness=None):
    """Distribute 1 - p_top over the other labels, with extra mass on the gold label if wrong."""
    rest = 1.0 - p_top
    weights = []
    for i in range(k):
        if i == top:
            weights.append(0.0)
            continue
        w = rng.random() + 0.05
        if closeness is not None:
            w *= closeness(i)
        if i == gold:
            w *= 4.0
        weights.append(w)
    total = sum(weights)
    probs = [p_top if i == top else rest * w / total for i, w in enumerate(weights)]
    hi = max(range(k), key=probs.__getitem__)
    if hi != top:  # keep the intended answer as the argmax
        probs[hi], probs[top] = probs[top], probs[hi]
    rounded = [round(p, 4) for p in probs]
    rounded[top] = round(rounded[top] + 1.0 - sum(rounded), 4)
    return rounded


def choice_answer(rng, qid, question, gold, correct):
    labels = list(question["criteria"])
    pred = gold if correct else wrong_label(rng, qid, gold, labels)
    k, top = len(labels), labels.index(pred)
    probs = spread(rng, k, top, peak(rng, correct, k), labels.index(gold))
    return {
        "type": "choice",
        "choice": pred,
        "probabilities": dict(zip(labels, probs)),
        # Decider's derive_confidence: normalised max-probability.
        "confidence": round(max(0.0, (k * max(probs) - 1) / (k - 1)), 4),
    }


def score_answer(rng, question, gold, correct):
    k = len(question["criteria"])
    if correct:
        pred = gold
    else:
        step = 2 if rng.random() < 0.15 else 1
        options = [lv for lv in (gold - step, gold + step) if 0 <= lv < k] or [
            gold - 1 if gold else 1
        ]
        pred = rng.choice(options)
    probs = spread(
        rng,
        k,
        pred,
        peak(rng, correct, k),
        gold,
        closeness=lambda i: math.exp(-abs(i - pred)),
    )
    mean = sum(i * p for i, p in enumerate(probs))
    sigma = math.sqrt(sum(p * (i - mean) ** 2 for i, p in enumerate(probs)))
    return {
        "type": "score",
        "score": round(mean, 2),
        "legend": {str(i): level for i, level in enumerate(question["criteria"])},
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        # Decider's derive_score_confidence with no ordinal smoothing.
        "confidence": round(max(0.0, ((k - 1) / 2 - sigma) / ((k - 1) / 2)), 4),
    }


def noul_answer(rng, gold, correct):
    pred = gold if correct else not gold
    u = rng.betavariate(5.0, 1.2) if correct else rng.betavariate(2.0, 2.0)
    p_pred = 0.5 + (0.995 - 0.5) * u
    return {"type": "noul", "noul": round(p_pred if pred else 1.0 - p_pred, 4)}


def main():
    rng = random.Random(SEED)
    questions = json.loads((HERE / "questions.json").read_text())
    tickets = read_jsonl(HERE / "tickets.jsonl")

    pairs = [(t["id"], qid) for t in tickets for qid in t["asked"]]
    refused = set(rng.sample(pairs, REFUSALS))

    lines = []
    for t in tickets:
        answers = {}
        for qid in t["asked"]:
            question, gold = questions[qid], t["gold"][qid]
            acc = ACCURACY[qid] - (HARD_PENALTY if t["id"] in HARD else 0.0)
            correct = rng.random() < acc
            if question["type"] == "choice":
                answer = choice_answer(rng, qid, question, gold, correct)
            elif question["type"] == "score":
                answer = score_answer(rng, question, gold, correct)
            else:
                answer = noul_answer(rng, gold, correct)
            if (t["id"], qid) not in refused:
                answers[qid] = answer
        prompt_chars = len(t["state"]) + sum(
            len(json.dumps(questions[q])) for q in t["asked"]
        )
        response = {
            "model": MODEL,
            "answers": answers,
            "usage": {
                "input_tokens": 40 + prompt_chars // 4,
                "output_tokens": len(answers),
            },
            "latency_ms": round(rng.uniform(250.0, 900.0), 2),
        }
        lines.append(json.dumps({"id": t["id"], "response": response}))
    (HERE / "responses.jsonl").write_text("\n".join(lines) + "\n")

    provenance = {
        "model_id": REPLACES,
        "response_model": MODEL,
        "hand_authored": True,
        "synthetic": True,
        "recorded": None,
        "strands_decider_version": None,
        "generator": {"script": "make_synthetic_responses.py", "seed": SEED},
        "note": (
            "NOT a recording. responses.jsonl is synthetic, drawn by make_synthetic_responses.py "
            "from the gold labels with a seeded generator, so the notebook and tests run offline. "
            "No model produced these answers. model_id names the model whose output they stand in for. "
            "Do not cite any score computed from them as a measurement. Replace them with "
            "record_systemone.py."
        ),
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"wrote {len(lines)} synthetic responses ({REFUSALS} refusals) to responses.jsonl"
    )


if __name__ == "__main__":
    main()
