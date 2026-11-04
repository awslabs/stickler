"""Record real System One answers for tickets.jsonl into responses.jsonl.

Run once against a System One server to replace the synthetic placeholders:

    strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000
    python record_systemone.py

For each ticket it POSTs ``{"state", "questions"}`` to ``--url``, sending only
the questions in that ticket's ``asked`` list, and stores the response as
returned. Any server with the System One interface works: strands-decider,
OpenJev or Jev. For a hosted Jev, pass ``--url https://api.typesafe.ai/v1/systemone``
and ``--model``, with the API key in ``TYPESAFE_API_KEY``.

Stdlib only. Nothing is written until every ticket has a response, so a failed
run leaves the fixture untouched. Exit codes:

    1  a request failed: server not running, wrong URL, HTTP error, timeout
    2  a response was not a System One response, or answered a question it was not asked
"""

import argparse
import datetime
import importlib.metadata
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_URL = "http://localhost:8000/v1/systemone"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def post(url, body, timeout, api_key):
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def local_decider_version():
    try:
        return importlib.metadata.version("strands-decider")
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"System One endpoint (default {DEFAULT_URL})",
    )
    ap.add_argument(
        "--model", help="model field to send; omit to use the server's default"
    )
    ap.add_argument(
        "--timeout", type=float, default=120.0, help="seconds per request (default 120)"
    )
    ap.add_argument(
        "--decider-version",
        help="strands-decider version the server runs; defaults to the version installed here, if any",
    )
    args = ap.parse_args(argv)

    questions = json.loads((HERE / "questions.json").read_text())
    tickets = read_jsonl(HERE / "tickets.jsonl")
    api_key = os.environ.get("TYPESAFE_API_KEY")

    lines, models = [], []
    for n, ticket in enumerate(tickets, 1):
        body = {
            "state": ticket["state"],
            "questions": {q: questions[q] for q in ticket["asked"]},
        }
        if args.model:
            body["model"] = args.model
        try:
            response = post(args.url, body, args.timeout, api_key)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            print(
                f"{ticket['id']}: HTTP {exc.code} from {args.url}: {detail}",
                file=sys.stderr,
            )
            return 1
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(
                f"{ticket['id']}: could not reach {args.url} ({exc}). Is the server running?",
                file=sys.stderr,
            )
            return 1
        except ValueError as exc:
            print(f"{ticket['id']}: response is not JSON ({exc})", file=sys.stderr)
            return 2

        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict):
            print(
                f"{ticket['id']}: response has no 'answers' object: {str(response)[:300]}",
                file=sys.stderr,
            )
            return 2
        extra = sorted(set(answers) - set(ticket["asked"]))
        if extra:
            print(
                f"{ticket['id']}: answered questions it was not asked: {extra}",
                file=sys.stderr,
            )
            return 2
        missing = sorted(set(ticket["asked"]) - set(answers))
        if missing:
            print(f"{ticket['id']}: no answer for {missing}; recorded as missing")
        models.append(response.get("model"))
        lines.append(json.dumps({"id": ticket["id"], "response": response}))
        print(f"[{n}/{len(tickets)}] {ticket['id']}", end="\r", flush=True)
    print()

    names = sorted({m for m in models if m})
    model = names[0] if len(names) == 1 else (names or None)
    version = args.decider_version or local_decider_version()
    provenance = {
        "model_id": model,
        "response_model": model,
        "hand_authored": False,
        "synthetic": False,
        "recorded": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "strands_decider_version": version,
        "endpoint": args.url,
        "note": (
            f"Recorded from a live System One server at {args.url} with record_systemone.py. "
            f"strands_decider_version is {'the version given or installed on the recording machine' if version else 'unknown'}."
        ),
    }
    (HERE / "responses.jsonl").write_text("\n".join(lines) + "\n")
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"recorded {len(lines)} responses from {model} to responses.jsonl")
    print(
        "review the diff before committing it; it changes every number the notebook prints."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
