#!/usr/bin/env python3
"""Dry-run smoke test for the QA service (standard library only).

Runs the 5 mandatory curl tests of the assignment against a base URL (local or
the public tunnel), then sends every item of the public sample set to /predict
and checks the answer, the offset invariant and the latency.

Usage
-----
    export QA_API_TOKEN=<token>            # PowerShell: $env:QA_API_TOKEN = "<token>"
    python smoke_test.py http://localhost:8000
    python smoke_test.py https://<name>.ngrok-free.app --extended

Exit code is 0 only if every contract check passes.
The token is read from the environment and is never printed.
"""
import argparse
import json
import os
import re
import string
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_SAMPLE = Path(__file__).resolve().parent / "samples" / "public_sample.json"
TIMEOUT_S = 5.0  # the instructor's harness waits at most 5 s per request

failures = []


def call(base, method, path, token=None, body=None, raw=None, headers=None):
    """Return (status, parsed_json_or_None, seconds, response_headers)."""
    url = base.rstrip("/") + path
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token is not None:
        req.add_header("Authorization", token)
    # Free ngrok shows a browser warning page unless this header is present; the harness is not affected.
    req.add_header("ngrok-skip-browser-warning", "1")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            status, text, hdrs = resp.status, resp.read().decode("utf-8", "replace"), resp.headers
    except urllib.error.HTTPError as exc:
        status, text, hdrs = exc.code, exc.read().decode("utf-8", "replace"), exc.headers
    elapsed = time.perf_counter() - started
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    return status, parsed, elapsed, hdrs


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


def norm(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(w for w in s.split() if w not in {"a", "an", "the"})


def is_error_body(body, code):
    return isinstance(body, dict) and body.get("error") == code


def contract_tests(base, token):
    print("\n== The 5 mandatory tests (assignment, section IX) ==")
    st, body, t, _ = call(base, "GET", "/health")
    check("1. /health, no token -> 200 {status: ok}", st == 200 and body == {"status": "ok"}, f"{st} {body} {t*1000:.0f} ms")

    st, body, t, hdrs = call(base, "GET", "/info")
    check("2. /info, no token -> 401 unauthorized", st == 401 and is_error_body(body, "unauthorized"), f"{st} {body}")

    st, body, t, _ = call(base, "GET", "/info", token="Bearer wrong")
    check("3. /info, wrong token -> 401 unauthorized", st == 401 and is_error_body(body, "unauthorized"), f"{st} {body}")

    st, body, t, _ = call(base, "GET", "/info", token=f"Bearer {token}")
    ok = (st == 200 and isinstance(body, dict) and isinstance(body.get("team"), str)
          and " " not in body["team"] and isinstance(body.get("model"), str)
          and isinstance(body.get("params"), int) and body["params"] >= 1)
    check("4. /info, valid token -> 200 with team, model, params", ok, f"{st} {body}")


def sample_tests(base, token, sample_path):
    print("\n== 5. /predict with the public sample set ==")
    items = json.loads(Path(sample_path).read_text(encoding="utf-8"))["items"]
    correct, latencies = 0, []
    for it in items:
        req = {"id": it["id"], "context": it["context"], "question": it["question"]}
        st, body, t, _ = call(base, "POST", "/predict", token=f"Bearer {token}", body=req)
        latencies.append(t)
        shape = (st == 200 and isinstance(body, dict) and body.get("id") == it["id"]
                 and all(k in body for k in ("answer", "start", "score"))
                 and isinstance(body["score"], (int, float)) and 0 <= body["score"] <= 1)
        ans, start = (body or {}).get("answer"), (body or {}).get("start")
        invariant = shape and ((ans is None and start is None) or (
            isinstance(ans, str) and isinstance(start, int)
            and it["context"][start:start + len(ans)] == ans))
        refs = [a["text"] for a in it.get("answers", [])]
        if it["is_impossible"]:
            right = shape and ans is None
        else:
            right = shape and ans is not None and any(norm(ans) == norm(r) for r in refs)
        correct += bool(right)
        check(f"{it['id']}: contract shape + offset invariant + < {TIMEOUT_S:.0f} s",
              bool(shape and invariant and t < TIMEOUT_S), f"{t*1000:.0f} ms")
        expected = "null" if it["is_impossible"] else " | ".join(refs)
        verdict = "correct" if right else "WRONG (informational, not a contract failure)"
        print(f"         expected: {expected}\n         got:      {ans!r} start={start}  -> {verdict}")
    print(f"\n  Public sample accuracy: {correct}/{len(items)}   "
          f"mean latency {sum(latencies)/len(latencies)*1000:.0f} ms, max {max(latencies)*1000:.0f} ms")


def extended_tests(base, token):
    print("\n== Extended contract checks ==")
    auth = f"Bearer {token}"
    ctx = "The Lempa River is the longest river in El Salvador, flowing about 422 km from Guatemala to the Pacific Ocean."
    good = {"id": "q-017", "context": ctx, "question": "How long is the Lempa River?"}

    st, body, _, hdrs = call(base, "GET", "/info")
    check("401 carries WWW-Authenticate: Bearer", st == 401 and (hdrs.get("WWW-Authenticate") or "") == "Bearer")
    st, body, _, _ = call(base, "GET", "/info", token=f"bearer {token}")
    check("lowercase scheme 'bearer' is rejected (case-sensitive)", st == 401)
    if token.swapcase() != token:
        st, body, _, _ = call(base, "GET", "/info", token=f"Bearer {token.swapcase()}")
        check("token compared case-sensitively", st == 401)
    st, body, _, _ = call(base, "GET", "/info", token="Bearer ")
    check("empty bearer token is rejected", st == 401)
    st, body, _, _ = call(base, "POST", "/predict", body=good)
    check("/predict without token -> 401", st == 401 and is_error_body(body, "unauthorized"), f"{st}")
    st, body, _, _ = call(base, "POST", "/predict", raw=b"{not json")
    check("/predict without token and malformed body -> 401 (auth first)", st == 401, f"{st}")

    st, body, _, _ = call(base, "POST", "/predict", token=auth, raw=b"{not json")
    check("malformed JSON -> 400/422 bad_request", st in (400, 422) and is_error_body(body, "bad_request"), f"{st} {body}")
    st, body, _, _ = call(base, "POST", "/predict", token=auth, body={"id": "x", "context": ctx})
    check("missing field -> 400/422 bad_request", st in (400, 422) and is_error_body(body, "bad_request"), f"{st}")
    st, body, _, _ = call(base, "POST", "/predict", token=auth, body={**good, "extra": 1})
    check("extra request field -> 400/422 (additionalProperties: false)", st in (400, 422), f"{st}")
    st, body, _, _ = call(base, "POST", "/predict", token=auth, body={**good, "id": ""})
    check("empty id -> 400/422", st in (400, 422), f"{st}")
    st, body, _, _ = call(base, "POST", "/predict", token=auth, body={**good, "id": "x" * 65})
    check("id longer than 64 -> 400/422", st in (400, 422), f"{st}")
    st, body, _, _ = call(base, "POST", "/predict", token=auth, body={**good, "context": "a" * 5001})
    check("context longer than 5000 -> 400/422", st in (400, 422), f"{st}")
    st, body, _, _ = call(base, "POST", "/predict", token=auth, body={**good, "question": "q" * 501})
    check("question longer than 500 -> 400/422", st in (400, 422), f"{st}")

    st, body, t, _ = call(base, "POST", "/predict", token=auth, body=good)
    b = body or {}
    check("Lempa example: id echoed, offset invariant",
          st == 200 and b.get("id") == "q-017" and isinstance(b.get("answer"), str)
          and ctx[b["start"]:b["start"] + len(b["answer"])] == b["answer"], f"{body} {t*1000:.0f} ms")
    st, body, _, _ = call(base, "POST", "/predict", token=auth,
                          body={"id": "q-018", "context": ctx, "question": "When was the first dam built on the Lempa River?"})
    print(f"         (informational) unanswerable example -> {body}")

    st, body, _, _ = call(base, "POST", "/predict", token=auth, body={**good, "question": "?" * 500})
    check("500-character question does not crash (200)", st == 200, f"{st} {body}")
    st, body, _, _ = call(base, "POST", "/predict", token=auth, body={**good, "context": " " * 50})
    check("whitespace-only context is valid -> 200 null", st == 200 and (body or {}).get("answer", 0) is None, f"{st} {body}")
    st, body, _, _ = call(base, "POST", "/predict", token=auth,
                          body={**good, "id": "ü-ñ-😀", "context": "Café Münich opened in 2019 — « naïve » 😀. " * 100})
    check("unicode id/context, 5000-char limit -> 200 and id echoed", st == 200 and (body or {}).get("id") == "ü-ñ-😀", f"{st}")
    st, body, _, _ = call(base, "GET", "/nope")
    check("unknown route uses the Error schema", st == 404 and isinstance(body, dict) and "error" in body, f"{st} {body}")
    st, body, _, _ = call(base, "GET", "/predict", token=auth)
    check("wrong method uses the Error schema", st == 405 and isinstance(body, dict) and "error" in body, f"{st} {body}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_url", help="http://localhost:8000 or the public tunnel URL")
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE), help="path to public_sample.json")
    ap.add_argument("--extended", action="store_true", help="also run extra contract/edge-case checks")
    args = ap.parse_args()

    token = os.environ.get("QA_API_TOKEN", "")
    if not token:
        sys.exit("QA_API_TOKEN is not set in this terminal. Set it first (it is never printed).")
    if not re.match(r"^https?://", args.base_url):
        sys.exit("base_url must start with http:// or https://")

    print(f"Target: {args.base_url}")
    contract_tests(args.base_url, token)
    sample_tests(args.base_url, token, args.sample)
    if args.extended:
        extended_tests(args.base_url, token)

    print()
    if failures:
        print(f"FAILED {len(failures)} contract check(s):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("All contract checks passed.")


if __name__ == "__main__":
    main()
