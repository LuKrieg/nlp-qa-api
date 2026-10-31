# -*- coding: utf-8 -*-
"""Integration tests for api.py.

Starts the FastAPI server in a background process, waits for /health to
return 200, runs all test cases, then shuts the server down.

Usage
-----
    $env:QA_API_TOKEN = "dev-token"
    .\.venv\Scripts\python.exe test_api.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
TOKEN = os.environ.get("QA_API_TOKEN", "dev-token")
WRONG_TOKEN = "wrong-token-xyz"
AUTH_HEADERS = {"Authorization": "Bearer " + TOKEN}
WRONG_HEADERS = {"Authorization": "Bearer " + WRONG_TOKEN}

PUBLIC_SAMPLES_PATH = Path(
    r"C:\Users\andre\Desktop\Semana 10 y 11 - NLP + APIs\public_sample.json"
)

VENV_PYTHON = str(Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results: list = []


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    mark = "+" if passed else "X"
    msg = "  [" + mark + "] " + name
    if detail:
        msg += " -- " + detail
    print(msg)
    results.append({"name": name, "passed": passed, "detail": detail})


def wait_for_server(timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(BASE_URL + "/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_health_public(client: httpx.Client) -> None:
    print("\n-- /health ------------------------------------------")
    r = client.get("/health")
    record("GET /health -> 200", r.status_code == 200, "status=" + str(r.status_code))
    body = r.json()
    record("GET /health body status=ok", body.get("status") == "ok", str(body))


def test_auth(client: httpx.Client) -> None:
    print("\n-- Authentication -----------------------------------")

    r = client.get("/info")
    record("GET /info no token -> 401", r.status_code == 401, "status=" + str(r.status_code))
    record("  error=unauthorized", r.json().get("error") == "unauthorized", str(r.json()))

    r = client.get("/info", headers=WRONG_HEADERS)
    record("GET /info wrong token -> 401", r.status_code == 401, "status=" + str(r.status_code))

    r = client.post("/predict", json={"id": "x", "context": "hello world", "question": "what?"})
    record("POST /predict no token -> 401", r.status_code == 401, "status=" + str(r.status_code))


def test_info(client: httpx.Client) -> None:
    print("\n-- /info --------------------------------------------")
    r = client.get("/info", headers=AUTH_HEADERS)
    record("GET /info valid token -> 200", r.status_code == 200, "status=" + str(r.status_code))
    body = r.json()
    for field in ("team", "model", "params", "max_seq_length", "doc_stride"):
        record("  has '" + field + "'", field in body, str(body.get(field)))
    record("  max_seq_length=384", body.get("max_seq_length") == 384)
    record("  doc_stride=128", body.get("doc_stride") == 128)


def test_validation(client: httpx.Client) -> None:
    print("\n-- /predict validation ------------------------------")

    r = client.post("/predict", headers=AUTH_HEADERS, json={"context": "hello", "question": "what?"})
    record("Missing 'id' -> 422", r.status_code == 422, "status=" + str(r.status_code))
    record("  error=bad_request", r.json().get("error") == "bad_request", str(r.json()))

    r = client.post("/predict", headers=AUTH_HEADERS,
                    json={"id": "t1", "context": "", "question": "what?"})
    record("Empty context -> 422", r.status_code == 422, "status=" + str(r.status_code))

    r = client.post("/predict", headers=AUTH_HEADERS,
                    json={"id": "t2", "context": "hello world", "question": ""})
    record("Empty question -> 422", r.status_code == 422, "status=" + str(r.status_code))


def test_public_samples(client: httpx.Client) -> None:
    print("\n-- /predict public samples --------------------------")

    if not PUBLIC_SAMPLES_PATH.exists():
        print("  WARNING: Public samples not found at " + str(PUBLIC_SAMPLES_PATH) + ", skipping")
        return

    samples = json.loads(PUBLIC_SAMPLES_PATH.read_text(encoding="utf-8"))
    # Support both list and {"items": [...]} formats
    if isinstance(samples, dict):
        samples = samples.get("items", [])

    for s in samples:
        sid = s["id"]
        t0 = time.perf_counter()
        r = client.post(
            "/predict",
            headers=AUTH_HEADERS,
            json={"id": sid, "context": s["context"], "question": s["question"]},
            timeout=120,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ok_status = r.status_code == 200
        body = r.json() if ok_status else {}
        id_echoed = body.get("id") == sid
        has_keys = "answer" in body and "start" in body and "score" in body

        record(sid + " -> 200", ok_status, "status=" + str(r.status_code))
        record("  id echoed", id_echoed, "got=" + repr(body.get("id")))
        record("  has answer/start/score", has_keys)

        ans = body.get("answer")
        start = body.get("start")
        score = body.get("score")
        ans_str = repr(ans)[:45]
        print("    answer=" + ans_str + "  start=" + str(start) + "  score=" + str(score) + "  " + str(int(elapsed_ms)) + "ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    server_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "api:app", "--port", "8000", "--log-level", "warning"],
        cwd=str(Path(__file__).resolve().parent),
        env={**os.environ, "QA_API_TOKEN": TOKEN},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print("Waiting for server to start and model to load (up to 90 s)...")
    ready = wait_for_server(timeout=90)

    if not ready:
        raw = server_proc.stderr.read() if server_proc.stderr else b""
        stderr = raw.decode("utf-8", errors="replace")
        print("FAIL -- server did not become ready in time.")
        print(stderr)
        server_proc.terminate()
        sys.exit(1)

    print("Server ready -- running tests")

    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        test_health_public(client)
        test_auth(client)
        test_info(client)
        test_validation(client)
        test_public_samples(client)

    server_proc.terminate()
    server_proc.wait(timeout=10)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    failed_tests = [r for r in results if not r["passed"]]
    print("\n" + "=" * 55)
    print("  Result: " + str(passed) + "/" + str(total) + " PASS")
    if failed_tests:
        print("  Failed:")
        for r in failed_tests:
            print("    * " + r["name"] + ": " + r["detail"])
    print("=" * 55)
    sys.exit(0 if not failed_tests else 1)


if __name__ == "__main__":
    main()
