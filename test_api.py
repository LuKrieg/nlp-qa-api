# -*- coding: utf-8 -*-
"""Integration tests for api.py (standard library only).

Starts the FastAPI server in a background process on 127.0.0.1:8000, waits for
/health to return 200, runs all test cases, then shuts the server down.
Do not run it while another server is already listening on port 8000.

Usage
-----
    export QA_API_TOKEN=<token>           # PowerShell: $env:QA_API_TOKEN = "<token>"
    python test_api.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:8000"
PUBLIC_SAMPLES_PATH = PROJECT_ROOT / "samples" / "public_sample.json"

TOKEN = os.environ.get("QA_API_TOKEN", "")
if not TOKEN:
    sys.exit("QA_API_TOKEN is not set. Set it first, e.g. export QA_API_TOKEN=<token> "
             "(PowerShell: $env:QA_API_TOKEN = \"<token>\"), then rerun.")

WRONG_TOKEN = "wrong-token-xyz"
AUTH_HEADERS = {"Authorization": "Bearer " + TOKEN}
WRONG_HEADERS = {"Authorization": "Bearer " + WRONG_TOKEN}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results: list = []


class Response:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        try:
            return json.loads(self.text)
        except ValueError:
            return {}


def request(method: str, path: str, headers: dict = None, body: dict = None, timeout: float = 120) -> Response:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Response(resp.status, resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return Response(exc.code, exc.read().decode("utf-8", "replace"))


def record(name: str, passed: bool, detail: str = "") -> None:
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
            if request("GET", "/health", timeout=3).status_code == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_health_public() -> None:
    print("\n-- /health ------------------------------------------")
    r = request("GET", "/health")
    record("GET /health -> 200", r.status_code == 200, "status=" + str(r.status_code))
    body = r.json()
    record("GET /health body status=ok", body.get("status") == "ok", str(body))


def test_auth() -> None:
    print("\n-- Authentication -----------------------------------")

    r = request("GET", "/info")
    record("GET /info no token -> 401", r.status_code == 401, "status=" + str(r.status_code))
    record("  error=unauthorized", r.json().get("error") == "unauthorized", str(r.json()))

    r = request("GET", "/info", headers=WRONG_HEADERS)
    record("GET /info wrong token -> 401", r.status_code == 401, "status=" + str(r.status_code))

    r = request("POST", "/predict", body={"id": "x", "context": "hello world", "question": "what?"})
    record("POST /predict no token -> 401", r.status_code == 401, "status=" + str(r.status_code))


def test_info() -> None:
    print("\n-- /info --------------------------------------------")
    r = request("GET", "/info", headers=AUTH_HEADERS)
    record("GET /info valid token -> 200", r.status_code == 200, "status=" + str(r.status_code))
    body = r.json()
    for field in ("team", "model", "params", "max_seq_length", "doc_stride"):
        record("  has '" + field + "'", field in body, str(body.get(field)))
    record("  max_seq_length=384", body.get("max_seq_length") == 384)
    record("  doc_stride=128", body.get("doc_stride") == 128)


def test_validation() -> None:
    print("\n-- /predict validation ------------------------------")

    r = request("POST", "/predict", headers=AUTH_HEADERS, body={"context": "hello", "question": "what?"})
    record("Missing 'id' -> 422", r.status_code == 422, "status=" + str(r.status_code))
    record("  error=bad_request", r.json().get("error") == "bad_request", str(r.json()))

    r = request("POST", "/predict", headers=AUTH_HEADERS,
                body={"id": "t1", "context": "", "question": "what?"})
    record("Empty context -> 422", r.status_code == 422, "status=" + str(r.status_code))

    r = request("POST", "/predict", headers=AUTH_HEADERS,
                body={"id": "t2", "context": "hello world", "question": ""})
    record("Empty question -> 422", r.status_code == 422, "status=" + str(r.status_code))


def test_public_samples() -> None:
    print("\n-- /predict public samples --------------------------")

    if not PUBLIC_SAMPLES_PATH.exists():
        record("public samples file present", False, str(PUBLIC_SAMPLES_PATH))
        return

    samples = json.loads(PUBLIC_SAMPLES_PATH.read_text(encoding="utf-8"))
    # Support both list and {"items": [...]} formats
    if isinstance(samples, dict):
        samples = samples.get("items", [])

    for s in samples:
        sid = s["id"]
        t0 = time.perf_counter()
        r = request("POST", "/predict", headers=AUTH_HEADERS,
                    body={"id": sid, "context": s["context"], "question": s["question"]})
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ok_status = r.status_code == 200
        body = r.json() if ok_status else {}
        id_echoed = body.get("id") == sid
        has_keys = "answer" in body and "start" in body and "score" in body

        record(sid + " -> 200", ok_status, "status=" + str(r.status_code))
        record("  id echoed", id_echoed, "got=" + repr(body.get("id")))
        record("  has answer/start/score", has_keys)

        ans_str = repr(body.get("answer"))[:45]
        print("    answer=" + ans_str + "  start=" + str(body.get("start")) + "  score="
              + str(body.get("score")) + "  " + str(int(elapsed_ms)) + "ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000",
         "--log-level", "warning"],
        cwd=str(PROJECT_ROOT),
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        print("Waiting for server to start and model to load (up to 90 s)...")
        if not wait_for_server(timeout=90):
            print("FAIL -- server did not become ready in time.")
            server_proc.terminate()
            _, raw = server_proc.communicate(timeout=10)
            print((raw or b"").decode("utf-8", errors="replace"))
            sys.exit(1)

        print("Server ready -- running tests")
        test_health_public()
        test_auth()
        test_info()
        test_validation()
        test_public_samples()
    finally:
        if server_proc.poll() is None:
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
