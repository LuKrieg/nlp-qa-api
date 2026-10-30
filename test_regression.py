"""Regression and benchmark tests for inference.py with modelo-final-v2."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import QAInference

SAMPLE_PATH = Path(r"C:\Users\andre\Desktop\Semana 10 y 11 - NLP + APIs\public_sample.json")


def test_different_working_directory():
    orig_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            qa = QAInference()
            return qa.model is not None and qa.no_answer_threshold is not None
        finally:
            os.chdir(orig_cwd)


def test_input_validation():
    qa = QAInference()
    invalid_cases = [
        ("", "Valid question?"),
        ("   ", "Valid question?"),
        (None, "Valid question?"),
        (12345, "Valid question?"),
        ("Valid context", ""),
        ("Valid context", "   "),
        ("Valid context", None),
        ("Valid context", 123),
        ("A" * 5001, "Valid question?"),
        ("Valid context", "B" * 501),
    ]
    for ctx, q in invalid_cases:
        try:
            qa.predict(ctx, q)
            return False
        except ValueError:
            pass
        except Exception:
            return False
    return True


def test_public_sample_and_contract():
    sample_file = SAMPLE_PATH if SAMPLE_PATH.exists() else PROJECT_ROOT / "public_sample.json"
    data = json.loads(sample_file.read_text(encoding="utf-8"))
    qa = QAInference()

    results_detail = []

    for item in data["items"]:
        sid = item["id"]
        ctx = item["context"]
        q = item["question"]

        t0 = time.perf_counter()
        res = qa.predict(ctx, q)
        latency = time.perf_counter() - t0

        ans = res["answer"]
        start = res["start"]
        score = res["score"]

        # 1. Invariant context and types
        if ans is not None:
            if not isinstance(ans, str) or not isinstance(start, int):
                return False, f"{sid}: Type error", []
            if ctx[start : start + len(ans)] != ans:
                return False, f"{sid}: Invariant context slice mismatch", []
        else:
            if start is not None:
                return False, f"{sid}: start is not None for null answer", []

        # 2. Score bounds [0, 1]
        if not (0.0 <= score <= 1.0):
            return False, f"{sid}: Score {score} out of [0, 1]", []

        results_detail.append({
            "id": sid,
            "answer": ans,
            "start": start,
            "score": score,
            "latency_ms": round(latency * 1000, 1),
        })

    return True, "Contract met on all 5 samples", results_detail


def test_cpu_device():
    qa = QAInference()
    param_devices = {p.device.type for p in qa.model.parameters()}
    return param_devices == {"cpu"}


def main():
    r1 = test_different_working_directory()
    print(f"1. Different working directory: {'PASS' if r1 else 'FAIL'}")

    r2 = test_input_validation()
    print(f"2. Input validation (ValueError): {'PASS' if r2 else 'FAIL'}")

    r3_ok, r3_msg, details = test_public_sample_and_contract()
    print(f"3 & 4. Offsets & scores contract: {'PASS' if r3_ok else 'FAIL'}")

    r5 = test_cpu_device()
    print(f"5. Model strictly on CPU: {'PASS' if r5 else 'FAIL'}")

    print("\nV2 Predictions & CPU Latency:")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
