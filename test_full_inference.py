"""Full evaluation of QAInference.predict() on all 5 items of public_sample.json."""
import json
import string
import time
from pathlib import Path
from inference import QAInference

SAMPLE_PATH = Path(r"C:\Users\andre\Desktop\Semana 10 y 11 - NLP + APIs\public_sample.json")

def norm(s):
    s = ''.join(c for c in (s or '').lower() if c not in string.punctuation)
    return ' '.join(w for w in s.split() if w not in {'a', 'an', 'the'})

def compute_f1(pred, ref):
    if pred is None and (ref is None or ref == ""):
        return 1.0
    if pred is None or (ref is None or ref == ""):
        return 0.0
    p, r = norm(pred).split(), norm(ref).split()
    if not p or not r:
        return float(p == r)
    overlap = sum(min(p.count(w), r.count(w)) for w in set(p))
    return 2 * overlap / (len(p) + len(r)) if overlap else 0.0

def compute_em(pred, ref):
    if pred is None and (ref is None or ref == ""):
        return 1.0
    if pred is None or (ref is None or ref == ""):
        return 0.0
    return float(norm(pred) == norm(ref))

def main():
    sample_file = SAMPLE_PATH if SAMPLE_PATH.exists() else Path("public_sample.json")
    data = json.loads(sample_file.read_text(encoding="utf-8"))
    
    qa = QAInference()
    print(f"Loaded QAInference. Calibrated threshold: {qa.no_answer_threshold:.4f}\n")

    items = data["items"]
    all_passed = True

    print(f"{'ID':<11} | {'Status':<6} | {'EM':<4} | {'F1':<4} | {'Latency':<8} | {'Pred Answer':<20} | {'Expected'}")
    print("-" * 85)

    for item in items:
        sid = item["id"]
        context = item["context"]
        question = item["question"]
        is_impossible = item.get("is_impossible", False)
        gold_answers = [a["text"] for a in item.get("answers", [])]
        gold_starts = [a["start"] for a in item.get("answers", [])]

        t0 = time.perf_counter()
        pred = qa.predict(context, question)
        latency = time.perf_counter() - t0

        answer = pred["answer"]
        start = pred["start"]

        # Check invariant context[start:start+len(answer)] == answer
        if answer is not None:
            invariant_ok = (start is not None) and (context[start : start + len(answer)] == answer)
        else:
            invariant_ok = (start is None)

        # Check answer correctness
        if is_impossible:
            em = 1.0 if answer is None else 0.0
            f1 = 1.0 if answer is None else 0.0
            correct = (answer is None and start is None and invariant_ok)
            expected_repr = "null"
        else:
            em = max((compute_em(answer, g) for g in gold_answers), default=0.0)
            f1 = max((compute_f1(answer, g) for g in gold_answers), default=0.0)
            start_ok = (start in gold_starts) if gold_starts else True
            correct = (em == 1.0 and invariant_ok and start_ok)
            expected_repr = gold_answers[0] if gold_answers else ""

        status = "PASS" if correct else "FAIL"
        if not correct:
            all_passed = False

        pred_disp = str(answer) if answer is not None else "null"
        print(f"{sid:<11} | {status:<6} | {em:<4.1f} | {f1:<4.1f} | {latency*1000:<6.1f}ms | {pred_disp:<20} | {expected_repr}")

    print("-" * 85)
    print(f"\nOverall Test Status: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")

if __name__ == "__main__":
    main()

