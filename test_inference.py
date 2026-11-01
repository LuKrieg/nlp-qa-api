"""Test QAInference.predict on answerable samples from public_sample.json."""
import json
import string
from pathlib import Path
from inference import QAInference

SAMPLE_PATH = Path(__file__).resolve().parent / "samples" / "public_sample.json"

def norm(s):
    s = ''.join(c for c in (s or '').lower() if c not in string.punctuation)
    return ' '.join(w for w in s.split() if w not in {'a', 'an', 'the'})

def compute_f1(pred, ref):
    p, r = norm(pred).split(), norm(ref).split()
    if not p or not r:
        return float(p == r)
    overlap = sum(min(p.count(w), r.count(w)) for w in set(p))
    return 2 * overlap / (len(p) + len(r)) if overlap else 0.0

def compute_em(pred, ref):
    return float(norm(pred) == norm(ref))

def main():
    data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    
    target_ids = {"sample-01", "sample-02", "sample-05"}
    samples = [item for item in data["items"] if item["id"] in target_ids]

    qa = QAInference()
    
    results = []
    for item in samples:
        sid = item["id"]
        context = item["context"]
        question = item["question"]
        expected_refs = [a["text"] for a in item["answers"]]
        expected_starts = [a["start"] for a in item["answers"]]
        
        pred = qa.predict(context, question)
        pred_answer = pred["answer"]
        pred_start = pred["start"]
        
        # Check verbatim context match
        if pred_answer is not None and pred_start is not None:
            slice_match = (context[pred_start : pred_start + len(pred_answer)] == pred_answer)
        else:
            slice_match = False
            
        em = max((compute_em(pred_answer, ref) for ref in expected_refs), default=0.0)
        f1_score = max((compute_f1(pred_answer, ref) for ref in expected_refs), default=0.0)
        
        results.append({
            "id": sid,
            "expected": expected_refs[0] if expected_refs else "null",
            "expected_start": expected_starts[0] if expected_starts else None,
            "predicted": pred_answer,
            "pred_start": pred_start,
            "slice_match": slice_match,
            "em": em,
            "f1": f1_score,
            "score": pred["score"]
        })

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

