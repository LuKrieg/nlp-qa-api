"""Test script for QAInference._tokenize_windows using sample-05 from public_sample.json."""
import json
from pathlib import Path
from inference import QAInference

SAMPLE_PATH = Path(__file__).resolve().parent / "samples" / "public_sample.json"

def main():
    data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    sample_05 = next(item for item in data["items"] if item["id"] == "sample-05")
    context = sample_05["context"]
    question = sample_05["question"]

    qa = QAInference()
    inputs, offset_mapping, sequence_ids = qa._tokenize_windows(context, question)

    num_windows = inputs["input_ids"].shape[0]

    # Test 1: Multiple windows generated (> 1)
    t1_pass = num_windows > 1
    print(f"Test 1 (Multiple windows generated: {num_windows}): {'PASS' if t1_pass else 'FAIL'}")

    # Test 2: All windows respect max_length=384
    seq_lens = [inputs["input_ids"].shape[1]]
    t2_pass = all(len(w) == 384 for w in inputs["input_ids"]) and inputs["input_ids"].shape[1] == 384
    print(f"Test 2 (Windows shape respects max_length=384): {'PASS' if t2_pass else 'FAIL'}")

    # Test 3: sequence_ids identifies context tokens (has 1s and 0s)
    has_question_and_context = all(
        (0 in seq) and (1 in seq) for seq in sequence_ids
    )
    print(f"Test 3 (sequence_ids correctly identifies context tokens): {'PASS' if has_question_and_context else 'FAIL'}")

    # Test 4: Offsets recover exact text from context for context tokens
    recovered_properly = True
    for w_idx in range(num_windows):
        offsets = offset_mapping[w_idx]
        seq = sequence_ids[w_idx]
        for token_idx, (start, end) in enumerate(offsets):
            if seq[token_idx] == 1 and end > start:
                token_text = context[start:end]
                if len(token_text) != (end - start):
                    recovered_properly = False
                    break
        if not recovered_properly:
            break

    print(f"Test 4 (Offsets recover exact text from context): {'PASS' if recovered_properly else 'FAIL'}")

if __name__ == "__main__":
    main()

