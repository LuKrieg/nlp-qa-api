"""Measure 50 sequential SQuAD 2.0 validation inferences on CPU with the served pipeline.

Uses inference.QAInference (same model folder, sliding windows and no-answer threshold
as the API; switch models with QA_MODEL_NAME). Requires data/validation-v2.0.json
(downloaded by train.py). Measures model inference only, not HTTP overhead.

Usage: python benchmark_cpu.py   ->  writes out/cpu_benchmark.json
"""
import json
import time
from pathlib import Path

from inference import MODEL_NAME, QAInference

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'validation-v2.0.json'
OUTPUT_PATH = BASE_DIR / 'out' / 'cpu_benchmark.json'
COUNT = 50

if not DATA_PATH.exists():
    raise SystemExit(f'{DATA_PATH} not found. Download the SQuAD 2.0 dev set there first.')

raw = json.loads(DATA_PATH.read_text(encoding='utf-8'))
examples = [(qa['question'], paragraph['context'])
            for article in raw['data']
            for paragraph in article['paragraphs']
            for qa in paragraph['qas']][:COUNT]

qa_model = QAInference()
qa_model.predict('Warm-up context sentence.', 'What is this?')

latencies = []
for question, context in examples:
    t0 = time.perf_counter()
    qa_model.predict(context, question)
    latencies.append(time.perf_counter() - t0)

elapsed = sum(latencies)
result = {
    'model': MODEL_NAME,
    'count': len(latencies),
    'total_seconds': elapsed,
    'mean_seconds': elapsed / len(latencies),
    'p95_seconds': sorted(latencies)[int(.95 * (len(latencies) - 1))],
    'max_seconds': max(latencies),
    'within_150_seconds': elapsed <= 150,
    'threshold': qa_model.no_answer_threshold,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
