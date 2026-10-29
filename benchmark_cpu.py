"""Measure 50 sequential SQuAD validation inferences on CPU."""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_DIR = 'weights/squad2-distilbert'
MAX_LENGTH = 384
THRESHOLD = json.loads(Path('logs/metrics.json').read_text(encoding='utf-8'))['no_answer_threshold']
raw = json.loads(Path('data/validation-v2.0.json').read_text(encoding='utf-8'))
examples = []
for article in raw['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            examples.append((qa['question'], paragraph['context']))
            if len(examples) == 50: break
        if len(examples) == 50: break
    if len(examples) == 50: break

torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR).eval()
latencies = []
with torch.no_grad():
    for question, context in examples:
        t0 = time.perf_counter()
        enc = tokenizer(question, context, truncation='only_second', max_length=MAX_LENGTH,
                        return_tensors='pt', return_offsets_mapping=True)
        offsets = enc.pop('offset_mapping')[0].tolist()
        out = model(**enc)
        sl, el = out.start_logits[0], out.end_logits[0]
        start = int(torch.argmax(sl)); end = int(torch.argmax(el))
        null_score = float(sl[0] + el[0])
        span_score = float(sl[start] + el[end])
        answer = context[offsets[start][0]:offsets[end][1]] if end >= start else ''
        if span_score - null_score <= THRESHOLD: answer = ''
        latencies.append(time.perf_counter() - t0)
elapsed = sum(latencies)
result = {'count': len(latencies), 'total_seconds': elapsed,
          'mean_seconds': elapsed / len(latencies), 'p95_seconds': sorted(latencies)[int(.95*(len(latencies)-1))],
          'within_150_seconds': elapsed <= 150, 'threshold': THRESHOLD}
Path('logs/cpu_benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
