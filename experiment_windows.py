"""Experiment to compare multi-window aggregation strategies on SQuAD 2.0 validation.

Method A: max_w (best_score_w - null_score_w) [local window margin]
Method B: max_w(best_score_w) - min_w(null_score_w) [global span minus min null]
"""
import argparse
import json
import random
import string
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_DIR = Path('weights/modelo-final-v1')
DATA_PATH = Path('data/validation-v2.0.json')


def cli():
    p = argparse.ArgumentParser(description="Compare multi-window aggregation methods.")
    p.add_argument('--limit', type=int, default=200, help='Number of validation examples (0 for all).')
    p.add_argument('--multi-window-only', action='store_true', help='Filter only examples that yield >1 window.')
    p.add_argument('--seed', type=int, default=42, help='Random seed.')
    p.add_argument('--max-length', type=int, default=384)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=16)
    return p.parse_args()


def load_all_raw_examples():
    raw = json.loads(DATA_PATH.read_text(encoding='utf-8'))
    examples = []
    for article in raw['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                examples.append({
                    'id': qa['id'],
                    'question': qa['question'],
                    'context': context,
                    'answers': [{'text': x['text'], 'answer_start': x['answer_start']} for x in qa.get('answers', [])],
                })
    return examples


def select_examples(raw_examples, tokenizer, max_length, stride, limit, multi_window_only, seed):
    if multi_window_only:
        # Pre-filter examples that produce > 1 window
        multi_examples = []
        for ex in raw_examples:
            # Fast check with return_overflowing_tokens
            enc = tokenizer(
                ex['question'].lstrip(),
                ex['context'],
                truncation='only_second',
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
            )
            if enc['input_ids'].shape[0] > 1 if hasattr(enc['input_ids'], 'shape') else len(enc['input_ids']) > 1:
                multi_examples.append(ex)

        print(f"Found {len(multi_examples)} multi-window examples in validation set.")
        target_pool = multi_examples
    else:
        target_pool = raw_examples

    if limit and limit < len(target_pool):
        neg = [x for x in target_pool if not x['answers']]
        pos = [x for x in target_pool if x['answers']]
        nneg = min(len(neg), round(limit * len(neg) / max(1, len(target_pool))))
        rng = random.Random(seed)
        selected = rng.sample(neg, nneg) + rng.sample(pos, limit - nneg)
        rng.shuffle(selected)
        return selected
    return target_pool


class Features(Dataset):
    def __init__(self, feats):
        self.feats = feats

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        f = self.feats[i]
        return {
            'input_ids': torch.tensor(f['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(f['attention_mask'], dtype=torch.long),
        }


def norm(s):
    s = ''.join(c for c in (s or '').lower() if c not in string.punctuation)
    return ' '.join(w for w in s.split() if w not in {'a', 'an', 'the'})


def f1(pred, ref):
    p, r = norm(pred).split(), norm(ref).split()
    if not p or not r:
        return float(p == r)
    overlap = sum(min(p.count(w), r.count(w)) for w in set(p))
    return 2 * overlap / (len(p) + len(r)) if overlap else 0.0


def compute_metrics(records, threshold):
    groups = {
        'all': [],
        'answerable': [],
        'unanswerable': [],
        'multi_window': [],
        'single_window': [],
    }
    for r in records:
        ans = r['pred'] if r['margin'] > threshold else ''
        em = max(float(norm(ans) == norm(g)) for g in r['refs'])
        fs = max(f1(ans, g) for g in r['refs'])

        pair = (em, fs)
        groups['all'].append(pair)
        if r['answerable']:
            groups['answerable'].append(pair)
        else:
            groups['unanswerable'].append(pair)

        if r['num_windows'] > 1:
            groups['multi_window'].append(pair)
        else:
            groups['single_window'].append(pair)

    res = {}
    for k, v in groups.items():
        count = len(v)
        res[k] = {
            'count': count,
            'em': (sum(x[0] for x in v) / count * 100) if count else 0.0,
            'f1': (sum(x[1] for x in v) / count * 100) if count else 0.0,
        }
    return res


def calibrate_threshold(cal_records):
    candidates = sorted({r['margin'] for r in cal_records} | {0.0})
    best_th, best_f1 = 0.0, -1.0
    for th in candidates:
        scores = []
        for r in cal_records:
            ans = r['pred'] if r['margin'] > th else ''
            scores.append(max(f1(ans, g) for g in r['refs']))
        avg_f1 = sum(scores) / max(1, len(scores))
        if avg_f1 > best_f1:
            best_th, best_f1 = th, avg_f1
    return best_th


def main():
    args = cli()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device for inference: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    raw_examples = load_all_raw_examples()
    examples = select_examples(raw_examples, tokenizer, args.max_length, args.stride, args.limit, args.multi_window_only, args.seed)
    print(f"Total examples selected for experiment: {len(examples)}")

    # Tokenize with sliding windows
    tokenized = tokenizer(
        [x['question'].lstrip() for x in examples],
        [x['context'] for x in examples],
        truncation='only_second',
        max_length=args.max_length,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )

    features = []
    offsets_by_feature = tokenized['offset_mapping']
    overflow_map = tokenized['overflow_to_sample_mapping']

    for fi, offsets in enumerate(offsets_by_feature):
        ei = overflow_map[fi]
        seq = tokenized.sequence_ids(fi)
        features.append({
            'input_ids': tokenized['input_ids'][fi],
            'attention_mask': tokenized['attention_mask'][fi],
            'offsets': offsets,
            'sequence_ids': seq,
            'example_index': ei,
        })

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    loader = DataLoader(Features(features), batch_size=args.batch_size, shuffle=False)

    # 1. Compute logits once for all windows
    all_start_logits, all_end_logits = [], []
    with torch.inference_mode():
        for batch in loader:
            b_ids = batch['input_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            out = model(input_ids=b_ids, attention_mask=b_mask)
            all_start_logits.append(out.start_logits.cpu())
            all_end_logits.append(out.end_logits.cpu())

    sl_all = torch.cat(all_start_logits, dim=0)
    el_all = torch.cat(all_end_logits, dim=0)

    # Organize windows by example
    windows_by_ex = [[] for _ in examples]
    for fi, feat in enumerate(features):
        ei = feat['example_index']
        offsets = feat['offsets']
        seq = feat['sequence_ids']
        sl = sl_all[fi]
        el = el_all[fi]

        valid = [k for k, sid in enumerate(seq) if sid == 1 and offsets[k][1] > offsets[k][0]]
        best_span = ('', -1e9)

        if valid:
            top_s = torch.topk(sl, min(20, sl.shape[0])).indices.tolist()
            top_e = torch.topk(el, min(20, el.shape[0])).indices.tolist()
            for s in top_s:
                if s not in valid:
                    continue
                for e in top_e:
                    if e not in valid or e < s or (e - s) > 30:
                        continue
                    score = float(sl[s] + el[e])
                    if score > best_span[1]:
                        cs, ce = offsets[s][0], offsets[e][1]
                        best_span = (examples[ei]['context'][cs:ce], score)

        null_score = float(sl[0] + el[0])
        windows_by_ex[ei].append({
            'best_span': best_span[0],
            'best_score': best_span[1],
            'null_score': null_score,
        })

    # Build predictions under Method A and Method B
    records_a, records_b = [], []
    for i, ex in enumerate(examples):
        wins = windows_by_ex[i]
        num_w = len(wins)
        refs = [a['text'] for a in ex['answers']] or ['']
        ans_bool = bool(ex['answers'])

        # Method A: max_w (best_score_w - null_score_w)
        cand_a = [(w['best_span'], w['best_score'] - w['null_score']) for w in wins]
        best_a = max(cand_a, key=lambda x: x[1]) if cand_a else ('', -1e9)
        records_a.append({
            'id': ex['id'],
            'pred': best_a[0],
            'margin': best_a[1],
            'refs': refs,
            'answerable': ans_bool,
            'num_windows': num_w,
        })

        # Method B: max_w(best_score_w) - min_w(null_score_w)
        best_span_b = max(wins, key=lambda w: w['best_score'])['best_span'] if wins else ''
        max_best_score = max(w['best_score'] for w in wins) if wins else -1e9
        min_null_score = min(w['null_score'] for w in wins) if wins else 0.0
        margin_b = max_best_score - min_null_score
        records_b.append({
            'id': ex['id'],
            'pred': best_span_b,
            'margin': margin_b,
            'refs': refs,
            'answerable': ans_bool,
            'num_windows': num_w,
        })

    # 2. Disjoint Calibration (50%) and Evaluation (50%)
    indices = list(range(len(examples)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    mid = len(indices) // 2
    cal_idx, eval_idx = indices[:mid], indices[mid:]

    cal_a = [records_a[i] for i in cal_idx]
    eval_a = [records_a[i] for i in eval_idx]

    cal_b = [records_b[i] for i in cal_idx]
    eval_b = [records_b[i] for i in eval_idx]

    # 3. Independent threshold search
    th_a = calibrate_threshold(cal_a)
    th_b = calibrate_threshold(cal_b)

    # 4. Evaluate metrics
    metrics_a = compute_metrics(eval_a, th_a)
    metrics_b = compute_metrics(eval_b, th_b)

    output = {
        'examples_total': len(examples),
        'eval_subset_size': len(eval_idx),
        'multi_window_in_eval': sum(1 for r in eval_a if r['num_windows'] > 1),
        'method_a': {
            'description': 'max_w(best_score_w - null_score_w)',
            'threshold': th_a,
            'metrics': metrics_a,
        },
        'method_b': {
            'description': 'max_w(best_score_w) - min_w(null_score_w)',
            'threshold': th_b,
            'metrics': metrics_b,
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
