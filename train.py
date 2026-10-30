"""Fine-tune encoder models (DistilBERT, BERT, RoBERTa) for extractive QA using PyTorch."""
import argparse
import csv
import json
import math
import os
import random
import string
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup

URLS = {
    'train': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
    'validation': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
}

def cli():
    p = argparse.ArgumentParser()
    # Support both --model-name and --model for backward compatibility
    p.add_argument('--model-name', '--model', dest='model_name', default='distilbert/distilbert-base-uncased')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--log-dir', default=None)
    p.add_argument('--train-limit', type=int, default=12000)
    p.add_argument('--eval-limit', type=int, default=3000)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--gradient-accumulation-steps', type=int, default=1)
    p.add_argument('--max-length', type=int, default=384)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def get_examples(split):
    path = Path('data') / f'{split}-v2.0.json'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        print(f'Downloading {split} split to {path}', flush=True)
        urllib.request.urlretrieve(URLS[split], path)
    data = json.loads(path.read_text(encoding='utf-8'))
    examples = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                answers = qa.get('answers', [])
                examples.append({'id': qa['id'], 'question': qa['question'],
                    'context': paragraph['context'],
                    'answers': [{'text': x['text'], 'answer_start': x['answer_start']} for x in answers]})
    return examples

def subset(items, limit, seed):
    if not limit or limit >= len(items): return items
    neg = [x for x in items if not x['answers']]
    pos = [x for x in items if x['answers']]
    nneg = min(len(neg), round(limit * len(neg) / len(items)))
    rng = random.Random(seed)
    selected = rng.sample(neg, nneg) + rng.sample(pos, limit - nneg)
    rng.shuffle(selected)
    return selected

def tokenize(examples, tokenizer, max_length, stride):
    result = tokenizer([x['question'].lstrip() for x in examples], [x['context'] for x in examples],
        truncation='only_second', max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length')
    features = []
    for fi, offsets in enumerate(result['offset_mapping']):
        ei = result['overflow_to_sample_mapping'][fi]
        seq = result.sequence_ids(fi)
        answer_list = examples[ei]['answers']
        start_pos = end_pos = 0
        if answer_list:
            answer = answer_list[0]
            astart, aend = answer['answer_start'], answer['answer_start'] + len(answer['text'])
            context_tokens = [j for j, sid in enumerate(seq) if sid == 1]
            if context_tokens and offsets[context_tokens[0]][0] <= astart and offsets[context_tokens[-1]][1] >= aend:
                lo, hi = context_tokens[0], context_tokens[-1]
                while lo <= hi and offsets[lo][0] <= astart: lo += 1
                start_pos = lo - 1
                while hi >= start_pos and offsets[hi][1] >= aend: hi -= 1
                end_pos = hi + 1
        features.append({'input_ids': result['input_ids'][fi], 'attention_mask': result['attention_mask'][fi],
            'start_positions': start_pos, 'end_positions': end_pos,
            'offsets': offsets, 'sequence_ids': seq, 'example_index': ei})
    return features

class Features(Dataset):
    def __init__(self, feats): self.feats = feats
    def __len__(self): return len(self.feats)
    def __getitem__(self, i):
        f = self.feats[i]
        return {k: torch.tensor(f[k], dtype=torch.long) for k in ('input_ids','attention_mask','start_positions','end_positions')}

def norm(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(w for w in s.split() if w not in {'a','an','the'})

def f1(pred, ref):
    p, r = norm(pred).split(), norm(ref).split()
    if not p or not r: return float(p == r)
    overlap = sum(min(p.count(w), r.count(w)) for w in set(p))
    return 2*overlap/(len(p)+len(r)) if overlap else 0.0

def evaluate(model, features, examples, device, seed=42):
    model.eval(); loader = DataLoader(Features(features), batch_size=8)
    by_ex = [[] for _ in examples]
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            sl, el = out.start_logits.cpu(), out.end_logits.cpu()
            for j in range(len(ids)):
                feat = features[cursor]; cursor += 1
                offsets = feat['offsets']; seq = feat['sequence_ids']
                valid = [k for k, sid in enumerate(seq) if sid == 1 and offsets[k][1] > offsets[k][0]]
                best = ('', -1e9)
                top_s = torch.topk(sl[j], min(20, sl.shape[1])).indices.tolist()
                top_e = torch.topk(el[j], min(20, el.shape[1])).indices.tolist()
                for s in top_s:
                    if s not in valid: continue
                    for e in top_e:
                        if e not in valid: continue
                        if e < s or e-s > 30: continue
                        score = float(sl[j,s] + el[j,e])
                        if score > best[1]:
                            cs, ce = offsets[s][0], offsets[e][1]
                            best = (examples[feat['example_index']]['context'][cs:ce], score)
                null = float(sl[j,0] + el[j,0])
                by_ex[feat['example_index']].append((best[0], best[1]-null))
    records = []
    for i, ex in enumerate(examples):
        pred, margin = max(by_ex[i], key=lambda x:x[1]) if by_ex[i] else ('', -1e9)
        refs = [a['text'] for a in ex['answers']] or ['']
        records.append({'id':ex['id'], 'pred':pred, 'margin':margin, 'refs':refs,
            'answerable':bool(ex['answers'])})

    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_point = len(indices) // 2
    cal_idx = indices[:split_point]
    eval_idx = indices[split_point:]

    cal_records = [records[i] for i in cal_idx] if cal_idx else records
    eval_records = [records[i] for i in eval_idx] if eval_idx else records

    candidates = sorted({r['margin'] for r in cal_records} | {0.0})
    threshold, best_f1 = 0.0, -1.0
    for th in candidates:
        scores = []
        for r in cal_records:
            answer = r['pred'] if r['margin'] > th else ''
            scores.append(max(f1(answer, gold) for gold in r['refs']))
        score = sum(scores) / max(1, len(scores))
        if score > best_f1:
            threshold, best_f1 = th, score

    groups = {'answerable': [], 'unanswerable': []}; rows = []
    for r in eval_records:
        answer = r['pred'] if r['margin'] > threshold else ''
        em = max(float(norm(answer) == norm(g)) for g in r['refs'])
        fs = max(f1(answer, g) for g in r['refs'])
        group = 'answerable' if r['answerable'] else 'unanswerable'
        groups[group].append((em, fs))
        rows.append({'id': r['id'], 'prediction': answer or 'null', 'references': json.dumps(r['refs']),
                     'answerable': r['answerable'], 'em': em, 'f1': fs})
    return threshold, groups, rows

def main():
    a = cli()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))

    model_slug = a.model_name.strip('/').split('/')[-1]
    outdir_path = Path(a.output_dir) if a.output_dir else Path('weights') / f'squad2-{model_slug}'
    logdir_path = Path(a.log_dir) if a.log_dir else Path('logs') / f'squad2-{model_slug}'
    logdir_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = f'GPU ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else 'CPU'
    print(f'Using device: {device_name} | Model: {a.model_name}', flush=True)

    train = subset(get_examples('train'), a.train_limit, a.seed)
    valid = subset(get_examples('validation'), a.eval_limit, a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model_name, use_fast=True)
    train_features = tokenize(train, tokenizer, a.max_length, a.stride)
    valid_features = tokenize(valid, tokenizer, a.max_length, a.stride)
    print(f'Train examples={len(train)} features={len(train_features)}; validation examples={len(valid)} features={len(valid_features)}', flush=True)

    model = AutoModelForQuestionAnswering.from_pretrained(a.model_name).to(device)
    loader = DataLoader(Features(train_features), batch_size=a.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=0.01)

    accum_steps = max(1, a.gradient_accumulation_steps)
    steps_per_epoch = math.ceil(len(loader) / accum_steps)
    total_update_steps = steps_per_epoch * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_update_steps * 0.1), total_update_steps)

    history = []
    started = time.time()
    global_update_step = 0

    for epoch in range(a.epochs):
        model.train()
        loss_sum = 0.0
        loss_count = 0
        opt.zero_grad()

        num_batches = len(loader)
        remainder = num_batches % accum_steps

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            raw_loss = out.loss

            # Scale loss according to the actual number of batches accumulated in the current window
            is_in_remainder = (remainder > 0 and batch_idx >= num_batches - remainder)
            current_accum = remainder if is_in_remainder else accum_steps

            loss = raw_loss / current_accum
            loss.backward()

            loss_sum += raw_loss.item()
            loss_count += 1

            is_accum_boundary = ((batch_idx + 1) % accum_steps == 0)
            is_epoch_end = (batch_idx + 1 == num_batches)

            if is_accum_boundary or is_epoch_end:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                global_update_step += 1

                if global_update_step % 100 == 0:
                    avg_loss = loss_sum / max(1, loss_count)
                    history.append({'step': global_update_step, 'loss': avg_loss, 'epoch': epoch + 1})
                    print(f'epoch={epoch+1} step={global_update_step} loss={avg_loss:.4f}', flush=True)
                    loss_sum = 0.0
                    loss_count = 0

        if loss_count > 0:
            avg_loss = loss_sum / loss_count
            history.append({'step': global_update_step, 'loss': avg_loss, 'epoch': epoch + 1})

        val_loader = DataLoader(Features(valid_features), batch_size=a.batch_size)
        model.eval()
        vl = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                vl.append(float(model(**batch).loss))
        eval_loss = sum(vl) / max(1, len(vl))
        history.append({'step': global_update_step, 'eval_loss': eval_loss, 'epoch': epoch + 1})
        print(f'epoch={epoch+1} validation_loss={eval_loss:.4f}', flush=True)

    threshold, groups, rows = evaluate(model, valid_features, valid, device, seed=a.seed)

    outdir_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir_path)
    tokenizer.save_pretrained(outdir_path)

    with (logdir_path / 'validation_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    (logdir_path / 'train_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    elapsed = time.time() - started
    metrics = {
        'model': a.model_name,
        'train_examples': len(train),
        'validation_examples': len(valid),
        'epochs': a.epochs,
        'learning_rate': a.learning_rate,
        'batch_size': a.batch_size,
        'gradient_accumulation_steps': accum_steps,
        'max_length': a.max_length,
        'doc_stride': a.stride,
        'training_seconds': elapsed,
        'no_answer_threshold': threshold,
    }
    for key, vals in groups.items():
        metrics[key] = {
            'count': len(vals),
            'EM': 100 * sum(x[0] for x in vals) / max(1, len(vals)),
            'F1': 100 * sum(x[1] for x in vals) / max(1, len(vals)),
        }
    (logdir_path / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == '__main__':
    main()
