"""Fine-tune DistilBERT for extractive QA using only PyTorch and stdlib JSON."""
import argparse
import csv
import json
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
    p.add_argument('--model', default='distilbert/distilbert-base-uncased')
    p.add_argument('--output-dir', default='weights/squad2-distilbert')
    p.add_argument('--log-dir', default='logs')
    p.add_argument('--train-limit', type=int, default=12000)
    p.add_argument('--eval-limit', type=int, default=3000)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--batch-size', type=int, default=8)
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
            'offsets': offsets, 'example_index': ei})
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

def evaluate(model, features, examples, device):
    model.eval(); loader = DataLoader(Features(features), batch_size=8)
    by_ex = [[] for _ in examples]
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            sl, el = out.start_logits.cpu(), out.end_logits.cpu()
            for j in range(len(ids)):
                feat = features[cursor]; cursor += 1; offsets = feat['offsets']
                # Only context tokens (nonzero offsets); CLS at index 0 is the null answer.
                valid = [k for k, (a,b) in enumerate(offsets) if b > a]
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
    # Pick null threshold by best validation F1; ties prefer the higher (more conservative) threshold.
    candidates = sorted({r['margin'] for r in records} | {0.0})
    threshold, best_f1 = 0.0, -1.0
    for th in candidates:
        scores=[]
        for r in records:
            answer = r['pred'] if r['margin'] > th else ''
            scores.append(max(f1(answer, gold) for gold in r['refs']))
        score=sum(scores)/max(1,len(scores))
        if score > best_f1: threshold,best_f1=th,score
    groups={'answerable':[],'unanswerable':[]}; rows=[]
    for r in records:
        answer=r['pred'] if r['margin'] > threshold else ''
        em=max(float(norm(answer)==norm(g)) for g in r['refs'])
        fs=max(f1(answer,g) for g in r['refs'])
        group='answerable' if r['answerable'] else 'unanswerable'
        groups[group].append((em,fs))
        rows.append({'id':r['id'],'prediction':answer or 'null','references':json.dumps(r['refs']),
                     'answerable':r['answerable'],'em':em,'f1':fs})
    return threshold,groups,rows

def main():
    a=cli(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    logdir=Path(a.log_dir); logdir.mkdir(exist_ok=True)
    train=subset(get_examples('train'),a.train_limit,a.seed)
    valid=subset(get_examples('validation'),a.eval_limit,a.seed)
    tokenizer=AutoTokenizer.from_pretrained(a.model, use_fast=True)
    train_features=tokenize(train,tokenizer,a.max_length,a.stride)
    valid_features=tokenize(valid,tokenizer,a.max_length,a.stride)
    print(f'Train examples={len(train)} features={len(train_features)}; validation examples={len(valid)} features={len(valid_features)}',flush=True)
    device=torch.device('cpu')
    model=AutoModelForQuestionAnswering.from_pretrained(a.model).to(device)
    loader=DataLoader(Features(train_features),batch_size=a.batch_size,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=0.01)
    total=len(loader)*a.epochs; sched=get_linear_schedule_with_warmup(opt,int(total*.1),total)
    history=[]; started=time.time(); step=0
    for epoch in range(a.epochs):
        model.train(); loss_sum=0; loss_count=0
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            out=model(**batch); loss=out.loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step(); opt.zero_grad(); step+=1; loss_sum+=loss.item(); loss_count+=1
            if step%100==0:
                history.append({'step':step,'loss':loss_sum/loss_count,'epoch':epoch+1}); loss_sum=0; loss_count=0
                print(f'epoch={epoch+1} step={step} loss={history[-1]["loss"]:.4f}',flush=True)
        if loss_count:
            history.append({'step':step,'loss':loss_sum/loss_count,'epoch':epoch+1})
        val_loader=DataLoader(Features(valid_features),batch_size=a.batch_size)
        model.eval(); vl=[]
        with torch.no_grad():
            for batch in val_loader:
                batch={k:v.to(device) for k,v in batch.items()}; vl.append(float(model(**batch).loss))
        history.append({'step':step,'eval_loss':sum(vl)/max(1,len(vl)),'epoch':epoch+1})
        print(f'epoch={epoch+1} validation_loss={history[-1]["eval_loss"]:.4f}',flush=True)
    threshold,groups,rows=evaluate(model,valid_features,valid,device)
    outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(outdir); tokenizer.save_pretrained(outdir)
    with (logdir/'validation_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (logdir/'train_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    elapsed=time.time()-started
    metrics={'model':a.model,'train_examples':len(train),'validation_examples':len(valid),'epochs':a.epochs,
        'learning_rate':a.learning_rate,'batch_size':a.batch_size,'max_length':a.max_length,'doc_stride':a.stride,
        'training_seconds':elapsed,'no_answer_threshold':threshold}
    for key,vals in groups.items():
        metrics[key]={'count':len(vals),'EM':100*sum(x[0] for x in vals)/max(1,len(vals)),
                      'F1':100*sum(x[1] for x in vals)/max(1,len(vals))}
    (logdir/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__': main()
