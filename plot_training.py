"""Create loss_curves.png from the Trainer log history."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

history = json.loads(Path('logs/train_history.json').read_text(encoding='utf-8'))
train = [(x['step'], x['loss']) for x in history if 'loss' in x and 'eval_loss' not in x]
valid = [(x['step'], x['eval_loss']) for x in history if 'eval_loss' in x]
if not train and not valid:
    raise SystemExit('No loss values found in logs/train_history.json')
plt.figure(figsize=(8, 4.5))
if train:
    plt.plot(*zip(*train), marker='o', label='Training loss')
if valid:
    plt.plot(*zip(*valid), marker='o', label='Validation loss')
plt.xlabel('Training step')
plt.ylabel('Loss')
plt.title('SQuAD 2.0 fine-tuning loss')
plt.grid(alpha=.25)
plt.legend()
plt.tight_layout()
Path('logs').mkdir(exist_ok=True)
plt.savefig('logs/loss_curves.png', dpi=160)
