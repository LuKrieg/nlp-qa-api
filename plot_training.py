"""Create loss curves from train history JSON."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

def cli():
    p = argparse.ArgumentParser(description="Plot loss curves from training history.")
    p.add_argument('--input', default=str(BASE_DIR / 'logs' / 'modelo-final-v2' / 'train_history.json'), help='Path to train history JSON file.')
    p.add_argument('--output', default=str(BASE_DIR / 'out' / 'loss_curves.png'), help='Path to output loss plot image.')
    return p.parse_args()

def main():
    args = cli()
    input_path = Path(args.input)
    output_path = Path(args.output)

    history = json.loads(input_path.read_text(encoding='utf-8'))
    train = [(x['step'], x['loss']) for x in history if 'loss' in x and 'eval_loss' not in x]
    valid = [(x['step'], x['eval_loss']) for x in history if 'eval_loss' in x]
    if not train and not valid:
        raise SystemExit(f'No loss values found in {input_path}')
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)

if __name__ == '__main__':
    main()
