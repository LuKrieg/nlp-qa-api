# Extractive QA training

Windows CPU workflow for fine-tuning DistilBERT on SQuAD 2.0. The project downloads the official JSON splits into `data/` and the base checkpoint from Hugging Face on first run. Trained weights are written under `weights/`; both folders are excluded by `.gitignore`.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the `py` launcher is unavailable, use an installed Python executable in the first command.

## Train

The default CPU run uses 12,000 training examples and 3,000 validation examples, 2 epochs, batch size 8, sequence length 384, and stride 128:

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe plot_training.py
```

For all SQuAD 2.0 examples, use `--train-limit 0 --eval-limit 0`. If CPU time is constrained, set smaller limits. `train.py` writes weights to `weights/squad2-distilbert/` and metrics, predictions, and history to `logs/`.

## CPU benchmark

After training, run 50 sequential model inferences on official validation questions:

```powershell
.\.venv\Scripts\python.exe benchmark_cpu.py
```

The benchmark result is saved to `logs/cpu_benchmark.json`. It measures model inference, not HTTP server overhead. A serving API is not included yet.

## Current pilot

The checked-in logs/curve describe a compact pilot (64 train examples, 32 validation examples, 1 epoch, 128-token sequence length). It verifies the pipeline but is not a final quality run. See [REPORT.md](REPORT.md) for metrics and known gaps. Do not commit model weights; distribute the ignored `weights/squad2-distilbert/` directory through the assignment's approved Release or shared folder.
