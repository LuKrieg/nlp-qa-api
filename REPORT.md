# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `TBD` |
| Members | `TBD` |
| Repository URL | https://github.com/jrivera937/nlp-qa-api |
| `submission` commit hash | `TBD` |
| Public service URL used at the dry run | `TBD` |
| AI tools used (which, and for what) | OpenAI Codex: repository scaffolding and training-script drafting; team must verify and disclose all tools used. |

## 1. Model choice and justification

- **Base checkpoint:** `distilbert/distilbert-base-uncased`. It is a compact encoder checkpoint intended for general language understanding, not a SQuAD-fine-tuned QA checkpoint. The training script adds a QA head and fine-tunes it on SQuAD 2.0. DistilBERT is selected as a CPU-serving candidate; the required 50-question/150-second limit still needs measurement on the team's laptop.
- **Alternatives considered:** `roberta-base` and `microsoft/deberta-v3-base` are larger candidates with potentially stronger QA quality, but greater CPU latency and memory costs. No measured comparison has been run for this project.
- **Trade-off:** The smaller DistilBERT encoder favors model footprint and CPU throughput. The pilot artifact processed 50 official validation questions in 3.50 seconds on CPU (model inference only); the API overhead and public sample set still need measurement.

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | Official SQuAD 2.0 train split; pilot run used a seeded, proportion-preserving subset of 64 examples. |
| Validation data (what you validated on, how many examples) | Official SQuAD 2.0 validation split; pilot run used a seeded subset of 32 examples. |
| Epochs | 1 (pilot run; configurable) |
| Learning rate and schedule (warmup?) | 3e-5, linear Trainer schedule, 10% warmup |
| Batch size | 8 |
| `max_seq_length` / `doc_stride` | 128 / 32 (pilot run) |
| Hardware and training time | CPU-only; 30.06 seconds. Record exact CPU/RAM before final submission. |

- **Loss curve:** [logs/loss_curves.png](logs/loss_curves.png), generated from the pilot run.
- **Validation results:** The 32-example pilot produced answerable EM/F1 0.00/0.00 (16 examples) and unanswerable EM/F1 100.00/100.00 (16 examples). This is a smoke-run result, not a representative quality estimate; rerun with a larger sample before submission.

Create the environment with `python -m venv .venv`, install with `.venv/Scripts/python -m pip install -r requirements.txt`, then run `.venv/Scripts/python train.py --train-limit 12000 --eval-limit 3000 --epochs 2` and `.venv/Scripts/python plot_training.py`. The script downloads official SQuAD 2.0 JSON splits into `data/` and writes model files under `weights/` and logs under `logs/`. It uses a PyTorch loop because Windows Application Control blocked pandas, which the Hugging Face Trainer/datasets stack imports in this environment.

## 3. No-answer threshold

The QA head scores candidate answer spans and the CLS null span. The validation script computes `best_non_null_score - null_score`; it chooses the threshold with the highest validation F1. The pilot threshold was `-0.8336`; recalibrate on a much larger validation subset before submission. A threshold that is too high returns too many nulls; a threshold that is too low returns unsupported spans for unanswerable questions.

## 4. Serving architecture

- **Framework and server:** `TBD (not present in the starter repository)`.
- **Model loading:** `TBD`.
- **Request flow:** `TBD`.
- **Access control:** `TBD`.
- **Tunnel:** `TBD`.
- **How to run from a clean machine:** `TBD: document the verified API setup, dependencies, and environment variables.`

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `TBD: from assignment sample` | `TBD` | `TBD` | `TBD` |
| sample-02 | `TBD: from assignment sample` | `TBD` | `TBD` | `TBD` |
| sample-03 | `null` | `TBD` | `TBD` | `TBD` |
| sample-04 | `null` | `TBD` | `TBD` | `TBD` |
| sample-05 | `TBD: from assignment sample` | `TBD` | `TBD` | `TBD` |

No public sample set was present in the repository, so no sample answers or correctness results are claimed.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | `TBD: inspect saved model` |
| Size on disk | 254.07 MB |
| Mean latency per `/predict` request on your laptop (CPU) | 0.070 seconds per local model call for 50 official validation questions (3.50 seconds total); this measures model inference, not API overhead. |
| Laptop CPU and RAM | `TBD` |

## 7. Problems encountered and how you fixed them

1. **Problem:** `TBD: record a real implementation problem.` **Cause:** `TBD` **Fix:** `TBD` **Commit:** `TBD`
2. **Problem:** `TBD: record a second real implementation problem.` **Cause:** `TBD` **Fix:** `TBD` **Commit:** `TBD`

## 8. Decoder investigation

1. **Problem:** Extractive QA can only return text that appears as one contiguous context span. A GPT-style decoder could generate a concise answer or normalize a discontinuous fact, but could also produce unsupported text; grounding and answerability checks would be needed. It would cost more CPU time and memory, so the 150-second limit would need a fresh benchmark.
2. **Problem:** The encoder QA head's null threshold may confuse answerable and unanswerable questions. A decoder could be prompted to abstain explicitly, but abstention is not guaranteed and requires calibration against the same held-out set. Generation would add latency and model-size costs.

## 9. Known limitations

The pilot completed training and validation but used only 64 train and 32 validation examples with 128-token sequences; its answerable score is not useful as a final model-quality claim. The repository still has no serving API, so `/predict`, authentication, tunnel, and public sample results remain to be implemented or supplied by the team. The 50-question CPU benchmark measured model inference only, not HTTP/API overhead. The default training run is larger and requires substantial CPU time. Weights are local and ignored by Git; upload them to a GitHub Release or shared folder before submission.
