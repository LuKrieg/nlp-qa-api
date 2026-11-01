# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `Kasane` |
| Members | Oliver, Camila, Enrique, Lucy |
| Repository URL | https://github.com/LuKrieg/nlp-qa-api |
| `submission` commit hash | The commit tagged `submission` (a report cannot contain the hash of its own commit). |
| Public service URL used at the dry run | `https://diffusion-briskly-craftily.ngrok-free.dev (fixed ngrok domain)` |
| AI tools used (which, and for what) | `ChatGPT (OpenAI): technical guidance, code development and debugging, assistance with DistilBERT training and CPU inference, FastAPI integration, testing, documentation, and deployment troubleshooting. Google Antigravity: AI-assisted development environment used for code implementation, review, and debugging.` |


## 1. Model choice and justification

- **Base checkpoint:** `distilbert/distilbert-base-uncased` (66,364,418 parameters). It is a general pre-trained encoder, not fine-tuned on SQuAD; we add a span-prediction head and fine-tune it ourselves.
- **Alternatives considered:** `roberta-base` and `microsoft/deberta-v3-base` usually score higher on SQuAD 2.0 but are larger and slower. We excluded them on expected CPU latency; we did not benchmark them.
- **Trade-off:** DistilBERT needs about 35–100 ms per 384-token window on CPU (measured on a Ryzen 7 9800X3D; a laptop will be slower), far below the 5 s cap. A larger encoder would add quality headroom on non-SQuAD text at a CPU cost we could not verify.

## 2. Training

| Item | Value |
|---|---|
| Training data | Full SQuAD 2.0 train split: 130,319 questions (131,754 windows), about 1/3 unanswerable. |
| Validation data | Full SQuAD 2.0 dev split: 11,873 questions (12,134 windows). |
| Epochs | 3 |
| Learning rate and schedule (warmup?) | 5e-5, linear decay, 10% warmup, AdamW (weight decay 0.01), gradient clipping 1.0 |
| Batch size | 32 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | Single GPU with bfloat16 autocast; 886 s for 3 epochs. Serving is CPU-only. |

- **Loss curve:** [logs/modelo-final-v3/loss.png](logs/modelo-final-v3/loss.png). Training loss fell from 5.7 to 0.65. Validation loss was 1.352, 1.336 and 1.467 after epochs 1–3: it rose in the last epoch, a sign of mild overfitting.
- **Validation results (full dev set, served threshold −1.0):** answerable EM 69.8%, correct `null` on unanswerable 58.8%. At the training-time threshold (−3.34): answerable EM 73.7%, F1 81.2%, unanswerable 49.6% (held-out half).

## 3. No-answer threshold

- **Decision:** we take the best span over all windows and compare `margin = best_span_score − null_score` (`null_score` = start and end logits at `[CLS]`, as in `train.py` and `inference.py`). The service returns `null` when `margin <= threshold`.
- **Value:** we swept the threshold over all dev margins. The score is flat between −3.3 and −1.0 (about 0.67 at 25% unanswerable, the hidden-test share). We chose **−1.0**: it loses about 0.003 against the optimum and gives more correct `null` answers, protecting against adversarial questions. It is stored in `metrics.json` next to the weights.
- **Too low or too high:** a low threshold answers unanswerable questions with unsupported spans; a high one returns `null` for answerable ones. On our 48-question set answerable EM goes from 0.83 at −2 to 0.75 at +3, while unanswerable accuracy goes from 0.67 to 0.92.

## 4. Serving architecture

- **Framework and server:** FastAPI + uvicorn, one worker, port 8000 (`api.py`).
- **Model loading:** a background thread loads the model once at start-up and runs a warm-up prediction. `/health` returns `503 {"status":"loading"}` until then, and `200 {"status":"ok"}` after.
- **Request flow:** `/predict` checks the token, validates the body (extra fields rejected), splits question and context into overlapping 384-token windows (stride 128), runs DistilBERT on CPU, keeps the best span per window (up to 30 tokens), picks the window with the highest margin and applies the threshold. Offsets map back to the original string, so `context[start:start+len(answer)] == answer` always holds. Questions over 96 tokens are clipped.
- **Access control:** `require_auth` runs before body validation. The token comes only from `QA_API_TOKEN`, compared with `hmac.compare_digest`; an unset or empty variable rejects every request with `401 {"error":"unauthorized"}`. `/health` is public.
- **Tunnel:** ngrok (`ngrok http 8000`) as primary and Cloudflare Quick Tunnel (`cloudflared tunnel --url http://localhost:8000`) as backup. `ngrok with a fixed domain (ngrok http --url=diffusion-briskly-craftily.ngrok-free.dev 127.0.0.1:8000) as primary and Cloudflare Quick Tunnel as backup`.
- **How to run** from a clean machine (details in `README.md`; weights are distributed outside git):

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements-serve.txt
export QA_API_TOKEN=<token> TEAM_NAME=<team>
./run.sh        # or run.ps1 on Windows; serves on 0.0.0.0:8000
python smoke_test.py http://localhost:8000 --extended
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` | `16` | 383 | Yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | `null` | Yes |
| sample-04 | `null` | `3 times` | 310 | No |
| sample-05 | `50,257` | `50,257` | 3157 | Yes |

sample-04 asks about BERT-base while the context reports LoRA results for GPT-3: a false premise using words from the context. The model returns a plausible span with margin 1.82, above the threshold. Correct answers elsewhere have margins of 6.4–9.2, so a threshold near 2 would fix this item, but we did not tune to one item: on SQuAD dev it costs about 0.025 of team score.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 253 MiB (`model.safetensors`, 265 MB) |
| Mean latency per `/predict` request on your laptop (CPU) | 96 ms mean, 327 ms max over the 5 public samples through local HTTP (the 3-window sample takes about 320 ms). Measured on the development machine, not yet on the serving laptop. |
 | Laptop CPU and RAM | **Option 1 (ideal)  — :** Intel Core Ultra 7 255HX, 31.43 GiB usable physical RAM (approximately 32 GB installed). **Option 2 — Development machine:** AMD Ryzen 7 9800X3D, 8 cores, 24 GB RAM allocated to WSL. Final serving machine to be confirmed before evaluation.  |

## 7. Problems encountered and how you fixed them

1. **Problem:** v2 (trained on 40,000 of 130,319 questions) answered the contract's example "How long is the Lempa River?" with "longest" and scored 0.58 on our 48-question set. **Cause:** too little training data. **Fix:** retrained on the full SQuAD 2.0 train set and re-calibrated the threshold; the set score rose to 0.77. **Commit:** `bb13389`
2. **Problem:** a 500-character question made the API return `500`. **Cause:** the tokenizer fails when the question leaves no room for the context. **Fix:** questions are clipped to 96 tokens (`inference.py`). **Commit:** `FILL: commit hash`
3. **Problem:** an invalid JSON body without token returned `422` instead of `401`, and an unset `QA_API_TOKEN` accepted the header `Bearer `. **Cause:** FastAPI parses the body before dependencies, and the expected token was empty. **Fix:** authentication is answered first and fails closed. **Commit:** `FILL: commit hash`
4. **Problem:** the repository held a duplicated `.git - Copy/` folder, Windows-only paths and a hard-coded default test token in the tests. **Cause:** files copied between machines. **Fix:** removed in a new commit (history preserved); tests now require `QA_API_TOKEN` and use relative paths. **Commit:** `FILL: commit hash`

## 8. Decoder investigation

Decoders are not allowed for the implementation: they are slow on a laptop CPU, their generated answers do not fit exact-match evaluation, and the time frame does not allow doing both well. We analysed how a decoder (GPT-style, Radford et al., 2018) or an encoder–decoder would address the problems we found.

1. **False-premise unanswerable question (sample-04).** Our model returns "3 times" with margin 1.82 (measured), while correct answers have margins of 6.4–9.2; the threshold was calibrated on SQuAD dev (about 50% unanswerable) but the hidden set is about 25% and not from SQuAD (Rajpurkar et al., 2018). A generative model could check the premise and emit an explicit "no answer", but abstention still needs held-out calibration.
2. **Partial or non-span answers.** For "How long is the Lempa River?" we return "422 km", not "about 422 km" (measured), and spans cannot answer yes/no questions or merge facts across sentences. A decoder can generate "about 422 km" or "yes", but not necessarily verbatim, so `start` would need post-hoc alignment and exact match could drop.
3. **Cost.** We serve 66.4M parameters (253 MiB) with one forward pass per 384-token window (measured). A useful decoder has roughly 350M–1B parameters and one pass per generated token (estimate). LoRA (Hu et al., 2021) reduces fine-tuning memory, not inference latency against the 5 s cap.

## 9. Known limitations

- False-premise unanswerable questions are still answered (sample-04; 4 of 6 in our own set).
- Answers can be partial spans ("422 km" instead of "about 422 km").
- Answers are one contiguous span: no yes/no, generated or merged answers.
- Quality on non-SQuAD text is estimated only from our 48-question set and the 5 public samples.
- An answer crossing a window boundary can be cut; long contexts cost more time (about 320 ms for 3 windows).
- One worker handles one request at a time, and the service is reachable only while the laptop, server and tunnel run.
