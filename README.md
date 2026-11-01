# Extractive QA as a Service

A DistilBERT model fine-tuned on SQuAD 2.0, served by FastAPI on the team laptop (CPU only, port 8000) and exposed to the instructor through a public tunnel. Given a `context` and a `question`, the API returns the answer span copied from the context, or `null` when the context does not contain the answer.

**New here? Start with [GUIA_PASO_A_PASO.md](GUIA_PASO_A_PASO.md)** (Spanish): start the API, open the tunnel and run the 5 curl tests, as in the instructor's guide.

Training runs outside this repository's serving setup; this repo holds the training code, the final training logs (`logs/modelo-final-v*/`), and the serving code. Model weights are never committed.

## API contract (summary)

The authoritative spec is [`qa_api_contract.yaml`](qa_api_contract.yaml).

| Endpoint | Auth | Success |
|---|---|---|
| `GET /health` | public | `200 {"status": "ok"}` when the model is loaded, `503 {"status": "loading"}` while loading |
| `GET /info` | Bearer | `200 {"team", "model", "params", "max_seq_length": 384, "doc_stride": 128}` |
| `POST /predict` | Bearer | `200 {"id", "answer", "start", "score"}` |

- Request body for `/predict`: `{"id": "...", "context": "...", "question": "..."}` (id 1-64 chars, context 1-5000, question 1-500; no extra fields).
- `answer` is an exact substring of `context` with `context[start:start+len(answer)] == answer`, or `answer = null` and `start = null` when there is no answer. `score` is in `[0, 1]`.
- Auth: header `Authorization: Bearer <token>`. The server reads the expected token only from the environment variable `QA_API_TOKEN`. If it is unset, every protected request gets 401.
- Errors use `{"error": "<code>", "detail": "..."}` with `code` in `unauthorized` (401), `bad_request` (400/422), `internal_error` (500).
- Long contexts are split into 384-token windows with a 128-token stride. A calibrated threshold decides when to return `null`.

## Weights

The weights are distributed outside git, as `modelo-final-v3.zip` on the team's GitHub Release or shared folder. Extract the zip so the files sit directly in `weights/modelo-final-v3/`:

```text
weights/modelo-final-v3/
  model.safetensors
  config.json
  tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json
  metrics.json        # optional here; holds no_answer_threshold
```

`no_answer_threshold` is read from `weights/<model>/metrics.json` or, if that file is missing, from `logs/<model>/metrics.json` (the committed copy exists for `modelo-final-v2`). To serve another model folder, set `QA_MODEL_NAME=<folder under weights/>`. Do not nest the folder twice (`weights/modelo-final-v3/modelo-final-v3/`). Never commit `*.zip`, `*.safetensors`, or `weights/`; `.gitignore` already excludes them.

## Setup and run

The token and team name are set in the terminal; never write them into a file tracked by git. The run scripts refuse to start without `QA_API_TOKEN`, warn if `TEAM_NAME` is missing, check that the weights are present, install `requirements-serve.txt` (CPU torch), then run `uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1`.

### Linux / WSL (uv)

```bash
export QA_API_TOKEN=<token>
export TEAM_NAME=<registered team name>
./run.sh
```

Manual equivalent:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-serve.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

### Windows (PowerShell, Python venv)

```powershell
$env:QA_API_TOKEN = "<token>"
$env:TEAM_NAME = "<registered team name>"
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Manual equivalent:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-serve.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

### Docker (optional)

Docker is a secondary option; the venv path above is the primary one. The image contains the code, `logs/` (`metrics.json`) and `samples/`. The weights are mounted as a volume.

```bash
docker build -t nlp-qa-api .
docker run --rm -p 8000:8000 -e QA_API_TOKEN -e TEAM_NAME \
    -v "$(pwd)/weights:/app/weights:ro" nlp-qa-api
```

PowerShell: `docker run --rm -p 8000:8000 -e QA_API_TOKEN -e TEAM_NAME -v "${PWD}\weights:/app/weights:ro" nlp-qa-api`.

## Test

With the server running, in a second terminal with the same `QA_API_TOKEN`:

```bash
python smoke_test.py http://localhost:8000 --extended
```

`smoke_test.py` uses only the standard library. It runs the 5 mandatory checks: `/health` without a token -> 200, `/info` without a token -> 401, `/info` with a wrong token -> 401, `/info` with a valid token -> 200, and `/predict` with a valid token -> 200. It then sends every item of `samples/public_sample.json` and checks the answers, the offsets, and the 5 s latency limit. Run the same command against the public tunnel URL.

The same 5 tests with curl (PowerShell: use `curl.exe` and `$env:QA_API_TOKEN`):

```bash
curl -i http://localhost:8000/health
curl -i http://localhost:8000/info
curl -i -H "Authorization: Bearer wrong" http://localhost:8000/info
curl -i -H "Authorization: Bearer $QA_API_TOKEN" http://localhost:8000/info
curl -i -X POST http://localhost:8000/predict -H "Authorization: Bearer $QA_API_TOKEN" \
    -H "Content-Type: application/json" -d @request.json
```

`request.json` holds the `id`, `context` and `question` of one item from `samples/public_sample.json`.

Other checks (in-process, CPU, no server needed): `python test_regression.py` (contract, offsets, CPU device), `python test_full_inference.py`, and `python test_windows.py`. `python test_api.py` starts its own server on port 8000, so do not run it while the API is already running.

## Public tunnel (dry run and presentation)

1. Start the API and check `curl http://localhost:8000/health`.
2. Primary, ngrok: run `ngrok config add-authtoken <ngrok authtoken>` once, then `ngrok http 8000`. The URL looks like `https://<name>.ngrok-free.app`.
3. Backup, Cloudflare Quick Tunnel (no account needed): `cloudflared tunnel --url http://localhost:8000`. The URL looks like `https://<random>.trycloudflare.com` and changes on every restart.
4. From a device on another network (for example, a phone on mobile data), run the 5 curl tests or `python smoke_test.py <public-url> --extended`.
5. Send the public URL to the instructor before the dry-run slot and again before the presentation, and again whenever it changes.
6. Keep the laptop plugged in with sleep disabled (Windows: `powercfg /change standby-timeout-ac 0`), and keep both the API and the tunnel running for the whole session.
7. On presentation day, restart the API with the new token in `QA_API_TOKEN`.

## Repository map

| Path | Purpose |
|---|---|
| `api.py` | FastAPI service (endpoints, Bearer auth, error schema) |
| `inference.py` | CPU inference: sliding windows, span selection, no-answer threshold |
| `train.py`, `plot_training.py` | Fine-tuning on SQuAD 2.0 and loss plots |
| `smoke_test.py` | Dry-run checks against a local or public URL |
| `test_*.py`, `benchmark_cpu.py`, `experiment_windows.py` | Local tests, CPU benchmark, window-aggregation experiment |
| `run.sh`, `run.ps1`, `Dockerfile` | Start the API (Linux/WSL, Windows, Docker) |
| `requirements-serve.txt` | Minimal CPU serving dependencies |
| `requirements.txt` | Serving dependencies plus training and evaluation extras |
| `qa_api_contract.yaml` | API contract (OpenAPI) |
| `samples/public_sample.json` | Public sample items with reference answers |
| `logs/modelo-final-v1/`, `logs/modelo-final-v2/` | Metrics, training history, and loss curves of the final models |
| `logs/pilot/` | Early pilot run (64 training examples). Not representative |
| `REPORT.md` | Project report |
| `weights/` | Model folders (ignored by git, distributed separately) |
