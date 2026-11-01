#!/usr/bin/env bash
# Start the QA API on 0.0.0.0:8000 (CPU only) from a clean checkout (Linux, WSL, macOS).
#
#   export QA_API_TOKEN=<token>        # required, never committed
#   export TEAM_NAME=<registered team name>
#   ./run.sh
#
# Optional: QA_MODEL_NAME=<folder under weights/> (default: modelo-final-v3),
#           VENV_DIR=<path> (default: .venv), SKIP_INSTALL=1 to skip the dependency sync.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR="${VENV_DIR:-.venv}"
MODEL_NAME="${QA_MODEL_NAME:-modelo-final-v3}"
MODEL_DIR="weights/${MODEL_NAME}"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

if [ -z "${QA_API_TOKEN:-}" ]; then
    echo "ERROR: QA_API_TOKEN is not set. Refusing to start." >&2
    echo "       Set it in this terminal first: export QA_API_TOKEN=<token>" >&2
    exit 1
fi

if [ -z "${TEAM_NAME:-}" ]; then
    echo "WARNING: TEAM_NAME is not set; /info will report the placeholder team name." >&2
    echo "         Set it with: export TEAM_NAME=<registered team name>" >&2
fi

if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo "ERROR: ${MODEL_DIR}/model.safetensors not found." >&2
    echo "       The weights are not in git. Download ${MODEL_NAME}.zip from the team's GitHub Release" >&2
    echo "       or shared folder and extract it so the files sit directly in ${MODEL_DIR}/" >&2
    echo "       (model.safetensors, config.json, tokenizer files), not in ${MODEL_DIR}/${MODEL_NAME}/." >&2
    exit 1
fi
if [ ! -f "${MODEL_DIR}/metrics.json" ] && [ ! -f "logs/${MODEL_NAME}/metrics.json" ]; then
    echo "ERROR: no metrics.json (no_answer_threshold) in ${MODEL_DIR}/ or logs/${MODEL_NAME}/." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "Creating virtual environment in ${VENV_DIR} ..."
    uv venv --python 3.12 "${VENV_DIR}"
fi

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    echo "Syncing serving dependencies (CPU torch) ..."
    uv pip install --python "${VENV_DIR}/bin/python" -r requirements-serve.txt \
        --extra-index-url "${TORCH_CPU_INDEX}" --index-strategy unsafe-best-match
fi

export QA_MODEL_NAME="${MODEL_NAME}"
echo "Starting API on http://0.0.0.0:8000 with model ${MODEL_NAME} (Ctrl+C to stop) ..."
exec "${VENV_DIR}/bin/python" -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
