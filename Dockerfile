# CPU-only image for the QA API. Weights are NOT baked in: mount them at /app/weights.
#
#   docker build -t nlp-qa-api .
#   docker run --rm -p 8000:8000 -e QA_API_TOKEN -e TEAM_NAME \
#       -v "$(pwd)/weights:/app/weights:ro" nlp-qa-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Code, the model metrics (no_answer_threshold fallback) and the public sample only.
COPY api.py inference.py smoke_test.py qa_api_contract.yaml ./
COPY logs/ ./logs/
COPY samples/ ./samples/

RUN useradd --create-home --uid 1000 app && mkdir -p /app/weights && chown app /app/weights
USER app

VOLUME ["/app/weights"]
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
