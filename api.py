"""FastAPI QA service — integrante 3, branch camila/parte3API.

Endpoints
---------
GET  /health   — public
GET  /info     — protected (Bearer token)
POST /predict  — protected (Bearer token)

Authentication
--------------
Token is read exclusively from the QA_API_TOKEN environment variable.
It is NEVER hard-coded here.
"""
import os
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="NLP QA API", version="1.0.0")

# ---------------------------------------------------------------------------
# Model loading — single instance, loaded once at startup in a daemon thread
# so /health can respond immediately while the model is still loading.
# ---------------------------------------------------------------------------
_qa: Any = None          # QAInference instance
_load_error: Optional[str] = None
_model_ready = threading.Event()


def _load_model() -> None:
    global _qa, _load_error
    try:
        from inference import QAInference  # noqa: PLC0415
        _qa = QAInference()
        _model_ready.set()
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
        _model_ready.set()   # unblock waiters even on error


threading.Thread(target=_load_model, daemon=True).start()

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _require_auth(request: Request) -> None:
    """Raise a 401 JSONResponse if the Bearer token is invalid or missing."""
    expected = os.environ.get("QA_API_TOKEN", "")
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header[7:] != expected:
        raise _Unauthorized()


class _Unauthorized(Exception):
    """Sentinel for 401 errors."""


@app.exception_handler(_Unauthorized)
async def _unauthorized_handler(request: Request, exc: _Unauthorized) -> JSONResponse:
    return JSONResponse({"error": "unauthorized"}, status_code=401)


# Override FastAPI's default 422 Unprocessable Entity to match contract format.
@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = "; ".join(
        f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
        for e in exc.errors()
    )
    return JSONResponse({"error": "bad_request", "detail": detail}, status_code=422)


# ---------------------------------------------------------------------------
# GET /health — public
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    """Return 200 when model is ready, 503 while loading or on error."""
    if _model_ready.is_set() and _load_error is None and _qa is not None:
        return JSONResponse({"status": "ok"}, status_code=200)
    if _load_error:
        return JSONResponse({"status": "loading", "detail": _load_error}, status_code=503)
    return JSONResponse({"status": "loading"}, status_code=503)


# ---------------------------------------------------------------------------
# GET /info — protected
# ---------------------------------------------------------------------------

@app.get("/info")
async def info(request: Request) -> JSONResponse:
    """Return model metadata. Requires valid Bearer token."""
    _require_auth(request)
    if not _model_ready.is_set() or _qa is None:
        return JSONResponse({"error": "internal_error", "detail": "model not ready"}, status_code=503)

    params = sum(p.numel() for p in _qa.model.parameters())
    return JSONResponse({
        "team": os.environ.get("TEAM_NAME", "team-03"),
        "model": "distilbert/distilbert-base-uncased",
        "params": params,
        "max_seq_length": _qa.max_length,
        "doc_stride": _qa.stride,
    })


# ---------------------------------------------------------------------------
# POST /predict — protected
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    context: str = Field(..., min_length=1, max_length=5000)
    question: str = Field(..., min_length=1, max_length=500)


@app.post("/predict")
async def predict(body: PredictRequest, request: Request) -> JSONResponse:
    """Run extractive QA and return the best answer span (or null)."""
    _require_auth(request)

    if not _model_ready.is_set() or _qa is None:
        return JSONResponse(
            {"error": "internal_error", "detail": "model not ready"},
            status_code=503,
        )

    try:
        result: Dict[str, Any] = _qa.predict(body.context, body.question)
    except ValueError as exc:
        return JSONResponse({"error": "bad_request", "detail": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": "internal_error", "detail": str(exc)}, status_code=500)

    return JSONResponse({
        "id": body.id,
        "answer": result["answer"],
        "start": result["start"],
        "score": result["score"],
    })

