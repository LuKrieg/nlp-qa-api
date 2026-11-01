"""FastAPI extractive-QA service implementing qa_api_contract.yaml.

Endpoints
---------
GET  /health   - public
GET  /info     - protected (Bearer token)
POST /predict  - protected (Bearer token)

Authentication
--------------
The token is read exclusively from the QA_API_TOKEN environment variable and is
never hard-coded. An unset/empty QA_API_TOKEN rejects every protected request.

Concurrency
-----------
Run with a single uvicorn worker (one copy of the model in memory). `predict` is
declared `async def` on purpose: FastAPI runs it on the event loop, so requests
are processed one at a time, which is required because the fast tokenizer is not
safe to call from several threads at once. If you ever change it to `def`, add a
threading.Lock around the model call.
"""
import hmac
import logging
import os
import threading
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("qa_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="NLP QA API", version="1.0.0")

TEAM_NAME = os.environ.get("TEAM_NAME", "team-03")
if "TEAM_NAME" not in os.environ:
    log.warning("TEAM_NAME is not set; /info will report the placeholder %r", TEAM_NAME)
if not os.environ.get("QA_API_TOKEN"):
    log.warning("QA_API_TOKEN is not set; every protected request will be rejected with 401")

# ---------------------------------------------------------------------------
# Model loading: one instance, loaded once at startup in a daemon thread so
# /health can answer 503 {"status": "loading"} while the model is loading.
# ---------------------------------------------------------------------------
_qa: Any = None
_load_error: Optional[str] = None
_model_ready = threading.Event()


def _load_model() -> None:
    global _qa, _load_error
    try:
        from inference import QAInference  # noqa: PLC0415

        qa = QAInference()
        # Warm-up so the first real request does not pay one-time initialization costs.
        qa.predict("Warm-up context sentence.", "What is this?")
        _qa = qa
        log.info("Model loaded and warmed up")
    except Exception as exc:  # noqa: BLE001
        _load_error = f"{type(exc).__name__}: {exc}"
        log.exception("Model failed to load")
    finally:
        _model_ready.set()


threading.Thread(target=_load_model, daemon=True).start()


def _ready() -> bool:
    return _model_ready.is_set() and _load_error is None and _qa is not None


# ---------------------------------------------------------------------------
# Errors: every error body is {"error": "<code>", "detail": "<optional>"}
# with <code> in {unauthorized, bad_request, internal_error}.
# ---------------------------------------------------------------------------
class _Unauthorized(Exception):
    """Raised by the auth dependency; rendered as 401."""


def _error(status: int, code: str, detail: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
    body: Dict[str, Any] = {"error": code}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status, headers=headers)


@app.exception_handler(_Unauthorized)
async def _unauthorized_handler(request: Request, exc: _Unauthorized) -> JSONResponse:
    return _error(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI parses the body before running dependencies, so a malformed body would
    # mask a missing/invalid token. Authentication must always be answered first.
    if request.url.path in ("/info", "/predict"):
        try:
            require_auth(request)
        except _Unauthorized:
            return _error(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})
    detail = "; ".join(
        f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in exc.errors()
    )
    return _error(422, "bad_request", detail)


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # Unknown routes (404), wrong methods (405), etc. must also use the Error schema.
    code = "unauthorized" if exc.status_code == 401 else (
        "internal_error" if exc.status_code >= 500 else "bad_request")
    return _error(exc.status_code, code, str(exc.detail))


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "internal_error")


# ---------------------------------------------------------------------------
# Auth dependency. FastAPI resolves dependencies before validating the body,
# so a missing/invalid token always yields 401, even with a malformed body.
# ---------------------------------------------------------------------------
def require_auth(request: Request) -> None:
    expected = os.environ.get("QA_API_TOKEN", "")
    header = request.headers.get("Authorization", "")
    # Fail closed: an unset/empty QA_API_TOKEN never authenticates anyone.
    # Scheme name and token are compared exactly (case-sensitive), in constant time.
    if (
        not expected
        or not header.startswith("Bearer ")
        or not hmac.compare_digest(header[7:].encode(), expected.encode())
    ):
        raise _Unauthorized()


# ---------------------------------------------------------------------------
# GET /health - public
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> JSONResponse:
    """200 {"status": "ok"} once the model is loaded; 503 {"status": "loading"} otherwise."""
    if _ready():
        return JSONResponse({"status": "ok"}, status_code=200)
    body: Dict[str, Any] = {"status": "loading"}
    if _load_error:
        body["detail"] = _load_error  # extra fields are allowed by the contract
    return JSONResponse(body, status_code=503)


# ---------------------------------------------------------------------------
# GET /info - protected
# ---------------------------------------------------------------------------
@app.get("/info", dependencies=[Depends(require_auth)])
async def info() -> JSONResponse:
    if not _ready():
        return _error(503, "internal_error", "model not ready")
    return JSONResponse({
        "team": TEAM_NAME,
        "model": _qa.base_model_id,
        "params": sum(p.numel() for p in _qa.model.parameters()),
        "max_seq_length": _qa.max_length,
        "doc_stride": _qa.stride,
    })


# ---------------------------------------------------------------------------
# POST /predict - protected
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # contract: additionalProperties: false

    id: str = Field(..., min_length=1, max_length=64)
    context: str = Field(..., min_length=1, max_length=5000)
    question: str = Field(..., min_length=1, max_length=500)


@app.post("/predict", dependencies=[Depends(require_auth)])
async def predict(body: PredictRequest) -> JSONResponse:
    """Run extractive QA and return the best answer span, or null when there is no answer."""
    if not _ready():
        return _error(503, "internal_error", "model not ready")
    try:
        result: Dict[str, Any] = _qa.predict(body.context, body.question)
    except ValueError as exc:
        return _error(400, "bad_request", str(exc))
    except Exception:  # noqa: BLE001
        log.exception("predict failed for id=%s", body.id)
        return _error(500, "internal_error")

    return JSONResponse({
        "id": body.id,
        "answer": result["answer"],
        "start": result["start"],
        "score": result["score"],
    })
