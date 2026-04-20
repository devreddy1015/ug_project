from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .celery_app import celery_app
from .config import SETTINGS
from .db import get_conn, init_db
from .jobs import (
    STATUS_CANCELED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    VALID_STATUSES,
    cancel_job,
    count_jobs,
    create_job,
    get_job,
    get_job_by_idempotency_key,
    get_job_metrics,
    list_jobs,
    mark_job_failed,
)
from .logging_utils import configure_logging
from .storage import UploadTooLargeError, UploadValidationError, ensure_storage, save_upload


ALLOWED_MODALITIES = {"text", "image", "video", "audio"}
TASK_NAME_BY_MODALITY = {
    "text": "service.tasks.score_text",
    "image": "service.tasks.score_image",
    "video": "service.tasks.score_video",
    "audio": "service.tasks.score_audio",
}
TASK_QUEUE_BY_MODALITY = {
    "text": "text",
    "image": "image",
    "video": "video",
    "audio": "audio",
}

logger = logging.getLogger("shield.service.api")


class TextJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    idempotency_key: Optional[str] = Field(default=None, max_length=128)
    webhook_url: Optional[str] = Field(default=None, max_length=1024)


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    created_new: bool
    status_url: str
    result_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    modality: str
    queue_name: Optional[str] = None
    attempts: int = 0
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    webhook_status: Optional[str] = None


class JobListResponse(BaseModel):
    items: list[JobStatusResponse]
    limit: int
    offset: int
    total: int


class HealthResponse(BaseModel):
    status: str
    checks: Dict[str, str]


def _route(path: str) -> str:
    return f"{SETTINGS.api_prefix}{path}"


def _job_status_url(job_id: str) -> str:
    return _route(f"/jobs/{job_id}")


def _job_result_url(job_id: str) -> str:
    return _route(f"/results/{job_id}")


def _as_job_status_response(record: Dict[str, Any]) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=record["id"],
        status=record["status"],
        modality=record["modality"],
        queue_name=record.get("queue_name"),
        attempts=int(record.get("attempts") or 0),
        created_at=record.get("created_at") or "",
        updated_at=record.get("updated_at") or "",
        started_at=record.get("started_at"),
        completed_at=record.get("completed_at"),
        result=record.get("result"),
        error_message=record.get("error_message"),
        webhook_status=record.get("webhook_status"),
    )


def _enqueue_task(modality: str, job_id: str) -> None:
    task_name = TASK_NAME_BY_MODALITY[modality]
    queue_name = TASK_QUEUE_BY_MODALITY[modality]
    celery_app.send_task(task_name, args=[job_id], queue=queue_name)


def _authorize_request(
    api_key: Optional[str] = Header(default=None, alias=SETTINGS.api_key_header),
) -> None:
    if not SETTINGS.require_api_key:
        return
    if api_key != SETTINGS.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(
    title=SETTINGS.app_name,
    version="3.0.0",
    description="Asynchronous multimodal Guard service with durable jobs and webhook callbacks.",
)

if SETTINGS.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(SETTINGS.allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get(SETTINGS.request_id_header, uuid.uuid4().hex)
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    response.headers[SETTINGS.request_id_header] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    logger.info(
        "request.completed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    return response


@app.on_event("startup")
def startup() -> None:
    configure_logging(SETTINGS.log_level)
    init_db()
    ensure_storage()
    logger.info("service.startup", extra={"environment": SETTINGS.environment})


@app.get("/health/live", response_model=HealthResponse)
def health_live() -> HealthResponse:
    return HealthResponse(status="ok", checks={"api": "ok"})


@app.get("/health/ready", response_model=HealthResponse)
def health_ready() -> HealthResponse:
    checks: Dict[str, str] = {}

    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"

    try:
        with celery_app.connection_for_read() as connection:
            connection.ensure_connection(max_retries=1)
        checks["broker"] = "ok"
    except Exception:
        checks["broker"] = "error"

    try:
        ensure_storage()
        checks["storage"] = "ok"
    except Exception:
        checks["storage"] = "error"

    status_value = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
    if status_value != "ok":
        raise HTTPException(status_code=503, detail=HealthResponse(status=status_value, checks=checks).model_dump())
    return HealthResponse(status=status_value, checks=checks)


@app.post(_route("/jobs/text"), response_model=JobCreatedResponse)
def create_text_job(
    payload: TextJobRequest,
    _: None = Depends(_authorize_request),
) -> JobCreatedResponse:
    if len(payload.text) > int(SETTINGS.text_max_chars):
        raise HTTPException(status_code=400, detail=f"Text exceeds max {SETTINGS.text_max_chars} characters")

    job_id, created_new = create_job(
        modality="text",
        queue_name=TASK_QUEUE_BY_MODALITY["text"],
        input_text=payload.text,
        idempotency_key=payload.idempotency_key,
        webhook_url=(payload.webhook_url or None),
    )

    try:
        if created_new:
            _enqueue_task("text", job_id)
    except Exception as error:
        mark_job_failed(job_id, f"Broker dispatch failed: {error}")
        raise HTTPException(status_code=503, detail="Could not enqueue job") from error

    current = get_job(job_id)
    status_value = current["status"] if current else STATUS_QUEUED

    return JobCreatedResponse(
        job_id=job_id,
        status=status_value,
        created_new=created_new,
        status_url=_job_status_url(job_id),
        result_url=_job_result_url(job_id),
    )


@app.post(_route("/jobs/file"), response_model=JobCreatedResponse)
def create_file_job(
    modality: str = Query(..., description="image, video, or audio"),
    file: UploadFile = File(...),
    idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
    webhook_url_header: Optional[str] = Header(default=None, alias="X-Webhook-Url"),
    webhook_url: Optional[str] = Query(default=None, description="Optional completion callback URL"),
    _: None = Depends(_authorize_request),
) -> JobCreatedResponse:
    normalized = modality.lower().strip()
    if normalized not in ALLOWED_MODALITIES or normalized == "text":
        raise HTTPException(status_code=400, detail="Unsupported modality")

    callback_url = webhook_url_header or webhook_url

    if idempotency_key:
        existing = get_job_by_idempotency_key(idempotency_key)
        if existing is not None:
            return JobCreatedResponse(
                job_id=existing["id"],
                status=existing["status"],
                created_new=False,
                status_url=_job_status_url(existing["id"]),
                result_url=_job_result_url(existing["id"]),
            )

    try:
        stored = save_upload(file, modality=normalized)
    except UploadTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except UploadValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    job_id, created_new = create_job(
        modality=normalized,
        queue_name=TASK_QUEUE_BY_MODALITY[normalized],
        input_path=str(stored.path),
        input_filename=stored.original_filename,
        input_content_type=stored.content_type,
        input_size_bytes=stored.size_bytes,
        idempotency_key=idempotency_key,
        webhook_url=callback_url,
    )

    try:
        if created_new:
            _enqueue_task(normalized, job_id)
    except Exception as error:
        mark_job_failed(job_id, f"Broker dispatch failed: {error}")
        raise HTTPException(status_code=503, detail="Could not enqueue job") from error

    current = get_job(job_id)
    status_value = current["status"] if current else STATUS_QUEUED

    return JobCreatedResponse(
        job_id=job_id,
        status=status_value,
        created_new=created_new,
        status_url=_job_status_url(job_id),
        result_url=_job_result_url(job_id),
    )


@app.get(_route("/jobs"), response_model=JobListResponse)
def get_jobs(
    status: Optional[str] = Query(default=None),
    modality: Optional[str] = Query(default=None),
    limit: int = Query(default=SETTINGS.default_page_size, ge=1),
    offset: int = Query(default=0, ge=0),
    _: None = Depends(_authorize_request),
) -> JobListResponse:
    status_filter = status.lower().strip() if status else None
    modality_filter = modality.lower().strip() if modality else None

    if status_filter and status_filter not in VALID_STATUSES:
        allowed = ", ".join(sorted(VALID_STATUSES))
        raise HTTPException(status_code=400, detail=f"Unsupported status filter. Allowed: {allowed}")

    if modality_filter and modality_filter not in ALLOWED_MODALITIES:
        allowed = ", ".join(sorted(ALLOWED_MODALITIES))
        raise HTTPException(status_code=400, detail=f"Unsupported modality filter. Allowed: {allowed}")

    safe_limit = min(limit, SETTINGS.max_page_size)
    records = list_jobs(limit=safe_limit, offset=offset, status=status_filter, modality=modality_filter)
    total = count_jobs(status=status_filter, modality=modality_filter)

    return JobListResponse(
        items=[_as_job_status_response(record) for record in records],
        limit=safe_limit,
        offset=offset,
        total=total,
    )


@app.get(_route("/jobs/{job_id}"), response_model=JobStatusResponse)
def get_job_status(job_id: str, _: None = Depends(_authorize_request)) -> JobStatusResponse:
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _as_job_status_response(record)


@app.post(_route("/jobs/{job_id}/cancel"), response_model=JobStatusResponse)
def cancel_job_endpoint(job_id: str, _: None = Depends(_authorize_request)) -> JobStatusResponse:
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if record["status"] in {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELED}:
        raise HTTPException(status_code=409, detail=f"Job already terminal: {record['status']}")

    canceled = cancel_job(job_id)
    if not canceled:
        raise HTTPException(status_code=409, detail="Job could not be canceled")

    updated = get_job(job_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _as_job_status_response(updated)


@app.get(_route("/results/{job_id}"))
def get_job_results(job_id: str, _: None = Depends(_authorize_request)) -> Dict[str, Any]:
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if record["status"] != STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed")

    return {
        "job_id": job_id,
        "modality": record["modality"],
        "result": record.get("result") or {},
    }


@app.get(_route("/metrics"))
def metrics(_: None = Depends(_authorize_request)) -> Dict[str, Any]:
    return {
        "service": SETTINGS.app_name,
        "environment": SETTINGS.environment,
        "job_metrics": get_job_metrics(),
        "page_limits": {
            "default": SETTINGS.default_page_size,
            "max": SETTINGS.max_page_size,
        },
    }
