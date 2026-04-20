from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from .analyzers import analyze_non_video_file, analyze_text_payload, analyze_video_file
from .celery_app import celery_app
from .jobs import (
    STATUS_CANCELED,
    get_job,
    mark_job_failed,
    save_job_result,
    update_job_status,
    update_webhook_delivery,
)
from .notifications import build_webhook_payload, post_job_webhook


logger = logging.getLogger("shield.service.worker")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_notify_webhook(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        return

    webhook_url = str(job.get("webhook_url") or "").strip()
    if not webhook_url:
        return

    payload = build_webhook_payload(job)
    delivered, error_message = post_job_webhook(webhook_url, payload)
    update_webhook_delivery(job_id, delivered=delivered, error_message=error_message)


def _handle_failure(job_id: str, error: Exception) -> Dict[str, object]:
    logger.exception("worker.job.failed", extra={"job_id": job_id})
    mark_job_failed(job_id, str(error))
    try:
        _try_notify_webhook(job_id)
    except Exception:
        logger.exception("worker.webhook.failed", extra={"job_id": job_id})
    return {"error": str(error), "job_id": job_id, "failed_at": _now()}


def _load_job_for_processing(job_id: str, task_id: str) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    job = get_job(job_id)
    if job is None:
        return None, _handle_failure(job_id, ValueError("Job not found"))

    if job.get("status") == STATUS_CANCELED:
        return None, {"job_id": job_id, "status": STATUS_CANCELED}

    update_job_status(job_id, "processing", task_id=task_id)
    return get_job(job_id), None


def _resolve_input_path(job: Dict[str, object]) -> Path:
    input_path = str(job.get("input_path") or "").strip()
    if not input_path:
        raise ValueError("Missing input path")

    path = Path(input_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Input file not found")
    return path


@celery_app.task(name="service.tasks.score_text", bind=True)
def score_text(self, job_id: str) -> Dict[str, object]:
    try:
        task_id = str(self.request.id or "")
        job, response = _load_job_for_processing(job_id, task_id)
        if response is not None:
            return response
        if job is None:
            return _handle_failure(job_id, ValueError("Missing job"))

        text = str(job.get("input_text") or "")
        if not text:
            return _handle_failure(job_id, ValueError("Missing input text"))

        result = analyze_text_payload(text)
        save_job_result(job_id, result)
        _try_notify_webhook(job_id)
        return result
    except Exception as error:
        return _handle_failure(job_id, error)


@celery_app.task(name="service.tasks.score_video", bind=True)
def score_video(self, job_id: str) -> Dict[str, object]:
    try:
        task_id = str(self.request.id or "")
        job, response = _load_job_for_processing(job_id, task_id)
        if response is not None:
            return response
        if job is None:
            return _handle_failure(job_id, ValueError("Missing job"))

        path = _resolve_input_path(job)
        result = analyze_video_file(path)
        save_job_result(job_id, result)
        _try_notify_webhook(job_id)
        return result
    except Exception as error:
        return _handle_failure(job_id, error)


@celery_app.task(name="service.tasks.score_image", bind=True)
def score_image(self, job_id: str) -> Dict[str, object]:
    return _score_non_video(self, job_id, modality="image")


@celery_app.task(name="service.tasks.score_audio", bind=True)
def score_audio(self, job_id: str) -> Dict[str, object]:
    return _score_non_video(self, job_id, modality="audio")


def _score_non_video(task, job_id: str, modality: str) -> Dict[str, object]:
    try:
        task_id = str(task.request.id or "")
        job, response = _load_job_for_processing(job_id, task_id)
        if response is not None:
            return response
        if job is None:
            return _handle_failure(job_id, ValueError("Missing job"))

        path = _resolve_input_path(job)
        result = analyze_non_video_file(path, modality=modality)
        save_job_result(job_id, result)
        _try_notify_webhook(job_id)
        return result
    except Exception as error:
        return _handle_failure(job_id, error)
