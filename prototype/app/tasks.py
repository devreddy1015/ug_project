from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from typing import Dict, Tuple

from .celery_app import celery_app
from .jobs import (
    STATUS_CANCELED,
    get_job,
    mark_job_failed,
    save_job_result,
    update_job_status,
)


logger = logging.getLogger("shield.worker")

HIGH_RISK_PHRASES = (
    "kill yourself",
    "how to make a bomb",
    "racial slur",
    "terror attack",
    "shoot them",
)

RISK_TERMS = (
    "kill",
    "hate",
    "bomb",
    "attack",
    "shoot",
    "terror",
    "suicide",
    "overdose",
    "fraud",
    "scam",
    "abuse",
)

SAFETY_TERMS = (
    "support",
    "help",
    "awareness",
    "education",
    "prevention",
    "recovery",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _risk_level(score: float) -> str:
    if score >= 75.0:
        return "critical"
    if score >= 55.0:
        return "high"
    if score >= 35.0:
        return "moderate"
    if score >= 15.0:
        return "low"
    return "minimal"


def _score_from_text(text: str) -> Tuple[float, float, Dict[str, object]]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_']+", lowered)
    token_count = max(len(tokens), 1)

    high_risk_hits = sum(lowered.count(phrase) for phrase in HIGH_RISK_PHRASES)
    risk_term_hits = sum(lowered.count(term) for term in RISK_TERMS)
    safety_term_hits = sum(lowered.count(term) for term in SAFETY_TERMS)

    risk_density = risk_term_hits / token_count
    high_risk_density = high_risk_hits / token_count
    lexical_diversity = min(len(set(tokens)) / token_count, 1.0)

    score = (
        6.0
        + 500.0 * risk_density
        + 1200.0 * high_risk_density
        - 35.0 * min(safety_term_hits / token_count, 1.0)
        + 8.0 * (1.0 - lexical_diversity)
    )
    score = round(_clamp(score, 0.0, 100.0), 4)

    coverage = min(token_count / 120.0, 1.0)
    signal = min((risk_term_hits + high_risk_hits + safety_term_hits) / 10.0, 1.0)
    confidence = round(_clamp(0.35 + 0.4 * coverage + 0.25 * signal, 0.0, 0.98), 4)

    signals = {
        "token_count": token_count,
        "risk_term_hits": risk_term_hits,
        "high_risk_phrase_hits": high_risk_hits,
        "safety_term_hits": safety_term_hits,
        "lexical_diversity": round(lexical_diversity, 4),
    }
    return score, confidence, signals


def _score_from_file(path: Path, modality: str) -> Tuple[float, float, Dict[str, object]]:
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024.0 * 1024.0)
    baseline = {"image": 20.0, "audio": 24.0, "video": 28.0}.get(modality, 22.0)
    size_component = min(size_mb / 300.0, 1.0) * 18.0

    name = path.name.lower()
    flagged_tokens = sum(
        token in name
        for token in ("hate", "violent", "blood", "gore", "nsfw", "extreme")
    )
    name_component = min(float(flagged_tokens) * 9.0, 27.0)

    score = round(_clamp(baseline + size_component + name_component, 0.0, 100.0), 4)
    confidence = round(_clamp(0.40 + min(size_mb / 150.0, 1.0) * 0.45, 0.0, 0.92), 4)

    signals = {
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_mb, 4),
        "filename_token_hits": flagged_tokens,
        "extension": path.suffix.lower(),
    }
    return score, confidence, signals


def _build_result(
    modality: str,
    score: float,
    confidence: float,
    signals: Dict[str, object],
) -> Dict[str, object]:
    return {
        "modality": modality,
        "risk_score": round(score, 4),
        "risk_level": _risk_level(score),
        "confidence": round(confidence, 4),
        "signals": signals,
        "analysis_mode": "heuristic_v2",
        "model_version": "proto-2",
        "created_at": _now(),
    }


def _handle_failure(job_id: str, error: Exception) -> Dict[str, object]:
    logger.exception("Worker job failed", extra={"job_id": job_id})
    mark_job_failed(job_id, str(error))
    return {"error": str(error), "job_id": job_id}


def _load_job_for_processing(job_id: str, task_id: str) -> Tuple[Dict[str, object] | None, Dict[str, object] | None]:
    job = get_job(job_id)
    if job is None:
        return None, _handle_failure(job_id, ValueError("Job not found"))
    if job.get("status") == STATUS_CANCELED:
        return None, {"job_id": job_id, "status": STATUS_CANCELED}

    update_job_status(job_id, "processing", task_id=task_id)
    return get_job(job_id), None


@celery_app.task(name="app.tasks.score_text", bind=True)
def score_text(self, job_id: str) -> Dict[str, object]:
    try:
        task_id = str(self.request.id or "")
        job, response = _load_job_for_processing(job_id, task_id)
        if response is not None:
            return response
        if job is None or not job.get("input_text"):
            return _handle_failure(job_id, ValueError("Missing input text"))

        score, confidence, signals = _score_from_text(str(job["input_text"]))
        result = _build_result("text", score, confidence, signals)
        save_job_result(job_id, result)
        return result
    except Exception as error:
        return _handle_failure(job_id, error)


@celery_app.task(name="app.tasks.score_image", bind=True)
def score_image(self, job_id: str) -> Dict[str, object]:
    return _score_file_job(self, job_id, "image")


@celery_app.task(name="app.tasks.score_video", bind=True)
def score_video(self, job_id: str) -> Dict[str, object]:
    return _score_file_job(self, job_id, "video")


@celery_app.task(name="app.tasks.score_audio", bind=True)
def score_audio(self, job_id: str) -> Dict[str, object]:
    return _score_file_job(self, job_id, "audio")


def _score_file_job(task, job_id: str, modality: str) -> Dict[str, object]:
    try:
        task_id = str(task.request.id or "")
        job, response = _load_job_for_processing(job_id, task_id)
        if response is not None:
            return response
        if job is None or not job.get("input_path"):
            return _handle_failure(job_id, ValueError("Missing input path"))

        path = Path(str(job["input_path"]))
        if not path.exists():
            return _handle_failure(job_id, FileNotFoundError("Input file not found"))

        score, confidence, signals = _score_from_file(path, modality)
        result = _build_result(modality, score, confidence, signals)
        save_job_result(job_id, result)
        return result
    except Exception as error:
        return _handle_failure(job_id, error)
